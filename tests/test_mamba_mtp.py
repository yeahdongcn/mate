"""Tests for the packed multi-token (MTP) SSU path.

The packed kernel is what speculative decoding needs from the decode side: several
tokens per sequence in one call, one destination state slot per speculative
position, and a read slot seeded by the acceptance count. These tests compare it
against ``mate.mamba_kernels.reference`` over the call shapes the serving path can
produce -- tokens per sequence 1..7, batch 1/2/8, every acceptance count in
``0..T``, shared and distinct slot tables, the flat ``[rows]`` tables vLLM's MTP6
call passes, and null block entries on either side.

The comparison is per element and against the oracle rather than against recorded
numbers, because a state that is merely *close* is a behaviour change here: under
MTP the state feeds the draft model, so it moves acceptance length and the served
output. Empty sequences and variable-length packing are covered too -- the packed
row layout is what makes an empty sequence reachable at all.
"""

from __future__ import annotations

import itertools

import pytest
import torch

import mate.mamba
from mate.mamba_kernels.reference import selective_state_update_multi_token_reference
from mate.testing import supported_musa_compute_capability

HEADS = 8
DIM = 8
DSTATE = 128
GROUPS = 2
IO_DTYPE = torch.bfloat16
STATE_DTYPE = torch.float16
SOFTPLUS = True
#: vLLM's ``NULL_BLOCK_ID``, which is also what its backend dispatch passes as the
#: pad slot; MATE refuses any other sentinel, so the tests use the serving one.
PAD = 0


def _device() -> torch.device:
    return torch.device("musa")


def _row_base(steps: list[int]) -> list[int]:
    """First packed row of each sequence."""
    return [0, *itertools.accumulate(steps)][: len(steps)]


def _case(
    steps: list[int],
    *,
    accepted: list[int] | None = None,
    heads: int = HEADS,
    dim: int = DIM,
    dstate: int = DSTATE,
    groups: int = GROUPS,
    state_dtype: torch.dtype = STATE_DTYPE,
    same_slots: bool = False,
    null_src: tuple[tuple[int, int], ...] = (),
    null_dst: tuple[tuple[int, int], ...] = (),
    dense: bool = False,
    disable_state_update: bool = False,
    flat: bool = False,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Build one packed decode call plus the oracle's description of the same call.

    Every ``(sequence, step)`` pair owns its own slot row, so a write to the wrong
    slot shows up as a large difference instead of a coincidence. ``null_src`` and
    ``null_dst`` poke the sentinel into chosen table entries. ``dense`` describes the
    same call through the 4-D ``[batch, steps, heads, dim]`` layout, and ``flat``
    describes it the way vLLM's MTP6 call does: one 1-D table with an entry per packed
    row, addressed as ``sequence + step``. A flat table only names every
    ``(sequence, step)`` pair separately when there is one sequence, so flat cases
    with several multi-token sequences are left to the non-spec regime, where only
    entry ``sequence`` is touched.
    """
    device = _device() if device is None else device
    # A CPU generator keeps the inputs of a case identical across runs, devices and
    # dtypes; the refusal tests then build the same tensors without a MUSA device.
    generator = torch.Generator(device="cpu").manual_seed(0)
    batch = len(steps)
    rows = sum(steps)
    width = max(steps)
    # Row 0 stays the sentinel; a flat table needs one entry per packed row, a 2-D
    # table one per (sequence, step) pair.
    slot_rows = 1 + 2 * (rows if flat else batch * width)

    def _rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.float32, generator=generator).to(device)

    state = (_rand(slot_rows, heads, dim, dstate) * 0.5).to(state_dtype)
    x = _rand(rows, heads, dim).to(IO_DTYPE)
    dt = (
        torch.rand(rows, heads, 1, dtype=torch.float32, generator=generator).to(device)
        * 0.3
        + 0.01
    )
    A = -torch.rand(heads, 1, dtype=torch.float32, generator=generator).to(device) - 0.1
    B = _rand(rows, groups, dstate).to(IO_DTYPE)
    C = _rand(rows, groups, dstate).to(IO_DTYPE)
    D = torch.rand(heads, 1, dtype=torch.float32, generator=generator).to(device)
    dt_bias = _rand(heads, 1) * 0.1
    z = _rand(rows, heads, dim).to(IO_DTYPE)

    table_elems = rows if flat else batch * width
    src_slots = torch.arange(1, 1 + table_elems, dtype=torch.int32, device=device)
    dst_slots = (
        src_slots.clone()
        if same_slots
        else torch.arange(
            1 + table_elems, 1 + 2 * table_elems, dtype=torch.int32, device=device
        )
    )
    if not flat:
        src_slots = src_slots.reshape(batch, width).contiguous()
        dst_slots = dst_slots.reshape(batch, width).contiguous()

    def _poke(table: torch.Tensor, entries: tuple[tuple[int, int], ...]) -> None:
        for sequence, step in entries:
            if flat:
                # A flat table is addressed by position: both coordinates select one
                # entry, which is why a flat multi-sequence table has to stay out of
                # the spec regime -- sequence 1's first token and sequence 0's second
                # would be the same slot.
                table[sequence + step] = PAD
            else:
                table[sequence, step] = PAD

    _poke(src_slots, null_src)
    _poke(dst_slots, null_dst)

    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "z": z,
        "dt_bias": dt_bias,
        "src_slots": src_slots,
        "dst_slots": dst_slots,
        "cu_seqlens": torch.tensor(
            [0, *itertools.accumulate(steps)], dtype=torch.int32, device=device
        ),
        "num_accepted_tokens": (
            None
            if accepted is None
            else torch.tensor(accepted, dtype=torch.int32, device=device)
        ),
        "steps": steps,
        "row_base": _row_base(steps),
        "dense": dense,
        "disable_state_update": disable_state_update,
    }


def _dense_operand(case: dict[str, object], name: str) -> torch.Tensor:
    """View a packed operand as the dense ``[batch, steps, ...]`` form of itself."""
    tensor = case[name]
    steps = case["steps"]
    return tensor.view(len(steps), max(steps), *tensor.shape[1:])


def _run_kernel(case: dict[str, object], *, state: torch.Tensor | None = None):
    x = case["x"]
    dt = case["dt"]
    B = case["B"]
    C = case["C"]
    z = case["z"]
    cu_seqlens = case["cu_seqlens"]
    if case["dense"]:
        x = _dense_operand(case, "x")
        dt = _dense_operand(case, "dt")
        B = _dense_operand(case, "B")
        C = _dense_operand(case, "C")
        z = _dense_operand(case, "z")
        cu_seqlens = None
    return mate.mamba.selective_state_update(
        case["state"] if state is None else state,
        x,
        dt,
        case["A"],
        B,
        C,
        case["D"],
        z=z,
        dt_bias=case["dt_bias"],
        dt_softplus=SOFTPLUS,
        state_batch_indices=case["src_slots"],
        dst_state_batch_indices=case["dst_slots"],
        pad_slot_id=PAD,
        disable_state_update=case["disable_state_update"],
        num_accepted_tokens=case["num_accepted_tokens"],
        cu_seqlens=cu_seqlens,
    )


def _run_reference(case: dict[str, object], *, state: torch.Tensor | None = None):
    return selective_state_update_multi_token_reference(
        case["state"] if state is None else state,
        case["x"],
        case["dt"],
        case["A"],
        case["B"],
        case["C"],
        case["D"],
        case["z"],
        case["dt_bias"],
        SOFTPLUS,
        case["src_slots"],
        case["dst_slots"],
        PAD,
        case["disable_state_update"],
        case["steps"],
        case["row_base"],
        case["num_accepted_tokens"],
    )


def _compare(case: dict[str, object]) -> None:
    expected_state = case["state"].clone()
    expected = _run_reference(case, state=expected_state)
    actual_state = case["state"].clone()
    actual = _run_kernel(case, state=actual_state)

    # A dense call returns its own rank; the oracle speaks the flattened layout.
    actual = actual.reshape(expected.shape)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-2, atol=1e-3)


#: (tokens per sequence, acceptance count) for every value in 0..T.
_TOKENS_AND_ACCEPTED = [
    (steps, accepted) for steps in range(1, 8) for accepted in range(steps + 1)
]


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("steps, accepted", _TOKENS_AND_ACCEPTED)
def test_uniform_packed_sequences_match_reference(batch, steps, accepted):
    """The MTP6 shape family: uniform sequences, one slot table for both sides.

    ``accepted == 0`` covers the floor of the ``max(count - 1, 0)`` read rule, and
    the identity form -- the caller passing the read table as the destination table,
    which is what the mixer does with prefix caching off -- is what the call itself
    uses here.
    """
    case = _case([steps] * batch, accepted=[accepted] * batch, same_slots=True)
    case["dst_slots"] = case["src_slots"]
    _compare(case)


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("steps, accepted", _TOKENS_AND_ACCEPTED)
def test_distinct_slot_tables_match_reference(batch, steps, accepted):
    """Prefix caching passes distinct read and destination tables.

    Every (sequence, step) pair then writes a row no other pair touches, so the
    oracle pins the write set exactly: a missed or extra store cannot hide.
    """
    _compare(_case([steps] * batch, accepted=[accepted] * batch))


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("steps", [1, 3, 7])
def test_null_block_entries_match_reference(batch, steps):
    """A null read slot reads as zero state; a null write slot is never written."""
    # Nulls on the first readable column: the seeded read when nothing was accepted.
    _compare(
        _case(
            [steps] * batch,
            accepted=[1] * batch,
            null_src=tuple((b, 0) for b in range(batch)),
            null_dst=tuple((b, steps - 1) for b in range(batch)),
        )
    )
    # Nulls on the column the acceptance counts select, with one shared table.
    _compare(
        _case(
            [steps] * batch,
            accepted=[steps] * batch,
            same_slots=True,
            null_src=tuple((b, steps - 1) for b in range(batch)),
            null_dst=tuple((b, 0) for b in range(batch)),
        )
    )


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize(
    "steps, accepted",
    [
        ([0, 3, 0, 4], [0, 2, 0, 4]),
        ([7, 1], [5, 1]),
        ([1, 6, 2, 5], [1, 3, 2, 5]),
        ([0, 1], [0, 1]),
    ],
)
def test_variable_length_sequences_match_reference(steps, accepted):
    """Packed rows of different lengths, including sequences of no tokens."""
    _compare(_case(steps, accepted=accepted))
    _compare(_case(steps, accepted=accepted, same_slots=True))


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("steps", [1, 7])
@pytest.mark.parametrize("accepted", [None, "full"])
def test_dense_four_dimensional_call_matches_reference(batch, steps, accepted):
    """The dense ``[batch, steps, heads, dim]`` form runs the same recurrence.

    That is the layout FlashInfer documents for multi-token SSU; the entry point
    flattens it into packed rows with the regular query starts that describe it.
    """
    _compare(
        _case(
            [steps] * batch,
            accepted=None if accepted is None else [steps] * batch,
            dense=True,
            same_slots=True,
        )
    )


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accepted", [None, [2, 2]])
def test_state_dtypes_match_reference(state_dtype, accepted):
    """The pool dtype only changes where the state is rounded, not the recurrence."""
    _compare(_case([2, 2], accepted=accepted, state_dtype=state_dtype))


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_production_shape_matches_reference():
    """Nemotron-3.5's MTP6 decode shape: B=8, T=7, H=64, D=64, N=128, G=8."""
    for accepted in (1, 4, 7):
        _compare(
            _case(
                [7] * 8,
                accepted=[accepted] * 8,
                heads=64,
                dim=64,
                dstate=128,
                groups=8,
                same_slots=True,
            )
        )


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("identical", [True, False])
@pytest.mark.parametrize("accepted", range(1, 8))
def test_flat_production_slot_tables_match_reference(accepted, identical):
    """The call ``--mamba-backend flashinfer`` dies on today: vLLM's MTP6 decode.

    ``mamba_mixer2`` passes ``x``/``dt``/``out`` of ``[total_tokens, heads, dim]``,
    flat 1-D ``state_batch_indices``/``dst_state_batch_indices``, an acceptance count
    and ``query_start_loc`` for one sequence of seven tokens at ``--max-num-seqs 1``.
    The tables carry one entry per packed row, so the accepted position is entry
    ``accepted - 1`` and token ``t`` publishes to entry ``t`` -- reading a 2-D table's
    ``[sequence, step]`` instead would decode from the wrong state, and refusing the
    flat form at all is what kept this configuration from starting.
    """
    case = _case([7], accepted=[accepted], flat=True)
    if identical:
        # The mixer hands the same table over as both source and destination when
        # prefix caching is off, so the chain moves forward in place.
        case["dst_slots"] = case["src_slots"]
    _compare(case)


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("steps", [1, 3, 7])
def test_flat_slot_tables_carry_the_plain_packed_call(batch, steps):
    """Without acceptance counts a flat table is one entry per sequence.

    Only entry ``sequence`` is read and written, so several multi-token sequences
    stay distinguishable -- which pins the sequence stride of the flat addressing.
    """
    _compare(_case([steps] * batch, flat=True))
    _compare(_case([steps] * batch, same_slots=True, flat=True))


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("null_entries", [((0, 0),), ((0, 6),)])
def test_null_entries_in_flat_slot_tables_match_reference(null_entries):
    """The sentinel reads as a zero state and is never written, in the flat form too."""
    _compare(
        _case(
            [7],
            accepted=[1],
            flat=True,
            null_src=null_entries,
            null_dst=null_entries,
        )
    )


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("flat", [True, False])
def test_absent_destination_table_publishes_to_the_read_table(flat):
    """With no destination table Triton aliases the read table onto the write side.

    Every token then publishes to its own entry of that table rather than to the slot
    the sequence started from, which is the behaviour to match: the read slot is only
    the *initial* state.
    """
    case = _case([4], accepted=[2], flat=flat)
    case["dst_slots"] = None
    expected_state = case["state"].clone()
    _run_reference(case, state=expected_state)
    # Entries 0 and 1 are the ones the accepted position and the token loop reach, so
    # a store that ignored the table would leave them at their initial values.
    assert not torch.equal(expected_state[1:3], case["state"][1:3])
    _compare(case)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_disable_state_update_keeps_the_packed_pool():
    case = _case([3, 3], accepted=[2, 3], disable_state_update=True)
    original = case["state"].clone()
    expected = _run_reference({**case, "state": original.clone()})
    actual = _run_kernel(case, state=original.clone())
    torch.testing.assert_close(
        actual.reshape(expected.shape), expected, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(case["state"], original)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_dispatch_keeps_the_one_token_kernel_for_the_plain_step(monkeypatch):
    """A plain decode step must not be sent through the packed kernel.

    vLLM's Mamba2 mixer makes this call on every decode step of every layer, and it
    passes query starts plus one slot per sequence. That shape has to keep reaching
    the kernel built for it; the packed kernel is for the multi-token calls.
    """
    calls: list[str] = []
    one_token = mate.mamba._ssu_launch()
    packed = mate.mamba._ssu_packed_launch()

    def _record(name: str, launcher):
        def _launch(*args, **kwargs):
            calls.append(name)
            return launcher(*args, **kwargs)

        return _launch

    monkeypatch.setattr(
        mate.mamba, "_ssu_launch", lambda: _record("one_token", one_token)
    )
    monkeypatch.setattr(
        mate.mamba, "_ssu_packed_launch", lambda: _record("packed", packed)
    )

    plain = _case([1, 1], same_slots=True)
    plain["src_slots"] = plain["src_slots"].reshape(-1)
    plain["dst_slots"] = None
    _run_kernel(plain)
    assert calls == ["one_token"], "the plain decode step left the one-token kernel"

    calls.clear()
    # One speculative step later: acceptance counts plus a per-token destination
    # table, so the state has to start from the accepted position.
    mtp = _case([1, 1], accepted=[1, 1], same_slots=True)
    _run_kernel(mtp)
    assert calls == ["packed"], "a one-token MTP call reached the one-token kernel"

    calls.clear()
    # The dense single-token form with one slot per sequence is the plain call too.
    dense = _case([1, 1], dense=True, same_slots=True)
    dense["src_slots"] = dense["src_slots"].reshape(-1)
    dense["dst_slots"] = None
    _run_kernel(dense)
    assert calls == ["one_token"], "the dense single-token call left the fast path"


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_one_token_call_agrees_across_the_two_kernels():
    """A one-token MTP call takes the packed kernel and must still match the fast path.

    That is the boundary the dispatch rule moves calls across, and the only shape
    where the two kernels can be compared directly.
    """
    vector = _case([1, 1], same_slots=True)
    vector["src_slots"] = vector["src_slots"].reshape(-1)
    vector["dst_slots"] = None
    table = _case([1, 1], accepted=[1, 1], same_slots=True)

    vector_state = vector["state"].clone()
    vector_out = _run_kernel(vector, state=vector_state)
    table_state = table["state"].clone()
    table_out = _run_kernel(table, state=table_state)

    torch.testing.assert_close(table_out, vector_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(table_state, vector_state, rtol=1e-2, atol=1e-3)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_prewarm_packed_then_run_uses_the_compiled_kernel():
    from mate.mamba import prewarm_selective_state_update

    prewarm_selective_state_update(
        state_dtype=STATE_DTYPE,
        io_dtype=IO_DTYPE,
        batch=2,
        heads=HEADS,
        dim=DIM,
        dstate=DSTATE,
        groups=GROUPS,
        slot_dtype=torch.int32,
        packed=True,
    )
    _compare(_case([3, 3], accepted=[2, 1]))


def test_packed_contract_is_validated_before_the_kernel():
    """A call that cannot express the packed contract fails instead of guessing.

    These run on CPU tensors: the validation happens before any kernel is built, so
    they cost nothing and they pin the errors a caller sees. The Triton kernel
    asserts the first two of them as well.
    """
    # A packed multi-token call with neither query starts nor a table that describes
    # the row split cannot be decoded.
    without_starts = _case([2, 2], device=torch.device("cpu"))
    without_starts["cu_seqlens"] = None
    with pytest.raises(ValueError, match="cu_seqlens"):
        _run_kernel(without_starts)

    # Acceptance needs a read mechanism, and that mechanism is the read table.
    without_table = _case([2, 2], accepted=[1, 1], device=torch.device("cpu"))
    without_table["src_slots"] = None
    with pytest.raises(ValueError, match="state_batch_indices"):
        _run_kernel(without_table)


def test_flat_slot_tables_are_normalized_rather_than_refused():
    """The tables are addressed by strides, so a 1-D table is a call, not an error.

    The serving path passes flat tables together with an acceptance count, and the
    kernel reaches position ``t`` of sequence ``b`` at entry ``b + t``: the strides the
    normalizer hands over are what say so. A 2-D table keeps its own row stride, so the
    two forms only differ in addressing -- with one sequence they name the same entries.
    """
    flat = torch.tensor([3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
    table, seq_stride, step_stride = mate.mamba._slot_table(
        flat, name="state_batch_indices"
    )
    assert table.data_ptr() == flat.data_ptr(), "the flat table was copied"
    assert (seq_stride, step_stride) == (1, 1)

    wide = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    table, seq_stride, step_stride = mate.mamba._slot_table(
        wide, name="state_batch_indices"
    )
    assert table.shape == (6,)
    assert (seq_stride, step_stride) == (3, 1)

    with pytest.raises(ValueError, match="flat"):
        mate.mamba._slot_table(
            torch.zeros(2, 2, 2, dtype=torch.int32), name="state_batch_indices"
        )
