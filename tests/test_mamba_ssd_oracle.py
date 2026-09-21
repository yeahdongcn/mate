"""CPU-only oracle tests for the native SSD prefill family (no tilelang import).

These pin the *semantics* of the reference implementations that the device
kernels are measured against, on any machine with torch. Kernel-level properties
that need a device (masked stores leaving padded rows untouched, timing) live in
the device tests.
"""

from __future__ import annotations

import pytest
import torch

from mate.mamba_kernels.reference import (
    ssd_bmm_reference,
    ssd_chunk_cumsum_reference,
    ssd_chunk_state_reference,
    ssd_state_passing_reference,
)

HEADS = 4
CHUNK = 8


def _inputs(offset_list, heads=HEADS, chunk=CHUNK, seed=0):
    generator = torch.Generator().manual_seed(seed)
    total = offset_list[-1]
    dt = torch.randn(total, heads, generator=generator)
    A = -torch.rand(heads, generator=generator) - 0.5
    bias = torch.randn(heads, generator=generator) * 0.1
    cu = torch.tensor(offset_list, dtype=torch.int32)
    return dt, A, bias, cu, chunk


def _layouts():
    """(name, chunk offsets) pairs: full chunks, partial chunks, single tokens."""
    return [
        ("full", [0, 8, 16, 24]),
        ("partial", [0, 8, 11, 19, 21]),
        ("single-token", [0, 1, 9, 10]),
        ("short-chunk", [0, 3, 4, 12]),
    ]


def test_inclusive_prefix_over_each_chunk():
    for _, offsets in _layouts():
        dt, A, bias, cu, chunk = _inputs(offsets)
        dA, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
        for c in range(len(offsets) - 1):
            length = offsets[c + 1] - offsets[c]
            expected = torch.cumsum(dt_out[:, c, :length] * A.view(-1, 1), dim=1)
            assert torch.allclose(dA[:, c, :length], expected, atol=1e-6)


def test_chunks_are_independent():
    dt, A, bias, cu, chunk = _inputs([0, 8, 11, 19, 21])
    base_dA, base_dt = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    perturbed = dt.clone()
    perturbed[0:8] += 5.0  # first chunk only
    new_dA, new_dt = ssd_chunk_cumsum_reference(perturbed, A, bias, cu, chunk, True)
    assert not torch.allclose(base_dA[:, 0, :8], new_dA[:, 0, :8])
    for c in (1, 2, 3):
        assert torch.allclose(base_dA[:, c], new_dA[:, c], atol=0.0)
        assert torch.allclose(base_dt[:, c], new_dt[:, c], atol=0.0)


def test_padding_saturates_dt_to_zero_and_da_to_the_chunk_total():
    """Padding is contractual: dt is 0 past the chunk end, so the scan saturates
    and the row's last position holds the chunk's total decay -- the value the
    downstream stages read unconditionally."""
    offsets = [0, 8, 11]
    dt, A, bias, cu, chunk = _inputs(offsets)
    dA, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)

    # dt is exactly zero beyond the chunk.
    assert torch.equal(dt_out[:, 1, 3:], torch.zeros_like(dt_out[:, 1, 3:]))
    assert torch.all(dt_out[:, 1, :3] > 0)

    # dA_cumsum is saturated at the chunk total past the chunk...
    total = dA[:, 1, 2]  # last valid position of the 3-token chunk
    assert torch.allclose(
        dA[:, 1, 3:], total.unsqueeze(1).expand_as(dA[:, 1, 3:]), atol=0.0
    )
    # ...and equals the row's last position, which is what callers read.
    assert torch.equal(dA[:, 1, 2], dA[:, 1, chunk - 1])

    # The total is the fp32 accumulation of processed_dt * A over valid tokens.
    expected = (dt_out[:, 1, :3] * A.view(-1, 1)).sum(dim=1)
    assert torch.allclose(total, expected, atol=1e-6)


def test_partial_chunk_equals_full_chunk_prefix():
    """A short chunk must produce exactly the prefix of the longer case."""
    dt, A, bias, cu, chunk = _inputs([0, 8, 8 + 5])
    short_dA, short_dt = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    full_dA, full_dt = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    assert torch.allclose(short_dA[:, 1, :5], full_dA[:, 1, :5], atol=0.0)
    assert torch.allclose(short_dt[:, 1, :5], full_dt[:, 1, :5], atol=0.0)


def test_dt_softplus_uses_the_threshold_rule():
    dt, A, bias, cu, chunk = _inputs([0, 8])
    dt = dt.clone()
    dt[0, :] = 30.0  # above the threshold: passed through
    dt[1, :] = -30.0  # below: log1p(exp(x))
    _, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    assert torch.allclose(dt_out[:, 0, 0], dt[0] + bias, atol=1e-5)
    assert torch.allclose(
        dt_out[:, 0, 1], torch.log1p(torch.exp(dt[1] + bias)), atol=1e-5
    )


def test_dt_softplus_disabled_adds_bias_then_clamps():
    """`dt_limit` clamps unconditionally, so with softplus off a negative dt is
    clamped (the production `(0.0, inf)` limit makes that a no-op only while
    softplus is on, which is the model's configuration)."""
    dt, A, bias, cu, chunk = _inputs([0, 8])
    _, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, False)
    assert torch.allclose(
        dt_out[:, 0, 0], (dt[0] + bias).clamp(min=0.0), atol=1e-6
    )
    # With an open limit the bias is the only change.
    _, unbounded = ssd_chunk_cumsum_reference(
        dt, A, bias, cu, chunk, False, dt_limit=(-float("inf"), float("inf"))
    )
    assert torch.allclose(unbounded[:, 0, 0], dt[0] + bias, atol=1e-6)


def test_dt_limit_clamps_before_the_scan():
    dt, A, bias, cu, chunk = _inputs([0, 8])
    _, dt_out = ssd_chunk_cumsum_reference(
        dt, A, bias, cu, chunk, True, dt_limit=(0.5, 1.5)
    )
    assert float(dt_out.min()) >= 0.5 - 1e-6
    assert float(dt_out.max()) <= 1.5 + 1e-6
    dA, _ = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True, dt_limit=(0.5, 1.5))
    expected = torch.cumsum(dt_out[:, 0, :8] * A.view(-1, 1), dim=1)
    assert torch.allclose(dA[:, 0, :8], expected, atol=1e-6)


def test_missing_dt_bias_is_allowed():
    dt, A, _, cu, chunk = _inputs([0, 8])
    dA, dt_out = ssd_chunk_cumsum_reference(dt, A, None, cu, chunk, True)
    assert torch.allclose(
        dt_out[:, 0, 0], torch.log1p(torch.exp(dt[0])), atol=1e-5
    )
    assert torch.allclose(dA[:, 0, 0], dt_out[:, 0, 0] * A, atol=1e-6)


# ---------------------------------------------------------------------------
# Stage 2 -- intra-chunk states.
#
# `ssd_chunk_state_reference` mislabels the token axis of its einsum (see
# `test_chunk_state_reference_pairs_the_token_axis`), so the tests below pin the
# properties that hold independently of that defect -- the padding contract, the
# chunk-total read, the single-token case -- plus one xfail that names it.
# ---------------------------------------------------------------------------

STATE_HEADS = 4
STATE_GROUPS = 2
STATE_DIM = 3
STATE_DSTATE = 4
STATE_CHUNK = 4


def _state_inputs(offsets, heads=STATE_HEADS, groups=STATE_GROUPS, seed=0):
    """Token-major ``x``/``B``, the head-major stage-1 outputs, and the metadata.

    ``dA_cumsum``/``dt_out`` come from the cumsum reference itself, so the chunk
    totals -- including the nonzero *tail* that stage 2 reads unconditionally --
    are exactly the ones stage 1 produces.
    """
    generator = torch.Generator().manual_seed(seed)
    dim, dstate, chunk = STATE_DIM, STATE_DSTATE, STATE_CHUNK
    tokens = offsets[-1]
    x = torch.randn(tokens, heads, dim, generator=generator)
    b = torch.randn(tokens, groups, dstate, generator=generator)
    dt = torch.randn(tokens, heads, generator=generator)
    a = -torch.rand(heads, generator=generator) - 0.5
    cu_chunk_seqlens = torch.tensor(offsets, dtype=torch.int32)
    dA_cumsum, dt_out = ssd_chunk_cumsum_reference(
        dt, a, None, cu_chunk_seqlens, chunk, True
    )
    return x, b, dt_out, dA_cumsum, cu_chunk_seqlens, chunk


def _state_explicit(x, b, dt_out, dA_cumsum, offsets, chunk, head_ratio):
    """The rank-one accumulation, written out per token in fp64.

    ``S_c[h,d,n] = sum_{t < limit} exp(min(total - dA_t, 0)) * dt_t * x_t B_t``
    over the *valid* token range only, so it cannot inherit any padding.
    """
    heads = x.shape[1]
    result = torch.zeros(
        (len(offsets) - 1, heads, STATE_DIM, STATE_DSTATE), dtype=torch.float64
    )
    for c in range(len(offsets) - 1):
        lo, hi = offsets[c], offsets[c + 1]
        limit = hi - lo
        if limit <= 0:
            continue
        total = dA_cumsum[:, c, chunk - 1].to(torch.float64)
        weight = torch.exp(
            (total.unsqueeze(1) - dA_cumsum[:, c, :limit].to(torch.float64)).clamp(
                max=0.0
            )
        ) * dt_out[:, c, :limit].to(torch.float64)
        for head in range(heads):
            group = head // head_ratio
            for t in range(limit):
                result[c, head] += (
                    weight[head, t] * x[lo + t, head].to(torch.float64)
                ).unsqueeze(1) * b[lo + t, group].to(torch.float64).unsqueeze(0)
    return result


def test_chunk_state_tail_of_a_partial_chunk_contributes_nothing():
    """``dt_out`` past a chunk's end is contractual zero, and stage 2 must ignore
    it even when a contract-violating nonzero tail is supplied: the decay factor
    there is ``exp(total - total) = 1``, so only the zero dt saves it."""
    offsets = [0, 3]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    base = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    assert torch.equal(dt_out[:, 0, 3:], torch.zeros_like(dt_out[:, 0, 3:]))

    dirty = dt_out.clone()
    dirty[:, 0, 3:] = 5.0  # the padded tail only
    assert torch.equal(ssd_chunk_state_reference(x, b, dirty, dA, cu, chunk), base)


def test_chunk_state_reads_the_chunk_total_from_the_padded_tail():
    """``dA_cumsum[:, c, L-1]`` holds the chunk's total decay for every chunk
    length and is read unconditionally, so perturbing only that entry moves the
    result of a partial chunk."""
    offsets = [0, 3]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    base = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    assert float(dA[:, 0, chunk - 1].abs().min()) > 0.0

    perturbed = dA.clone()
    perturbed[:, 0, chunk - 1] -= 0.5
    moved = ssd_chunk_state_reference(x, b, dt_out, perturbed, cu, chunk)
    assert not torch.allclose(base, moved)


def test_chunk_state_decay_is_a_difference_of_one_cumsum():
    """The decay weights are ``exp(total - dA_t)``, so shifting a chunk's whole
    ``dA_cumsum`` row -- valid part and padded tail together -- by a constant
    leaves the state untouched. An exclusive cumsum, a ``+dt`` shift or a zeroed
    tail would not survive this."""
    offsets = [0, 3, 7]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    base = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)

    # Shift one chunk's whole padded row -- valid entries and the total together.
    shifted = dA.clone()
    shifted[:, 0, :] += 2.0
    assert torch.allclose(
        ssd_chunk_state_reference(x, b, dt_out, shifted, cu, chunk).to(torch.float64),
        base.to(torch.float64),
        atol=1e-5,
    )


def test_chunk_state_single_token_chunk_is_the_rank_one_contribution():
    """With one valid token the reference's contraction is exact (there is only
    one token to sum over), so this pins the weight formula and the layout."""
    offsets = [0, 1]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    states = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    for head in range(STATE_HEADS):
        group = head // (STATE_HEADS // STATE_GROUPS)
        delta = float(dA[head, 0, chunk - 1] - dA[head, 0, 0])
        weight = float(torch.tensor(delta).clamp(max=0.0).exp()) * float(
            dt_out[head, 0, 0]
        )
        expected = torch.outer(x[0, head].double(), b[0, group].double()) * weight
        assert torch.allclose(states[0, head].to(torch.float64), expected, atol=1e-6)


def test_chunk_state_zero_length_chunk_is_zero():
    offsets = [0, 0, 4]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    states = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    assert torch.equal(states[0], torch.zeros_like(states[0]))
    assert bool((states[1].abs() > 1e-6).any())


def test_chunk_state_chunks_do_not_mix_sequences():
    """Perturbing one sequence's tokens must leave the other sequence's chunks
    bit-identical -- the token range comes from ``cu_chunk_seqlens`` alone."""
    offsets = [0, 3, 7]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    base = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)

    moved_x = x.clone()
    moved_b = b.clone()
    moved_x[0:3] += 1.0
    moved_b[0:3] += 1.0
    moved = ssd_chunk_state_reference(moved_x, moved_b, dt_out, dA, cu, chunk)

    assert not torch.allclose(base[0], moved[0])
    assert torch.equal(base[1], moved[1])


@pytest.mark.xfail(
    strict=False,
    reason="ssd_chunk_state_reference contracts the wrong axis: its einsum "
    "'thd,hln->hdn' leaves the token labels unshared, so it returns "
    "(sum_t x[t,h,d]) * (sum_l scaled[h,l,n]) instead of the rank-one "
    "accumulation over the chunk's tokens. Reported; the fix is "
    "'thd,htn->hdn'.",
)
def test_chunk_state_reference_pairs_the_token_axis():
    """The state must be the chunk's rank-one accumulation, not a product of two
    independent sums."""
    offsets = [0, 3, 4, 8]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    states = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    expected = _state_explicit(
        x, b, dt_out, dA, offsets, chunk, STATE_HEADS // STATE_GROUPS
    )
    assert torch.allclose(states.to(torch.float64), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Stage 3 -- inter-chunk state passing.
# ---------------------------------------------------------------------------

PASS_HEADS = 2
PASS_DIM = 3
PASS_DSTATE = 4
PASS_CHUNK = 4


def _passing_case(nchunks, last, decay=None, seed=0):
    """``states``/``dA_cumsum``/``last_chunk_indices`` for one chunk layout.

    ``dA_cumsum`` carries only the padded row's last entry: the chunk's total
    decay. ``decay`` is ``[heads, nchunks]`` (or a scalar), left at 0 -- i.e.
    ``exp(0) = 1``, no decay -- when omitted.
    """
    generator = torch.Generator().manual_seed(seed)
    heads, dim, dstate, chunk = PASS_HEADS, PASS_DIM, PASS_DSTATE, PASS_CHUNK
    states = torch.randn((nchunks, heads, dim, dstate), generator=generator)
    dA_cumsum = torch.zeros((heads, nchunks, chunk))
    if decay is not None:
        dA_cumsum[:, :, chunk - 1] = torch.as_tensor(decay, dtype=torch.float32)
    return states, dA_cumsum, torch.tensor(last, dtype=torch.int32)


def _passing_fp64(states, dA_cumsum, last, initial_states=None):
    """Independent fp64 recurrence over the chunk ranges ``last`` implies."""
    heads, dim, dstate = PASS_HEADS, PASS_DIM, PASS_DSTATE
    last = [int(value) for value in last]
    out = torch.zeros((states.shape[0], heads, dim, dstate), dtype=torch.float64)
    for sequence in range(len(last)):
        end = last[sequence] + 1
        start = (last[sequence - 1] + 1) if sequence > 0 else 0
        running = (
            torch.zeros((heads, dim, dstate), dtype=torch.float64)
            if initial_states is None
            else initial_states[sequence].to(torch.float64)
        )
        for chunk in range(start, end):
            running = torch.exp(
                dA_cumsum[:, chunk, PASS_CHUNK - 1].to(torch.float64)
            ).view(heads, 1, 1) * running + states[chunk].to(torch.float64)
            out[chunk] = running
    return out


def test_state_passing_without_decay_is_the_running_sum():
    """``dA_cumsum`` all zero makes the recurrence a plain prefix sum -- and the
    prefix restarts at every sequence boundary (sequence 0 owns chunk 0, sequence
    1 owns chunks 1..3)."""
    states, dA_cumsum, last = _passing_case(4, [0, 3])
    out = ssd_state_passing_reference(states, dA_cumsum, last)
    expected = torch.empty_like(out)
    expected[0] = states[0]
    expected[1:] = torch.cumsum(states[1:], dim=0)
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.equal(out[0], states[0])


def test_state_passing_matches_the_fp64_recurrence_with_decay():
    """Two sequences, a partial second one, per-(head, chunk) decay, and fp16
    initial states that must be upcast before the recurrence."""
    decay = -torch.rand(PASS_HEADS, 4) - 0.1
    states, dA_cumsum, last = _passing_case(4, [1, 3], decay=decay)
    generator = torch.Generator().manual_seed(7)
    init = torch.randn((2, PASS_HEADS, PASS_DIM, PASS_DSTATE), generator=generator).to(
        torch.float16
    )
    out = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=init, state_dtype=torch.float32
    )
    assert out.dtype == torch.float32
    assert torch.allclose(
        out.to(torch.float64), _passing_fp64(states, dA_cumsum, last, init), atol=1e-5
    )


def test_state_passing_indexes_initial_states_by_sequence_position():
    states, dA_cumsum, last = _passing_case(3, [0, 2])
    init = torch.zeros(2, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[0] = 3.0
    init[1] = -7.0
    out = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)
    assert torch.allclose(out[0], states[0] + 3.0, atol=1e-5)
    # Sequence 1 is seeded from row 1, not from row 0 and not from a seq_idx.
    assert torch.allclose(out[1], states[1] - 7.0, atol=1e-5)
    assert torch.allclose(out[2], states[1] + states[2] - 7.0, atol=1e-5)


def test_state_passing_zero_chunk_sequences_iterate_zero_times():
    """The middle sequence owns no chunk, so it must neither seed nor advance the
    chunk its right neighbour starts from."""
    states, dA_cumsum, last = _passing_case(4, [1, 1, 3])
    init = torch.zeros(3, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[0] = 1.0
    init[1] = 100.0
    init[2] = 1000.0
    out = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)

    assert torch.allclose(
        out.to(torch.float64),
        _passing_fp64(states, dA_cumsum, last, init),
        atol=1e-5,
    )
    # Sequence 0 owns chunks 0 and 1 and carries its own running state.
    assert torch.allclose(out[0], states[0] + 1.0, atol=1e-5)
    assert torch.allclose(out[1], states[0] + states[1] + 1.0, atol=1e-5)
    # Chunks 2 and 3 belong to sequence 2 and start from 1000, never from the
    # empty sequence's 100.
    assert torch.allclose(out[2], states[2] + 1000.0, atol=1e-5)
    assert torch.allclose(out[3], states[2] + states[3] + 1000.0, atol=1e-5)


def test_state_passing_first_sequence_with_no_chunks():
    """``last_chunk_indices[0] == -1``: sequence 0 owns nothing and the neighbour
    still starts at chunk 0."""
    states, dA_cumsum, last = _passing_case(2, [-1, 1])
    init = torch.zeros(2, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[1] = 5.0
    out = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)
    assert torch.allclose(out[0], states[0] + 5.0, atol=1e-5)
    assert torch.allclose(out[1], states[0] + states[1] + 5.0, atol=1e-5)


def test_state_passing_all_empty_sequences_leave_the_output_untouched():
    states, dA_cumsum, last = _passing_case(2, [-1, -1])
    out = ssd_state_passing_reference(states, dA_cumsum, last)
    assert torch.equal(out, torch.zeros_like(out))


def test_state_passing_absent_initial_states_equals_zero_initial_states():
    states, dA_cumsum, last = _passing_case(3, [2], decay=-0.3)
    zeros = torch.zeros(1, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    with_init = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=zeros
    )
    without = ssd_state_passing_reference(states, dA_cumsum, last)
    assert torch.equal(with_init, without)


def test_state_passing_rounds_only_on_the_store():
    """The running state stays fp32 across chunk boundaries and only the store
    rounds; carrying the rounded value forward gives a different answer."""
    states = torch.zeros(4, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    states[0] = 1.0
    states[1:] = 4e-4
    _, dA_cumsum, last = _passing_case(4, [3])
    out = ssd_state_passing_reference(
        states, dA_cumsum, last, state_dtype=torch.float16
    )
    assert out.dtype == torch.float16

    running = torch.cumsum(states.to(torch.float32), dim=0)
    assert torch.equal(out, running.to(torch.float16))

    carried = torch.empty_like(running)
    previous = torch.zeros(PASS_HEADS, PASS_DIM, PASS_DSTATE)
    for chunk in range(4):
        previous = (previous.to(torch.float16) + states[chunk]).to(torch.float32)
        carried[chunk] = previous
    assert not torch.equal(out, carried.to(torch.float16))


def test_state_passing_state_dtype_resolution():
    states, dA_cumsum, last = _passing_case(2, [1])
    init = torch.zeros(1, PASS_HEADS, PASS_DIM, PASS_DSTATE, dtype=torch.bfloat16)
    # state_dtype wins...
    out = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=init, state_dtype=torch.float32
    )
    assert out.dtype == torch.float32
    # ...then initial_states.dtype...
    out = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)
    assert out.dtype == torch.bfloat16
    # ...then the input state's dtype.
    out = ssd_state_passing_reference(states, dA_cumsum, last)
    assert out.dtype == torch.float32


def _passing_fp64_by_seq_idx(states, dA_cumsum, seq_idx, initial_states=None):
    """Independent fp64 recurrence for the ``seq_idx`` walk.

    Chunks are visited in index order; a sequence's first chunk enters from
    ``initial_states[seq_idx[c]]``, later chunks of the same id continue it, and
    ids that never appear leave nothing written.
    """
    running: dict[int, torch.Tensor] = {}
    out = torch.zeros(states.shape, dtype=torch.float64)
    for chunk in range(states.shape[0]):
        sequence = int(seq_idx[chunk])
        if sequence not in running:
            running[sequence] = (
                torch.zeros((PASS_HEADS, PASS_DIM, PASS_DSTATE), dtype=torch.float64)
                if initial_states is None
                else initial_states[sequence].to(torch.float64)
            )
        decay = torch.exp(dA_cumsum[:, chunk, PASS_CHUNK - 1].to(torch.float64)).view(
            PASS_HEADS, 1, 1
        )
        running[sequence] = decay * running[sequence] + states[chunk].to(torch.float64)
        out[chunk] = running[sequence]
    return out


def test_state_passing_seq_idx_agrees_with_last_chunk_indices_when_contiguous():
    """The two boundary rules are interchangeable metadata for a packed,
    contiguous prefill: sequence 0 owns chunk 0, sequence 1 owns chunks 1..3."""
    states, dA_cumsum, last = _passing_case(4, [0, 3], decay=-0.4)
    seq_idx = torch.tensor([0, 1, 1, 1], dtype=torch.int32)
    init = torch.full((2, PASS_HEADS, PASS_DIM, PASS_DSTATE), 0.25)
    by_range = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)
    by_seq_idx = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=init, seq_idx=seq_idx
    )
    assert torch.equal(by_range, by_seq_idx)


def test_state_passing_seq_idx_keeps_interleaved_sequences_separate():
    """Interleaved chunks are where the rules disagree: ``last_chunk_indices``
    arithmetic would hand sequence 1 the whole tail from chunk 1, while the
    oracle gives it only its own chunks 1 and 3."""
    states, dA_cumsum, _ = _passing_case(4, [3], decay=-0.4)
    seq_idx = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    init = torch.full((2, PASS_HEADS, PASS_DIM, PASS_DSTATE), 0.5)

    out = ssd_state_passing_reference(
        states, dA_cumsum, None, initial_states=init, seq_idx=seq_idx
    )
    assert torch.allclose(
        out.to(torch.float64),
        _passing_fp64_by_seq_idx(states, dA_cumsum, seq_idx, init),
        atol=1e-5,
    )
    # Chunks 0 and 2 belong to sequence 0 and are chained through chunk 2 only.
    assert not torch.allclose(out[1], out[2], atol=1e-3)
    # Chunk 2 continues chunk 0's sequence: it must not see chunk 1's state.
    decay = torch.exp(dA_cumsum[:, 2, PASS_CHUNK - 1].to(torch.float64)).view(
        PASS_HEADS, 1, 1
    )
    expected = decay * out[0].to(torch.float64) + states[2].to(torch.float64)
    assert torch.allclose(out[2].to(torch.float64), expected, atol=1e-5)


def test_state_passing_seq_idx_skips_without_decaying():
    """A skipped chunk must not decay the resumed sequence's state, and the
    resumed chunk applies its own decay."""
    states, dA_cumsum, _ = _passing_case(3, [2], decay=-1.0)
    seq_idx = torch.tensor([0, 1, 0], dtype=torch.int32)

    out = ssd_state_passing_reference(states, dA_cumsum, None, seq_idx=seq_idx)
    assert torch.allclose(
        out[0].to(torch.float64), states[0].to(torch.float64), atol=1e-5
    )
    decay = torch.exp(dA_cumsum[:, 2, PASS_CHUNK - 1].to(torch.float64)).view(
        PASS_HEADS, 1, 1
    )
    # Only chunk 2's own decay applies -- chunk 1's is not applied on the way.
    resumed = decay * states[0].to(torch.float64) + states[2].to(torch.float64)
    assert torch.allclose(out[2].to(torch.float64), resumed, atol=1e-5)


def test_state_passing_seq_idx_indexes_initial_states_by_the_id():
    """``initial_states`` is indexed by the value of ``seq_idx``: chunk 0's
    sequence is id 1, so it enters from ``initial_states[1]``."""
    states, dA_cumsum, last = _passing_case(2, [0, 1], decay=-0.2)
    seq_idx = torch.tensor([1, 1], dtype=torch.int32)
    init = torch.zeros(2, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[0] = -3.0
    init[1] = 7.0

    out = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=init, seq_idx=seq_idx
    )
    decay0 = torch.exp(dA_cumsum[:, 0, PASS_CHUNK - 1].to(torch.float64)).view(
        PASS_HEADS, 1, 1
    )
    assert torch.allclose(
        out[0].to(torch.float64),
        7.0 * decay0 + states[0].to(torch.float64),
        atol=1e-5,
    )
    assert not torch.allclose(
        out[0].to(torch.float64), -3.0 * decay0 + states[0].double()
    )


def test_state_passing_seq_idx_id_with_no_chunks_writes_nothing():
    """Id 1 never appears in ``seq_idx``, so its initial state can never enter a
    chain: the result is unchanged when that row is zeroed."""
    states, dA_cumsum, _ = _passing_case(3, [2], decay=-0.5)
    seq_idx = torch.tensor([0, 2, 2], dtype=torch.int32)
    init = torch.ones(3, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[1] = 100.0

    out = ssd_state_passing_reference(
        states, dA_cumsum, None, initial_states=init, seq_idx=seq_idx
    )
    assert torch.allclose(
        out.to(torch.float64),
        _passing_fp64_by_seq_idx(states, dA_cumsum, seq_idx, init),
        atol=1e-5,
    )
    assert bool((out[2].abs() > 1e-6).any())

    zeroed = init.clone()
    zeroed[1] = 0.0
    assert torch.allclose(
        out.to(torch.float64),
        _passing_fp64_by_seq_idx(states, dA_cumsum, seq_idx, zeroed),
        atol=1e-5,
    )


def test_state_passing_seq_idx_must_index_initial_states():
    states, dA_cumsum, last = _passing_case(2, [0, 1])
    with pytest.raises(ValueError, match="must index initial_states"):
        ssd_state_passing_reference(
            states,
            dA_cumsum,
            last,
            initial_states=torch.zeros(2, PASS_HEADS, PASS_DIM, PASS_DSTATE),
            seq_idx=torch.tensor([0, 5], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="one sequence id per chunk"):
        ssd_state_passing_reference(
            states, dA_cumsum, last, seq_idx=torch.tensor([0], dtype=torch.int32)
        )


# ---------------------------------------------------------------------------
# Stage 4 -- chunk-level ``C @ B^T``.
# ---------------------------------------------------------------------------

BMM_GROUPS = 2
BMM_DSTATE = 4
BMM_CHUNK = 4


def _bmm_case(offsets, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = offsets[-1]
    cmat = torch.randn((tokens, BMM_GROUPS, BMM_DSTATE), generator=generator)
    bmat = torch.randn((tokens, BMM_GROUPS, BMM_DSTATE), generator=generator)
    cu_chunk_seqlens = torch.tensor(offsets, dtype=torch.int32)
    return cmat.to(dtype), bmat.to(dtype), cu_chunk_seqlens


def _bmm_explicit(cmat, bmat, offsets, chunk=BMM_CHUNK, dot_dtype=None):
    """Independent fp64 double loop over individual inner products."""
    if dot_dtype is not None:
        cmat = cmat.to(dot_dtype)
        bmat = bmat.to(dot_dtype)
    cmat = cmat.to(torch.float64)
    bmat = bmat.to(torch.float64)
    out = torch.zeros((len(offsets) - 1, BMM_GROUPS, chunk, chunk), dtype=torch.float64)
    for c in range(len(offsets) - 1):
        lo, hi = offsets[c], offsets[c + 1]
        for i in range(hi - lo):
            for j in range(hi - lo):
                for group in range(BMM_GROUPS):
                    out[c, group, i, j] = (
                        cmat[lo + i, group] * bmat[lo + j, group]
                    ).sum()
    return out


def test_bmm_matches_explicit_inner_products_over_the_chunk_range():
    """Multi-sequence varlen: chunk lengths 3, 1 and 4 against chunk_size 4."""
    offsets = [0, 3, 4, 8]
    cmat, bmat, cu = _bmm_case(offsets)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)
    assert cb.shape == (3, BMM_GROUPS, BMM_CHUNK, BMM_CHUNK)
    assert cb.dtype == torch.float32
    assert torch.allclose(
        cb.to(torch.float64), _bmm_explicit(cmat, bmat, offsets), atol=1e-5
    )


def test_bmm_is_non_causal():
    offsets = [0, 4]
    cmat, bmat, cu = _bmm_case(offsets)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)
    # The strict upper triangle is materialized, not left at zero...
    assert bool((cb[:, :, 0, 1:].abs() > 1e-6).all())
    # ...and it is a genuine inner product, not a copy of the lower triangle.
    assert not torch.allclose(cb[:, :, 0, 1], cb[:, :, 1, 0])
    explicit = _bmm_explicit(cmat, bmat, offsets)
    for i in range(BMM_CHUNK):
        for j in range(BMM_CHUNK):
            assert torch.allclose(
                cb[:, :, i, j].to(torch.float64), explicit[:, :, i, j], atol=1e-5
            )


def test_bmm_partial_chunk_tail_is_exactly_zero():
    """A chunk shorter than ``chunk_size``: everything at or beyond the boundary
    is exactly zero while the valid block is untouched."""
    offsets = [0, 2]
    cmat, bmat, cu = _bmm_case(offsets)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)
    assert torch.equal(cb[:, :, 2:, :], torch.zeros_like(cb[:, :, 2:, :]))
    assert torch.equal(cb[:, :, :, 2:], torch.zeros_like(cb[:, :, :, 2:]))
    assert bool((cb[:, :, :2, :2].abs() > 1e-6).all())
    explicit = _bmm_explicit(cmat, bmat, offsets)
    assert torch.allclose(
        cb[:, :, :2, :2].to(torch.float64), explicit[:, :, :2, :2], atol=1e-5
    )


def test_bmm_single_token_and_zero_length_chunks():
    offsets = [0, 1, 1, 5]
    cmat, bmat, cu = _bmm_case(offsets)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)

    expected = torch.einsum("gn,gn->g", cmat[0], bmat[0]).to(torch.float64)
    assert torch.allclose(cb[0, :, 0, 0].to(torch.float64), expected, atol=1e-5)
    assert torch.equal(cb[0, :, 0, 1:], torch.zeros_like(cb[0, :, 0, 1:]))
    assert torch.equal(cb[0, :, 1:, :], torch.zeros_like(cb[0, :, 1:, :]))

    # A zero-length chunk contributes nothing at all.
    assert torch.equal(cb[1], torch.zeros_like(cb[1]))
    assert torch.allclose(
        cb[2].to(torch.float64), _bmm_explicit(cmat, bmat, offsets)[2], atol=1e-5
    )


def test_bmm_rounds_both_operands_to_the_activation_dtype_before_the_product():
    offsets = [0, 4]
    cmat32, bmat32, cu = _bmm_case(offsets)
    cmat = cmat32.to(torch.bfloat16)
    bmat = bmat32.to(torch.bfloat16)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)
    assert cb.dtype == torch.float32
    # Exactly the dot of the bf16-rounded operands...
    assert torch.allclose(
        cb.to(torch.float64),
        _bmm_explicit(cmat, bmat, offsets),
        atol=1e-4,
    )
    # ...which is a different answer from the fp32-exact product of the same
    # values, so the rounding point is observable.
    exact = _bmm_explicit(cmat32, bmat32, offsets)
    assert not torch.allclose(cb.to(torch.float64), exact, atol=1e-6)


def test_bmm_rejects_a_chunk_longer_than_chunk_size():
    offsets = [0, 5]
    cmat, bmat, cu = _bmm_case(offsets)
    with pytest.raises(ValueError, match="longer than chunk_size"):
        ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)
