"""CPU-only semantics tests for the multi-token SSU oracle.

These need no MUSA device: the multi-token oracle must reduce to the validated
one-token oracle when the token loop is written out explicitly, and it must honour
the slot regimes the consumers rely on. Keeping them in their own module avoids
importing the TileLang kernel module, so they run anywhere.
"""

from __future__ import annotations

import torch

from mate.mamba_kernels.reference import (
    selective_state_update_multi_token_reference,
    selective_state_update_one_token_reference,
)

HEADS, DIM, DSTATE, GROUPS, SLOTS, PAD = 4, 3, 8, 2, 8, -1
# An fp32 state pool keeps the token-by-token chain exact: the recurrence carries the
# state in registers between tokens, so rounding the pool between steps would differ
# by rounding only (covered by the last test in this module).
POOL_DTYPE = torch.float32
IO_DTYPE = torch.bfloat16


def _make_inputs(rows: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    return {
        "x": torch.randn(rows, HEADS, DIM, generator=generator).to(IO_DTYPE).to(device),
        "dt": (torch.rand(rows, HEADS, DIM, generator=generator) * 0.3 + 0.01).to(
            device
        ),
        "B": torch.randn(rows, GROUPS, DSTATE, generator=generator)
        .to(IO_DTYPE)
        .to(device),
        "C": torch.randn(rows, GROUPS, DSTATE, generator=generator)
        .to(IO_DTYPE)
        .to(device),
        "z": torch.randn(rows, HEADS, DIM, generator=generator).to(IO_DTYPE).to(device),
        "A": (-torch.rand(HEADS, generator=generator) - 0.1).to(device),
        "D": torch.randn(HEADS, generator=generator).to(device),
        "bias": (torch.randn(HEADS, generator=generator) * 0.1).to(device),
    }


def _make_pool(dtype: torch.dtype = POOL_DTYPE) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return (torch.randn(SLOTS, HEADS, DIM, DSTATE, generator=generator) * 0.5).to(dtype)


def _one_token(
    pool: torch.Tensor,
    inp: dict[str, torch.Tensor],
    row: int,
    src: int,
    dst: int | None,
    softplus: bool = False,
    disable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-token oracle on a single row, keeping the batch dimension."""
    state = pool.clone()
    y = selective_state_update_one_token_reference(
        state,
        inp["x"][row : row + 1],
        inp["dt"][row : row + 1],
        inp["A"],
        inp["B"][row : row + 1],
        inp["C"][row : row + 1],
        inp["D"],
        inp["z"][row : row + 1],
        inp["bias"],
        softplus,
        torch.tensor([src]),
        None if dst is None else torch.tensor([dst]),
        PAD,
        disable,
    )
    return y[0], state


def _sequential(
    pool: torch.Tensor,
    inp: dict[str, torch.Tensor],
    chains: list[list[tuple[int, int, int | None]]],
    softplus: bool = False,
    disable: bool = False,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Run the one-token oracle token by token over explicit (row, src, dst) chains."""
    outputs: dict[int, torch.Tensor] = {}
    state = pool.clone()
    for chain in chains:
        for row, src, dst in chain:
            y, state = _one_token(state, inp, row, src, dst, softplus, disable)
            outputs[row] = y
    return outputs, state


def _multi(
    pool: torch.Tensor,
    inp: dict[str, torch.Tensor],
    steps: list[int],
    row_base: list[int],
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor | None,
    softplus: bool = False,
    disable: bool = False,
    accepted: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = pool.clone()
    y = selective_state_update_multi_token_reference(
        state,
        inp["x"],
        inp["dt"],
        inp["A"],
        inp["B"],
        inp["C"],
        inp["D"],
        inp["z"],
        inp["bias"],
        softplus,
        src_slots,
        dst_slots,
        PAD,
        disable,
        steps,
        row_base,
        accepted,
    )
    return y, state


def test_single_step_matches_one_token_oracle() -> None:
    inp, pool = _make_inputs(3), _make_pool()
    y, state = _multi(pool, inp, [1, 1, 1], [0, 1, 2], torch.tensor([1, 3, 5]), None)
    expected_y, expected_state = _sequential(
        pool, inp, [[(0, 1, None)], [(1, 3, None)], [(2, 5, None)]]
    )
    for row in range(3):
        assert torch.equal(y[row], expected_y[row])
    assert torch.equal(state, expected_state)


def test_mtp_plain_writes_the_read_slot_once() -> None:
    inp, pool = _make_inputs(2), _make_pool()
    y, state = _multi(pool, inp, [2], [0], torch.tensor([2]), None)
    expected_y, expected_state = _sequential(pool, inp, [[(0, 2, None), (1, 2, None)]])
    assert torch.equal(y[0], expected_y[0])
    assert torch.equal(y[1], expected_y[1])
    assert torch.equal(state, expected_state)


def test_packed_varlen_writes_the_final_slot_once() -> None:
    inp, pool = _make_inputs(3), _make_pool()
    y, state = _multi(
        pool, inp, [2, 1], [0, 2], torch.tensor([6, 7]), torch.tensor([6, 7])
    )
    expected_y, expected_state = _sequential(
        pool, inp, [[(0, 6, 6), (1, 6, 6)], [(2, 7, 7)]]
    )
    for row in range(3):
        assert torch.equal(y[row], expected_y[row])
    assert torch.equal(state, expected_state)


def test_speculative_decoding_chains_the_destination_slots() -> None:
    inp, pool = _make_inputs(2), _make_pool()
    y, state = _multi(
        pool,
        inp,
        [2],
        [0],
        torch.tensor([1]),
        torch.tensor([[4, 5]]),
        accepted=torch.tensor([1]),
    )
    expected_y, expected_state = _sequential(pool, inp, [[(0, 1, 4), (1, 4, 5)]])
    assert torch.equal(y[0], expected_y[0])
    assert torch.equal(y[1], expected_y[1])
    assert torch.equal(state, expected_state)
    # The source slot is only read: the chain moves forward slot by slot.
    assert torch.equal(state[1], pool[1])


def test_num_accepted_tokens_indexes_the_read_slot() -> None:
    inp, pool = _make_inputs(2), _make_pool()
    slots = torch.tensor([[0, 3]])
    # The consumer passes the accepted-token count, and the state is read at
    # ``count - 1`` so that the accepted token itself is the starting state.
    for count, start in ((1, 0), (2, 3)):
        y, state = _multi(
            pool, inp, [2], [0], slots, None, accepted=torch.tensor([count])
        )
        expected_y, expected_state = _sequential(
            pool, inp, [[(0, start, None), (1, start, None)]]
        )
        assert torch.equal(y[0], expected_y[0])
        assert torch.equal(y[1], expected_y[1])
        assert torch.equal(state, expected_state)


def test_pad_slots_disable_state_update_and_empty_sequences() -> None:
    inp, pool = _make_inputs(2), _make_pool()

    # A padded source reads as a zero state and a padded destination is never written.
    zero_pool = pool.clone()
    zero_pool[0] = 0.0
    y_pad, state_pad = _multi(
        pool, inp, [2], [0], torch.tensor([PAD]), torch.tensor([PAD])
    )
    expected_y, _ = _sequential(zero_pool, inp, [[(0, 0, 0), (1, 0, 0)]])
    assert torch.equal(state_pad, pool)
    assert torch.equal(y_pad[0], expected_y[0])
    assert torch.equal(y_pad[1], expected_y[1])

    # disable_state_update and empty sequences leave the pool untouched.
    _, state_disabled = _multi(
        pool, inp, [2], [0], torch.tensor([1]), torch.tensor([1]), disable=True
    )
    _, expected_disabled = _sequential(
        pool, inp, [[(0, 1, 1), (1, 1, 1)]], disable=True
    )
    assert torch.equal(state_disabled, expected_disabled)
    _, state_empty = _multi(pool, inp, [0], [0], torch.tensor([1]), torch.tensor([1]))
    assert torch.equal(state_empty, pool)


def test_optional_features_match_the_sequential_chain() -> None:
    inp, pool = _make_inputs(2), _make_pool()
    # dt_bias and dt_softplus are active in both runs; D and z are always passed.
    for softplus in (False, True):
        y, state = _multi(
            pool, inp, [2], [0], torch.tensor([1]), torch.tensor([1]), softplus=softplus
        )
        expected_y, expected_state = _sequential(
            pool, inp, [[(0, 1, 1), (1, 1, 1)]], softplus=softplus
        )
        assert torch.equal(y[0], expected_y[0])
        assert torch.equal(y[1], expected_y[1])
        assert torch.equal(state, expected_state)


def test_rounding_state_pool_differs_by_rounding_only() -> None:
    inp = _make_inputs(2)
    pool = _make_pool(torch.float16)
    y, state = _multi(pool, inp, [2], [0], torch.tensor([2]), None)
    expected_y, expected_state = _sequential(pool, inp, [[(0, 2, None), (1, 2, None)]])
    scale = float(expected_state[2].abs().max())
    assert torch.allclose(
        state[2].to(torch.float32),
        expected_state[2].to(torch.float32),
        atol=8 * 2**-11 * max(scale, 1.0),
    )
    assert torch.allclose(
        y[1].to(torch.float32), expected_y[1].to(torch.float32), atol=2e-2
    )
