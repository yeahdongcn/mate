"""Tests for the native mamba/SSU family.

The contract test pins the frozen argument list shared with the FlashInfer-shaped
compatibility wrapper. The numeric tests compare the tilelang kernel against the
fp32 oracle on real MUSA hardware, and the gap tests assert that unsupported
paths fail loudly instead of silently narrowing semantics.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import mate.mamba
from mate.mamba_kernels.reference import selective_state_update_one_token_reference
from mate.testing import supported_musa_compute_capability

# Frozen argument list: the wrapper forwards exactly these, in this order.
CONTRACT_PARAMETERS = (
    "state",
    "x",
    "dt",
    "A",
    "B",
    "C",
    "D",
    "z",
    "dt_bias",
    "dt_softplus",
    "state_batch_indices",
    "pad_slot_id",
    "state_scale",
    "out",
    "disable_state_update",
    "intermediate_states_buffer",
    "intermediate_state_indices",
    "intermediate_state_scales",
    "rand_seed",
    "philox_rounds",
    "cache_steps",
    "algorithm",
    "dst_state_batch_indices",
    "cu_seqlens",
    "num_accepted_tokens",
    "backend",
)

_HEADS = 8
_DIM = 8
_DSTATE = 128
_GROUPS = 2
_BATCH = 2
_SLOTS = 4


def test_selective_state_update_contract():
    parameters = inspect.signature(mate.mamba.selective_state_update).parameters
    assert tuple(parameters) == CONTRACT_PARAMETERS
    assert parameters["state"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["pad_slot_id"].default == -1
    assert parameters["philox_rounds"].default == 10
    assert parameters["algorithm"].default == "auto"
    assert parameters["backend"].default == "auto"


def _device() -> torch.device:
    return torch.device("musa")


def _case(
    *,
    with_D: bool = True,
    with_z: bool = False,
    with_bias: bool = True,
    softplus: bool = False,
    disable_state_update: bool = False,
    pad_slot_id: int = -1,
    src_slots: tuple[int, ...] = (0, 1),
    dst_slots: tuple[int, ...] | None = None,
):
    device = _device()
    torch.manual_seed(0)
    state = torch.randn(
        _SLOTS, _HEADS, _DIM, _DSTATE, dtype=torch.float32, device=device
    ).to(torch.float16)
    x = torch.randn(_BATCH, _HEADS, _DIM, dtype=torch.float32, device=device).to(
        torch.bfloat16
    )
    dt = torch.rand(_BATCH, _HEADS, _DIM, dtype=torch.float32, device=device) * 0.1
    A = -torch.rand(_HEADS, dtype=torch.float32, device=device)
    B = torch.randn(_BATCH, _GROUPS, _DSTATE, dtype=torch.float32, device=device).to(
        torch.bfloat16
    )
    C = torch.randn(_BATCH, _GROUPS, _DSTATE, dtype=torch.float32, device=device).to(
        torch.bfloat16
    )
    D = torch.rand(_HEADS, dtype=torch.float32, device=device) if with_D else None
    dt_bias = (
        torch.rand(_HEADS, dtype=torch.float32, device=device) if with_bias else None
    )
    z = (
        torch.randn(_BATCH, _HEADS, _DIM, dtype=torch.float32, device=device).to(
            torch.bfloat16
        )
        if with_z
        else None
    )
    src = torch.tensor(src_slots, dtype=torch.int32, device=device)
    dst = (
        torch.tensor(dst_slots, dtype=torch.int32, device=device) if dst_slots else None
    )
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
        "dt_softplus": softplus,
        "src_slots": src,
        "dst_slots": dst,
        "pad_slot_id": pad_slot_id,
        "disable_state_update": disable_state_update,
    }


def _run_kernel(case, *, out=None):
    return mate.mamba.selective_state_update(
        case["state"],
        case["x"],
        case["dt"],
        case["A"],
        case["B"],
        case["C"],
        case["D"],
        z=case["z"],
        dt_bias=case["dt_bias"],
        dt_softplus=case["dt_softplus"],
        state_batch_indices=case["src_slots"],
        dst_state_batch_indices=case["dst_slots"],
        pad_slot_id=case["pad_slot_id"],
        disable_state_update=case["disable_state_update"],
        out=out,
    )


def _run_reference(case):
    return selective_state_update_one_token_reference(
        case["state"],
        case["x"],
        case["dt"],
        case["A"],
        case["B"],
        case["C"],
        case["D"],
        case["z"],
        case["dt_bias"],
        case["dt_softplus"],
        case["src_slots"],
        case["dst_slots"],
        case["pad_slot_id"],
        case["disable_state_update"],
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("with_D", [True, False])
@pytest.mark.parametrize("with_z", [True, False])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("softplus", [True, False])
@torch.inference_mode
def test_one_token_matches_reference(with_D, with_z, with_bias, softplus):
    case = _case(with_D=with_D, with_z=with_z, with_bias=with_bias, softplus=softplus)
    expected_state = case["state"].clone()
    expected = _run_reference({**case, "state": expected_state})
    actual_state = case["state"].clone()
    actual = _run_kernel({**case, "state": actual_state})

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-2, atol=1e-3)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_separate_destination_slots_and_padding():
    # Source slot 2 is the padding sentinel and slot 3 is an untouched source.
    case = _case(src_slots=(2, 3), dst_slots=(0, 1), pad_slot_id=2)
    original = case["state"].clone()
    expected_state = original.clone()
    expected = _run_reference({**case, "state": expected_state})
    actual_state = original.clone()
    actual = _run_kernel({**case, "state": actual_state})

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-2, atol=1e-3)
    # The padded source contributes a zero state and the untouched slot is intact.
    torch.testing.assert_close(actual_state[3], original[3])


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_disable_state_update_keeps_the_pool():
    case = _case(disable_state_update=True)
    original = case["state"].clone()
    expected = _run_reference({**case, "state": original.clone()})
    actual_state = original.clone()
    actual = _run_kernel({**case, "state": actual_state})

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, original)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_prewarm_then_run_uses_the_compiled_kernel():
    from mate.mamba import prewarm_selective_state_update

    prewarm_selective_state_update(
        state_dtype=torch.float16,
        io_dtype=torch.bfloat16,
        batch=_BATCH,
        heads=_HEADS,
        dim=_DIM,
        dstate=_DSTATE,
        groups=_GROUPS,
        slot_dtype=torch.int32,
    )
    case = _case()
    expected = _run_reference({**case, "state": case["state"].clone()})
    actual = _run_kernel(case)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_out_buffer_is_written_in_place():
    case = _case()
    out = torch.empty(_BATCH, _HEADS, _DIM, dtype=torch.bfloat16, device=_device())
    returned = _run_kernel(case, out=out)
    assert returned is out
    assert torch.isfinite(out).all()


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize(
    "overrides, message",
    [
        # One token per sequence is exactly what the native SSU implements, so it
        # is accepted (see test_one_token_per_sequence_cu_seqlens_is_accepted).
        ({"cu_seqlens": torch.tensor([0, 2], dtype=torch.int32)}, "variable-length"),
        ({"rand_seed": torch.zeros(1, dtype=torch.int32)}, "stochastic rounding"),
        ({"state_scale": torch.ones(_SLOTS, dtype=torch.float32)}, "quantized state"),
        ({"cache_steps": 4}, "intermediate"),
        (
            {"philox_rounds": 0, "cache_steps": 0, "intermediate_states_buffer": None},
            None,
        ),
    ],
)
def test_unsupported_paths_fail_loudly(overrides, message):
    case = _case()
    kwargs = {
        "D": case["D"],
        "z": case["z"],
        "dt_bias": case["dt_bias"],
        "pad_slot_id": case["pad_slot_id"],
        **overrides,
    }
    if message is None:
        # A no-op override set must still run, so the test proves the guard list
        # is not simply rejecting everything.
        mate.mamba.selective_state_update(
            case["state"],
            case["x"],
            case["dt"],
            case["A"],
            case["B"],
            case["C"],
            **kwargs,
        )
        return
    with pytest.raises(NotImplementedError, match=message):
        mate.mamba.selective_state_update(
            case["state"],
            case["x"],
            case["dt"],
            case["A"],
            case["B"],
            case["C"],
            **kwargs,
        )


def test_channel_wise_D_is_reported_as_a_gap():
    case = _case()
    channel_D = torch.rand(_HEADS, _DIM, dtype=torch.float32, device=_device())
    with pytest.raises(NotImplementedError, match="per-head"):
        _run_kernel({**case, "D": channel_D})
