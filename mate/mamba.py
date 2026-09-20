"""Public MUSA-native mamba/SSU API.

``selective_state_update`` is the decode-side entry point consumed by the
FlashInfer-shaped compatibility wrapper (``flashinfer.mamba``) in a MUSA build
whose provider is MATE. It implements the one-token selective state update
natively in tilelang and mirrors the reference contract exactly for the
supported configuration.

Supported today (single token per sequence):

- ``state`` fp16/bf16/fp32 ``[slots, heads, dim, dstate]``, contiguous
- ``x``/``B``/``C``/``z`` fp16 or bf16; ``dt``/``A``/``D``/``dt_bias`` fp32
- per-head (tied) ``A``, ``D`` and ``dt_bias``, i.e. ``A[h]`` broadcasting over
  ``dim``/``dstate``, which is the layout the Mamba2 consumers pass
- ``pad_slot_id`` and ``disable_state_update`` honoured in kernel

Everything else raises ``NotImplementedError`` naming the missing capability
rather than silently narrowing semantics. Not implemented yet: packed
variable-length and MTP decoding (``cu_seqlens``/``num_accepted_tokens``),
stochastic rounding (``rand_seed``), quantized state (``state_scale``),
intermediate/replay state capture (``cache_steps`` and
``intermediate_*``), and channel-wise ``D``/``dt_bias``.

Graph capture: the tilelang kernel is compiled on first use for each
(dtype, shape, flag) tuple. Call :func:`prewarm_selective_state_update` during
warmup, outside any capture context — a first-call compile inside a captured
graph stalls the worker (see `.claude/rules/tilelang-musa.md`). The op itself
issues a single kernel launch, allocates nothing when ``out`` is provided, and
performs no host/device scalar copies.
"""

from __future__ import annotations

import torch

from mate.api_logging import mate_api
from mate.mamba_kernels.tilelang.ssu_one_token import ssu_one_token_launch

__all__ = ["prewarm_selective_state_update", "selective_state_update"]

_SUPPORTED_STATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_SLOT_DTYPES = (torch.int32, torch.int64)
_SUPPORTED_ALGORITHMS = ("auto", "native", "simple")
_SUPPORTED_BACKENDS = ("auto", "musa", "native", "flashinfer")
_DEFAULT_LANES_PER_ROW = 32
_DEFAULT_ROWS_PER_CTA = 4


def _per_head(tensor: torch.Tensor, heads: int, *, name: str) -> torch.Tensor:
    """Return ``tensor`` as a contiguous per-head vector, or explain the gap."""
    if tensor.dtype != torch.float32:
        raise NotImplementedError(
            f"native mamba SSU requires fp32 {name}, got {tensor.dtype}."
        )
    if tensor.dim() == 1:
        if tensor.shape[0] != heads:
            raise ValueError(
                f"{name} must have one entry per head, got {tensor.shape}."
            )
        return tensor.contiguous()
    if tensor.dim() == 3 and tensor.stride(1) == 0 and tensor.stride(2) == 0:
        # The consumers pass A as a tied strided view [heads, dim, dstate].
        if tensor.shape[0] != heads:
            raise ValueError(
                f"{name} must have one entry per head, got {tensor.shape}."
            )
        return tensor[:, 0, 0].contiguous()
    raise NotImplementedError(
        f"native mamba SSU implements per-head (tied) {name} only; got shape "
        f"{tuple(tensor.shape)} with strides {tuple(tensor.stride())}."
    )


def _slots(
    indices: torch.Tensor | None,
    batch: int,
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if indices is None:
        # One tensor op; the consumers pass explicit indices on the capture path.
        return torch.arange(batch, dtype=torch.int32, device=device)
    if indices.dim() != 1 or indices.shape[0] != batch:
        raise NotImplementedError(
            f"native mamba SSU implements the one-token path only, so {name} must "
            f"be a 1D tensor with one entry per batch row; got {tuple(indices.shape)}."
        )
    if indices.dtype not in _SUPPORTED_SLOT_DTYPES:
        raise ValueError(f"{name} must be int32 or int64, got {indices.dtype}.")
    return indices.contiguous()


@mate_api
def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    pad_slot_id: int = -1,
    state_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
    intermediate_state_scales: torch.Tensor | None = None,
    rand_seed: torch.Tensor | None = None,
    philox_rounds: int = 10,
    cache_steps: int = 0,
    algorithm: str = "auto",
    dst_state_batch_indices: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Apply one decode-step selective state update and return the output ``y``.

    The argument order, defaults and semantics follow the FlashInfer-shaped
    contract the MUSA consumers already call, so this function can serve as the
    implementation behind that surface without an adapter layer.
    """
    if algorithm not in _SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"native mamba SSU implements {_SUPPORTED_ALGORITHMS}, got algorithm={algorithm!r}."
        )
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"native mamba SSU accepts backends {_SUPPORTED_BACKENDS}, got backend={backend!r}."
        )
    if cu_seqlens is not None or num_accepted_tokens is not None or x.dim() != 3:
        raise NotImplementedError(
            "native mamba SSU implements single-token decoding only; packed "
            "variable-length/MTP decoding (cu_seqlens/num_accepted_tokens, or a "
            "4D x) is not implemented yet."
        )
    if rand_seed is not None:
        raise NotImplementedError(
            "native mamba SSU does not implement stochastic rounding (rand_seed) yet."
        )
    if state_scale is not None:
        raise NotImplementedError(
            "native mamba SSU does not implement quantized state (state_scale) yet."
        )
    if (
        intermediate_states_buffer is not None
        or intermediate_state_indices is not None
        or intermediate_state_scales is not None
        or cache_steps
    ):
        raise NotImplementedError(
            "native mamba SSU does not implement intermediate/replay state capture yet."
        )
    if state.dim() != 4:
        raise ValueError(
            f"state must be [slots, heads, dim, dstate], got {tuple(state.shape)}."
        )
    slots, heads, dim, dstate = state.shape
    if state.dtype not in _SUPPORTED_STATE_DTYPES:
        raise NotImplementedError(
            f"native mamba SSU supports state dtypes {_SUPPORTED_STATE_DTYPES}, "
            f"got {state.dtype}."
        )
    if x.dtype not in _SUPPORTED_IO_DTYPES:
        raise NotImplementedError(
            f"native mamba SSU supports x dtypes {_SUPPORTED_IO_DTYPES}, got {x.dtype}."
        )
    if x.shape != dt.shape or x.shape != (x.shape[0], heads, dim):
        raise ValueError(
            f"x/dt must match the state head layout: x={tuple(x.shape)}, "
            f"dt={tuple(dt.shape)}, state={tuple(state.shape)}."
        )
    if B.dim() != 3 or C.shape != B.shape:
        raise ValueError(
            f"single-token B/C must be [batch, groups, dstate]; got B="
            f"{tuple(B.shape)}, C={tuple(C.shape)}."
        )
    if B.dtype != x.dtype or C.dtype != x.dtype:
        raise ValueError("B and C must share the x dtype for the native SSU kernel.")
    batch = x.shape[0]

    a_head = _per_head(A, heads, name="A")
    d_head = _per_head(D, heads, name="D") if D is not None else None
    bias_head = (
        _per_head(dt_bias, heads, name="dt_bias") if dt_bias is not None else None
    )
    if z is not None and (z.shape != x.shape or z.dtype != x.dtype):
        raise ValueError("z must match x in shape and dtype for the native SSU kernel.")
    if out is None:
        out = torch.empty((batch, heads, dim), dtype=x.dtype, device=x.device)
    elif out.shape != (batch, heads, dim) or out.dtype != x.dtype:
        raise ValueError(
            f"out must be {x.dtype} with shape {tuple(x.shape)}, got "
            f"{out.dtype} {tuple(out.shape)}."
        )

    src_slots = _slots(
        state_batch_indices, batch, name="state_batch_indices", device=x.device
    )
    dst_slots = (
        None
        if dst_state_batch_indices is None
        else _slots(
            dst_state_batch_indices,
            batch,
            name="dst_state_batch_indices",
            device=x.device,
        )
    )

    return ssu_one_token_launch(
        state,
        x,
        dt,
        a_head,
        B,
        C,
        d_head,
        bias_head,
        z,
        src_slots,
        dst_slots,
        out,
        dt_softplus=bool(dt_softplus),
        pad_slot_id=int(pad_slot_id),
        disable_state_update=bool(disable_state_update),
    )


def prewarm_selective_state_update(
    *,
    state_dtype: torch.dtype,
    io_dtype: torch.dtype,
    batch: int,
    heads: int,
    dim: int,
    dstate: int,
    groups: int,
    slot_dtype: torch.dtype,
    rows_per_cta: int = _DEFAULT_ROWS_PER_CTA,
    lanes_per_row: int = _DEFAULT_LANES_PER_ROW,
) -> None:
    """Compile the native kernel for one configuration, outside capture.

    Call this during warmup (before any graph capture) for each shape/dtype
    combination the serving configuration will use. Compiling inside a captured
    graph stalls the worker, so prewarming is required rather than optional.
    """
    from mate.mamba_kernels.tilelang.ssu_one_token import prewarm_ssu_one_token

    prewarm_ssu_one_token(
        state_dtype=state_dtype,
        io_dtype=io_dtype,
        dim=dim,
        dstate=dstate,
        head_ratio=heads // groups,
        slot_dtype=slot_dtype,
        rows_per_cta=rows_per_cta,
        lanes_per_row=lanes_per_row,
    )
