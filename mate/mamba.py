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

__all__ = [
    "prewarm_selective_state_update",
    "prewarm_ssd_combined_fwd_varlen",
    "reset_ssd_workspaces",
    "selective_state_update",
    "ssd_combined_fwd_varlen",
]

def _ssu_launch():
    """Import the SSU launcher on first use.

    ``mate.mamba`` is the symbol surface the FlashInfer-shaped compatibility
    wrapper probes, so importing it must not require tilelang: on a CPU-only node
    (or in a test environment) the probe still has to answer.
    """
    from mate.mamba_kernels.tilelang.ssu_one_token import ssu_one_token_launch

    return ssu_one_token_launch


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

    return _ssu_launch()(
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


@mate_api
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


# --- SSD packed prefill (the `--mamba-backend flashinfer` prefill entry) ------

_SSD_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)
_SSD_SUPPORTED_DT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_SSD_SUPPORTED_STATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class _SsdWorkspace:
    """Per-shape scratch for the SSD stages.

    Allocating the intermediates per call is what makes a wrapper expensive: five
    kernels' worth of scratch inside a captured graph is allocation churn on every
    replay. The workspace is keyed by everything the buffers depend on, so a
    serving run allocates once per shape and replays allocation-free after that.
    """

    __slots__ = ("dt_out", "dA_cumsum", "states", "CB")

    def __init__(self, heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, device):
        self.dt_out = torch.empty(
            (heads, nchunks, chunk_size), device=device, dtype=torch.float32
        )
        self.dA_cumsum = torch.empty_like(self.dt_out)
        self.states = torch.empty(
            (nchunks, heads, dim, dstate), device=device, dtype=state_dtype
        )
        self.CB = torch.empty(
            (nchunks, groups, chunk_size, chunk_size), device=device, dtype=torch.float32
        )


_WORKSPACES: dict[tuple, _SsdWorkspace] = {}


def _workspace(
    heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, device
) -> _SsdWorkspace:
    key = (heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, str(device))
    space = _WORKSPACES.get(key)
    if space is None:
        space = _SsdWorkspace(
            heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, device
        )
        _WORKSPACES[key] = space
    return space


def reset_ssd_workspaces() -> None:
    """Drop cached scratch, e.g. before changing device or freeing memory."""
    _WORKSPACES.clear()


def _stage(name: str):
    """Resolve one stage launcher.

    Imported lazily so that a partial checkout fails with the name of the missing
    stage rather than an ImportError at module import, and so the stage table
    stays patchable for tests.
    """
    from mate.mamba_kernels.tilelang import (
        ssd_bmm,
        ssd_chunk_cumsum,
        ssd_chunk_scan,
        ssd_chunk_state,
        ssd_state_passing,
    )

    launchers = {
        "cumsum": ssd_chunk_cumsum.chunk_cumsum_launch,
        "chunk_state": ssd_chunk_state.ssd_chunk_state_launch,
        "state_passing": ssd_state_passing.ssd_state_passing_launch,
        "bmm": ssd_bmm.ssd_bmm_launch,
        "chunk_scan": ssd_chunk_scan.chunk_scan_launch,
    }
    return launchers[name]


@mate_api
def ssd_combined_fwd_varlen(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    seq_idx: torch.Tensor,
    out: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    initial_states: torch.Tensor | None = None,
    return_intermediate_states: bool = False,
    state_dtype: torch.dtype | None = None,
    checkpoint_token_indices: torch.Tensor | None = None,
    checkpoint_state_slots: torch.Tensor | None = None,
    checkpoint_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed variable-length SSD prefill, native on MUSA.

    The five stages run as five tilelang launches with per-shape scratch taken
    from a cached workspace, so a steady-state call allocates nothing but the
    returned state tensor.

    Contract, matching vLLM's Mamba2 prefill caller:

    - ``x`` is ``[tokens, heads, dim]`` and ``dt`` is ``[tokens, heads]``, both
      packed over the flattened (variable-length) batch;
    - ``cu_chunk_seqlens`` is the only authority for the chunk-to-token mapping:
      chunk ``c`` covers ``[cu_chunk_seqlens[c], cu_chunk_seqlens[c+1])``. A chunk
      never spans two sequences, and every chunk but the last of a sequence spans
      ``chunk_size`` tokens;
    - ``last_chunk_indices[b]`` is the ordinal of sequence ``b``'s last logical
      chunk and ``seq_idx[c]`` is the request owning chunk ``c``. A chunk opens a
      sequence when ``seq_idx`` differs from the previous chunk's, and that
      transition -- not the ``last_chunk_indices`` arithmetic -- decides where the
      entering state comes from. The two can be made to disagree only by an
      inconsistent caller, and the two independent implementations of this
      contract resolve that disagreement differently, so ``seq_idx`` is the
      authority;
    - ``out`` is written in place when supplied;
    - the return value is a single tensor: the full ``[nchunks, heads, dim,
      dstate]`` state buffer when ``return_intermediate_states``, otherwise
      ``states[last_chunk_indices]`` with shape ``[batch, heads, dim, dstate]``,
      both in ``state_dtype`` (which defaults to ``initial_states.dtype``, else
      ``C.dtype``).

    ``checkpoint_*`` is not supported and raises rather than silently ignoring a
    capability the caller asked for.
    """
    if any(
        value is not None
        for value in (checkpoint_token_indices, checkpoint_state_slots, checkpoint_states)
    ):
        raise NotImplementedError(
            "ssd_combined_fwd_varlen does not support the checkpoint_* arguments; "
            "the caller must dispatch to its reference path for checkpointed runs."
        )
    if x.dim() != 3 or dt.dim() != 2 or B.dim() != 3 or C.dim() != 3:
        raise ValueError("packed SSD expects x [tokens, heads, dim] and dt [tokens, heads].")
    tokens, heads, dim = x.shape
    groups, dstate = B.shape[1], B.shape[2]
    if dt.shape != (tokens, heads):
        raise ValueError("dt must be [tokens, heads].")
    if x.dtype not in _SSD_SUPPORTED_IO_DTYPES or B.dtype != x.dtype or C.dtype != x.dtype:
        raise ValueError("x, B and C must share one of fp16/bf16.")
    if dt.dtype not in _SSD_SUPPORTED_DT_DTYPES:
        raise ValueError("dt must be fp32, fp16 or bf16.")
    if heads % groups:
        raise ValueError("heads must be divisible by groups.")
    if A.shape != (heads,):
        raise ValueError("A must be [heads].")
    if A.dtype != torch.float32:
        raise ValueError("A must be fp32.")
    if dt_bias is not None and dt_bias.shape != (heads,):
        raise ValueError("dt_bias must be [heads].")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    nchunks = cu_chunk_seqlens.numel() - 1
    if nchunks < 1:
        raise ValueError("cu_chunk_seqlens must have at least two entries.")
    if seq_idx.numel() != nchunks:
        raise ValueError("seq_idx must hold one request id per logical chunk.")
    if cu_seqlens.numel() != last_chunk_indices.numel() + 1:
        raise ValueError(
            "cu_seqlens and last_chunk_indices must describe the same batch."
        )
    batch = last_chunk_indices.numel()
    if initial_states is not None:
        if tuple(initial_states.shape[1:]) != (heads, dim, dstate):
            raise ValueError("initial_states must be [batch, heads, dim, dstate].")
        if int(initial_states.shape[0]) != batch:
            raise ValueError(
                "initial_states must hold one row per sequence in cu_seqlens."
            )
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError("out must match x's shape and dtype.")
    if D is not None and D.shape not in ((heads, dim), (heads,)):
        raise ValueError("D must be [heads, dim] or [heads].")
    if z is not None and z.shape != x.shape:
        raise ValueError("z must match x's shape.")
    if state_dtype is None:
        state_dtype = initial_states.dtype if initial_states is not None else C.dtype
    if state_dtype not in _SSD_SUPPORTED_STATE_DTYPES:
        raise ValueError("state_dtype must be fp16, bf16 or fp32.")
    for tensor in (x, dt, A, B, C, cu_seqlens, cu_chunk_seqlens, last_chunk_indices, seq_idx):
        if not tensor.is_contiguous():
            raise ValueError(
                "x, dt, A, B, C and the chunk metadata must be contiguous."
            )

    space = _workspace(
        heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, x.device
    )

    _stage("cumsum")(
        dt,
        A,
        dt_bias,
        cu_chunk_seqlens,
        chunk_size,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        dA_cumsum=space.dA_cumsum,
        dt_out=space.dt_out,
    )
    _stage("chunk_state")(
        x,
        B,
        space.dt_out,
        space.dA_cumsum,
        cu_chunk_seqlens,
        chunk_size,
        states=space.states,
    )
    _stage("state_passing")(
        space.states,
        space.dA_cumsum,
        initial_states,
        last_chunk_indices,
        seq_idx,
        state_dtype=state_dtype,
    )
    _stage("bmm")(
        C,
        B,
        cu_chunk_seqlens,
        chunk_size,
        cb=space.CB,
    )
    _stage("chunk_scan")(
        x,
        C,
        space.CB,
        space.dt_out,
        space.dA_cumsum,
        space.states,
        initial_states,
        seq_idx,
        cu_chunk_seqlens,
        out,
        D_param=D,
        z=z,
    )
    if return_intermediate_states:
        return space.states
    return space.states.index_select(0, last_chunk_indices)


@mate_api
def prewarm_ssd_combined_fwd_varlen(
    heads: int,
    groups: int,
    dim: int,
    dstate: int,
    chunk_size: int,
    io_dtype: torch.dtype = torch.bfloat16,
    state_dtype: torch.dtype = torch.bfloat16,
    dt_dtype: torch.dtype = torch.float32,
    block_M: int = 32,
) -> None:
    """Compile every SSD stage for one shape, outside any captured graph.

    Call during warmup for each configuration the server will serve: a first-call
    compile inside a captured graph stalls the worker. The compile-time degrees of
    freedom are the four shape parameters, the three dtypes and the scan's
    ``block_M``; the token and chunk counts are runtime dimensions.
    """
    from mate.mamba_kernels.tilelang.ssd_bmm import prewarm_ssd_bmm
    from mate.mamba_kernels.tilelang.ssd_chunk_cumsum import prewarm_ssd_chunk_cumsum
    from mate.mamba_kernels.tilelang.ssd_chunk_scan import prewarm_ssd_chunk_scan
    from mate.mamba_kernels.tilelang.ssd_chunk_state import prewarm_ssd_chunk_state
    from mate.mamba_kernels.tilelang.ssd_state_passing import prewarm_ssd_state_passing

    prewarm_ssd_chunk_cumsum(heads, chunk_size, dt_dtype)
    prewarm_ssd_chunk_state(heads, groups, dim, dstate, chunk_size, io_dtype)
    prewarm_ssd_state_passing(heads, dim, dstate, io_dtype, state_dtype)
    prewarm_ssd_bmm(groups, dstate, chunk_size, io_dtype)
    prewarm_ssd_chunk_scan(
        heads, groups, dim, dstate, chunk_size, io_dtype, state_dtype, block_M
    )
