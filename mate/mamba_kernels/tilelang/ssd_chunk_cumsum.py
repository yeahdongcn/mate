"""Native MUSA tilelang kernel for the SSD chunk-local cumsum stage.

This is stage 1 of the Mamba2/SSD chunked prefill behind
``mate.mamba.ssd_combined_fwd_varlen``. For every chunk it produces

* the **processed dt** -- ``softplus(dt + dt_bias)`` clamped to ``dt_limit`` --
  which the chunk-state stage consumes, and
* ``dA_cumsum`` -- the **inclusive** prefix sum of ``processed_dt * A`` over the
  chunk's tokens, which the state-passing and chunk-scan stages consume.

Layout contract (matches the production MUSA implementation, see the notes at the
bottom): ``dt`` is token-major ``[tokens, heads]`` while **both outputs are
head-major** ``[heads, nchunks, chunk_size]``. A chunk never spans two sequences
(the caller guarantees it), so the scan is per chunk and `cu_seqlens` is not
needed -- only `cu_chunk_seqlens`, the chunk start offsets in token units.

Chunks are padded rows: chunk ``c`` covers tokens
``[cu_chunk_seqlens[c], cu_chunk_seqlens[c+1])`` and lives at ``[0:heads, c, 0:]``
of the output. The **whole row is written**, and padding is part of the contract
rather than slack space: ``dt_out`` is exactly zero past the chunk's end, so the
scan saturates and ``dA_cumsum`` holds the chunk's total decay in the tail --
which is what the downstream stages read, unconditionally, from the padded row's
last position ``dA_cumsum[:, c, chunk_size - 1]``. Loads past the chunk's end are
masked by hand, since the kernel is compiled with bounds checking disabled.

The whole chunk is scanned with ``T.cumsum`` over a ``(heads, chunk_size)`` fp32
fragment, mirroring the warp-scan in the shipped native implementation but at
tile granularity.
"""

import tilelang
import tilelang.language as T
import torch

__all__ = ["chunk_cumsum_launch", "prewarm_ssd_chunk_cumsum"]

#: Softplus switches to the identity above this bound. Same constant as the SSU
#: kernel and its oracle, so the family stays self-consistent.
SOFTPLUS_THRESHOLD = 20.0

_THREADS = 128

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}

_COMPILE_FLAGS = [
    "-O3",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


@tilelang.jit(pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS)
def tilelang_ssd_chunk_cumsum(
    heads,
    chunk_size,
    dt_dtype,
    param_dtype,
    out_dtype,
    seqlen_dtype,
    accum_dtype="float32",
):
    """Build the chunk-local cumsum kernel for one (heads, chunk_size, dtype) set."""
    nchunks = T.dynamic("nchunks")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size
    dt_shape = (num_tokens, heads)
    out_shape = (heads, nchunks, block_S)

    @T.prim_func
    def tilelang_ssd_chunk_cumsum_kernel(
        dt: T.Tensor(dt_shape, dtype=dt_dtype),
        A: T.Tensor((heads,), dtype=param_dtype),
        dt_bias: T.Tensor((heads,), dtype=param_dtype),
        cu_chunk_seqlens: T.Tensor((nchunks + 1,), dtype=seqlen_dtype),
        dA_cumsum: T.Tensor(out_shape, dtype=out_dtype),
        dt_out: T.Tensor(out_shape, dtype=out_dtype),
        use_dt_bias: T.int32,
        dt_softplus: T.int32,
        dt_min: T.float32,
        dt_max: T.float32,
    ):
        with T.Kernel(nchunks, threads=_THREADS) as (bc,):
            chunk_start = T.alloc_var("int32")
            chunk_end = T.alloc_var("int32")
            chunk_start = cu_chunk_seqlens[bc]
            chunk_end = cu_chunk_seqlens[bc + 1]

            # Token-major staging of the chunk, then transposed into the
            # (heads, chunk_size) fragment the scan works on.
            dtT_fragment = T.alloc_fragment((block_S, heads), dtype=accum_dtype)
            dtT_shared = T.alloc_shared((block_S, heads + 1), dtype=accum_dtype)
            scaled = T.alloc_fragment((heads, block_S), dtype=accum_dtype)
            processed = T.alloc_fragment((heads, block_S), dtype=accum_dtype)

            for j, i in T.Parallel(block_S, heads):
                if chunk_start + j < chunk_end:
                    dtT_fragment[j, i] = T.cast(dt[chunk_start + j, i], accum_dtype)
                else:
                    dtT_fragment[j, i] = T.cast(0.0, accum_dtype)
            T.copy(dtT_fragment, dtT_shared[:, 0:heads])

            for i, j in T.Parallel(heads, block_S):
                value = T.alloc_var(accum_dtype)
                value = dtT_shared[j, i]
                if use_dt_bias != 0:
                    value = value + T.cast(dt_bias[i], accum_dtype)
                if dt_softplus != 0:
                    value = T.if_then_else(
                        value > SOFTPLUS_THRESHOLD,
                        value,
                        T.log(1.0 + T.exp(value)),
                    )
                value = T.max(value, dt_min)
                value = T.min(value, dt_max)
                if chunk_start + j < chunk_end:
                    processed[i, j] = value
                    scaled[i, j] = value * T.cast(A[i], accum_dtype)
                else:
                    # dt is zero beyond the chunk's end, so the scan saturates at
                    # the chunk total -- the value the downstream stages read
                    # from the padded row's last position.
                    processed[i, j] = T.cast(0.0, accum_dtype)
                    scaled[i, j] = T.cast(0.0, accum_dtype)

            T.cumsum(scaled, dim=1)

            # Every row is written in full: zero dt in the tail, saturated total
            # of dA_cumsum in the tail.
            for i, j in T.Parallel(heads, block_S):
                dA_cumsum[i, bc, j] = T.cast(scaled[i, j], out_dtype)
                dt_out[i, bc, j] = T.cast(processed[i, j], out_dtype)

    symbol = (
        f"tilelang_ssd_chunk_cumsum_h{heads}_cs{chunk_size}"
        f"_dt{str(dt_dtype).replace('torch.', '')}_o{str(out_dtype).replace('torch.', '')}"
    )
    return tilelang_ssd_chunk_cumsum_kernel.with_attr("global_symbol", symbol)


_DUMMY_CACHE: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor] = {}


def _dummy(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return an unread placeholder for an absent optional input.

    The kernel never dereferences an input whose runtime flag is zero, so one
    cached buffer per (shape, dtype, device) keeps the op at a single launch and
    allocates nothing per call.
    """
    key = (shape, dtype, str(device))
    tensor = _DUMMY_CACHE.get(key)
    if tensor is None:
        tensor = torch.empty(shape, device=device, dtype=dtype)
        _DUMMY_CACHE[key] = tensor
    return tensor


def chunk_cumsum_launch(
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor | None,
    cu_chunk_seqlens: torch.Tensor,
    chunk_size: int,
    *,
    dt_softplus: bool,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    dA_cumsum: torch.Tensor | None = None,
    dt_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the native chunk-local cumsum.

    Returns ``(dA_cumsum, dt_out)``, both head-major
    ``[heads, nchunks, chunk_size]`` fp32. Pass ``dA_cumsum``/``dt_out`` to
    reuse caller-owned scratch (the orchestration does, so a captured graph
    allocates nothing).

    ``dt`` may be fp32 or low precision; the scan itself is fp32.
    """
    if dt.dim() != 2:
        raise RuntimeError("dt must be [tokens, heads].")
    if not dt.is_contiguous():
        raise RuntimeError("dt must be contiguous.")
    heads = dt.shape[1]
    if A.numel() != heads or A.dim() != 1:
        raise RuntimeError("A must be [heads].")
    if A.dtype != torch.float32:
        raise RuntimeError("A must be fp32.")
    if dt_bias is not None and (dt_bias.numel() != heads or dt_bias.dtype != torch.float32):
        raise RuntimeError("dt_bias must be fp32 [heads].")
    if dt_bias is not None and not dt_bias.is_contiguous():
        raise RuntimeError("dt_bias must be contiguous.")
    if cu_chunk_seqlens.dtype != torch.int32:
        raise RuntimeError("cu_chunk_seqlens must be int32.")
    if cu_chunk_seqlens.dim() != 1 or cu_chunk_seqlens.numel() < 2:
        raise RuntimeError("cu_chunk_seqlens must be [nchunks + 1].")
    if not cu_chunk_seqlens.is_contiguous():
        raise RuntimeError("cu_chunk_seqlens must be contiguous.")
    nchunks = cu_chunk_seqlens.numel() - 1
    if chunk_size <= 0:
        raise RuntimeError("chunk_size must be positive.")

    out_shape = (heads, nchunks, chunk_size)
    if dA_cumsum is None:
        dA_cumsum = torch.empty(out_shape, device=dt.device, dtype=torch.float32)
    if dt_out is None:
        dt_out = torch.empty(out_shape, device=dt.device, dtype=torch.float32)
    if dA_cumsum.shape != out_shape or dt_out.shape != out_shape:
        raise RuntimeError(
            f"outputs must be {out_shape}, got {tuple(dA_cumsum.shape)} / "
            f"{tuple(dt_out.shape)}."
        )
    if dA_cumsum.dtype != torch.float32 or dt_out.dtype != torch.float32:
        raise RuntimeError("dA_cumsum and dt_out must be fp32.")
    if not dA_cumsum.is_contiguous() or not dt_out.is_contiguous():
        raise RuntimeError("dA_cumsum and dt_out must be contiguous.")

    kernel = tilelang_ssd_chunk_cumsum(
        heads,
        chunk_size,
        dt.dtype,
        torch.float32,
        torch.float32,
        cu_chunk_seqlens.dtype,
    )
    dt_min, dt_max = dt_limit
    kernel(
        dt,
        A,
        dt_bias if dt_bias is not None else _dummy((heads,), torch.float32, dt.device),
        cu_chunk_seqlens,
        dA_cumsum,
        dt_out,
        1 if dt_bias is not None else 0,
        1 if dt_softplus else 0,
        float(dt_min),
        float(dt_max),
    )
    return dA_cumsum, dt_out


def prewarm_ssd_chunk_cumsum(
    heads: int,
    chunk_size: int,
    dt_dtype: torch.dtype = torch.float32,
) -> None:
    """Compile the kernel for one shape outside any captured graph.

    Call during warmup for every ``(heads, chunk_size, dt dtype)`` the serving
    configuration will use: a first-call compile inside a captured graph stalls
    the worker.
    """
    tilelang_ssd_chunk_cumsum(
        heads, chunk_size, dt_dtype, torch.float32, torch.float32, torch.int32
    )
