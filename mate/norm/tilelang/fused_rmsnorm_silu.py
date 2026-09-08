# mypy: ignore-errors

import functools

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

from mate.api_logging import mate_api

from ._norm_common import (
    _JIT_CONFIG,
    _ceil_pow2_expr,
    _make_row_sum_reduce_macro,
    _make_sum_squares_macro,
)


__all__ = ["fused_rmsnorm_silu"]


_MXFP8_BLOCK_SIZE = 32
_FP8_MAX = 448.0
_LOG2E = 1.4426950408889634

_SILU_JIT_CONFIG = {
    **_JIT_CONFIG,
    "pass_configs": {
        **_JIT_CONFIG["pass_configs"],
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
}


@functools.cache
def _select_norm_schedule(hidden_size: int, num_rows: int) -> tuple[int, int, int]:
    values_per_thread = 8
    if hidden_size >= 8192:
        threads_per_row = 256
        rows_per_block = 2
    elif hidden_size >= 4096:
        threads_per_row = 128
        rows_per_block = 4
    elif hidden_size >= 1024:
        threads_per_row = 64
        rows_per_block = 8
    else:
        threads_per_row = 32
        rows_per_block = 16
    while rows_per_block > 1 and num_rows % rows_per_block != 0:
        rows_per_block //= 2
    return rows_per_block, threads_per_row, values_per_thread


@functools.cache
def _select_mxfp8_schedule(hidden_size: int) -> tuple[int, int, int]:
    rows_per_block = 1
    threads_per_row = 128
    values_per_thread = 8
    return rows_per_block, threads_per_row, values_per_thread


@tilelang.jit(**_SILU_JIT_CONFIG)
def tilelang_rmsnorm_silu_dense(
    is_fp8: bool,
    rows_per_block: int,
    threads_per_row: int,
    values_per_thread: int,
):
    num_tokens = T.dynamic("num_tokens")
    hidden_size = T.dynamic("hidden_size")
    x_stride_t = T.dynamic("x_stride_t")
    out_stride_t = T.dynamic("out_stride_t")

    num_warps_per_row = threads_per_row // 32
    row_reduce_shuffles = (
        num_warps_per_row.bit_length() - 1 if num_warps_per_row > 1 else 0
    )
    threads = rows_per_block * threads_per_row
    tile_width = threads_per_row * values_per_thread
    full_groups = hidden_size // tile_width
    tail_elems = hidden_size % tile_width
    output_dtype = T.float8_e4m3fn if is_fp8 else T.bfloat16
    sum_squares = _make_sum_squares_macro(
        full_groups,
        tile_width,
        values_per_thread,
        tail_elems,
        hidden_size,
        is_3d=False,
    )
    row_sum_reduce = _make_row_sum_reduce_macro(num_warps_per_row, row_reduce_shuffles)

    x_type = T.StridedTensor((num_tokens, hidden_size), (x_stride_t, 1), T.bfloat16)
    weight_type = T.StridedTensor((hidden_size,), (1,), T.bfloat16)
    out_type = T.StridedTensor(
        (num_tokens, hidden_size), (out_stride_t, 1), output_dtype
    )

    @T.prim_func
    def tilelang_rmsnorm_silu_dense_kernel(
        x: x_type,
        weight: weight_type,
        out: out_type,
        eps: T.float32,
    ):
        num_blocks = num_tokens // rows_per_block

        with T.Kernel(num_blocks, threads=threads) as (block_idx,):
            T.assume(x_stride_t % values_per_thread == 0)
            T.assume(out_stride_t % values_per_thread == 0)

            tid = T.get_thread_binding()
            lane = tid % threads_per_row
            row_in_block = tid // threads_per_row
            warp_lane = lane % 32
            warp_in_row = lane // 32
            row = block_idx * rows_per_block + row_in_block

            sum_sq = T.alloc_local((1,), T.float32)
            x_local = T.alloc_local((values_per_thread,), T.bfloat16)
            weight_shared = T.alloc_shared((hidden_size,), T.bfloat16)
            warp_sum_sq_shared = T.alloc_shared(
                (rows_per_block, num_warps_per_row), T.float32
            )

            sum_sq[0] = 0.0
            T.copy(weight, weight_shared)
            T.sync_threads()

            sum_squares(x, x_local, sum_sq, row, 0, 0, lane)
            row_sum_reduce(
                sum_sq,
                warp_sum_sq_shared,
                row_in_block,
                warp_lane,
                warp_in_row,
            )
            T.sync_threads()

            inv_rms = T.rsqrt(warp_sum_sq_shared[row_in_block, 0] / hidden_size + eps)

            for group in T.serial(full_groups):
                base = group * tile_width + lane * values_per_thread
                for v in T.vectorized(values_per_thread):
                    x_local[v] = x[row, base + v]
                for v in T.vectorized(values_per_thread):
                    norm = (
                        T.cast(x_local[v], T.float32)
                        * inv_rms
                        * T.cast(weight_shared[base + v], T.float32)
                    )
                    silu = norm / (T.exp2(-norm * _LOG2E) + 1.0)
                    if is_fp8:
                        out[row, base + v] = T.cast(
                            T.clamp(silu, -_FP8_MAX, _FP8_MAX), output_dtype
                        )
                    else:
                        out[row, base + v] = T.cast(silu, output_dtype)

            if tail_elems != 0:
                base = full_groups * tile_width + lane * values_per_thread
                for v in T.unroll(values_per_thread):
                    col = base + v
                    if col < hidden_size:
                        x_local[v] = x[row, col]
                        norm = (
                            T.cast(x_local[v], T.float32)
                            * inv_rms
                            * T.cast(weight_shared[col], T.float32)
                        )
                        silu = norm / (T.exp2(-norm * _LOG2E) + 1.0)
                        if is_fp8:
                            out[row, col] = T.cast(
                                T.clamp(silu, -_FP8_MAX, _FP8_MAX), output_dtype
                            )
                        else:
                            out[row, col] = T.cast(silu, output_dtype)

    mode = "fp8" if is_fp8 else "bf16"
    symbol = f"tilelang_rmsnorm_silu_dense_{mode}_h{hidden_size}_r{rows_per_block}_l{threads_per_row}_v{values_per_thread}"
    return tilelang_rmsnorm_silu_dense_kernel.with_attr("global_symbol", symbol)


@tilelang.jit(**_SILU_JIT_CONFIG)
def tilelang_rmsnorm_silu_mxfp8(
    rows_per_block: int,
    threads_per_row: int,
    values_per_thread: int,
):
    if values_per_thread != 8:
        raise RuntimeError("MXFP8 kernel requires values_per_thread=8.")

    num_tokens = T.dynamic("num_tokens")
    hidden_size = T.dynamic("hidden_size")
    x_stride_t = T.dynamic("x_stride_t")
    out_stride_t = T.dynamic("out_stride_t")
    scale_stride_t = T.dynamic("scale_stride_t")

    num_warps_per_row = threads_per_row // 32
    row_reduce_shuffles = (
        num_warps_per_row.bit_length() - 1 if num_warps_per_row > 1 else 0
    )
    threads = rows_per_block * threads_per_row
    tile_width = threads_per_row * values_per_thread
    full_groups = hidden_size // tile_width
    tail_elems = hidden_size % tile_width
    scale_blocks_per_tile = tile_width // _MXFP8_BLOCK_SIZE
    sum_squares = _make_sum_squares_macro(
        full_groups,
        tile_width,
        values_per_thread,
        tail_elems,
        hidden_size,
        is_3d=False,
    )
    row_sum_reduce = _make_row_sum_reduce_macro(num_warps_per_row, row_reduce_shuffles)

    x_type = T.StridedTensor((num_tokens, hidden_size), (x_stride_t, 1), T.bfloat16)
    weight_type = T.StridedTensor((hidden_size,), (1,), T.bfloat16)
    out_type = T.StridedTensor(
        (num_tokens, hidden_size), (out_stride_t, 1), T.float8_e4m3fn
    )
    scale_type = T.StridedTensor(
        (num_tokens, hidden_size // _MXFP8_BLOCK_SIZE),
        (scale_stride_t, 1),
        T.float8_e8m0fnu,
    )

    @T.prim_func
    def tilelang_rmsnorm_silu_mxfp8_kernel(
        x: x_type,
        weight: weight_type,
        out: out_type,
        block_scale: scale_type,
        eps: T.float32,
    ):
        num_blocks = num_tokens // rows_per_block

        with T.Kernel(num_blocks, threads=threads) as (block_idx,):
            T.assume(x_stride_t % values_per_thread == 0)
            T.assume(out_stride_t % values_per_thread == 0)
            tid = T.get_thread_binding()
            lane = tid % threads_per_row
            row_in_block = tid // threads_per_row
            warp_lane = lane % 32
            warp_in_row = lane // 32
            row = block_idx * rows_per_block + row_in_block

            sum_sq = T.alloc_local((1,), T.float32)
            x_local = T.alloc_local((values_per_thread,), T.bfloat16)
            local_max = T.alloc_local((1,), T.float32)
            silu_local = T.alloc_local((values_per_thread,), T.float32)
            warp_sum_sq_shared = T.alloc_shared(
                (rows_per_block, num_warps_per_row), T.float32
            )

            sum_sq[0] = 0.0
            sum_squares(x, x_local, sum_sq, row, 0, 0, lane)
            row_sum_reduce(
                sum_sq,
                warp_sum_sq_shared,
                row_in_block,
                warp_lane,
                warp_in_row,
            )
            T.sync_threads()
            inv_rms = T.rsqrt(warp_sum_sq_shared[row_in_block, 0] / hidden_size + eps)

            for group in T.serial(full_groups):
                base = group * tile_width + lane * values_per_thread
                local_max[0] = 0.0
                for v in T.vectorized(values_per_thread):
                    x_local[v] = x[row, base + v]
                for v in T.vectorized(values_per_thread):
                    norm = (
                        T.cast(x_local[v], T.float32)
                        * inv_rms
                        * T.cast(weight[base + v], T.float32)
                    )
                    silu_local[v] = norm / (T.exp2(-norm * _LOG2E) + 1.0)
                for v in T.unroll(values_per_thread):
                    local_max[0] = T.max(local_max[0], T.abs(silu_local[v]))

                local_max[0] = T.max(local_max[0], T.shfl_xor(local_max[0], 2))
                local_max[0] = T.max(local_max[0], T.shfl_xor(local_max[0], 1))

                scale_f32 = T.alloc_local((1,), T.float32)
                scale_f32[0] = 0.0
                if lane % 4 == 0:
                    scale_f32[0] = T.if_then_else(
                        local_max[0] > 0.0,
                        _ceil_pow2_expr(local_max[0] / _FP8_MAX),
                        0.0,
                    )
                    block_col = group * scale_blocks_per_tile + lane // 4
                    block_scale[row, block_col] = T.cast(scale_f32[0], T.float8_e8m0fnu)
                scale_f32[0] = T.shfl_sync(scale_f32[0], (warp_lane // 4) * 4, 32)
                inv_scale = T.if_then_else(scale_f32[0] != 0.0, 1.0 / scale_f32[0], 0.0)

                for v in T.vectorized(values_per_thread):
                    out[row, base + v] = T.cast(
                        silu_local[v] * inv_scale, T.float8_e4m3fn
                    )

            if tail_elems != 0:
                base = full_groups * tile_width + lane * values_per_thread
                if base < hidden_size:
                    local_max[0] = 0.0
                    for v in T.unroll(values_per_thread):
                        col = base + v
                        if col < hidden_size:
                            x_local[v] = x[row, col]
                            norm = (
                                T.cast(x_local[v], T.float32)
                                * inv_rms
                                * T.cast(weight[col], T.float32)
                            )
                            silu_local[v] = norm / (T.exp2(-norm * _LOG2E) + 1.0)
                            local_max[0] = T.max(local_max[0], T.abs(silu_local[v]))
                    local_max[0] = T.max(local_max[0], T.shfl_xor(local_max[0], 2))
                    local_max[0] = T.max(local_max[0], T.shfl_xor(local_max[0], 1))
                    scale_f32 = T.if_then_else(
                        local_max[0] > 0.0,
                        _ceil_pow2_expr(local_max[0] / _FP8_MAX),
                        0.0,
                    )
                    scale = T.cast(scale_f32, T.float8_e8m0fnu)
                    inv_scale = T.if_then_else(scale_f32 != 0.0, 1.0 / scale_f32, 0.0)
                    for v in T.unroll(values_per_thread):
                        col = base + v
                        if col < hidden_size:
                            out[row, col] = T.cast(
                                silu_local[v] * inv_scale, T.float8_e4m3fn
                            )
                    if lane % 4 == 0:
                        block_scale[
                            row, full_groups * scale_blocks_per_tile + lane // 4
                        ] = scale

    symbol = (
        f"tilelang_rmsnorm_silu_mxfp8_h{hidden_size}_r{rows_per_block}_l{threads_per_row}_"
        f"v{values_per_thread}"
    )
    return tilelang_rmsnorm_silu_mxfp8_kernel.with_attr("global_symbol", symbol)


def _output_mode(out: torch.Tensor, block_scale: torch.Tensor | None) -> str:
    if out.dtype == torch.bfloat16:
        return "bf16"
    if out.dtype == torch.float8_e4m3fn:
        if block_scale is not None:
            return "mxfp8"
        return "fp8"
    raise RuntimeError(
        "fused_rmsnorm_silu output dtype must be bfloat16 or float8_e4m3fn."
    )


def _fused_rmsnorm_silu_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    block_scale: torch.Tensor | None,
    output_mode: str,
    rows_per_block_override: int | None = None,
    threads_per_row_override: int | None = None,
):
    num_tokens, hidden_size = input.shape
    if output_mode == "mxfp8":
        rows_per_block, threads_per_row, values_per_thread = _select_mxfp8_schedule(
            hidden_size
        )
        rows_per_block = (
            rows_per_block_override
            if rows_per_block_override is not None
            else rows_per_block
        )
        threads_per_row = (
            threads_per_row_override
            if threads_per_row_override is not None
            else threads_per_row
        )
    else:
        default_rows, default_threads, values_per_thread = _select_norm_schedule(
            hidden_size, num_tokens
        )
        rows_per_block = (
            rows_per_block_override
            if rows_per_block_override is not None
            else default_rows
        )
        threads_per_row = (
            threads_per_row_override
            if threads_per_row_override is not None
            else default_threads
        )

    if output_mode in ("bf16", "fp8"):
        kernel = tilelang_rmsnorm_silu_dense(
            is_fp8=output_mode == "fp8",
            rows_per_block=rows_per_block,
            threads_per_row=threads_per_row,
            values_per_thread=values_per_thread,
        )
        kernel(input, weight, out, float(eps))
        return out

    if output_mode != "mxfp8":
        raise RuntimeError(f"Unsupported output_mode {output_mode}.")

    if block_scale is None:
        raise RuntimeError("MXFP8 output requires block_scale.")
    kernel = tilelang_rmsnorm_silu_mxfp8(
        rows_per_block=rows_per_block,
        threads_per_row=threads_per_row,
        values_per_thread=values_per_thread,
    )
    kernel(input, weight, out, block_scale, float(eps))
    return out, block_scale


@mate_api
def fused_rmsnorm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    block_scale: torch.Tensor | None = None,
    rows_per_block: int | None = None,
    threads_per_row: int | None = None,
):
    if input.dim() != 2:
        raise RuntimeError(
            "fused_rmsnorm_silu expects input [num_tokens, hidden_size]."
        )
    if input.dtype != torch.bfloat16:
        raise RuntimeError("fused_rmsnorm_silu expects bfloat16 input.")
    if input.stride(-1) != 1:
        raise RuntimeError(
            "fused_rmsnorm_silu requires a contiguous input last dimension."
        )
    if weight.dim() != 1 or weight.size(0) != input.size(1):
        raise RuntimeError("fused_rmsnorm_silu expects weight [hidden_size].")
    if weight.dtype != torch.bfloat16:
        raise RuntimeError("fused_rmsnorm_silu expects bfloat16 weight.")
    if weight.device != input.device:
        raise RuntimeError(
            "fused_rmsnorm_silu requires input and weight on the same device."
        )
    if weight.stride(0) != 1:
        raise RuntimeError("fused_rmsnorm_silu requires contiguous weight.")

    if out is None:
        out = torch.empty_like(input)
    if out.device != input.device:
        raise RuntimeError("fused_rmsnorm_silu requires out on the input device.")
    mode = _output_mode(out, block_scale)

    if mode in ("bf16", "fp8"):
        if out.shape != input.shape:
            raise RuntimeError("BF16/FP8 out shape must match input shape.")
        if out.stride(-1) != 1:
            raise RuntimeError("BF16/FP8 out requires a contiguous last dimension.")
        return _fused_rmsnorm_silu_impl(
            input,
            weight,
            eps,
            out,
            None,
            mode,
            rows_per_block_override=rows_per_block,
            threads_per_row_override=threads_per_row,
        )

    hidden_size = input.size(1)
    if hidden_size % _MXFP8_BLOCK_SIZE != 0:
        raise RuntimeError("MXFP8 output requires hidden_size divisible by 32.")

    expected_out_shape = (input.size(0), hidden_size)
    if out.shape != expected_out_shape:
        raise RuntimeError(f"MXFP8 out shape must be {expected_out_shape}.")
    if out.stride(-1) != 1:
        raise RuntimeError("MXFP8 out requires a contiguous last dimension.")

    expected_scale_shape = (input.size(0), hidden_size // _MXFP8_BLOCK_SIZE)
    if block_scale is None:
        block_scale = torch.empty(
            expected_scale_shape,
            device=input.device,
            dtype=torch.float8_e8m0fnu,
        )
    if block_scale.shape != expected_scale_shape:
        raise RuntimeError(f"MXFP8 block_scale shape must be {expected_scale_shape}.")
    if block_scale.dtype != torch.float8_e8m0fnu:
        raise RuntimeError("MXFP8 block_scale dtype must be float8_e8m0fnu.")
    if block_scale.device != input.device:
        raise RuntimeError("MXFP8 block_scale must be on the input device.")
    if not block_scale.is_contiguous():
        raise RuntimeError("MXFP8 block_scale must be contiguous.")

    return _fused_rmsnorm_silu_impl(
        input,
        weight,
        eps,
        out,
        block_scale,
        "mxfp8",
        rows_per_block_override=rows_per_block,
        threads_per_row_override=threads_per_row,
    )


if __name__ == "__main__":
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is not available.")

    device = torch.device("musa")
    torch.manual_seed(0)
    eps = 1e-6
    input = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    norm_ref = (
        input.float()
        * torch.rsqrt(input.float().square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    )
    ref = F.silu(norm_ref)

    out = torch.empty_like(input)
    result = fused_rmsnorm_silu(input, weight, eps, out=out)
    torch.testing.assert_close(result, ref.to(torch.bfloat16), rtol=2e-2, atol=2e-2)

    fp8_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    fp8_result = fused_rmsnorm_silu(input, weight, eps, out=fp8_out)
    torch.testing.assert_close(
        fp8_result.float(), ref.to(fp8_out.dtype).float(), rtol=2e-2, atol=0.25
    )

    mxfp8_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    block_scale = torch.empty(
        (input.size(0), input.size(1) // _MXFP8_BLOCK_SIZE),
        device=device,
        dtype=torch.float8_e8m0fnu,
    )
    mxfp8_result, block_scale = fused_rmsnorm_silu(
        input, weight, eps, out=mxfp8_out, block_scale=block_scale
    )
    scale_u8 = block_scale.contiguous().view(torch.uint8)
    scale_f32 = (scale_u8.to(torch.int32) << 23).view(torch.float32)
    scale = scale_f32.repeat_interleave(_MXFP8_BLOCK_SIZE, dim=1)
    torch.testing.assert_close(mxfp8_result.float() * scale, ref, rtol=2e-2, atol=0.5)

    print(
        f"fused_rmsnorm_silu smoke ok: shape={tuple(input.shape)}, dtype={input.dtype}"
    )
