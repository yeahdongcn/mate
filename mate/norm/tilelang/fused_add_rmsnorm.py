# mypy: ignore-errors

import functools

import torch
import tilelang
import tilelang.language as T

from mate.api_logging import mate_api

from ._norm_common import _JIT_CONFIG, _fp8_torch_max, _tl_fp8_dtype

__all__ = [
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_fp8_block_quant",
    "fused_add_rmsnorm_quant",
]


_FP8_BLOCK_SIZE = 128
_FP8_BLOCK_AMAX_FLOOR = 1e-4
_FP8_E4M3_MAX = 448.0
_FP8_BLOCK_QUANT_JIT_CONFIG = {
    **_JIT_CONFIG,
    "pass_configs": {
        **_JIT_CONFIG["pass_configs"],
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
}


@functools.cache
def _select_norm_schedule(
    hidden_size: int, num_rows: int, *, is_quant: bool
) -> tuple[int, int, int]:
    if hidden_size >= 8192:
        threads_per_row = 128
        rows_per_block = 1
        values_per_thread = 8 if is_quant else 16
    elif hidden_size >= 4096:
        rows_per_block = 1
        if is_quant:
            if num_rows <= 1024:
                threads_per_row = 128
                values_per_thread = 16
            elif num_rows >= 8192:
                threads_per_row = 64
                values_per_thread = 8
            else:
                threads_per_row = 128
                values_per_thread = 8
        else:
            threads_per_row = 128 if num_rows <= 1024 else 64
            values_per_thread = 16
    elif hidden_size >= 1024:
        threads_per_row = 64
        rows_per_block = 1
        values_per_thread = 16
    else:
        threads_per_row = 32
        rows_per_block = 16
        values_per_thread = 8

    while rows_per_block > 1 and num_rows % rows_per_block != 0:
        rows_per_block //= 2
    return rows_per_block, threads_per_row, values_per_thread


@functools.cache
def _select_fp8_block_quant_schedule(
    hidden_size: int, num_rows: int
) -> tuple[int, int, int]:
    rows_per_block = 1
    if hidden_size == 4096 and num_rows > 256:
        return rows_per_block, 256, 4
    if 6144 <= hidden_size <= 8192 and hidden_size % 1024 == 0:
        if hidden_size == 6144 and num_rows <= 256:
            return rows_per_block, 768, 8
        return rows_per_block, 128, 8
    values_per_thread = 8 if hidden_size <= 8192 else 16
    return rows_per_block, hidden_size // values_per_thread, values_per_thread


@tilelang.jit(**_FP8_BLOCK_QUANT_JIT_CONFIG)
def tilelang_fused_add_rmsnorm_fp8_block_quant(
    input_dtype,
    rows_per_block: int,
    threads_per_row: int,
    values_per_thread: int,
    hidden_size_static: int,
):
    if input_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "tilelang_fused_add_rmsnorm_fp8_block_quant expects fp16 or bf16 input."
        )

    rows = T.dynamic("rows")
    hidden_size = T.dynamic("hidden_size")
    x_stride_r = T.dynamic("x_stride_r")
    residual_stride_r = T.dynamic("residual_stride_r")
    out_stride_r = T.dynamic("out_stride_r")
    scale_stride_r = T.dynamic("scale_stride_r")
    normed_stride_r = T.dynamic("normed_stride_r")

    num_warps_per_row = threads_per_row // 32
    threads = rows_per_block * threads_per_row
    tile_width = threads_per_row * values_per_thread
    if hidden_size_static % tile_width != 0:
        raise RuntimeError(
            "tilelang_fused_add_rmsnorm_fp8_block_quant requires hidden_size "
            "divisible by threads_per_row * values_per_thread."
        )
    full_groups = hidden_size_static // tile_width
    cached_groups = max(full_groups - 1, 0)
    cache_width = cached_groups * tile_width
    lanes_per_quant_block = _FP8_BLOCK_SIZE // values_per_thread
    quant_reduce_shuffles = lanes_per_quant_block.bit_length() - 1
    scale_blocks_per_tile = tile_width // _FP8_BLOCK_SIZE

    x_type = T.StridedTensor((rows, hidden_size), (x_stride_r, 1), input_dtype)
    residual_type = T.StridedTensor(
        (rows, hidden_size), (residual_stride_r, 1), input_dtype
    )
    weight_type = T.StridedTensor((hidden_size,), (1,), input_dtype)
    out_type = T.StridedTensor((rows, hidden_size), (out_stride_r, 1), T.float8_e4m3)
    scale_type = T.StridedTensor(
        (rows, hidden_size // _FP8_BLOCK_SIZE),
        (scale_stride_r, 1),
        T.float32,
    )
    normed_type = T.StridedTensor(
        (rows, hidden_size), (normed_stride_r, 1), input_dtype
    )

    @T.prim_func
    def tilelang_fused_add_rmsnorm_fp8_block_quant_kernel(
        x: x_type,
        residual: residual_type,
        weight: weight_type,
        out: out_type,
        block_scale: scale_type,
        normed_out: normed_type,
        eps: T.float32,
    ):
        with T.Kernel(rows // rows_per_block, threads=threads) as (block_idx,):
            T.assume(x_stride_r % values_per_thread == 0)
            T.assume(residual_stride_r % values_per_thread == 0)
            T.assume(out_stride_r % values_per_thread == 0)
            T.assume(normed_stride_r % values_per_thread == 0)

            tid = T.get_thread_binding()
            lane = tid % threads_per_row
            row_in_block = tid // threads_per_row
            warp_lane = lane % 32
            warp_in_row = lane // 32
            row = block_idx * rows_per_block + row_in_block

            sum_sq = T.alloc_local((1,), T.float32)
            local_amax = T.alloc_local((1,), T.float32)
            residual_local = T.alloc_local((values_per_thread,), T.float32)
            weight_local = T.alloc_local((values_per_thread,), input_dtype)
            normed_local = T.alloc_local((values_per_thread,), input_dtype)
            normed_f32_local = T.alloc_local((values_per_thread,), T.float32)
            out_local = T.alloc_local((values_per_thread,), T.float8_e4m3)
            warp_sum_sq_shared = T.alloc_shared(
                (rows_per_block, num_warps_per_row), T.float32
            )
            residual_cache_shared = T.alloc_shared(
                (rows_per_block, max(cache_width, 1)), T.float32
            )

            sum_sq[0] = 0.0
            for group in T.serial(full_groups):
                base = group * tile_width + lane * values_per_thread
                for v in T.vectorized(values_per_thread):
                    residual_local[v] = T.cast(x[row, base + v], T.float32) + T.cast(
                        residual[row, base + v], T.float32
                    )
                for v in T.unroll(values_per_thread):
                    sum_sq[0] += residual_local[v] * residual_local[v]
                for v in T.vectorized(values_per_thread):
                    residual[row, base + v] = T.cast(residual_local[v], input_dtype)
                    if group < cached_groups:
                        residual_cache_shared[row_in_block, base + v] = residual_local[
                            v
                        ]

            for offset in T.unroll(5):
                sum_sq[0] += T.shfl_xor(sum_sq[0], 16 >> offset)
            if warp_lane == 0:
                warp_sum_sq_shared[row_in_block, warp_in_row] = sum_sq[0]
            T.sync_threads()

            if warp_in_row == 0:
                if warp_lane < num_warps_per_row:
                    sum_sq[0] = warp_sum_sq_shared[row_in_block, warp_lane]
                else:
                    sum_sq[0] = 0.0
                for offset in T.unroll(5):
                    sum_sq[0] += T.shfl_xor(sum_sq[0], 16 >> offset)
                if warp_lane == 0:
                    warp_sum_sq_shared[row_in_block, 0] = T.rsqrt(
                        sum_sq[0] / hidden_size_static + eps
                    )
            T.sync_threads()

            inv_rms = warp_sum_sq_shared[row_in_block, 0]
            for reverse_group in T.serial(full_groups):
                group = full_groups - 1 - reverse_group
                base = group * tile_width + lane * values_per_thread
                if group < cached_groups:
                    for v in T.vectorized(values_per_thread):
                        residual_local[v] = residual_cache_shared[
                            row_in_block, base + v
                        ]
                for v in T.vectorized(values_per_thread):
                    weight_local[v] = weight[base + v]

                local_amax[0] = 0.0
                for v in T.vectorized(values_per_thread):
                    normed_local[v] = T.cast(
                        residual_local[v]
                        * inv_rms
                        * T.cast(weight_local[v], T.float32),
                        input_dtype,
                    )
                for v in T.vectorized(values_per_thread):
                    normed_f32_local[v] = T.cast(normed_local[v], T.float32)
                for v in T.unroll(values_per_thread):
                    local_amax[0] = T.max(local_amax[0], T.abs(normed_f32_local[v]))
                for v in T.vectorized(values_per_thread):
                    normed_out[row, base + v] = normed_local[v]

                for offset in T.unroll(quant_reduce_shuffles):
                    local_amax[0] = T.max(
                        local_amax[0],
                        T.shfl_xor(local_amax[0], lanes_per_quant_block // 2 >> offset),
                    )
                amax = T.max(local_amax[0], _FP8_BLOCK_AMAX_FLOOR)
                inv_scale = T.alloc_local((1,), T.float32)
                inv_scale[0] = 0.0
                if lane % lanes_per_quant_block == 0:
                    block_col = (
                        group * scale_blocks_per_tile + lane // lanes_per_quant_block
                    )
                    block_scale[row, block_col] = amax * (1.0 / _FP8_E4M3_MAX)
                    inv_scale[0] = _FP8_E4M3_MAX / amax
                inv_scale[0] = T.shfl_sync(
                    inv_scale[0],
                    (warp_lane // lanes_per_quant_block) * lanes_per_quant_block,
                    32,
                )

                for v in T.vectorized(values_per_thread):
                    out_local[v] = T.cast(
                        normed_f32_local[v] * inv_scale[0],
                        T.float8_e4m3,
                    )
                for v in T.vectorized(values_per_thread):
                    out[row, base + v] = out_local[v]

    symbol = (
        "tilelang_fused_add_rmsnorm_fp8_block_quant"
        f"_{str(input_dtype).replace('torch.', '')}"
        f"_h{hidden_size_static}_r{rows_per_block}"
        f"_l{threads_per_row}_v{values_per_thread}"
    )
    return tilelang_fused_add_rmsnorm_fp8_block_quant_kernel.with_attr(
        "global_symbol", symbol
    )


@tilelang.jit(**_JIT_CONFIG)
def tilelang_fused_add_rmsnorm(
    input_dtype,
    weight_bias: float,
    is_quant: bool,
    rows_per_block: int,
    threads_per_row: int,
    values_per_thread: int,
    hidden_size_static: int,
    output_dtype=None,
    fp8_max: float | None = None,
):
    if input_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("tilelang_fused_add_rmsnorm expects fp16 or bf16 input.")
    if is_quant:
        if output_dtype is None or fp8_max is None:
            raise RuntimeError(
                "tilelang_fused_add_rmsnorm quant path requires output_dtype and fp8_max."
            )

    batch_size = T.dynamic("batch_size")
    hidden_size = T.dynamic("hidden_size")
    x_stride_b = T.dynamic("x_stride_b")
    residual_stride_b = T.dynamic("residual_stride_b")
    output_stride_b = T.dynamic("output_stride_b")
    if threads_per_row not in (32, 64, 128, 256):
        raise RuntimeError(
            "tilelang_fused_add_rmsnorm expects threads_per_row in {32, 64, 128, 256}."
        )
    num_warps_per_row = threads_per_row // 32
    num_shuffles = 5
    row_reduce_shuffles = (
        num_warps_per_row.bit_length() - 1 if num_warps_per_row > 1 else 0
    )
    threads = rows_per_block * threads_per_row
    tile_width = threads_per_row * values_per_thread
    full_groups = hidden_size_static // tile_width
    smem_groups = max(full_groups - 1, 0)
    cache_width = smem_groups * tile_width
    tail_elems = hidden_size_static % tile_width

    x_type = T.StridedTensor((batch_size, hidden_size), (x_stride_b, 1), input_dtype)
    residual_type = T.StridedTensor(
        (batch_size, hidden_size),
        (residual_stride_b, 1),
        input_dtype,
    )
    w_type = T.StridedTensor((hidden_size,), (1,), input_dtype)
    s_type = T.StridedTensor((1,), (1,), T.float32)
    if is_quant:
        output_type = T.StridedTensor(
            (batch_size, hidden_size), (output_stride_b, 1), output_dtype
        )
        local_output_dtype = output_dtype
    else:
        output_type = x_type
        local_output_dtype = input_dtype

    @T.prim_func
    def tilelang_fused_add_rmsnorm_kernel(
        x: x_type,
        residual: residual_type,
        weight: w_type,
        output: output_type,
        scale: s_type,
        eps: T.float32,
    ):
        num_blocks = batch_size // rows_per_block

        with T.Kernel(num_blocks, threads=threads) as (block_idx,):
            T.assume(x_stride_b % values_per_thread == 0)
            T.assume(residual_stride_b % values_per_thread == 0)
            if is_quant:
                T.assume(output_stride_b % values_per_thread == 0)

            tid = T.get_thread_binding()
            lane = tid % threads_per_row
            row_in_block = tid // threads_per_row
            warp_lane = lane % 32
            warp_in_row = lane // 32
            row = block_idx * rows_per_block + row_in_block

            sum_sq = T.alloc_local([1], dtype="float32")
            w_local = T.alloc_local([values_per_thread], dtype=input_dtype)
            residual_local = T.alloc_local([values_per_thread], dtype="float32")
            residual_tail_local = T.alloc_local([values_per_thread], dtype="float32")
            norm_local = T.alloc_local([values_per_thread], dtype=local_output_dtype)
            residual_cache_shared = T.alloc_shared(
                (rows_per_block, cache_width), dtype="float32"
            )
            warp_sum_sq_shared = T.alloc_shared(
                (rows_per_block, num_warps_per_row), dtype="float32"
            )

            sum_sq[0] = 0.0

            for group in T.serial(0, full_groups):
                base_sum = group * tile_width + lane * values_per_thread
                for v in T.vectorized(values_per_thread):
                    residual_local[v] = T.cast(
                        x[row, base_sum + v], "float32"
                    ) + T.cast(residual[row, base_sum + v], "float32")
                for v in T.unroll(values_per_thread):
                    sum_sq[0] += residual_local[v] * residual_local[v]
                for v in T.vectorized(values_per_thread):
                    if group < smem_groups:
                        residual_cache_shared[row_in_block, base_sum + v] = (
                            residual_local[v]
                        )

            if tail_elems != 0:
                tail_base_sum = full_groups * tile_width + lane * values_per_thread
                for v in T.unroll(values_per_thread):
                    j_h = tail_base_sum + v
                    if j_h < hidden_size_static:
                        residual_tail_local[v] = T.cast(
                            x[row, j_h], "float32"
                        ) + T.cast(residual[row, j_h], "float32")
                        sum_sq[0] += residual_tail_local[v] * residual_tail_local[v]

            for offset in T.unroll(num_shuffles):
                sum_sq[0] += T.shfl_xor(sum_sq[0], 16 >> offset)

            if num_warps_per_row == 1:
                if warp_lane == 0:
                    warp_sum_sq_shared[row_in_block, 0] = sum_sq[0]
            else:
                if warp_lane == 0:
                    warp_sum_sq_shared[row_in_block, warp_in_row] = sum_sq[0]
                T.sync_threads()
                if warp_in_row == 0:
                    if warp_lane < num_warps_per_row:
                        sum_sq[0] = warp_sum_sq_shared[row_in_block, warp_lane]
                    else:
                        sum_sq[0] = 0.0
                    for offset in T.unroll(row_reduce_shuffles):
                        sum_sq[0] += T.shfl_xor(
                            sum_sq[0], (num_warps_per_row // 2) >> offset
                        )
                    if warp_lane == 0:
                        warp_sum_sq_shared[row_in_block, 0] = sum_sq[0]
                T.sync_threads()

            T.sync_threads()
            sum_sq[0] = warp_sum_sq_shared[row_in_block, 0]

            inv_rms = T.rsqrt(sum_sq[0] / hidden_size_static + eps)
            scale_inv = 1.0
            if is_quant:
                scale_inv = 1.0 / scale[0]

            if smem_groups < full_groups:
                base_out = (full_groups - 1) * tile_width + lane * values_per_thread
                for v in T.vectorized(values_per_thread):
                    w_local[v] = weight[base_out + v]
                for v in T.unroll(values_per_thread):
                    w = T.cast(w_local[v], "float32") + weight_bias
                    if is_quant:
                        q = T.clamp(
                            residual_local[v] * inv_rms * w * scale_inv,
                            -fp8_max,
                            fp8_max,
                        )
                        norm_local[v] = T.cast(q, output_dtype)
                    else:
                        norm_local[v] = T.cast(
                            residual_local[v] * inv_rms * w, input_dtype
                        )
                for v in T.vectorized(values_per_thread):
                    residual[row, base_out + v] = T.cast(residual_local[v], input_dtype)
                    if is_quant:
                        output[row, base_out + v] = norm_local[v]
                    else:
                        x[row, base_out + v] = norm_local[v]

            for rev_group in T.serial(0, smem_groups):
                group_out = smem_groups - 1 - rev_group
                base_cached_out = group_out * tile_width + lane * values_per_thread
                for v in T.vectorized(values_per_thread):
                    residual_local[v] = residual_cache_shared[
                        row_in_block, base_cached_out + v
                    ]
                    w_local[v] = weight[base_cached_out + v]
                for v in T.unroll(values_per_thread):
                    w = T.cast(w_local[v], "float32") + weight_bias
                    if is_quant:
                        q = T.clamp(
                            residual_local[v] * inv_rms * w * scale_inv,
                            -fp8_max,
                            fp8_max,
                        )
                        norm_local[v] = T.cast(q, output_dtype)
                    else:
                        norm_local[v] = T.cast(
                            residual_local[v] * inv_rms * w, input_dtype
                        )
                for v in T.vectorized(values_per_thread):
                    residual[row, base_cached_out + v] = T.cast(
                        residual_local[v], input_dtype
                    )
                    if is_quant:
                        output[row, base_cached_out + v] = norm_local[v]
                    else:
                        x[row, base_cached_out + v] = norm_local[v]

            if tail_elems != 0:
                tail_base_out = full_groups * tile_width + lane * values_per_thread
                for v in T.unroll(values_per_thread):
                    j_h = tail_base_out + v
                    if j_h < hidden_size_static:
                        residual_value = residual_tail_local[v]
                        w = T.cast(weight[j_h], "float32") + weight_bias
                        residual[row, j_h] = T.cast(residual_value, input_dtype)
                        if is_quant:
                            q = T.clamp(
                                residual_value * inv_rms * w * scale_inv,
                                -fp8_max,
                                fp8_max,
                            )
                            output[row, j_h] = T.cast(q, output_dtype)
                        else:
                            x[row, j_h] = T.cast(
                                residual_value * inv_rms * w, input_dtype
                            )

    def _symbol_part(value):
        return (
            str(value)
            .replace("torch.", "")
            .replace(".", "p")
            .replace("-", "m")
            .replace(" ", "_")
        )

    symbol = (
        f"tilelang_fused_add_rmsnorm_kernel_{_symbol_part(input_dtype)}"
        f"_h{hidden_size_static}_wb{_symbol_part(weight_bias)}"
        f"_q{int(is_quant)}"
        f"_r{rows_per_block}_l{threads_per_row}_v{values_per_thread}"
    )
    return tilelang_fused_add_rmsnorm_kernel.with_attr("global_symbol", symbol)


@mate_api
def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    gemma: bool = False,
) -> None:
    _fused_add_rmsnorm_impl(
        x,
        residual,
        weight,
        eps=eps,
        gemma=gemma,
        is_quant=False,
        out=None,
    )


@mate_api
def fused_add_rmsnorm_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    gemma: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return _fused_add_rmsnorm_impl(
        x,
        residual,
        weight,
        eps=eps,
        gemma=gemma,
        is_quant=True,
        scale=scale,
        out=out,
    )


@mate_api
def fused_add_rmsnorm_fp8_block_quant(
    out: torch.Tensor,
    block_scale: torch.Tensor,
    normed_out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """Fused add RMSNorm with row-major 1x128 FP8 block quantization."""
    if input.dim() != 2:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant expects rank-2 input [rows, hidden_size]."
        )
    if residual.shape != input.shape:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant residual shape must match input shape."
        )
    if weight.dim() != 1 or weight.size(0) != input.size(1):
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant expects weight [hidden_size]."
        )
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant expects fp16 or bf16 input."
        )
    if residual.dtype != input.dtype or weight.dtype != input.dtype:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant requires matching input/residual/weight dtype."
        )

    rows, hidden_size = input.shape
    values_per_thread = 8 if hidden_size <= 8192 else 16
    if (
        hidden_size == 0
        or hidden_size > 16384
        or hidden_size % (32 * values_per_thread) != 0
    ):
        divisor = 32 * values_per_thread
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant requires hidden_size <= 16384 "
            f"and divisible by {divisor}."
        )
    if rows == 0:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant requires at least one input row."
        )

    expected_scale_shape = (rows, hidden_size // _FP8_BLOCK_SIZE)
    if out.shape != input.shape:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant out shape must match input shape."
        )
    if out.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant out dtype must be torch.float8_e4m3fn."
        )
    if block_scale.shape != expected_scale_shape:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant block_scale shape must be "
            f"{expected_scale_shape}."
        )
    if block_scale.dtype != torch.float32:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant block_scale dtype must be float32."
        )
    if not block_scale.is_contiguous():
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant block_scale must be contiguous row-major."
        )
    if normed_out.shape != input.shape or normed_out.dtype != input.dtype:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant normed_out must match input shape and dtype."
        )

    tensors = (residual, weight, out, block_scale, normed_out)
    if any(tensor.device != input.device for tensor in tensors):
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant requires all tensors on the same device."
        )
    matrix_tensors = (input, residual, out, normed_out)
    if any(tensor.stride(-1) != 1 for tensor in matrix_tensors):
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant requires contiguous last dimensions."
        )
    if weight.stride(0) != 1:
        raise RuntimeError(
            "fused_add_rmsnorm_fp8_block_quant requires contiguous weight."
        )

    rows_per_block, threads_per_row, values_per_thread = (
        _select_fp8_block_quant_schedule(hidden_size, rows)
    )
    kernel = tilelang_fused_add_rmsnorm_fp8_block_quant(
        input_dtype=input.dtype,
        rows_per_block=rows_per_block,
        threads_per_row=threads_per_row,
        values_per_thread=values_per_thread,
        hidden_size_static=hidden_size,
    )
    kernel(input, residual, weight, out, block_scale, normed_out, float(eps))


def _fused_add_rmsnorm_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    gemma: bool = False,
    is_quant: bool = False,
    scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if x.dim() != 2:
        raise RuntimeError("fused_add_rmsnorm expects rank-2 input [B, H].")
    if residual.dim() != 2:
        raise RuntimeError("fused_add_rmsnorm expects rank-2 residual [B, H].")
    if weight.dim() != 1:
        raise RuntimeError("fused_add_rmsnorm expects rank-1 weight [H].")
    if residual.shape != x.shape:
        raise RuntimeError("fused_add_rmsnorm residual shape must match input shape.")
    if x.size(1) != weight.size(0):
        raise RuntimeError(
            f"fused_add_rmsnorm hidden size mismatch: x.size(1)={x.size(1)}, "
            f"weight.size(0)={weight.size(0)}."
        )
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("fused_add_rmsnorm expects fp16 or bf16 input.")
    if residual.dtype != x.dtype or weight.dtype != x.dtype:
        raise RuntimeError(
            "fused_add_rmsnorm requires matching input/residual/weight dtype."
        )
    if residual.device != x.device or weight.device != x.device:
        raise RuntimeError("fused_add_rmsnorm requires all tensors on the same device.")
    if x.stride(-1) != 1 or residual.stride(-1) != 1:
        raise RuntimeError("fused_add_rmsnorm requires contiguous last dimension.")
    if weight.stride(-1) != 1:
        raise RuntimeError("fused_add_rmsnorm requires contiguous weight.")

    rows_per_block, threads_per_row, values_per_thread = _select_norm_schedule(
        x.size(1), x.size(0), is_quant=is_quant
    )

    if rows_per_block < 1 or rows_per_block * threads_per_row > 1024:
        raise RuntimeError(
            "fused_add_rmsnorm requires 1 <= rows_per_block * threads_per_row <= 1024."
        )
    if x.size(0) % rows_per_block != 0:
        raise RuntimeError(
            "fused_add_rmsnorm requires batch size divisible by rows_per_block."
        )

    if is_quant:
        if scale is None:
            raise RuntimeError("fused_add_rmsnorm_quant expects a scalar scale tensor.")
        if scale.numel() != 1:
            raise RuntimeError("fused_add_rmsnorm_quant expects a scalar scale tensor.")
        if scale.dtype != torch.float32:
            raise RuntimeError("fused_add_rmsnorm_quant expects scale dtype float32.")
        if x.device != scale.device:
            raise RuntimeError(
                "fused_add_rmsnorm_quant requires all tensors on the same device."
            )
        if out is None:
            out = torch.empty_strided(
                x.shape, x.stride(), device=x.device, dtype=torch.float8_e4m3fn
            )
        if out.shape != x.shape:
            raise RuntimeError(
                "fused_add_rmsnorm_quant output shape must match input shape."
            )
        if out.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise RuntimeError(
                "fused_add_rmsnorm_quant output dtype must be torch.float8_e4m3fn or torch.float8_e5m2."
            )
        if out.device != x.device:
            raise RuntimeError(
                "fused_add_rmsnorm_quant output device must match input device."
            )
        if out.stride(-1) != 1:
            raise RuntimeError(
                "fused_add_rmsnorm_quant requires output contiguous last dimension."
            )
        kernel = tilelang_fused_add_rmsnorm(
            input_dtype=x.dtype,
            weight_bias=1.0 if gemma else 0.0,
            is_quant=True,
            rows_per_block=rows_per_block,
            threads_per_row=threads_per_row,
            values_per_thread=values_per_thread,
            hidden_size_static=x.size(1),
            output_dtype=_tl_fp8_dtype(out.dtype),
            fp8_max=_fp8_torch_max(out.dtype),
        )
        kernel(x, residual, weight, out, scale, float(eps))
        return out
    else:
        kernel = tilelang_fused_add_rmsnorm(
            input_dtype=x.dtype,
            weight_bias=1.0 if gemma else 0.0,
            is_quant=False,
            rows_per_block=rows_per_block,
            threads_per_row=threads_per_row,
            values_per_thread=values_per_thread,
            hidden_size_static=x.size(1),
        )
        scale = torch.ones(1, device=x.device, dtype=torch.float32)
        kernel(x, residual, weight, x, scale, float(eps))
        return None


if __name__ == "__main__":

    def _torch_reference(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        gemma: bool,
        is_quant: bool,
        scale: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_out = (x.float() + residual.float()).to(x.dtype)
        weight_f32 = weight.float() + (1.0 if gemma else 0.0)
        norm_out = residual_out.float()
        norm_out = norm_out * torch.rsqrt(
            norm_out.square().mean(dim=-1, keepdim=True) + eps
        )
        norm_out = norm_out * weight_f32
        if is_quant:
            if scale is None or out_dtype is None:
                raise RuntimeError("quant reference requires scale and out_dtype.")
            fp8_max = _fp8_torch_max(out_dtype)
            norm_out = (norm_out / scale.float()).clamp(-fp8_max, fp8_max).to(out_dtype)
        else:
            norm_out = norm_out.to(x.dtype)
        return norm_out, residual_out

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is not available.")

    device = torch.device("musa")
    torch.manual_seed(0)

    eps = 1e-6
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    residual = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    scale = torch.ones(1, device=device, dtype=torch.float32)

    x_in = x.clone()
    residual_in = residual.clone()
    norm_ref, residual_ref = _torch_reference(
        x_in, residual_in, weight, eps, False, False
    )
    fused_add_rmsnorm(x, residual, weight, eps=eps)
    torch.testing.assert_close(x, norm_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(residual, residual_ref, rtol=2e-2, atol=2e-2)

    x = x_in.clone()
    residual = residual_in.clone()
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    q_ref, residual_ref = _torch_reference(
        x_in,
        residual_in,
        weight,
        eps,
        False,
        True,
        scale=scale,
        out_dtype=out.dtype,
    )
    fused_add_rmsnorm_quant(x, residual, weight, scale, eps=eps, out=out)
    torch.testing.assert_close(out.float(), q_ref.float(), rtol=0.0, atol=0.25)
    torch.testing.assert_close(residual, residual_ref, rtol=2e-2, atol=2e-2)

    print(f"fused_add_rmsnorm smoke ok: shape={tuple(x.shape)}, dtype={x.dtype}")
