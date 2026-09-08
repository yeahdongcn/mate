# mypy: ignore-errors

import torch
import tilelang
import tilelang.language as T

from mate.api_logging import mate_api

from ._norm_common import _JIT_CONFIG, _ceil_pow2_expr

__all__ = [
    "fused_dit_gate_residual_layernorm_gamma_beta",
    "fused_dit_gate_residual_layernorm_scale_shift",
    "fused_dit_residual_layernorm_scale_shift",
]


_DIT_LN_MODE_GATE_RESIDUAL_GAMMA_BETA = 0
_DIT_LN_MODE_RESIDUAL_SCALE_SHIFT = 1
_DIT_LN_MODE_GATE_RESIDUAL_SCALE_SHIFT = 2

_DIT_LN_OUTPUT_BF16 = 0
_DIT_LN_OUTPUT_NVFP4 = 1
_DIT_LN_OUTPUT_MXFP8 = 2

_DIT_HIDDEN_SIZE = 3072
_DIT_ROWS_PER_BLOCK = 1
_DIT_THREADS_PER_ROW = 384
_DIT_THREADS = _DIT_ROWS_PER_BLOCK * _DIT_THREADS_PER_ROW
_DIT_VALUES_PER_THREAD = 8
_MXFP8_BLOCK_SIZE = 32
_MXFP8_FP8_MAX = 448.0
_DIT_JIT_CONFIG = {
    "pass_configs": dict(_JIT_CONFIG["pass_configs"]),
    "compile_flags": _JIT_CONFIG["compile_flags"],
}
_DIT_JIT_CONFIG["pass_configs"][tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = True


def _as_dit_param_3d(
    tensor: torch.Tensor, input: torch.Tensor, name: str
) -> torch.Tensor:
    hidden_size = input.size(2)
    if tensor.dim() == 3:
        if tensor.size(0) != input.size(0) or tensor.size(1) != input.size(1):
            raise RuntimeError(f"{name} shape must match input batch and rows.")
        return tensor
    if tensor.dim() != 2:
        raise RuntimeError(f"{name} must be 2D or 3D.")
    if tensor.size(0) != input.size(1) or tensor.size(1) != hidden_size:
        raise RuntimeError(f"{name} shape must be [num_rows, hidden_size].")
    return torch.as_strided(
        tensor,
        size=(input.size(0), input.size(1), hidden_size),
        stride=(0, tensor.stride(0), tensor.stride(1)),
    )


def _check_common_inputs(
    input: torch.Tensor, residual: torch.Tensor | None, name: str
) -> None:
    if input.dim() != 3:
        raise RuntimeError(f"{name} expects input [batch, num_rows, 3072].")
    if input.dtype != torch.bfloat16:
        raise RuntimeError(f"{name} expects bfloat16 input.")
    if input.size(2) != _DIT_HIDDEN_SIZE:
        raise RuntimeError(f"{name} only supports hidden_size={_DIT_HIDDEN_SIZE}.")
    if input.stride(-1) != 1:
        raise RuntimeError(f"{name} requires contiguous hidden dimension for input.")
    if residual is not None:
        if residual.shape != input.shape:
            raise RuntimeError(f"{name} residual shape must match input.")
        if residual.dtype != torch.bfloat16:
            raise RuntimeError(f"{name} expects bfloat16 residual.")
        if residual.stride(-1) != 1:
            raise RuntimeError(
                f"{name} requires contiguous hidden dimension for residual."
            )
        if residual.device != input.device:
            raise RuntimeError(f"{name} requires residual on the input device.")


def _check_dit_param(
    tensor: torch.Tensor, input: torch.Tensor, name: str
) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"{name} must be bfloat16.")
    if tensor.device != input.device:
        raise RuntimeError(f"{name} must be on the input device.")
    if tensor.size(-1) != input.size(2):
        raise RuntimeError(f"{name} last dim must be hidden_size.")
    if tensor.stride(-1) != 1:
        raise RuntimeError(f"{name} must have stride 1 in the hidden dimension.")
    row_stride = tensor.stride(1) if tensor.dim() == 3 else tensor.stride(0)
    hidden_size = input.size(2)
    if row_stride not in (hidden_size, 6 * hidden_size):
        raise RuntimeError(f"{name} row stride must be hidden_size or 6 * hidden_size.")
    return _as_dit_param_3d(tensor, input, name)


def _check_bias(
    tensor: torch.Tensor | None, input: torch.Tensor, name: str
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dim() not in (1, 2):
        raise RuntimeError(f"{name} must be 1D or 2D.")
    if tensor.dim() == 2 and tensor.size(0) != 1:
        raise RuntimeError(f"{name} must have shape [hidden_size] or [1, hidden_size].")
    if tensor.size(-1) != input.size(2):
        raise RuntimeError(f"{name} last dim must be hidden_size.")
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"{name} must be float32.")
    if tensor.device != input.device:
        raise RuntimeError(f"{name} must be on the input device.")
    if tensor.stride(-1) != 1:
        raise RuntimeError(f"{name} must be contiguous in the hidden dimension.")
    return tensor.view(-1)


def _check_scale_factor(
    tensor: torch.Tensor | None, input: torch.Tensor, name: str
) -> None:
    if tensor is None:
        return
    if tensor.shape != (1,):
        raise RuntimeError(f"{name} must have shape [1].")
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"{name} must be float32.")
    if tensor.device != input.device:
        raise RuntimeError(f"{name} must be on the input device.")


def _dummy_f32(input: torch.Tensor) -> torch.Tensor:
    return torch.empty(input.size(2), dtype=torch.float32, device=input.device)


def _dummy_scale(input: torch.Tensor) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=input.device)


def _resolve_output_format(use_nvfp4: bool, use_mxfp8: bool) -> int:
    if use_nvfp4 and use_mxfp8:
        raise RuntimeError("Cannot enable both NVFP4 and MXFP8 output.")
    if use_nvfp4:
        return _DIT_LN_OUTPUT_NVFP4
    if use_mxfp8:
        return _DIT_LN_OUTPUT_MXFP8
    return _DIT_LN_OUTPUT_BF16


def _prepare_outputs(
    input: torch.Tensor,
    output_format: int,
    residual_out: torch.Tensor | None,
    norm_out: torch.Tensor | None,
    sf_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_rows, hidden_size = input.shape
    if residual_out is None:
        residual_out = torch.empty_like(input)
    if residual_out.shape != input.shape or residual_out.dtype != torch.bfloat16:
        raise RuntimeError("residual_out must be bfloat16 with input shape.")
    if residual_out.device != input.device or residual_out.stride(-1) != 1:
        raise RuntimeError(
            "residual_out must be on the input device and contiguous in the hidden dimension."
        )

    if output_format == _DIT_LN_OUTPUT_BF16:
        if norm_out is None:
            norm_out = torch.empty_like(input)
        if norm_out.shape != input.shape or norm_out.dtype != torch.bfloat16:
            raise RuntimeError("BF16 norm_out must be bfloat16 with input shape.")
        if norm_out.device != input.device or norm_out.stride(-1) != 1:
            raise RuntimeError(
                "norm_out must be on the input device and contiguous in the hidden dimension."
            )
        if sf_out is None:
            sf_out = torch.empty(1, dtype=torch.uint8, device=input.device)
        return residual_out, norm_out, sf_out

    if output_format == _DIT_LN_OUTPUT_NVFP4:
        raise NotImplementedError("NVFP4 output is not implemented yet.")

    expected_norm_shape = (batch_size, num_rows, hidden_size // 4)
    if norm_out is None:
        norm_out = torch.empty(
            expected_norm_shape, dtype=torch.int32, device=input.device
        )
    if norm_out.shape != expected_norm_shape or norm_out.dtype != torch.int32:
        raise RuntimeError(
            "MXFP8 norm_out must be int32 [batch, num_rows, hidden_size // 4]."
        )
    if norm_out.device != input.device or norm_out.stride(-1) != 1:
        raise RuntimeError(
            "MXFP8 norm_out must be on the input device and contiguous in the hidden dimension."
        )

    expected_sf_shape = (batch_size, num_rows, hidden_size // _MXFP8_BLOCK_SIZE)
    if sf_out is None:
        sf_out = torch.empty(expected_sf_shape, dtype=torch.uint8, device=input.device)
    if sf_out.shape != expected_sf_shape or sf_out.dtype != torch.uint8:
        raise RuntimeError(
            "MXFP8 sf_out must be uint8 [batch, num_rows, hidden_size // 32]."
        )
    if sf_out.device != input.device or sf_out.stride(-1) != 1:
        raise RuntimeError(
            "MXFP8 sf_out must be on the input device and contiguous in the hidden dimension."
        )
    return residual_out, norm_out, sf_out


@tilelang.jit(**_DIT_JIT_CONFIG)
def tilelang_fused_dit_layernorm(
    mode: int,
    output_format: int,
    use_residual: bool,
    use_gate_bias: bool,
    use_scale_bias: bool,
    use_shift_bias: bool,
    use_input_global_scaling_factor: bool,
):
    if output_format == _DIT_LN_OUTPUT_NVFP4:
        raise RuntimeError("NVFP4 output is not implemented.")
    is_gate_mode = mode in (
        _DIT_LN_MODE_GATE_RESIDUAL_GAMMA_BETA,
        _DIT_LN_MODE_GATE_RESIDUAL_SCALE_SHIFT,
    )
    is_gamma_beta = mode == _DIT_LN_MODE_GATE_RESIDUAL_GAMMA_BETA
    is_bf16_output = output_format == _DIT_LN_OUTPUT_BF16

    batch_size = T.dynamic("batch_size")
    num_rows = T.dynamic("num_rows")
    input_stride_b = T.dynamic("input_stride_b", "int64")
    input_stride_r = T.dynamic("input_stride_r", "int64")
    residual_stride_b = T.dynamic("residual_stride_b", "int64")
    residual_stride_r = T.dynamic("residual_stride_r", "int64")
    residual_out_stride_b = T.dynamic("residual_out_stride_b", "int64")
    residual_out_stride_r = T.dynamic("residual_out_stride_r", "int64")
    norm_out_stride_b = T.dynamic("norm_out_stride_b", "int64")
    norm_out_stride_r = T.dynamic("norm_out_stride_r", "int64")
    sf_out_stride_b = T.dynamic("sf_out_stride_b", "int64")
    sf_out_stride_r = T.dynamic("sf_out_stride_r", "int64")
    gate_stride_b = T.dynamic("gate_stride_b", "int64")
    gate_stride_r = T.dynamic("gate_stride_r", "int64")
    scale_stride_b = T.dynamic("scale_stride_b", "int64")
    scale_stride_r = T.dynamic("scale_stride_r", "int64")
    shift_stride_b = T.dynamic("shift_stride_b", "int64")
    shift_stride_r = T.dynamic("shift_stride_r", "int64")

    hidden_size = _DIT_HIDDEN_SIZE
    threads = _DIT_THREADS
    rows_per_block = _DIT_ROWS_PER_BLOCK
    threads_per_row = _DIT_THREADS_PER_ROW
    values_per_thread = _DIT_VALUES_PER_THREAD
    tile_width = threads_per_row * values_per_thread
    full_groups = hidden_size // tile_width
    blocks_per_row = hidden_size // _MXFP8_BLOCK_SIZE
    scale_blocks_per_tile = tile_width // _MXFP8_BLOCK_SIZE
    num_warps_per_row = threads_per_row // 32
    row_reduce_shuffles = num_warps_per_row.bit_length()

    x_type = T.StridedTensor(
        (batch_size, num_rows, hidden_size),
        (input_stride_b, input_stride_r, 1),
        T.bfloat16,
    )
    residual_type = T.StridedTensor(
        (batch_size, num_rows, hidden_size),
        (residual_stride_b, residual_stride_r, 1),
        T.bfloat16,
    )
    residual_out_type = T.StridedTensor(
        (batch_size, num_rows, hidden_size),
        (residual_out_stride_b, residual_out_stride_r, 1),
        T.bfloat16,
    )
    gate_type = T.StridedTensor(
        (batch_size, num_rows, hidden_size),
        (gate_stride_b, gate_stride_r, 1),
        T.bfloat16,
    )
    scale_type = T.StridedTensor(
        (batch_size, num_rows, hidden_size),
        (scale_stride_b, scale_stride_r, 1),
        T.bfloat16,
    )
    shift_type = T.StridedTensor(
        (batch_size, num_rows, hidden_size),
        (shift_stride_b, shift_stride_r, 1),
        T.bfloat16,
    )
    gate_bias_type = T.Tensor((hidden_size,), T.float32)
    gamma_type = T.Tensor((hidden_size,), T.float32)
    beta_type = T.Tensor((hidden_size,), T.float32)
    scale_bias_type = T.Tensor((hidden_size,), T.float32)
    shift_bias_type = T.Tensor((hidden_size,), T.float32)
    scale_factor_type = T.Tensor((1,), T.float32)
    if output_format == _DIT_LN_OUTPUT_MXFP8:
        norm_out_type = T.StridedTensor(
            (batch_size, num_rows, hidden_size),
            (norm_out_stride_b, norm_out_stride_r, 1),
            T.float8_e4m3fn,
        )
        sf_out_type = T.StridedTensor(
            (batch_size, num_rows, blocks_per_row),
            (sf_out_stride_b, sf_out_stride_r, 1),
            T.uint8,
        )
    else:
        norm_out_type = T.StridedTensor(
            (batch_size, num_rows, hidden_size),
            (norm_out_stride_b, norm_out_stride_r, 1),
            T.bfloat16,
        )
        sf_out_type = T.Tensor((1,), T.uint8)

    @T.prim_func
    def tilelang_fused_dit_layernorm_kernel(
        input: x_type,
        residual: residual_type,
        gate: gate_type,
        gate_bias: gate_bias_type,
        gamma: gamma_type,
        beta: beta_type,
        scale: scale_type,
        scale_bias: scale_bias_type,
        shift: shift_type,
        shift_bias: shift_bias_type,
        residual_out: residual_out_type,
        norm_out: norm_out_type,
        sf_out: sf_out_type,
        input_global_scaling_factor: scale_factor_type,
        eps: T.float32,
    ):
        num_blocks = num_rows // rows_per_block

        with T.Kernel(num_blocks, batch_size, threads=threads) as (block_idx, batch):
            T.assume(input_stride_b % values_per_thread == 0)
            T.assume(input_stride_r % values_per_thread == 0)
            T.assume(residual_stride_b % values_per_thread == 0)
            T.assume(residual_stride_r % values_per_thread == 0)
            T.assume(residual_out_stride_b % values_per_thread == 0)
            T.assume(residual_out_stride_r % values_per_thread == 0)
            T.assume(norm_out_stride_b % values_per_thread == 0)
            T.assume(norm_out_stride_r % values_per_thread == 0)
            T.assume(gate_stride_b % values_per_thread == 0)
            T.assume(gate_stride_r % values_per_thread == 0)
            T.assume(scale_stride_b % values_per_thread == 0)
            T.assume(scale_stride_r % values_per_thread == 0)
            T.assume(shift_stride_b % values_per_thread == 0)
            T.assume(shift_stride_r % values_per_thread == 0)
            tid = T.get_thread_binding()
            lane = tid % threads_per_row
            row_in_block = tid // threads_per_row
            warp_lane = lane % 32
            warp_in_row = lane // 32
            row = block_idx * rows_per_block + row_in_block

            sum_x = T.alloc_local((1,), T.float32)
            sum_sq = T.alloc_local((1,), T.float32)
            input_local = T.alloc_local((values_per_thread,), T.bfloat16)
            gate_local = T.alloc_local((values_per_thread,), T.bfloat16)
            scale_local = T.alloc_local((values_per_thread,), T.bfloat16)
            shift_local = T.alloc_local((values_per_thread,), T.bfloat16)
            residual_local = T.alloc_local((values_per_thread,), T.bfloat16)
            norm_bf16_local = T.alloc_local((values_per_thread,), T.bfloat16)
            norm_local = T.alloc_local((values_per_thread,), T.float32)
            fp8_local = T.alloc_local((values_per_thread,), T.float8_e4m3fn)
            gamma_local = T.alloc_local((values_per_thread,), T.float32)
            beta_local = T.alloc_local((values_per_thread,), T.float32)
            scale_bias_local = T.alloc_local((values_per_thread,), T.float32)
            shift_bias_local = T.alloc_local((values_per_thread,), T.float32)
            warp_sum_x_shared = T.alloc_shared(
                (rows_per_block, num_warps_per_row), T.float32
            )
            warp_sum_sq_shared = T.alloc_shared(
                (rows_per_block, num_warps_per_row), T.float32
            )
            block_scale_shared = T.alloc_shared(
                (rows_per_block, blocks_per_row), T.float32
            )

            sum_x[0] = 0.0
            sum_sq[0] = 0.0

            for group in T.serial(0, full_groups):
                residual_base = group * tile_width + lane * values_per_thread
                for v in T.vectorized(values_per_thread):
                    input_local[v] = input[batch, row, residual_base + v]
                if use_residual:
                    for v in T.vectorized(values_per_thread):
                        residual_local[v] = residual[batch, row, residual_base + v]
                if is_gate_mode:
                    for v in T.vectorized(values_per_thread):
                        gate_local[v] = gate[batch, row, residual_base + v]
                for v in T.unroll(values_per_thread):
                    col = residual_base + v
                    input_value = T.cast(input_local[v], T.float32)
                    if use_input_global_scaling_factor:
                        input_value = input_value * input_global_scaling_factor[0]
                    residual_value = 0.0
                    if use_residual:
                        residual_value = T.cast(residual_local[v], T.float32)
                    if is_gate_mode:
                        if use_gate_bias:
                            gate_value = (
                                T.cast(gate_local[v], T.float32) + gate_bias[col]
                            )
                        else:
                            gate_value = T.cast(gate_local[v], T.float32)
                        fused_value = input_value * gate_value + residual_value
                    else:
                        fused_value = input_value + residual_value
                    residual_local[v] = fused_value
                for v in T.vectorized(values_per_thread):
                    col = residual_base + v
                    residual_out[batch, row, col] = residual_local[v]
                for v in T.unroll(values_per_thread):
                    sum_x[0] += residual_local[v]
                    sum_sq[0] += residual_local[v] * residual_local[v]

            for offset in T.unroll(5):
                sum_x[0] += T.shfl_xor(sum_x[0], 16 >> offset)
                sum_sq[0] += T.shfl_xor(sum_sq[0], 16 >> offset)

            if warp_lane == 0:
                warp_sum_x_shared[row_in_block, warp_in_row] = sum_x[0]
                warp_sum_sq_shared[row_in_block, warp_in_row] = sum_sq[0]
            T.sync_threads()

            if warp_in_row == 0:
                if warp_lane < num_warps_per_row:
                    sum_x[0] = warp_sum_x_shared[row_in_block, warp_lane]
                    sum_sq[0] = warp_sum_sq_shared[row_in_block, warp_lane]
                else:
                    sum_x[0] = 0.0
                    sum_sq[0] = 0.0
                for offset in T.unroll(row_reduce_shuffles):
                    mask = (1 << (row_reduce_shuffles - 1)) >> offset
                    sum_x[0] += T.shfl_xor(sum_x[0], mask)
                    sum_sq[0] += T.shfl_xor(sum_sq[0], mask)
                if warp_lane == 0:
                    warp_sum_x_shared[row_in_block, 0] = sum_x[0]
                    warp_sum_sq_shared[row_in_block, 0] = sum_sq[0]
            T.sync_threads()

            sum_x[0] = warp_sum_x_shared[row_in_block, 0]
            sum_sq[0] = warp_sum_sq_shared[row_in_block, 0]
            mean = sum_x[0] / hidden_size
            variance = sum_sq[0] / hidden_size - mean * mean
            inv_std = T.rsqrt(T.max(variance, 0.0) + eps)

            for group in T.serial(0, full_groups):
                norm_base = group * tile_width + lane * values_per_thread
                if full_groups != 1:
                    for v in T.vectorized(values_per_thread):
                        residual_local[v] = residual_out[batch, row, norm_base + v]
                if is_gamma_beta:
                    for v in T.vectorized(values_per_thread):
                        gamma_local[v] = gamma[norm_base + v]
                        beta_local[v] = beta[norm_base + v]
                else:
                    for v in T.vectorized(values_per_thread):
                        scale_local[v] = scale[batch, row, norm_base + v]
                        shift_local[v] = shift[batch, row, norm_base + v]
                    if use_scale_bias:
                        for v in T.vectorized(values_per_thread):
                            scale_bias_local[v] = scale_bias[norm_base + v]
                    if use_shift_bias:
                        for v in T.vectorized(values_per_thread):
                            shift_bias_local[v] = shift_bias[norm_base + v]
                for v in T.unroll(values_per_thread):
                    col = norm_base + v
                    norm_input = residual_local[v]
                    if is_bf16_output:
                        if is_gamma_beta:
                            norm_bf16_local[v] = T.cast(
                                (norm_input - mean) * inv_std * gamma_local[v]
                                + beta_local[v],
                                T.bfloat16,
                            )
                        else:
                            if use_scale_bias:
                                scale_value = (
                                    T.cast(scale_local[v], T.float32)
                                    + scale_bias_local[v]
                                )
                            else:
                                scale_value = T.cast(scale_local[v], T.float32)
                            if use_shift_bias:
                                shift_value = (
                                    T.cast(shift_local[v], T.float32)
                                    + shift_bias_local[v]
                                )
                            else:
                                shift_value = T.cast(shift_local[v], T.float32)
                            norm_bf16_local[v] = T.cast(
                                (norm_input - mean) * inv_std * (1.0 + scale_value)
                                + shift_value,
                                T.bfloat16,
                            )
                    else:
                        if is_gamma_beta:
                            norm_local[v] = (norm_input - mean) * inv_std * gamma_local[
                                v
                            ] + beta_local[v]
                        else:
                            if use_scale_bias:
                                scale_value = (
                                    T.cast(scale_local[v], T.float32)
                                    + scale_bias_local[v]
                                )
                            else:
                                scale_value = T.cast(scale_local[v], T.float32)
                            if use_shift_bias:
                                shift_value = (
                                    T.cast(shift_local[v], T.float32)
                                    + shift_bias_local[v]
                                )
                            else:
                                shift_value = T.cast(shift_local[v], T.float32)
                            norm_local[v] = (norm_input - mean) * inv_std * (
                                1.0 + scale_value
                            ) + shift_value

                if is_bf16_output:
                    for v in T.vectorized(values_per_thread):
                        norm_out[batch, row, norm_base + v] = norm_bf16_local[v]
                else:
                    local_max = T.alloc_local((1,), T.float32)
                    local_max[0] = 0.0
                    for v in T.unroll(values_per_thread):
                        local_max[0] = T.max(local_max[0], T.abs(norm_local[v]))

                    local_max[0] = T.max(local_max[0], T.shfl_xor(local_max[0], 2))
                    local_max[0] = T.max(local_max[0], T.shfl_xor(local_max[0], 1))
                    if lane % 4 == 0:
                        block_col = group * scale_blocks_per_tile + lane // 4
                        scale_f32 = T.if_then_else(
                            local_max[0] > 0.0,
                            _ceil_pow2_expr(local_max[0] / _MXFP8_FP8_MAX),
                            0.0,
                        )
                        block_scale_shared[row_in_block, block_col] = scale_f32
                        sf_out[batch, row, block_col] = T.reinterpret(
                            T.cast(scale_f32, T.float8_e8m0fnu), T.uint8
                        )
                    T.sync_threads()

                    inv_scale = T.if_then_else(
                        block_scale_shared[
                            row_in_block, group * scale_blocks_per_tile + lane // 4
                        ]
                        != 0.0,
                        1.0
                        / block_scale_shared[
                            row_in_block, group * scale_blocks_per_tile + lane // 4
                        ],
                        0.0,
                    )
                    for v in T.vectorized(values_per_thread):
                        fp8_local[v] = T.cast(
                            norm_local[v] * inv_scale, T.float8_e4m3fn
                        )
                    for v in T.vectorized(values_per_thread):
                        norm_out[batch, row, norm_base + v] = fp8_local[v]

    symbol = (
        f"tilelang_fused_dit_layernorm_m{mode}_o{output_format}"
        f"_rs{int(use_residual)}_gb{int(use_gate_bias)}_sb{int(use_scale_bias)}_hb{int(use_shift_bias)}"
        f"_igs{int(use_input_global_scaling_factor)}_r{rows_per_block}_l{threads_per_row}_v{values_per_thread}"
    )
    return tilelang_fused_dit_layernorm_kernel.with_attr("global_symbol", symbol)


def _launch_fused_dit_layernorm(
    input: torch.Tensor,
    residual: torch.Tensor | None,
    gate: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    gamma: torch.Tensor | None,
    beta: torch.Tensor | None,
    scale: torch.Tensor | None,
    scale_bias: torch.Tensor | None,
    shift: torch.Tensor | None,
    shift_bias: torch.Tensor | None,
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    sf_out: torch.Tensor,
    input_global_scaling_factor: torch.Tensor | None,
    eps: float,
    mode: int,
    output_format: int,
) -> None:
    kernel = tilelang_fused_dit_layernorm(
        mode=mode,
        output_format=output_format,
        use_residual=residual is not None,
        use_gate_bias=gate_bias is not None,
        use_scale_bias=scale_bias is not None,
        use_shift_bias=shift_bias is not None,
        use_input_global_scaling_factor=input_global_scaling_factor is not None,
    )
    kernel(
        input,
        residual if residual is not None else input,
        gate if gate is not None else input,
        gate_bias if gate_bias is not None else _dummy_f32(input),
        gamma if gamma is not None else _dummy_f32(input),
        beta if beta is not None else _dummy_f32(input),
        scale if scale is not None else input,
        scale_bias if scale_bias is not None else _dummy_f32(input),
        shift if shift is not None else input,
        shift_bias if shift_bias is not None else _dummy_f32(input),
        residual_out,
        norm_out.view(torch.float8_e4m3fn)
        if output_format == _DIT_LN_OUTPUT_MXFP8
        else norm_out,
        sf_out,
        input_global_scaling_factor
        if input_global_scaling_factor is not None
        else _dummy_scale(input),
        float(eps),
    )


@mate_api
def fused_dit_gate_residual_layernorm_gamma_beta(
    input: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    *,
    gate_bias: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: torch.Tensor | None = None,
    input_global_scaling_factor: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    norm_out: torch.Tensor | None = None,
    sf_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    name = "fused_dit_gate_residual_layernorm_gamma_beta"
    output_format = _resolve_output_format(use_nvfp4, use_mxfp8)
    _check_common_inputs(input, residual, name)
    gate = _check_dit_param(gate, input, "gate")
    gate_bias = _check_bias(gate_bias, input, "gate_bias")
    _check_scale_factor(global_scaling_factor, input, "global_scaling_factor")
    _check_scale_factor(
        input_global_scaling_factor, input, "input_global_scaling_factor"
    )
    if gamma.shape != (input.size(2),) or beta.shape != (input.size(2),):
        raise RuntimeError("gamma and beta must be [hidden_size].")
    if gamma.dtype != torch.float32 or beta.dtype != torch.float32:
        raise RuntimeError("gamma and beta must be float32.")
    if gamma.device != input.device or beta.device != input.device:
        raise RuntimeError("gamma and beta must be on the input device.")
    residual_out, norm_out, sf_out = _prepare_outputs(
        input, output_format, residual_out, norm_out, sf_out
    )
    _launch_fused_dit_layernorm(
        input,
        residual,
        gate,
        gate_bias,
        gamma,
        beta,
        None,
        None,
        None,
        None,
        residual_out,
        norm_out,
        sf_out,
        input_global_scaling_factor,
        epsilon,
        _DIT_LN_MODE_GATE_RESIDUAL_GAMMA_BETA,
        output_format,
    )
    return residual_out, norm_out


@mate_api
def fused_dit_gate_residual_layernorm_scale_shift(
    input: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    gate_bias: torch.Tensor | None = None,
    scale_bias: torch.Tensor | None = None,
    shift_bias: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: torch.Tensor | None = None,
    input_global_scaling_factor: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    norm_out: torch.Tensor | None = None,
    sf_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    name = "fused_dit_gate_residual_layernorm_scale_shift"
    output_format = _resolve_output_format(use_nvfp4, use_mxfp8)
    _check_common_inputs(input, residual, name)
    gate = _check_dit_param(gate, input, "gate")
    scale = _check_dit_param(scale, input, "scale")
    shift = _check_dit_param(shift, input, "shift")
    gate_bias = _check_bias(gate_bias, input, "gate_bias")
    scale_bias = _check_bias(scale_bias, input, "scale_bias")
    shift_bias = _check_bias(shift_bias, input, "shift_bias")
    _check_scale_factor(global_scaling_factor, input, "global_scaling_factor")
    _check_scale_factor(
        input_global_scaling_factor, input, "input_global_scaling_factor"
    )
    residual_out, norm_out, sf_out = _prepare_outputs(
        input, output_format, residual_out, norm_out, sf_out
    )
    _launch_fused_dit_layernorm(
        input,
        residual,
        gate,
        gate_bias,
        None,
        None,
        scale,
        scale_bias,
        shift,
        shift_bias,
        residual_out,
        norm_out,
        sf_out,
        input_global_scaling_factor,
        epsilon,
        _DIT_LN_MODE_GATE_RESIDUAL_SCALE_SHIFT,
        output_format,
    )
    return residual_out, norm_out


@mate_api
def fused_dit_residual_layernorm_scale_shift(
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    scale_bias: torch.Tensor | None = None,
    shift_bias: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: torch.Tensor | None = None,
    input_global_scaling_factor: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    norm_out: torch.Tensor | None = None,
    sf_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    name = "fused_dit_residual_layernorm_scale_shift"
    output_format = _resolve_output_format(use_nvfp4, use_mxfp8)
    _check_common_inputs(input, residual, name)
    scale = _check_dit_param(scale, input, "scale")
    shift = _check_dit_param(shift, input, "shift")
    scale_bias = _check_bias(scale_bias, input, "scale_bias")
    shift_bias = _check_bias(shift_bias, input, "shift_bias")
    _check_scale_factor(global_scaling_factor, input, "global_scaling_factor")
    _check_scale_factor(
        input_global_scaling_factor, input, "input_global_scaling_factor"
    )
    residual_out, norm_out, sf_out = _prepare_outputs(
        input, output_format, residual_out, norm_out, sf_out
    )
    _launch_fused_dit_layernorm(
        input,
        residual,
        None,
        None,
        None,
        None,
        scale,
        scale_bias,
        shift,
        shift_bias,
        residual_out,
        norm_out,
        sf_out,
        input_global_scaling_factor,
        epsilon,
        _DIT_LN_MODE_RESIDUAL_SCALE_SHIFT,
        output_format,
    )
    return residual_out, norm_out


def _dit_layernorm_core_f32(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = x_f32.square().mean(dim=-1, keepdim=True) - mean.square()
    return (x_f32 - mean) * torch.rsqrt(var + eps)


def _dequant_mxfp8_linear(norm_out: torch.Tensor, sf_out: torch.Tensor) -> torch.Tensor:
    fp8 = norm_out.view(torch.float8_e4m3fn)
    scale = (
        (sf_out.to(torch.int32) << 23)
        .view(torch.float32)
        .repeat_interleave(_MXFP8_BLOCK_SIZE, dim=-1)
    )
    return fp8.float() * scale


def _make_dit_param(
    batch_size: int,
    num_rows: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(
        batch_size,
        num_rows * 6,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )[:, ::6, :]


def _check_output(
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    ref_residual: torch.Tensor,
    ref_norm: torch.Tensor,
    output_mode: str,
    sf_out: torch.Tensor | None,
) -> None:
    torch.testing.assert_close(residual_out, ref_residual, rtol=2e-2, atol=2e-2)
    if output_mode == "mxfp8":
        if sf_out is None:
            raise RuntimeError("MXFP8 check requires sf_out.")
        torch.testing.assert_close(
            _dequant_mxfp8_linear(norm_out, sf_out),
            ref_norm.float(),
            rtol=2e-2,
            atol=2.1,
        )
    else:
        torch.testing.assert_close(norm_out, ref_norm, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is not available.")

    device = torch.device("musa")
    torch.manual_seed(0)
    batch_size = 1
    num_rows = 1
    eps = 1e-6
    output_mode = "bf16"
    shape = (batch_size, num_rows, _DIT_HIDDEN_SIZE)
    input = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    residual = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    gate = _make_dit_param(batch_size, num_rows, _DIT_HIDDEN_SIZE, device)
    scale = _make_dit_param(batch_size, num_rows, _DIT_HIDDEN_SIZE, device)
    shift = _make_dit_param(batch_size, num_rows, _DIT_HIDDEN_SIZE, device)
    gamma = torch.randn(_DIT_HIDDEN_SIZE, device=device, dtype=torch.float32)
    beta = torch.randn(_DIT_HIDDEN_SIZE, device=device, dtype=torch.float32)
    gate_bias = torch.randn(_DIT_HIDDEN_SIZE, device=device, dtype=torch.float32)
    scale_bias = torch.randn(_DIT_HIDDEN_SIZE, device=device, dtype=torch.float32)
    shift_bias = torch.randn(_DIT_HIDDEN_SIZE, device=device, dtype=torch.float32)

    cases = (
        (
            "gate_gamma_beta",
            lambda residual_out,
            norm_out,
            sf_out: fused_dit_gate_residual_layernorm_gamma_beta(
                input,
                residual,
                gate,
                gamma,
                beta,
                gate_bias=gate_bias,
                epsilon=eps,
                residual_out=residual_out,
                norm_out=norm_out,
                sf_out=sf_out,
            ),
            lambda: (
                (
                    residual.float()
                    + input.float() * (gate.float() + gate_bias.float())
                ).to(torch.bfloat16),
                _dit_layernorm_core_f32(
                    (
                        residual.float()
                        + input.float() * (gate.float() + gate_bias.float())
                    ).to(torch.bfloat16),
                    eps,
                )
                .mul(gamma.float())
                .add(beta.float())
                .to(torch.bfloat16),
            ),
        ),
        (
            "gate_scale_shift",
            lambda residual_out,
            norm_out,
            sf_out: fused_dit_gate_residual_layernorm_scale_shift(
                input,
                residual,
                gate,
                scale,
                shift,
                gate_bias=gate_bias,
                scale_bias=scale_bias,
                shift_bias=shift_bias,
                epsilon=eps,
                residual_out=residual_out,
                norm_out=norm_out,
                sf_out=sf_out,
            ),
            lambda: (
                (
                    residual.float()
                    + input.float() * (gate.float() + gate_bias.float())
                ).to(torch.bfloat16),
                (
                    _dit_layernorm_core_f32(
                        (
                            residual.float()
                            + input.float() * (gate.float() + gate_bias.float())
                        ).to(torch.bfloat16),
                        eps,
                    )
                    * (1.0 + scale.float() + scale_bias.float())
                    + shift.float()
                    + shift_bias.float()
                ).to(torch.bfloat16),
            ),
        ),
        (
            "residual_scale_shift",
            lambda residual_out,
            norm_out,
            sf_out: fused_dit_residual_layernorm_scale_shift(
                input,
                scale,
                shift,
                residual=None,
                scale_bias=scale_bias,
                shift_bias=shift_bias,
                epsilon=eps,
                residual_out=residual_out,
                norm_out=norm_out,
                sf_out=sf_out,
            ),
            lambda: (
                input.to(torch.bfloat16),
                (
                    _dit_layernorm_core_f32(input.to(torch.bfloat16), eps)
                    * (1.0 + scale.float() + scale_bias.float())
                    + shift.float()
                    + shift_bias.float()
                ).to(torch.bfloat16),
            ),
        ),
    )

    for mode_name, run_case, ref_case in cases:
        norm_out = torch.empty_like(input)
        sf_out = None
        residual_out = torch.empty_like(input)
        ref_residual, ref_norm = ref_case()
        run_case(residual_out, norm_out, sf_out)
        torch.musa.synchronize()
        _check_output(
            residual_out, norm_out, ref_residual, ref_norm, output_mode, sf_out
        )
        print(f"{mode_name} ok")

    print(f"fused_dit_layernorm smoke ok: shape={shape}, output_mode={output_mode}")
