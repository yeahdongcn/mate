# mypy: ignore-errors

import functools

import torch
import tilelang
import tilelang.language as T

from mate.api_logging import mate_api

from ._norm_common import _JIT_CONFIG, _fp8_torch_max, _tl_fp8_dtype

__all__ = ["layernorm", "layernorm_quant"]


@functools.cache
def _select_norm_schedule(hidden_size: int, num_rows: int) -> tuple[int, int, int]:
    if hidden_size >= 4096:
        threads_per_row = 128
        rows_per_block = 4
    elif hidden_size >= 2048:
        threads_per_row = 64
        rows_per_block = 8
    else:
        threads_per_row = 32
        rows_per_block = 16

    values_per_thread = 8
    while rows_per_block > 1 and num_rows % rows_per_block != 0:
        rows_per_block //= 2
    return rows_per_block, threads_per_row, values_per_thread


@tilelang.jit(**_JIT_CONFIG)
def tilelang_layernorm(
    input_dtype,
    is_quant: bool,
    rows_per_block: int,
    threads_per_row: int,
    values_per_thread: int,
    output_dtype=None,
    output_torch_dtype: torch.dtype | None = None,
    fp8_max: float | None = None,
):
    if input_dtype != torch.bfloat16:
        raise RuntimeError("tilelang_layernorm expects bfloat16 input.")
    if is_quant and (
        output_dtype is None or output_torch_dtype is None or fp8_max is None
    ):
        raise RuntimeError(
            "tilelang_layernorm quant path requires output dtype and fp8 max."
        )
    if threads_per_row not in (32, 64, 128, 256):
        raise RuntimeError(
            "tilelang_layernorm expects threads_per_row in {32, 64, 128, 256}."
        )

    batch_size = T.dynamic("batch_size")
    hidden_size = T.dynamic("hidden_size")
    x_stride_b = T.dynamic("x_stride_b")
    y_stride_b = T.dynamic("y_stride_b")
    output_stride_b = T.dynamic("output_stride_b")

    num_warps_per_row = threads_per_row // 32
    num_shuffles = 5
    row_reduce_shuffles = (
        num_warps_per_row.bit_length() - 1 if num_warps_per_row > 1 else 0
    )
    threads = rows_per_block * threads_per_row
    tile_width = threads_per_row * values_per_thread
    full_groups = hidden_size // tile_width
    tail_elems = hidden_size % tile_width

    x_type = T.StridedTensor((batch_size, hidden_size), (x_stride_b, 1), input_dtype)
    gemma_type = T.StridedTensor((hidden_size,), (1,), T.float32)
    beta_type = T.StridedTensor((hidden_size,), (1,), T.float32)
    if is_quant:
        y_dtype = output_dtype
        y_type = T.StridedTensor(
            (batch_size, hidden_size), (output_stride_b, 1), y_dtype
        )
    else:
        y_dtype = input_dtype
        y_type = T.StridedTensor(
            (batch_size, hidden_size), (y_stride_b, 1), input_dtype
        )

    @T.prim_func
    def tilelang_layernorm_kernel(
        x: x_type,
        gemma: gemma_type,
        beta: beta_type,
        y: y_type,
        scale: T.StridedTensor((1,), (1,), T.float32),
        eps: T.float32,
    ):
        num_blocks = batch_size // rows_per_block

        with T.Kernel(num_blocks, threads=threads) as (block_idx,):
            T.assume(x_stride_b % values_per_thread == 0)
            if is_quant:
                T.assume(output_stride_b % values_per_thread == 0)
            else:
                T.assume(y_stride_b % values_per_thread == 0)

            tid = T.get_thread_binding()
            lane = tid % threads_per_row
            row_in_block = tid // threads_per_row
            warp_lane = lane % 32
            warp_in_row = lane // 32
            row = block_idx * rows_per_block + row_in_block

            sum_x = T.alloc_local([1], dtype="float32")
            sum_sq = T.alloc_local([1], dtype="float32")
            x_local = T.alloc_local([values_per_thread], dtype=input_dtype)
            gamma_local = T.alloc_local([values_per_thread], dtype=T.float32)
            beta_local = T.alloc_local([values_per_thread], dtype=T.float32)
            y_local = T.alloc_local([values_per_thread], dtype=input_dtype)
            warp_sum_x_shared = T.alloc_shared(
                (rows_per_block, num_warps_per_row), dtype="float32"
            )
            warp_sum_sq_shared = T.alloc_shared(
                (rows_per_block, num_warps_per_row), dtype="float32"
            )

            sum_x[0] = 0.0
            sum_sq[0] = 0.0

            for group in T.serial(0, full_groups):
                base = group * tile_width + lane * values_per_thread
                for v in T.vectorized(values_per_thread):
                    x_local[v] = x[row, base + v]
                for v in T.unroll(values_per_thread):
                    x_value = T.cast(x_local[v], "float32")
                    sum_x[0] += x_value
                    sum_sq[0] += x_value * x_value

            if tail_elems != 0:
                tail_base = full_groups * tile_width + lane * values_per_thread
                for v in T.unroll(values_per_thread):
                    h = tail_base + v
                    if h < hidden_size:
                        x_value = T.cast(x[row, h], "float32")
                        sum_x[0] += x_value
                        sum_sq[0] += x_value * x_value

            for offset in T.unroll(num_shuffles):
                sum_x[0] += T.shfl_xor(sum_x[0], 16 >> offset)
                sum_sq[0] += T.shfl_xor(sum_sq[0], 16 >> offset)

            if num_warps_per_row == 1:
                if warp_lane == 0:
                    warp_sum_x_shared[row_in_block, 0] = sum_x[0]
                    warp_sum_sq_shared[row_in_block, 0] = sum_sq[0]
            else:
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
                        sum_x[0] += T.shfl_xor(
                            sum_x[0], (num_warps_per_row // 2) >> offset
                        )
                        sum_sq[0] += T.shfl_xor(
                            sum_sq[0], (num_warps_per_row // 2) >> offset
                        )
                    if warp_lane == 0:
                        warp_sum_x_shared[row_in_block, 0] = sum_x[0]
                        warp_sum_sq_shared[row_in_block, 0] = sum_sq[0]
                T.sync_threads()

            T.sync_threads()
            sum_x[0] = warp_sum_x_shared[row_in_block, 0]
            sum_sq[0] = warp_sum_sq_shared[row_in_block, 0]

            mean = sum_x[0] / hidden_size
            var = sum_sq[0] / hidden_size - mean * mean
            inv_norm = T.rsqrt(var + eps)
            scale_inv = 1.0
            if is_quant:
                scale_inv = 1.0 / scale[0]

            for group in T.serial(0, full_groups):
                base = group * tile_width + lane * values_per_thread
                for v in T.vectorized(values_per_thread):
                    x_local[v] = x[row, base + v]
                    gamma_local[v] = gemma[base + v]
                    beta_local[v] = beta[base + v]
                for v in T.unroll(values_per_thread):
                    x_value = T.cast(x_local[v], "float32") - mean
                    gamma_value = T.cast(gamma_local[v], "float32")
                    beta_value = T.cast(beta_local[v], "float32")
                    if is_quant:
                        q = T.clamp(
                            (x_value * inv_norm * gamma_value + beta_value) * scale_inv,
                            -fp8_max,
                            fp8_max,
                        )
                        y_local[v] = T.cast(q, y_dtype)
                    else:
                        y_local[v] = T.cast(
                            x_value * inv_norm * gamma_value + beta_value, input_dtype
                        )
                for v in T.vectorized(values_per_thread):
                    y[row, base + v] = y_local[v]

            if tail_elems != 0:
                tail_base = full_groups * tile_width + lane * values_per_thread
                for v in T.unroll(values_per_thread):
                    h = tail_base + v
                    if h < hidden_size:
                        x_value = T.cast(x[row, h], "float32") - mean
                        gamma_value = T.cast(gemma[h], "float32")
                        beta_value = T.cast(beta[h], "float32")
                        if is_quant:
                            q = T.clamp(
                                (x_value * inv_norm * gamma_value + beta_value)
                                * scale_inv,
                                -fp8_max,
                                fp8_max,
                            )
                            y[row, h] = T.cast(q, y_dtype)
                        else:
                            y[row, h] = T.cast(
                                x_value * inv_norm * gamma_value + beta_value,
                                input_dtype,
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
        f"tilelang_layernorm_kernel_{_symbol_part(input_dtype)}"
        f"_r{rows_per_block}_l{threads_per_row}_v{values_per_thread}"
    )
    return tilelang_layernorm_kernel.with_attr("global_symbol", symbol)


@mate_api
def layernorm(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return _layernorm_impl(
        input=input,
        gemma=gemma,
        beta=beta,
        eps=eps,
        is_quant=False,
        scale=None,
        out=None,
        name="layernorm",
    )


@mate_api
def layernorm_quant(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return _layernorm_impl(
        input=input,
        gemma=gemma,
        beta=beta,
        eps=eps,
        is_quant=True,
        scale=scale,
        out=out,
        name="layernorm_quant",
    )


def _layernorm_impl(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
    is_quant: bool = False,
    scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    name: str = "layernorm",
) -> torch.Tensor:
    if input.dim() != 2:
        raise RuntimeError(f"{name} expects a rank-2 [B, H] tensor.")
    if input.dtype != torch.bfloat16:
        raise RuntimeError(f"{name} expects bfloat16 input.")
    if gemma.dim() != 1 or beta.dim() != 1:
        raise RuntimeError(f"{name} expects gemma and beta to be rank-1 tensors [H].")
    if input.size(1) != gemma.size(0) or input.size(1) != beta.size(0):
        raise RuntimeError(f"{name} hidden size mismatch.")
    if gemma.dtype != torch.float32 or beta.dtype != torch.float32:
        raise RuntimeError(f"{name} expects gemma and beta to be float32.")
    if input.device != gemma.device or input.device != beta.device:
        raise RuntimeError(f"{name} requires input, gemma and beta on the same device.")
    if input.stride(-1) != 1:
        raise RuntimeError(f"{name} requires input contiguous last dimension.")
    if gemma.stride(-1) != 1 or beta.stride(-1) != 1:
        raise RuntimeError(f"{name} requires contiguous gemma and beta.")

    batch_size, hidden_size = input.shape
    rows_per_block, threads_per_row, values_per_thread = _select_norm_schedule(
        hidden_size, batch_size
    )
    if rows_per_block < 1 or rows_per_block * threads_per_row > 1024:
        raise RuntimeError(
            f"{name} requires 1 <= rows_per_block * threads_per_row <= 1024."
        )
    if batch_size % rows_per_block != 0:
        raise RuntimeError(f"{name} requires batch size divisible by rows_per_block.")

    if is_quant:
        if scale is None:
            raise RuntimeError(f"{name} expects a scalar scale tensor.")
        if scale.numel() != 1:
            raise RuntimeError(f"{name} expects a scalar scale tensor.")
        if scale.dtype != torch.float32:
            raise RuntimeError(f"{name} expects scale dtype float32.")
        if input.device != scale.device:
            raise RuntimeError(f"{name} requires all tensors on the same device.")
        if out is None:
            out = torch.empty_strided(
                tuple(input.shape),
                tuple(input.stride()),
                device=input.device,
                dtype=torch.float8_e4m3fn,
            )
        if out.shape != input.shape:
            raise RuntimeError(f"{name} output shape must match input shape.")
        if out.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise RuntimeError(
                f"{name} output dtype must be torch.float8_e4m3fn or torch.float8_e5m2."
            )
        if out.device != input.device:
            raise RuntimeError(f"{name} output device must match input device.")
        if out.stride(-1) != 1:
            raise RuntimeError(f"{name} requires output contiguous last dimension.")
    else:
        if scale is None:
            scale = torch.ones(1, device=input.device, dtype=torch.float32)
        if out is None:
            out = torch.empty_strided(
                tuple(input.shape),
                tuple(input.stride()),
                device=input.device,
                dtype=input.dtype,
            )
        if out.shape != input.shape:
            raise RuntimeError(f"{name} output shape must match input shape.")
        if out.dtype != input.dtype:
            raise RuntimeError(f"{name} output dtype must match input dtype.")
        if out.device != input.device:
            raise RuntimeError(f"{name} output device must match input device.")
        if out.stride(-1) != 1:
            raise RuntimeError(f"{name} requires output contiguous last dimension.")

    kernel = tilelang_layernorm(
        input_dtype=input.dtype,
        is_quant=is_quant,
        rows_per_block=rows_per_block,
        threads_per_row=threads_per_row,
        values_per_thread=values_per_thread,
        output_dtype=_tl_fp8_dtype(out.dtype) if is_quant else None,
        output_torch_dtype=out.dtype if is_quant else None,
        fp8_max=_fp8_torch_max(out.dtype) if is_quant else None,
    )
    kernel(input, gemma, beta, out, scale, float(eps))
    return out


if __name__ == "__main__":

    def _torch_reference(
        input: torch.Tensor,
        gemma: torch.Tensor,
        beta: torch.Tensor,
        scale: torch.Tensor | None,
        eps: float,
        is_quant: bool,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        x_f32 = input.float()
        mean = x_f32.mean(dim=-1, keepdim=True)
        var = x_f32.square().mean(dim=-1, keepdim=True) - mean.square()
        ref = (x_f32 - mean) * torch.rsqrt(var + eps) * gemma.float() + beta.float()
        if is_quant:
            if scale is None or out_dtype is None:
                raise RuntimeError(
                    "layernorm quant reference requires scale and out_dtype."
                )
            fp8_max = _fp8_torch_max(out_dtype)
            return (ref / scale.float()).clamp(-fp8_max, fp8_max).to(out_dtype)
        return ref.to(input.dtype)

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is not available.")

    device = torch.device("musa")
    torch.manual_seed(0)

    eps = 1e-6
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    gemma = torch.randn(64, device=device, dtype=torch.float32)
    beta = torch.randn(64, device=device, dtype=torch.float32)
    out = layernorm(x, gemma, beta, eps=eps)
    ref = _torch_reference(x, gemma, beta, None, eps, False)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    scale = torch.ones(1, device=device, dtype=torch.float32)
    q_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    q_out = layernorm_quant(x, gemma, beta, scale, eps=eps, out=q_out)
    q_ref = _torch_reference(x, gemma, beta, scale, eps, True, out_dtype=q_out.dtype)
    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=0.0, atol=0.25)

    print(f"layernorm smoke ok: shape={tuple(x.shape)}, dtype={x.dtype}")
