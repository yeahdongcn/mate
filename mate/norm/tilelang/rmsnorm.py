# mypy: ignore-errors

import functools

import torch
import tilelang
import tilelang.language as T

from mate.api_logging import mate_api

from ._norm_common import (
    _JIT_CONFIG,
    _fp8_torch_max,
    _tl_fp8_dtype,
    _make_row_sum_reduce_macro,
    _make_sum_squares_macro,
)

__all__ = ["rmsnorm", "rmsnorm_quant"]


@functools.cache
def _select_norm_schedule(hidden_size: int, num_rows: int) -> tuple[int, int, int]:
    if hidden_size >= 8192:
        threads_per_row = 256
        rows_per_block = 2
    elif hidden_size >= 4096:
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
def tilelang_rmsnorm(
    input_dtype,
    weight_bias: float,
    is_quant: bool,
    rows_per_block: int,
    threads_per_row: int,
    values_per_thread: int,
    output_dtype=None,
    output_torch_dtype: torch.dtype | None = None,
    fp8_max: float | None = None,
):
    if input_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("tilelang_rmsnorm expects fp16 or bf16 input.")
    if is_quant and (
        output_dtype is None or output_torch_dtype is None or fp8_max is None
    ):
        raise RuntimeError(
            "tilelang_rmsnorm quant path requires output dtype and fp8 max."
        )

    batch_size = T.dynamic("batch_size")
    seq_len = T.dynamic("seq_len")
    hidden_size = T.dynamic("hidden_size")
    x_stride_b = T.dynamic("x_stride_b")
    x_stride_s = T.dynamic("x_stride_s")
    y_stride_b = T.dynamic("y_stride_b")
    y_stride_s = T.dynamic("y_stride_s")
    if threads_per_row not in (32, 64, 128, 256):
        raise RuntimeError(
            "tilelang_rmsnorm expects threads_per_row in {32, 64, 128, 256}."
        )

    num_warps_per_row = threads_per_row // 32
    row_reduce_shuffles = (
        num_warps_per_row.bit_length() - 1 if num_warps_per_row > 1 else 0
    )
    threads = rows_per_block * threads_per_row
    tile_width = threads_per_row * values_per_thread
    full_groups = hidden_size // tile_width
    tail_elems = hidden_size % tile_width
    sum_squares = _make_sum_squares_macro(
        full_groups,
        tile_width,
        values_per_thread,
        tail_elems,
        hidden_size,
        is_3d=True,
    )
    row_sum_reduce = _make_row_sum_reduce_macro(
        num_warps_per_row,
        row_reduce_shuffles,
    )

    x_type = T.StridedTensor(
        (batch_size, seq_len, hidden_size),
        (x_stride_b, x_stride_s, 1),
        input_dtype,
    )
    w_type = T.StridedTensor((hidden_size,), (1,), input_dtype)
    s_type = T.StridedTensor((1,), (1,), T.float32)
    if is_quant:
        y_dtype = output_dtype
        y_torch_dtype = output_torch_dtype
    else:
        y_dtype = input_dtype
        y_torch_dtype = input_dtype
    y_type = T.StridedTensor(
        (batch_size, seq_len, hidden_size),
        (y_stride_b, y_stride_s, 1),
        y_dtype,
    )

    @T.prim_func
    def tilelang_rmsnorm_kernel(
        x: x_type,
        weight: w_type,
        y: y_type,
        scale: s_type,
        eps: T.float32,
    ):
        num_rows = batch_size * seq_len
        num_blocks = num_rows // rows_per_block

        with T.Kernel(num_blocks, threads=threads) as (block_idx,):
            T.assume(x_stride_s % values_per_thread == 0)
            T.assume(x_stride_b % values_per_thread == 0)
            T.assume(y_stride_s % values_per_thread == 0)
            T.assume(y_stride_b % values_per_thread == 0)

            tid = T.get_thread_binding()
            lane = tid % threads_per_row
            row_in_block = tid // threads_per_row
            warp_lane = lane % 32
            warp_in_row = lane // 32
            row = block_idx * rows_per_block + row_in_block
            batch_idx = row // seq_len
            seq_idx = row - batch_idx * seq_len

            sum_sq = T.alloc_local([1], dtype="float32")
            x_local = T.alloc_local([values_per_thread], dtype=input_dtype)
            w_local = T.alloc_local([values_per_thread], dtype=input_dtype)
            y_local = T.alloc_local([values_per_thread], dtype=y_dtype)
            weight_shared = T.alloc_shared((hidden_size,), dtype=input_dtype)
            warp_sum_sq_shared = T.alloc_shared(
                (rows_per_block, num_warps_per_row), dtype="float32"
            )

            sum_sq[0] = 0.0
            T.copy(weight, weight_shared)
            T.sync_threads()

            sum_squares(x, x_local, sum_sq, row, batch_idx, seq_idx, lane)

            row_sum_reduce(
                sum_sq,
                warp_sum_sq_shared,
                row_in_block,
                warp_lane,
                warp_in_row,
            )

            T.sync_threads()
            sum_sq[0] = warp_sum_sq_shared[row_in_block, 0]
            inv_norm = T.rsqrt(sum_sq[0] / hidden_size + eps)
            scale_inv = 1.0
            if is_quant:
                scale_inv = 1.0 / scale[0]

            for group in T.serial(0, full_groups):
                base_out = group * tile_width + lane * values_per_thread
                for v in T.vectorized(values_per_thread):
                    x_local[v] = x[batch_idx, seq_idx, base_out + v]
                    w_local[v] = weight_shared[base_out + v]
                for v in T.unroll(values_per_thread):
                    x_value = T.cast(x_local[v], "float32")
                    w = T.cast(w_local[v], "float32") + weight_bias
                    if is_quant:
                        q = T.clamp(
                            x_value * inv_norm * w * scale_inv, -fp8_max, fp8_max
                        )
                        y_local[v] = T.cast(q, y_dtype)
                    else:
                        y_local[v] = T.cast(x_value * inv_norm * w, input_dtype)
                for v in T.vectorized(values_per_thread):
                    y[batch_idx, seq_idx, base_out + v] = y_local[v]

            if tail_elems != 0:
                tail_base_out = full_groups * tile_width + lane * values_per_thread
                for v in T.unroll(values_per_thread):
                    j_h = tail_base_out + v
                    if j_h < hidden_size:
                        x_value = T.cast(x[batch_idx, seq_idx, j_h], "float32")
                        w = T.cast(weight_shared[j_h], "float32") + weight_bias
                        if is_quant:
                            q = T.clamp(
                                x_value * inv_norm * w * scale_inv, -fp8_max, fp8_max
                            )
                            y[batch_idx, seq_idx, j_h] = T.cast(q, y_dtype)
                        else:
                            y[batch_idx, seq_idx, j_h] = T.cast(
                                x_value * inv_norm * w, input_dtype
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
        f"tilelang_rmsnorm_kernel_{_symbol_part(input_dtype)}"
        f"_out{_symbol_part(y_torch_dtype)}_h{hidden_size}_wb{_symbol_part(weight_bias)}"
        f"_q{int(is_quant)}"
        f"_r{rows_per_block}_l{threads_per_row}_v{values_per_thread}"
    )
    return tilelang_rmsnorm_kernel.with_attr("global_symbol", symbol)


def _rmsnorm_3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    gemma: bool = False,
    y: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    is_quant: bool = False,
) -> torch.Tensor:
    if x.dim() not in (2, 3):
        raise RuntimeError(
            "_rmsnorm_3d expects a rank-2 [B, H] or rank-3 [B, S, H] tensor."
        )
    if weight.dim() != 1:
        raise RuntimeError("_rmsnorm_3d expects a rank-1 weight tensor [H].")
    if x.size(-1) != weight.size(0):
        raise RuntimeError(
            f"_rmsnorm_3d hidden size mismatch: x.size(-1)={x.size(-1)}, "
            f"weight.size(0)={weight.size(0)}."
        )
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("_rmsnorm_3d expects fp16 or bf16 input.")
    if weight.dtype != x.dtype:
        raise RuntimeError("_rmsnorm_3d requires weight dtype to match input dtype.")
    if x.stride(-1) != 1:
        raise RuntimeError("_rmsnorm_3d requires a contiguous last dimension.")
    if weight.stride(-1) != 1:
        raise RuntimeError("_rmsnorm_3d requires contiguous weight.")

    out = y
    if is_quant:
        if scale is None:
            raise RuntimeError("_rmsnorm_3d quant path requires a scale tensor.")
        if scale.numel() != 1:
            raise RuntimeError("_rmsnorm_3d quant path expects a scalar scale tensor.")
        if scale.dtype != torch.float32:
            raise RuntimeError("_rmsnorm_3d quant path expects scale dtype float32.")
        if scale.device != x.device:
            raise RuntimeError(
                "_rmsnorm_3d quant path requires scale on the input device."
            )
        if out is None:
            out = torch.empty_strided(
                tuple(x.shape),
                tuple(x.stride()),
                device=x.device,
                dtype=torch.float8_e4m3fn,
            )
    else:
        if scale is None:
            scale = torch.ones(1, device=x.device, dtype=torch.float32)
        if out is None:
            out = torch.empty_strided(
                tuple(x.shape),
                tuple(x.stride()),
                device=x.device,
                dtype=x.dtype,
            )
    if out.shape != x.shape:
        raise RuntimeError("_rmsnorm_3d output shape must match input shape.")
    if is_quant:
        if out.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise RuntimeError(
                "_rmsnorm_3d quant output dtype must be torch.float8_e4m3fn or torch.float8_e5m2."
            )
    elif out.dtype != x.dtype:
        raise RuntimeError("_rmsnorm_3d output dtype must match input dtype.")
    if out.device != x.device:
        raise RuntimeError("_rmsnorm_3d output device must match input device.")
    if out.stride(-1) != 1:
        raise RuntimeError("_rmsnorm_3d requires output contiguous last dimension.")

    if x.dim() == 2:
        x_3d = x.unsqueeze(1)
        y_3d = out.unsqueeze(1)
    else:
        x_3d = x
        y_3d = out

    num_rows = x_3d.size(0) * x_3d.size(1)
    rows_per_block, threads_per_row, values_per_thread = _select_norm_schedule(
        x_3d.size(2), num_rows
    )

    kernel = tilelang_rmsnorm(
        input_dtype=x_3d.dtype,
        weight_bias=1.0 if gemma else 0.0,
        is_quant=is_quant,
        rows_per_block=rows_per_block,
        threads_per_row=threads_per_row,
        values_per_thread=values_per_thread,
        output_dtype=_tl_fp8_dtype(out.dtype) if is_quant else None,
        output_torch_dtype=out.dtype if is_quant else None,
        fp8_max=_fp8_torch_max(out.dtype) if is_quant else None,
    )
    kernel(x_3d, weight, y_3d, scale, float(eps))
    return out


@mate_api
def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    gemma: bool = False,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    return _rmsnorm_3d(
        x,
        weight,
        eps=eps,
        gemma=gemma,
        y=y,
        is_quant=False,
    )


@mate_api
def rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    gemma: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return _rmsnorm_3d(
        x,
        weight,
        eps=eps,
        gemma=gemma,
        y=out,
        scale=scale,
        is_quant=True,
    )


if __name__ == "__main__":

    def _torch_reference(
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        eps: float,
        gemma: bool,
        is_quant: bool,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        x_f32 = x.float()
        weight_f32 = weight.float() + (1.0 if gemma else 0.0)
        ref = (
            x_f32
            * torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
            * weight_f32
        )
        if is_quant:
            if scale is None:
                raise RuntimeError("quant reference requires scale.")
            fp8_max = _fp8_torch_max(out_dtype)
            return (ref / scale.float()).clamp(-fp8_max, fp8_max).to(out_dtype)
        return ref.to(x.dtype)

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is not available.")

    device = torch.device("musa")
    torch.manual_seed(0)

    eps = 1e-6
    x = torch.randn(2, 3, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    out = rmsnorm(x, weight, eps=eps, y=y)
    ref = _torch_reference(x, weight, None, eps, False, False, out.dtype)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    scale = torch.ones(1, device=device, dtype=torch.float32)
    q_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    q_out = rmsnorm_quant(x, weight, scale, eps=eps, out=q_out)
    q_ref = _torch_reference(x, weight, scale, eps, False, True, q_out.dtype)
    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=2e-2, atol=0.25)

    print(f"rmsnorm smoke ok: shape={tuple(x.shape)}, dtype={x.dtype}")
