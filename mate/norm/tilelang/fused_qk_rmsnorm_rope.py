# mypy: ignore-errors

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T

from mate.api_logging import mate_api

from ._norm_common import _JIT_CONFIG, _make_row_sum_reduce_macro

__all__ = ["fused_qk_rmsnorm_rope"]


_ROPE_TABLE_CACHE_SIZE = 32
_FP8_MAX = 448.0


def _symbol_part(value) -> str:
    return (
        str(value)
        .replace("torch.", "")
        .replace(".", "p")
        .replace("-", "m")
        .replace(" ", "_")
    )


@tilelang.jit(**_JIT_CONFIG)
def tilelang_fused_qk_rmsnorm_rope(
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    num_frame_channels: int,
    num_height_channels: int,
    num_width_channels: int,
    ppf: int,
    pph: int,
    ppw: int,
    interleave: bool,
    is_qk_norm: bool,
    output_fp8: bool,
):
    if head_dim not in (64, 128, 256):
        raise RuntimeError(
            "tilelang_fused_qk_rmsnorm_rope expects head_dim in {64, 128, 256}."
        )
    if head_dim % 64 != 0:
        raise RuntimeError(
            "tilelang_fused_qk_rmsnorm_rope expects head_dim divisible by 64."
        )
    if num_frame_channels + num_height_channels + num_width_channels != head_dim:
        raise RuntimeError("RoPE channel counts must sum to head_dim.")
    if (
        num_frame_channels % 2 != 0
        or num_height_channels % 2 != 0
        or num_width_channels % 2 != 0
    ):
        raise RuntimeError("RoPE channel counts must be even.")

    batch_size = T.dynamic("batch_size")
    seq_len = T.dynamic("seq_len")
    hidden_qkv = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    qkv_stride_b = T.dynamic("qkv_stride_b", "int64")
    qkv_stride_s = T.dynamic("qkv_stride_s", "int64")
    q_stride_b = T.dynamic("q_stride_b", "int64")
    q_stride_s = T.dynamic("q_stride_s", "int64")
    q_stride_h = T.dynamic("q_stride_h", "int64")
    k_stride_b = T.dynamic("k_stride_b", "int64")
    k_stride_s = T.dynamic("k_stride_s", "int64")
    k_stride_h = T.dynamic("k_stride_h", "int64")
    v_stride_b = T.dynamic("v_stride_b", "int64")
    v_stride_s = T.dynamic("v_stride_s", "int64")
    v_stride_h = T.dynamic("v_stride_h", "int64")

    max_heads = max(num_heads_q, num_heads_k, num_heads_v)
    if max_heads > 32:
        raise RuntimeError("tilelang_fused_qk_rmsnorm_rope expects max heads <= 32.")

    threads_per_warp = 32
    reduce_warps_per_block = 1 << (max_heads - 1).bit_length()
    row_reduce_shuffles = reduce_warps_per_block.bit_length() - 1
    warps_per_block = reduce_warps_per_block if is_qk_norm else max_heads
    threads = warps_per_block * threads_per_warp
    elems_per_thread = head_dim // threads_per_warp
    pairs_per_thread = elems_per_thread // 2
    neox_pairs_per_thread = head_dim // (threads_per_warp * 2)
    height_slice_start = num_frame_channels
    width_slice_start = num_frame_channels + num_height_channels
    pphppw = pph * ppw
    row_sum_reduce = _make_row_sum_reduce_macro(
        reduce_warps_per_block, row_reduce_shuffles
    )
    output_dtype = T.float8_e4m3fn if output_fp8 else T.bfloat16

    qkv_type = T.StridedTensor(
        (batch_size, seq_len, hidden_qkv),
        (qkv_stride_b, qkv_stride_s, 1),
        T.bfloat16,
    )
    q_weight_type = T.StridedTensor((num_heads_q * head_dim,), (1,), T.bfloat16)
    k_weight_type = T.StridedTensor((num_heads_k * head_dim,), (1,), T.bfloat16)
    q_out_type = T.StridedTensor(
        (batch_size, seq_len, num_heads_q, head_dim),
        (q_stride_b, q_stride_s, q_stride_h, 1),
        output_dtype,
    )
    k_out_type = T.StridedTensor(
        (batch_size, seq_len, num_heads_k, head_dim),
        (k_stride_b, k_stride_s, k_stride_h, 1),
        output_dtype,
    )
    v_out_type = T.StridedTensor(
        (batch_size, seq_len, num_heads_v, head_dim),
        (v_stride_b, v_stride_s, v_stride_h, 1),
        output_dtype,
    )
    frame_table_type = T.Tensor((ppf, num_frame_channels // 2, 2), T.float32)
    height_table_type = T.Tensor((pph, num_height_channels // 2, 2), T.float32)
    width_table_type = T.Tensor((ppw, num_width_channels // 2, 2), T.float32)

    @T.macro
    def process_qk(
        qkv,
        weight,
        out,
        b,
        s,
        head_idx,
        lane,
        warp_lane,
        input_segment_start: int,
        num_heads_this: int,
        eps,
        output_quant_scale,
        frame_cos_sin,
        height_cos_sin,
        width_cos_sin,
        rope_cos_sin_shared,
        warp_sum_sq_shared,
    ):
        valid_head = head_idx < num_heads_this
        sum_sq = T.alloc_local((1,), T.float32)
        inv_rms = T.alloc_local((1,), T.float32)
        qk_local_elems = elems_per_thread if interleave else neox_pairs_per_thread * 2
        elems = T.alloc_local((qk_local_elems,), T.float32)
        rotated = T.alloc_local((qk_local_elems,), T.bfloat16)
        sum_sq[0] = 0.0
        inv_rms[0] = 1.0

        dim_base = lane * elems_per_thread
        neox_pair_base = lane * neox_pairs_per_thread
        if valid_head:
            if interleave:
                for v in T.vectorized(elems_per_thread):
                    dim = dim_base + v
                    elems[v] = qkv[
                        b, s, input_segment_start + head_idx * head_dim + dim
                    ]
                if is_qk_norm:
                    for v in T.unroll(elems_per_thread):
                        sum_sq[0] += elems[v] * elems[v]
            else:
                for v in T.vectorized(neox_pairs_per_thread):
                    dim = neox_pair_base + v
                    elems[v] = qkv[
                        b, s, input_segment_start + head_idx * head_dim + dim
                    ]
                for v in T.vectorized(neox_pairs_per_thread):
                    dim = neox_pair_base + v
                    elems[v + neox_pairs_per_thread] = qkv[
                        b,
                        s,
                        input_segment_start + head_idx * head_dim + dim + head_dim // 2,
                    ]
                if is_qk_norm:
                    for v in T.unroll(neox_pairs_per_thread):
                        sum_sq[0] += (
                            elems[v] * elems[v]
                            + elems[v + neox_pairs_per_thread]
                            * elems[v + neox_pairs_per_thread]
                        )

        if is_qk_norm:
            row_sum_reduce(sum_sq, warp_sum_sq_shared, 0, warp_lane, head_idx)
            sum_sq[0] = warp_sum_sq_shared[0, 0]
            inv_rms[0] = T.rsqrt(
                sum_sq[0] / T.cast(num_heads_this * head_dim, T.float32) + eps
            )
            if valid_head:
                if interleave:
                    for v in T.unroll(elems_per_thread):
                        dim = dim_base + v
                        elems[v] = (
                            elems[v]
                            * inv_rms[0]
                            * T.cast(weight[head_idx * head_dim + dim], T.float32)
                        )
                else:
                    for v in T.unroll(neox_pairs_per_thread):
                        dim = neox_pair_base + v
                        elems[v] = (
                            elems[v]
                            * inv_rms[0]
                            * T.cast(weight[head_idx * head_dim + dim], T.float32)
                        )
                        elems[v + neox_pairs_per_thread] = (
                            elems[v + neox_pairs_per_thread]
                            * inv_rms[0]
                            * T.cast(
                                weight[head_idx * head_dim + dim + head_dim // 2],
                                T.float32,
                            )
                        )

        token_idx_in_seq = s
        pos_t = token_idx_in_seq // pphppw
        pos_x = token_idx_in_seq - pos_t * pphppw
        pos_h = pos_x // ppw
        pos_w = pos_x - pos_h * ppw

        if valid_head:
            if interleave:
                for pair in T.unroll(pairs_per_thread):
                    dim = dim_base + pair * 2
                    cos_v = T.alloc_var(T.float32, init=0.0)
                    sin_v = T.alloc_var(T.float32, init=0.0)
                    rope_pair = dim // 2
                    if num_heads_this >= 4 and head_dim >= 128:
                        cos_v = rope_cos_sin_shared[rope_pair, 0]
                        sin_v = rope_cos_sin_shared[rope_pair, 1]
                    else:
                        table_dim = T.alloc_var(T.int32, init=0)
                        if dim >= width_slice_start:
                            table_dim = (dim - width_slice_start) // 2
                            cos_v = width_cos_sin[pos_w, table_dim, 0]
                            sin_v = width_cos_sin[pos_w, table_dim, 1]
                        elif dim >= height_slice_start:
                            table_dim = (dim - height_slice_start) // 2
                            cos_v = height_cos_sin[pos_h, table_dim, 0]
                            sin_v = height_cos_sin[pos_h, table_dim, 1]
                        else:
                            table_dim = dim // 2
                            cos_v = frame_cos_sin[pos_t, table_dim, 0]
                            sin_v = frame_cos_sin[pos_t, table_dim, 1]
                    x0 = elems[pair * 2]
                    x1 = elems[pair * 2 + 1]
                    y0 = T.alloc_var(T.float32, init=x0 * cos_v - x1 * sin_v)
                    y1 = T.alloc_var(T.float32, init=x1 * cos_v + x0 * sin_v)
                    if output_fp8:
                        out[b, s, head_idx, dim] = T.cast(
                            T.clamp(y0 / output_quant_scale, -_FP8_MAX, _FP8_MAX),
                            output_dtype,
                        )
                        out[b, s, head_idx, dim + 1] = T.cast(
                            T.clamp(y1 / output_quant_scale, -_FP8_MAX, _FP8_MAX),
                            output_dtype,
                        )
                    else:
                        rotated[pair * 2] = T.cast(y0, T.bfloat16)
                        rotated[pair * 2 + 1] = T.cast(y1, T.bfloat16)
                if not output_fp8:
                    if head_dim == 256:
                        packed = T.reinterpret(rotated[0:8], T.uint32x4)
                        T.stg128(out[b, s, head_idx, dim_base : dim_base + 8], packed)
                    else:
                        for v in T.unroll(elems_per_thread):
                            dim = dim_base + v
                            out[b, s, head_idx, dim] = rotated[v]
            else:
                for v in T.unroll(neox_pairs_per_thread):
                    dim = neox_pair_base + v
                    partner_dim = dim + head_dim // 2

                    freq_dim = dim * 2
                    rope_pair = dim
                    cos_v = T.alloc_var(T.float32, init=0.0)
                    sin_v = T.alloc_var(T.float32, init=0.0)
                    if num_heads_this >= 4 and head_dim >= 128:
                        cos_v = rope_cos_sin_shared[rope_pair, 0]
                        sin_v = rope_cos_sin_shared[rope_pair, 1]
                    else:
                        table_dim = T.alloc_var(T.int32, init=0)
                        if freq_dim >= width_slice_start:
                            table_dim = (freq_dim - width_slice_start) // 2
                            cos_v = width_cos_sin[pos_w, table_dim, 0]
                            sin_v = width_cos_sin[pos_w, table_dim, 1]
                        elif freq_dim >= height_slice_start:
                            table_dim = (freq_dim - height_slice_start) // 2
                            cos_v = height_cos_sin[pos_h, table_dim, 0]
                            sin_v = height_cos_sin[pos_h, table_dim, 1]
                        else:
                            table_dim = dim
                            cos_v = frame_cos_sin[pos_t, table_dim, 0]
                            sin_v = frame_cos_sin[pos_t, table_dim, 1]
                    x0 = elems[v]
                    x1 = elems[v + neox_pairs_per_thread]
                    y0 = T.alloc_var(T.float32, init=x0 * cos_v - x1 * sin_v)
                    y1 = T.alloc_var(T.float32, init=x1 * cos_v + x0 * sin_v)
                    if output_fp8:
                        out[b, s, head_idx, dim] = T.cast(
                            T.clamp(y0 / output_quant_scale, -_FP8_MAX, _FP8_MAX),
                            output_dtype,
                        )
                        out[b, s, head_idx, partner_dim] = T.cast(
                            T.clamp(y1 / output_quant_scale, -_FP8_MAX, _FP8_MAX),
                            output_dtype,
                        )
                    else:
                        out[b, s, head_idx, dim] = T.cast(y0, T.bfloat16)
                        out[b, s, head_idx, partner_dim] = T.cast(y1, T.bfloat16)

    @T.macro
    def load_rope_cos_sin_shared(
        frame_cos_sin,
        height_cos_sin,
        width_cos_sin,
        rope_cos_sin_shared,
        tid,
        pos_t,
        pos_h,
        pos_w,
    ):
        for load_iter in T.unroll(T.ceildiv(head_dim, threads)):
            load_idx = tid + load_iter * threads
            if load_idx < head_dim:
                rope_pair = load_idx // 2
                cs_idx = load_idx - rope_pair * 2
                dim = rope_pair * 2
                table_dim = T.alloc_var(T.int32, init=0)
                if dim >= width_slice_start:
                    table_dim = (dim - width_slice_start) // 2
                    rope_cos_sin_shared[rope_pair, cs_idx] = width_cos_sin[
                        pos_w, table_dim, cs_idx
                    ]
                elif dim >= height_slice_start:
                    table_dim = (dim - height_slice_start) // 2
                    rope_cos_sin_shared[rope_pair, cs_idx] = height_cos_sin[
                        pos_h, table_dim, cs_idx
                    ]
                else:
                    table_dim = dim // 2
                    rope_cos_sin_shared[rope_pair, cs_idx] = frame_cos_sin[
                        pos_t, table_dim, cs_idx
                    ]
        T.sync_threads()

    @T.macro
    def process_v(qkv, out, b, s, head_idx, lane, v_quant_scale):
        if head_idx < num_heads_v:
            dim_base = lane * elems_per_thread
            input_segment_start = (num_heads_q + num_heads_k) * head_dim
            for v in T.unroll(elems_per_thread):
                dim = dim_base + v
                value = T.cast(
                    qkv[b, s, input_segment_start + head_idx * head_dim + dim],
                    T.float32,
                )
                if output_fp8:
                    out[b, s, head_idx, dim] = T.cast(
                        T.clamp(value / v_quant_scale, -_FP8_MAX, _FP8_MAX),
                        output_dtype,
                    )
                else:
                    out[b, s, head_idx, dim] = T.cast(value, T.bfloat16)

    @T.prim_func
    def tilelang_fused_qk_rmsnorm_rope_kernel(
        qkv: qkv_type,
        q_weight: q_weight_type,
        k_weight: k_weight_type,
        q_out: q_out_type,
        k_out: k_out_type,
        v_out: v_out_type,
        frame_cos_sin: frame_table_type,
        height_cos_sin: height_table_type,
        width_cos_sin: width_table_type,
        eps: T.float32,
        output_quant_scale: T.float32,
        v_quant_scale: T.float32,
    ):
        with T.Kernel(seq_len, batch_size, 3, threads=threads) as (s, b, kind):
            T.assume(qkv_stride_s % elems_per_thread == 0)
            T.assume(qkv_stride_b % elems_per_thread == 0)
            T.assume(q_stride_s % elems_per_thread == 0)
            T.assume(q_stride_b % elems_per_thread == 0)
            T.assume(k_stride_s % elems_per_thread == 0)
            T.assume(k_stride_b % elems_per_thread == 0)
            T.assume(v_stride_s % elems_per_thread == 0)
            T.assume(v_stride_b % elems_per_thread == 0)

            tid = T.get_thread_binding()
            head_idx = tid // threads_per_warp
            lane = tid - head_idx * threads_per_warp
            warp_lane = lane
            warp_sum_sq_shared = T.alloc_shared((1, reduce_warps_per_block), T.float32)
            rope_cos_sin_shared = T.alloc_shared((head_dim // 2, 2), T.float32)
            token_idx_in_seq = s
            pos_t = token_idx_in_seq // pphppw
            pos_x = token_idx_in_seq - pos_t * pphppw
            pos_h = pos_x // ppw
            pos_w = pos_x - pos_h * ppw

            if kind == 0:
                if num_heads_q >= 4 and head_dim >= 128:
                    load_rope_cos_sin_shared(
                        frame_cos_sin,
                        height_cos_sin,
                        width_cos_sin,
                        rope_cos_sin_shared,
                        tid,
                        pos_t,
                        pos_h,
                        pos_w,
                    )
                process_qk(
                    qkv,
                    q_weight,
                    q_out,
                    b,
                    s,
                    head_idx,
                    lane,
                    warp_lane,
                    0,
                    num_heads_q,
                    eps,
                    output_quant_scale,
                    frame_cos_sin,
                    height_cos_sin,
                    width_cos_sin,
                    rope_cos_sin_shared,
                    warp_sum_sq_shared,
                )
            elif kind == 1:
                if num_heads_k >= 4 and head_dim >= 128:
                    load_rope_cos_sin_shared(
                        frame_cos_sin,
                        height_cos_sin,
                        width_cos_sin,
                        rope_cos_sin_shared,
                        tid,
                        pos_t,
                        pos_h,
                        pos_w,
                    )
                process_qk(
                    qkv,
                    k_weight,
                    k_out,
                    b,
                    s,
                    head_idx,
                    lane,
                    warp_lane,
                    num_heads_q * head_dim,
                    num_heads_k,
                    eps,
                    output_quant_scale,
                    frame_cos_sin,
                    height_cos_sin,
                    width_cos_sin,
                    rope_cos_sin_shared,
                    warp_sum_sq_shared,
                )
            else:
                process_v(qkv, v_out, b, s, head_idx, lane, v_quant_scale)

    symbol = (
        "tilelang_fused_qk_rmsnorm_rope"
        f"_{'fp8' if output_fp8 else 'bf16'}"
        f"_nq{num_heads_q}_nk{num_heads_k}_nv{num_heads_v}_d{head_dim}"
        f"_cf{num_frame_channels}_ch{num_height_channels}_cw{num_width_channels}"
        f"_pp{ppf}x{pph}x{ppw}_i{int(interleave)}_norm{int(is_qk_norm)}"
    )
    return tilelang_fused_qk_rmsnorm_rope_kernel.with_attr("global_symbol", symbol)


def _check_contiguous_last_dim(name: str, tensor: torch.Tensor) -> None:
    if tensor.stride(-1) != 1:
        raise RuntimeError(f"{name} requires contiguous last dimension.")


def _as_qkv_3d(qkv: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, bool]:
    if qkv.dim() == 3:
        if qkv.size(1) != seq_len:
            raise RuntimeError(
                f"qkv seq_len ({qkv.size(1)}) != ppf*pph*ppw ({seq_len})."
            )
        return qkv, True
    if qkv.dim() != 2:
        raise RuntimeError("fused_qk_rmsnorm_rope expects qkv rank 2 or 3.")
    if qkv.size(0) % seq_len != 0:
        raise RuntimeError(
            f"qkv num_tokens ({qkv.size(0)}) must be divisible by seq_len ({seq_len})."
        )
    batch_size = qkv.size(0) // seq_len
    qkv_3d = qkv.as_strided(
        (batch_size, seq_len, qkv.size(1)),
        (seq_len * qkv.stride(0), qkv.stride(0), qkv.stride(1)),
    )
    return qkv_3d, False


def _output_4d_view(out: torch.Tensor, seq_len: int, is_input_3d: bool) -> torch.Tensor:
    if is_input_3d:
        return out
    batch_size = out.size(0) // seq_len
    return out.as_strided(
        (batch_size, seq_len, out.size(1), out.size(2)),
        (seq_len * out.stride(0), out.stride(0), out.stride(1), out.stride(2)),
    )


def _make_axis_rope_table(
    num_positions: int,
    num_channels: int,
    *,
    device: torch.device,
    base: float,
    factor: float,
    low: float,
    high: float,
) -> torch.Tensor:
    half_channels = num_channels // 2
    dim = torch.arange(half_channels, device=device, dtype=torch.float32)
    freq = torch.pow(
        torch.tensor(float(base), device=device, dtype=torch.float32),
        -2.0 * dim / float(num_channels),
    )
    if factor != 1.0:
        high_adj = high + (0.001 if abs(low - high) <= 1.0e-6 else 0.0)
        ramp = torch.clamp(
            (dim - float(low)) / (float(high_adj) - float(low)), 0.0, 1.0
        )
        inv_freq_extrapolation_factor = 1.0 - ramp
        freq = (freq / float(factor)) * (
            1.0 - inv_freq_extrapolation_factor
        ) + freq * inv_freq_extrapolation_factor

    pos = torch.arange(num_positions, device=device, dtype=torch.float32).reshape(
        num_positions, 1
    )
    theta = pos * freq.reshape(1, half_channels)
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1).contiguous()


@lru_cache(maxsize=_ROPE_TABLE_CACHE_SIZE)
def _make_rope_tables_cached(
    device: torch.device,
    ppf: int,
    pph: int,
    ppw: int,
    num_frame_channels: int,
    num_height_channels: int,
    num_width_channels: int,
    base: float,
    factor: float,
    low: float,
    high: float,
    attention_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_cos_sin = _make_axis_rope_table(
        ppf,
        num_frame_channels,
        device=device,
        base=base,
        factor=factor,
        low=low,
        high=high,
    )
    height_cos_sin = _make_axis_rope_table(
        pph,
        num_height_channels,
        device=device,
        base=base,
        factor=factor,
        low=low,
        high=high,
    )
    width_cos_sin = _make_axis_rope_table(
        ppw,
        num_width_channels,
        device=device,
        base=base,
        factor=factor,
        low=low,
        high=high,
    )
    if attention_factor != 1.0:
        frame_cos_sin = frame_cos_sin * float(attention_factor)
        height_cos_sin = height_cos_sin * float(attention_factor)
        width_cos_sin = width_cos_sin * float(attention_factor)
    return frame_cos_sin, height_cos_sin, width_cos_sin


def _make_rope_tables(
    *,
    device: torch.device,
    ppf: int,
    pph: int,
    ppw: int,
    num_frame_channels: int,
    num_height_channels: int,
    num_width_channels: int,
    base: float,
    factor: float,
    low: float,
    high: float,
    attention_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = device.index
    if device.type == "musa" and device_index is None:
        device_index = torch.musa.current_device()
    resolved_device = (
        torch.device(device.type)
        if device_index is None
        else torch.device(device.type, device_index)
    )
    return _make_rope_tables_cached(
        resolved_device,
        ppf,
        pph,
        ppw,
        num_frame_channels,
        num_height_channels,
        num_width_channels,
        float(base),
        float(factor),
        float(low),
        float(high),
        float(attention_factor),
    )


@mate_api
def fused_qk_rmsnorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    ppf: int,
    pph: int,
    ppw: int,
    num_frame_channels: int,
    num_height_channels: int,
    num_width_channels: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float = 1e-6,
    base: float = 10000.0,
    interleave: bool = True,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    is_qk_norm: bool = True,
    output_fp8: bool = False,
    output_quant_scale: float = 1.0,
    v_quant_scale: float = 1.0,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if qkv.dtype != torch.bfloat16:
        raise RuntimeError("qkv must be bfloat16.")
    if q_weight.dtype != torch.bfloat16 or k_weight.dtype != torch.bfloat16:
        raise RuntimeError("q_weight and k_weight must be bfloat16.")
    if qkv.device != q_weight.device or qkv.device != k_weight.device:
        raise RuntimeError("qkv, q_weight, and k_weight must be on the same device.")
    _check_contiguous_last_dim("qkv", qkv)
    _check_contiguous_last_dim("q_weight", q_weight)
    _check_contiguous_last_dim("k_weight", k_weight)

    if head_dim not in (64, 128, 256):
        raise RuntimeError(f"head_dim must be 64, 128, or 256, got {head_dim}.")
    max_heads = max(num_heads_q, num_heads_k, num_heads_v)
    if max_heads > 32:
        raise RuntimeError(
            f"max(num_heads_q, num_heads_k, num_heads_v) must be <= 32, got {max_heads}."
        )
    if num_frame_channels + num_height_channels + num_width_channels != head_dim:
        raise RuntimeError(
            "num_frame_channels + num_height_channels + num_width_channels must equal head_dim."
        )
    if (
        num_frame_channels % 2 != 0
        or num_height_channels % 2 != 0
        or num_width_channels % 2 != 0
    ):
        raise RuntimeError(
            "num_frame_channels, num_height_channels, and num_width_channels must be even."
        )
    if ppf <= 0 or pph <= 0 or ppw <= 0:
        raise RuntimeError("ppf, pph, and ppw must be positive.")
    seq_len = ppf * pph * ppw
    if factor == 1.0 and attention_factor != 1.0:
        raise RuntimeError("attention_factor must be 1.0 when factor is 1.0.")
    if output_fp8 and (output_quant_scale == 0.0 or v_quant_scale == 0.0):
        raise RuntimeError(
            "output_quant_scale and v_quant_scale must be non-zero for FP8 output."
        )

    out_dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
    hidden_qkv = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    if qkv.size(-1) != hidden_qkv:
        raise RuntimeError(f"qkv hidden size {qkv.size(-1)} != expected {hidden_qkv}.")
    if q_weight.numel() != num_heads_q * head_dim:
        raise RuntimeError("q_weight size must equal num_heads_q * head_dim.")
    if k_weight.numel() != num_heads_k * head_dim:
        raise RuntimeError("k_weight size must equal num_heads_k * head_dim.")

    qkv_3d, input_is_3d = _as_qkv_3d(qkv, seq_len)
    batch_size = qkv_3d.size(0)
    out_shape_q = (
        (batch_size, seq_len, num_heads_q, head_dim)
        if input_is_3d
        else (batch_size * seq_len, num_heads_q, head_dim)
    )
    out_shape_k = (
        (batch_size, seq_len, num_heads_k, head_dim)
        if input_is_3d
        else (batch_size * seq_len, num_heads_k, head_dim)
    )
    out_shape_v = (
        (batch_size, seq_len, num_heads_v, head_dim)
        if input_is_3d
        else (batch_size * seq_len, num_heads_v, head_dim)
    )

    if q_out is None:
        q_out = torch.empty(out_shape_q, device=qkv.device, dtype=out_dtype)
    if k_out is None:
        k_out = torch.empty(out_shape_k, device=qkv.device, dtype=out_dtype)
    if v_out is None:
        v_out = torch.empty(out_shape_v, device=qkv.device, dtype=out_dtype)

    for name, out, shape in (
        ("q_out", q_out, out_shape_q),
        ("k_out", k_out, out_shape_k),
        ("v_out", v_out, out_shape_v),
    ):
        if tuple(out.shape) != shape:
            raise RuntimeError(f"{name} shape {tuple(out.shape)} != expected {shape}.")
        if out.dtype != out_dtype:
            raise RuntimeError(f"{name} dtype {out.dtype} != expected {out_dtype}.")
        if out.device != qkv.device:
            raise RuntimeError(f"{name} must be on the qkv device.")
        if not out.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous.")

    q_out_4d = _output_4d_view(q_out, seq_len, input_is_3d)
    k_out_4d = _output_4d_view(k_out, seq_len, input_is_3d)
    v_out_4d = _output_4d_view(v_out, seq_len, input_is_3d)
    frame_cos_sin, height_cos_sin, width_cos_sin = _make_rope_tables(
        device=qkv.device,
        ppf=ppf,
        pph=pph,
        ppw=ppw,
        num_frame_channels=num_frame_channels,
        num_height_channels=num_height_channels,
        num_width_channels=num_width_channels,
        base=base,
        factor=factor,
        low=low,
        high=high,
        attention_factor=attention_factor,
    )

    kernel = tilelang_fused_qk_rmsnorm_rope(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        num_frame_channels=num_frame_channels,
        num_height_channels=num_height_channels,
        num_width_channels=num_width_channels,
        ppf=ppf,
        pph=pph,
        ppw=ppw,
        interleave=interleave,
        is_qk_norm=is_qk_norm,
        output_fp8=output_fp8,
    )
    kernel(
        qkv_3d,
        q_weight,
        k_weight,
        q_out_4d,
        k_out_4d,
        v_out_4d,
        frame_cos_sin,
        height_cos_sin,
        width_cos_sin,
        float(eps),
        float(output_quant_scale),
        float(v_quant_scale),
    )
    return q_out, k_out, v_out


if __name__ == "__main__":

    def _adjusted_freq(
        half_dim_val: int,
        dim_size: int,
        base: float,
        factor: float,
        low: float,
        high: float,
    ) -> float:
        freq = base ** (-2.0 * half_dim_val / dim_size)
        if factor != 1.0:
            high_adj = high + (0.001 if abs(low - high) <= 1.0e-6 else 0.0)
            ramp = min(max((half_dim_val - low) / (high_adj - low), 0.0), 1.0)
            inv_freq_extrapolation_factor = 1.0 - ramp
            freq = (freq / factor) * (
                1.0 - inv_freq_extrapolation_factor
            ) + freq * inv_freq_extrapolation_factor
        return freq

    def _qkv_as_3d(qkv: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, bool]:
        if qkv.dim() == 3:
            return qkv, True
        batch_size = qkv.size(0) // seq_len
        return qkv.as_strided(
            (batch_size, seq_len, qkv.size(1)),
            (seq_len * qkv.stride(0), qkv.stride(0), qkv.stride(1)),
        ), False

    def _apply_rope_interleave(
        x: torch.Tensor,
        *,
        pph: int,
        ppw: int,
        num_frame_channels: int,
        num_height_channels: int,
        num_width_channels: int,
        base: float,
        factor: float,
        low: float,
        high: float,
        attention_factor: float,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = x.shape
        out = torch.empty_like(x)
        height_slice_start = num_frame_channels
        width_slice_start = num_frame_channels + num_height_channels
        pphppw = pph * ppw

        for b in range(batch_size):
            for s in range(seq_len):
                pos_t = s // pphppw
                pos_x = s % pphppw
                pos_h = pos_x // ppw
                pos_w = pos_x % ppw
                for h in range(num_heads):
                    for d in range(0, head_dim, 2):
                        if d >= width_slice_start:
                            pos_id = pos_w
                            half_dim_val = (d - width_slice_start) // 2
                            dim_size = num_width_channels
                        elif d >= height_slice_start:
                            pos_id = pos_h
                            half_dim_val = (d - height_slice_start) // 2
                            dim_size = num_height_channels
                        else:
                            pos_id = pos_t
                            half_dim_val = d // 2
                            dim_size = num_frame_channels

                        freq = _adjusted_freq(
                            half_dim_val, dim_size, base, factor, low, high
                        )
                        theta = pos_id * freq
                        theta_t = torch.tensor(
                            theta, device=x.device, dtype=torch.float32
                        )
                        cos_v = torch.cos(theta_t)
                        sin_v = torch.sin(theta_t)
                        x0 = x[b, s, h, d]
                        x1 = x[b, s, h, d + 1]
                        y0 = x0 * cos_v - x1 * sin_v
                        y1 = x1 * cos_v + x0 * sin_v
                        if factor != 1.0:
                            y0 = y0 * attention_factor
                            y1 = y1 * attention_factor
                        out[b, s, h, d] = y0
                        out[b, s, h, d + 1] = y1
        return out

    def _apply_rope_neox(
        x: torch.Tensor,
        *,
        pph: int,
        ppw: int,
        num_frame_channels: int,
        num_height_channels: int,
        num_width_channels: int,
        base: float,
        factor: float,
        low: float,
        high: float,
        attention_factor: float,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = x.shape
        out = torch.empty_like(x)
        height_slice_start = num_frame_channels
        width_slice_start = num_frame_channels + num_height_channels
        pphppw = pph * ppw
        half_head_dim = head_dim // 2

        for b in range(batch_size):
            for s in range(seq_len):
                pos_t = s // pphppw
                pos_x = s % pphppw
                pos_h = pos_x // ppw
                pos_w = pos_x % ppw
                for h in range(num_heads):
                    for d in range(head_dim):
                        partner_dim = d + half_head_dim
                        rotate_sign = -1.0
                        if d >= half_head_dim:
                            partner_dim = d - half_head_dim
                            rotate_sign = 1.0

                        freq_dim = (d * 2) % head_dim
                        if freq_dim >= width_slice_start:
                            pos_id = pos_w
                            half_dim_val = (freq_dim - width_slice_start) // 2
                            dim_size = num_width_channels
                        elif freq_dim >= height_slice_start:
                            pos_id = pos_h
                            half_dim_val = (freq_dim - height_slice_start) // 2
                            dim_size = num_height_channels
                        else:
                            pos_id = pos_t
                            half_dim_val = freq_dim // 2
                            dim_size = num_frame_channels

                        freq = _adjusted_freq(
                            half_dim_val, dim_size, base, factor, low, high
                        )
                        theta = pos_id * freq
                        theta_t = torch.tensor(
                            theta, device=x.device, dtype=torch.float32
                        )
                        y = x[b, s, h, d] * torch.cos(theta_t) + rotate_sign * x[
                            b, s, h, partner_dim
                        ] * torch.sin(theta_t)
                        if factor != 1.0:
                            y = y * attention_factor
                        out[b, s, h, d] = y
        return out

    def _apply_rope(
        x: torch.Tensor,
        *,
        interleave: bool,
        pph: int,
        ppw: int,
        num_frame_channels: int,
        num_height_channels: int,
        num_width_channels: int,
        base: float,
        factor: float,
        low: float,
        high: float,
        attention_factor: float,
    ) -> torch.Tensor:
        apply_fn = _apply_rope_interleave if interleave else _apply_rope_neox
        return apply_fn(
            x,
            pph=pph,
            ppw=ppw,
            num_frame_channels=num_frame_channels,
            num_height_channels=num_height_channels,
            num_width_channels=num_width_channels,
            base=base,
            factor=factor,
            low=low,
            high=high,
            attention_factor=attention_factor,
        )

    def _torch_reference(
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        *,
        ppf: int,
        pph: int,
        ppw: int,
        num_frame_channels: int,
        num_height_channels: int,
        num_width_channels: int,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
        eps: float,
        base: float,
        factor: float,
        low: float,
        high: float,
        attention_factor: float,
        interleave: bool,
        is_qk_norm: bool,
        output_fp8: bool,
        output_quant_scale: float,
        v_quant_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = ppf * pph * ppw
        qkv_3d, input_is_3d = _qkv_as_3d(qkv, seq_len)
        batch_size = qkv_3d.size(0)

        q = (
            qkv_3d[..., : num_heads_q * head_dim]
            .reshape(batch_size, seq_len, num_heads_q, head_dim)
            .float()
        )
        k_start = num_heads_q * head_dim
        k_end = (num_heads_q + num_heads_k) * head_dim
        k = (
            qkv_3d[..., k_start:k_end]
            .reshape(batch_size, seq_len, num_heads_k, head_dim)
            .float()
        )
        v = (
            qkv_3d[..., k_end:]
            .reshape(batch_size, seq_len, num_heads_v, head_dim)
            .clone()
        )

        if is_qk_norm:
            q = (
                q
                * torch.rsqrt(
                    q.square().sum(dim=(2, 3), keepdim=True) / (num_heads_q * head_dim)
                    + eps
                )
                * q_weight.reshape(1, 1, num_heads_q, head_dim).float()
            )
            k = (
                k
                * torch.rsqrt(
                    k.square().sum(dim=(2, 3), keepdim=True) / (num_heads_k * head_dim)
                    + eps
                )
                * k_weight.reshape(1, 1, num_heads_k, head_dim).float()
            )
            if not output_fp8:
                q = q.to(torch.bfloat16).float()
                k = k.to(torch.bfloat16).float()

        q = _apply_rope(
            q,
            interleave=interleave,
            pph=pph,
            ppw=ppw,
            num_frame_channels=num_frame_channels,
            num_height_channels=num_height_channels,
            num_width_channels=num_width_channels,
            base=base,
            factor=factor,
            low=low,
            high=high,
            attention_factor=attention_factor,
        )
        k = _apply_rope(
            k,
            interleave=interleave,
            pph=pph,
            ppw=ppw,
            num_frame_channels=num_frame_channels,
            num_height_channels=num_height_channels,
            num_width_channels=num_width_channels,
            base=base,
            factor=factor,
            low=low,
            high=high,
            attention_factor=attention_factor,
        )
        if output_fp8:
            q = (
                (q / output_quant_scale)
                .clamp(-_FP8_MAX, _FP8_MAX)
                .to(torch.float8_e4m3fn)
            )
            k = (
                (k / output_quant_scale)
                .clamp(-_FP8_MAX, _FP8_MAX)
                .to(torch.float8_e4m3fn)
            )
            v = (
                (v.float() / v_quant_scale)
                .clamp(-_FP8_MAX, _FP8_MAX)
                .to(torch.float8_e4m3fn)
            )
        else:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        if input_is_3d:
            return q, k, v
        return (
            q.reshape(batch_size * seq_len, num_heads_q, head_dim),
            k.reshape(batch_size * seq_len, num_heads_k, head_dim),
            v.reshape(batch_size * seq_len, num_heads_v, head_dim),
        )

    def _make_qkv(
        batch_size: int,
        seq_len: int,
        hidden_qkv: int,
        device: torch.device,
        noncontiguous: bool,
    ) -> torch.Tensor:
        if noncontiguous:
            return torch.randn(
                batch_size,
                seq_len * 2,
                hidden_qkv,
                device=device,
                dtype=torch.bfloat16,
            )[:, ::2, :]
        return torch.randn(
            batch_size, seq_len, hidden_qkv, device=device, dtype=torch.bfloat16
        )

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is not available.")
    device = torch.device("musa")
    torch.manual_seed(0)

    batch_size = 1
    ppf = 2
    pph = 2
    ppw = 2
    seq_len = ppf * pph * ppw
    num_heads_q = 2
    num_heads_k = 2
    num_heads_v = 2
    head_dim = 64
    num_frame_channels = 16
    num_height_channels = 16
    num_width_channels = 32
    hidden_qkv = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    qkv = _make_qkv(batch_size, seq_len, hidden_qkv, device, False)
    q_weight = torch.randn(num_heads_q * head_dim, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(num_heads_k * head_dim, device=device, dtype=torch.bfloat16)

    got = fused_qk_rmsnorm_rope(
        qkv,
        q_weight,
        k_weight,
        ppf=ppf,
        pph=pph,
        ppw=ppw,
        num_frame_channels=num_frame_channels,
        num_height_channels=num_height_channels,
        num_width_channels=num_width_channels,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
    )
    ref = _torch_reference(
        qkv,
        q_weight,
        k_weight,
        ppf=ppf,
        pph=pph,
        ppw=ppw,
        num_frame_channels=num_frame_channels,
        num_height_channels=num_height_channels,
        num_width_channels=num_width_channels,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=1e-6,
        base=10000.0,
        factor=1.0,
        low=0.0,
        high=0.0,
        attention_factor=1.0,
        interleave=True,
        is_qk_norm=True,
        output_fp8=False,
        output_quant_scale=1.0,
        v_quant_scale=1.0,
    )
    for actual, expected in zip(got, ref):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.0, atol=0.03125
        )

    print(
        "fused_qk_rmsnorm_rope ok: "
        f"B={batch_size}, S={seq_len}, H={hidden_qkv}, "
        f"Nq/Nk/Nv=({num_heads_q},{num_heads_k},{num_heads_v}), D={head_dim}"
    )
