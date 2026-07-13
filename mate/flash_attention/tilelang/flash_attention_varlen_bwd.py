# ruff: noqa
# type: ignore
import torch

from ._flash_attention_bwd_common import (
    _check_attention_strides,
    _contiguous_cosize_bytes,
    _jit_for_index_type_promotion,
    _needs_index_type_promotion,
    _tensor_cosize_bytes,
)
from ._flash_attention_bwd_post import (
    ceil_div,
    compute_delta_ws,
    pack_dq_from_accum_ws,
    reduce_kv_grads_ws,
    to_tilelang_dtype,
)
from ._flash_attention_bwd_split import flashattn_bwd_ws_split
from ._flash_attention_bwd_split_dkdv import flashattn_bwd_ws_split_dkdv
from ._flash_attention_bwd_split_dq import flashattn_bwd_ws_split_dq
from ._flash_attention_bwd_unsplit import flashattn_bwd_ws_unsplit
from ._flash_attention_bwd_unsplit_dkdv import flashattn_bwd_ws_unsplit_dkdv
from ._flash_attention_bwd_unsplit_dq import flashattn_bwd_ws_unsplit_dq


_THREADS = 640
_SPLIT_BLOCK_M = 64
_SPLIT_BLOCK_N = 64
_UNSPLIT_BLOCK_M = 64
_UNSPLIT_BLOCK_N = 128
_SPLIT_DKDV_BLOCK_M = 64
_SPLIT_DKDV_BLOCK_N = 128
_SPLIT_DQ_BLOCK_M = 128
_SPLIT_DQ_BLOCK_N = 64
_UNSPLIT_DKDV_BLOCK_M = 64
_UNSPLIT_DKDV_BLOCK_N = 128
_UNSPLIT_DQ_BLOCK_M = 128
_UNSPLIT_DQ_BLOCK_N = 64


def _select_bwd_plan(dim, deterministic, heads_q_eq_heads_kv=True):
    if dim == 256:
        return "split_separate" if deterministic else "split"
    if dim == 128:
        if deterministic or not heads_q_eq_heads_kv:
            return "unsplit_separate"
        return "unsplit"
    raise NotImplementedError(
        f"TileLang FlashAttention backward currently supports dim 128 or 256, got {dim}"
    )


def _unified_blocks(plan):
    if plan == "split":
        return _SPLIT_BLOCK_M, _SPLIT_BLOCK_N
    if plan == "unsplit":
        return _UNSPLIT_BLOCK_M, _UNSPLIT_BLOCK_N
    raise ValueError(f"{plan} is not a unified backward plan")


def _separate_blocks(plan):
    if plan == "split_separate":
        return (
            _SPLIT_DKDV_BLOCK_M,
            _SPLIT_DKDV_BLOCK_N,
            _SPLIT_DQ_BLOCK_M,
            _SPLIT_DQ_BLOCK_N,
        )
    if plan == "unsplit_separate":
        return (
            _UNSPLIT_DKDV_BLOCK_M,
            _UNSPLIT_DKDV_BLOCK_N,
            _UNSPLIT_DQ_BLOCK_M,
            _UNSPLIT_DQ_BLOCK_N,
        )
    raise ValueError(f"{plan} is not a split backward plan")


def _make_unified_bwd_kernel(
    plan,
    *,
    dim,
    is_causal,
    is_varlen,
    is_bhsd,
    heads_q_eq_heads_kv,
    smscale,
    dtype,
    enable_index_type_promotion=False,
):
    block_M, block_N = _unified_blocks(plan)
    jit = flashattn_bwd_ws_split if plan == "split" else flashattn_bwd_ws_unsplit
    jit = _jit_for_index_type_promotion(jit, enable_index_type_promotion)
    return jit(
        dim=dim,
        is_causal=is_causal,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        heads_q_eq_heads_kv=heads_q_eq_heads_kv,
        block_M=block_M,
        block_N=block_N,
        smscale=smscale,
        dtype=dtype,
    )


def _make_separate_bwd_kernel(
    plan,
    *,
    dim,
    is_causal,
    is_varlen,
    is_bhsd,
    heads_q_eq_heads_kv,
    smscale,
    dtype,
    enable_index_type_promotion=False,
):
    dkdv_block_M, dkdv_block_N, dq_block_M, dq_block_N = _separate_blocks(plan)
    if plan == "split_separate":
        dkdv_jit = flashattn_bwd_ws_split_dkdv
        dq_jit = flashattn_bwd_ws_split_dq
    else:
        dkdv_jit = flashattn_bwd_ws_unsplit_dkdv
        dq_jit = flashattn_bwd_ws_unsplit_dq

    dkdv_jit = _jit_for_index_type_promotion(dkdv_jit, enable_index_type_promotion)
    dq_jit = _jit_for_index_type_promotion(dq_jit, enable_index_type_promotion)
    dkdv_kernel = dkdv_jit(
        dim=dim,
        is_causal=is_causal,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        heads_q_eq_heads_kv=heads_q_eq_heads_kv,
        block_M=dkdv_block_M,
        block_N=dkdv_block_N,
        smscale=smscale,
        dtype=dtype,
    )
    dq_kernel = dq_jit(
        dim=dim,
        is_causal=is_causal,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        heads_q_eq_heads_kv=heads_q_eq_heads_kv,
        block_M=dq_block_M,
        block_N=dq_block_N,
        smscale=smscale,
        dtype=dtype,
    )

    def run(
        Q,
        K,
        V,
        dQ,
        dK,
        dV,
        dO,
        cu_seq_q,
        cu_seq_kv,
        Lse,
        Delta,
        max_seq_q,
        max_seq_kv,
    ):
        dkdv_kernel(
            Q,
            K,
            V,
            dK,
            dV,
            dO,
            cu_seq_q,
            cu_seq_kv,
            Lse,
            Delta,
            max_seq_kv,
        )
        dq_kernel(
            Q,
            K,
            V,
            dQ,
            dO,
            cu_seq_q,
            cu_seq_kv,
            Lse,
            Delta,
            max_seq_q,
        )

    return run


def flashattn_bwd_ws(
    dim,
    is_causal,
    is_varlen,
    is_bhsd=False,
    heads_q_eq_heads_kv=False,
    smscale=None,
    dtype="bfloat16",
    deterministic=False,
):
    plan = _select_bwd_plan(dim, deterministic, heads_q_eq_heads_kv)
    if plan.endswith("_separate"):
        return _make_separate_bwd_kernel(
            plan,
            dim=dim,
            is_causal=is_causal,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            heads_q_eq_heads_kv=heads_q_eq_heads_kv,
            smscale=smscale,
            dtype=dtype,
        )
    return _make_unified_bwd_kernel(
        plan,
        dim=dim,
        is_causal=is_causal,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        heads_q_eq_heads_kv=heads_q_eq_heads_kv,
        smscale=smscale,
        dtype=dtype,
    )


flashattn_bwd_ws_unsplit_d128 = flashattn_bwd_ws_unsplit


def flashattn_varlen_bwd_interface(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    is_causal=False,
    smscale=None,
    dtype=None,
    is_bhsd=False,
    deterministic=False,
):
    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_varlen and (cu_seqlens_q is None or cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")

    if is_varlen:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("varlen inputs must have shape [total_seq, heads, dim]")
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("max_seqlen_q and max_seqlen_k are required for varlen")
        q_flat, k_flat, v_flat = q, k, v
        out_flat, dout_flat = out, dout
        total_seq_q, heads_q, dim = q_flat.shape
        total_seq_kv, heads_kv, _ = k_flat.shape
        out_expected_shape = q_flat.shape
        lse_expected_shape = (heads_q, total_seq_q)
    else:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "non-varlen inputs must have shape [batch, seqlen, heads, dim]"
            )
        if softmax_lse.ndim != 3:
            raise ValueError(
                "non-varlen softmax_lse must have shape [batch, heads, seqlen]"
            )
        batch = q.shape[0]
        if softmax_lse.shape[0] != batch:
            raise ValueError("q and softmax_lse must have the same batch size")
        _, seqlen_q, heads_q, dim = q.shape
        batch_k, seqlen_k, heads_kv, dim_k = k.shape
        expected_v_shape = (batch, seqlen_k, heads_kv, dim)
        if batch_k != batch:
            raise ValueError("q and k must have the same batch size for non-varlen")
        if dim_k != dim:
            raise ValueError("q and k must have the same head dimension")
        if v.shape != expected_v_shape:
            raise ValueError(
                f"v must match k layout for non-varlen backward, expected {expected_v_shape}, "
                f"got {tuple(v.shape)}"
            )
        total_seq_q = batch * seqlen_q
        total_seq_kv = batch * seqlen_k
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k
        q_flat, k_flat, v_flat = q, k, v
        out_flat, dout_flat = out, dout
        out_expected_shape = q.shape
        lse_expected_shape = (batch, heads_q, seqlen_q)

    if is_varlen and v_flat.shape != (total_seq_kv, heads_kv, dim):
        raise ValueError(
            "v must match k shape for the TileLang varlen backward interface"
        )
    if out.shape != out_expected_shape or dout.shape != out_expected_shape:
        raise ValueError("out and dout must have the same shape as q")
    if is_varlen and (cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1D tensors")
    if softmax_lse.shape != lse_expected_shape:
        raise ValueError(
            f"softmax_lse must be shaped {lse_expected_shape}, got {tuple(softmax_lse.shape)}"
        )
    if not softmax_lse.is_contiguous():
        raise ValueError("softmax_lse must be contiguous")

    for name, tensor in (
        ("q", q_flat),
        ("k", k_flat),
        ("v", v_flat),
        ("out", out_flat),
        ("dout", dout_flat),
    ):
        _check_attention_strides(name, tensor)

    kernel_dtype = dtype if dtype is not None else to_tilelang_dtype(q.dtype)
    heads_q_eq_heads_kv = heads_q == heads_kv
    if heads_q % heads_kv != 0:
        raise ValueError(
            f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv})"
        )

    plan = _select_bwd_plan(dim, deterministic, heads_q_eq_heads_kv)
    use_separate = plan.endswith("_separate")

    batch = cu_seqlens_q.numel() - 1 if is_varlen else q.shape[0]
    if use_separate:
        _, _, block_M, block_N = _separate_blocks(plan)
    else:
        block_M, block_N = _unified_blocks(plan)
    max_seq_q_padded = ceil_div(max_seqlen_q, block_M) * block_M
    dq_shape = (total_seq_q, heads_q, dim) if is_varlen else q.shape

    kernel_arg_byte_spans = [
        _tensor_cosize_bytes(q_flat),
        _tensor_cosize_bytes(k_flat),
        _tensor_cosize_bytes(v_flat),
        _tensor_cosize_bytes(out_flat),
        _tensor_cosize_bytes(dout_flat),
        _tensor_cosize_bytes(softmax_lse),
        _contiguous_cosize_bytes((total_seq_q, heads_q), torch.float32),
    ]
    if use_separate:
        kernel_arg_byte_spans.append(_contiguous_cosize_bytes(dq_shape, q.dtype))
    else:
        kernel_arg_byte_spans.append(
            _contiguous_cosize_bytes(
                (batch, heads_q, max_seq_q_padded, dim), torch.float32
            )
        )
    if not heads_q_eq_heads_kv:
        kv_accum_shape = (
            (total_seq_kv, heads_q, dim)
            if is_varlen
            else (batch, seqlen_k, heads_q, dim)
        )
        kv_grad_shape = (total_seq_kv, heads_kv, dim) if is_varlen else k.shape
        kernel_arg_byte_spans.extend(
            (
                _contiguous_cosize_bytes(kv_accum_shape, torch.float32),
                _contiguous_cosize_bytes(kv_grad_shape, torch.float32),
            )
        )
    enable_index_type_promotion = _needs_index_type_promotion(*kernel_arg_byte_spans)

    delta = torch.empty((total_seq_q, heads_q), device=q.device, dtype=torch.float32)
    compute_delta = _jit_for_index_type_promotion(
        compute_delta_ws, enable_index_type_promotion
    )
    compute_delta(
        dim,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        dtype=kernel_dtype,
        block_M=block_M,
        threads=_THREADS,
        use_strided_tensors=True,
    )(out_flat, dout_flat, delta)

    dq = torch.empty(dq_shape, dtype=q.dtype, device=q.device)
    dQ_accum = None
    if not use_separate:
        dQ_accum = torch.zeros(
            (batch, heads_q, max_seq_q_padded, dim),
            dtype=torch.float32,
            device=q.device,
        )
    if heads_q_eq_heads_kv:
        dK_accum = torch.zeros_like(k)
        dV_accum = torch.zeros_like(v)
    else:
        kv_accum_shape = (
            (total_seq_kv, heads_q, dim)
            if is_varlen
            else (batch, seqlen_k, heads_q, dim)
        )
        dK_accum = torch.zeros(kv_accum_shape, dtype=torch.float32, device=q.device)
        dV_accum = torch.zeros(kv_accum_shape, dtype=torch.float32, device=q.device)

    if is_varlen:
        cu_seqlens_q_arg = cu_seqlens_q
        cu_seqlens_k_arg = cu_seqlens_k
    else:
        cu_seqlens_dummy = torch.empty((batch + 1,), device=q.device, dtype=torch.int32)
        cu_seqlens_q_arg = cu_seqlens_dummy
        cu_seqlens_k_arg = cu_seqlens_dummy

    if use_separate:
        kernel = _make_separate_bwd_kernel(
            plan,
            dim=dim,
            is_causal=is_causal,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            heads_q_eq_heads_kv=heads_q_eq_heads_kv,
            smscale=smscale,
            dtype=kernel_dtype,
            enable_index_type_promotion=enable_index_type_promotion,
        )
        kernel(
            q_flat,
            k_flat,
            v_flat,
            dq,
            dK_accum,
            dV_accum,
            dout_flat,
            cu_seqlens_q_arg,
            cu_seqlens_k_arg,
            softmax_lse,
            delta,
            max_seqlen_q,
            max_seqlen_k,
        )
    else:
        kernel = _make_unified_bwd_kernel(
            plan,
            dim=dim,
            is_causal=is_causal,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            heads_q_eq_heads_kv=heads_q_eq_heads_kv,
            smscale=smscale,
            dtype=kernel_dtype,
            enable_index_type_promotion=enable_index_type_promotion,
        )
        kernel(
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            dQ_accum,
            dK_accum,
            dV_accum,
            dout_flat,
            cu_seqlens_q_arg,
            cu_seqlens_k_arg,
            softmax_lse,
            delta,
            max_seqlen_k,
        )
        pack_dq = _jit_for_index_type_promotion(
            pack_dq_from_accum_ws, enable_index_type_promotion
        )
        pack_dq(
            dim,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            dtype=kernel_dtype,
            use_strided_tensors=True,
        )(dQ_accum, cu_seqlens_q_arg, dq)

    if heads_q_eq_heads_kv:
        dk, dv = dK_accum, dV_accum
    else:
        kv_shape = (total_seq_kv, heads_kv, dim) if is_varlen else k.shape
        dk = torch.empty(kv_shape, dtype=torch.float32, device=q.device)
        dv = torch.empty(kv_shape, dtype=torch.float32, device=q.device)
        reduce_kv_grads = _jit_for_index_type_promotion(
            reduce_kv_grads_ws, enable_index_type_promotion
        )
        reduce_kv_grads(
            heads_q // heads_kv,
            dim,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            use_strided_tensors=True,
        )(dK_accum, dV_accum, dk, dv)
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)
    return dq, dk, dv


__all__ = [
    "ceil_div",
    "compute_delta_ws",
    "flashattn_bwd_ws",
    "flashattn_bwd_ws_split",
    "flashattn_bwd_ws_split_dkdv",
    "flashattn_bwd_ws_split_dq",
    "flashattn_bwd_ws_unsplit",
    "flashattn_bwd_ws_unsplit_d128",
    "flashattn_bwd_ws_unsplit_dkdv",
    "flashattn_bwd_ws_unsplit_dq",
    "flashattn_varlen_bwd_interface",
    "pack_dq_from_accum_ws",
    "reduce_kv_grads_ws",
    "to_tilelang_dtype",
]
