# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_musa  # noqa: F401
from mate import flash_attn_varlen_func
from typing import Optional  # noqa: F401
from mate.testing import supported_musa_compute_capability


def ref_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    cu_kv_lens: torch.Tensor,
    max_query_len: int,
    max_kv_len: int,
    is_causal: bool = True,
    is_varlen: bool = True,
    scale: float = 1.0,
    upcast: bool = True,
    reorder_ops: bool = False,
):
    batch = len(cu_query_lens) - 1 if is_varlen else query.shape[0]
    dtype_og = query.dtype
    # head_size_qk = key_cache.shape[-1]

    outputs: list[torch.Tensor] = []
    lse_outputs: list[torch.Tensor] = []

    for i in range(batch):
        query_len = (
            cu_query_lens[i + 1] - cu_query_lens[i] if is_varlen else query.shape[1]
        )
        kv_len = (
            cu_kv_lens[i + 1] - cu_kv_lens[i] if is_varlen else value_cache.shape[1]
        )
        if is_varlen:
            q = query[cu_query_lens[i] : cu_query_lens[i + 1]]

            k = key_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
            v = value_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
        else:
            q = query[i]
            k = key_cache[i]
            v = value_cache[i]

        if upcast:
            q, k, v = q.float(), k.float(), v.float()

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        # raw attention score
        if reorder_ops:
            attn = torch.einsum("qhd,khd->hqk", q, k * scale)
        else:
            attn = torch.einsum("qhd,khd->hqk", q * scale, k)
        attn = attn.float()

        # mask
        if is_causal:
            empty_mask = torch.ones(query_len, kv_len, device=query.device)
            mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
            attn.masked_fill_(mask, float("-inf"))

        # LSE (head, query)
        lse = torch.logsumexp(attn, dim=-1)  # shape [H, Q]
        lse = torch.nan_to_num(lse)
        lse_outputs.append(lse)

        # softmax = exp(score - lse)
        attn = torch.exp(attn - lse.unsqueeze(-1)).to(v.dtype)

        # output
        out = torch.einsum("hqk,khd->qhd", attn, v)
        out = torch.nan_to_num(out)
        outputs.append(out.to(dtype=dtype_og))
    if is_varlen:
        return torch.cat(outputs, dim=0), torch.cat(lse_outputs, dim=1)
    else:
        return torch.stack(outputs, dim=0), torch.stack(lse_outputs, dim=0)


def clone_with_grad(t: torch.Tensor) -> torch.Tensor:
    return t.clone().detach().requires_grad_(True)


def strided_last_dim_tensor(
    shape: tuple[int, ...],
    *,
    device: str,
    dtype: torch.dtype,
    requires_grad: bool = False,
    pad: int = 8,
) -> torch.Tensor:
    base = torch.randn((*shape[:-1], shape[-1] + pad), device=device, dtype=dtype)
    tensor = base[..., : shape[-1]].detach()
    if requires_grad:
        tensor.requires_grad_(True)
    assert tensor.stride(-1) == 1
    assert not tensor.is_contiguous()
    return tensor


def copy_to_strided_last_dim(tensor: torch.Tensor, pad: int = 8) -> torch.Tensor:
    base = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] + pad),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    out = base[..., : tensor.shape[-1]]
    out.copy_(tensor)
    assert out.stride(-1) == 1
    assert not out.is_contiguous()
    return out


def assert_fa_error_within_pt(
    actual: torch.Tensor,
    ref: torch.Tensor,
    pt: torch.Tensor,
    name: str,
    *,
    multiplier: float = 2.0,
) -> None:
    actual_err = (actual - ref).abs().max().item()
    pt_err = (pt - ref).abs().max().item()
    atol_floor = 2 * (ref + 0.3 - 0.3 - ref).abs().max().item()
    assert actual_err <= multiplier * pt_err + atol_floor, (
        f"{name} max error {actual_err} exceeds "
        f"{multiplier}x PyTorch low-precision baseline {pt_err} "
        f"+ numerical floor {atol_floor}"
    )


def fa_error_limit(
    ref: torch.Tensor,
    pt: torch.Tensor,
    *,
    multiplier: float = 2.0,
) -> float:
    pt_err = (pt - ref).abs().max().item()
    atol_floor = 2 * (ref + 0.3 - 0.3 - ref).abs().max().item()
    return multiplier * pt_err + atol_floor


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [
            (512, 512),
            (511, 511),
            (384, 384),
            (257, 257),
            (128, 128),
            (65, 65),
            (320, 320),
            (192, 192),
        ]
    ],
)
@pytest.mark.parametrize("num_heads", [(6, 1), (6, 6)])
@pytest.mark.parametrize("head_size", [(128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("is_varlen", [True])
def test_varlen_fast_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: tuple[int, int],
    dtype: torch.dtype,
    is_causal: bool,
    is_varlen: bool,
) -> None:
    torch.set_default_device("musa")
    torch.manual_seed(42)
    torch.musa.manual_seed(42)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    head_size_qk = head_size[0]
    head_size_v = head_size[1]
    scale = head_size_qk**-0.5

    if is_varlen:
        query = torch.randn(
            (sum(query_lens), num_query_heads, head_size_qk),
            dtype=dtype,
            requires_grad=True,
        )
        key_cache = torch.randn(
            (sum(kv_lens), num_kv_heads, head_size_qk), dtype=dtype, requires_grad=True
        )
        value_cache = torch.randn(
            (sum(kv_lens), num_kv_heads, head_size_v), dtype=dtype, requires_grad=True
        )
        query.retain_grad()
        key_cache.retain_grad()
        value_cache.retain_grad()
    else:
        return

    q_fa, k_fa, v_fa = map(clone_with_grad, (query, key_cache, value_cache))
    q_t, k_t, v_t = map(clone_with_grad, (query, key_cache, value_cache))
    q_pt, k_pt, v_pt = map(clone_with_grad, (query, key_cache, value_cache))

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    cu_kv_lens = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        causal=is_causal,
    )

    ref_output, _ = ref_attn(
        query=q_t,
        key_cache=k_t,
        value_cache=v_t,
        cu_query_lens=cu_query_lens,
        cu_kv_lens=cu_kv_lens,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
        upcast=True,
    )
    pt_output, _ = ref_attn(
        query=q_pt,
        key_cache=k_pt,
        value_cache=v_pt,
        cu_query_lens=cu_query_lens,
        cu_kv_lens=cu_kv_lens,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
        upcast=False,
        reorder_ops=True,
    )
    grad_out = torch.randn_like(out_fa)

    grad_fa = clone_with_grad(grad_out)
    grad_t = clone_with_grad(grad_out)
    grad_pt = clone_with_grad(grad_out)

    out_fa.backward(gradient=grad_fa, retain_graph=True)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    ref_output.backward(gradient=grad_t, retain_graph=True)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    pt_output.backward(gradient=grad_pt, retain_graph=True)
    dq_pt, dk_pt, dv_pt = q_pt.grad, k_pt.grad, v_pt.grad

    assert_fa_error_within_pt(out_fa, ref_output, pt_output, "out")
    assert_fa_error_within_pt(dk_fa, dk_t, dk_pt, "dk")
    assert_fa_error_within_pt(dv_fa, dv_t, dv_pt, "dv")
    assert_fa_error_within_pt(dq_fa, dq_t, dq_pt, "dq")


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_varlen_fast_attn_gqa_dkv_stability(dtype: torch.dtype) -> None:
    pytest.importorskip("tilelang")
    from mate.flash_attention.tilelang.flash_attention_varlen_bwd import (
        ceil_div,
        compute_delta_ws,
        flashattn_bwd_ws,
        reduce_kv_grads_ws,
        to_tilelang_dtype,
    )

    torch.set_default_device("musa")
    torch.manual_seed(42)
    torch.musa.manual_seed(42)
    seq_lens = [
        (512, 512),
        (511, 511),
        (384, 384),
        (257, 257),
        (128, 128),
        (65, 65),
        (320, 320),
        (192, 192),
    ]
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_k = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    num_query_heads, num_kv_heads, head_dim = 6, 1, 256
    scale = head_dim**-0.5

    q = torch.randn(
        (sum(query_lens), num_query_heads, head_dim),
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        (sum(kv_lens), num_kv_heads, head_dim),
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        (sum(kv_lens), num_kv_heads, head_dim),
        dtype=dtype,
        requires_grad=True,
    )
    q_t, k_t, v_t = map(clone_with_grad, (q, k, v))
    q_pt, k_pt, v_pt = map(clone_with_grad, (q, k, v))

    ref_output, _ = ref_attn(
        q_t,
        k_t,
        v_t,
        cu_q,
        cu_k,
        max_query_len,
        max_kv_len,
        is_causal=False,
        is_varlen=True,
        scale=scale,
        upcast=True,
    )
    pt_output, _ = ref_attn(
        q_pt,
        k_pt,
        v_pt,
        cu_q,
        cu_k,
        max_query_len,
        max_kv_len,
        is_causal=False,
        is_varlen=True,
        scale=scale,
        upcast=False,
        reorder_ops=True,
    )
    grad_out = torch.randn_like(ref_output)
    ref_output.backward(gradient=grad_out, retain_graph=True)
    pt_output.backward(gradient=grad_out, retain_graph=True)
    q_static = q.detach()
    k_static = k.detach()
    v_static = v.detach()
    dk_limit = fa_error_limit(k_t.grad, k_pt.grad)
    dv_limit = fa_error_limit(v_t.grad, v_pt.grad)
    kernel_dtype = to_tilelang_dtype(dtype)
    max_seq_q_padded = ceil_div(max_query_len, 64) * 64
    num_blocks_kv = ceil_div(max_kv_len, 64)
    delta_kernel = compute_delta_ws(
        head_dim,
        is_varlen=True,
        dtype=kernel_dtype,
        block_M=64,
        threads=640,
    )
    bwd_kernel = flashattn_bwd_ws(
        dim=head_dim,
        is_causal=False,
        is_varlen=True,
        heads_q_eq_heads_kv=False,
        block_M=64,
        block_N=64,
        smscale=scale,
        threads=640,
        dtype=kernel_dtype,
    )
    reduce_kernel = reduce_kv_grads_ws(
        num_query_heads // num_kv_heads,
        head_dim,
        is_varlen=True,
    )

    dK_accum_results: list[torch.Tensor] = []
    dV_accum_results: list[torch.Tensor] = []
    dk_results: list[torch.Tensor] = []
    dv_results: list[torch.Tensor] = []
    repeat = 8
    for _ in range(repeat):
        with torch.no_grad():
            out_fa, softmax_lse = flash_attn_varlen_func(
                q=q_static,
                k=k_static,
                v=v_static,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                causal=False,
                return_softmax_lse=True,
            )

        delta = torch.empty(
            (q_static.shape[0], num_query_heads),
            device=q_static.device,
            dtype=torch.float32,
        )
        delta_kernel(out_fa, grad_out, delta)
        dQ_accum = torch.zeros(
            (len(seq_lens), num_query_heads, max_seq_q_padded, head_dim),
            dtype=torch.float32,
            device=q_static.device,
        )
        dK_accum = torch.zeros(
            (k_static.shape[0], num_query_heads, head_dim),
            dtype=torch.float32,
            device=k_static.device,
        )
        dV_accum = torch.zeros(
            (v_static.shape[0], num_query_heads, head_dim),
            dtype=torch.float32,
            device=v_static.device,
        )
        debug = torch.empty((num_blocks_kv,), device=q_static.device, dtype=torch.int32)
        bwd_kernel(
            q_static,
            k_static,
            v_static,
            out_fa,
            dQ_accum,
            dK_accum,
            dV_accum,
            grad_out,
            cu_q,
            cu_k,
            softmax_lse,
            delta,
            debug,
        )
        torch.musa.synchronize()
        dK_accum_results.append(dK_accum.detach().clone())
        dV_accum_results.append(dV_accum.detach().clone())
        dk = torch.empty(k_static.shape, dtype=torch.float32, device=k_static.device)
        dv = torch.empty(v_static.shape, dtype=torch.float32, device=v_static.device)
        reduce_kernel(dK_accum, dV_accum, dk, dv)
        torch.musa.synchronize()
        dk_results.append(dk.to(dtype))
        dv_results.append(dv.to(dtype))

    base_dK_accum = dK_accum_results[0]
    base_dV_accum = dV_accum_results[0]
    base_dk = dk_results[0]
    base_dv = dv_results[0]
    for i, (dK_accum, dV_accum, dk, dv) in enumerate(
        zip(dK_accum_results, dV_accum_results, dk_results, dv_results)
    ):
        dK_accum_stable_err = (dK_accum - base_dK_accum).abs().max().item()
        dV_accum_stable_err = (dV_accum - base_dV_accum).abs().max().item()
        dk_ref_err = (dk - k_t.grad).abs().max().item()
        dv_ref_err = (dv - v_t.grad).abs().max().item()
        dk_stable_err = (dk - base_dk).abs().max().item()
        dv_stable_err = (dv - base_dv).abs().max().item()
        assert dK_accum_stable_err == 0.0, (
            f"run {i} dK_accum is not stable across repeated identical inputs; "
            f"max drift from run 0 is {dK_accum_stable_err}"
        )
        assert dV_accum_stable_err == 0.0, (
            f"run {i} dV_accum is not stable across repeated identical inputs; "
            f"max drift from run 0 is {dV_accum_stable_err}"
        )
        assert dk_ref_err <= dk_limit, (
            f"run {i} dk max error {dk_ref_err} exceeds reference limit {dk_limit}; "
            f"max drift from run 0 is {dk_stable_err}"
        )
        assert dv_ref_err <= dv_limit, (
            f"run {i} dv max error {dv_ref_err} exceeds reference limit {dv_limit}; "
            f"max drift from run 0 is {dv_stable_err}"
        )
        assert dk_stable_err == 0.0, (
            f"run {i} dk is not stable across repeated identical inputs; "
            f"max drift from run 0 is {dk_stable_err}"
        )
        assert dv_stable_err == 0.0, (
            f"run {i} dv is not stable across repeated identical inputs; "
            f"max drift from run 0 is {dv_stable_err}"
        )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("num_query_heads,num_kv_heads", [(4, 2), (2, 2)])
def test_flash_attn_bwd_non_varlen_bshd_fa3_api(
    dtype: torch.dtype,
    is_causal: bool,
    num_query_heads: int,
    num_kv_heads: int,
) -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(1)
    torch.musa.manual_seed(1)
    device = "musa"
    batch, seqlen_q, seqlen_k = 2, 16, 16
    head_dim = 256
    scale = head_dim**-0.5

    q_fa = strided_last_dim_tensor(
        (batch, seqlen_q, num_query_heads, head_dim),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k_fa = strided_last_dim_tensor(
        (batch, seqlen_k, num_kv_heads, head_dim),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v_fa = strided_last_dim_tensor(
        (batch, seqlen_k, num_kv_heads, head_dim),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    q_t, k_t, v_t = map(clone_with_grad, (q_fa, k_fa, v_fa))
    q_pt, k_pt, v_pt = map(clone_with_grad, (q_fa, k_fa, v_fa))

    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        causal=is_causal,
        softmax_scale=scale,
        backend="mutlass",
    )
    ref_output, _ = ref_attn(
        q_t,
        k_t,
        v_t,
        None,
        None,
        seqlen_q,
        seqlen_k,
        is_causal=is_causal,
        is_varlen=False,
        scale=scale,
        upcast=True,
    )
    pt_output, _ = ref_attn(
        q_pt,
        k_pt,
        v_pt,
        None,
        None,
        seqlen_q,
        seqlen_k,
        is_causal=is_causal,
        is_varlen=False,
        scale=scale,
        upcast=False,
        reorder_ops=True,
    )

    grad_out = copy_to_strided_last_dim(torch.randn_like(out_fa))
    out_fa.backward(gradient=grad_out, retain_graph=True)
    ref_output.backward(gradient=grad_out, retain_graph=True)
    pt_output.backward(gradient=grad_out, retain_graph=True)

    assert_fa_error_within_pt(out_fa, ref_output, pt_output, "out")
    assert_fa_error_within_pt(q_fa.grad, q_t.grad, q_pt.grad, "dq")
    assert_fa_error_within_pt(k_fa.grad, k_t.grad, k_pt.grad, "dk")
    assert_fa_error_within_pt(v_fa.grad, v_t.grad, v_pt.grad, "dv")


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_bwd_varlen_strided_inputs_fa3_api(
    dtype: torch.dtype,
    is_causal: bool,
) -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(3)
    torch.musa.manual_seed(3)
    device = "musa"
    cu_q = torch.tensor([0, 9, 25], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 11, 27], dtype=torch.int32, device=device)
    total_q, total_k = int(cu_q[-1].item()), int(cu_k[-1].item())
    max_seqlen_q, max_seqlen_k = 16, 16
    num_query_heads, num_kv_heads, head_dim = 2, 2, 256
    scale = head_dim**-0.5

    q_fa = strided_last_dim_tensor(
        (total_q, num_query_heads, head_dim),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k_fa = strided_last_dim_tensor(
        (total_k, num_kv_heads, head_dim),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v_fa = strided_last_dim_tensor(
        (total_k, num_kv_heads, head_dim),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    q_t, k_t, v_t = map(clone_with_grad, (q_fa, k_fa, v_fa))
    q_pt, k_pt, v_pt = map(clone_with_grad, (q_fa, k_fa, v_fa))

    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=is_causal,
        softmax_scale=scale,
        backend="mutlass",
    )
    out_ref, _ = ref_attn(
        q_t,
        k_t,
        v_t,
        cu_q,
        cu_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal=is_causal,
        is_varlen=True,
        scale=scale,
        upcast=True,
    )
    pt_output, _ = ref_attn(
        q_pt,
        k_pt,
        v_pt,
        cu_q,
        cu_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal=is_causal,
        is_varlen=True,
        scale=scale,
        upcast=False,
        reorder_ops=True,
    )
    dout = copy_to_strided_last_dim(torch.randn_like(out_fa))
    out_fa.backward(gradient=dout)
    out_ref.backward(gradient=dout)
    pt_output.backward(gradient=dout)
    torch.musa.synchronize()

    assert_fa_error_within_pt(out_fa, out_ref, pt_output, "out")
    assert_fa_error_within_pt(q_fa.grad, q_t.grad, q_pt.grad, "dq")
    assert_fa_error_within_pt(k_fa.grad, k_t.grad, k_pt.grad, "dk")
    assert_fa_error_within_pt(v_fa.grad, v_t.grad, v_pt.grad, "dv")
