from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from mate.norm.tilelang import (
    fused_add_rmsnorm,
    fused_add_rmsnorm_fp8_block_quant,
    fused_add_rmsnorm_quant,
    fused_dit_gate_residual_layernorm_gamma_beta,
    fused_dit_gate_residual_layernorm_scale_shift,
    fused_dit_residual_layernorm_scale_shift,
    fused_qk_rmsnorm_rope,
    fused_rmsnorm_silu,
    layernorm,
    layernorm_quant,
    rmsnorm,
    rmsnorm_quant,
)

pytest.importorskip("tilelang")
pytest.importorskip("torch_musa")

_FP8_MAX = 448.0
_NORM_PUBLIC_API_NAMES = (
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_fp8_block_quant",
    "fused_add_rmsnorm_quant",
    "fused_dit_gate_residual_layernorm_gamma_beta",
    "fused_dit_gate_residual_layernorm_scale_shift",
    "fused_dit_residual_layernorm_scale_shift",
    "fused_qk_rmsnorm_rope",
    "fused_rmsnorm_silu",
    "layernorm",
    "layernorm_quant",
    "rmsnorm",
    "rmsnorm_quant",
)


def test_norm_public_apis_are_logged() -> None:
    script = textwrap.dedent(
        f"""
        import mate.norm

        api_names = {_NORM_PUBLIC_API_NAMES!r}
        for name in api_names:
            try:
                getattr(mate.norm, name)()
            except TypeError:
                pass
        """
    )
    env = os.environ.copy()
    env["MATE_LOGLEVEL"] = "1"
    env["MATE_LOGDEST"] = "stdout"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    for name in _NORM_PUBLIC_API_NAMES:
        assert f"MATE API Call: {name}" in result.stdout


class QKRMSNormRoPEReference:
    @staticmethod
    def adjusted_freq(
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

    @staticmethod
    def qkv_as_3d(qkv: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, bool]:
        if qkv.dim() == 3:
            return qkv, True
        batch_size = qkv.size(0) // seq_len
        return qkv.as_strided(
            (batch_size, seq_len, qkv.size(1)),
            (seq_len * qkv.stride(0), qkv.stride(0), qkv.stride(1)),
        ), False

    @classmethod
    def apply_rope_interleave(
        cls,
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

                        freq = cls.adjusted_freq(
                            half_dim_val, dim_size, base, factor, low, high
                        )
                        theta_t = torch.tensor(
                            pos_id * freq, device=x.device, dtype=torch.float32
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

    @classmethod
    def apply_rope_neox(
        cls,
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

                        freq = cls.adjusted_freq(
                            half_dim_val, dim_size, base, factor, low, high
                        )
                        theta_t = torch.tensor(
                            pos_id * freq, device=x.device, dtype=torch.float32
                        )
                        y = x[b, s, h, d] * torch.cos(theta_t) + rotate_sign * x[
                            b, s, h, partner_dim
                        ] * torch.sin(theta_t)
                        if factor != 1.0:
                            y = y * attention_factor
                        out[b, s, h, d] = y
        return out

    @classmethod
    def apply_rope(
        cls,
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
        apply_fn = cls.apply_rope_interleave if interleave else cls.apply_rope_neox
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

    @classmethod
    def run(
        cls,
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
        qkv_3d, input_is_3d = cls.qkv_as_3d(qkv, seq_len)
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

        q = cls.apply_rope(
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
        k = cls.apply_rope(
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


def _make_input(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    noncontiguous: bool,
) -> torch.Tensor:
    if not noncontiguous:
        return torch.randn(*shape, device=device, dtype=dtype)
    if len(shape) == 2:
        return torch.randn(shape[0], shape[1] * 2, device=device, dtype=dtype)[
            :, : shape[1]
        ]
    return torch.randn(
        shape[0],
        shape[1] * 2,
        shape[2] * 2,
        device=device,
        dtype=dtype,
    )[:, : shape[1], : shape[2]]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("shape", [(4, 1024), (2, 3, 1024)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_rmsnorm(
    dtype: torch.dtype,
    gemma: bool,
    shape: tuple[int, ...],
    noncontiguous: bool,
) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    hidden_size = shape[-1]
    eps = 1e-6

    x = _make_input(shape, dtype, device, noncontiguous)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    y = torch.empty_strided(
        tuple(x.shape), tuple(x.stride()), device=device, dtype=dtype
    )

    out = rmsnorm(x, weight, eps=eps, gemma=gemma, y=y)

    weight_ref = weight.float() + (1.0 if gemma else 0.0)
    ref = x.float()
    ref = ref * torch.rsqrt(ref.square().mean(dim=-1, keepdim=True) + eps)
    ref = (ref * weight_ref).to(dtype)

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("use_out", [False, True])
def test_rmsnorm_quant(
    dtype: torch.dtype,
    gemma: bool,
    out_dtype: torch.dtype,
    use_out: bool,
) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    shape = (4, 1024)
    hidden_size = shape[-1]
    eps = 1e-6

    x = _make_input(shape, dtype, device, False)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    out = torch.empty(shape, device=device, dtype=out_dtype) if use_out else None
    expected_dtype = out_dtype if use_out else torch.float8_e4m3fn

    out = rmsnorm_quant(x, weight, scale, eps=eps, gemma=gemma, out=out)

    weight_ref = weight.float() + (1.0 if gemma else 0.0)
    ref = x.float()
    ref = ref * torch.rsqrt(ref.square().mean(dim=-1, keepdim=True) + eps)
    fp8_max = 448.0 if expected_dtype == torch.float8_e4m3fn else 57344.0
    ref = (ref * weight_ref / scale.float()).clamp(-fp8_max, fp8_max)
    ref = ref.to(expected_dtype)

    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("shape", [(4, 1024)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_layernorm(
    shape: tuple[int, ...],
    noncontiguous: bool,
) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    hidden_size = shape[-1]
    eps = 1e-6

    dtype = torch.bfloat16
    x = _make_input(shape, dtype, device, noncontiguous)
    gemma = torch.randn(hidden_size, device=device, dtype=torch.float32)
    beta = torch.randn(hidden_size, device=device, dtype=torch.float32)

    out = layernorm(x, gemma, beta, eps=eps)

    x_ref = x.float()
    mean = x_ref.mean(dim=-1, keepdim=True)
    var = x_ref.square().mean(dim=-1, keepdim=True) - mean.square()
    ref = ((x_ref - mean) * torch.rsqrt(var + eps) * gemma + beta).to(dtype)

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_layernorm_quant(out_dtype: torch.dtype) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    shape = (4, 1024)
    eps = 1e-6
    dtype = torch.bfloat16

    x = _make_input(shape, dtype, device, True)
    gemma = torch.randn(shape[-1], device=device, dtype=torch.float32)
    beta = torch.randn(shape[-1], device=device, dtype=torch.float32)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    out = torch.empty(shape, device=device, dtype=out_dtype)

    ref = x.float()
    mean = ref.mean(dim=-1, keepdim=True)
    var = ref.square().mean(dim=-1, keepdim=True) - mean.square()
    ref = (ref - mean) * torch.rsqrt(var + eps) * gemma.float() + beta.float()
    fp8_max = 448.0 if out_dtype == torch.float8_e4m3fn else 57344.0
    ref = (ref / scale.float()).clamp(-fp8_max, fp8_max).to(out_dtype)

    out = layernorm_quant(x, gemma, beta, scale, eps=eps, out=out)

    atol = 0.25 if out_dtype == torch.float8_e4m3fn else 1.0
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.0, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_fused_add_rmsnorm(
    dtype: torch.dtype,
    gemma: bool,
    noncontiguous: bool,
) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    shape = (4, 1024)
    hidden_size = shape[-1]
    eps = 1e-6

    x = _make_input(shape, dtype, device, noncontiguous)
    residual = _make_input(shape, dtype, device, noncontiguous)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    x_ref_in = x.clone()
    residual_ref_in = residual.clone()

    fused_add_rmsnorm(x, residual, weight, eps=eps, gemma=gemma)

    h_ref = x_ref_in.float() + residual_ref_in.float()
    residual_ref = h_ref.to(dtype)
    weight_ref = weight.float() + (1.0 if gemma else 0.0)
    norm_ref = h_ref * torch.rsqrt(h_ref.square().mean(dim=-1, keepdim=True) + eps)
    norm_ref = (norm_ref * weight_ref).to(dtype)

    torch.testing.assert_close(residual, residual_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x, norm_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("use_out", [False, True])
def test_fused_add_rmsnorm_quant(
    dtype: torch.dtype,
    gemma: bool,
    out_dtype: torch.dtype,
    use_out: bool,
) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    shape = (4, 1024)
    hidden_size = shape[-1]
    eps = 1e-6

    x = _make_input(shape, dtype, device, False)
    residual = _make_input(shape, dtype, device, False)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    out = torch.empty(shape, device=device, dtype=out_dtype) if use_out else None
    expected_dtype = out_dtype if use_out else torch.float8_e4m3fn

    x_ref_in = x.clone()
    residual_ref_in = residual.clone()

    out = fused_add_rmsnorm_quant(
        x,
        residual,
        weight,
        scale,
        eps=eps,
        gemma=gemma,
        out=out,
    )

    h_ref = x_ref_in.float() + residual_ref_in.float()
    residual_ref = h_ref.to(dtype)
    weight_ref = weight.float() + (1.0 if gemma else 0.0)
    ref = h_ref  # nv store fp32 to do next，code must be this instead of residual_ref.float()
    ref = ref * torch.rsqrt(ref.square().mean(dim=-1, keepdim=True) + eps)
    fp8_max = 448.0 if expected_dtype == torch.float8_e4m3fn else 57344.0
    ref = (ref * weight_ref / scale.float()).clamp(-fp8_max, fp8_max)
    ref = ref.to(expected_dtype)

    torch.testing.assert_close(residual, residual_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((4, 1024), torch.float16),
        ((4, 1024), torch.bfloat16),
        ((3, 6144), torch.bfloat16),
    ],
)
def test_fused_add_rmsnorm_fp8_block_quant(
    shape: tuple[int, int], dtype: torch.dtype
) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    torch.manual_seed(0)
    rows, hidden_size = shape
    eps = 1e-6
    x = torch.randn(shape, device=device, dtype=dtype) * 0.1
    residual = torch.randn_like(x) * 0.1
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    block_scale = torch.empty(
        (rows, hidden_size // 128), device=device, dtype=torch.float32
    )
    normed_out = torch.empty_like(x)
    x_before = x.clone()
    residual_before = residual.clone()

    result = fused_add_rmsnorm_fp8_block_quant(
        out, block_scale, normed_out, x, residual, weight, eps=eps
    )

    assert result is None
    assert block_scale.shape == (rows, hidden_size // 128)
    assert block_scale.is_contiguous()
    torch.testing.assert_close(x, x_before, rtol=0.0, atol=0.0)

    residual_f32 = x_before.float() + residual_before.float()
    residual_ref = residual_f32.to(dtype)
    normed_ref = (
        residual_f32
        * torch.rsqrt(residual_f32.square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    ).to(dtype)
    normed_blocks = normed_ref.float().view(rows, hidden_size // 128, 128)
    scale_ref = normed_blocks.abs().amax(dim=-1).clamp_min(1e-4) / _FP8_MAX

    torch.testing.assert_close(residual, residual_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(normed_out, normed_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(block_scale, scale_ref, rtol=2e-2, atol=1e-6)
    dequantized = (
        out.float().view(rows, hidden_size // 128, 128) * block_scale.unsqueeze(-1)
    ).view(shape)
    torch.testing.assert_close(dequantized, normed_ref.float(), rtol=0.1, atol=0.3)


def test_fused_add_rmsnorm_fp8_block_quant_rejects_non_row_major_scale() -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    x = torch.randn((2, 1024), device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(1024, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    normed_out = torch.empty_like(x)
    block_scale = torch.empty((8, 2), device=device, dtype=torch.float32).t()

    with pytest.raises(RuntimeError, match="contiguous row-major"):
        fused_add_rmsnorm_fp8_block_quant(
            out, block_scale, normed_out, x, residual, weight
        )


@pytest.mark.parametrize("quantized", [False, True])
def test_fused_add_rmsnorm_tail(quantized: bool) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    shape = (4, 320)
    eps = 1e-6
    x = torch.randn(shape, device=device, dtype=torch.bfloat16)
    residual = torch.randn(shape, device=device, dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], device=device, dtype=torch.bfloat16)
    x_ref_in = x.clone()
    residual_ref_in = residual.clone()

    h_ref = x_ref_in.float() + residual_ref_in.float()
    residual_ref = h_ref.to(torch.bfloat16)
    norm_ref = h_ref * torch.rsqrt(h_ref.square().mean(dim=-1, keepdim=True) + eps)
    norm_ref = norm_ref * weight.float()

    if quantized:
        scale = torch.ones(1, device=device, dtype=torch.float32)
        result = fused_add_rmsnorm_quant(x, residual, weight, scale, eps=eps)
        expected = norm_ref.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        torch.testing.assert_close(
            result.float(), expected.float(), rtol=2e-2, atol=2e-2
        )
    else:
        fused_add_rmsnorm(x, residual, weight, eps=eps)
        torch.testing.assert_close(x, norm_ref.to(torch.bfloat16), rtol=2e-2, atol=2e-2)

    torch.testing.assert_close(residual, residual_ref, rtol=2e-2, atol=2e-2)


def fused_rmsnorm_silu_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out_dtype: torch.dtype | None = None,
    block_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    norm = (
        input.float()
        * torch.rsqrt(input.float().square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    )
    silu = norm / (torch.exp(-norm) + 1.0)
    if out_dtype is None or out_dtype == torch.bfloat16:
        return silu.to(torch.bfloat16)
    if out_dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            "fused_rmsnorm_silu_reference expects bfloat16 or float8_e4m3fn output."
        )

    return silu.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("output_mode", ["bf16", "fp8", "mxfp8"])
def test_fused_rmsnorm_silu(output_mode: str) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")
    if output_mode == "mxfp8" and not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("MXFP8 dtype is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    input = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)

    if output_mode == "bf16":
        out = torch.empty_like(input)
        result = fused_rmsnorm_silu(input, weight, eps=1e-6, out=out)
        ref = fused_rmsnorm_silu_reference(
            input, weight, eps=1e-6, out_dtype=torch.bfloat16
        )
        torch.testing.assert_close(result, ref, rtol=2e-2, atol=2e-2)
        return

    if output_mode == "fp8":
        out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        result = fused_rmsnorm_silu(input, weight, eps=1e-6, out=out)
        ref = fused_rmsnorm_silu_reference(
            input, weight, eps=1e-6, out_dtype=torch.float8_e4m3fn
        )
        torch.testing.assert_close(result.float(), ref.float(), rtol=2e-2, atol=0.25)
        return

    out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    block_scale = torch.empty(
        (input.size(0), input.size(1) // 32),
        device=device,
        dtype=torch.float8_e8m0fnu,
    )
    result, block_scale = fused_rmsnorm_silu(
        input, weight, eps=1e-6, out=out, block_scale=block_scale
    )
    ref = fused_rmsnorm_silu_reference(
        input, weight, eps=1e-6, out_dtype=torch.bfloat16
    )
    scale_u8 = block_scale.contiguous().view(torch.uint8)
    scale = (
        (scale_u8.to(torch.int32) << 23)
        .view(torch.float32)
        .repeat_interleave(32, dim=-1)
    )
    torch.testing.assert_close(result.float() * scale, ref.float(), rtol=2e-2, atol=0.5)


def test_fused_rmsnorm_silu_mxfp8_zero_full_group() -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("MXFP8 dtype is not available")

    device = torch.device("musa")
    input = torch.zeros(2, 1024, device=device, dtype=torch.bfloat16)
    weight = torch.randn(1024, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    block_scale = torch.empty(2, 32, device=device, dtype=torch.float8_e8m0fnu)

    result, block_scale = fused_rmsnorm_silu(
        input,
        weight,
        eps=1e-6,
        out=out,
        block_scale=block_scale,
    )

    result_f32 = result.float()
    assert torch.isfinite(result_f32).all()
    assert torch.count_nonzero(result_f32) == 0
    assert torch.count_nonzero(block_scale.view(torch.uint8)) == 0


@pytest.mark.parametrize(
    "shape_mode, interleave, is_qk_norm, output_fp8, factor, low, high, attention_factor",
    [
        ("3d", True, True, False, 1.0, 0.0, 0.0, 1.0),
        ("3d", False, True, True, 1.0, 0.0, 0.0, 1.0),
        ("2d", True, False, False, 1.0, 0.0, 0.0, 1.0),
        ("2d", False, False, True, 1.0, 0.0, 0.0, 1.0),
        ("3d", True, True, False, 1.1, 0.0, 2.0, 0.9),
    ],
)
def test_fused_qk_rmsnorm_rope(
    shape_mode: str,
    interleave: bool,
    is_qk_norm: bool,
    output_fp8: bool,
    factor: float,
    low: float,
    high: float,
    attention_factor: float,
) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

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
    batch_size = 2

    if shape_mode == "3d":
        qkv = torch.randn(
            batch_size, seq_len, hidden_qkv, device=device, dtype=torch.bfloat16
        )
        q_shape = (batch_size, seq_len, num_heads_q, head_dim)
        k_shape = (batch_size, seq_len, num_heads_k, head_dim)
        v_shape = (batch_size, seq_len, num_heads_v, head_dim)
    else:
        qkv = torch.randn(
            batch_size * seq_len, hidden_qkv, device=device, dtype=torch.bfloat16
        )
        q_shape = (batch_size * seq_len, num_heads_q, head_dim)
        k_shape = (batch_size * seq_len, num_heads_k, head_dim)
        v_shape = (batch_size * seq_len, num_heads_v, head_dim)

    q_weight = torch.randn(num_heads_q * head_dim, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(num_heads_k * head_dim, device=device, dtype=torch.bfloat16)
    out_dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
    q_out = torch.empty(q_shape, device=device, dtype=out_dtype)
    k_out = torch.empty(k_shape, device=device, dtype=out_dtype)
    v_out = torch.empty(v_shape, device=device, dtype=out_dtype)

    actual_q, actual_k, actual_v = fused_qk_rmsnorm_rope(
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
        factor=factor,
        low=low,
        high=high,
        attention_factor=attention_factor,
        interleave=interleave,
        is_qk_norm=is_qk_norm,
        output_fp8=output_fp8,
        q_out=q_out,
        k_out=k_out,
        v_out=v_out,
    )
    ref_q, ref_k, ref_v = QKRMSNormRoPEReference.run(
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
        factor=factor,
        low=low,
        high=high,
        attention_factor=attention_factor,
        interleave=interleave,
        is_qk_norm=is_qk_norm,
        output_fp8=output_fp8,
        output_quant_scale=1.0,
        v_quant_scale=1.0,
    )
    if output_fp8:
        torch.testing.assert_close(
            actual_q.float(), ref_q.float(), rtol=0.0, atol=0.03125
        )
        torch.testing.assert_close(
            actual_k.float(), ref_k.float(), rtol=0.0, atol=0.03125
        )
        torch.testing.assert_close(
            actual_v.float(), ref_v.float(), rtol=0.0, atol=0.03125
        )
    else:
        torch.testing.assert_close(actual_q, ref_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_k, ref_k, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_v, ref_v, rtol=2e-2, atol=2e-2)


def _make_strided_gate(
    batch: int, num_rows: int, hidden_size: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        batch, num_rows * 6, hidden_size, device=device, dtype=torch.bfloat16
    )[:, ::6, :]


def _make_strided_input(
    batch: int, num_rows: int, hidden_size: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        batch, num_rows + 3, hidden_size, device=device, dtype=torch.bfloat16
    )[:, ::2, :]


def _layernorm_f32(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = x_f32.square().mean(dim=-1, keepdim=True) - mean.square()
    return (x_f32 - mean) * torch.rsqrt(var + eps)


def _gamma_beta_ref(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float
) -> torch.Tensor:
    return (_layernorm_f32(x, eps) * gamma.float() + beta.float()).to(torch.bfloat16)


def _scale_shift_ref(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    scale_bias: torch.Tensor | None = None,
    shift_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    scale_f32 = scale.float()
    shift_f32 = shift.float()
    if scale_bias is not None:
        scale_f32 = scale_f32 + scale_bias.float()
    if shift_bias is not None:
        shift_f32 = shift_f32 + shift_bias.float()
    return (_layernorm_f32(x, eps) * (1.0 + scale_f32) + shift_f32).to(torch.bfloat16)


def _mxfp8_dequant(out: torch.Tensor, sf_out: torch.Tensor) -> torch.Tensor:
    fp8 = out.view(torch.float8_e4m3fn)
    scale = (
        (sf_out.to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, dim=-1)
    )
    return fp8.float() * scale


@pytest.mark.parametrize("output_mode", ["bf16", "mxfp8"])
def test_fused_dit_gate_residual_layernorm_gamma_beta(output_mode: str) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")
    if output_mode == "mxfp8" and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("MXFP8 dtype is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    shape = (2, 3, 3072)
    input = _make_strided_input(*shape[:2], shape[2], device)
    residual = _make_strided_input(*shape[:2], shape[2], device)
    gate = _make_strided_gate(*shape[:2], shape[2], device)
    gate_bias = torch.randn(1, shape[2], device=device, dtype=torch.float32)
    gamma = torch.randn(shape[2], device=device, dtype=torch.float32)
    beta = torch.randn(shape[2], device=device, dtype=torch.float32)
    sf_out = (
        torch.empty(
            shape[0], shape[1], shape[2] // 32, device=device, dtype=torch.uint8
        )
        if output_mode == "mxfp8"
        else None
    )

    residual_out, norm_out = fused_dit_gate_residual_layernorm_gamma_beta(
        input,
        residual,
        gate,
        gamma,
        beta,
        gate_bias=gate_bias,
        use_mxfp8=output_mode == "mxfp8",
        sf_out=sf_out,
    )

    ref_residual = (
        residual.float() + input.float() * (gate.float() + gate_bias.float())
    ).to(torch.bfloat16)
    ref_norm = _gamma_beta_ref(ref_residual, gamma, beta, 1e-6)

    torch.testing.assert_close(residual_out, ref_residual, rtol=2e-2, atol=2e-2)
    if output_mode == "mxfp8":
        assert sf_out is not None
        torch.testing.assert_close(
            _mxfp8_dequant(norm_out, sf_out), ref_norm.float(), rtol=2e-2, atol=2.1
        )
    else:
        torch.testing.assert_close(norm_out, ref_norm, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("output_mode", ["bf16", "mxfp8"])
def test_fused_dit_gate_residual_layernorm_scale_shift(output_mode: str) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")
    if output_mode == "mxfp8" and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("MXFP8 dtype is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    shape = (2, 3, 3072)
    input = _make_strided_input(*shape[:2], shape[2], device)
    residual = _make_strided_input(*shape[:2], shape[2], device)
    gate = _make_strided_gate(*shape[:2], shape[2], device)
    scale = _make_strided_gate(*shape[:2], shape[2], device)
    shift = _make_strided_gate(*shape[:2], shape[2], device)
    gate_bias = torch.randn(1, shape[2], device=device, dtype=torch.float32)
    scale_bias = torch.randn(1, shape[2], device=device, dtype=torch.float32)
    shift_bias = torch.randn(1, shape[2], device=device, dtype=torch.float32)
    sf_out = (
        torch.empty(
            shape[0], shape[1], shape[2] // 32, device=device, dtype=torch.uint8
        )
        if output_mode == "mxfp8"
        else None
    )

    residual_out, norm_out = fused_dit_gate_residual_layernorm_scale_shift(
        input,
        residual,
        gate,
        scale,
        shift,
        gate_bias=gate_bias,
        scale_bias=scale_bias,
        shift_bias=shift_bias,
        use_mxfp8=output_mode == "mxfp8",
        sf_out=sf_out,
    )

    ref_residual = (
        residual.float() + input.float() * (gate.float() + gate_bias.float())
    ).to(torch.bfloat16)
    ref_norm = _scale_shift_ref(
        ref_residual, scale, shift, 1e-6, scale_bias, shift_bias
    )

    torch.testing.assert_close(residual_out, ref_residual, rtol=2e-2, atol=2e-2)
    if output_mode == "mxfp8":
        assert sf_out is not None
        torch.testing.assert_close(
            _mxfp8_dequant(norm_out, sf_out), ref_norm.float(), rtol=2e-2, atol=2.1
        )
    else:
        torch.testing.assert_close(norm_out, ref_norm, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("output_mode", ["bf16", "mxfp8"])
def test_fused_dit_residual_layernorm_scale_shift(output_mode: str) -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")
    if output_mode == "mxfp8" and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("MXFP8 dtype is not available")

    device = torch.device("musa")
    torch.manual_seed(0)

    shape = (2, 3, 3072)
    input = _make_strided_input(*shape[:2], shape[2], device)
    scale = _make_strided_gate(*shape[:2], shape[2], device)
    shift = _make_strided_gate(*shape[:2], shape[2], device)
    scale_bias = torch.randn(1, shape[2], device=device, dtype=torch.float32)
    shift_bias = torch.randn(1, shape[2], device=device, dtype=torch.float32)
    sf_out = (
        torch.empty(
            shape[0], shape[1], shape[2] // 32, device=device, dtype=torch.uint8
        )
        if output_mode == "mxfp8"
        else None
    )

    residual_out, norm_out = fused_dit_residual_layernorm_scale_shift(
        input,
        scale,
        shift,
        residual=None,
        scale_bias=scale_bias,
        shift_bias=shift_bias,
        use_mxfp8=output_mode == "mxfp8",
        sf_out=sf_out,
    )

    ref_residual = input.to(torch.bfloat16)
    ref_norm = _scale_shift_ref(
        ref_residual, scale, shift, 1e-6, scale_bias, shift_bias
    )

    torch.testing.assert_close(residual_out, ref_residual, rtol=2e-2, atol=2e-2)
    if output_mode == "mxfp8":
        assert sf_out is not None
        torch.testing.assert_close(
            _mxfp8_dequant(norm_out, sf_out), ref_norm.float(), rtol=2e-2, atol=2.1
        )
    else:
        torch.testing.assert_close(norm_out, ref_norm, rtol=2e-2, atol=2e-2)


def test_fused_dit_bias_rejects_multiple_rows() -> None:
    if not torch.musa.is_available():
        pytest.skip("MUSA is not available")

    device = torch.device("musa")
    shape = (1, 1, 3072)
    input = torch.randn(shape, device=device, dtype=torch.bfloat16)
    scale = torch.randn(shape, device=device, dtype=torch.bfloat16)
    shift = torch.randn(shape, device=device, dtype=torch.bfloat16)
    scale_bias = torch.randn(2, shape[-1], device=device, dtype=torch.float32)

    with pytest.raises(
        RuntimeError, match=r"shape \[hidden_size\] or \[1, hidden_size\]"
    ):
        fused_dit_residual_layernorm_scale_shift(
            input,
            scale,
            shift,
            scale_bias=scale_bias,
        )
