import torch

from mate.norm import (
    fused_add_rmsnorm as _mate_fused_add_rmsnorm,
)
from mate.norm import (
    fused_add_rmsnorm_fp8_block_quant as _mate_fused_add_rmsnorm_fp8_block_quant,
)
from mate.norm import (
    fused_add_rmsnorm_quant as _mate_fused_add_rmsnorm_quant,
)
from mate.norm import (
    fused_dit_gate_residual_layernorm_gamma_beta,
    fused_dit_gate_residual_layernorm_scale_shift,
    fused_dit_residual_layernorm_scale_shift,
    fused_qk_rmsnorm_rope,
)
from mate.norm import fused_rmsnorm_silu as _mate_fused_rmsnorm_silu
from mate.norm import layernorm as _mate_layernorm
from mate.norm import layernorm_quant as _mate_layernorm_quant
from mate.norm import rmsnorm as _mate_rmsnorm
from mate.norm import rmsnorm_quant as _mate_rmsnorm_quant


def _scale_tensor(scale: float | torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    if isinstance(scale, torch.Tensor):
        return scale
    return torch.tensor([scale], dtype=torch.float32, device=input.device)


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    """Apply FlashInfer-compatible RMSNorm."""
    del enable_pdl
    return _mate_rmsnorm(input, weight, eps=eps, y=out)


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    """Apply FlashInfer-compatible Gemma RMSNorm."""
    del enable_pdl
    return _mate_rmsnorm(input, weight, eps=eps, gemma=True, y=out)


def rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: float | torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    """Apply RMSNorm and write the quantized result to ``out``."""
    del enable_pdl
    _mate_rmsnorm_quant(
        input,
        weight,
        _scale_tensor(scale, input),
        eps=eps,
        out=out,
    )


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    """Add ``input`` to ``residual`` and write RMSNorm back to ``input``."""
    del enable_pdl
    _mate_fused_add_rmsnorm(input, residual, weight, eps=eps)


def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    """Apply FlashInfer-compatible Gemma fused add RMSNorm."""
    del enable_pdl
    _mate_fused_add_rmsnorm(input, residual, weight, eps=eps, gemma=True)


def fused_add_rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float | torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    """Apply fused add RMSNorm and write the quantized result to ``out``."""
    del enable_pdl
    _mate_fused_add_rmsnorm_quant(
        input,
        residual,
        weight,
        _scale_tensor(scale, input),
        eps=eps,
        out=out,
    )


def fused_add_rmsnorm_fp8_block_quant(
    out: torch.Tensor,
    block_scale: torch.Tensor,
    normed_out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    """Apply fused add RMSNorm and row-major 1x128 FP8 quantization."""
    del enable_pdl
    _mate_fused_add_rmsnorm_fp8_block_quant(
        out,
        block_scale,
        normed_out,
        input,
        residual,
        weight,
        eps=eps,
    )


def layernorm(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply FlashInfer-compatible LayerNorm."""
    return _mate_layernorm(input, gemma, beta, eps=eps)


def layernorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    scale: float | torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """Apply LayerNorm and write the quantized result to ``out``."""
    _mate_layernorm_quant(
        input,
        gemma,
        beta,
        _scale_tensor(scale, input),
        eps=eps,
        out=out,
    )


def fused_rmsnorm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    block_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply fused RMSNorm and SiLU activation."""
    return _mate_fused_rmsnorm_silu(
        input,
        weight,
        eps=eps,
        out=out,
        block_scale=block_scale,
    )


__all__ = [
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_fp8_block_quant",
    "fused_add_rmsnorm_quant",
    "fused_dit_gate_residual_layernorm_gamma_beta",
    "fused_dit_gate_residual_layernorm_scale_shift",
    "fused_dit_residual_layernorm_scale_shift",
    "fused_qk_rmsnorm_rope",
    "fused_rmsnorm_silu",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "layernorm",
    "layernorm_quant",
    "rmsnorm",
    "rmsnorm_quant",
]
