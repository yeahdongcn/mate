from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from mate.api_logging import mate_api
from mate.jit.kda_ops import (
    get_kda_fused_ops_function_name,
    get_kda_fused_ops_module,
    make_kda_fused_ops_config,
)

_SUPPORTED_QKVA_DTYPES = (torch.float16, torch.bfloat16)


def _as_4d_varlen_input(
    x: torch.Tensor,
    *,
    name: str,
    cu_seqlens: Optional[torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    if cu_seqlens is None:
        if x.ndim != 4:
            raise ValueError(f"{name} must be a 4D tensor [B, T, H, 128].")
        return x, False
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(
                f"{name}.shape[0] must be 1 when cu_seqlens is provided; "
                "flatten variable-length input as [S, H, 128]."
            )
        return x, False
    raise ValueError(f"{name} must be [S, H, 128] or [1, S, H, 128] for varlen.")


def _check_state_dtype(
    x: torch.Tensor,
    *,
    name: str,
    value_dtype: torch.dtype,
) -> None:
    if x.dtype not in (value_dtype, torch.float32):
        raise TypeError(f"{name} must have dtype {value_dtype} or torch.float32.")


@mate_api
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: float = -5.0,
    use_qk_l2norm_in_kernel: bool = True,
    output: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run the fused chunk KDA kernel.

    Args:
        q: Query tensor with shape ``[B, T, Hqk, 128]`` for dense mode or
            ``[S, Hqk, 128]`` / ``[1, S, Hqk, 128]`` for varlen mode.
        k: Key tensor with the same shape and dtype as ``q``.
        v: Value tensor with shape ``[B, T, Hv, 128]`` or varlen equivalent.
        g: Gate input tensor with the same shape as ``v``.
        beta: Beta logits tensor with shape ``[B, T, Hv]`` or varlen equivalent.
        scale: Optional QK scaling factor. Defaults to ``128**-0.5``.
        initial_state: Optional recurrent state tensor.
        output_final_state: Whether to return the final recurrent state.
        cu_seqlens: Optional cumulative sequence lengths for varlen mode.
        A_log: Optional per-head gate parameter tensor.
        dt_bias: Optional per-head, per-channel gate bias tensor.
        lower_bound: Gate lower bound used when gate parameters ``A_log`` and
            ``dt_bias`` are enabled. Defaults to ``-5.0``.
        use_qk_l2norm_in_kernel: Whether to normalize Q/K in the kernel.
        output: Optional preallocated output tensor.
        final_state: Optional preallocated final-state tensor.
    """

    q, squeeze_varlen = _as_4d_varlen_input(q, name="q", cu_seqlens=cu_seqlens)
    k, _ = _as_4d_varlen_input(k, name="k", cu_seqlens=cu_seqlens)
    v, _ = _as_4d_varlen_input(v, name="v", cu_seqlens=cu_seqlens)
    g, _ = _as_4d_varlen_input(g, name="g", cu_seqlens=cu_seqlens)
    if beta.ndim == 2 and cu_seqlens is not None:
        beta = beta.unsqueeze(0)

    if (
        q.shape[-1] != 128
        or k.shape[-1] != 128
        or v.shape[-1] != 128
        or g.shape[-1] != 128
    ):
        raise ValueError("chunk_kda currently requires D=128.")
    if k.shape != q.shape:
        raise ValueError("k must have the same shape as q.")
    if q.shape[:2] != v.shape[:2] or q.shape[:2] != g.shape[:2]:
        raise ValueError("q, v and g must have matching [B, T] dimensions.")
    if g.shape != v.shape:
        raise ValueError("g must have the same shape as v.")
    if v.shape[2] % q.shape[2] != 0:
        raise ValueError("GVA requires v/g heads to be divisible by q/k heads.")
    if beta.shape != v.shape[:3]:
        raise ValueError("beta must have shape [B, T, Hv].")
    if q.dtype not in _SUPPORTED_QKVA_DTYPES:
        raise TypeError("chunk_kda supports torch.float16 and torch.bfloat16 inputs.")
    if (
        k.dtype != q.dtype
        or v.dtype != q.dtype
        or g.dtype != q.dtype
        or beta.dtype != q.dtype
    ):
        raise TypeError("k, v, g and beta must have the same dtype as q.")
    if (A_log is None) != (dt_bias is None):
        raise ValueError("A_log and dt_bias must be provided together.")
    if initial_state is not None:
        _check_state_dtype(initial_state, name="initial_state", value_dtype=q.dtype)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if cu_seqlens is not None:
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must have dtype torch.int32 or torch.int64.")
        cu_seqlens = cu_seqlens.contiguous()

    if output is None:
        output = torch.empty_like(v)
    elif squeeze_varlen and output.ndim == 3:
        output = output.unsqueeze(0)
    if output.shape != v.shape:
        raise ValueError("output must have the same shape as v.")
    if output.dtype != q.dtype:
        raise TypeError("output must have the same dtype as q.")

    if output_final_state and final_state is None:
        nseq = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else q.shape[0]
        state_dtype = initial_state.dtype if initial_state is not None else q.dtype
        final_state = torch.empty(
            (nseq, v.shape[2], 128, 128),
            device=q.device,
            dtype=state_dtype,
        )
    elif not output_final_state:
        final_state = None
    elif final_state is not None:
        _check_state_dtype(final_state, name="final_state", value_dtype=q.dtype)

    if (
        initial_state is not None
        and final_state is not None
        and initial_state.dtype != final_state.dtype
    ):
        raise TypeError("initial_state and final_state must have the same dtype.")

    state_fp32 = (
        initial_state is not None and initial_state.dtype == torch.float32
    ) or (final_state is not None and final_state.dtype == torch.float32)
    state_dtype = (
        initial_state.dtype
        if initial_state is not None
        else final_state.dtype
        if final_state is not None
        else q.dtype
    )
    kda_config = make_kda_fused_ops_config(
        q.dtype,
        state_dtype=state_dtype,
        cu_seqlens_dtype=cu_seqlens.dtype if cu_seqlens is not None else None,
        has_state_in=initial_state is not None,
        has_state_out=final_state is not None,
        state_fp32=state_fp32,
        has_gate_params=A_log is not None,
        is_varlen=cu_seqlens is not None,
        normalize_qk=bool(use_qk_l2norm_in_kernel),
    )
    kda_func_name = get_kda_fused_ops_function_name(kda_config)

    get_kda_fused_ops_module(kda_config).get_function(kda_func_name)(
        q,
        k,
        v,
        g,
        beta,
        output,
        initial_state,
        final_state,
        cu_seqlens,
        A_log,
        dt_bias,
        float(scale),
        float(lower_bound),
        bool(use_qk_l2norm_in_kernel),
    )

    if squeeze_varlen:
        output = output.squeeze(0)
    if output_final_state:
        assert final_state is not None
        return output, final_state
    return output
