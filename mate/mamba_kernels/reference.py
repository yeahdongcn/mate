"""PyTorch oracle for the native mamba/SSU family.

The kernel tests compare against these functions, and the family falls back to
them only where documented: they are the definition of correct behaviour, not a
shipping fast path. The recurrence arithmetic is fp32 with the same operand order
as the kernels, so a mismatch points at the kernel rather than at rounding.
"""

from __future__ import annotations

import torch

__all__ = ["selective_state_update_one_token_reference"]

_SOFTPLUS_THRESHOLD = 20.0


def selective_state_update_one_token_reference(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor | None,
    pad_slot_id: int,
    disable_state_update: bool,
) -> torch.Tensor:
    """Reference one-token selective state update.

    Per-head ``A``, ``D`` and ``dt_bias`` broadcast over the state dimensions.
    ``state`` is updated in place through the destination slots; the returned
    tensor is the per-token output ``y``.
    """
    slots, heads, dim, dstate = state.shape
    batch = x.shape[0]
    groups = B.shape[1]
    head_ratio = heads // groups

    dt_value = dt.to(torch.float32)
    if dt_bias is not None:
        dt_value = dt_value + dt_bias.to(torch.float32).view(1, heads, 1)
    if dt_softplus:
        dt_value = torch.where(
            dt_value > _SOFTPLUS_THRESHOLD,
            dt_value,
            torch.log1p(torch.exp(dt_value)),
        )

    y = torch.empty((batch, heads, dim), dtype=torch.float32, device=x.device)
    dst_slots = src_slots if dst_slots is None else dst_slots
    head_groups = torch.arange(heads, device=x.device) // head_ratio
    a_head = A.to(torch.float32).view(heads, 1, 1)

    for b in range(batch):
        src = int(src_slots[b])
        dst = int(dst_slots[b])
        old = (
            torch.zeros((heads, dim, dstate), dtype=torch.float32, device=x.device)
            if src == pad_slot_id
            else state[src].to(torch.float32)
        )
        xb = x[b].to(torch.float32)
        dtb = dt_value[b]
        decay = torch.exp(a_head * dtb[:, :, None])
        # Same operand order as the kernel: (dt * B) * x.
        update = (
            dtb[:, :, None] * B[b].to(torch.float32)[head_groups][:, None, :]
        ) * xb[:, :, None]
        new = old * decay + update
        y[b] = (new * C[b].to(torch.float32)[head_groups][:, None, :]).sum(dim=-1)
        if D is not None:
            y[b] = y[b] + D.to(torch.float32).view(heads, 1) * xb
        if z is not None:
            zb = z[b].to(torch.float32)
            y[b] = y[b] * zb * torch.sigmoid(zb)
        if not disable_state_update and dst != pad_slot_id:
            state[dst].copy_(new.to(state.dtype))

    return y.to(x.dtype)
