"""PyTorch oracle for the native mamba/SSU family.

The kernel tests compare against these functions, and the family falls back to
them only where documented: they are the definition of correct behaviour, not a
shipping fast path. The recurrence arithmetic is fp32 with the same operand order
as the kernels, so a mismatch points at the kernel rather than at rounding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = [
    "selective_state_update_multi_token_reference",
    "selective_state_update_one_token_reference",
    "ssd_chunk_cumsum_reference",
]

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


def _slot_at(slots: torch.Tensor, batch_idx: int, step: int) -> int:
    """Read a slot id from either the ``[batch]`` or ``[batch, width]`` form."""
    if slots.dim() == 1:
        return int(slots[batch_idx])
    return int(slots[batch_idx, step])


def selective_state_update_multi_token_reference(
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
    steps: Sequence[int],
    row_base: Sequence[int],
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference selective state update over several tokens per sequence.

    Covers both packed variable-length decode (``x`` is ``[total_tokens, heads,
    dim]`` and ``row_base[b]`` is ``cu_seqlens[b]``) and MTP decode (``x`` is the
    flattened ``[batch * steps, heads, dim]`` view of ``[batch, steps, heads,
    dim]`` with ``row_base[b] == b * steps``). The caller derives ``steps`` and
    ``row_base``; the oracle itself only needs the flattened layout.

    The state stays in registers across a sequence's tokens (a single read at the
    start), and the three write regimes follow the contract the consumers rely on:

    - plain packed varlen: one write after the sequence's token loop, to
      ``dst_slots[b, 0]`` (falling back to the read slot);
    - plain MTP without destination slots: one write after the token loop, back
      to the read slot;
    - speculative decoding (``num_accepted_tokens`` given): one write per token
      through the ``dst_slots[b, t]`` slot chain, where each written slot becomes
      the next token's read slot.

    A padded source slot reads as a zero state, a padded destination slot is never
    written, and ``disable_state_update`` suppresses every write. Empty sequences
    (``steps[b] == 0``) are skipped entirely, with no read and no write.

    One deliberate difference from the fork's reference: plain MTP that carries
    destination slots without ``num_accepted_tokens`` still writes its final state
    once, to ``dst_slots[b, 0]``, where the fork's ``elif`` chain happens to skip
    that write. No consumer configuration reaches that combination (the serving
    path passes destination slots only together with ``num_accepted_tokens``, which
    takes the slot-chain regime above).
    """
    heads, dim, dstate = state.shape[1:]
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

    batch = len(steps)
    total_rows = x.shape[0]
    y = torch.empty((total_rows, heads, dim), dtype=torch.float32, device=x.device)
    head_groups = torch.arange(heads, device=x.device) // head_ratio
    a_head = A.to(torch.float32).view(heads, 1, 1)
    spec_decoding = num_accepted_tokens is not None

    for b in range(batch):
        n_steps = int(steps[b])
        if n_steps == 0:
            continue
        first_row = int(row_base[b])
        accepted = max(int(num_accepted_tokens[b]) - 1, 0) if spec_decoding else 0
        read_slot = _slot_at(src_slots, b, accepted)
        running = (
            torch.zeros((heads, dim, dstate), dtype=torch.float32, device=x.device)
            if read_slot == pad_slot_id
            else state[read_slot].to(torch.float32)
        )

        for token in range(n_steps):
            row = first_row + token
            xb = x[row].to(torch.float32)
            dtb = dt_value[row]
            decay = torch.exp(a_head * dtb[:, :, None])
            b_h = B[row].to(torch.float32)[head_groups]
            # Same operand order as the kernel: (dt * B) * x.
            running = (
                running * decay + (dtb[:, :, None] * b_h[:, None, :]) * xb[:, :, None]
            )
            c_h = C[row].to(torch.float32)[head_groups]
            y[row] = (running * c_h[:, None, :]).sum(dim=-1)
            if D is not None:
                y[row] = y[row] + D.to(torch.float32).view(heads, 1) * xb
            if z is not None:
                zb = z[row].to(torch.float32)
                y[row] = y[row] * zb * torch.sigmoid(zb)

            if disable_state_update:
                continue
            if spec_decoding:
                write_slot = (
                    _slot_at(dst_slots, b, token)
                    if dst_slots is not None
                    else read_slot
                )
                if write_slot != pad_slot_id:
                    state[write_slot].copy_(running.to(state.dtype))
                # The slot just written is the next token's read slot.
                read_slot = write_slot

        if disable_state_update or spec_decoding:
            continue
        if dst_slots is not None:
            final_slot = _slot_at(dst_slots, b, 0)
        else:
            # Plain MTP without destination slots updates the read slot in place.
            final_slot = read_slot
        if final_slot != pad_slot_id:
            state[final_slot].copy_(running.to(state.dtype))

    return y.to(x.dtype)


def ssd_chunk_cumsum_reference(
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor | None,
    cu_chunk_seqlens: torch.Tensor,
    chunk_size: int,
    dt_softplus: bool,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference SSD chunk-local cumsum.

    Returns ``(dA_cumsum, dt_out)``, both head-major
    ``[heads, nchunks, chunk_size]`` fp32, where ``dt_out`` is the processed
    ``dt`` (bias, softplus, clamp) and ``dA_cumsum`` is the *inclusive* prefix
    sum of ``dt_out * A`` over each chunk's tokens.

    Chunks are scanned independently -- never across a sequence boundary -- and a
    chunk's padded row positions past its end are left untouched, mirroring the
    kernel and the production contract.
    """
    if dt.dim() != 2:
        raise ValueError("dt must be [tokens, heads].")
    heads = dt.shape[1]
    offsets = [int(x) for x in cu_chunk_seqlens.reshape(-1).tolist()]
    nchunks = len(offsets) - 1
    if nchunks < 1:
        raise ValueError("cu_chunk_seqlens must have at least two entries.")

    dt_min, dt_max = dt_limit
    dA_cumsum = torch.zeros(
        (heads, nchunks, chunk_size), dtype=torch.float32, device=dt.device
    )
    dt_out = torch.zeros_like(dA_cumsum)

    a_value = A.to(torch.float32)
    bias = None if dt_bias is None else dt_bias.to(torch.float32)

    for chunk in range(nchunks):
        lo, hi = offsets[chunk], offsets[chunk + 1]
        length = hi - lo
        if length < 0 or hi > dt.shape[0]:
            raise ValueError(
                f"chunk {chunk} spans [{lo}, {hi}) outside 0..{dt.shape[0]}."
            )
        if length == 0:
            continue
        segment = dt[lo:hi].to(torch.float32)
        value = segment if bias is None else segment + bias.view(1, heads)
        if dt_softplus:
            value = torch.where(
                value > _SOFTPLUS_THRESHOLD,
                value,
                torch.log1p(torch.exp(value)),
            )
        value = value.clamp(min=dt_min, max=dt_max)
        prefix = torch.cumsum(value * a_value.view(1, heads), dim=0)
        dA_cumsum[:, chunk, :length] = prefix.transpose(0, 1)
        dt_out[:, chunk, :length] = value.transpose(0, 1)

    return dA_cumsum, dt_out
