"""Native MUSA tilelang kernel for the one-token selective state update (SSU).

This is the S5000 decode recurrence behind ``mate.mamba.selective_state_update``
for the shapes the Mamba2/Nemotron consumers use: state ``[slots, heads, dim,
dstate]``, one token per sequence, and per-head ``A``, ``D`` and ``dt_bias``
broadcasting over ``dim``/``dstate``.

Dataflow (one CTA owns ``rows_per_cta`` consecutive ``dim`` rows of one head):
each lane owns ``dstate / lanes_per_row`` state values, stages ``B``/``C`` once
per head, and keeps the whole recurrence in fp32. The per-row output reduction is
a warp shuffle tree, so no shared-memory staging or cross-warp synchronization is
involved.

Optional inputs are selected by runtime flags rather than compile-time
specialization, so one compiled kernel covers every flag combination and
prewarming stays meaningful. The ``pad_slot_id``/``disable_state_update``
contract is honoured in kernel: a padded source reads as zero state and a padded
or disabled destination is not written.
"""

import tilelang
import tilelang.language as T
import torch

__all__ = ["prewarm_ssu_one_token", "ssu_one_token_launch"]

SOFTPLUS_THRESHOLD = 20.0

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}

_COMPILE_FLAGS = [
    "-O3",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


@tilelang.jit(pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS)
def tilelang_ssu_one_token(
    state_dtype,
    io_dtype,
    slot_dtype,
    dim,
    dstate,
    head_ratio,
    rows_per_cta,
    lanes_per_row,
    dt_dtype=None,
    dt_dim=None,
):
    """Build the one-token SSU kernel for one (dtype, shape, config) tuple.

    ``dt_dim`` is ``1`` when the caller passes the per-head step instead of a
    [batch, heads, dim] tensor. vLLM expands it across head_dim, and the two
    forms carry identical values, so reading one element per head lets the caller
    hand over the unexpanded buffer and skip the copy that a widening cast on the
    decode path would otherwise cost per mixer per step.
    """
    if dt_dtype is None:
        dt_dtype = torch.float32
    if dt_dim is None:
        dt_dim = dim
    batch = T.dynamic("batch")
    heads = T.dynamic("heads")
    groups = T.dynamic("groups")
    slots = T.dynamic("slots")

    vec = tilelang.cdiv(dstate, lanes_per_row)
    num_shuffles = lanes_per_row.bit_length() - 1
    threads = rows_per_cta * lanes_per_row
    dim_blocks = tilelang.cdiv(dim, rows_per_cta)

    @T.prim_func
    def tilelang_ssu_one_token_kernel(
        state: T.Tensor((slots, heads, dim, dstate), state_dtype),
        x: T.Tensor((batch, heads, dim), io_dtype),
        dt: T.Tensor((batch, heads, dt_dim), dt_dtype),
        A: T.Tensor((heads,), "float32"),
        B: T.Tensor((batch, groups, dstate), io_dtype),
        C: T.Tensor((batch, groups, dstate), io_dtype),
        Dv: T.Tensor((heads,), "float32"),
        dt_bias: T.Tensor((heads,), "float32"),
        z: T.Tensor((batch, heads, dim), io_dtype),
        src_slots: T.Tensor((batch,), slot_dtype),
        dst_slots: T.Tensor((batch,), slot_dtype),
        out: T.Tensor((batch, heads, dim), io_dtype),
        pad_slot_id: T.int32,
        use_dt_bias: T.int32,
        use_z: T.int32,
        has_D: T.int32,
        dt_softplus: T.int32,
        write_state: T.int32,
    ):
        with T.Kernel(dim_blocks, heads, batch, threads=threads) as (
            d_block,
            head,
            batch_idx,
        ):
            tid = T.get_thread_binding()
            lane = tid % lanes_per_row
            row_in_block = tid // lanes_per_row
            d = d_block * rows_per_cta + row_in_block
            group = head // head_ratio

            state_row = T.alloc_local((vec,), dtype="float32")
            b_val = T.alloc_local((vec,), dtype="float32")
            c_val = T.alloc_local((vec,), dtype="float32")
            out_acc = T.alloc_local((1,), dtype="float32")
            dt_value = T.alloc_var("float32")
            decay = T.alloc_var("float32")
            x_value = T.alloc_var("float32")
            src = T.alloc_var("int32")
            dst = T.alloc_var("int32")

            if d < dim:
                src = T.cast(src_slots[batch_idx], "int32")
                dst = T.cast(dst_slots[batch_idx], "int32")

                # dt -> decay, shared by every dstate lane of this row.
                # dt_dim == 1 is the broadcast form: every d of this head reads
                # the same value, so the unexpanded buffer can be used as-is.
                dt_value = T.cast(dt[batch_idx, head, 0 if dt_dim == 1 else d], "float32")
                if use_dt_bias == 1:
                    dt_value = dt_value + dt_bias[head]
                if dt_softplus == 1:
                    dt_value = T.if_then_else(
                        dt_value > SOFTPLUS_THRESHOLD,
                        dt_value,
                        T.log(1.0 + T.exp(dt_value)),
                    )
                decay = T.exp(A[head] * dt_value)
                x_value = T.cast(x[batch_idx, head, d], "float32")

                # B/C are shared by every (dim, dstate) element of this head.
                for i in T.vectorized(vec):
                    b_val[i] = T.cast(B[batch_idx, group, lane * vec + i], "float32")
                    c_val[i] = T.cast(C[batch_idx, group, lane * vec + i], "float32")

                # A padded source slot has no state: branch around the load instead
                # of relying on a select, so no out-of-range address is formed and
                # the load stays vectorized.
                if src == pad_slot_id:
                    for i in T.vectorized(vec):
                        state_row[i] = 0.0
                else:
                    for i in T.vectorized(vec):
                        state_row[i] = T.cast(
                            state[src, head, d, lane * vec + i], "float32"
                        )

                out_acc[0] = 0.0
                for i in T.unroll(vec):
                    state_row[i] = state_row[i] * decay + dt_value * b_val[i] * x_value
                    out_acc[0] += state_row[i] * c_val[i]

                for offset in T.unroll(num_shuffles):
                    out_acc[0] += T.shfl_xor(out_acc[0], (lanes_per_row // 2) >> offset)

                if write_state == 1 and dst != pad_slot_id:
                    for i in T.vectorized(vec):
                        state[dst, head, d, lane * vec + i] = T.cast(
                            state_row[i], state_dtype
                        )

                if lane == 0:
                    if has_D == 1:
                        out_acc[0] += Dv[head] * x_value
                    if use_z == 1:
                        out_acc[0] = (
                            out_acc[0]
                            * T.cast(z[batch_idx, head, d], "float32")
                            * T.sigmoid(T.cast(z[batch_idx, head, d], "float32"))
                        )
                    out[batch_idx, head, d] = T.cast(out_acc[0], io_dtype)

    return tilelang_ssu_one_token_kernel


def _validate_launch(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    dstate: int,
    rows_per_cta: int,
    lanes_per_row: int,
) -> None:
    if dstate % lanes_per_row != 0:
        raise RuntimeError(
            f"dstate {dstate} must be divisible by lanes_per_row {lanes_per_row}."
        )
    if lanes_per_row != 32:
        raise RuntimeError(
            "the native SSU kernel assigns one dim row per warp so that the "
            "`d < dim` guard stays warp-uniform while the lane group shuffles; "
            f"lanes_per_row must be 32, got {lanes_per_row}. A sub-warp variant "
            "must hoist the reduction out of the guard first."
        )
    if rows_per_cta < 1 or rows_per_cta * lanes_per_row > 1024:
        raise RuntimeError(
            "the native SSU kernel requires 1 <= rows_per_cta * lanes_per_row <= 1024."
        )
    if not x.is_contiguous() or not dt.is_contiguous():
        raise RuntimeError("x and dt must be contiguous for the native SSU kernel.")
    if state.stride() != (
        state.shape[1] * state.shape[2] * dstate,
        state.shape[2] * dstate,
        dstate,
        1,
    ):
        raise RuntimeError(
            "state must be a contiguous [slots, heads, dim, dstate] pool."
        )
    if dt.shape != (x.shape[0], x.shape[1], dt.shape[2]) or dt.shape[2] not in (
        1,
        x.shape[2],
    ):
        raise RuntimeError(
            "dt must be [batch, heads, dim] or the per-head broadcast [batch, heads, 1]; "
            f"got {tuple(dt.shape)} for x={tuple(x.shape)}."
        )
    if dt.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise RuntimeError(f"dt must be fp32, bf16 or fp16; got {dt.dtype}.")
    if A.dtype != torch.float32:
        raise RuntimeError("A must be fp32 for the native SSU kernel.")
    if B.shape[1] == 0 or state.shape[1] % B.shape[1] != 0:
        raise RuntimeError("B/C groups must divide heads for the native SSU kernel.")


def _kernel_for(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    src_slots: torch.Tensor,
    *,
    rows_per_cta: int,
    lanes_per_row: int,
):
    return tilelang_ssu_one_token(
        state_dtype=state.dtype,
        io_dtype=x.dtype,
        slot_dtype=src_slots.dtype,
        dim=state.shape[2],
        dstate=state.shape[3],
        head_ratio=state.shape[1] // B.shape[1],
        rows_per_cta=rows_per_cta,
        lanes_per_row=lanes_per_row,
        dt_dtype=dt.dtype,
        dt_dim=dt.shape[2],
    )


def prewarm_ssu_one_token(
    *,
    state_dtype: torch.dtype,
    io_dtype: torch.dtype,
    slot_dtype: torch.dtype,
    dim: int,
    dstate: int,
    head_ratio: int,
    rows_per_cta: int = 4,
    lanes_per_row: int = 32,
    dt_dtype: torch.dtype = torch.float32,
    dt_dim: int | None = None,
) -> None:
    """Compile one kernel configuration without launching it."""
    tilelang_ssu_one_token(
        state_dtype=state_dtype,
        io_dtype=io_dtype,
        slot_dtype=slot_dtype,
        dim=dim,
        dstate=dstate,
        head_ratio=head_ratio,
        rows_per_cta=rows_per_cta,
        lanes_per_row=lanes_per_row,
        dt_dtype=dt_dtype,
        dt_dim=dt_dim,
    )


_DUMMY_CACHE: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor] = {}


def _dummy(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return an unread placeholder for an absent optional input.

    The kernel never dereferences an optional input whose flag is zero, so one
    cached buffer per (shape, dtype, device) suffices. Caching keeps the op at a
    single kernel launch per call and avoids a fresh allocation inside a captured
    graph.
    """
    key = (shape, dtype, str(device))
    tensor = _DUMMY_CACHE.get(key)
    if tensor is None:
        tensor = torch.empty(shape, device=device, dtype=dtype)
        _DUMMY_CACHE[key] = tensor
    return tensor


def ssu_one_token_launch(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    Dv: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    z: torch.Tensor | None,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor | None,
    out: torch.Tensor,
    *,
    dt_softplus: bool,
    pad_slot_id: int,
    disable_state_update: bool,
    rows_per_cta: int = 4,
    lanes_per_row: int = 32,
) -> torch.Tensor:
    """Run the native one-token SSU kernel.

    ``mate.mamba.selective_state_update`` has already validated the contract:
    single-token inputs, per-head (tied) ``A``/``D``/``dt_bias`` in fp32,
    contiguous slots and outputs.
    """
    batch, heads, dim = x.shape
    slots, state_heads, state_dim, dstate = state.shape
    if state_heads != heads or state_dim != dim:
        raise RuntimeError("state must match x as [slots, heads, dim, dstate].")
    if dst_slots is not None and dst_slots.dtype != src_slots.dtype:
        raise RuntimeError("source and destination slots must share a dtype.")
    if any(
        tensor is not None and not tensor.is_contiguous()
        for tensor in (B, C, Dv, dt_bias, z, src_slots, dst_slots)
    ):
        raise RuntimeError(
            "B, C, D, dt_bias, z and the slot tensors must be contiguous."
        )
    _validate_launch(state, x, dt, A, B, dstate, rows_per_cta, lanes_per_row)

    kernel = _kernel_for(
        state,
        x,
        dt,
        B,
        src_slots,
        rows_per_cta=rows_per_cta,
        lanes_per_row=lanes_per_row,
    )

    kernel(
        state,
        x,
        dt,
        A,
        B,
        C,
        Dv if Dv is not None else _dummy((heads,), torch.float32, x.device),
        dt_bias if dt_bias is not None else _dummy((heads,), torch.float32, x.device),
        z if z is not None else _dummy((batch, heads, dim), x.dtype, x.device),
        src_slots,
        dst_slots if dst_slots is not None else src_slots,
        out,
        int(pad_slot_id),
        1 if dt_bias is not None else 0,
        1 if z is not None else 0,
        1 if Dv is not None else 0,
        1 if dt_softplus else 0,
        0 if disable_state_update else 1,
    )
    return out
