"""Selective state update, FlashInfer-compatible on MUSA.

The signature mirrors upstream FlashInfer's ``selective_state_update`` exactly,
including argument order and defaults, because vLLM's SSU dispatch calls it with
keyword arguments and probes it for capability flags. MUSA-specific selection
(native kernel versus portability fallback, reduction splits, Philox mode) is a
MATE implementation concern or an environment switch; it must not appear as a new
parameter on this name.
"""

from __future__ import annotations

from typing import Any

from . import _backend

__all__ = ["selective_state_update"]

#: vLLM's sentinel for "this state row belongs to no block"
#: (``vllm.v1.attention.backends.utils.NULL_BLOCK_ID``).
NULL_BLOCK_ID = 0


import torch


def _materialized(tensor: Any) -> Any:
    """Return a contiguous tensor, copying only when vLLM passed a view.

    vLLM expands `dt_d` across head_dim, so it arrives as a zero-stride view that
    MATE's kernel refuses. The copy is tokens x heads x head_dim elements -- a few
    kilobytes per mixer per step -- and it is captured with the graph, so it does
    not reallocate at replay.
    """
    if tensor is None or not hasattr(tensor, "is_contiguous"):
        return tensor
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def _flat_slots(indices: Any, batch: int) -> Any:
    """Return the slot vector MATE expects, or leave it for MATE to refuse.

    vLLM hands over `state_indices_tensor_d[:num_decode_tokens]`, which arrives
    shaped [rows, width]. Flattening is only correct when the tensor carries one
    slot per row; a speculative call carries one slot per accepted token, and
    guessing there would pair rows with the wrong state. Those calls keep their
    shape and MATE refuses them explicitly.
    """
    if indices is None or batch is None or not hasattr(indices, "dim"):
        return indices
    if indices.dim() > 1 and indices.numel() == batch:
        return indices.reshape(-1)
    return indices


_FP32_VECTOR_CACHE: dict[tuple, Any] = {}
_FP32_VECTOR_CACHE_LIMIT = 256


def _per_head_fp32(tensor: Any) -> Any:
    """Return MATE's per-head fp32 view of a vLLM per-head vector.

    D and dt_bias are loaded model parameters, so the widened copy is made once
    per tensor and reused. Widening bf16/fp16 to fp32 is exact; doing it on every
    decode step instead put one cast launch per mixer per step on the critical
    path, which is measurable at 52 mixers.
    """
    if tensor is None or not hasattr(tensor, "dtype"):
        # Forwarding tests and non-tensor sentinels pass through untouched; MATE
        # validates whatever it actually receives.
        return tensor
    if tensor.dim() >= 2 and all(stride == 0 for stride in tensor.stride()[1:]):
        tensor = tensor[:, 0]
    if tensor.dtype == torch.float32:
        return tensor
    key = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )
    cached = _FP32_VECTOR_CACHE.get(key)
    if cached is None:
        cached = tensor.float().contiguous()
        if len(_FP32_VECTOR_CACHE) < _FP32_VECTOR_CACHE_LIMIT:
            _FP32_VECTOR_CACHE[key] = cached
    return cached


def _dt_input(tensor: Any) -> Any:
    """Return dt in the form MATE reads without a copy.

    vLLM expands dt across head_dim, so it arrives as a zero-stride view in the
    model dtype. MATE reads one step value per head and widens it inside the
    kernel, so the broadcast's base -- a contiguous [batch, heads] buffer -- is
    handed over as [batch, heads, 1]. The values are identical by construction,
    and the decode path loses a per-layer cast launch.
    """
    if tensor is None or not hasattr(tensor, "stride"):
        return tensor
    if tensor.dim() == 3 and tensor.stride(-1) == 0:
        return tensor.select(-1, 0).unsqueeze(-1)
    return tensor


def selective_state_update(
    state: Any,
    x: Any,
    dt: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    z: Any = None,
    dt_bias: Any = None,
    dt_softplus: bool = False,
    state_batch_indices: Any = None,
    pad_slot_id: int = -1,
    state_scale: Any = None,
    out: Any = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Any = None,
    intermediate_state_indices: Any = None,
    intermediate_state_scales: Any = None,
    rand_seed: Any = None,
    philox_rounds: int = 10,
    cache_steps: int = 0,
    algorithm: str = "auto",
    dst_state_batch_indices: Any = None,
    cu_seqlens: Any = None,
    num_accepted_tokens: Any = None,
    backend: str = "auto",
    *,
    null_block_id: int = NULL_BLOCK_ID,
    is_blackwell: bool = False,
    enable_stochastic_rounding: bool = False,
    cache_philox_rounds: int | None = None,
) -> Any:
    """Update the Mamba SSM state for one decode step.

    The first block of parameters is MATE's own order and is forwarded
    positionally. The keyword-only block after it exists for vLLM's backend
    dispatch, which is the only production caller and passes these four by
    keyword: without them the call raises ``TypeError`` before any kernel runs,
    which is how an otherwise-correct MATE SSU stayed unreachable behind
    ``--mamba-backend flashinfer``.

    ``is_blackwell`` and a ``null_block_id`` other than the sentinel are refused
    rather than ignored: MATE has no Blackwell path, and a non-sentinel null
    block would change which state rows a kernel may skip, so accepting it
    silently would produce wrong states instead of an error.
    """
    if is_blackwell:
        raise NotImplementedError(
            "is_blackwell=True selects a Blackwell-only SSU path; this is the MUSA "
            "surface and MATE implements no such path."
        )
    if null_block_id != NULL_BLOCK_ID:
        raise NotImplementedError(
            f"null_block_id={null_block_id} asks the SSU to treat that block as "
            "carrying no state, which MATE's selective_state_update has no concept "
            "of; only the sentinel is accepted."
        )
    if enable_stochastic_rounding and rand_seed is None:
        raise NotImplementedError(
            "enable_stochastic_rounding=True needs a seed: MATE takes stochastic "
            "rounding from rand_seed, and this call supplied none."
        )
    implementation = _backend.resolve("selective_state_update")
    # vLLM builds its per-head vectors as `self.D[:, None].expand(-1, head_dim)`:
    # a zero-stride 2-D broadcast in the model dtype. MATE's native kernel takes
    # one fp32 value per head, and widening bf16/fp16 to fp32 is exact, so the
    # adapter hands over the broadcast's first column instead of materializing a
    # [heads, head_dim] copy. Refusing bf16 D here made the flag fail closed for a
    # reason the caller could not act on: vLLM has no fp32 D to pass.
    D = _per_head_fp32(D)
    dt_bias = _per_head_fp32(dt_bias)
    x = _materialized(x)
    dt = _dt_input(dt)
    # x can be a non-tensor sentinel on the pure-forwarding path, in which case
    # there is no batch to compare against and the indices pass through.
    batch = x.shape[0] if hasattr(x, "shape") else None
    state_batch_indices = _flat_slots(state_batch_indices, batch)
    dst_state_batch_indices = _flat_slots(dst_state_batch_indices, batch)
    return implementation(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        state_scale,
        out,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        intermediate_state_scales,
        rand_seed,
        cache_philox_rounds if cache_philox_rounds is not None else philox_rounds,
        cache_steps,
        algorithm,
        dst_state_batch_indices,
        cu_seqlens,
        num_accepted_tokens,
        backend,
    )
