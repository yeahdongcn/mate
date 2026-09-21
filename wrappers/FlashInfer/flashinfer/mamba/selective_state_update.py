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
