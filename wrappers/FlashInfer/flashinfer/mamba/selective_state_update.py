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
) -> Any:
    """Update the Mamba SSM state for one decode step.

    Arguments are forwarded positionally to the MATE implementation in the
    upstream FlashInfer order.
    """
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
        philox_rounds,
        cache_steps,
        algorithm,
        dst_state_batch_indices,
        cu_seqlens,
        num_accepted_tokens,
        backend,
    )
