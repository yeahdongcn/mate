"""Variable-length SSD combined forward, FlashInfer-compatible on MUSA.

``ssd_combined_fwd_varlen`` is the MUSA entry point that vLLM's Mamba2/SSD
implementation calls for packed variable-length prefill. Its parameter list is a
frozen contract: consumers pass the first twenty-two arguments positionally, so
the names, order, and arity must not change. ``mamba_chunk_scan_combined_varlen``
is the historical alias and must stay an alias of the same object rather than a
second definition.

Upstream FlashInfer's ``ssd_combined_fwd`` (the dense, CUTLASS/CuTe-backed
entry point) is intentionally not provided here; this wrapper covers the
packed-varlen path the MUSA consumers use. MUSA-specific tuning knobs are not
added as parameters, because the surface has to stay upstream-shaped; they
belong to the MATE implementation or its environment.
"""

from __future__ import annotations

from typing import Any

from . import _backend

__all__ = [
    "mamba_chunk_scan_combined_varlen",
    "ssd_combined_fwd_varlen",
]


def ssd_combined_fwd_varlen(
    x: Any,
    dt: Any,
    A: Any,
    B: Any,
    C: Any,
    chunk_size: int,
    cu_seqlens: Any,
    cu_chunk_seqlens: Any,
    last_chunk_indices: Any,
    seq_idx: Any,
    out: Any,
    D: Any = None,
    z: Any = None,
    dt_bias: Any = None,
    initial_states: Any = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_intermediate_states: bool = False,
    state_dtype: Any = None,
    checkpoint_token_indices: Any = None,
    checkpoint_state_slots: Any = None,
    checkpoint_states: Any = None,
) -> Any:
    """Run the combined SSD forward pass for packed variable-length sequences.

    Arguments are forwarded positionally in the frozen order above; see the
    MATE mamba family for the shape and dtype contract of each argument.
    """
    implementation = _backend.resolve("ssd_combined_fwd_varlen")
    return implementation(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        cu_seqlens,
        cu_chunk_seqlens,
        last_chunk_indices,
        seq_idx,
        out,
        D,
        z,
        dt_bias,
        initial_states,
        dt_softplus,
        dt_limit,
        return_intermediate_states,
        state_dtype,
        checkpoint_token_indices,
        checkpoint_state_slots,
        checkpoint_states,
    )


# Historical name kept as a true alias: downstream code asserts identity, not
# just equal behavior.
mamba_chunk_scan_combined_varlen = ssd_combined_fwd_varlen
