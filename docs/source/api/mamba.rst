.. _apimamba:

Mamba / SSU
===========

.. currentmodule:: mate.mamba

Mamba2 selective state update (SSU) on MUSA. ``selective_state_update`` applies
one decode step of the SSM recurrence for a single token per sequence, and is the
MUSA-native implementation behind the FlashInfer-shaped
``flashinfer.mamba.selective_state_update`` compatibility surface.

Minimal SSU example:

.. code-block:: python

   import torch
   from mate.mamba import selective_state_update

   batch, heads, dim, dstate, groups, slots = 2, 8, 8, 128, 2, 8
   state = torch.zeros(slots, heads, dim, dstate, device="musa", dtype=torch.float16)
   x = torch.randn(batch, heads, dim, device="musa", dtype=torch.bfloat16)
   dt = torch.rand(batch, heads, dim, device="musa", dtype=torch.float32) * 0.1
   A = -torch.rand(heads, device="musa", dtype=torch.float32)
   B = torch.randn(batch, groups, dstate, device="musa", dtype=torch.bfloat16)
   C = torch.randn(batch, groups, dstate, device="musa", dtype=torch.bfloat16)
   slots_index = torch.tensor([0, 1], device="musa", dtype=torch.int32)

   y = selective_state_update(state, x, dt, A, B, C, None)
   y = selective_state_update(
       state,
       x,
       dt,
       A,
       B,
       C,
       None,
       state_batch_indices=slots_index,
       dst_state_batch_indices=slots_index,
   )

SSU at a glance
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Capability
     - Supported scope
   * - Device
     - MUSA through the TileLang backend
   * - Sequence layout
     - One token per sequence: ``x``, ``dt`` and ``z`` are
       ``[batch, heads, dim]``
   * - State
     - ``[slots, heads, dim, dstate]``, contiguous, fp16/bf16/fp32, updated in
       place through the destination slots
   * - ``A`` / ``D`` / ``dt_bias``
     - per-head fp32, broadcasting over ``dim``/``dstate``
   * - ``B`` / ``C``
     - ``[batch, groups, dstate]`` with the ``x`` dtype; ``heads`` must be
       divisible by ``groups``
   * - Output
     - ``[batch, heads, dim]`` in the ``x`` dtype, optionally written into a
       caller-provided ``out``
   * - ``dt_softplus``
     - supported
   * - Slot semantics
     - ``pad_slot_id`` reads as a zero state and is never written;
       ``disable_state_update`` leaves the pool unchanged
   * - Stochastic rounding
     - not implemented; ``rand_seed`` raises ``NotImplementedError``
   * - Variable-length and MTP decode
     - not implemented; ``cu_seqlens`` and ``num_accepted_tokens`` raise
       ``NotImplementedError``

SSU toolchain requirements
--------------------------

- TileLang for MUSA (``tilelang-musa``) is required; the kernel is compiled at
  run time for the shape and dtype combination it is first called with.
- The recurrence runs in fp32 with the state updated in the state dtype.

Graph capture
-------------

Call ``prewarm_selective_state_update`` during warmup, before any graph capture:
the TileLang kernel is compiled on first use, and a first-call compile inside a
captured graph stalls the worker. The operator itself issues a single kernel
launch and performs no host/device scalar copies.

SSD packed prefill
-------------------

``ssd_combined_fwd_varlen`` is the prefill-side counterpart of the decode
entrypoint: one packed call over a variable-length batch, running the five SSD
stages (chunk-local dt/decay cumsum, intra-chunk states, state passing, the
``C·Bᵀ`` products, and the chunk scan) as five TileLang launches. It is the
symbol the FlashInfer-shaped compatibility wrapper dispatches to for
``--mamba-backend flashinfer`` on MUSA, and it drives the same implementation for
every Mamba2/Nemotron-H prefill shape.

The per-shape intermediates come from a cached workspace, so a steady-state call
allocates nothing beyond the returned state tensor. ``cu_chunk_seqlens`` is the
only authority for the chunk-to-token mapping, and ``seq_idx`` -- not the
``last_chunk_indices`` arithmetic -- decides which chunks open a sequence and
therefore where their entering state comes from.

Everything outside the supported contract raises ``NotImplementedError`` or
``ValueError`` naming the missing capability rather than silently narrowing
semantics: the ``checkpoint_*`` replay arguments are not implemented.

API reference
-------------

.. autofunction:: selective_state_update

.. autofunction:: prewarm_selective_state_update

.. autofunction:: ssd_combined_fwd_varlen

.. autofunction:: prewarm_ssd_combined_fwd_varlen
