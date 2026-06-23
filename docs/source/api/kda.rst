.. _apikda:

KDA
===

.. currentmodule:: mate.kda

For framework integrations that already target FlashKDA Python APIs, prefer
the ``flash_kda`` wrapper package first. Use the MATE API below when wrapper
coverage is not enough or when direct operator-level control is required.

``chunk_kda`` is MATE's fused chunked KDA operator on MUSA.

At a glance
-----------

- Public API: ``mate.chunk_kda`` / ``mate.kda.chunk_kda``
- Device: MUSA
- Input dtypes: ``torch.float16`` and ``torch.bfloat16``
- Head dimension: currently fixed to ``128``
- Sequence modes:
  - Dense: ``[B, T, H, 128]``
  - Varlen: ``[S, H, 128]`` or ``[1, S, H, 128]`` with ``cu_seqlens``
- Optional recurrent state input/output
- Optional preallocated ``output`` and ``final_state``

Shape contract
--------------

For dense mode:

- ``q`` / ``k``: ``[B, T, Hqk, 128]``
- ``v`` / ``g``: ``[B, T, Hv, 128]``
- ``beta``: ``[B, T, Hv]``

For varlen mode:

- ``q`` / ``k``: ``[S, Hqk, 128]`` or ``[1, S, Hqk, 128]``
- ``v`` / ``g``: ``[S, Hv, 128]`` or ``[1, S, Hv, 128]``
- ``beta``: ``[S, Hv]`` or ``[1, S, Hv]``
- ``cu_seqlens``: cumulative sequence lengths with shape ``[num_seqs + 1]``

Additional constraints:

- ``k.shape == q.shape``
- ``g.shape == v.shape``
- ``beta.shape == v.shape[:3]``
- ``Hv`` must be divisible by ``Hqk``
- the last dimension of every tensor passed to the kernel must be contiguous

State tensors
-------------

- ``initial_state`` is optional
- ``final_state`` is optional unless ``output_final_state=True``
- state shape: ``[num_seqs, Hv, 128, 128]``
- state dtype: same as value dtype or ``torch.float32``
- if both ``initial_state`` and ``final_state`` are provided, their dtypes must match

Gate parameters
---------------

When gate parameters are enabled:

- ``A_log`` and ``dt_bias`` must be provided together
- ``A_log`` shape: ``[Hv]``
- ``dt_bias`` shape: ``[Hv, 128]``
- ``lower_bound`` defaults to ``-5.0``

Outputs
-------

- default return: ``output``
- if ``output_final_state=True``: returns ``(output, final_state)``

API reference
-------------

.. autofunction:: chunk_kda
