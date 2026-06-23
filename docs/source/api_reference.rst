Python APIs
===========

This section covers the direct MATE Python APIs.
Use these interfaces only when your framework or model architecture requires
direct symbol-level integration.

.. note::

   Check wrapper availability first. The wrapper-first workflow preserves more
   of the upstream package behavior and usually requires less code change.

Use direct MATE Python APIs when:

- no wrapper matches your framework's package surface
- a wrapper exists but does not cover the feature you need
- you need symbol-level control over a specific operator path

Supported API Entrypoints
-------------------------

Attention
---------

Optimized entrypoints for FlashAttention, varlen, KV-cache, and MLA-related :doc:`attention <api/attention>` paths.

- ``mate.flash_attn_varlen_func``
- ``mate.flash_attn_with_kvcache``
- ``mate.get_mla_metadata``
- ``mate.flash_mla_with_kvcache``

GEMM
----

Low-precision :doc:`GEMM <api/gemm>` entrypoints, including batched FP8 and groupwise GEMM
paths.

- ``mate.gemm.bmm_fp8``
- ``mate.gemm.gemm_fp8_nt_groupwise``

Hyperconnection
---------------

Direct :doc:`Hyperconnection <api/hyperconnection>` APIs exposed through ``mate.hyperconnection``.

KDA
---

Direct :doc:`KDA <api/kda>` entrypoints for fused chunked KDA when the
``flash_kda`` wrapper is not the right integration surface.

- ``mate.kda.chunk_kda``
