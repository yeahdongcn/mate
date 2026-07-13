.. _apiattention:

Attention
=========

For framework integrations that already target FlashAttention-3 or FlashMLA
Python APIs, prefer the ``flash_attn_3`` or ``flash_mla`` wrapper packages
first. Use the MATE APIs below when wrapper coverage is not enough.

FMHA
----

.. currentmodule:: mate.mha_interface

Forward dtype notes:

- ``flash_attn_varlen_func`` and ``flash_attn_with_kvcache`` accept
  ``torch.float16``, ``torch.bfloat16``, and ``torch.float8_e4m3fn`` inputs on
  the MATE FMHA forward path.
- For FP8 inputs, optional ``q_descale``, ``k_descale``, and ``v_descale``
  tensors are ``torch.float32`` scale factors with shape
  ``(batch_size, num_heads_kv)``.
- FP8 inputs produce ``torch.bfloat16`` outputs by default.
- The FMHA forward path also supports an optional ``qv`` input, including FP8
  inputs with ``qv``.
- When both ``q`` and ``qv`` are FP8, ``q_descale`` applies to both query
  tensors.
- For best FP8 attention performance, use MUSA SDK 5.2.0 or newer when
  available.
- Use ``flash_attn_combine`` to merge partial FMHA outputs and log-sum-exp
  buffers when your integration splits the attention computation.
- Use ``get_scheduler_metadata`` to precompute the ``scheduler_metadata`` input
  for ``flash_attn_with_kvcache`` when you call the FMHA path directly.

.. autofunction:: flash_attn_combine
.. autofunction:: flash_attn_with_kvcache
.. autofunction:: get_scheduler_metadata
.. autofunction:: flash_attn_varlen_func


MLA
---

.. currentmodule:: mate.flashmla

.. autofunction:: get_mla_metadata
.. autofunction:: flash_mla_with_kvcache
