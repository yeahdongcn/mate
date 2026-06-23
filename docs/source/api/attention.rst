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
- FP8 inputs produce ``torch.bfloat16`` outputs by default, and FP8 with
  ``qv`` is not part of the current supported scope.

.. autofunction:: flash_attn_with_kvcache
.. autofunction:: flash_attn_varlen_func


MLA
---

.. currentmodule:: mate.flashmla

.. autofunction:: get_mla_metadata
.. autofunction:: flash_mla_with_kvcache
