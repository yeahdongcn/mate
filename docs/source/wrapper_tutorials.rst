Wrappers
========

MATE uses the packages in the ``wrappers/`` directory as a compatibility
layer to run CUDA software on MUSA. They preserve familiar package names and
high-level APIs while routing execution to MATE operators and kernels. This
enables existing integrations to migrate to Moore Threads platforms with
minimal code changes.

Wrappers are the default integration path when your framework already targets
a supported CUDA-oriented Python package. Install MATE first, then choose the
wrapper that matches your upstream import path.

How the wrappers work
---------------------

Each wrapper keeps the upstream-facing Python package surface stable while
routing supported execution paths to MATE-backed implementations on MUSA.

Key mechanisms
~~~~~~~~~~~~~~

- API mapping: Maps upstream-style calls to MATE operator paths.
- Namespace preservation: Preserves expected package names and import paths.
- Kernel routing: Runs calls on MATE-optimized operators and MUSA kernels.

Why use wrappers
~~~~~~~~~~~~~~~~

- Lower migration overhead: Minimizes code changes and avoids separate
  hardware-specific code paths.
- Faster integration: Accelerates deployment of common tools and libraries on
  Moore Threads GPUs.

Wrapper support at a glance
---------------------------

Select a wrapper package to open its documentation page.

.. list-table::
   :header-rows: 1

   * - Wrapper package
     - Import path
     - Best fit
     - Current scope
   * - :doc:`flash_attn_3 <wrappers/flash_attention_wrapper>`
     - ``flash_attn_interface``
     - FlashAttention-3 style APIs
     - Dense FMHA, varlen FMHA, KV-cache attention, scheduler metadata
   * - :doc:`sageattention <wrappers/sageattention_wrapper>`
     - ``sageattention``
     - SageAttention style APIs
     - Dense SageAttention-compatible path
   * - :doc:`flash_mla <wrappers/flash_mla_wrapper>`
     - ``flash_mla``
     - FlashMLA style APIs
     - MLA metadata, decode, sparse prefill
   * - :doc:`flash_kda <wrappers/flash_kda_wrapper>`
     - ``flash_kda``
     - FlashKDA style APIs
     - KDA forward, workspace-size compatibility helper
   * - :doc:`deep-gemm <wrappers/deep_gemm_wrapper>`
     - ``deep_gemm``
     - DeepGEMM style APIs
     - Grouped GEMM, dense GEMM, prenorm GEMM, MQA logits
