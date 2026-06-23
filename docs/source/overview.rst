Overview
========

MUSA AI Tensor Engine (MATE) accelerates generative AI workloads on Moore
Threads GPUs by providing high-performance operator implementations,
especially Attention and GEMM, along with compatibility wrappers for selected
CUDA-oriented Python operator interfaces.

For a deeper explanation of how MATE is structured, see
:doc:`Design and Architecture <design_and_architecture>`.

Key Principles
--------------

- Wrapper-first: when a wrapper matches your existing package surface, keep
  the upstream import path and high-level API shape as stable as possible.
- Direct API fallback: use MATE Python APIs when no wrapper matches your
  workload or wrapper coverage is insufficient.
- Diagnostics early: verify the runtime with ``mate check``,
  ``mate show-config``, and ``mate env`` before debugging deeper failures.

Key Goals
---------

- Run high-performance generative AI workloads on Moore Threads GPUs with
  optimized Attention and GEMM operators.
- Reduce migration work for CUDA-oriented integrations by preserving familiar
  package surfaces when wrapper coverage exists.
- Provide a clear path from installation to wrapper selection, runtime
  verification, and failure diagnosis.
- Surface actionable debug artifacts, including logs, configuration, dumps,
  and replay data, when an integration fails.

Typical Workflow
----------------

1. Prepare a supported runtime.

   Start with a MUSA-enabled ``torch`` / ``torch_musa`` stack.

2. Install MATE.

   Avoid replacing the MUSA PyTorch stack during installation.

3. Choose the matching wrapper.

   Start with FlashAttention-3, SageAttention, FlashMLA, FlashKDA, or DeepGEMM
   when one matches your framework surface.

4. Verify the runtime.

   Run ``mate check``, ``mate show-config``, and ``mate env``.

5. Debug or fall back to APIs.

   If wrapper coverage does not meet your needs, continue with direct MATE
   Python APIs.

Next steps: :doc:`Installing MATE <install>` -> :doc:`Wrappers <wrapper_tutorials>` -> :doc:`CLI & Diagnostics <diagnostics>` -> :doc:`Python APIs <api_reference>`
