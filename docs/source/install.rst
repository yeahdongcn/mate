Installing MATE
===============

This page describes how to install MATE from source on top of an existing
MUSA-enabled ``torch`` / ``torch_musa`` stack.

Steps at a glance
-----------------

1. Check requirements.
2. Get source from GitHub.
3. Install MATE.
4. Validate installation.
5. Install a wrapper.
6. Optionally pre-build AOT kernels.

Step 1. Check Requirements
--------------------------

MATE currently requires the following runtime baseline:

+----------------+----------------------+
| Component      | Requirement          |
+================+======================+
| MUSA Toolkit   | ``4.3.6`` or later   |
+----------------+----------------------+
| TorchMUSA      | ``2.7`` or later     |
+----------------+----------------------+
| Architecture   | ``Pinghu (MP31)``    |
+----------------+----------------------+

Before continuing, make sure the MUSA-enabled ``torch`` / ``torch_musa`` stack
is already installed and working in your environment.

Step 2. Get Source from GitHub
------------------------------

Clone the repository with submodules:

.. code-block:: bash

   git clone https://github.com/MooreThreads/mate.git --recursive
   cd mate

.. note::

   If the repository was cloned without ``--recursive``, run
   ``git submodule update --init --recursive`` in the repository root before
   building.

Step 3. Install MATE
--------------------

For local builds, keep dependency resolution disabled so ``pip`` does not
replace the MUSA PyTorch stack with upstream PyPI packages.

- Use ``--no-build-isolation`` for source installs.
- Use ``--no-isolation`` for local wheel builds.
- Use ``--no-deps`` when installing local builds.

Development install
~~~~~~~~~~~~~~~~~~~

Use an editable install when you are iterating on the local checkout:

.. code-block:: bash

   pip install --no-build-isolation --no-deps -e . -v

Build and install a wheel
~~~~~~~~~~~~~~~~~~~~~~~~~

Use a local wheel when you want a built artifact instead of an editable install:

.. code-block:: bash

   python -m build --wheel --no-isolation
   python -m pip install --no-deps dist/mate-*.whl

Step 4. Validate Installation
-----------------------------

After installing MATE, validate the runtime before integrating wrappers or
calling ``mate`` APIs directly.

.. code-block:: bash

   python -m mate --help
   mate check
   mate show-config
   mate env

If the ``mate`` executable entrypoint is not available in your environment,
use ``python -m mate ...`` for supported subcommands.

Step 5. Install a Wrapper
-------------------------

After MATE is installed and importable, install the wrapper that matches
the Python package surface your framework already expects.

.. list-table::
   :header-rows: 1

   * - Wrapper directory
     - Package name
     - Import path
     - Typical use
   * - ``wrappers/flash-attention``
     - ``flash_attn_3``
     - ``flash_attn_interface``
     - FlashAttention-3 style integration
   * - ``wrappers/FlashMLA``
     - ``flash_mla``
     - ``flash_mla``
     - FlashMLA style integration
   * - ``wrappers/FlashKDA``
     - ``flash_kda``
     - ``flash_kda``
     - FlashKDA style integration
   * - ``wrappers/DeepGEMM``
     - ``deep-gemm``
     - ``deep_gemm``
     - DeepGEMM style integration
   * - ``wrappers/SageAttention``
     - ``sageattention``
     - ``sageattention``
     - SageAttention style integration

Editable install pattern:

.. code-block:: bash

   cd wrappers/flash-attention
   pip install --no-build-isolation -e .

Wheel install pattern:

.. code-block:: bash

   cd wrappers/flash-attention
   python -m build --wheel
   pip install dist/flash_attn_3-*.whl

Repeat the same workflow for ``wrappers/FlashMLA``, ``wrappers/FlashKDA``,
``wrappers/DeepGEMM``, or ``wrappers/SageAttention`` when those package
surfaces match your framework.

Optional Step 6. Pre-Build AOT Kernels
--------------------------------------

If you want to pre-build AOT kernels before producing a wheel, run:

.. code-block:: bash

   MATE_MUSA_ARCH_LIST=3.1 python -m mate.aot
   python -m build --wheel --no-isolation

Customize AOT coverage when needed:

.. code-block:: bash

   python -m mate.aot --attention-aot-level 0 --add-gemm true --add-moe false

Next Steps
----------

- Continue with :doc:`wrapper_tutorials` for wrapper-specific quickstarts.
- Continue with :doc:`diagnostics` if validation or runtime behavior fails.
- Continue with :doc:`api_reference` when no wrapper matches your workload.
