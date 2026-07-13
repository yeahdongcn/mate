Installing MATE
===============

This page describes how to install MATE or a supported wrapper package from
the MUSA Python wheel source, or how to build MATE from source on top of an
existing MUSA-enabled ``torch`` / ``torch_musa`` stack.

Steps at a glance
-----------------

1. Check requirements.
2. Choose a package source.
3. Install a delivered package or build from source.
4. Validate installation.
5. Build a wrapper from source if needed.
6. Optionally pre-build AOT kernels.

Step 1. Check Requirements
--------------------------

MATE currently requires the following runtime baseline:

+----------------+----------------------+
| Component      | Requirement          |
+================+======================+
| Python         | ``3.10`` or later    |
+----------------+----------------------+
| MUSA Toolkit   | ``4.3.6`` or later   |
+----------------+----------------------+
| TorchMUSA      | ``2.7`` or later     |
+----------------+----------------------+
| Architecture   | ``Pinghu (MP31)``    |
+----------------+----------------------+

The table above shows the repository-wide baseline. Some feature paths need
newer toolchains. The current external delivery source mainly covers
``x86_64`` and Python ``3.10`` / ``3.12`` wheels. When a wrapper or API page
lists a stricter requirement, follow that page. For example, FlashKDA
currently builds on MUSA SDK / MTCC 5.1.0+, and FlashAttention ``Local +
attention_chunk`` requires MUSA SDK 5.1.0+.

Before continuing, make sure the MUSA-enabled ``torch`` / ``torch_musa`` stack
is already installed and working in your environment.

Step 2. Choose a Package Source
-------------------------------

For delivered packages, use the external MUSA wheel source.

- Index:
  ``https://dl.mthreads.com/repo/api/pypi/pypi/simple``
- Wheel root:
  ``https://dl.mthreads.com/repo/repository/public-release-pypi/``

Choose one configuration method:

Temporary shell or CI
   .. code-block:: bash

      export PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple

Persistent pip config
   .. code-block:: bash

      python -m pip config set global.index-url \
        https://dl.mthreads.com/repo/api/pypi/pypi/simple

One-off install
   .. code-block:: bash

      python -m pip install <package> \
        --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

   .. important::

      Use one package index per install step. Do not mix the MUSA wheel
      source with a public PyPI mirror in the same ``pip install`` command.

Optional:

Direct wheel URL
   Use this when you need one exact wheel. The URL format is
   ``<package>/<version>/<wheel-file>`` under ``public-release-pypi``.

   .. code-block:: bash

      python -m pip install \
        https://dl.mthreads.com/repo/repository/public-release-pypi/<package>/<version>/<wheel-file>

   Keep one ``index-url`` configured for dependency resolution, or add
   ``--no-deps`` if dependencies are already installed.

Check available versions
   .. code-block:: bash

      python -m pip index versions mate \
        --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

   If ``pip index versions`` is unavailable, upgrade ``pip`` first.

Step 3. Install a Delivered Package or Build from Source
--------------------------------------------------------

Choose one installation path:

- Delivered wrapper install: recommended when your framework already expects
  ``flash_attn_3``, ``flash_mla``, ``flash_kda``, ``deep-gemm``, or
  ``sageattention``. Each delivered wrapper installs the matching ``mate``
  dependency automatically.
- Direct MATE install: use this when you need direct ``mate`` APIs without a
  wrapper.
- Build from source: use this when you are developing MATE locally or need a
  local build artifact.

Delivered wrapper install
~~~~~~~~~~~~~~~~~~~~~~~~~

Supported delivered wrapper packages are ``flash_attn_3``, ``flash_mla``,
``flash_kda``, ``deep-gemm``, and ``sageattention``.

Before reinstalling a delivered package set, uninstall the packages you plan
to replace:

.. code-block:: bash

   python -m pip uninstall -y \
     mate flash_mla flash_kda deep-gemm sageattention flash_attn_3

Install the wrapper package that matches your framework surface. ``pip``
installs the matching ``mate`` dependency automatically.

.. code-block:: bash

   python -m pip install flash_attn_3 \
     --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

Replace ``flash_attn_3`` with ``flash_mla``, ``flash_kda``, ``deep-gemm``, or
``sageattention`` when those package surfaces match your framework.

Delivered MUSA wrapper versions use the PEP 440 local version suffix
``+musa``, for example ``0.2.4+musa``. Check the installed distribution to
distinguish the MATE-backed MUSA wrapper from the native implementation:

.. code-block:: bash

   python -m pip show flash_attn_3

The ``Version`` field should end in ``+musa``. Replace ``flash_attn_3`` with
the wrapper package you installed when checking another wrapper. The matching
``mate`` dependency keeps its normal version; ``+musa`` is a wrapper-only
identifier and does not encode the MUSA Toolkit version.

Direct MATE install
~~~~~~~~~~~~~~~~~~~

Install ``mate`` directly when no wrapper matches your workload or when you
want direct MATE Python APIs without a wrapper package.

.. code-block:: bash

   python -m pip install mate \
     --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

Build from source
~~~~~~~~~~~~~~~~~

Get the source checkout
^^^^^^^^^^^^^^^^^^^^^^^

Clone the repository with submodules:

.. code-block:: bash

   git clone https://github.com/MooreThreads/mate.git --recursive
   cd mate

.. note::

   If the repository was cloned without ``--recursive``, run
   ``git submodule update --init --recursive`` in the repository root before
   building.

Install from the local checkout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For local builds, keep dependency resolution disabled so ``pip`` does not
replace the MUSA PyTorch stack with upstream PyPI packages.

- Use ``--no-build-isolation`` for source installs.
- Use ``--no-isolation`` for local wheel builds.
- Use ``--no-deps`` when installing local builds.

Choose one local install mode:

- Use an editable install when you are iterating on the local checkout.
- Build and install a local wheel when you need a built artifact.

Editable install
""""""""""""""""

Use an editable install when you are iterating on the local checkout:

.. code-block:: bash

   python -m pip install --no-build-isolation --no-deps -e . -v

Local wheel install
"""""""""""""""""""

Use a local wheel when you want a built artifact instead of an editable
install:

.. code-block:: bash

   python -m build --wheel --no-isolation
   python -m pip install --no-deps dist/mate-*.whl

Step 4. Validate Installation
-----------------------------

After installing MATE directly or through a wrapper package, validate the
MATE runtime first.

.. code-block:: bash

   python - <<'PY'
   import mate
   print("mate import ok")
   PY

.. code-block:: bash

   python -m mate --help
   mate check
   mate show-config
   mate env

If the ``mate`` executable entrypoint is not available in your environment,
use ``python -m mate ...`` for supported subcommands.

If you installed a wrapper package in Step 3, follow that wrapper page for the
wrapper import path and package-specific validation snippet.

Step 5. Build a Wrapper from Source if Needed
---------------------------------------------

Skip this step if you installed a delivered wrapper package in Step 3.

Use this path only when you are developing a wrapper locally from the
repository checkout. Install local ``mate`` from the Step 3 source-build path
first. The commands below are for local editable installs or local wheel
installs, not the delivered wheel source.

- Install local ``mate`` first.
- Use ``--no-deps`` for local wrapper installs.

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

   cd /path/to/mate
   python -m pip install --no-build-isolation --no-deps -e . -v
   cd wrappers/flash-attention
   python -m pip install --no-build-isolation --no-deps -e .

Wheel install pattern:

.. code-block:: bash

   cd /path/to/mate
   python -m pip install --no-build-isolation --no-deps -e . -v
   cd wrappers/flash-attention
   python -m build --wheel
   python -m pip install --no-deps dist/flash_attn_3-*.whl

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
