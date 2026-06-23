Command Line Interface
======================

.. toctree::
   :hidden:
   :maxdepth: 1

   MATE CLI Reference <mate_cli>

MATE provides a command-line interface for validating the runtime environment,
inspecting resolved configuration, viewing environment variables, inspecting
registered JIT or AOT modules, exporting compile commands for development
tools, clearing the JIT cache, and replaying captured dumps.

Quick Reference
---------------

View all available commands:

.. code-block:: bash

   mate --help

If the ``mate`` executable entrypoint is not available in your environment, use
``python -m mate --help`` and ``python -m mate ...`` for supported subcommands.

For the full command reference, see :doc:`MATE CLI Reference <mate_cli>`.

Environment and Status
----------------------

Use these commands first after installation or when runtime behavior does not
match expectations.

Validate the runtime environment:

.. code-block:: bash

   mate check

Inspect resolved configuration, package versions, detected devices,
architecture resolution, and JIT or AOT state:

.. code-block:: bash

   mate show-config

Show the MATE-related environment variables visible to the current shell:

.. code-block:: bash

   mate env

JIT and Development Tools
-------------------------

Inspect the registered JIT and AOT modules without compiling them:

.. code-block:: bash

   mate module-status
   mate module-status --detailed

List all registered modules, or inspect one module in more detail:

.. code-block:: bash

   mate list-modules
   mate list-modules module_name

Export ``compile_commands.json`` for IDEs and language servers such as
``clangd`` or ``ccls``:

.. code-block:: bash

   mate export-compile-commands
   mate export-compile-commands my_compile_commands.json

Cache Management
----------------

Clear the runtime JIT cache without removing packaged AOT libraries:

.. code-block:: bash

   mate clear-cache

Replay Recorded Calls
---------------------

Replay API dumps captured with Level 10 logging.

List recorded dump directories under a dump root:

.. code-block:: bash

   mate list-dumps mate_dumps/

Replay all recorded calls in a dump session:

.. code-block:: bash

   mate replay --dir mate_dumps/

Replay a single recorded call:

.. code-block:: bash

   mate replay --dir mate_dumps/<dump_directory>

``mate replay`` accepts either the dump root directory or a single dump
subdirectory.

See Also
--------

- For the full dump and replay workflow, see :doc:`logging_debugging`.
- For the broader troubleshooting path, see :doc:`diagnostics`.
