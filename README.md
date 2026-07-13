# MUSA AI Tensor Engine

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://mate-docs.mthreads.com/latest/)

MATE (**M**USA **A**I **T**ensor **E**ngine) is a centralized library for Generative AI workloads on MUSA. It provides high-performance Attention and GEMM operators, and compatibility wrappers for CUDA-oriented Python APIs.

## Highlights

- High-performance attention and GEMM operators for MUSA
- Mixed-dtype W4A8 MoE GEMM APIs for ragged and masked MoE paths in `0.2.4`
- Compatibility wrappers for `flash_attn_3`, `sageattention`, `flash_mla`,
  `flash_kda`, and `deep-gemm`
- CLI tools for environment checks, configuration inspection, and replay

## Requirements

| Component | Requirement |
| --- | --- |
| Python | `3.10` or later |
| MUSA Toolkit | `4.3.6` or later |
| TorchMUSA | `2.7` or later |
| Architecture | `Pinghu (MP31)` |

The current external delivery source mainly covers `x86_64` and Python `3.10`
/ `3.12` wheels.

## Recommended Workflow

For Moore Threads platforms, the normal integration flow is wrapper-first:

1. Start with an existing MUSA-enabled `torch` / `torch_musa` stack
2. If a supported wrapper matches your framework surface, install that wrapper from the MUSA wheel source
3. The wrapper install pulls in the matching `mate` package automatically
4. Install `mate` directly only when no wrapper matches your workload or you need direct MATE APIs
5. Keep the upstream import path and high-level API shape as stable as possible
6. Use `mate check`, `mate show-config`, `mate env`, logging, and replay if something fails

## Quick Start

### Install from the MUSA Wheel Source

For delivered builds, use the external delivery source after the MUSA-enabled
`torch` / `torch_musa` stack is already installed.

```bash
python -m pip install flash_attn_3 \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

This installs the matching `mate` dependency automatically. Install other
wrapper packages such as `flash_mla`, `flash_kda`, `deep-gemm`, or
`sageattention` the same way.

MUSA compatibility wrapper releases use the PEP 440 local version suffix
`+musa`, for example `0.2.4+musa`. Check the installed distribution with
`python -m pip show flash_attn_3`; a version ending in `+musa` identifies the
MATE-backed MUSA wrapper rather than the native implementation. The matching
`mate` dependency keeps its normal version without this wrapper-only suffix.

Install `mate` directly only when you need direct MATE APIs without a wrapper:

```bash
python -m pip install mate \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

Use exactly one package index per install step. Do not mix the MUSA wheel
source with a public PyPI mirror in the same `pip install` invocation.

### Build from Source

```bash
git clone https://github.com/MooreThreads/mate.git --recursive
cd mate
python -m pip install --no-build-isolation --no-deps -e . -v
```

For a local wheel instead of an editable install:

```bash
python -m build --wheel --no-isolation
python -m pip install --no-deps dist/mate-*.whl
```

See [`docs/source/install.rst`](docs/source/install.rst) for the full
wheel-source workflow, wrapper package commands, `PIP_INDEX_URL`, `pip config`,
`pip index versions`, direct wheel URLs, and local AOT build options.

### Choose a Wrapper

Install the wrapper that matches your framework's expected Python package
surface. Each delivered wrapper install pulls in the matching `mate`
dependency automatically.

| Wrapper directory | Package | Import path | Typical use |
| --- | --- | --- | --- |
| `wrappers/flash-attention` | `flash_attn_3` | `flash_attn_interface` | FlashAttention-3 style integration |
| `wrappers/FlashMLA` | `flash_mla` | `flash_mla` | FlashMLA style integration |
| `wrappers/FlashKDA` | `flash_kda` | `flash_kda` | FlashKDA style integration |
| `wrappers/DeepGEMM` | `deep-gemm` | `deep_gemm` | DeepGEMM style integration |
| `wrappers/SageAttention` | `sageattention` | `sageattention` | SageAttention style integration |

For wrapper install commands, use the same external index or follow the local
build workflow in [`docs/source/install.rst`](docs/source/install.rst).

### Verify and Diagnose

Start with these commands after installation:

```bash
mate check
mate show-config
mate env
```

### Notes

- If the checkout was cloned without `--recursive`, run `git submodule update --init --recursive`.
- Do not let pip resolve and replace the MUSA PyTorch dependencies unless that
  is intentional.
- See [docs/mate_cli.md](docs/mate_cli.md) for CLI extras and local wheel
  installation details.
- See [docs/environment_variables.md](docs/environment_variables.md) for build
  and runtime environment variables.

## MATE CLI

MATE provides a command-line interface for configuration, debugging, diagnostics, and replay.

| Command | Purpose |
| --- | --- |
| `mate check` | Validate the runtime environment |
| `mate show-config` | Display installation and runtime configuration |
| `mate env` | Show relevant environment variables |
| `mate guard-run -- COMMAND` | Run a workload with the guarded MUSA allocator installed at startup |
| `mate replay --dir PATH` | Replay API calls from Level 10 dumps |
| `mate list-dumps PATH` | List recorded dump directories |

Example:

```bash
mate check
mate show-config
mate env
mate guard-run -- python your_script.py
mate replay --dir mate_dumps/
mate list-dumps mate_dumps/
```

See [docs/mate_cli.md](docs/mate_cli.md) for full CLI documentation.
See [docs/environment_variables.md](docs/environment_variables.md) for the
complete environment variable reference.

## Memory Debug / Guard Allocator

MATE includes a guarded MUSA allocator that can replace the default `torch_musa` allocator during debugging to help localize out-of-bounds reads and writes across MUSA workloads. Start with `mate guard-run --mode tail -- python your_script.py`, or enable it in tests with `pytest --guard-alloc`. The detailed workflow, pytest defaults, and limitations are documented in [docs/guard_allocator.md](docs/guard_allocator.md).

## Repository Layout

| Path | Purpose |
| --- | --- |
| `mate/` | Core Python package and public APIs |
| `wrappers/` | Compatibility wrapper packages for existing Python ecosystems |
| `docs/` | Markdown docs and Sphinx sources |
| `tests/` | Correctness and integration tests |
| `benchmarks/` | Performance and benchmarking scripts |

## Quick Links

- Official documentation: [https://mate-docs.mthreads.com/latest/](https://mate-docs.mthreads.com/latest/)
- Wrapper quickstarts: [docs/source/wrapper_tutorials.rst](docs/source/wrapper_tutorials.rst)
- CLI documentation: [docs/mate_cli.md](docs/mate_cli.md)
- Guard allocator debugging: [docs/guard_allocator.md](docs/guard_allocator.md)
- Environment variables: [docs/environment_variables.md](docs/environment_variables.md)
- FlashAttention-3 compatibility summary: [docs/source/wrappers/flash_attention_forward_compatibility.md](docs/source/wrappers/flash_attention_forward_compatibility.md)

## Acknowledgement

MATE is inspired by [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [cutlass](https://github.com/NVIDIA/cutlass), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), and [DeepGemm](https://github.com/deepseek-ai/DeepGEMM).
