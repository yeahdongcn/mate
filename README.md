# MUSA AI Tensor Engine

MATE (**M**USA **A**I **T**ensor **E**ngine) is a centralized library for Generative AI workloads on MUSA. It provides high-performance Attention and GEMM operators, and compatibility wrappers for CUDA-oriented Python APIs.

## Highlights

- High-performance attention and GEMM operators for MUSA
- Compatibility wrappers for `flash_attn_3`, `sageattention`, `flash_mla`,
  `flash_kda`, and `deep_gemm`
- CLI tools for environment checks, configuration inspection, and replay

## Requirements

| Component | Requirement |
| --- | --- |
| MUSA Toolkit | `4.3.6` or later |
| TorchMUSA | `2.7` or later |
| Architecture | `Pinghu (MP31)` |

## Recommended Workflow

For Moore Threads platforms, the normal integration flow is wrapper-first:

1. Install MATE on top of an existing MUSA-enabled `torch` / `torch_musa` stack
2. Install the wrapper package that matches the Python package surface your framework already expects
3. Keep the upstream import path and high-level API shape as stable as possible
4. Use `mate check`, `mate show-config`, `mate env`, logging, and replay if something fails
5. Use native MATE APIs only when no wrapper matches your workload or the wrapper does not cover the feature you need

## Quick Start

Use these commands after the MUSA-enabled `torch` / `torch_musa` stack is
installed. Keep dependency resolution disabled for local builds so pip does not
replace that stack with upstream PyPI packages.

- Use `--no-build-isolation` for source installs.
- Use `--no-isolation` for wheel builds.
- Use `--no-deps` when installing local builds.

### Development Install

```bash
git clone https://github.com/MooreThreads/mate.git --recursive
cd mate
pip install --no-build-isolation --no-deps -e . -v
```

### Build a Wheel

```bash
git clone https://github.com/MooreThreads/mate.git --recursive
cd mate
python -m build --wheel --no-isolation
python -m pip install --no-deps dist/mate-*.whl
```

### Optional: Pre-Build AOT Kernels

```bash
MATE_MUSA_ARCH_LIST=3.1 python -m mate.aot
python -m build --wheel --no-isolation
```

Customize AOT coverage when needed:

```bash
python -m mate.aot --attention-aot-level 0 --add-gemm true --add-moe false
```

### Install a Wrapper

After MATE is installed, install the wrapper that matches your framework's
expected Python package surface.

| Wrapper directory | Package | Import path | Typical use |
| --- | --- | --- | --- |
| `wrappers/flash-attention` | `flash_attn_3` | `flash_attn_interface` | FlashAttention-3 style integration |
| `wrappers/FlashMLA` | `flash_mla` | `flash_mla` | FlashMLA style integration |
| `wrappers/FlashKDA` | `flash_kda` | `flash_kda` | FlashKDA style integration |
| `wrappers/DeepGEMM` | `deep-gemm` | `deep_gemm` | DeepGEMM style integration |
| `wrappers/SageAttention` | `sageattention` | `sageattention` | SageAttention style integration |

Generic editable install pattern:

```bash
cd wrappers/flash-attention
pip install --no-build-isolation -e .
```

Generic wheel install pattern:

```bash
cd wrappers/flash-attention
python -m build --wheel
pip install dist/flash_attn_3-*.whl
```

Repeat the same workflow for `wrappers/FlashMLA`, `wrappers/FlashKDA`,
`wrappers/DeepGEMM`, or `wrappers/SageAttention` when those package surfaces
match your framework.

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

## Wrappers

MATE uses the packages under `wrappers/` as a compatibility layer for CUDA-oriented software stacks on MUSA. These wrappers preserve familiar package names and high-level APIs while routing execution to MATE operators and kernels on MUSA, which helps existing integrations migrate with smaller code changes.

For the guided wrapper-first documentation path, start with [docs/source/overview.rst](docs/source/overview.rst) and [docs/source/wrapper_tutorials.rst](docs/source/wrapper_tutorials.rst) for the wrapper quickstart flow.

| Wrapper | Package | Import Path | Purpose | Documentation |
| --- | --- | --- | --- | --- |
| `wrappers/flash-attention` | `flash_attn_3` | `flash_attn_interface` | FlashAttention-3-compatible APIs on top of MATE attention operators on MUSA | [wrapper README](wrappers/flash-attention/README.md), [compatibility summary](docs/source/wrappers/flash_attention_forward_compatibility.md) |
| `wrappers/SageAttention` | `sageattention` | `sageattention` | SageAttention-compatible dense quantized attention wrapper on top of MATE on MUSA | [wrapper README](wrappers/SageAttention/README.md) |
| `wrappers/FlashMLA` | `flash_mla` | `flash_mla` | FlashMLA-compatible MLA dense/sparse decode and sparse prefill APIs on top of MATE MLA operators on MUSA | [wrapper README](wrappers/FlashMLA/README.md) |
| `wrappers/FlashKDA` | `flash_kda` | `flash_kda` | FlashKDA-compatible KDA forward APIs on top of MATE KDA operators on MUSA | [wrapper README](wrappers/FlashKDA/README.md) |
| `wrappers/DeepGEMM` | `deep-gemm` | `deep_gemm` | DeepGEMM-compatible APIs on top of MATE GEMM operators on MUSA | [wrapper README](wrappers/DeepGEMM/README.md) |

## Repository Layout

| Path | Purpose |
| --- | --- |
| `mate/` | Core Python package and public APIs |
| `wrappers/` | Compatibility wrapper packages for existing Python ecosystems |
| `docs/` | Markdown docs and Sphinx sources |
| `tests/` | Correctness and integration tests |
| `benchmarks/` | Performance and benchmarking scripts |

## Build Documentation

After installing `mate`, build the Sphinx docs with:

```bash
pip install sphinx furo myst-parser
cd docs
make html
```

## Quick Links

- CLI documentation: [docs/mate_cli.md](docs/mate_cli.md)
- Guard allocator debugging: [docs/guard_allocator.md](docs/guard_allocator.md)
- Environment variables: [docs/environment_variables.md](docs/environment_variables.md)
- FlashAttention-3 compatibility summary: [docs/source/wrappers/flash_attention_forward_compatibility.md](docs/source/wrappers/flash_attention_forward_compatibility.md)
- FlashAttention-3 wrapper: [wrappers/flash-attention/README.md](wrappers/flash-attention/README.md)
- SageAttention wrapper: [wrappers/SageAttention/README.md](wrappers/SageAttention/README.md)
- FlashMLA wrapper: [wrappers/FlashMLA/README.md](wrappers/FlashMLA/README.md)
- FlashKDA wrapper: [wrappers/FlashKDA/README.md](wrappers/FlashKDA/README.md)
- DeepGEMM wrapper: [wrappers/DeepGEMM/README.md](wrappers/DeepGEMM/README.md)

## Acknowledgement

MATE is inspired by [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [cutlass](https://github.com/NVIDIA/cutlass), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), and [DeepGemm](https://github.com/deepseek-ai/DeepGEMM).
