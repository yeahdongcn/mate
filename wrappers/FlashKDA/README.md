# FlashKDA Compatibility Wrapper (flash_kda)

`flash_kda` is a compatibility wrapper package that preserves the official
`flash_kda` package name and import path while running on MUSA through MATE
Kimi Delta Attention (KDA) operators.

## Overview

This wrapper is designed for projects that already target the FlashKDA Python
API. It keeps the public `flash_kda` import surface and forwards execution to
`mate.kda.chunk_kda`.

The current compatibility scope includes:

- `flash_kda.fwd`
- `flash_kda.get_workspace_size`

## Package and import

- Package name: `flash_kda`
- Import path: `flash_kda`
- Runtime backend: MATE KDA operators on MUSA

## Requirements

Before using this wrapper, make sure the following are available:

- MATE is installed and importable.
- TorchMUSA is installed and the MUSA runtime environment is configured.
- The target workload is configured to run on MUSA devices.

## Build

Build a wheel from the `wrappers/FlashKDA` directory:

```bash
python -m build --wheel
```

The generated wheel will be placed under:

```text
dist/
```

## Installation

Install from source:

```bash
pip install --no-build-isolation -e .
```

Install a built wheel:

```bash
pip install dist/flash_kda-*.whl
```

## Import

Import the package directly:

```python
import flash_kda
```

Import individual APIs:

```python
from flash_kda import fwd, get_workspace_size
```

## Behavior

- `fwd(...)` preserves the FlashKDA Python call signature and forwards to
  `mate.kda.chunk_kda(...)` with `use_qk_l2norm_in_kernel=True`.
- `out` and `final_state` are treated as preallocated output buffers and are
  written in place, matching the official package surface.
- `get_workspace_size(...)` is preserved for compatibility and always returns
  `0`. The wrapper does not allocate or consume an explicit workspace tensor
  because MATE manages the kernel internals itself.
- `cu_seqlens` follows the MATE runtime behavior on MUSA and accepts both
  `torch.int32` and `torch.int64`.

## Notes

- This wrapper keeps the official FlashKDA import surface, but execution is
  provided by MATE on MUSA.
- For the authoritative operator behavior, refer to `mate.kda.chunk_kda`.
