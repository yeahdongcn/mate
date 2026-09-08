"""Base type for concrete operator outputs."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class OperatorOutputs(Mapping[str, Any]):
    """Expose dataclass outputs as a mapping for recursive comparisons."""

    def __getitem__(self, key: str) -> Any:
        for field in fields(self):
            if field.name == key:
                return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (field.name for field in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    def clone(self) -> OperatorOutputs:
        """Snapshot outputs before a later call can overwrite shared buffers."""
        return copy.deepcopy(self)


__all__ = ["OperatorOutputs"]
