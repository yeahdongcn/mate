"""Base type for concrete operator inputs."""

from __future__ import annotations

import copy

import torch


class OperatorInputs:
    """Provide value snapshots for operators that mutate their inputs."""

    def clone(self) -> OperatorInputs:
        """Snapshot top-level tensor values while preserving other input state."""
        snapshot = copy.copy(self)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(snapshot, name, value.detach().clone())
        return snapshot

    def _copy_from(self, other: OperatorInputs) -> None:
        """Restore this input in place from a snapshot of the same type."""
        if type(self) is not type(other):
            raise TypeError(
                "operator input snapshots must have the same type: "
                f"{type(self).__name__} != {type(other).__name__}"
            )

        for name, snapshot_value in vars(other).items():
            current_value = getattr(self, name)
            if isinstance(snapshot_value, torch.Tensor):
                if not isinstance(current_value, torch.Tensor):
                    setattr(self, name, snapshot_value.detach().clone())
                else:
                    current_value.copy_(snapshot_value)
            else:
                setattr(self, name, snapshot_value)


__all__ = ["OperatorInputs"]
