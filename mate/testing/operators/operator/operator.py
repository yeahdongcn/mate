"""Core abstraction for generated and direct-input operator execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from enum import Enum
import os

import torch

from mate.execution_context import dry_run_context, raise_complete_if_dry_run

from .operator_inputs import OperatorInputs
from .operator_outputs import OperatorOutputs
from .operator_reference import OperatorReference
from .operator_workload import OperatorWorkload

MATE_BITWISE_CHECKS_ENV = "MATE_BITWISE_CHECKS"


def _bitwise_checks_from_env() -> int:
    value = os.environ.get(MATE_BITWISE_CHECKS_ENV)
    if value is None:
        return 0
    try:
        bitwise_checks = int(value)
    except ValueError as error:
        raise ValueError(
            f"{MATE_BITWISE_CHECKS_ENV} must be an integer, got {value!r}"
        ) from error
    if bitwise_checks < 1:
        raise ValueError(
            f"{MATE_BITWISE_CHECKS_ENV} must be at least 1, got {bitwise_checks}"
        )
    return bitwise_checks


class OpMode(Enum):
    NORMAL = "normal"
    DRY_RUN = "dry-run"
    DUMP = "dump"


class Operator(ABC):
    """Define how an operator is prepared, executed, and verified."""

    def __init__(self) -> None:
        self.mode = OpMode.NORMAL

    @abstractmethod
    def generate(self, workload: OperatorWorkload) -> OperatorInputs:
        """Generate concrete operator inputs from a workload description."""
        raise NotImplementedError

    @abstractmethod
    def call(self, inputs: OperatorInputs) -> OperatorOutputs:
        """Execute the operator with concrete inputs once."""
        raise NotImplementedError

    @abstractmethod
    def reference(self, inputs: OperatorInputs) -> OperatorReference:
        """Produce the reference result for concrete inputs."""
        raise NotImplementedError

    @abstractmethod
    def verify(
        self,
        inputs: OperatorInputs,
        outputs: OperatorOutputs,
        reference: OperatorReference,
    ) -> None:
        """Raise when operator outputs do not satisfy the reference result."""
        raise NotImplementedError

    def _test(
        self,
        workload: OperatorWorkload,
        *,
        mode: OpMode = OpMode.NORMAL,
    ) -> None:
        """
        Internal method to demo how to test the operator.
        Given a workload, generate inputs, execute the operator, and verify outputs.

        Args:
            workload: Workload description to generate concrete inputs.
            mode: Operator execution mode, either NORMAL, DRY_RUN, or DUMP for now.
        """
        with self.use_mode(mode):
            inputs = self.generate(workload)

            reference = self.reference(inputs) if mode is OpMode.NORMAL else None
            bitwise_checks = _bitwise_checks_from_env() if mode is OpMode.NORMAL else 0
            stable_inputs = inputs.clone() if bitwise_checks else None

            outputs = self.call(inputs)
            baseline = outputs.clone() if bitwise_checks else None
            raise_complete_if_dry_run()

            if stable_inputs is not None and baseline is not None:
                for repeat in range(1, bitwise_checks + 1):
                    inputs._copy_from(stable_inputs)
                    current = self.call(inputs).clone()
                    torch.testing.assert_close(
                        baseline,
                        current,
                        atol=0,
                        rtol=0,
                        msg=lambda message, repeat=repeat: (
                            f"bitwise check failed on repeat "
                            f"{repeat}/{bitwise_checks}: {message}"
                        ),
                    )

            if reference is not None:
                self.verify(inputs, outputs, reference)

    def _bench(
        self,
        inputs: OperatorInputs,
        *,
        verify: bool = True,
        warmup: int = 1,
        iterations: int = 10,
        method: str = "event",
        flush_l2: bool = True,
    ) -> None:
        """
        Internal method to demo how to bench the operator.
        Given inputs, benchmark the operator execution with optional verification.

        Args:
            inputs: Concrete operator inputs.
            verify: Whether to verify the outputs against the reference result.
            warmup: Number of warmup iterations before benchmarking.
            iterations: Number of iterations to benchmark.
            method: Benchmarking method, either "trace" or "event".
            flush_l2: Whether to flush L2 cache before each iteration.
        """
        assert method in ["trace", "event"], f"Invalid bench method: {method}"

        stable_inputs = inputs.clone()

        if verify:
            reference = self.reference(inputs)
            outputs = self.call(inputs)
            self.verify(inputs, outputs, reference)
            inputs._copy_from(stable_inputs)

        for _ in range(warmup):
            self.call(inputs)
            inputs._copy_from(stable_inputs)
        # TODO: Implement trace and event-based benchmarking
        for _ in range(iterations):
            # Call operator execution
            self.call(inputs)
            # Restore stable inputs
            inputs._copy_from(stable_inputs)
            # TODO: L2 Cache Flush
            pass

    def __call__(self, inputs: OperatorInputs) -> OperatorOutputs:
        """Execute the operator with concrete inputs once."""
        return self.call(inputs)

    @contextmanager
    def use_mode(self, mode: OpMode):
        prev_mode = self.mode
        self.mode = mode
        try:
            if mode is OpMode.NORMAL:
                yield self
            else:
                with self._mode_context(mode):
                    yield self
        finally:
            self.mode = prev_mode

    def _mode_context(self, mode: OpMode):
        # OpMode.NORMAL will never enter here.
        match mode:
            case OpMode.DRY_RUN:
                return dry_run_context()
            case OpMode.DUMP:
                return nullcontext()
        return nullcontext()

    def dry_run(self):
        return self.use_mode(OpMode.DRY_RUN)

    def dump(self):
        return self.use_mode(OpMode.DUMP)
