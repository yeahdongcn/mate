from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from functools import lru_cache, wraps
from typing import Any, Callable, ContextManager, TypeVar

from torch._guards import active_fake_mode

from torch._subclasses.fake_tensor import FakeTensorMode

MATE_DRY_RUN_ENV = "MATE_DRY_RUN"


class MateDryRunComplete(RuntimeError):
    """Raised to stop the current dry-run path once compilation is complete."""


@lru_cache()
def is_dry_run_enabled() -> bool:
    return os.environ.get(MATE_DRY_RUN_ENV, "0") == "1"


def _fake_tensor_mode(
    allow_non_fake_inputs: bool = False,
) -> ContextManager[Any]:
    if active_fake_mode() is not None:
        return nullcontext()
    return FakeTensorMode(allow_non_fake_inputs=allow_non_fake_inputs)


@contextmanager
def dry_run_context(allow_non_fake_inputs: bool = False):
    """Enable MATE dry-run behavior and fake tensors for one execution scope."""
    previous = os.environ.get(MATE_DRY_RUN_ENV)
    os.environ[MATE_DRY_RUN_ENV] = "1"
    is_dry_run_enabled.cache_clear()
    try:
        with _fake_tensor_mode(allow_non_fake_inputs=allow_non_fake_inputs):
            yield
    finally:
        if previous is None:
            os.environ.pop(MATE_DRY_RUN_ENV, None)
        else:
            os.environ[MATE_DRY_RUN_ENV] = previous
        is_dry_run_enabled.cache_clear()


def is_fake_mode() -> bool:
    return active_fake_mode() is not None


def raise_complete_if_dry_run():
    if is_dry_run_enabled():
        raise MateDryRunComplete


def skip_kernel_launch_if_dry_run() -> bool:
    """Return whether a compiled kernel launch should be skipped."""
    return is_dry_run_enabled()


ResultT = TypeVar("ResultT")


def call_or_return_if_dry_run(
    fn: Callable[..., ResultT],
    *,
    dry_run_result: ResultT,
) -> Callable[..., ResultT]:
    """Call ``fn`` normally or return a caller-provided dry-run result."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> ResultT:
        if is_dry_run_enabled():
            return dry_run_result
        return fn(*args, **kwargs)

    return wrapper


def empty_if_dry_run(
    fn: Callable[..., ResultT],
    *,
    empty_values: ResultT,
) -> Callable[..., ResultT]:
    """Compatibility alias for ``call_or_return_if_dry_run``."""
    return call_or_return_if_dry_run(fn, dry_run_result=empty_values)
