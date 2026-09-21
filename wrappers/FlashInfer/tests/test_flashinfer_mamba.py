"""Contract tests for the FlashInfer-compatible mamba surface.

These tests are about the *surface*: which names exist, in what order their
positional parameters must appear, and whether calls reach the MATE backend
without being reshaped. They do not need a MUSA device, because the backend is
replaced by a recording stub. Numeric behavior is covered by the MATE mamba
family's own tests.
"""

import importlib
import inspect
import sys
import types

import pytest

from flashinfer.mamba import (
    allocate_checkpointing_ssu_scratch,
    checkpointing_ssu,
    mamba_chunk_scan_combined_varlen,
    replayssm_materialize,
    selective_state_update,
    ssd_combined_fwd_varlen,
)
from flashinfer.mamba import _backend

# Frozen consumer contract: vLLM's Mamba2/SSD implementation passes these
# arguments positionally, so names and order are part of the API.
VARLEN_POSITIONAL_PREFIX = [
    "x",
    "dt",
    "A",
    "B",
    "C",
    "chunk_size",
    "cu_seqlens",
    "cu_chunk_seqlens",
    "last_chunk_indices",
    "seq_idx",
    "out",
    "D",
    "z",
    "dt_bias",
    "initial_states",
    "dt_softplus",
    "dt_limit",
    "return_intermediate_states",
    "state_dtype",
    "checkpoint_token_indices",
    "checkpoint_state_slots",
    "checkpoint_states",
]

# Upstream FlashInfer parameter order, which vLLM's SSU dispatch relies on.
SSU_PARAMETERS = [
    "state",
    "x",
    "dt",
    "A",
    "B",
    "C",
    "D",
    "z",
    "dt_bias",
    "dt_softplus",
    "state_batch_indices",
    "pad_slot_id",
    "state_scale",
    "out",
    "disable_state_update",
    "intermediate_states_buffer",
    "intermediate_state_indices",
    "intermediate_state_scales",
    "rand_seed",
    "philox_rounds",
    "cache_steps",
    "algorithm",
    "dst_state_batch_indices",
    "cu_seqlens",
    "num_accepted_tokens",
    "backend",
    # vLLM's MambaSSUBackend.__call__ passes these four by keyword. They are part
    # of the surface because the dispatch is the only production caller.
    "null_block_id",
    "is_blackwell",
    "enable_stochastic_rounding",
    "cache_philox_rounds",
]


def test_varlen_ssd_matches_consumer_positional_prefix():
    assert list(inspect.signature(ssd_combined_fwd_varlen).parameters) == (
        VARLEN_POSITIONAL_PREFIX
    )
    assert mamba_chunk_scan_combined_varlen is ssd_combined_fwd_varlen


def test_selective_state_update_matches_upstream_parameter_order():
    parameters = inspect.signature(selective_state_update).parameters
    assert list(parameters) == SSU_PARAMETERS
    assert parameters["algorithm"].default == "auto"
    assert parameters["backend"].default == "auto"
    assert parameters["philox_rounds"].default == 10


def test_selective_state_update_accepts_the_dispatch_keywords():
    """vLLM's backend dispatch must reach MATE instead of raising TypeError.

    ``ssu_dispatch.FlashInferSSUBackend.__call__`` passes these four by keyword;
    a wrapper whose parameter list mirrors MATE alone rejects the call before any
    kernel runs.
    """
    parameters = inspect.signature(selective_state_update).parameters
    for name in (
        "null_block_id",
        "is_blackwell",
        "enable_stochastic_rounding",
        "cache_philox_rounds",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert parameters["null_block_id"].default == 0
    assert parameters["is_blackwell"].default is False
    assert parameters["enable_stochastic_rounding"].default is False


def test_selective_state_update_refuses_what_it_cannot_honour():
    """Refuse, do not silently ignore: both cases would change state contents."""
    with pytest.raises(NotImplementedError, match="is_blackwell"):
        selective_state_update(
            None, None, None, None, None, None, None, is_blackwell=True
        )
    with pytest.raises(NotImplementedError, match="null_block_id"):
        selective_state_update(
            None, None, None, None, None, None, None, null_block_id=7
        )


def test_checkpointing_surface_is_importable():
    assert callable(checkpointing_ssu)
    assert callable(allocate_checkpointing_ssu_scratch)
    assert callable(replayssm_materialize)


@pytest.fixture
def recording_backend(monkeypatch):
    """Install a stub ``mate.mamba`` and record how it is called."""
    calls = []

    def make(name):
        def _record(*args, **kwargs):
            calls.append((name, args, kwargs))
            return f"{name}:result"

        _record.__name__ = name
        return _record

    names = [
        "ssd_combined_fwd_varlen",
        "selective_state_update",
        "checkpointing_ssu",
        "allocate_checkpointing_ssu_scratch",
        "replayssm_materialize",
    ]
    stub = types.ModuleType(_backend.BACKEND_PACKAGE)
    for name in names:
        setattr(stub, name, make(name))
    monkeypatch.setitem(sys.modules, "mate", types.ModuleType("mate"))
    monkeypatch.setitem(sys.modules, _backend.BACKEND_PACKAGE, stub)
    monkeypatch.setattr(_backend, "_CACHE", {})

    module = importlib.import_module("flashinfer.mamba")
    return calls, module


def test_varlen_call_is_forwarded_by_name_with_every_argument(recording_backend):
    calls, _ = recording_backend
    args = [object() for _ in range(11)] + [None] * 11
    result = ssd_combined_fwd_varlen(*args)

    name, forwarded, kwargs = calls[0]
    assert name == "ssd_combined_fwd_varlen"
    # Forwarded by name, never by position: upstream orders
    # initial_states, dt_softplus, dt_limit where MATE orders
    # dt_softplus, dt_limit, initial_states, so position would misalign them.
    assert forwarded == ()
    mate_parameters = list(inspect.signature(ssd_combined_fwd_varlen).parameters)
    assert list(kwargs) == mate_parameters
    assert list(kwargs.values()) == args
    assert result == "ssd_combined_fwd_varlen:result"


# The consumer (vLLM's Mamba2 SSD routing) calls this entry point by keyword,
# passing exactly these nineteen names. Renaming a parameter breaks it.
CONSUMER_VARLEN_KEYWORDS = [
    "x",
    "dt",
    "A",
    "B",
    "C",
    "chunk_size",
    "cu_seqlens",
    "cu_chunk_seqlens",
    "last_chunk_indices",
    "seq_idx",
    "out",
    "D",
    "z",
    "dt_bias",
    "initial_states",
    "dt_softplus",
    "dt_limit",
    "return_intermediate_states",
    "state_dtype",
]


def test_varlen_accepts_the_consumer_keyword_call(recording_backend):
    calls, _ = recording_backend
    sentinel = object()
    keywords = {name: sentinel for name in CONSUMER_VARLEN_KEYWORDS}
    keywords["chunk_size"] = 128

    ssd_combined_fwd_varlen(**keywords)

    _, forwarded, kwargs = calls[0]
    assert forwarded == ()
    # Names reach MATE unchanged, and the three checkpointing arguments the
    # consumer omits keep their defaults.
    assert list(kwargs) == CONSUMER_VARLEN_KEYWORDS + [
        "checkpoint_token_indices",
        "checkpoint_state_slots",
        "checkpoint_states",
    ]
    assert kwargs["chunk_size"] == 128
    assert kwargs["x"] is sentinel and kwargs["state_dtype"] is sentinel
    assert kwargs["checkpoint_token_indices"] is None


def test_selective_state_update_forwards_keyword_capabilities(recording_backend):
    calls, _ = recording_backend
    args = [object() for _ in range(7)]
    result = selective_state_update(
        *args, algorithm="auto", backend="flashinfer", philox_rounds=10
    )

    name, forwarded, kwargs = calls[0]
    assert name == "selective_state_update"
    assert list(forwarded[:7]) == args
    # Forwarding is positional over MATE's own parameter list; the keyword-only
    # additions are the consumer contract and are consumed here, not forwarded.
    mate_parameters = [
        n
        for n, p in inspect.signature(selective_state_update).parameters.items()
        if p.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert len(forwarded) == len(mate_parameters)
    assert kwargs == {}
    assert result == "selective_state_update:result"


def test_replayssm_materialize_keeps_keyword_only_shape(recording_backend):
    calls, _ = recording_backend
    positional = [object() for _ in range(16)]
    replayssm_materialize(
        *positional,
        state_dtype="float16",
        input_dtype="float16",
        matrixA_dtype="float32",
        dim=64,
        dstate=128,
        num_heads=8,
        heads_per_group=1,
        max_window=4,
        ring_buffer_len=16,
    )

    name, forwarded, kwargs = calls[0]
    assert name == "replayssm_materialize"
    assert list(forwarded) == positional
    assert kwargs["dstate"] == 128
    assert kwargs["ring_buffer_len"] == 16


def test_runner_class_is_exported_only_when_backend_provides_it(monkeypatch):
    # Both modules must be reloaded: the conditional export lives in the
    # submodule, and the package mirrors it into its own namespace.
    package = importlib.import_module("flashinfer.mamba")
    backend_module = importlib.import_module("flashinfer.mamba.checkpointing_ssu")

    monkeypatch.setattr(_backend, "has_symbol", lambda name: False)
    importlib.reload(backend_module)
    package = importlib.reload(package)
    assert not hasattr(package, "CheckpointingSSURunner")
    assert "CheckpointingSSURunner" not in package.__all__

    monkeypatch.setattr(_backend, "has_symbol", lambda name: True)
    monkeypatch.setattr(_backend, "resolve", lambda name: object)
    importlib.reload(backend_module)
    package = importlib.reload(package)
    assert package.CheckpointingSSURunner is object
    assert "CheckpointingSSURunner" in package.__all__

    monkeypatch.undo()
    importlib.reload(backend_module)
    importlib.reload(package)


def test_calling_without_a_backend_reports_the_gap(monkeypatch):
    monkeypatch.setattr(_backend, "_CACHE", {"module": None})
    with pytest.raises(NotImplementedError) as excinfo:
        selective_state_update(*[object() for _ in range(7)])
    message = str(excinfo.value)
    assert "selective_state_update" in message
    assert _backend.BACKEND_PACKAGE in message
