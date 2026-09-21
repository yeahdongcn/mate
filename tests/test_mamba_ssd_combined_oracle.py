"""CPU end-to-end parity of the packed SSD orchestration.

``mate.mamba.ssd_combined_fwd_varlen`` is checked against the *independent*
FlashInfer-MUSA reference implementation that lives in the fork — a separate
transcription of the same contract, written by a different author for a different
runtime. A disagreement means one of them is wrong; agreement over partial chunks,
sequence boundaries, initial states and the epilogue is real evidence.

The native tilelang stages need MUSA, so the stage table is patched with
reference-backed launchers that keep the native calling convention (preallocated
outputs, keyword-only flags). The plumbing under test — metadata handling, the
workspace, the entering-state rule, the return contract — is the same code that
runs on device.

The oracle's file is not part of this repository: the module skips when it is
absent so the suite stays runnable anywhere. Point ``MATE_SSD_ORACLE`` at it to
run elsewhere.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest
import torch

DEFAULT_ORACLE = pathlib.Path(
    "/home/xiaodongye/ws/flashinfer-nemotron-musa/flashinfer/mamba/musa_reference.py"
)

# The tolerance the fork's own device tests use for exactly this comparison:
# the native path rounds the running state to state_dtype at every chunk
# boundary while the oracle keeps it in fp32 across the sequence.
RTOL = 0.05
ATOL = 0.02


def _load_oracle():
    path = pathlib.Path(os.environ.get("MATE_SSD_ORACLE", DEFAULT_ORACLE))
    if not path.exists():
        pytest.skip(f"FlashInfer-MUSA oracle not present at {path}")
    spec = importlib.util.spec_from_file_location("musa_reference_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ssd_combined_fwd_varlen_musa_reference


@pytest.fixture(scope="module")
def oracle():
    return _load_oracle()


def reference_stage_table(round_operands=True):
    """Reference-backed launchers that keep the native calling convention.

    ``round_operands=False`` drops the pre-dot rounding to the activation dtype,
    turning the chain into a pure fp32 transcription of the same arithmetic.
    """
    from mate.mamba_kernels.reference import (
        ssd_chunk_cumsum_reference,
        ssd_chunk_scan_reference,
        ssd_chunk_state_reference,
    )

    def cumsum(
        dt, A, dt_bias, cu_chunk_seqlens, chunk_size, *, dt_softplus, dt_limit,
        dA_cumsum, dt_out,
    ):
        dA, processed = ssd_chunk_cumsum_reference(
            dt, A, dt_bias, cu_chunk_seqlens, chunk_size, dt_softplus, dt_limit
        )
        dA_cumsum.copy_(dA)
        dt_out.copy_(processed)
        return dA_cumsum, dt_out

    def chunk_state(x, B, dt_out, dA_cumsum, cu_chunk_seqlens, chunk_size, *, states):
        computed = ssd_chunk_state_reference(
            x, B, dt_out, dA_cumsum, cu_chunk_seqlens, chunk_size, round_operands
        )
        states.copy_(computed)
        return states

    def state_passing(
        states, dA_cumsum, initial_states, last_chunk_indices, seq_idx, *, state_dtype
    ):
        nchunks, heads, dim, dstate = states.shape
        block = dA_cumsum.shape[2]
        for chunk in range(nchunks):
            opens_sequence = chunk == 0 or int(seq_idx[chunk]) != int(seq_idx[chunk - 1])
            if opens_sequence:
                if initial_states is None:
                    running = torch.zeros(
                        (heads, dim, dstate), dtype=torch.float32, device=states.device
                    )
                else:
                    running = initial_states[int(seq_idx[chunk])].to(torch.float32)
            # decay the entering state across the chunk, then add the chunk's
            # own contribution; rounded at every boundary, as the fast path does
            decay = torch.exp(dA_cumsum[:, chunk, block - 1]).to(torch.float32)
            running = (
                decay.unsqueeze(1).unsqueeze(2) * running
                + states[chunk].to(torch.float32)
            )
            states[chunk].copy_(running.to(state_dtype))
        return states

    def bmm(C, B, cu_chunk_seqlens, chunk_size, *, cb):
        offsets = [int(v) for v in cu_chunk_seqlens.reshape(-1).tolist()]
        c32, b32 = C.to(torch.float32), B.to(torch.float32)
        cb.zero_()
        for chunk in range(len(offsets) - 1):
            start, end = offsets[chunk], offsets[chunk + 1]
            limit = end - start
            cb[chunk, :, :limit, :limit] = torch.einsum(
                "ign,jgn->gij", c32[start:end], b32[start:end]
            )
        return cb

    def chunk_scan(
        x, C, CB, dt_out, dA_cumsum, states, initial_states, seq_idx,
        cu_chunk_seqlens, out, *, D_param, z,
    ):
        return ssd_chunk_scan_reference(
            x,
            C,
            torch.zeros_like(C),
            dt_out,
            dA_cumsum,
            states,
            initial_states,
            seq_idx,
            cu_chunk_seqlens,
            dA_cumsum.shape[2],
            D_param=D_param,
            z=z,
            out=out,
            CB=CB,
            round_operands=round_operands,
        )

    return {
        "cumsum": cumsum,
        "chunk_state": chunk_state,
        "state_passing": state_passing,
        "bmm": bmm,
        "chunk_scan": chunk_scan,
    }


@pytest.fixture(autouse=True)
def reference_stages(monkeypatch):
    """Patch the five native stage launchers with reference-backed equivalents."""
    from mate import mamba

    table = reference_stage_table()
    monkeypatch.setattr(mamba, "_stage", lambda name: table[name])
    mamba.reset_ssd_workspaces()
    yield
    mamba.reset_ssd_workspaces()


def _tensors(case, device="cpu", seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = case["cu_chunk_seqlens"][-1]
    heads, dim, groups, dstate = case["heads"], case["dim"], case["groups"], case["dstate"]
    x = torch.randn(tokens, heads, dim, generator=generator).to(torch.bfloat16).to(device)
    dt = torch.randn(tokens, heads, generator=generator).abs().to(device) + 0.1
    A = (-torch.rand(heads, generator=generator) - 0.5).to(device)
    B = torch.randn(tokens, groups, dstate, generator=generator).to(torch.bfloat16).to(device)
    C = torch.randn(tokens, groups, dstate, generator=generator).to(torch.bfloat16).to(device)
    return x, dt, A, B, C


# The recipe from the specification's oracle section: one 5-token sequence and one
# 17-token sequence, so the first chunk is partial and both shapes of chunk appear.
TINY = {
    # sequence 0 is 5 tokens (one partial chunk), sequence 1 is 17 (three chunks)
    "cu_seqlens": [0, 5, 22],
    "cu_chunk_seqlens": [0, 5, 8, 16, 22],
    "last_chunk_indices": [0, 3],
    "seq_idx": [0, 1, 1, 1],
    "heads": 4,
    "dim": 8,
    "groups": 2,
    "dstate": 16,
    "chunk_size": 8,
}

FULL_CHUNKS = {
    "cu_seqlens": [0, 16, 32],
    "cu_chunk_seqlens": [0, 8, 16, 24, 32],
    "last_chunk_indices": [1, 3],
    "seq_idx": [0, 0, 1, 1],
    "heads": 4,
    "dim": 8,
    "groups": 2,
    "dstate": 16,
    "chunk_size": 8,
}


def _run(case, oracle, *, with_D=True, with_z=True, with_initial=False, extra=None):
    from mate.mamba import ssd_combined_fwd_varlen

    device = "cpu"
    x, dt, A, B, C = _tensors(case, device)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    generator = torch.Generator().manual_seed(11)
    batch = len(case["cu_seqlens"]) - 1
    kwargs = dict(
        chunk_size=case["chunk_size"],
        D=torch.randn(case["heads"], case["dim"], generator=generator) if with_D else None,
        z=torch.randn_like(x).to(torch.bfloat16) if with_z else None,
        dt_bias=torch.randn(case["heads"], generator=generator) * 0.1,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        initial_states=(
            torch.randn(
                batch, case["heads"], case["dim"], case["dstate"], generator=generator
            ).to(torch.bfloat16)
            if with_initial
            else None
        ),
    )
    if extra:
        kwargs.update(extra)

    mine_out = torch.zeros_like(x)
    expected_out = torch.zeros_like(x)
    mine_states = ssd_combined_fwd_varlen(
        x, dt, A, B, C, out=mine_out, return_intermediate_states=True, **meta, **kwargs
    )
    expected_states = oracle(
        x, dt, A, B, C, out=expected_out, return_intermediate_states=True, **meta, **kwargs
    )
    return mine_out, mine_states, expected_out, expected_states


def assert_states_close(mine, expected, nchunks, case):
    """Compare states under the documented rounding difference.

    The native path rounds the running state to ``state_dtype`` at every chunk
    boundary while the oracle keeps it in fp32 across a sequence, so the two
    disagree by the accumulated bf16 resolution -- and elements produced by
    cancellation inherit the error of the *scale*, not of their own size. The
    bound therefore scales with the largest state and with the number of
    boundaries instead of being a per-element tolerance.

    The tolerance must stay far below the size of a structural error: a mis-
    contracted operand axis in the state stage measured 3.8 here, an order of
    magnitude above this bound.
    """
    scale = float(expected.float().abs().max())
    bound = ATOL + RTOL * scale + nchunks * 2 ** -9 * scale
    worst = float((mine.float() - expected.float()).abs().max())
    assert worst <= bound, (
        f"{case}: worst state difference {worst:.4f} exceeds {bound:.4f} "
        f"(state scale {scale:.3f}, {nchunks} chunk boundaries)"
    )


def assert_out_close(mine, expected):
    """Compare ``out`` with the fork's tolerance applied to the output scale.

    The fork's device tests use ``rtol=0.05, atol=0.02`` per element, which holds
    for model-like activations where outputs are O(1). These cases use
    unnormalized random inputs whose outputs reach 33, and the error floor set by
    the bf16 output quantization plus the pre-dot operand rounding is inherited
    from that scale -- cancellation then makes it large *relatively* on elements
    near zero while staying at 0.05% of the scale in absolute terms. Measured:
    max absolute difference 1.56e-02 on a scale of 33, with no element above the
    fork's own atol. A structural error moves every element, which the mean bound
    below catches.
    """
    scale = float(expected.float().abs().max())
    worst = float((mine.float() - expected.float()).abs().max())
    mean = float((mine.float() - expected.float()).abs().mean())
    assert worst <= ATOL + RTOL * scale, (
        f"worst output difference {worst:.4e} exceeds {ATOL + RTOL * scale:.4e} "
        f"(output scale {scale:.2f})"
    )
    assert mean <= 1e-3 * scale, (
        f"mean output difference {mean:.4e} exceeds 0.1% of the scale {scale:.2f}; "
        f"this is not a cancellation artifact"
    )


@pytest.mark.parametrize("case", [TINY, FULL_CHUNKS], ids=["partial-first-chunk", "full-chunks"])
def test_out_and_states_match_the_independent_oracle(case, oracle):
    mine_out, mine_states, expected_out, expected_states = _run(case, oracle)
    assert_out_close(mine_out, expected_out)
    assert_states_close(mine_states, expected_states, len(case["seq_idx"]), case["cu_chunk_seqlens"])


def _states_fp32(case, oracle, round_operands):
    """Run the chain in fp32 with the pre-dot rounding on or off."""
    from mate import mamba

    x, dt, A, B, C = _tensors(case)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    kwargs = dict(
        chunk_size=case["chunk_size"],
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        state_dtype=torch.float32,
        return_intermediate_states=True,
    )
    saved = mamba._stage
    table = reference_stage_table(round_operands=round_operands)
    mamba._stage = lambda name: table[name]
    mamba.reset_ssd_workspaces()
    try:
        mine = mamba.ssd_combined_fwd_varlen(
            x, dt, A, B, C, out=torch.zeros_like(x), **meta, **kwargs
        )
    finally:
        mamba._stage = saved
        mamba.reset_ssd_workspaces()
    expected = oracle(x, dt, A, B, C, out=torch.zeros_like(x), **meta, **kwargs)
    return mine, expected


def test_fp32_chain_is_equivalent_to_the_oracle(oracle):
    """Disabling the pre-dot rounding turns the chain into a pure fp32
    transcription of the same arithmetic, and it then reproduces the oracle to
    fp32 summation order. This is what makes the shipped path's residual
    difference attributable to the rounding convention rather than to a defect."""
    mine, expected = _states_fp32(TINY, oracle, round_operands=False)
    assert mine.dtype == torch.float32
    scale = max(1.0, float(expected.abs().max()))
    worst = float((mine - expected).abs().max())
    assert worst <= 1e-4 * scale, (
        f"pure fp32 states differ by {worst:.3e} (scale {scale:.3f}); the "
        f"orchestration is not equivalent to the oracle"
    )


def test_the_pre_dot_rounding_costs_about_one_bf16_ulp(oracle):
    """The shipped path rounds the scaled operands to the activation dtype before
    every dot, so it is *by construction* less accurate than an fp32 oracle. The
    measurement, which bounds how much a state comparison may be relaxed:

        rounding on  : 1.35e-02 = 0.123% of the state scale
        rounding off : 1.43e-06 = 0.000013%

    bf16's own resolution is 2**-9 = 0.195%, so the convention costs about one
    ulp and nothing more. A structural error in this stage measured 3.8 (44%),
    forty times this bound, so the bound still fails loudly on real defects.
    """
    mine, expected = _states_fp32(TINY, oracle, round_operands=True)
    scale = float(expected.abs().max())
    worst = float((mine - expected).abs().max())
    assert 0 < worst <= 2 ** -8 * scale, (
        f"shipped states differ from the oracle by {worst:.4e} "
        f"({worst / scale:.4%} of scale {scale:.2f}); expected a difference "
        f"bounded by the bf16 operand rounding"
    )


def test_partial_chunks_do_not_leave_garbage_in_the_state_buffer(oracle):
    """The first chunk of the 5-token sequence is partial: its padded rows must not
    contribute, so the state must match the oracle rather than explode."""
    _, mine_states, _, expected_states = _run(TINY, oracle)
    assert torch.isfinite(mine_states.float()).all()
    assert_states_close(mine_states, expected_states, len(TINY["seq_idx"]), TINY)


def test_final_states_are_selected_by_last_chunk_indices(oracle):
    from mate.mamba import ssd_combined_fwd_varlen

    case = TINY
    x, dt, A, B, C = _tensors(case)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    full = ssd_combined_fwd_varlen(
        x, dt, A, B, C, chunk_size=case["chunk_size"], return_intermediate_states=True, **meta
    )
    final = ssd_combined_fwd_varlen(
        x, dt, A, B, C, chunk_size=case["chunk_size"], return_intermediate_states=False, **meta
    )
    expected = oracle(
        x, dt, A, B, C, chunk_size=case["chunk_size"], return_intermediate_states=False, **meta
    )
    assert final.shape == (2, case["heads"], case["dim"], case["dstate"])
    assert final.dtype == torch.bfloat16  # C.dtype when state_dtype is unset
    torch.testing.assert_close(final.float(), expected.float(), rtol=RTOL, atol=ATOL)
    # The selected rows are exactly the last chunk of each sequence.
    for b, chunk in enumerate(case["last_chunk_indices"]):
        assert torch.equal(final[b], full[chunk])


def test_state_dtype_overrides_the_default_and_is_honoured(oracle):
    from mate.mamba import ssd_combined_fwd_varlen

    case = FULL_CHUNKS
    x, dt, A, B, C = _tensors(case)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    states = ssd_combined_fwd_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=case["chunk_size"],
        state_dtype=torch.float32,
        return_intermediate_states=True,
        **meta,
    )
    assert states.dtype == torch.float32


def test_checkpoint_arguments_raise_instead_of_being_ignored(oracle):
    from mate.mamba import ssd_combined_fwd_varlen

    case = TINY
    x, dt, A, B, C = _tensors(case)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    with pytest.raises(NotImplementedError, match="checkpoint_"):
        ssd_combined_fwd_varlen(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=case["chunk_size"],
            checkpoint_token_indices=torch.zeros(1, dtype=torch.int32),
            **meta,
        )


def test_workspace_is_reused_across_calls(oracle):
    """Steady state must not allocate the stage scratch again."""
    from mate import mamba

    _run(FULL_CHUNKS, oracle)
    first = dict(mamba._WORKSPACES)
    _run(FULL_CHUNKS, oracle)
    second = dict(mamba._WORKSPACES)
    assert first.keys() == second.keys()
    for key, space in first.items():
        assert space.dt_out is second[key].dt_out
        assert space.states is second[key].states
        assert space.CB is second[key].CB
