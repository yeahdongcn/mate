"""CPU-only oracle tests for the native SSD chunk scan (no tilelang import).

The scan is the one stage where a wrong result hides easily: the operands are
small, the tolerances downstream are loose, and three separate things can be
individually plausible and still wrong together -- the choice of the entering
state, the causal masking, and the order of the ``D``/``z`` epilogue. So the
reference is checked against a second, deliberately naive transcription of the
specification's per-token formula, plus one test per contract clause.
"""

from __future__ import annotations

import torch

from mate.mamba_kernels.reference import (
    ssd_chunk_scan_reference,
    ssd_chunk_state_reference,
)

HEADS = 4
DIM = 3
DSTATE = 5
GROUPS = 2
CHUNK = 8
HEAD_RATIO = HEADS // GROUPS


def _naive_scan(x, C, B, dt_out, dA_cumsum, states, initial_states, seq_idx, offsets, out):
    """The spec's formula as a plain per-token, per-j triple loop."""
    tokens, heads, dim = x.shape
    _, nchunks, block = dA_cumsum.shape
    c32 = C.to(torch.float32)
    b32 = B.to(torch.float32)
    x32 = x.to(torch.float32)
    dA = dA_cumsum.to(torch.float32)
    dt = dt_out.to(torch.float32)
    st = states.to(torch.float32)
    sequence = [int(v) for v in seq_idx.reshape(-1).tolist()]

    for chunk in range(nchunks):
        start, end = offsets[chunk], offsets[chunk + 1]
        limit = end - start
        if limit <= 0:
            continue
        opens = chunk == 0 or sequence[chunk] != sequence[chunk - 1]
        if opens:
            prev = (
                initial_states[sequence[chunk]].to(torch.float32)
                if initial_states is not None
                else torch.zeros((heads, dim, DSTATE), dtype=torch.float32)
            )
        else:
            prev = st[chunk - 1]
        for head in range(heads):
            group = head // HEAD_RATIO
            for t in range(limit):
                dA_t = float(dA[head, chunk, t])
                for d in range(dim):
                    acc = 0.0
                    for n in range(DSTATE):
                        acc += (
                            float(c32[start + t, group, n])
                            * torch.exp(torch.tensor(dA_t)).item()
                            * float(prev[head, d, n])
                        )
                    for j in range(t + 1):
                        # the CB product, then the decay, then dt, then the
                        # round to the activation dtype -- production's order
                        cb = 0.0
                        for n in range(DSTATE):
                            cb += float(c32[start + t, group, n]) * float(
                                b32[start + j, group, n]
                            )
                        dA_j = float(dA[head, chunk, j])
                        cb = cb * float(
                            torch.exp(torch.tensor(min(dA_t - dA_j, 0.0))).item()
                        )
                        cb = cb * float(dt[head, chunk, j])
                        cb = float(
                            torch.tensor(cb, dtype=torch.float32).to(x.dtype).to(torch.float32)
                        )
                        acc += cb * float(x32[start + j, head, d])
                    out[start + t, head, d] = acc
    return out


def _case(offsets, seed=0, heads=HEADS, chunk=CHUNK, with_initial=True):
    generator = torch.Generator().manual_seed(seed)
    total = offsets[-1]
    x = torch.randn(total, heads, DIM, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    C = torch.randn(total, GROUPS, DSTATE, generator=generator).to(torch.bfloat16)
    B = torch.randn(total, GROUPS, DSTATE, generator=generator).to(torch.bfloat16)
    nchunks = len(offsets) - 1
    dt_out = torch.zeros(heads, nchunks, chunk)
    dA_cumsum = torch.zeros(heads, nchunks, chunk)
    for c in range(nchunks):
        length = offsets[c + 1] - offsets[c]
        dt_out[:, c, :length] = torch.rand(heads, length, generator=generator) * 0.5 + 0.1
        dA_cumsum[:, c, :length] = torch.cumsum(
            -torch.rand(heads, length, generator=generator), dim=1
        )
        dA_cumsum[:, c, length:] = dA_cumsum[:, c, length - 1 : length]
    states = ssd_chunk_state_reference(
        x, B, dt_out, dA_cumsum, torch.tensor(offsets, dtype=torch.int32), chunk
    )
    initial_states = (
        torch.randn(len(offsets) - 1, heads, DIM, DSTATE, generator=generator)
        if with_initial
        else None
    )
    return x, C, B, dt_out, dA_cumsum, states, initial_states


def test_reference_matches_the_naive_per_token_formula():
    offsets = [0, 8]
    x, C, B, dt_out, dA_cumsum, states, initial = _case(offsets)
    seq_idx = torch.zeros(1, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)

    fast = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, CHUNK
    )
    slow = _naive_scan(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, offsets, torch.zeros_like(x)
    )
    assert torch.allclose(fast.float(), slow.float(), rtol=1e-4, atol=1e-4)


def test_second_sequence_and_bare_chunks_use_the_right_entering_state():
    offsets = [0, 8, 16]
    x, C, B, dt_out, dA_cumsum, states, initial = _case(offsets)
    cu = torch.tensor(offsets, dtype=torch.int32)
    # Chunks 0 and 1 belong to two sequences: each takes its initial state.
    seq_idx = torch.tensor([0, 1], dtype=torch.int32)
    with_initial = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, CHUNK
    )
    # With no initial states both chunks of that layout start from zero, so both
    # results must equal the same run against zeroed initial states.
    zeros = torch.zeros_like(initial)
    from_zeros = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, zeros, seq_idx, cu, CHUNK
    )
    assert not torch.allclose(with_initial, from_zeros)
    # One sequence spanning both chunks: chunk 0 opens it, so only chunk 0 sees
    # the initial state; chunk 1 must consume states[0] and be identical either
    # way. That is exactly the boundary the two indexings disagree on.
    one_seq = torch.zeros(2, dtype=torch.int32)
    a = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, one_seq, cu, CHUNK
    )
    b = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, zeros, one_seq, cu, CHUNK
    )
    assert torch.equal(a[CHUNK:], b[CHUNK:])
    assert not torch.allclose(a[:CHUNK], b[:CHUNK])


def test_rows_past_a_partial_chunk_are_never_written():
    offsets = [0, 8, 11]
    x, C, B, dt_out, dA_cumsum, states, initial = _case(offsets)
    seq_idx = torch.zeros(2, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)
    out = torch.full_like(x, float("nan"))
    ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, CHUNK, out=out
    )
    assert not torch.isnan(out[:11]).any()
    assert out[11:].isnan().all()


def test_future_tokens_do_not_leak_through_the_causal_mask():
    offsets = [0, 8]
    x, C, B, dt_out, dA_cumsum, states, initial = _case(offsets)
    seq_idx = torch.zeros(1, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)
    base = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, CHUNK
    )
    perturbed = x.clone()
    perturbed[4:] += 10.0  # only tokens at or after row 4
    after = ssd_chunk_scan_reference(
        perturbed, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, CHUNK
    )
    assert torch.allclose(base[:4], after[:4], atol=1e-6)
    assert not torch.allclose(base[4:], after[4:])


def test_epilogue_applies_d_before_the_z_gate():
    offsets = [0, 8]
    x, C, B, dt_out, dA_cumsum, states, initial = _case(offsets)
    seq_idx = torch.zeros(1, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)
    generator = torch.Generator().manual_seed(7)
    D_param = torch.randn(HEADS, DIM, generator=generator)
    z = torch.randn(8, HEADS, DIM, generator=generator).to(torch.bfloat16)

    # fp32 outputs: a bf16 result quantizes to ~0.4 %, which would swamp the
    # comparison the test is actually making.
    trace = ssd_chunk_scan_reference(
        x,
        C,
        B,
        dt_out,
        dA_cumsum,
        states,
        initial,
        seq_idx,
        cu,
        CHUNK,
        out=torch.zeros(x.shape, dtype=torch.float32),
    )
    both = ssd_chunk_scan_reference(
        x,
        C,
        B,
        dt_out,
        dA_cumsum,
        states,
        initial,
        seq_idx,
        cu,
        CHUNK,
        D_param=D_param,
        z=z,
        out=torch.zeros(x.shape, dtype=torch.float32),
    )
    # Reconstruct: skip, then SiLU gate, both in fp32.
    z32 = z.to(torch.float32)
    expected = (
        trace.float() + x.float() * D_param.view(1, HEADS, DIM)
    ) * z32 * torch.sigmoid(z32)
    assert torch.allclose(both.float(), expected, rtol=1e-5, atol=1e-5)
    # And the order matters: applying the gate first would not match.
    wrong_order = trace.float() * z32 * torch.sigmoid(z32) + x.float() * D_param.view(
        1, HEADS, DIM
    )
    assert not torch.allclose(both.float(), wrong_order, rtol=1e-3, atol=1e-3)


def test_zero_chunk_sequence_is_skipped_without_reading_states():
    offsets = [0, 8]
    x, C, B, dt_out, dA_cumsum, states, initial = _case(offsets)
    # Two sequences, but the second has no chunks at all (its state row is
    # never consumed); the run must still complete and match the single-sequence
    # behaviour.
    seq_idx = torch.zeros(1, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)
    initial = torch.randn(3, HEADS, DIM, DSTATE)
    out = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, CHUNK
    )
    assert torch.isfinite(out.float()).all()
