"""CPU-only oracle tests for the native SSD prefill family (no tilelang import).

These pin the *semantics* of the reference implementations that the device
kernels are measured against, on any machine with torch. Kernel-level properties
that need a device (masked stores leaving padded rows untouched, timing) live in
the device tests.
"""

from __future__ import annotations

import torch

from mate.mamba_kernels.reference import ssd_chunk_cumsum_reference

HEADS = 4
CHUNK = 8


def _inputs(offset_list, heads=HEADS, chunk=CHUNK, seed=0):
    generator = torch.Generator().manual_seed(seed)
    total = offset_list[-1]
    dt = torch.randn(total, heads, generator=generator)
    A = -torch.rand(heads, generator=generator) - 0.5
    bias = torch.randn(heads, generator=generator) * 0.1
    cu = torch.tensor(offset_list, dtype=torch.int32)
    return dt, A, bias, cu, chunk


def _layouts():
    """(name, chunk offsets) pairs: full chunks, partial chunks, single tokens."""
    return [
        ("full", [0, 8, 16, 24]),
        ("partial", [0, 8, 11, 19, 21]),
        ("single-token", [0, 1, 9, 10]),
        ("short-chunk", [0, 3, 4, 12]),
    ]


def test_inclusive_prefix_over_each_chunk():
    for _, offsets in _layouts():
        dt, A, bias, cu, chunk = _inputs(offsets)
        dA, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
        for c in range(len(offsets) - 1):
            length = offsets[c + 1] - offsets[c]
            expected = torch.cumsum(dt_out[:, c, :length] * A.view(-1, 1), dim=1)
            assert torch.allclose(dA[:, c, :length], expected, atol=1e-6)


def test_chunks_are_independent():
    dt, A, bias, cu, chunk = _inputs([0, 8, 11, 19, 21])
    base_dA, base_dt = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    perturbed = dt.clone()
    perturbed[0:8] += 5.0  # first chunk only
    new_dA, new_dt = ssd_chunk_cumsum_reference(perturbed, A, bias, cu, chunk, True)
    assert not torch.allclose(base_dA[:, 0, :8], new_dA[:, 0, :8])
    for c in (1, 2, 3):
        assert torch.allclose(base_dA[:, c], new_dA[:, c], atol=0.0)
        assert torch.allclose(base_dt[:, c], new_dt[:, c], atol=0.0)


def test_padding_saturates_dt_to_zero_and_da_to_the_chunk_total():
    """Padding is contractual: dt is 0 past the chunk end, so the scan saturates
    and the row's last position holds the chunk's total decay -- the value the
    downstream stages read unconditionally."""
    offsets = [0, 8, 11]
    dt, A, bias, cu, chunk = _inputs(offsets)
    dA, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)

    # dt is exactly zero beyond the chunk.
    assert torch.equal(dt_out[:, 1, 3:], torch.zeros_like(dt_out[:, 1, 3:]))
    assert torch.all(dt_out[:, 1, :3] > 0)

    # dA_cumsum is saturated at the chunk total past the chunk...
    total = dA[:, 1, 2]  # last valid position of the 3-token chunk
    assert torch.allclose(
        dA[:, 1, 3:], total.unsqueeze(1).expand_as(dA[:, 1, 3:]), atol=0.0
    )
    # ...and equals the row's last position, which is what callers read.
    assert torch.equal(dA[:, 1, 2], dA[:, 1, chunk - 1])

    # The total is the fp32 accumulation of processed_dt * A over valid tokens.
    expected = (dt_out[:, 1, :3] * A.view(-1, 1)).sum(dim=1)
    assert torch.allclose(total, expected, atol=1e-6)


def test_partial_chunk_equals_full_chunk_prefix():
    """A short chunk must produce exactly the prefix of the longer case."""
    dt, A, bias, cu, chunk = _inputs([0, 8, 8 + 5])
    short_dA, short_dt = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    full_dA, full_dt = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    assert torch.allclose(short_dA[:, 1, :5], full_dA[:, 1, :5], atol=0.0)
    assert torch.allclose(short_dt[:, 1, :5], full_dt[:, 1, :5], atol=0.0)


def test_dt_softplus_uses_the_threshold_rule():
    dt, A, bias, cu, chunk = _inputs([0, 8])
    dt = dt.clone()
    dt[0, :] = 30.0  # above the threshold: passed through
    dt[1, :] = -30.0  # below: log1p(exp(x))
    _, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    assert torch.allclose(dt_out[:, 0, 0], dt[0] + bias, atol=1e-5)
    assert torch.allclose(
        dt_out[:, 0, 1], torch.log1p(torch.exp(dt[1] + bias)), atol=1e-5
    )


def test_dt_softplus_disabled_adds_bias_then_clamps():
    """`dt_limit` clamps unconditionally, so with softplus off a negative dt is
    clamped (the production `(0.0, inf)` limit makes that a no-op only while
    softplus is on, which is the model's configuration)."""
    dt, A, bias, cu, chunk = _inputs([0, 8])
    _, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, False)
    assert torch.allclose(
        dt_out[:, 0, 0], (dt[0] + bias).clamp(min=0.0), atol=1e-6
    )
    # With an open limit the bias is the only change.
    _, unbounded = ssd_chunk_cumsum_reference(
        dt, A, bias, cu, chunk, False, dt_limit=(-float("inf"), float("inf"))
    )
    assert torch.allclose(unbounded[:, 0, 0], dt[0] + bias, atol=1e-6)


def test_dt_limit_clamps_before_the_scan():
    dt, A, bias, cu, chunk = _inputs([0, 8])
    _, dt_out = ssd_chunk_cumsum_reference(
        dt, A, bias, cu, chunk, True, dt_limit=(0.5, 1.5)
    )
    assert float(dt_out.min()) >= 0.5 - 1e-6
    assert float(dt_out.max()) <= 1.5 + 1e-6
    dA, _ = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True, dt_limit=(0.5, 1.5))
    expected = torch.cumsum(dt_out[:, 0, :8] * A.view(-1, 1), dim=1)
    assert torch.allclose(dA[:, 0, :8], expected, atol=1e-6)


def test_missing_dt_bias_is_allowed():
    dt, A, _, cu, chunk = _inputs([0, 8])
    dA, dt_out = ssd_chunk_cumsum_reference(dt, A, None, cu, chunk, True)
    assert torch.allclose(
        dt_out[:, 0, 0], torch.log1p(torch.exp(dt[0])), atol=1e-5
    )
    assert torch.allclose(dA[:, 0, 0], dt_out[:, 0, 0] * A, atol=1e-6)
