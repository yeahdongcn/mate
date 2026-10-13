# SPDX-License-Identifier: Apache-2.0
"""Marlin-frame to MATE-kernel-frame conversion, and the compute entry the consumer calls.

Read this first if you are writing the vllm-musa side. **What the consumer calls:**

    from mate.marlin import prepare_marlin_w4a16, apply_gptq_marlin_linear

    prepared = prepare_marlin_w4a16(
        qweight,        # upstream gptq_marlin_repack output, [padded_k/16, 2*padded_n] int32
        scales,         # upstream marlin_permute_scales output, (padded_k/group, padded_n)
        zero_points,    # upstream marlin_zero_points output, or None for symmetric
        size_k,         # logical K (before tile padding)
        size_n,         # logical N (before tile padding)
        group_size=128, # MATE's kernel is structurally group 128
    )
    out = apply_gptq_marlin_linear(input_a, prepared, bias=bias)

``prepared`` carries the kernel-ready tensors and the extents; it is a plain dataclass with no
behaviour, so the consumer can log it. Load-time work happens inside ``prepare_marlin_w4a16`` and
nowhere else, which is the point: the serving path must not do conversion per token.

The consumer hands over upstream's **output**, not raw GPTQ tensors — by the time vLLM's Marlin path
reaches a kernel it has already run the load-time utilities. So every direction below is an
*inverse* of an upstream layout, and each one says which direction the host path uses:

===========================  ==============================  ==========================
upstream (what arrives)      MATE kernel (what is needed)     function
===========================  ==============================  ==========================
``[K/16, 2N]`` int32 words   ``(N, K//2)`` int8, even k low   :func:`marlin_weight_to_musa`
permuted scales ``(G, N)``   plain ``(N, G)``, fp16           :func:`marlin_scales_to_musa`
packed + permuted zp         plain ``(N, G)``, fp16 or None   :func:`marlin_zero_points_to_musa`
===========================  ==============================  ==========================

Why the weight needs a real conversion and not a stride trick: hardware measurement on the existing
MATE kernel showed its int4 B operand is consumed as **consecutive k**, even k in the low nibble, odd
k in the high nibble — while upstream's words are a 16x64 tile map. On top of that the two sides
disagree about what a nibble *means* (upstream: GPTQ ``uint4b8``; MATE's atom: two's complement, which
the declared frame reconciles by ``u ^ 0x8``); see :func:`marlin_weight_to_musa`. No TME stride
descriptor expresses either difference, so route B pays one load-time pass.

Act order (``g_idx``): supported by *materialisation*, not by a kernel feature. The weight must have
been repacked with ``perm=argsort(g_idx)`` (upstream's own convention), which makes the permuted k
order contiguous in the repacked words; the per-group scales need no extra reordering, because they
are indexed by group rather than by k. Pass ``g_idx`` so the entry can check the contract it can
check, and note that the landed kernel asserts identity-only act order, so this is the only way to
serve an act-order checkpoint without changing the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # lazy by design, as in the rest of the package
    import torch
except ImportError:  # pragma: no cover - exercised on the numpy-only gate host
    torch = None  # type: ignore[assignment]

from . import compat
from . import quant

WEIGHT_B_DTYPE_INT8 = "int8"


@dataclass(frozen=True)
class MarlinW4A16KernelFrame:
    """Kernel-ready operands plus the extents the caller needs to interpret the output.

    ``padded_n`` may exceed ``logical_n``: MATE's kernel accepts a tile-padded N (padding works
    through zero *scales*, so the extra columns compute to zero), and the caller slices the output to
    ``logical_n``. ``padded_k`` never exceeds ``size_k`` in practice — under group 128 both upstream
    candidate families round K up to ``size_k`` itself — but the field is here so nobody has to
    assume it.
    """

    weight_b: "torch.Tensor"  # (padded_n, padded_k//2) int8
    scale_b: "torch.Tensor"  # (padded_n, padded_k//group_size) fp16
    zero_points_b: "torch.Tensor | None"  # (padded_n, padded_k//group_size) fp16, or None
    logical_n: int
    padded_n: int
    padded_k: int
    group_size: int
    has_zero_points: bool

    @property
    def padded(self) -> bool:
        return self.padded_n != self.logical_n

    def unpad_output(self, output):
        """Strip the padded N columns; a no-op when nothing was padded."""
        return compat.marlin_unpad_output(output, self.logical_n, self.padded_n)


def marlin_weight_to_musa(
    repacked, size_k: int, size_n: int, num_bits: int = 4, biased_codes: bool = True
) -> np.ndarray:
    """Upstream ``[size_k/16, 2*size_n]`` int32 words -> MATE ``(size_n, size_k//2)`` int8.

    Two layouts differ, and so do two *conventions* about what a nibble means; both have to be handled
    here, and only the second one is subtle.

    Layout: consecutive k, even k in the low nibble, odd k in the high nibble, against upstream's
    16x64 tile map. That is index arithmetic.

    Convention, and the reason for ``^ 0x8``. MATE's declared operand frame is *biased*: the
    documented dequant is ``w = scale * (q + nibble_bias - zero_point)`` and the packed-weight contract
    says each nibble "already holds the signed INT4 value ``unsigned_nibble - nibble_bias``"
    (``mate/gemm.py:1503-1507``, ``mate/jit/gemm/dense_w4a16_gemm.py:28-31,46-49``), with
    ``nibble_bias = 8`` and a symmetric checkpoint's zero point equal to it
    (``tests/test_dense_w4a16_gemm.py:42-47``: "GPTQ uint4b8 decodes as ``nibble - 8`` with no further
    bias", and the symmetric correction must vanish). So the atom is expected to read a presented
    nibble ``v`` as ``v - 8``. The atom actually reads two's complement -- measured on hardware:
    code 0 reads 0 and code 8 reads -8, which is why the neighbouring ``-8`` for code 8 was mistaken
    for ``u - 8``: that identity holds only *at* ``u == 8``. Upstream's words carry GPTQ ``uint4b8``
    codes ``u``, whose value is ``u - 8``, so presenting them verbatim is wrong for the whole negative
    half by ``+8 * scale`` per element -- a uniform error that no shape-based comparison catches, and
    one that shows up as exactly ``8 * scale * sum_k(a)`` per group in the R3 sensitivity gate.
    The unique presentation satisfying the declared frame is ``v = u ^ 0x8``, since
    ``s(u ^ 0x8) = u - 8`` for all 16 codes. That identity is asserted as a gate, not trusted.

    ``biased_codes=False`` is for words that are already in MATE's frame (two's-complement patterns
    whose signed read is the dequantized integer), e.g. a MATE-native packer; it is not a tolerance
    knob. Returns numpy because the conversion is pure index arithmetic;
    :func:`prepare_marlin_w4a16` moves it to the device once, at load time.
    """
    if num_bits != 4:
        raise NotImplementedError(f"MATE's Marlin wrapper is int4 only; got num_bits={num_bits}")
    from . import layout

    words = layout.unpack_gptq(repacked, size_k, size_n)  # (size_k//8, size_n) packed words
    codes = layout.unpack_rows(words, size_k, size_n)  # (size_k, size_n) uint4b8 codes
    if biased_codes:
        codes = codes ^ 0x8  # uint4b8 code -> the two's-complement pattern the atom must see

    half = codes.reshape(size_k // 2, 2, size_n)
    packed_kn = (half[:, 0, :] | (half[:, 1, :] << 4)).astype(np.uint8)  # even k low, odd k high
    return np.ascontiguousarray(packed_kn.T).view(np.int8)  # (size_n, size_k//2)


def marlin_scales_to_musa(permuted, size_k: int, size_n: int, group_size: int):
    """Inverse of ``marlin_permute_scales``: permuted ``(G, size_n)`` -> plain ``(size_n, G)``.

    Direction matters and is easy to get backwards: the *host* path receives scales that are already
    permuted, so it needs this inverse. ``quant.marlin_permute_scales`` stays upstream-identical and
    is used by the conversion path (and byte-gated by the ported tests); it is not what the serving
    path calls. The transpose is not cosmetic — upstream's layout is K-major ``(G, size_n)`` while the
    kernel's scale operand is ``(n, k_blocks)``.
    """
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError("marlin_scales_to_musa needs torch")
    return _unpermute(permuted, size_k, size_n, group_size).t().contiguous().to(torch.float16)


def _unpermute(permuted, size_k: int, size_n: int, group_size: int):
    scale_perm, scale_perm_single = quant.get_scale_perms()
    if group_size < size_k and group_size != -1:
        width = len(scale_perm)
        inverse = np.argsort(np.array(scale_perm))
    else:
        width = len(scale_perm_single)
        inverse = np.argsort(np.array(scale_perm_single))
    rows = permuted.reshape((-1, width))[:, inverse]
    return rows.reshape((-1, size_n))


def marlin_zero_points_to_musa(packed_zp, size_k: int, size_n: int, group_size: int):
    """Inverse of ``marlin_zero_points``: packed + permuted zp -> plain ``(size_n, G)`` fp16.

    ``quant.unpack_marlin_zero_points`` undoes both the column packing and the permutation and
    returns ``(G, size_n)`` in the same K-major frame as upstream's scales, which is what the
    kernel's zero-point operand mirrors; this adds the transpose and the fp16 cast the staging
    expects. The values are upstream's **raw** per-group zero points -- the ``zero_point`` term of
    the documented ``w = scale * (q + nibble_bias - zero_point)``, with the same shape and dtype
    contract as ``scale_b`` (``mate/gemm.py:1512-1515``). Pass them raw: the op applies the
    ``(nibble_bias - zero_point)`` term itself, so a stream already folded by
    ``fold_zero_points`` would be folded twice, and a *symmetric* checkpoint's zero point is
    ``nibble_bias`` (8), which folds to exactly zero. Passing 0 where 8 belongs is the double-fold
    error in its most common shape.
    """
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError("marlin_zero_points_to_musa needs torch")
    plain = quant.unpack_marlin_zero_points(packed_zp, size_k, size_n)
    return plain.t().contiguous().to(torch.float16)


def prepare_marlin_w4a16(
    qweight,
    scales,
    zero_points=None,
    size_k: int | None = None,
    size_n: int | None = None,
    group_size: int = compat.MUSA_MARLIN_GROUP_SIZE,
    g_idx=None,
    device=None,
) -> MarlinW4A16KernelFrame:
    """The single load-time entry: upstream Marlin frame in, MATE kernel frame out.

    ``qweight`` and ``scales`` are upstream's already-converted tensors. When ``size_k``/``size_n``
    are omitted they are recovered with ``marlin_repacked_nk``, i.e. the *padded* extents, and
    ``logical_n`` is then the padded N — pass the logical extents when the model padded them, and the
    returned frame's ``unpad_output`` will slice the result.

    Refusals are deliberate and carry reasons, because the consumer must not discover these by
    crashing: a group size other than 128, ``K % 256 != 0`` (a TP-sharded ``down`` at TP=2 is
    rank-local K=4480, the concrete casualty), and a repack whose K axis is not tile-aligned.
    """
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError("prepare_marlin_w4a16 needs torch; the numpy gates live in mate.marlin.reference")

    qweight = qweight.contiguous()
    padded_n, padded_k = compat.marlin_repacked_nk(qweight, num_bits=4)
    if size_k is None:
        size_k = padded_k
    if size_n is None:
        size_n = padded_n

    if padded_k < size_k or padded_n < size_n:
        raise ValueError(
            f"the repacked weight is smaller than the logical shape: packed (n={padded_n}, k={padded_k}) "
            f"vs logical (n={size_n}, k={size_k}); pad the weight before repacking, not after"
        )

    ok, reason = compat.check_musa_marlin_supports_shape(size_k, size_n, group_size)
    if not ok:
        raise ValueError(reason)

    if g_idx is not None:
        # Materialisation contract: argsort(g_idx) must already be baked into the repacked words.
        g = g_idx.detach().cpu().to(torch.int64).numpy().reshape(-1)
        if g.shape[0] not in (size_k, padded_k):
            raise ValueError(
                f"g_idx has {g.shape[0]} entries but the weight has K={size_k}; pass the g_idx that "
                "produced this repack"
            )
        order = np.argsort(g, kind="stable")
        if not np.array_equal(order, np.arange(order.shape[0])):
            raise ValueError(
                "act order is present but this wrapper cannot verify that the k permutation was "
                "materialised: repack the weights with perm=argsort(g_idx) (upstream's convention, "
                "which makes the permuted k order contiguous) and drop g_idx here; the landed kernel "
                "asserts identity-only act order, so an unmaterialised g_idx would be served wrongly. "
                f"({int((order != np.arange(order.shape[0])).sum())} of {order.shape[0]} k positions "
                "are out of order in this g_idx.)"
            )

    weight_b = torch.from_numpy(marlin_weight_to_musa(qweight.cpu().numpy(), padded_k, padded_n))
    scale_b = marlin_scales_to_musa(scales, padded_k, padded_n, group_size)
    zp_b = (
        marlin_zero_points_to_musa(zero_points, padded_k, padded_n, group_size)
        if zero_points is not None
        else None
    )
    if device is not None:
        weight_b = weight_b.to(device)
        scale_b = scale_b.to(device)
        zp_b = zp_b.to(device) if zp_b is not None else None
    else:
        weight_b = weight_b.to(scales.device)
        scale_b = scale_b.to(scales.device)
        zp_b = zp_b.to(scales.device) if zp_b is not None else None

    return MarlinW4A16KernelFrame(
        weight_b=weight_b.contiguous(),
        scale_b=scale_b.contiguous(),
        zero_points_b=zp_b.contiguous() if zp_b is not None else None,
        logical_n=size_n,
        padded_n=padded_n,
        padded_k=padded_k,
        group_size=group_size,
        has_zero_points=zero_points is not None,
    )


def apply_gptq_marlin_linear(input_a, prepared: MarlinW4A16KernelFrame, bias=None, num_mps=None):
    """Run one W4A16 linear on a prepared frame. No branch on M: decode and prefill take this path.

    The symmetric case passes ``zero_points=None``, which is inert by design on the kernel's symmetric
    instantiation (its ``sym`` build never stages the tensor). The caller slices the result with
    ``prepared.unpad_output`` when the model padded N.
    """
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError("apply_gptq_marlin_linear needs torch")

    from mate.gemm import dense_w4a16_gemm  # imported here so this module stays torch-free

    out = torch.empty(
        input_a.shape[:-1] + (prepared.padded_n,), dtype=input_a.dtype, device=input_a.device
    )
    dense_w4a16_gemm(
        input_a,
        prepared.weight_b,
        prepared.scale_b,
        out,
        zero_points=prepared.zero_points_b,
        g_idx=None,
        num_mps=num_mps,
    )
    if bias is not None:
        out = out + bias
    return out
