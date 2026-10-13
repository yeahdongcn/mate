# SPDX-License-Identifier: Apache-2.0
"""Shippable MUSA repack for the upstream Marlin W4A16 weight layout (the GPTQ one the model uses).

``layout_ref.py`` holds a *scalar* transcription of ``gptq_marlin_repack_kernel`` and stays the
oracle. This module is the form MATE would ship: the index arithmetic is vectorised per 16x64 tile
and the outer loops are over tiles, which is exactly the grid a device kernel wants — there is no
Python loop over elements. ``check_repack_musa.py`` asserts the two agree byte-for-byte on the map
and on values, so this file is a port candidate while the oracle stays independent evidence.

Offset equation, 4-bit non-a8bit path (one tile; ``w`` is the unpacked ``[size_k, size_n]`` nibbles)::

    out[k_tile, n_tile * TILE_WORDS + word] |= value << (4 * nib)
    value     = w[k_tile * 16 + k_local, n_tile * 64 + n_local]
    k_local   = 2 * ((word // 4) % 4) + TC_OFFSETS[src % 4]
    n_local   = (word % 4) * 16 + word // 16 + 8 * (src >= 4)
    src       = PACK_IDX[nib]
    PACK_IDX = (0, 2, 4, 6, 1, 3, 5, 7),  TC_OFFSETS = (0, 1, 8, 9)
    TILE_WORDS = 128,  word = th_id * 4 + warp_id   (th_id 0..31, warp_id 0..3)

Sources: ``gptq_marlin_repack.cu:122-137, 149-162, 196, 200-209``, ``marlin.cuh:30,38-39``. Each
``(word, nib)`` slot is written exactly once, so the OR is order-independent, and the per-tile map is
a permutation of the 1024 ``(k, n)`` slots — asserted, not assumed, by the checker.

Model path: always this repack. AWQ checkpoints and their zero points are converted to the GPTQ
packed layout at load, so there is no second repack (contract §0 item 7). Values here are unsigned
nibbles; ``uint4b8``'s fused ``-8`` belongs to the dequant, not to the repack.
"""

from __future__ import annotations

import numpy as np

TILE_K = 16
TILE_N = 64
PACK_IDX = (0, 2, 4, 6, 1, 3, 5, 7)
TC_OFFSETS = (0, 1, 8, 9)
TILE_WORDS = TILE_K * TILE_N // 8  # 128 int32 words per 16x64 int4 tile


def tile_index_arrays():
    """Per-tile ``[TILE_WORDS, 8]`` coordinate arrays: ``(k_local, n_local, word, nib)``.

    Shape-independent by construction: every tile uses the same map, translated by
    ``k_tile * 16`` / ``n_tile * 64``. This is the per-thread/per-word work a kernel would do.
    """
    words = np.arange(TILE_WORDS)
    nib = np.arange(8)
    src = np.asarray(PACK_IDX)[nib]
    k_local = (2 * ((words // 4) % 4))[:, None] + np.asarray(TC_OFFSETS)[src % 4][None, :]
    n_local = ((words % 4) * 16 + words // 16)[:, None] + 8 * (src >= 4)[None, :]
    word = np.broadcast_to(words[:, None], (TILE_WORDS, 8))
    nib = np.broadcast_to(nib[None, :], (TILE_WORDS, 8))
    return k_local, n_local, word, nib


def _require_tiles(size_k: int, size_n: int) -> tuple[int, int]:
    if size_k % TILE_K or size_n % TILE_N:
        raise ValueError(
            f"repack needs K % {TILE_K} == 0 and N % {TILE_N} == 0, got K={size_k}, N={size_n}; "
            "upstream tile-pads first (marlin_utils.py:221-243, marlin_padded_nk)"
        )
    return size_k // TILE_K, size_n // TILE_N


def unpack_rows(qweight: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
    """GPTQ input layout ``[size_k/8, size_n]`` int words -> ``[size_k, size_n]`` nibble values."""
    w = np.asarray(qweight, dtype=np.int64)
    vals = (w[:, None, :] >> (4 * np.arange(8))[None, :, None]) & 0xF  # [k/8, nib, n]
    return vals.reshape(size_k, size_n)


def pack_rows(w: np.ndarray) -> np.ndarray:
    """Inverse of :func:`unpack_rows`: ``[size_k, size_n]`` nibbles -> ``[size_k/8, size_n]``."""
    w = np.asarray(w, dtype=np.int64)
    out = np.zeros((w.shape[0] // 8, w.shape[1]), dtype=np.int64)
    for i in range(8):
        out |= (w[i::8, :] & 0xF) << (4 * i)
    return out


def pack_cols(w: np.ndarray) -> np.ndarray:
    """AWQ input layout ``[size_k, size_n]`` -> ``[size_k, size_n/8]`` (nibbles of word n // 8)."""
    w = np.asarray(w, dtype=np.int64)
    out = np.zeros((w.shape[0], w.shape[1] // 8), dtype=np.int64)
    for i in range(8):
        out |= (w[:, i::8] & 0xF) << (4 * i)
    return out


def repack_gptq(
    qweight: np.ndarray, size_k: int, size_n: int, perm: np.ndarray | None = None
) -> np.ndarray:
    """``[size_k/8, size_n]`` GPTQ words -> Marlin ``[size_k/16, 2*size_n]`` int32.

    ``perm`` is the act-order / ``g_idx`` order: destination k-slot ``k`` of a k-tile reads source k
    ``perm[k]`` — a pure k-axis gather inside the tile (``gptq_marlin_repack.cu:61-70, 146-162``).
    """
    kt, nt = _require_tiles(size_k, size_n)
    w = unpack_rows(qweight, size_k, size_n)
    k_local, n_local, word, nib = tile_index_arrays()
    out = np.zeros((kt, 2 * size_n), dtype=np.int64)
    for k_tile in range(kt):
        k_src = k_tile * TILE_K + k_local
        if perm is not None:
            k_src = k_tile * TILE_K + np.asarray(perm, dtype=np.int64)[k_src]
        for n_tile in range(nt):
            vals = w[k_src, n_tile * TILE_N + n_local]
            np.bitwise_or.at(out[k_tile], n_tile * TILE_WORDS + word, vals << (4 * nib))
    return out


def unpack_gptq(
    repacked: np.ndarray, size_k: int, size_n: int, perm: np.ndarray | None = None
) -> np.ndarray:
    """Inverse of :func:`repack_gptq`: ``[size_k/16, 2*size_n]`` -> ``[size_k/8, size_n]`` words."""
    kt, nt = _require_tiles(size_k, size_n)
    k_local, n_local, word, nib = tile_index_arrays()
    r = np.asarray(repacked, dtype=np.int64)
    w = np.zeros((size_k, size_n), dtype=np.int64)
    for k_tile in range(kt):
        k_dst = k_tile * TILE_K + k_local
        k_src = k_dst
        if perm is not None:
            k_src = k_tile * TILE_K + np.asarray(perm, dtype=np.int64)[k_dst]
        for n_tile in range(nt):
            vals = (r[k_tile, n_tile * TILE_WORDS + word] >> (4 * nib)) & 0xF
            w[k_src, n_tile * TILE_N + n_local] = vals  # exactly-once slots
    return pack_rows(w)
