"""Dense W4A16 GEMM: fp16 activations, signed INT4 weights, per-K128-group fp16 scales.

The kernel keeps every M-dependent decision inside one compiled op, so the dispatch name never
depends on M and the caller cannot freeze a shape-specific choice at trace time. One tile shape
covers the whole range; the persistent scheduler adapts the work distribution instead.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import torch
from jinja2 import Environment, FileSystemLoader

from ...mate_runtime import resolve_num_mps
from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec
from ..utils import EXPORT_FUNC, TVM_HEADER, dtype_torch2mutlass_map
from . import gemm_utils

# The K256 tile carries exactly two K128 weight-scale blocks, which is the group size W4A16
# checkpoints are quantized with; a different group size needs a different tile, not a flag.
DENSE_W4A16_GEMM_TILE = (32, 256, 256, 3)
DENSE_W4A16_GEMM_GROUP_SIZE = 128

# Signed INT4 nibbles are stored biased so that a zero point of kWeightNibbleBias is a no-op, which is
# what the symmetric GPTQ (uint4b8) checkpoints use and what the atom's signed product already
# computes. Any other zero point needs the rank-1 correction the kernel applies.
DENSE_W4A16_GEMM_NIBBLE_BIAS = 8

DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

DENSE_W4A16_GEMM_MCC_FLAGS = ["-Od3", "-O3", "-std=c++17"]
DENSE_W4A16_GEMM_CXX_FLAGS = ["-O3"]


@dataclass(frozen=True)
class MateW4A16WeightLayout:
    """The packed-weight contract this kernel consumes today.

    Contract: `b_packed[n, k // 2]` is one byte holding the pair of K-adjacent weights of output
    column `n`: the even K index occupies the low nibble, the odd K index the high nibble, and each
    nibble already holds the signed INT4 value `unsigned_nibble - nibble_bias`, so the atom's signed
    product needs no per-element correction. `scale_b` and the folded `zero_b` are `(N, K // 128)`.

    TODO: the Marlin layout contract (upstream `[K // 16, 2 * N]` int32 words) replaces this. Upstream
    stores the same logical `(k, n) -> (word, nibble)` index map in a different physical order, so
    consuming it means translating its nibble addresses into this atom's fragment addresses by a
    static permutation, plus re-deriving the group-scale and zero-point layouts that ride with it.
    Nothing else in this module depends on the physical order, so that permutation is the only change.
    """

    even_k_low_nibble: bool = True
    nibble_bias: int = DENSE_W4A16_GEMM_NIBBLE_BIAS
    scale_group_size: int = DENSE_W4A16_GEMM_GROUP_SIZE


MATE_W4A16_WEIGHT_LAYOUT = MateW4A16WeightLayout()


def dense_w4a16_gemm_dispatch_name(out_dtype: torch.dtype, zero_points: bool) -> str:
    if out_dtype not in DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS:
        raise ValueError(f"out_dtype must be fp16 or bf16, got {out_dtype!r}")
    tile_m, tile_n, tile_k, stages = DENSE_W4A16_GEMM_TILE
    zp_tag = "zp" if zero_points else "sym"
    return f"mp31_dense_w4a16_gemm_{DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS[out_dtype]}_{tile_m}x{tile_n}x{tile_k}_s{stages}_{zp_tag}"


def render_dense_w4a16_gemm_source(out_dtype: torch.dtype, zero_points: bool) -> str:
    tile_m, tile_n, tile_k, stages = DENSE_W4A16_GEMM_TILE
    render_config = {
        "func_name": dense_w4a16_gemm_dispatch_name(out_dtype, zero_points),
        "element_a": dtype_torch2mutlass_map[torch.float16],
        "element_scale_b": dtype_torch2mutlass_map[torch.float16],
        "element_zero_b": dtype_torch2mutlass_map[torch.float16],
        "element_d": dtype_torch2mutlass_map[out_dtype],
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "stages": stages,
        "zero_points": zero_points,
    }
    return (
        TVM_HEADER
        + Environment(
            loader=FileSystemLoader(
                (gemm_utils.gemm_template_dir / "dense_w4a16_gemm").as_posix()
            ),
            keep_trailing_newline=True,
        )
        .get_template("dense_w4a16_gemm_kern.j2")
        .render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def gen_dense_w4a16_gemm_spec(out_dtype: torch.dtype, zero_points: bool) -> JitSpec:
    dispatch_name = dense_w4a16_gemm_dispatch_name(out_dtype, zero_points)
    source_path = jit_env.MATE_GEN_SRC_DIR / "dense_w4a16_gemm" / f"{dispatch_name}.mu"
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cflags=DENSE_W4A16_GEMM_CXX_FLAGS,
        extra_cuda_cflags=DENSE_W4A16_GEMM_MCC_FLAGS,
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
        generated_sources={
            source_path: render_dense_w4a16_gemm_source(out_dtype, zero_points)
        },
    )


def gen_dense_w4a16_gemm_aot() -> list[JitSpec]:
    return [
        gen_dense_w4a16_gemm_spec(out_dtype, zero_points)
        for out_dtype in DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS
        for zero_points in (False, True)
    ]


@functools.cache
def _load_dense_w4a16_gemm_module(out_dtype: torch.dtype, zero_points: bool):
    return gen_dense_w4a16_gemm_spec(out_dtype, zero_points).build_and_load()


def get_dense_w4a16_gemm_module(out_dtype: torch.dtype, zero_points: bool):
    return (
        dense_w4a16_gemm_dispatch_name(out_dtype, zero_points),
        _load_dense_w4a16_gemm_module(out_dtype, zero_points),
    )


def apply_gidx_permutation(
    weight: torch.Tensor, g_idx: Optional[torch.Tensor]
) -> torch.Tensor:
    """Materialize the act-order (g_idx) permutation into the packed weight.

    The kernel copies K contiguously and asserts that it is given an identity ordering, so a
    permutation has to land here, in the packed operand, rather than inside the kernel. Only the
    identity ordering is accepted until the act-order layout itself is defined.
    """
    if g_idx is None:
        return weight
    if g_idx.dim() != 1:
        raise ValueError(f"g_idx must be 1-D, got shape {tuple(g_idx.shape)}")
    if not bool(
        torch.equal(
            g_idx, torch.arange(g_idx.numel(), device=g_idx.device, dtype=g_idx.dtype)
        )
    ):
        raise NotImplementedError(
            "act-order (desc_act) checkpoints are not supported yet: g_idx must be the identity"
        )
    return weight


def fold_zero_points(
    zero_points: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Fold per-group zero points into the single correction stream the kernel consumes.

    `w = scale * (q + nibble_bias - zero_point)` splits into the atom's signed product `scale * q` plus
    the rank-1 term `(nibble_bias - zero_point) * scale * reduce_k(a)`. The kernel takes the second
    factor, so this function returns `(nibble_bias - zero_point) * scale` in fp16.

    A plain INT4 to fp16 dequant that ignores the bias is wrong by a constant `nibble_bias * sum(scale)`
    per group, so any parity test must compare against the biased reference, not a naive dequant.
    """
    if zero_points.shape != scale_b.shape:
        raise ValueError(
            f"zero_points shape {tuple(zero_points.shape)} must match scale_b shape {tuple(scale_b.shape)}"
        )
    correction = (
        float(MATE_W4A16_WEIGHT_LAYOUT.nibble_bias) - zero_points.to(torch.float32)
    ) * scale_b.to(torch.float32)
    return correction.to(scale_b.dtype).contiguous()


def _check_contiguous_last_dim(tensor: torch.Tensor, name: str) -> None:
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in its last dimension")


def _pad_last_dim_to_even(tensor: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Pad the group dimension to an even count, which the paired 16-bit scale words require."""
    if tensor.shape[-1] % 2 == 0:
        return tensor
    padded = torch.full(
        (*tensor.shape[:-1], tensor.shape[-1] + 1),
        fill,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[..., :-1].copy_(tensor)
    return padded.contiguous()


def dense_w4a16_gemm_mutlass(
    input_a: torch.Tensor,
    weight_b: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    g_idx: Optional[torch.Tensor] = None,
    num_mps: Optional[int] = None,
) -> torch.Tensor:
    """Run the dense W4A16 GEMM. See `MateW4A16WeightLayout` for the packed-weight contract.

    `input_a` is (M, K) fp16, `weight_b` is (N, K // 2) int8, `scale_b` is (N, K // 128) fp16,
    `zero_points` (optional, same shape as `scale_b`) holds the per-group AWQ zero points, and `out`
    is the preallocated (M, N) result.
    """
    tile_k = DENSE_W4A16_GEMM_TILE[2]
    group_size = MATE_W4A16_WEIGHT_LAYOUT.scale_group_size

    if input_a.dim() != 2 or weight_b.dim() != 2 or out.dim() != 2:
        raise ValueError("input_a, weight_b and out must all be 2-D")
    if input_a.dtype != torch.float16:
        raise ValueError(f"input_a must be fp16, got {input_a.dtype}")
    if weight_b.dtype != torch.int8:
        raise ValueError(
            f"weight_b must be int8 holding packed INT4 pairs, got {weight_b.dtype}"
        )
    if out.dtype not in DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS:
        raise ValueError(f"out must be fp16 or bf16, got {out.dtype}")

    m, k = input_a.shape
    n = weight_b.shape[0]
    if k % tile_k:
        raise ValueError(f"K must be a multiple of the tile K ({tile_k}), got {k}")
    if weight_b.shape[1] * 2 != k:
        raise ValueError(
            f"weight_b holds {weight_b.shape[1] * 2} K values but input_a has K = {k}"
        )
    expected_scale_cols = k // group_size
    if scale_b.shape != (n, expected_scale_cols):
        raise ValueError(
            f"scale_b must have shape {(n, expected_scale_cols)}, got {tuple(scale_b.shape)}"
        )
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")

    for tensor, name in (
        (input_a, "input_a"),
        (weight_b, "weight_b"),
        (scale_b, "scale_b"),
        (out, "out"),
    ):
        _check_contiguous_last_dim(tensor, name)

    weight_b = apply_gidx_permutation(weight_b, g_idx)

    # The packed scale words pair two 16-bit group values, so the group dimension has to be even. A
    # padded zero-point column is filled with the nibble bias, which makes its folded correction
    # exactly zero, and a padded scale column is zero, which keeps the product zero as well.
    scale_b_padded = _pad_last_dim_to_even(scale_b)
    bias = float(MATE_W4A16_WEIGHT_LAYOUT.nibble_bias)
    correction = None
    if zero_points is not None:
        if zero_points.shape != scale_b.shape:
            raise ValueError(
                f"zero_points shape {tuple(zero_points.shape)} must match scale_b shape "
                f"{tuple(scale_b.shape)}"
            )
        correction = fold_zero_points(
            _pad_last_dim_to_even(zero_points, fill=bias), scale_b_padded
        )

    dispatch_name, module = get_dense_w4a16_gemm_module(
        out.dtype, correction is not None
    )
    module.get_function(dispatch_name)(
        input_a,
        weight_b,
        scale_b_padded,
        correction if correction is not None else scale_b_padded,
        out,
        resolve_num_mps(input_a.device, num_mps),
    )
    return out
