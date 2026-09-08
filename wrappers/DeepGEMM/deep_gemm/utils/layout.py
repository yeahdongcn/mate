import functools
import os

import torch
from mate.utils import ceil_div, round_up

__all__ = [
    "get_tma_aligned_size",
    "get_mn_major_tma_aligned_packed_ue8m0_tensor",
    "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
    "transform_sf_into_required_layout",
    "get_mk_alignment_for_contiguous_layout",
    "get_col_major_tma_aligned_tensor",
    "get_mn_major_tma_aligned_tensor",
]

_MK_ALIGNMENT_ENV = "MATE_DEEPGEMM_MK_ALIGNMENT"
_VALID_MK_ALIGNMENTS = (128, 256)


def get_tma_aligned_size(x: int, element_size: int) -> int:
    alignment = 16 // element_size
    return round_up(x, alignment)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor,
) -> torch.Tensor:
    ue8m0_tensor = (sf.view(torch.int32) >> 23).to(torch.uint8)

    mn, k = sf.shape[-2:]
    remove_dim = sf.dim() == 2
    if remove_dim:
        sf = sf.unsqueeze(0)
    batch = sf.shape[0]

    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = round_up(k, 4)
    padded = torch.zeros(
        (batch, aligned_mn, aligned_k), device=sf.device, dtype=torch.uint8
    )
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(torch.int32).view(batch, aligned_mn, aligned_k // 4)

    transposed = torch.empty(
        (batch, aligned_k // 4, aligned_mn), device=sf.device, dtype=torch.int32
    ).mT
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor,
    ks_tensor: torch.Tensor,
    ks: list[int],
) -> torch.Tensor:
    sf_chunks = sf.split([ceil_div(k, 128) for k in ks])
    packed_chunks = [
        get_mn_major_tma_aligned_packed_ue8m0_tensor(chunk.T).T for chunk in sf_chunks
    ]
    return torch.cat(packed_chunks)


@functools.cache
def get_mk_alignment_for_contiguous_layout() -> int:
    """Return the M-axis alignment requirement for contiguous grouped GEMM.

    Each expert segment in a contiguous-layout grouped GEMM must have its
    token count padded to a multiple of this value before being passed to
    m_grouped_{fp8,bf16}_gemm_nt_contiguous.

    Defaults to 128 and can be overridden with MATE_DEEPGEMM_MK_ALIGNMENT.
    The value is resolved once per process and cached.
    """
    value = os.environ.get(_MK_ALIGNMENT_ENV)
    if value is None:
        return 128

    value = value.strip()
    if not value:
        raise ValueError(
            f"{_MK_ALIGNMENT_ENV} must be one of {_VALID_MK_ALIGNMENTS}, "
            "got an empty value"
        )

    try:
        alignment = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{_MK_ALIGNMENT_ENV} must be one of {_VALID_MK_ALIGNMENTS}, got {value!r}"
        ) from exc

    if alignment not in _VALID_MK_ALIGNMENTS:
        raise ValueError(
            f"{_MK_ALIGNMENT_ENV} must be one of {_VALID_MK_ALIGNMENTS}, got {value!r}"
        )

    return alignment


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Return a column-major, TMA-aligned view of a scale tensor.

    This function returns a contiguous copy of x without reordering data.

    Callers porting DeepGEMM code should call this on the LHS FP32 scale
    tensor (shape [m, ceil(k/128)]) before packaging it as a (fp8, scale)
    tuple — the call is a no-op here but keeps the call-site portable.
    """
    return x.contiguous()


get_mn_major_tma_aligned_tensor = get_col_major_tma_aligned_tensor


def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: tuple[int, int, int],
    num_groups: int | None = None,
    is_sfa: bool = False,
    disable_ue8m0_cast: bool = False,
) -> torch.Tensor:
    return get_mn_major_tma_aligned_tensor(sf)
