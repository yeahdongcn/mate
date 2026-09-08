from __future__ import annotations

import functools
from pathlib import Path
from typing import Mapping

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .utils import dtype_torch2mutlass_map


CXX_FLAGS = ["-O3", "-Wno-switch-bool"]
CUDA_FLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]
INCLUDE_PATHS = [
    jit_env.MATE_INCLUDE_DIR,
    jit_env.MATE_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
]

KDA_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}
KDA_DTYPE_FFI = {
    torch.float16: "dl_float16",
    torch.bfloat16: "dl_bfloat16",
}
KDA_CU_SEQLENS_ELEMENT_NAMES = {torch.int32: "int32_t", torch.int64: "int64_t"}
KDA_CU_SEQLENS_DTYPE_NAMES = {torch.int32: "i32", torch.int64: "i64"}
KDA_CU_SEQLENS_DTYPE_FFI = {torch.int32: "dl_int32", torch.int64: "dl_int64"}

# Logical KDA prefill ABI shared by architecture backends.  SQMMA layouts and
# launch occupancy remain backend-specific.
KDA_PREFILL_CHUNK_SIZE = 16
KDA_PREFILL_HEAD_DIM = 128
KDA_MP31_PREPARE_CTAS_PER_MP = 6

KDA_BOOL_KEYS = (
    "has_state_in",
    "has_state_out",
    "state_fp32",
    "has_gate_params",
    "is_varlen",
    "normalize_qk",
)
KDA_CONFIG_KEYS = (
    "element",
    "element_dtype",
    "dtype_name",
    "state_element",
    "state_dtype_name",
    "cu_seqlens_element",
    "cu_seqlens_dtype",
    "cu_seqlens_dtype_name",
    *KDA_BOOL_KEYS,
)
KDA_DEFAULT_CONFIG = {
    "element": dtype_torch2mutlass_map[torch.bfloat16],
    "element_dtype": KDA_DTYPE_FFI[torch.bfloat16],
    "dtype_name": KDA_DTYPE_NAMES[torch.bfloat16],
    "state_element": dtype_torch2mutlass_map[torch.bfloat16],
    "state_dtype_name": KDA_DTYPE_NAMES[torch.bfloat16],
    "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[torch.int64],
    "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[torch.int64],
    "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[torch.int64],
    "has_state_in": True,
    "has_state_out": True,
    "state_fp32": False,
    "has_gate_params": True,
    "is_varlen": False,
    "normalize_qk": True,
}


@functools.lru_cache
def _get_kda_template_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _get_kda_template(template_name: str):
    template_dir = (jit_env.MATE_TEMPLATE_DIR / "flat" / "kda").as_posix()
    return _get_kda_template_env(template_dir).get_template(template_name)


KDA_PREPARE_TEMPLATE = _get_kda_template("chunk_kda_prepare_kernel.j2")
KDA_RECURRENCE_TEMPLATE = _get_kda_template("chunk_kda_recurrence_kernel.j2")


def make_kda_ops_config(
    dtype: torch.dtype,
    *,
    state_dtype: torch.dtype,
    cu_seqlens_dtype: torch.dtype | None = None,
    has_state_in: bool,
    has_state_out: bool,
    state_fp32: bool,
    has_gate_params: bool,
    is_varlen: bool,
    normalize_qk: bool,
) -> dict[str, object]:
    if dtype not in KDA_DTYPE_FFI:
        raise TypeError("chunk_kda supports torch.float16 and torch.bfloat16 inputs")
    if state_dtype not in KDA_DTYPE_NAMES:
        raise TypeError("chunk_kda state must be fp16, bf16, or fp32")
    cu_seqlens_dtype = cu_seqlens_dtype or torch.int64
    if cu_seqlens_dtype not in KDA_CU_SEQLENS_DTYPE_FFI:
        raise TypeError("chunk_kda cu_seqlens must be int32 or int64")
    return {
        "element": dtype_torch2mutlass_map[dtype],
        "element_dtype": KDA_DTYPE_FFI[dtype],
        "dtype_name": KDA_DTYPE_NAMES[dtype],
        "state_element": dtype_torch2mutlass_map[state_dtype],
        "state_dtype_name": KDA_DTYPE_NAMES[state_dtype],
        "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[cu_seqlens_dtype],
        "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[cu_seqlens_dtype],
        "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[cu_seqlens_dtype],
        "has_state_in": bool(has_state_in),
        "has_state_out": bool(has_state_out),
        "state_fp32": bool(state_fp32),
        "has_gate_params": bool(has_gate_params),
        "is_varlen": bool(is_varlen),
        "normalize_qk": bool(normalize_qk),
    }


def _normalize_kda_ops_config(
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized = dict(KDA_DEFAULT_CONFIG)
    if config is not None:
        normalized.update(config)
    for key in KDA_CONFIG_KEYS:
        if key not in normalized:
            raise ValueError(f"missing KDA JIT config key: {key}")
    for key in KDA_BOOL_KEYS:
        normalized[key] = bool(normalized[key])
    if not normalized["is_varlen"]:
        for key in (
            "cu_seqlens_element",
            "cu_seqlens_dtype",
            "cu_seqlens_dtype_name",
        ):
            normalized[key] = KDA_DEFAULT_CONFIG[key]
    return {key: normalized[key] for key in KDA_CONFIG_KEYS}


def _render_config(config: Mapping[str, object], func_name: str) -> dict[str, object]:
    render_config = dict(config)
    render_config["func_name"] = func_name
    render_config["chunk_size"] = KDA_PREFILL_CHUNK_SIZE
    render_config["head_dim"] = KDA_PREFILL_HEAD_DIM
    for key in KDA_BOOL_KEYS:
        render_config[key] = "true" if bool(config[key]) else "false"
    return render_config


def _common_suffix(config: Mapping[str, object]) -> str:
    seqlens = (
        f"_seqlens_{config['cu_seqlens_dtype_name']}"
        if bool(config["is_varlen"])
        else ""
    )
    return (
        f"dtype_{config['dtype_name']}"
        f"_gate_{int(bool(config['has_gate_params']))}"
        f"_varlen_{int(bool(config['is_varlen']))}"
        f"_normqk_{int(bool(config['normalize_qk']))}{seqlens}"
    )


def get_kda_prepare_ops_function_name(
    config: Mapping[str, object] | None = None,
) -> str:
    config = _normalize_kda_ops_config(config)
    return f"prepare_kda_{_common_suffix(config)}"


def get_kda_recurrence_ops_function_name(
    config: Mapping[str, object] | None = None,
) -> str:
    config = _normalize_kda_ops_config(config)
    return (
        f"kda_recurrence_{_common_suffix(config)}"
        f"_state_{config['state_dtype_name']}"
        f"_in_{int(bool(config['has_state_in']))}"
        f"_out_{int(bool(config['has_state_out']))}"
    )


def _gen_spec(
    config: Mapping[str, object],
    *,
    func_name: str,
    template,
) -> JitSpec:
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "kda" / f"{func_name}.mu")
    return gen_jit_spec(
        func_name,
        [source_file],
        generated_sources={
            source_file: template.render(_render_config(config, func_name))
        },
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


def gen_kda_prepare_ops_spec(config: Mapping[str, object] | None = None) -> JitSpec:
    config = _normalize_kda_ops_config(config)
    config.update(
        state_element=config["element"],
        state_dtype_name=config["dtype_name"],
        has_state_in=False,
        has_state_out=False,
        state_fp32=False,
    )
    return _gen_spec(
        config,
        func_name=get_kda_prepare_ops_function_name(config),
        template=KDA_PREPARE_TEMPLATE,
    )


def gen_kda_recurrence_ops_spec(config: Mapping[str, object] | None = None) -> JitSpec:
    config = _normalize_kda_ops_config(config)
    return _gen_spec(
        config,
        func_name=get_kda_recurrence_ops_function_name(config),
        template=KDA_RECURRENCE_TEMPLATE,
    )


def gen_kda_ops_aot() -> list[JitSpec]:
    return [gen_kda_prepare_ops_spec(), gen_kda_recurrence_ops_spec()]


@functools.cache
def _load_prepare(frozen_config: tuple[tuple[str, object], ...]):
    return gen_kda_prepare_ops_spec(dict(frozen_config)).build_and_load()


@functools.cache
def _load_recurrence(frozen_config: tuple[tuple[str, object], ...]):
    return gen_kda_recurrence_ops_spec(dict(frozen_config)).build_and_load()


def get_kda_prepare_ops_module(config: Mapping[str, object] | None = None):
    normalized = _normalize_kda_ops_config(config)
    normalized.update(
        state_element=normalized["element"],
        state_dtype_name=normalized["dtype_name"],
        has_state_in=False,
        has_state_out=False,
        state_fp32=False,
    )
    return _load_prepare(tuple(sorted(normalized.items())))


def get_kda_recurrence_ops_module(config: Mapping[str, object] | None = None):
    normalized = _normalize_kda_ops_config(config)
    return _load_recurrence(tuple(sorted(normalized.items())))
