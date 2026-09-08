from __future__ import annotations

import argparse
import contextlib
import gc
import io
import sys
from collections.abc import Callable
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mate.norm.tilelang import (  # noqa: E402
    fused_add_rmsnorm,
    fused_add_rmsnorm_fp8_block_quant,
    fused_add_rmsnorm_quant,
    fused_dit_gate_residual_layernorm_gamma_beta,
    fused_dit_gate_residual_layernorm_scale_shift,
    fused_dit_residual_layernorm_scale_shift,
    fused_qk_rmsnorm_rope,
    fused_rmsnorm_silu,
    layernorm,
    layernorm_quant,
    rmsnorm,
    rmsnorm_quant,
)
from mate.norm.tilelang.fused_add_rmsnorm import (  # noqa: E402
    _select_fp8_block_quant_schedule,
)
from mate.testing.utils import bench_kineto  # noqa: E402


DIT_HIDDEN_SIZE = 3072
DIT_THREADS = 384
MXFP8_BLOCK_SIZE = 32

KERNELS = (
    "rmsnorm",
    "layernorm",
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_fp8_block_quant",
    "fused_rmsnorm_silu",
    "fused_qk_rmsnorm_rope",
    "fused_dit_layernorm",
)


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _select_norm_schedule(hidden_size: int, num_rows: int) -> tuple[int, int, int]:
    if hidden_size >= 4096:
        threads_per_row = 128
        rows_per_block = 4
    elif hidden_size >= 2048:
        threads_per_row = 64
        rows_per_block = 8
    else:
        threads_per_row = 32
        rows_per_block = 8

    values_per_thread = 8
    while rows_per_block > 1 and num_rows % rows_per_block != 0:
        rows_per_block //= 2
    return rows_per_block, threads_per_row, values_per_thread


def _select_layernorm_schedule(hidden_size: int, num_rows: int) -> tuple[int, int, int]:
    if hidden_size >= 4096:
        threads_per_row = 128
        rows_per_block = 4
    elif hidden_size >= 2048:
        threads_per_row = 64
        rows_per_block = 8
    else:
        threads_per_row = 32
        rows_per_block = 16

    values_per_thread = 8
    while rows_per_block > 1 and num_rows % rows_per_block != 0:
        rows_per_block //= 2
    return rows_per_block, threads_per_row, values_per_thread


def _select_silu_schedule(hidden_size: int, num_rows: int) -> tuple[int, int, int]:
    if hidden_size >= 8192:
        threads_per_row = 256
        rows_per_block = 2
    elif hidden_size >= 2048:
        threads_per_row = 128
        rows_per_block = 4
    else:
        threads_per_row = 64
        rows_per_block = 8

    values_per_thread = 8
    while rows_per_block > 1 and num_rows % rows_per_block != 0:
        rows_per_block //= 2
    return rows_per_block, threads_per_row, values_per_thread


def _bench_seconds(
    fn: Callable[[], object],
    kernel_names: str | tuple[str, ...],
    *,
    warmup: int,
    num_tests: int,
    flush_l2: bool,
) -> float:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    seconds = bench_kineto(
        fn,
        kernel_names=kernel_names,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    if isinstance(seconds, (list, tuple)):
        seconds = sum(float(item) for item in seconds)
    return float(seconds)


def _release_musa_cache() -> None:
    gc.collect()
    if hasattr(torch, "musa"):
        torch.musa.empty_cache()


def _print_header() -> None:
    header = (
        f"{'kernel':<28s} {'shape':<24s} {'mode':<18s} "
        f"{'latency(us)':>12s} {'TFLOPS':>9s} {'GB/s':>9s}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)


def _print_result(
    kernel: str, shape: str, mode: str, seconds: float, flops: int, bytes_rw: int
) -> None:
    if seconds <= 0.0:
        latency_us = "n/a"
        tflops = "n/a"
        bandwidth = "n/a"
    else:
        latency_us = f"{seconds * 1e6:.2f}"
        tflops = f"{flops / seconds / 1e12:.4f}"
        bandwidth = f"{bytes_rw / seconds / 1e9:.2f}"
    print(
        f"{kernel:<28s} {shape:<24s} {mode:<18s} {latency_us:>12s} {tflops:>9s} {bandwidth:>9s}",
        flush=True,
    )


def _rmsnorm_flops(
    num_rows: int,
    hidden_size: int,
    threads_per_row: int,
    *,
    gemma: bool,
    quant: bool,
) -> int:
    per_row = (5 if quant else 4) * hidden_size + (threads_per_row - 1)
    if gemma:
        per_row += hidden_size
    return num_rows * per_row


def _rmsnorm_bytes(
    num_rows: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    *,
    quant: bool,
) -> int:
    input_elem_size = _dtype_size(input_dtype)
    output_elem_size = _dtype_size(output_dtype)
    if quant:
        return (
            2 * num_rows * hidden_size * input_elem_size
            + hidden_size * input_elem_size
            + num_rows * hidden_size * output_elem_size
            + _dtype_size(torch.float32)
        )
    return (3 * num_rows * hidden_size + hidden_size) * input_elem_size


def _layernorm_flops(
    num_rows: int, hidden_size: int, threads_per_row: int, *, quant: bool
) -> int:
    per_row = (6 if quant else 5) * hidden_size + (threads_per_row - 1)
    return num_rows * per_row


def _layernorm_bytes(
    num_rows: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    *,
    quant: bool,
) -> int:
    input_elem_size = _dtype_size(input_dtype)
    output_elem_size = _dtype_size(output_dtype)
    if quant:
        return (
            2 * num_rows * hidden_size * input_elem_size
            + 2 * hidden_size * input_elem_size
            + num_rows * hidden_size * output_elem_size
            + _dtype_size(torch.float32)
        )
    return (3 * num_rows * hidden_size + 2 * hidden_size) * input_elem_size


def _fused_add_rmsnorm_flops(
    num_rows: int,
    hidden_size: int,
    threads_per_row: int,
    *,
    gemma: bool,
) -> int:
    per_row = 6 * hidden_size + (threads_per_row - 1)
    if gemma:
        per_row += hidden_size
    return num_rows * per_row


def _fused_add_rmsnorm_bytes(
    num_rows: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    *,
    quant: bool,
) -> int:
    elem = _dtype_size(input_dtype)
    if not quant:
        return (4 * num_rows * hidden_size + hidden_size) * elem
    return (
        (4 * num_rows * hidden_size + hidden_size) * elem
        + num_rows * hidden_size * _dtype_size(output_dtype)
        + _dtype_size(torch.float32)
    )


def _fused_add_rmsnorm_fp8_block_quant_bytes(
    num_rows: int, hidden_size: int, input_dtype: torch.dtype
) -> int:
    elem = _dtype_size(input_dtype)
    return (
        (4 * num_rows * hidden_size + hidden_size) * elem
        + num_rows * hidden_size
        + num_rows * (hidden_size // 128) * _dtype_size(torch.float32)
    )


def _fused_rmsnorm_silu_flops(
    num_tokens: int, hidden_size: int, threads_per_row: int
) -> int:
    return num_tokens * (8 * hidden_size + threads_per_row - 1)


def _fused_rmsnorm_silu_bytes(
    num_tokens: int, hidden_size: int, output_mode: str
) -> int:
    input_bytes = _dtype_size(torch.bfloat16)
    bytes_rw = 2 * num_tokens * hidden_size * input_bytes + hidden_size * input_bytes
    if output_mode == "bf16":
        return bytes_rw + num_tokens * hidden_size * input_bytes
    if output_mode == "fp8":
        return bytes_rw + num_tokens * hidden_size
    return (
        bytes_rw
        + num_tokens * hidden_size
        + num_tokens * hidden_size // MXFP8_BLOCK_SIZE
    )


def _qk_rope_flops(
    batch_size: int,
    seq_len: int,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    is_qk_norm: bool,
) -> int:
    tokens = batch_size * seq_len
    qk_elems = (num_heads_q + num_heads_k) * head_dim
    rope_flops = (qk_elems // 2) * 6
    norm_flops = qk_elems * 4 + 3 * 2 if is_qk_norm else 0
    return tokens * (rope_flops + norm_flops)


def _qk_rope_bytes(
    batch_size: int,
    seq_len: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    *,
    is_qk_norm: bool,
    output_fp8: bool,
) -> int:
    tokens = batch_size * seq_len
    qkv_elems = tokens * (num_heads_q + num_heads_k + num_heads_v) * head_dim
    weight_elems = (num_heads_q + num_heads_k) * head_dim if is_qk_norm else 0
    output_dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
    return (qkv_elems + weight_elems) * _dtype_size(
        torch.bfloat16
    ) + qkv_elems * _dtype_size(output_dtype)


def _make_dit_param(
    batch_size: int,
    num_rows: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(
        batch_size,
        num_rows * 6,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )[:, ::6, :]


def _dit_flops(
    batch_size: int,
    num_rows: int,
    hidden_size: int,
    mode: str,
    use_gate_bias: bool,
    use_scale_bias: bool,
    use_shift_bias: bool,
    use_residual: bool,
    output_mode: str,
) -> int:
    rows = batch_size * num_rows
    per_elem = 0
    if mode == "gate_gamma_beta":
        per_elem += 2 + int(use_gate_bias)
        per_elem += 3
        per_elem += 4
    elif mode == "gate_scale_shift":
        per_elem += 2 + int(use_gate_bias)
        per_elem += 3
        per_elem += 5 + int(use_scale_bias) + int(use_shift_bias)
    elif mode == "residual_scale_shift":
        per_elem += int(use_residual)
        per_elem += 3
        per_elem += 5 + int(use_scale_bias) + int(use_shift_bias)
    else:
        raise RuntimeError(f"Unsupported mode {mode}.")
    if output_mode == "mxfp8":
        per_elem += 3
    return rows * (per_elem * hidden_size + DIT_THREADS - 1)


def _dit_bytes(
    batch_size: int,
    num_rows: int,
    hidden_size: int,
    mode: str,
    use_gate_bias: bool,
    use_scale_bias: bool,
    use_shift_bias: bool,
    use_residual: bool,
    output_mode: str,
) -> int:
    rows = batch_size * num_rows
    bf16_bytes = _dtype_size(torch.bfloat16)
    f32_bytes = _dtype_size(torch.float32)
    bytes_rw = rows * hidden_size * bf16_bytes
    if use_residual:
        bytes_rw += rows * hidden_size * bf16_bytes
    if mode in ("gate_gamma_beta", "gate_scale_shift"):
        bytes_rw += rows * hidden_size * bf16_bytes
        if use_gate_bias:
            bytes_rw += hidden_size * f32_bytes
    if mode == "gate_gamma_beta":
        bytes_rw += 2 * hidden_size * f32_bytes
    else:
        bytes_rw += 2 * rows * hidden_size * bf16_bytes
        if use_scale_bias:
            bytes_rw += hidden_size * f32_bytes
        if use_shift_bias:
            bytes_rw += hidden_size * f32_bytes
    bytes_rw += rows * hidden_size * bf16_bytes
    if output_mode == "mxfp8":
        bytes_rw += rows * hidden_size
        bytes_rw += rows * hidden_size // MXFP8_BLOCK_SIZE
    else:
        bytes_rw += rows * hidden_size * bf16_bytes
    return bytes_rw


def _run_rmsnorm(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> None:
    for batch_size in args.batch_sizes:
        for hidden_size in args.hidden_sizes:
            x = torch.randn((batch_size, hidden_size), device=device, dtype=dtype)
            weight = torch.randn((hidden_size,), device=device, dtype=dtype)
            _, threads_per_row, _ = _select_layernorm_schedule(hidden_size, batch_size)
            if args.include_quant:
                scale = torch.ones(1, device=device, dtype=torch.float32)
                out = torch.empty_like(x, dtype=torch.float8_e4m3fn)

                def runner() -> torch.Tensor:
                    return rmsnorm_quant(
                        x, weight, scale, eps=args.eps, gemma=args.gemma, out=out
                    )

                mode = "quant"
                output_dtype = torch.float8_e4m3fn
            else:
                y = torch.empty_like(x)

                def runner() -> torch.Tensor:
                    return rmsnorm(x, weight, eps=args.eps, gemma=args.gemma, y=y)

                mode = "bf16/fp16"
                output_dtype = dtype
            seconds = _bench_seconds(
                runner,
                "tilelang_rmsnorm_kernel",
                warmup=args.warmup,
                num_tests=args.num_tests,
                flush_l2=args.flush_l2,
            )
            _print_result(
                "rmsnorm",
                f"{batch_size}x{hidden_size}",
                mode,
                seconds,
                _rmsnorm_flops(
                    batch_size,
                    hidden_size,
                    threads_per_row,
                    gemma=args.gemma,
                    quant=args.include_quant,
                ),
                _rmsnorm_bytes(
                    batch_size,
                    hidden_size,
                    dtype,
                    output_dtype,
                    quant=args.include_quant,
                ),
            )


def _run_layernorm(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> None:
    for batch_size in args.batch_sizes:
        for hidden_size in args.hidden_sizes:
            x = torch.randn((batch_size, hidden_size), device=device, dtype=dtype)
            gamma = torch.randn((hidden_size,), device=device, dtype=torch.float32)
            beta = torch.randn((hidden_size,), device=device, dtype=torch.float32)
            _, threads_per_row, _ = _select_norm_schedule(hidden_size, batch_size)
            if args.include_quant:
                scale = torch.ones(1, device=device, dtype=torch.float32)
                out = torch.empty_like(x, dtype=torch.float8_e4m3fn)

                def runner() -> torch.Tensor:
                    return layernorm_quant(x, gamma, beta, scale, eps=args.eps, out=out)

                mode = "quant"
                output_dtype = torch.float8_e4m3fn
            else:

                def runner() -> torch.Tensor:
                    return layernorm(x, gamma, beta, eps=args.eps)

                mode = "bf16/fp16"
                output_dtype = dtype
            seconds = _bench_seconds(
                runner,
                "tilelang_layernorm_kernel",
                warmup=args.warmup,
                num_tests=args.num_tests,
                flush_l2=args.flush_l2,
            )
            _print_result(
                "layernorm",
                f"{batch_size}x{hidden_size}",
                mode,
                seconds,
                _layernorm_flops(
                    batch_size, hidden_size, threads_per_row, quant=args.include_quant
                ),
                _layernorm_bytes(
                    batch_size,
                    hidden_size,
                    dtype,
                    output_dtype,
                    quant=args.include_quant,
                ),
            )


def _run_fused_add_rmsnorm(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> None:
    for batch_size in args.batch_sizes:
        for hidden_size in args.hidden_sizes:
            x = torch.randn((batch_size, hidden_size), device=device, dtype=dtype)
            residual = torch.randn(
                (batch_size, hidden_size), device=device, dtype=dtype
            )
            weight = torch.randn((hidden_size,), device=device, dtype=dtype)
            _, threads_per_row, _ = _select_norm_schedule(hidden_size, batch_size)
            if args.include_quant:
                scale = torch.ones(1, device=device, dtype=torch.float32)
                out = torch.empty_like(x, dtype=torch.float8_e4m3fn)

                def runner() -> torch.Tensor:
                    with contextlib.redirect_stdout(io.StringIO()):
                        return fused_add_rmsnorm_quant(
                            x,
                            residual,
                            weight,
                            scale,
                            eps=args.eps,
                            gemma=args.gemma,
                            out=out,
                        )

                mode = "quant"
                output_dtype = torch.float8_e4m3fn
            else:

                def runner() -> None:
                    with contextlib.redirect_stdout(io.StringIO()):
                        return fused_add_rmsnorm(
                            x, residual, weight, eps=args.eps, gemma=args.gemma
                        )

                mode = "inplace"
                output_dtype = dtype
            seconds = _bench_seconds(
                runner,
                "tilelang_fused_add_rmsnorm_kernel",
                warmup=args.warmup,
                num_tests=args.num_tests,
                flush_l2=args.flush_l2,
            )
            _print_result(
                "fused_add_rmsnorm",
                f"{batch_size}x{hidden_size}",
                mode,
                seconds,
                _fused_add_rmsnorm_flops(
                    batch_size,
                    hidden_size,
                    threads_per_row,
                    gemma=args.gemma,
                ),
                _fused_add_rmsnorm_bytes(
                    batch_size,
                    hidden_size,
                    dtype,
                    output_dtype,
                    quant=args.include_quant,
                ),
            )


def _run_fused_add_rmsnorm_fp8_block_quant(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> None:
    for batch_size in args.batch_sizes:
        for hidden_size in args.hidden_sizes:
            values_per_thread = 8 if hidden_size <= 8192 else 16
            if (
                hidden_size == 0
                or hidden_size > 16384
                or hidden_size % (32 * values_per_thread) != 0
            ):
                continue
            _, threads_per_row, values_per_thread = _select_fp8_block_quant_schedule(
                hidden_size, batch_size
            )

            x = torch.randn((batch_size, hidden_size), device=device, dtype=dtype)
            residual = torch.randn_like(x)
            weight = torch.randn((hidden_size,), device=device, dtype=dtype)
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            block_scale = torch.empty(
                (batch_size, hidden_size // 128),
                device=device,
                dtype=torch.float32,
            )
            normed_out = torch.empty_like(x)

            def runner() -> None:
                with contextlib.redirect_stdout(io.StringIO()):
                    fused_add_rmsnorm_fp8_block_quant(
                        out,
                        block_scale,
                        normed_out,
                        x,
                        residual,
                        weight,
                        eps=args.eps,
                    )

            seconds = _bench_seconds(
                runner,
                "tilelang_fused_add_rmsnorm_fp8_block_quant",
                warmup=args.warmup,
                num_tests=args.num_tests,
                flush_l2=args.flush_l2,
            )
            _print_result(
                "fused_add_rmsnorm_fp8_bq",
                f"{batch_size}x{hidden_size}",
                "rowmajor-1x128",
                seconds,
                _fused_add_rmsnorm_flops(
                    batch_size,
                    hidden_size,
                    threads_per_row,
                    gemma=False,
                )
                + 2 * batch_size * hidden_size,
                _fused_add_rmsnorm_fp8_block_quant_bytes(
                    batch_size, hidden_size, dtype
                ),
            )


def _run_fused_rmsnorm_silu(args: argparse.Namespace, device: torch.device) -> None:
    for batch_size in args.batch_sizes:
        for hidden_size in args.hidden_sizes:
            x = torch.randn(
                (batch_size, hidden_size), device=device, dtype=torch.bfloat16
            )
            weight = torch.randn((hidden_size,), device=device, dtype=torch.bfloat16)
            rows_per_block, threads_per_row, _ = _select_silu_schedule(
                hidden_size, batch_size
            )
            for output_mode in args.output_modes:
                if output_mode == "fp8":
                    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                    block_scale = None
                    kernel_name = "tilelang_rmsnorm_silu_dense"
                elif output_mode == "mxfp8":
                    if hidden_size % MXFP8_BLOCK_SIZE != 0:
                        continue
                    out = torch.empty_like(x, dtype=torch.uint8)
                    block_scale = torch.empty(
                        (batch_size, hidden_size // MXFP8_BLOCK_SIZE),
                        device=device,
                        dtype=torch.uint8,
                    )
                    kernel_name = "tilelang_rmsnorm_silu_mxfp8"
                else:
                    out = torch.empty_like(x)
                    block_scale = None
                    kernel_name = "tilelang_rmsnorm_silu_dense"

                def runner() -> tuple[torch.Tensor, torch.Tensor | None]:
                    return fused_rmsnorm_silu(
                        x,
                        weight,
                        eps=args.eps,
                        out=out,
                        block_scale=block_scale,
                        rows_per_block=rows_per_block,
                        threads_per_row=threads_per_row,
                    )

                seconds = _bench_seconds(
                    runner,
                    kernel_name,
                    warmup=args.warmup,
                    num_tests=args.num_tests,
                    flush_l2=args.flush_l2,
                )
                _print_result(
                    "fused_rmsnorm_silu",
                    f"{batch_size}x{hidden_size}",
                    output_mode,
                    seconds,
                    _fused_rmsnorm_silu_flops(batch_size, hidden_size, threads_per_row),
                    _fused_rmsnorm_silu_bytes(batch_size, hidden_size, output_mode),
                )


def _run_fused_qk_rmsnorm_rope(args: argparse.Namespace, device: torch.device) -> None:
    batch_size = args.qk_batch_size
    seq_len = args.ppf * args.pph * args.ppw
    hidden_qkv = (
        args.num_heads_q + args.num_heads_k + args.num_heads_v
    ) * args.head_dim
    qkv = torch.randn(
        (batch_size, seq_len, hidden_qkv), device=device, dtype=torch.bfloat16
    )
    q_weight = torch.randn(
        (args.num_heads_q * args.head_dim,), device=device, dtype=torch.bfloat16
    )
    k_weight = torch.randn(
        (args.num_heads_k * args.head_dim,), device=device, dtype=torch.bfloat16
    )
    out_dtype = torch.float8_e4m3fn if args.qk_output_fp8 else torch.bfloat16
    q_out = torch.empty(
        (batch_size, seq_len, args.num_heads_q, args.head_dim),
        device=device,
        dtype=out_dtype,
    )
    k_out = torch.empty(
        (batch_size, seq_len, args.num_heads_k, args.head_dim),
        device=device,
        dtype=out_dtype,
    )
    v_out = torch.empty(
        (batch_size, seq_len, args.num_heads_v, args.head_dim),
        device=device,
        dtype=out_dtype,
    )

    def runner() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fused_qk_rmsnorm_rope(
            qkv,
            q_weight,
            k_weight,
            ppf=args.ppf,
            pph=args.pph,
            ppw=args.ppw,
            num_frame_channels=args.num_frame_channels,
            num_height_channels=args.num_height_channels,
            num_width_channels=args.num_width_channels,
            num_heads_q=args.num_heads_q,
            num_heads_k=args.num_heads_k,
            num_heads_v=args.num_heads_v,
            head_dim=args.head_dim,
            eps=args.eps,
            interleave=not args.no_interleave,
            is_qk_norm=not args.no_qk_norm,
            output_fp8=args.qk_output_fp8,
            output_quant_scale=args.output_quant_scale,
            v_quant_scale=args.v_quant_scale,
            q_out=q_out,
            k_out=k_out,
            v_out=v_out,
        )

    seconds = _bench_seconds(
        runner,
        "tilelang_fused_qk_rmsnorm_rope",
        warmup=args.warmup,
        num_tests=args.num_tests,
        flush_l2=args.flush_l2,
    )
    _print_result(
        "fused_qk_rmsnorm_rope",
        f"{batch_size}x{seq_len}x{hidden_qkv}",
        "fp8" if args.qk_output_fp8 else "bf16",
        seconds,
        _qk_rope_flops(
            batch_size,
            seq_len,
            args.num_heads_q,
            args.num_heads_k,
            args.head_dim,
            not args.no_qk_norm,
        ),
        _qk_rope_bytes(
            batch_size,
            seq_len,
            args.num_heads_q,
            args.num_heads_k,
            args.num_heads_v,
            args.head_dim,
            is_qk_norm=not args.no_qk_norm,
            output_fp8=args.qk_output_fp8,
        ),
    )


def _run_fused_dit_layernorm(args: argparse.Namespace, device: torch.device) -> None:
    shape = (args.dit_batch_size, args.dit_num_rows, DIT_HIDDEN_SIZE)
    x = torch.randn(shape, device=device, dtype=torch.bfloat16)
    residual = torch.randn(shape, device=device, dtype=torch.bfloat16)
    gate = _make_dit_param(
        args.dit_batch_size, args.dit_num_rows, DIT_HIDDEN_SIZE, device
    )
    scale = _make_dit_param(
        args.dit_batch_size, args.dit_num_rows, DIT_HIDDEN_SIZE, device
    )
    shift = _make_dit_param(
        args.dit_batch_size, args.dit_num_rows, DIT_HIDDEN_SIZE, device
    )
    gamma = torch.randn((DIT_HIDDEN_SIZE,), device=device, dtype=torch.float32)
    beta = torch.randn((DIT_HIDDEN_SIZE,), device=device, dtype=torch.float32)
    gate_bias = torch.randn((DIT_HIDDEN_SIZE,), device=device, dtype=torch.float32)
    scale_bias = torch.randn((DIT_HIDDEN_SIZE,), device=device, dtype=torch.float32)
    shift_bias = torch.randn((DIT_HIDDEN_SIZE,), device=device, dtype=torch.float32)

    residual_out = torch.empty_like(x)
    if args.dit_output_mode == "mxfp8":
        norm_out = torch.empty(
            (args.dit_batch_size, args.dit_num_rows, DIT_HIDDEN_SIZE // 4),
            device=device,
            dtype=torch.int32,
        )
        sf_out = torch.empty(
            (args.dit_batch_size, args.dit_num_rows, DIT_HIDDEN_SIZE // 32),
            device=device,
            dtype=torch.uint8,
        )
    else:
        norm_out = torch.empty_like(x)
        sf_out = None

    cases: tuple[tuple[str, Callable[[], object]], ...] = (
        (
            "gate_gamma_beta",
            lambda: fused_dit_gate_residual_layernorm_gamma_beta(
                x,
                residual,
                gate,
                gamma,
                beta,
                gate_bias=gate_bias,
                epsilon=args.eps,
                use_mxfp8=args.dit_output_mode == "mxfp8",
                residual_out=residual_out,
                norm_out=norm_out,
                sf_out=sf_out,
            ),
        ),
        (
            "gate_scale_shift",
            lambda: fused_dit_gate_residual_layernorm_scale_shift(
                x,
                residual,
                gate,
                scale,
                shift,
                gate_bias=gate_bias,
                scale_bias=scale_bias,
                shift_bias=shift_bias,
                epsilon=args.eps,
                use_mxfp8=args.dit_output_mode == "mxfp8",
                residual_out=residual_out,
                norm_out=norm_out,
                sf_out=sf_out,
            ),
        ),
        (
            "residual_scale_shift",
            lambda: fused_dit_residual_layernorm_scale_shift(
                x,
                scale,
                shift,
                residual=residual,
                scale_bias=scale_bias,
                shift_bias=shift_bias,
                epsilon=args.eps,
                use_mxfp8=args.dit_output_mode == "mxfp8",
                residual_out=residual_out,
                norm_out=norm_out,
                sf_out=sf_out,
            ),
        ),
    )
    for mode, runner in cases:
        seconds = _bench_seconds(
            runner,
            "tilelang_fused_dit_layernorm",
            warmup=args.warmup,
            num_tests=args.num_tests,
            flush_l2=args.flush_l2,
        )
        _print_result(
            "fused_dit_layernorm",
            f"{args.dit_batch_size}x{args.dit_num_rows}x{DIT_HIDDEN_SIZE}",
            f"{mode}/{args.dit_output_mode}",
            seconds,
            _dit_flops(
                args.dit_batch_size,
                args.dit_num_rows,
                DIT_HIDDEN_SIZE,
                mode,
                True,
                mode != "gate_gamma_beta",
                mode != "gate_gamma_beta",
                True,
                args.dit_output_mode,
            ),
            _dit_bytes(
                args.dit_batch_size,
                args.dit_num_rows,
                DIT_HIDDEN_SIZE,
                mode,
                True,
                mode != "gate_gamma_beta",
                mode != "gate_gamma_beta",
                True,
                args.dit_output_mode,
            ),
        )


def _selected_kernels(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        return KERNELS
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TileLang norm kernels with one entrypoint."
    )
    parser.add_argument(
        "--kernels", nargs="+", choices=(*KERNELS, "all"), default=["all"]
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller default shapes for smoke performance runs.",
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--num-tests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--no-l2-flush", action="store_true", help="Disable L2 flush during timing."
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--gemma",
        action="store_true",
        help="Use Gemma-style weight offset for RMSNorm kernels.",
    )
    parser.add_argument(
        "--include-quant",
        action="store_true",
        help="Run FP8 quant variants for kernels that support it.",
    )
    parser.add_argument(
        "--output-modes", nargs="+", choices=("bf16", "fp8", "mxfp8"), default=["bf16"]
    )

    parser.add_argument("--qk-batch-size", type=int, default=None)
    parser.add_argument("--ppf", type=int, default=8)
    parser.add_argument("--pph", type=int, default=8)
    parser.add_argument("--ppw", type=int, default=8)
    parser.add_argument("--num-heads-q", type=int, default=4)
    parser.add_argument("--num-heads-k", type=int, default=4)
    parser.add_argument("--num-heads-v", type=int, default=4)
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument("--num-frame-channels", type=int, default=64)
    parser.add_argument("--num-height-channels", type=int, default=64)
    parser.add_argument("--num-width-channels", type=int, default=128)
    parser.add_argument("--no-interleave", action="store_true")
    parser.add_argument("--no-qk-norm", action="store_true")
    parser.add_argument("--qk-output-fp8", action="store_true")
    parser.add_argument("--output-quant-scale", type=float, default=1.32)
    parser.add_argument("--v-quant-scale", type=float, default=1.0)

    parser.add_argument("--dit-batch-size", type=int, default=None)
    parser.add_argument("--dit-num-rows", type=int, default=None)
    parser.add_argument("--dit-output-mode", choices=("bf16", "mxfp8"), default="bf16")
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")
    if args.warmup < 0:
        raise RuntimeError("--warmup must be >= 0.")
    if args.num_tests < 1:
        raise RuntimeError("--num-tests must be >= 1.")
    if (
        args.num_frame_channels + args.num_height_channels + args.num_width_channels
        != args.head_dim
    ):
        raise RuntimeError("RoPE channel counts must sum to --head-dim.")
    if args.qk_output_fp8 and not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for FP8 output.")
    args.flush_l2 = not args.no_l2_flush

    if args.batch_sizes is None:
        args.batch_sizes = [128] if args.quick else [1024, 4096, 8192]
    if args.hidden_sizes is None:
        args.hidden_sizes = [1024] if args.quick else [1024, 4096, 8192]
    if args.qk_batch_size is None:
        args.qk_batch_size = 1 if args.quick else 16
    if args.dit_batch_size is None:
        args.dit_batch_size = 8 if args.quick else 384
    if args.dit_num_rows is None:
        args.dit_num_rows = 16 if args.quick else 512

    device = torch.device("musa")
    dtype = _dtype(args.dtype)
    torch.manual_seed(0)

    print(f"\nMUSA: {torch.musa.get_device_name(0)}", flush=True)
    print(
        f"Config: kernels={_selected_kernels(args.kernels)}, dtype={args.dtype}, "
        f"warmup={args.warmup}, num_tests={args.num_tests}, l2_flush={args.flush_l2}",
        flush=True,
    )
    print(flush=True)
    _print_header()

    runners: dict[str, Callable[[argparse.Namespace, torch.device], None]] = {
        "rmsnorm": lambda ns, dev: _run_rmsnorm(ns, dev, dtype),
        "layernorm": lambda ns, dev: _run_layernorm(ns, dev, dtype),
        "fused_add_rmsnorm": lambda ns, dev: _run_fused_add_rmsnorm(ns, dev, dtype),
        "fused_add_rmsnorm_fp8_block_quant": lambda ns, dev: (
            _run_fused_add_rmsnorm_fp8_block_quant(ns, dev, dtype)
        ),
        "fused_rmsnorm_silu": _run_fused_rmsnorm_silu,
        "fused_qk_rmsnorm_rope": _run_fused_qk_rmsnorm_rope,
        "fused_dit_layernorm": _run_fused_dit_layernorm,
    }
    for kernel in _selected_kernels(args.kernels):
        runners[kernel](args, device)
        _release_musa_cache()


if __name__ == "__main__":
    main()
