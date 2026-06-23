#!/usr/bin/env python3
"""Benchmark TileLang FlashAttention backward 5-MM path.

Default big case mirrors:

    test_flashattn_bwd_5mm(1, 28, 28, 8192, 8192, 256, torch.bfloat16, True)

The script first runs a small correctness check, then benchmarks a preallocated
kernel path with MUSA graph replay and profiles the key kernels on the path.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch_musa  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mate.flash_attention.tilelang.flash_attention_varlen_bwd import (  # noqa: E402
    ceil_div,
    compute_delta_ws,
    flashattn_bwd_ws,
    pack_dq_from_accum_ws,
    reduce_kv_grads_ws,
    to_tilelang_dtype,
)

_GRAPH_KEEPALIVE: list[object] = []


@dataclasses.dataclass(frozen=True)
class BwdCase:
    batch: int
    heads: int
    heads_kv: int
    total_seq_q: int
    total_seq_kv: int
    dim: int
    dtype: torch.dtype
    causal: bool
    block_m: int = 64
    block_n: int = 64
    threads: int = 640

    @property
    def head_groups(self) -> int:
        return self.heads // self.heads_kv

    @property
    def max_seq_q(self) -> int:
        return self.total_seq_q // self.batch

    @property
    def max_seq_kv(self) -> int:
        return self.total_seq_kv // self.batch

    @property
    def total_flops(self) -> float:
        factor = 5 if self.causal else 10
        return (
            self.total_seq_q
            * self.total_seq_kv
            / self.batch
            * self.heads
            * factor
            * self.dim
        )


def _sync() -> None:
    torch.musa.synchronize()


def _new_event() -> torch.musa.Event:
    return torch.musa.Event(enable_timing=True)


def _new_graph() -> torch.musa.MUSAGraph:
    return torch.musa.MUSAGraph()


def _profiler_activity():
    if hasattr(torch.profiler.ProfilerActivity, "MUSA"):
        return torch.profiler.ProfilerActivity.MUSA
    return torch.profiler.ProfilerActivity.CUDA


def _kernel_time_us(evt) -> float:
    return float(
        getattr(evt, "device_time_total", 0.0)
        or getattr(evt, "self_device_time_total", 0.0)
        or getattr(evt, "musa_time_total", 0.0)
        or getattr(evt, "self_musa_time_total", 0.0)
        or getattr(evt, "cuda_time_total", 0.0)
        or getattr(evt, "self_cuda_time_total", 0.0)
        or 0.0
    )


def _kernel_role(name: str) -> str | None:
    low = name.lower()
    if "flashattn_bwd_ws_kernel" in low:
        return "bwd"
    if "compute_delta" in low or "compute_delta_ws" in low:
        return "delta"
    if "pack_dq" in low:
        return "pack_dq"
    if "reduce" in low and ("kv" in low or "grad" in low):
        return "reduce_kv"
    if "fill" in low or "zero" in low or "setitem" in low or "memset" in low:
        return "clear"
    if "copy" in low or "cast" in low or "to_copy" in low:
        return "copy_cast"
    return None


def _dtype_from_name(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _make_cu_seqlens(total_seq: int, batch: int, device: str) -> torch.Tensor:
    if total_seq % batch != 0:
        raise ValueError("this benchmark expects total_seq divisible by batch")
    step = total_seq // batch
    return torch.arange(batch + 1, device=device, dtype=torch.int32) * step


def _ref_flashattn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    causal: bool,
    smscale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, heads_q, dim = q.shape
    heads_kv = k.shape[1]
    head_groups = heads_q // heads_kv
    scale = smscale if smscale is not None else dim**-0.5
    out = torch.zeros_like(q, dtype=torch.float32)
    lse_out = torch.zeros((q.shape[0], heads_q), dtype=torch.float32, device=q.device)

    for batch_idx in range(cu_q.numel() - 1):
        q0, q1 = int(cu_q[batch_idx].item()), int(cu_q[batch_idx + 1].item())
        k0, k1 = int(cu_k[batch_idx].item()), int(cu_k[batch_idx + 1].item())
        q_batch = q[q0:q1].float()
        k_batch = k[k0:k1].float()
        v_batch = v[k0:k1].float()
        q_len = q1 - q0
        kv_len = k1 - k0
        for head_q in range(heads_q):
            head_kv = head_q // head_groups
            scores = q_batch[:, head_q, :] @ k_batch[:, head_kv, :].T
            scores = scores * scale
            if causal:
                mask = torch.tril(
                    torch.ones((q_len, kv_len), device=q.device, dtype=torch.bool)
                )
                scores = scores.masked_fill(~mask, float("-inf"))
            lse = torch.logsumexp(scores, dim=-1)
            lse = torch.nan_to_num(lse)
            prob = torch.exp(scores - lse.unsqueeze(-1))
            out[q0:q1, head_q, :] = prob @ v_batch[:, head_kv, :]
            lse_out[q0:q1, head_q] = lse
    return out.to(q.dtype), lse_out


def _ref_flashattn_bwd_5mm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    causal: bool,
    smscale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, heads_q, dim = q.shape
    heads_kv = k.shape[1]
    head_groups = heads_q // heads_kv
    scale = smscale if smscale is not None else dim**-0.5
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    for batch_idx in range(cu_q.numel() - 1):
        q0, q1 = int(cu_q[batch_idx].item()), int(cu_q[batch_idx + 1].item())
        k0, k1 = int(cu_k[batch_idx].item()), int(cu_k[batch_idx + 1].item())
        q_batch = q[q0:q1].float()
        k_batch = k[k0:k1].float()
        v_batch = v[k0:k1].float()
        do_batch = do[q0:q1].float()
        q_len = q1 - q0
        kv_len = k1 - k0
        for head_q in range(heads_q):
            head_kv = head_q // head_groups
            scores = q_batch[:, head_q, :] @ k_batch[:, head_kv, :].T
            scores = scores * scale
            if causal:
                mask = torch.tril(
                    torch.ones((q_len, kv_len), device=q.device, dtype=torch.bool)
                )
                scores = scores.masked_fill(~mask, float("-inf"))
            prob = torch.exp(scores - lse[q0:q1, head_q].unsqueeze(-1))
            dp = do_batch[:, head_q, :] @ v_batch[:, head_kv, :].T
            dp = prob * (dp - delta[q0:q1, head_q].unsqueeze(-1))
            dp = dp * scale
            dq[q0:q1, head_q, :] += dp @ k_batch[:, head_kv, :]
            dk[k0:k1, head_kv, :] += dp.T @ q_batch[:, head_q, :]
            dv[k0:k1, head_kv, :] += prob.T @ do_batch[:, head_q, :]
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def _make_inputs(case: BwdCase, device: str):
    torch.manual_seed(42)
    torch.musa.manual_seed(42)
    q = (
        torch.randn(
            (case.total_seq_q, case.heads, case.dim),
            device=device,
            dtype=case.dtype,
        )
        / 10
    ).contiguous()
    k = (
        torch.randn(
            (case.total_seq_kv, case.heads_kv, case.dim),
            device=device,
            dtype=case.dtype,
        )
        / 10
    ).contiguous()
    v = (
        torch.randn(
            (case.total_seq_kv, case.heads_kv, case.dim),
            device=device,
            dtype=case.dtype,
        )
        / 10
    ).contiguous()
    do = (torch.randn_like(q) / 10).contiguous()
    cu_q = _make_cu_seqlens(case.total_seq_q, case.batch, device)
    cu_k = _make_cu_seqlens(case.total_seq_kv, case.batch, device)
    out, lse = _ref_flashattn_fwd(q, k, v, cu_q, cu_k, causal=case.causal)
    return q, k, v, do, out.contiguous(), lse.transpose(0, 1).contiguous(), cu_q, cu_k


class BwdKernelPath:
    def __init__(self, case: BwdCase, device: str = "musa") -> None:
        self.case = case
        self.device = device
        (
            self.q,
            self.k,
            self.v,
            self.do,
            self.out,
            self.lse,
            self.cu_q,
            self.cu_k,
        ) = _make_inputs(case, device)
        self.kernel_dtype = to_tilelang_dtype(case.dtype)
        self.num_blocks_kv = ceil_div(case.max_seq_kv, case.block_n)
        self.max_seq_q_padded = ceil_div(case.max_seq_q, case.block_m) * case.block_m
        self.heads_q_eq_heads_kv = case.heads == case.heads_kv

        self.delta = torch.empty(
            (case.total_seq_q, case.heads), device=device, dtype=torch.float32
        )
        self.dq_accum = torch.empty(
            (case.batch, case.heads, self.max_seq_q_padded, case.dim),
            device=device,
            dtype=torch.float32,
        )
        if self.heads_q_eq_heads_kv:
            self.dk_accum = torch.empty_like(self.k)
            self.dv_accum = torch.empty_like(self.v)
            self.dk = self.dk_accum
            self.dv = self.dv_accum
            self.reduce_kernel = None
        else:
            accum_shape = (case.total_seq_kv, case.heads, case.dim)
            self.dk_accum = torch.empty(accum_shape, device=device, dtype=torch.float32)
            self.dv_accum = torch.empty(accum_shape, device=device, dtype=torch.float32)
            self.dk = torch.empty_like(self.k, dtype=torch.float32)
            self.dv = torch.empty_like(self.v, dtype=torch.float32)
            self.reduce_kernel = reduce_kv_grads_ws(
                case.head_groups,
                case.dim,
                is_varlen=True,
            )
        self.dq = torch.empty_like(self.q)
        self.debug = torch.empty(
            (self.num_blocks_kv,), device=device, dtype=torch.int32
        )

        self.delta_kernel = compute_delta_ws(
            case.dim,
            is_varlen=True,
            dtype=self.kernel_dtype,
            block_M=case.block_m,
            threads=case.threads,
        )
        self.bwd_kernel = flashattn_bwd_ws(
            dim=case.dim,
            is_causal=case.causal,
            is_varlen=True,
            heads_q_eq_heads_kv=self.heads_q_eq_heads_kv,
            block_M=case.block_m,
            block_N=case.block_n,
            smscale=None,
            threads=case.threads,
            dtype=self.kernel_dtype,
        )
        self.pack_dq_kernel = pack_dq_from_accum_ws(
            case.dim,
            is_varlen=True,
            dtype=self.kernel_dtype,
        )

    def clear(self) -> None:
        self.dq_accum.zero_()
        self.dk_accum.zero_()
        self.dv_accum.zero_()

    def run(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.clear()
        self.delta_kernel(self.out, self.do, self.delta)
        self.bwd_kernel(
            self.q,
            self.k,
            self.v,
            self.out,
            self.dq_accum,
            self.dk_accum,
            self.dv_accum,
            self.do,
            self.cu_q,
            self.cu_k,
            self.lse,
            self.delta,
            self.debug,
        )
        self.pack_dq_kernel(self.dq_accum, self.cu_q, self.dq)
        if self.reduce_kernel is not None:
            self.reduce_kernel(self.dk_accum, self.dv_accum, self.dk, self.dv)
            self.dk.copy_(self.dk.to(self.case.dtype))
            self.dv.copy_(self.dv.to(self.case.dtype))
        return self.dq, self.dk, self.dv


def _bench_event_us(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start_events = [_new_event() for _ in range(repeat)]
    end_events = [_new_event() for _ in range(repeat)]
    for idx in range(repeat):
        start_events[idx].record()
        fn()
        end_events[idx].record()
    _sync()
    return statistics.median(
        start_events[idx].elapsed_time(end_events[idx]) * 1000.0
        for idx in range(repeat)
    )


def _bench_graph_us(
    fn: Callable[[], object],
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    graph = _new_graph()
    outputs = []
    with torch.musa.graph(graph):
        for _ in range(graph_iters):
            outputs.append(fn())
    _sync()
    for _ in range(2):
        graph.replay()
    _sync()
    start_events = [_new_event() for _ in range(repeat)]
    end_events = [_new_event() for _ in range(repeat)]
    for idx in range(repeat):
        start_events[idx].record()
        graph.replay()
        end_events[idx].record()
    _sync()
    _GRAPH_KEEPALIVE.append((graph, outputs))
    return statistics.median(
        start_events[idx].elapsed_time(end_events[idx]) * 1000.0 / graph_iters
        for idx in range(repeat)
    )


def _profile_kernel_path(
    fn: Callable[[], object],
    repeat: int,
) -> list[tuple[str, str, float, int]]:
    fn()
    _sync()
    with torch.profiler.profile(activities=[_profiler_activity()]) as prof:
        for _ in range(repeat):
            fn()
            prof.step()
    rows: list[tuple[str, str, float, int]] = []
    for evt in prof.key_averages():
        us = _kernel_time_us(evt)
        if us <= 0:
            continue
        role = _kernel_role(evt.key) or "other"
        if evt.key == "main_kernel":
            # compute_delta_ws and pack_dq_from_accum_ws both lower to this
            # generic TileLang name, and key_averages aggregates them.
            role = "delta_pack"
        rows.append((role, evt.key, us / repeat, evt.count))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows


def _validate_correctness() -> None:
    case = BwdCase(
        batch=1,
        heads=4,
        heads_kv=4,
        total_seq_q=256,
        total_seq_kv=256,
        dim=256,
        dtype=torch.bfloat16,
        causal=True,
    )
    path = BwdKernelPath(case)
    dq, dk, dv = path.run()
    _sync()
    delta = torch.empty(
        (case.total_seq_q, case.heads), device="musa", dtype=torch.float32
    )
    compute_delta_ws(
        case.dim,
        is_varlen=True,
        dtype=to_tilelang_dtype(case.dtype),
        block_M=case.block_m,
        threads=case.threads,
    )(path.out, path.do, delta)
    ref_dq, ref_dk, ref_dv = _ref_flashattn_bwd_5mm(
        path.q,
        path.k,
        path.v,
        path.do,
        path.lse.transpose(0, 1).contiguous(),
        delta,
        path.cu_q,
        path.cu_k,
        causal=case.causal,
    )
    _sync()
    limit = 2.01 / 128
    for name, actual, ref in (
        ("dQ", dq, ref_dq),
        ("dK", dk, ref_dk),
        ("dV", dv, ref_dv),
    ):
        err = (actual.float() - ref.float()).abs().max().item()
        mean = (actual.float() - ref.float()).abs().mean().item()
        print(f"[correctness] {name} max={err:.6f} mean={mean:.6f}")
        if err > limit:
            raise AssertionError(f"{name} max error {err} exceeds {limit}")
    print("[correctness] pass")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--heads-kv", type=int, default=28)
    parser.add_argument("--total-seq-q", type=int, default=8192)
    parser.add_argument("--total-seq-kv", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--graph-iters", type=int, default=10)
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-profiler", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is required for this benchmark")
    torch.set_default_device("musa")
    if not args.skip_correctness:
        _validate_correctness()

    case = BwdCase(
        batch=args.batch,
        heads=args.heads,
        heads_kv=args.heads_kv,
        total_seq_q=args.total_seq_q,
        total_seq_kv=args.total_seq_kv,
        dim=args.dim,
        dtype=_dtype_from_name(args.dtype),
        causal=args.causal,
    )
    path = BwdKernelPath(case)
    path.run()
    _sync()

    event_us = _bench_event_us(path.run, args.warmup, args.repeat)
    graph_us = _bench_graph_us(path.run, args.warmup, args.repeat, args.graph_iters)
    print(
        "[bench] "
        f"batch={case.batch} heads={case.heads} heads_kv={case.heads_kv} "
        f"seq_q={case.total_seq_q} seq_kv={case.total_seq_kv} dim={case.dim} "
        f"dtype={args.dtype} causal={case.causal}"
    )
    print(f"[bench] event_path={event_us:.3f} us")
    print(
        f"[bench] graph_path={graph_us:.3f} us "
        f"TFLOPS={case.total_flops / (graph_us * 1e-6) / 1e12:.3f}"
    )
    print(
        f"[bench] flops={case.total_flops:.0f} "
        f"graph_iters={args.graph_iters} repeat={args.repeat}"
    )

    if args.skip_profiler:
        return
    rows = _profile_kernel_path(path.run, args.profile_repeat)
    role_totals: dict[str, float] = {}
    for role, _, us, _ in rows:
        role_totals[role] = role_totals.get(role, 0.0) + us
    print("[profile] role totals per run:")
    for role, us in sorted(role_totals.items(), key=lambda item: item[1], reverse=True):
        role_tflops = case.total_flops / (us * 1e-6) / 1e12 if role == "bwd" else 0.0
        print(f"  {role:<10} {us:9.3f} us {role_tflops:8.3f} TFLOPS")
    print("[profile] top kernels per run:")
    for role, name, us, count in rows[:20]:
        print(f"  {role:<10} {us:9.3f} us count={count:<3} {name}")


if __name__ == "__main__":
    main()
