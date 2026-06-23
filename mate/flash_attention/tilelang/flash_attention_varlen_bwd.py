# ruff: noqa
# type: ignore
from dataclasses import replace
import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
from tvm import tir
from ...utils import cosize

TARGET = "musa"
DEVICE = "musa"

INT32_ADDRESS_SPACE_BYTES = torch.iinfo(torch.int32).max


def _make_jit_pass_configs(disable_index_type_promotion=True):
    return {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: disable_index_type_promotion,
    }


JIT_PASS_CONFIGS = _make_jit_pass_configs(disable_index_type_promotion=True)
JIT_PASS_CONFIGS_PROMOTE_INDEX = _make_jit_pass_configs(
    disable_index_type_promotion=False
)

JIT_COMPILE_FLAGS = [
    # "-Od3",
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-if-convert=1",
    # "-mllvm",
    # "-mtgpu-combine-instr-with-burst=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


def perm_n(n, block_N):
    # assume n in blockN
    l = block_N // 8
    return (n % 8) * l + (n // 8)


def _check_last_dim_stride_one(name, tensor):
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have contiguous last dimension")


def _check_attention_strides(name, tensor, multiple=8):
    _check_last_dim_stride_one(name, tensor)
    for dim, stride in enumerate(tensor.stride()[:-1]):
        if stride <= 0:
            raise ValueError(f"{name} stride({dim}) must be positive, got {stride}")
        if stride % multiple != 0:
            raise ValueError(
                f"{name} stride({dim}) must be divisible by {multiple}, got {stride}"
            )


def _tilelang_dtype_nbytes(dtype):
    if dtype in ("float16", "bfloat16"):
        return 2
    if dtype in ("float", "float32"):
        return 4
    raise ValueError(f"Unsupported TileLang dtype: {dtype}")


def _cosize_bytes(shape, strides, element_size):
    if any(extent == 0 for extent in shape):
        return 0
    return cosize(shape, strides) * element_size


def _tensor_cosize_bytes(tensor):
    return _cosize_bytes(
        tuple(tensor.shape), tuple(tensor.stride()), tensor.element_size()
    )


def _contiguous_cosize_bytes(shape, torch_dtype):
    return _cosize_bytes(
        tuple(shape), None, torch.empty((), dtype=torch_dtype).element_size()
    )


def _needs_index_type_promotion(*byte_spans):
    return any(byte_span > INT32_ADDRESS_SPACE_BYTES for byte_span in byte_spans)


def _clone_jit_with_pass_configs(jit_impl, pass_configs):
    return replace(jit_impl, pass_configs=dict(pass_configs))


_JIT_INDEX_PROMOTION_VARIANTS = {}


def _jit_for_index_type_promotion(jit_impl, enable_index_type_promotion):
    if not enable_index_type_promotion:
        return jit_impl
    key = id(jit_impl)
    variant = _JIT_INDEX_PROMOTION_VARIANTS.get(key)
    if variant is None:
        variant = _clone_jit_with_pass_configs(jit_impl, JIT_PASS_CONFIGS_PROMOTE_INDEX)
        _JIT_INDEX_PROMOTION_VARIANTS[key] = variant
    return variant


def _annotate_sqmma(buffer, k_major, continuity=None):
    if continuity is None:
        layout = tilelang.layout.make_sqmma_swizzled_layout(
            buffer[:, :], k_major=k_major
        )
    else:
        layout = tilelang.layout.make_sqmma_swizzled_layout(
            buffer[:, :], k_major=k_major, continuity=continuity
        )
    T.annotate_layout(
        {buffer[:, :]: layout},
        allow_reannotation=True,
        allow_buffer_region=True,
    )


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=True,
    compile_flags=JIT_COMPILE_FLAGS,
)
# fmt: off
def flashattn_bwd_ws(
    dim,
    is_causal,
    is_varlen,
    is_bhsd=False,
    heads_q_eq_heads_kv=False,
    block_M=64,
    block_N=64,
    smscale=None,
    threads=640,
    dtype="bfloat16",
    use_strided_tensors=True,
):
    if smscale is not None:
        rln2_scale = smscale * 1.44269504  # log2(e)
    else:
        smscale = 1.0 / (dim**0.5)
        rln2_scale = smscale * 1.44269504  # log2(e)
    num_blocks_kv = T.dynamic("num_blocks_kv")
    log2e = 1.44269504
    batch = T.dynamic("batch")
    heads_q = T.dynamic("heads_q")
    heads_kv = heads_q if heads_q_eq_heads_kv else T.dynamic("heads_kv")
    total_seq_q = T.dynamic("total_seq_q")
    total_seq_kv = T.dynamic("total_seq_kv")
    max_seq_q = T.dynamic("max_seq_q")
    seqlen_q = T.dynamic("seqlen_q")
    seqlen_k = T.dynamic("seqlen_k")
    head_groups = 1 if heads_q_eq_heads_kv else heads_q // heads_kv
    safe_seq_check = True
    safe_dim_check = False
    accum_dtype = "float"
    kv_grad_dtype = dtype if heads_q_eq_heads_kv else accum_dtype
    kv_grad_heads = heads_kv if heads_q_eq_heads_kv else heads_q
    L = block_N // 8  # 8 means 8 threads, L means Local Elem
    consumer0_threads = 256
    consumer1_threads = 256
    producer_threads = 128
    q_stride_b = T.dynamic("q_stride_b")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    k_stride_b = T.dynamic("k_stride_b")
    k_stride_s = T.dynamic("k_stride_s")
    k_stride_h = T.dynamic("k_stride_h")
    v_stride_b = T.dynamic("v_stride_b")
    v_stride_s = T.dynamic("v_stride_s")
    v_stride_h = T.dynamic("v_stride_h")
    o_stride_b = T.dynamic("o_stride_b")
    o_stride_s = T.dynamic("o_stride_s")
    o_stride_h = T.dynamic("o_stride_h")
    do_stride_b = T.dynamic("do_stride_b")
    do_stride_s = T.dynamic("do_stride_s")
    do_stride_h = T.dynamic("do_stride_h")
    dk_stride_b = T.dynamic("dk_stride_b")
    dk_stride_s = T.dynamic("dk_stride_s")
    dk_stride_h = T.dynamic("dk_stride_h")
    dv_stride_b = T.dynamic("dv_stride_b")
    dv_stride_s = T.dynamic("dv_stride_s")
    dv_stride_h = T.dynamic("dv_stride_h")
    q_shape = (
        (total_seq_q, heads_q, dim) if is_varlen else (batch, seqlen_q, heads_q, dim)
    )
    kv_shape = (
        (total_seq_kv, heads_kv, dim) if is_varlen else (batch, seqlen_k, heads_kv, dim)
    )
    kv_grad_shape = (
        (total_seq_kv, kv_grad_heads, dim)
        if is_varlen
        else (batch, seqlen_k, kv_grad_heads, dim)
    )
    q_strides = (
        (q_stride_s, q_stride_h, 1)
        if is_varlen
        else (q_stride_b, q_stride_s, q_stride_h, 1)
    )
    k_strides = (
        (k_stride_s, k_stride_h, 1)
        if is_varlen
        else (k_stride_b, k_stride_s, k_stride_h, 1)
    )
    v_strides = (
        (v_stride_s, v_stride_h, 1)
        if is_varlen
        else (v_stride_b, v_stride_s, v_stride_h, 1)
    )
    o_strides = (
        (o_stride_s, o_stride_h, 1)
        if is_varlen
        else (o_stride_b, o_stride_s, o_stride_h, 1)
    )
    do_strides = (
        (do_stride_s, do_stride_h, 1)
        if is_varlen
        else (do_stride_b, do_stride_s, do_stride_h, 1)
    )
    dk_strides = (
        (dk_stride_s, dk_stride_h, 1)
        if is_varlen
        else (dk_stride_b, dk_stride_s, dk_stride_h, 1)
    )
    dv_strides = (
        (dv_stride_s, dv_stride_h, 1)
        if is_varlen
        else (dv_stride_b, dv_stride_s, dv_stride_h, 1)
    )
    q_type = T.StridedTensor(q_shape, q_strides, dtype)
    k_type = T.StridedTensor(kv_shape, k_strides, dtype)
    v_type = T.StridedTensor(kv_shape, v_strides, dtype)
    o_type = T.StridedTensor(q_shape, o_strides, dtype)
    do_type = T.StridedTensor(q_shape, do_strides, dtype)
    dk_type = T.StridedTensor(kv_grad_shape, dk_strides, kv_grad_dtype)
    dv_type = T.StridedTensor(kv_grad_shape, dv_strides, kv_grad_dtype)
    dtype_bytes = _tilelang_dtype_nbytes(dtype)
    k_robust_bytes = cosize(kv_shape, k_strides) * dtype_bytes
    v_robust_bytes = cosize(kv_shape, v_strides) * dtype_bytes

    def q_block(tensor, batch_idx, q_start, begin_seq, head, dim_start):
        if is_varlen:
            return tensor[
                q_start : q_start + block_M,
                head,
                dim_start : dim_start + dim // 2,
            ]
        local_q_start = q_start - begin_seq
        return tensor[
            batch_idx,
            local_q_start : local_q_start + block_M,
            head,
            dim_start : dim_start + dim // 2,
        ]

    def kv_block(tensor, batch_idx, kv_start, begin_seq, head, dim_start):
        if is_varlen:
            return tensor[
                kv_start : kv_start + block_N,
                head,
                dim_start : dim_start + dim // 2,
            ]
        local_kv_start = kv_start - begin_seq
        return tensor[
            batch_idx,
            local_kv_start : local_kv_start + block_N,
            head,
            dim_start : dim_start + dim // 2,
        ]

    def kv_elem(tensor, batch_idx, kv_idx, begin_seq, head, dim_idx):
        if is_varlen:
            return tensor[kv_idx, head, dim_idx]
        return tensor[batch_idx, kv_idx - begin_seq, head, dim_idx]

    def kv_tile_e(tensor, batch_idx, kv_offset, begin_seq, head, dim_idx):
        return kv_elem(
            tensor, batch_idx, begin_seq + kv_offset, begin_seq, head, dim_idx
        )

    def kv_tile_b(tensor, batch_idx, kv_offset, begin_seq, head, dim_start):
        return kv_block(
            tensor, batch_idx, begin_seq + kv_offset, begin_seq, head, dim_start
        )

    def permuted_kv_idx(begin_seq, block_start, idx):
        return begin_seq + block_start + perm_n(idx, block_N)

    def permuted_kv_valid(begin_seq, block_start, idx, end_seq):
        return permuted_kv_idx(begin_seq, block_start, idx) < end_seq

    def lse_elem(tensor, batch_idx, q_idx, begin_seq, head):
        if is_varlen:
            return tensor[head, q_idx]
        return tensor[batch_idx, head, q_idx - begin_seq]

    @T.prim_func
    def flashattn_bwd_ws_kernel(
        Q: q_type,  # type: ignore
        K: k_type,  # type: ignore
        V: v_type,  # type: ignore
        Output: o_type,  # type: ignore
        dQ_accum: T.Tensor([batch, heads_q, max_seq_q, dim], accum_dtype),  # type: ignore
        dK: dk_type,  # type: ignore
        dV: dv_type,  # type: ignore
        dO: do_type,  # type: ignore
        cu_seq_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        cu_seq_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
        Lse: T.Tensor(
            [heads_q, total_seq_q] if is_varlen else [batch, heads_q, seqlen_q],
            accum_dtype,
        ),  # type: ignore
        Delta: T.Tensor([total_seq_q, heads_q], accum_dtype),  # type: ignore
        debug: T.Tensor([num_blocks_kv], T.int32),  # type: ignore
    ):
        with T.Kernel(
            num_blocks_kv,
            heads_q,
            batch,
            threads=threads,
        ) as (bx, by, bz):
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(k_stride_s % 8 == 0)
            T.assume(k_stride_h % 8 == 0)
            T.assume(v_stride_s % 8 == 0)
            T.assume(v_stride_h % 8 == 0)
            T.assume(o_stride_s % 8 == 0)
            T.assume(o_stride_h % 8 == 0)
            T.assume(do_stride_s % 8 == 0)
            T.assume(do_stride_h % 8 == 0)
            T.assume(dk_stride_s % 8 == 0)
            T.assume(dk_stride_h % 8 == 0)
            T.assume(dv_stride_s % 8 == 0)
            T.assume(dv_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(q_stride_b % 8 == 0)
                T.assume(k_stride_b % 8 == 0)
                T.assume(v_stride_b % 8 == 0)
                T.assume(o_stride_b % 8 == 0)
                T.assume(do_stride_b % 8 == 0)
                T.assume(dk_stride_b % 8 == 0)
                T.assume(dv_stride_b % 8 == 0)
            begin_seq_q = T.alloc_var(T.int32)
            end_seq_q = T.alloc_var(T.int32)
            begin_seq_kv = T.alloc_var(T.int32)
            end_seq_kv = T.alloc_var(T.int32)
            kv_group = T.alloc_var(T.int32)
            kv_group = by if heads_q_eq_heads_kv else by // head_groups
            if is_varlen:
                begin_seq_q = cu_seq_q[bz]
                end_seq_q = cu_seq_q[bz + 1]
                begin_seq_kv = cu_seq_kv[bz]
                end_seq_kv = cu_seq_kv[bz + 1]
            else:
                begin_seq_q = bz * seqlen_q
                end_seq_q = begin_seq_q + seqlen_q
                begin_seq_kv = bz * seqlen_k
                end_seq_kv = begin_seq_kv + seqlen_k

            Qa_shared_0 = T.alloc_shared([block_M, dim // 2], dtype)
            dOb_shared_0 = Qa_shared_0
            # dOb_shared_0 = T.view(Qa_shared_0, [block_M, dim//2], dtype)
            Qa_shared_1 = T.alloc_shared([block_M, dim // 2], dtype)
            dOb_shared_1 = Qa_shared_1
            # dOb_shared_1 = T.view(Qa_shared_1, [block_M, dim//2], dtype)
            Qb_shared_0 = T.alloc_shared([block_M, dim // 2], dtype)
            dOa_shared_0 = Qb_shared_0
            # dOa_shared_0 = T.view(Qb_shared_0, [block_M, dim//2], dtype)
            Qb_shared_1 = T.alloc_shared([block_M, dim // 2], dtype)
            dOa_shared_1 = Qb_shared_1
            # dOa_shared_1 = T.view(Qb_shared_1, [block_M, dim//2], dtype)
            K_shared_0 = T.alloc_shared([block_N, dim // 2], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim // 2], dtype)
            Kt_shared_0 = T.alloc_shared([block_N, dim // 2], dtype)
            Kt_shared_1 = T.alloc_shared([block_N, dim // 2], dtype)
            Vt_shared_0 = T.alloc_shared([block_N, dim // 2], dtype)
            Vt_shared_1 = T.alloc_shared([block_N, dim // 2], dtype)
            Pt_shared = T.alloc_shared([block_M, block_N], dtype)
            dSt_shared = T.alloc_shared([block_M, block_N], dtype)
            dS_shared = T.alloc_shared([block_M, block_N], dtype)

            bar_k = T.alloc_barrier(arrive_count=128)
            bar_kt = T.alloc_barrier(arrive_count=128)
            bar_vt = T.alloc_barrier(arrive_count=128)
            bar_qa0_ready = T.alloc_barrier(arrive_count=128)
            bar_qa1_ready = T.alloc_barrier(arrive_count=128)
            bar_qb0_ready = T.alloc_barrier(arrive_count=128)
            bar_qb1_ready = T.alloc_barrier(arrive_count=128)
            bar_doa0_ready = T.alloc_barrier(arrive_count=128)
            bar_doa1_ready = T.alloc_barrier(arrive_count=128)
            bar_dob0_ready = T.alloc_barrier(arrive_count=128)
            bar_dob1_ready = T.alloc_barrier(arrive_count=128)
            bar_qa0_free = T.alloc_barrier(arrive_count=256)
            bar_qa1_free = T.alloc_barrier(arrive_count=256)
            bar_qb0_free = T.alloc_barrier(arrive_count=256)
            bar_qb1_free = T.alloc_barrier(arrive_count=256)
            bar_doa0_free = T.alloc_barrier(arrive_count=256)
            bar_doa1_free = T.alloc_barrier(arrive_count=256)
            bar_dob0_free = T.alloc_barrier(arrive_count=256)
            bar_dob1_free = T.alloc_barrier(arrive_count=256)
            bar_p_ready = T.alloc_barrier(arrive_count=256)
            bar_p_free = T.alloc_barrier(arrive_count=256)
            bar_ds_ready = T.alloc_barrier(arrive_count=256)
            bar_ds_free = T.alloc_barrier(arrive_count=256)
            bar_dst_ready = T.alloc_barrier(arrive_count=256)
            bar_dst_free = T.alloc_barrier(arrive_count=256)

            T.sync_threads()
            if is_varlen:
                k_robust_desc = T.make_robust_desc(
                    T.address_of(K[0, 0, 0]), k_robust_bytes
                )
                v_robust_desc = T.make_robust_desc(
                    T.address_of(V[0, 0, 0]), v_robust_bytes
                )
            else:
                k_robust_desc = T.make_robust_desc(
                    T.address_of(K[0, 0, 0, 0]), k_robust_bytes
                )
                v_robust_desc = T.make_robust_desc(
                    T.address_of(V[0, 0, 0, 0]), v_robust_bytes
                )
            if is_varlen:
                lse_robust_desc = T.make_robust_desc(
                    T.address_of(Lse[0, 0]), (total_seq_q * heads_q * 4)
                )
            else:
                lse_robust_desc = T.make_robust_desc(
                    T.address_of(Lse[0, 0, 0]), (batch * seqlen_q * heads_q * 4)
                )
            delta_robust_desc = T.make_robust_desc(
                T.address_of(Delta[0, 0]), (total_seq_q * heads_q * 4)
            )

            tid = T.get_thread_binding()
            block_kv_start = bx * block_N
            block_kv_end = (bx + 1) * block_N
            local_seq_q = T.alloc_var(T.int32)
            local_seq_kv = T.alloc_var(T.int32)
            causal_offset = T.alloc_var(T.int32)
            causal_q_local_start = T.alloc_var(T.int32)
            causal_q_local_mask_end = T.alloc_var(T.int32)
            causal_q_start = T.alloc_var(T.int32)
            causal_q_mask_end = T.alloc_var(T.int32)
            local_seq_q = end_seq_q - begin_seq_q
            local_seq_kv = end_seq_kv - begin_seq_kv
            causal_offset = local_seq_kv - local_seq_q
            causal_q_local_start = block_kv_start - causal_offset
            causal_q_local_start = T.if_then_else(
                causal_q_local_start > 0, causal_q_local_start, 0
            )
            causal_q_start = (
                begin_seq_q + T.floordiv(causal_q_local_start, block_M) * block_M
            )
            causal_q_local_mask_end = block_kv_end - causal_offset
            causal_q_local_mask_end = T.if_then_else(
                causal_q_local_mask_end > 0, causal_q_local_mask_end, 0
            )
            causal_q_mask_end = (
                begin_seq_q + T.ceildiv(causal_q_local_mask_end, block_M) * block_M
            )
            causal_q_mask_end = T.if_then_else(
                causal_q_mask_end < end_seq_q, causal_q_mask_end, end_seq_q
            )
            block_kv_all_invalid = (block_kv_start + begin_seq_kv) >= end_seq_kv
            block_kv_all_valid = (block_kv_end + begin_seq_kv) <= end_seq_kv
            block_kv_partial_valid = (block_kv_start + begin_seq_kv) < end_seq_kv and (
                block_kv_end + begin_seq_kv
            ) > end_seq_kv

            # if not block_kv_all_invalid:
            if (block_kv_start + begin_seq_kv) < end_seq_kv:
                if tid >= 512:
                    # tme load k
                    phase_producer = T.alloc_var(T.int32)
                    phase_producer = 0
                    T.copy(
                        kv_block(
                            K,
                            bz,
                            begin_seq_kv + block_kv_start,
                            begin_seq_kv,
                            kv_group,
                            0,
                        ),
                        K_shared_0,
                        barrier=bar_k,
                    )
                    T.copy(
                        kv_block(
                            K,
                            bz,
                            begin_seq_kv + block_kv_start,
                            begin_seq_kv,
                            kv_group,
                            dim // 2,
                        ),
                        K_shared_1,
                        barrier=bar_k,
                    )
                    T.barrier_arrive(bar_k)
                    # K_buffer debug hook for direct global-memory addressing.
                    for i, j in T.Parallel(block_N, dim // 2):
                        T.copy(
                            kv_elem(
                                K,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j,
                            ),
                            Kt_shared_0[i, j],
                            force_async_copy=True,
                            src_robust_desc=k_robust_desc,
                        )
                    for i, j in T.Parallel(block_N, dim // 2):
                        T.copy(
                            kv_elem(
                                K,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j + dim // 2,
                            ),
                            Kt_shared_1[i, j],
                            force_async_copy=True,
                            src_robust_desc=k_robust_desc,
                        )
                    for i, j in T.Parallel(block_N, dim // 2):
                        T.copy(
                            kv_elem(
                                V,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j,
                            ),
                            Vt_shared_0[i, j],
                            force_async_copy=True,
                            src_robust_desc=v_robust_desc,
                        )
                    for i, j in T.Parallel(block_N, dim // 2):
                        T.copy(
                            kv_elem(
                                V,
                                bz,
                                permuted_kv_idx(begin_seq_kv, block_kv_start, i),
                                begin_seq_kv,
                                kv_group,
                                j + dim // 2,
                            ),
                            Vt_shared_1[i, j],
                            force_async_copy=True,
                            src_robust_desc=v_robust_desc,
                        )
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    # if oob_n
                    if (block_kv_end + begin_seq_kv) > end_seq_kv:
                        for i, j in T.Parallel(block_N, dim // 2):
                            valid = permuted_kv_valid(
                                begin_seq_kv, block_kv_start, i, end_seq_kv
                            )
                            Kt_shared_0[i, j] = T.if_then_else(
                                valid, Kt_shared_0[i, j], 0.0
                            )
                            Kt_shared_1[i, j] = T.if_then_else(
                                valid, Kt_shared_1[i, j], 0.0
                            )
                            Vt_shared_0[i, j] = T.if_then_else(
                                valid, Vt_shared_0[i, j], 0.0
                            )
                            Vt_shared_1[i, j] = T.if_then_else(
                                valid, Vt_shared_1[i, j], 0.0
                            )
                    T.lma_wait()
                    T.barrier_arrive(bar_kt)
                    T.barrier_arrive(bar_vt)
                    for q_start in range(
                        causal_q_start if is_causal else begin_seq_q, end_seq_q, block_M
                    ):
                        T.barrier_wait(bar_qa0_free, phase_producer ^ 1)
                        _annotate_sqmma(Qa_shared_0, k_major=True)
                        T.copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, 0),
                            Qa_shared_0,
                            barrier=bar_qa0_ready,
                        )
                        T.barrier_arrive(bar_qa0_ready)

                        T.barrier_wait(bar_qa1_free, phase_producer ^ 1)
                        _annotate_sqmma(Qa_shared_1, k_major=True)
                        T.copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, dim // 2),
                            Qa_shared_1,
                            barrier=bar_qa1_ready,
                        )
                        T.barrier_arrive(bar_qa1_ready)

                        T.barrier_wait(bar_doa0_free, phase_producer ^ 1)
                        _annotate_sqmma(dOa_shared_0, k_major=True)
                        T.copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, 0),
                            dOa_shared_0,
                            barrier=bar_doa0_ready,
                        )
                        T.barrier_arrive(bar_doa0_ready)

                        T.barrier_wait(bar_doa1_free, phase_producer ^ 1)
                        _annotate_sqmma(dOa_shared_1, k_major=True)
                        T.copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, dim // 2),
                            dOa_shared_1,
                            barrier=bar_doa1_ready,
                        )
                        T.barrier_arrive(bar_doa1_ready)

                        T.barrier_wait(bar_dob0_free, phase_producer)
                        _annotate_sqmma(dOb_shared_0, k_major=False, continuity=64)
                        T.copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, 0),
                            dOb_shared_0,
                            barrier=bar_dob0_ready,
                        )
                        T.barrier_arrive(bar_dob0_ready)

                        T.barrier_wait(bar_dob1_free, phase_producer)
                        _annotate_sqmma(dOb_shared_1, k_major=False, continuity=64)
                        T.copy(
                            q_block(dO, bz, q_start, begin_seq_q, by, dim // 2),
                            dOb_shared_1,
                            barrier=bar_dob1_ready,
                        )
                        T.barrier_arrive(bar_dob1_ready)

                        T.barrier_wait(bar_qb0_free, phase_producer)
                        _annotate_sqmma(Qb_shared_0, k_major=False, continuity=64)
                        T.copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, 0),
                            Qb_shared_0,
                            barrier=bar_qb0_ready,
                        )
                        T.barrier_arrive(bar_qb0_ready)

                        T.barrier_wait(bar_qb1_free, phase_producer)
                        _annotate_sqmma(Qb_shared_1, k_major=False, continuity=64)
                        T.copy(
                            q_block(Q, bz, q_start, begin_seq_q, by, dim // 2),
                            Qb_shared_1,
                            barrier=bar_qb1_ready,
                        )
                        T.barrier_arrive(bar_qb1_ready)
                        phase_producer = phase_producer ^ 1
                    if begin_seq_q > total_seq_q:
                        debug[0] = 0

                elif tid >= 256:
                    # P*dOa -> dV
                    # ds*K -> dQ
                    phase_consumer1 = T.alloc_var(T.int32)
                    phase_consumer1 = 0
                    T.barrier_wait(bar_k, 0)
                    T.barrier_wait(bar_kt, 0)
                    T.barrier_wait(bar_vt, 0)
                    dv_accum_0 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    dq_accum_0 = T.alloc_fragment([block_M, dim // 2], accum_dtype)
                    dv_accum_1 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    dq_accum_1 = T.alloc_fragment([block_M, dim // 2], accum_dtype)
                    dv_cast_0 = T.alloc_fragment([block_N, dim // 2], dtype)
                    dv_cast_1 = T.alloc_fragment([block_N, dim // 2], dtype)
                    T.fill(dv_accum_0, 0.0)
                    T.fill(dv_accum_1, 0.0)
                    for q_start in range(
                        causal_q_start if is_causal else begin_seq_q, end_seq_q, block_M
                    ):
                        # is_last_q_tile = (q_start+block_M) >= end_seq_q
                        T.barrier_wait(bar_p_ready, phase_consumer1)
                        T.barrier_wait(bar_dob0_ready, phase_consumer1)
                        _annotate_sqmma(Pt_shared, k_major=False, continuity=64)
                        _annotate_sqmma(dOb_shared_0, k_major=False, continuity=64)
                        T.gemm(
                            Pt_shared,
                            dOb_shared_0,
                            dv_accum_0,
                            wg_wait=0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.barrier_wait(bar_dob1_ready, phase_consumer1)
                        _annotate_sqmma(dOb_shared_1, k_major=False, continuity=64)
                        T.gemm(
                            Pt_shared,
                            dOb_shared_1,
                            dv_accum_1,
                            wg_wait=0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_qa0_free)
                        T.barrier_arrive(bar_qa1_free)
                        T.barrier_arrive(bar_p_free)

                        T.barrier_wait(bar_ds_ready, phase_consumer1)
                        _annotate_sqmma(dS_shared, k_major=True)
                        T.gemm(
                            dS_shared,
                            K_shared_0,
                            dq_accum_0,
                            wg_wait=0,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.gemm(
                            dS_shared,
                            K_shared_1,
                            dq_accum_1,
                            wg_wait=0,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_ds_free)

                        local_q_start = q_start - begin_seq_q
                        for i, j in T.Parallel(block_M, dim // 2):
                            if local_q_start + i < local_seq_q:
                                T.atomic_add(
                                    dQ_accum[bz, by, local_q_start + i, j],
                                    dq_accum_0[i, j],
                                )
                                T.atomic_add(
                                    dQ_accum[bz, by, local_q_start + i, dim // 2 + j],
                                    dq_accum_1[i, j],
                                )
                        phase_consumer1 = phase_consumer1 ^ 1
                    if heads_q_eq_heads_kv:
                        T.copy(dv_accum_0, dv_cast_0)
                        T.copy(dv_accum_1, dv_cast_1)
                    if (block_kv_end + begin_seq_kv) > end_seq_kv:
                        for i, j in T.Parallel(block_N, dim // 2):
                            kv_offset = block_kv_start + i
                            if begin_seq_kv + kv_offset < end_seq_kv:
                                if heads_q_eq_heads_kv:
                                    T.copy(
                                        dv_cast_0[i, j],
                                        kv_tile_e(
                                            dV, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                    T.copy(
                                        dv_cast_1[i, j],
                                        kv_tile_e(
                                            dV,
                                            bz,
                                            kv_offset,
                                            begin_seq_kv,
                                            by,
                                            dim // 2 + j,
                                        ),
                                    )
                                else:
                                    T.copy(
                                        dv_accum_0[i, j],
                                        kv_tile_e(
                                            dV, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                    T.copy(
                                        dv_accum_1[i, j],
                                        kv_tile_e(
                                            dV,
                                            bz,
                                            kv_offset,
                                            begin_seq_kv,
                                            by,
                                            dim // 2 + j,
                                        ),
                                    )
                    else:
                        if heads_q_eq_heads_kv:
                            T.copy(
                                dv_cast_0,
                                kv_tile_b(dV, bz, block_kv_start, begin_seq_kv, by, 0),
                            )
                            T.copy(
                                dv_cast_1,
                                kv_tile_b(
                                    dV, bz, block_kv_start, begin_seq_kv, by, dim // 2
                                ),
                            )
                        else:
                            T.copy(
                                dv_accum_0,
                                kv_tile_b(dV, bz, block_kv_start, begin_seq_kv, by, 0),
                            )
                            T.copy(
                                dv_accum_1,
                                kv_tile_b(
                                    dV, bz, block_kv_start, begin_seq_kv, by, dim // 2
                                ),
                            )

                else:
                    # Qa * Kt -> s
                    # dOb * Vt -> dp
                    # dst * Qb -> dK
                    phase_consumer0 = T.alloc_var(T.int32)
                    phase_consumer0 = 0
                    dk_accum_0 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    dk_accum_1 = T.alloc_fragment([block_N, dim // 2], accum_dtype)
                    accs_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                    dp_accum = T.alloc_fragment([block_M, block_N], accum_dtype)
                    accs_cast = T.alloc_fragment([block_M, block_N], dtype)
                    lse_buffer = T.alloc_fragment([block_M], accum_dtype)
                    delta_buffer = T.alloc_fragment([block_M], accum_dtype)
                    T.barrier_wait(bar_k, 0)
                    T.barrier_wait(bar_kt, 0)
                    T.barrier_wait(bar_vt, 0)
                    T.fill(dk_accum_0, 0.0)
                    T.fill(dk_accum_1, 0.0)
                    dk_cast_0 = T.alloc_fragment([block_N, dim // 2], dtype)
                    dk_cast_1 = T.alloc_fragment([block_N, dim // 2], dtype)
                    for q_start in range(
                        causal_q_start if is_causal else begin_seq_q, end_seq_q, block_M
                    ):
                        # is_last_q_tile = (q_start+block_M) >= end_seq_q
                        T.barrier_wait(bar_qa0_ready, phase_consumer0)
                        _annotate_sqmma(Qa_shared_0, k_major=True)
                        T.gemm(
                            Qa_shared_0,
                            Kt_shared_0,
                            accs_accum,
                            clear_accum=True,
                            wg_wait=0,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.barrier_wait(bar_qa1_ready, phase_consumer0)
                        _annotate_sqmma(Qa_shared_1, k_major=True)
                        T.gemm(
                            Qa_shared_1,
                            Kt_shared_1,
                            accs_accum,
                            wg_wait=0,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_dob0_free)
                        T.barrier_arrive(bar_dob1_free)
                        T.barrier_wait(bar_doa0_ready, phase_consumer0)
                        _annotate_sqmma(dOa_shared_0, k_major=True)
                        T.gemm(
                            dOa_shared_0,
                            Vt_shared_0,
                            dp_accum,
                            clear_accum=True,
                            wg_wait=-1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.barrier_wait(bar_doa1_ready, phase_consumer0)
                        _annotate_sqmma(dOa_shared_1, k_major=True)
                        T.gemm(
                            dOa_shared_1,
                            Vt_shared_1,
                            dp_accum,
                            wg_wait=-1,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.wait_wgmma(0)
                        T.barrier_wait(bar_dob0_ready, phase_consumer0)
                        for i in T.Parallel(block_M):
                            T.copy(
                                lse_elem(Lse, bz, q_start + i, begin_seq_q, by),
                                lse_buffer[i],
                                src_robust_desc=lse_robust_desc,
                            )
                        # if is_last_q -> check oob_m
                        if (q_start + block_M) >= end_seq_q:
                            for i in T.Parallel(block_M):
                                lse_buffer[i] = T.if_then_else(
                                    q_start + i < end_seq_q, lse_buffer[i], 2**30
                                )
                            # for i, j in T.Parallel(block_M, block_N):
                            #     accs_accum[i, j] = T.if_then_else(
                            #                                     -2**30)
                        if (block_kv_end + begin_seq_kv) > end_seq_kv:
                            for i, j in T.Parallel(block_M, block_N):
                                accs_accum[i, j] = T.if_then_else(
                                    permuted_kv_valid(
                                        begin_seq_kv, block_kv_start, j, end_seq_kv
                                    ),
                                    accs_accum[i, j],
                                    -(2**30),
                                )

                        if is_causal:
                            if q_start < causal_q_mask_end:
                                for i, j in T.Parallel(block_M, block_N):
                                    q_idx = q_start - begin_seq_q + i
                                    k_idx = bx * block_N + perm_n(j, block_N)
                                    valid = q_idx + causal_offset >= k_idx
                                    accs_accum[i, j] = T.if_then_else(
                                        valid, accs_accum[i, j], -(2**30)
                                    )
                        for i, j in T.Parallel(block_M, block_N):
                            accs_accum[i, j] = accs_accum[i, j] * rln2_scale
                        for i, j in T.Parallel(block_M, block_N):
                            accs_accum[i, j] = T.exp2(
                                accs_accum[i, j] - lse_buffer[i] * log2e
                            )
                        T.copy(accs_accum, accs_cast)
                        T.barrier_wait(bar_p_free, phase_consumer0 ^ 1)
                        for i, t in T.Parallel(block_M, 8):
                            base = t * L
                            for l in T.vectorized(L):
                                Pt_shared[i, base + l] = accs_cast[i, l * 8 + t]
                        T.sync_warp()
                        T.barrier_arrive(bar_p_ready)
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_qb0_free)
                        T.barrier_arrive(bar_qb1_free)
                        for i in T.Parallel(block_M):
                            T.copy(
                                Delta[q_start + i, by],
                                delta_buffer[i],
                                src_robust_desc=delta_robust_desc,
                            )
                        if (block_kv_end + begin_seq_kv) > end_seq_kv:
                            for i, j in T.Parallel(block_M, block_N):
                                dp_accum[i, j] = T.if_then_else(
                                    permuted_kv_valid(
                                        begin_seq_kv, block_kv_start, j, end_seq_kv
                                    ),
                                    dp_accum[i, j],
                                    0.0,
                                )
                        # if oob_m
                        if (q_start + block_M) >= end_seq_q:
                            for i in T.Parallel(block_M):
                                delta_buffer[i] = T.if_then_else(
                                    q_start + i < end_seq_q, delta_buffer[i], 0.0
                                )
                        for i, j in T.Parallel(block_M, block_N):
                            dp_accum[i, j] = (
                                (dp_accum[i, j] - delta_buffer[i])
                                * accs_accum[i, j]
                                * smscale
                            )
                        if (q_start + block_M) >= end_seq_q:
                            for i, j in T.Parallel(block_M, block_N):
                                dp_accum[i, j] = T.if_then_else(
                                    q_start + i < end_seq_q, dp_accum[i, j], 0.0
                                )
                        T.barrier_wait(bar_ds_free, phase_consumer0 ^ 1)
                        T.copy(dp_accum, accs_cast)
                        for i, t in T.Parallel(block_M, 8):
                            base = t * L
                            for l in T.vectorized(L):
                                dS_shared[i, base + l] = accs_cast[i, l * 8 + t]
                        T.sync_warp()
                        T.barrier_arrive(bar_ds_ready)
                        T.barrier_wait(bar_dst_free, phase_consumer0 ^ 1)
                        for i, t in T.Parallel(block_M, 8):
                            base = t * L
                            for l in T.vectorized(L):
                                dSt_shared[i, base + l] = accs_cast[i, l * 8 + t]
                        T.sync_warp()
                        T.barrier_wait(bar_qb0_ready, phase_consumer0)
                        T.barrier_arrive(bar_dst_ready)
                        T.barrier_wait(bar_dst_ready, phase_consumer0)
                        _annotate_sqmma(dSt_shared, k_major=False, continuity=64)
                        _annotate_sqmma(Qb_shared_0, k_major=False, continuity=64)
                        T.gemm(
                            dSt_shared,
                            Qb_shared_0,
                            dk_accum_0,
                            wg_wait=0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.barrier_wait(bar_qb1_ready, phase_consumer0)
                        _annotate_sqmma(Qb_shared_1, k_major=False, continuity=64)
                        T.gemm(
                            dSt_shared,
                            Qb_shared_1,
                            dk_accum_1,
                            wg_wait=0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_doa0_free)
                        T.barrier_arrive(bar_doa1_free)
                        T.barrier_arrive(bar_dst_free)
                        phase_consumer0 = phase_consumer0 ^ 1
                    if heads_q_eq_heads_kv:
                        T.copy(dk_accum_0, dk_cast_0)
                        T.copy(dk_accum_1, dk_cast_1)
                    if (block_kv_end + begin_seq_kv) > end_seq_kv:
                        for i, j in T.Parallel(block_N, dim // 2):
                            kv_offset = block_kv_start + i
                            if begin_seq_kv + kv_offset < end_seq_kv:
                                if heads_q_eq_heads_kv:
                                    T.copy(
                                        dk_cast_0[i, j],
                                        kv_tile_e(
                                            dK, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                    T.copy(
                                        dk_cast_1[i, j],
                                        kv_tile_e(
                                            dK,
                                            bz,
                                            kv_offset,
                                            begin_seq_kv,
                                            by,
                                            dim // 2 + j,
                                        ),
                                    )
                                else:
                                    T.copy(
                                        dk_accum_0[i, j],
                                        kv_tile_e(
                                            dK, bz, kv_offset, begin_seq_kv, by, j
                                        ),
                                    )
                                    T.copy(
                                        dk_accum_1[i, j],
                                        kv_tile_e(
                                            dK,
                                            bz,
                                            kv_offset,
                                            begin_seq_kv,
                                            by,
                                            dim // 2 + j,
                                        ),
                                    )
                    else:
                        if heads_q_eq_heads_kv:
                            T.copy(
                                dk_cast_0,
                                kv_tile_b(dK, bz, block_kv_start, begin_seq_kv, by, 0),
                            )
                            T.copy(
                                dk_cast_1,
                                kv_tile_b(
                                    dK, bz, block_kv_start, begin_seq_kv, by, dim // 2
                                ),
                            )
                        else:
                            T.copy(
                                dk_accum_0,
                                kv_tile_b(dK, bz, block_kv_start, begin_seq_kv, by, 0),
                            )
                            T.copy(
                                dk_accum_1,
                                kv_tile_b(
                                    dK, bz, block_kv_start, begin_seq_kv, by, dim // 2
                                ),
                            )

    # if heads_kv == heads_q -> output dk dv,
    # if heads_kv != heads_q -> output dk_accum dv_accum， -> reduce to dk dv
    return flashattn_bwd_ws_kernel


# fmt: on


def reduce_kv_grads(dK_accum, dV_accum, heads_kv):
    total_seq_kv, heads_q, dim = dK_accum.shape
    if dV_accum.shape != (total_seq_kv, heads_q, dim):
        raise ValueError("dK_accum and dV_accum must have the same shape")
    if heads_q % heads_kv != 0:
        raise ValueError(
            f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv})"
        )

    head_groups = heads_q // heads_kv
    if head_groups == 1:
        return dK_accum, dV_accum

    dK = dK_accum.reshape(total_seq_kv, heads_kv, head_groups, dim).sum(dim=2)
    dV = dV_accum.reshape(total_seq_kv, heads_kv, head_groups, dim).sum(dim=2)
    return dK.contiguous(), dV.contiguous()


def pack_dq_from_accum(dQ_accum, cu_seq_q, total_seq_q, dtype):
    batch, heads, _, dim = dQ_accum.shape
    dQ = torch.empty((total_seq_q, heads, dim), dtype=dtype, device=dQ_accum.device)
    for b in range(batch):
        q0 = cu_seq_q[b].item()
        q1 = cu_seq_q[b + 1].item()
        valid_q = q1 - q0
        dQ[q0:q1] = dQ_accum[b, :, :valid_q, :].permute(1, 0, 2).to(dtype)
    return dQ.contiguous()


def to_tilelang_dtype(torch_dtype):
    if torch_dtype == torch.bfloat16:
        return "bfloat16"
    if torch_dtype == torch.float16:
        return "float16"
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=False,
    compile_flags=JIT_COMPILE_FLAGS,
)
def pack_dq_from_accum_ws(
    dim,
    is_varlen,
    is_bhsd=False,
    dtype="bfloat16",
    block_M=64,
    block_D=32,
    threads=256,
    use_strided_tensors=True,
):
    max_seq_q = T.dynamic("max_seq_q")
    batch = T.dynamic("batch")
    heads = T.dynamic("heads")
    total_seq_q = T.dynamic("total_seq_q")
    seqlen_q = T.dynamic("seqlen_q")
    dq_stride_b = T.dynamic("dq_stride_b")
    dq_stride_s = T.dynamic("dq_stride_s")
    dq_stride_h = T.dynamic("dq_stride_h")
    num_dim_tiles = (dim + block_D - 1) // block_D
    dq_shape = (total_seq_q, heads, dim) if is_varlen else (batch, seqlen_q, heads, dim)
    dq_strides = (
        (dq_stride_s, dq_stride_h, 1)
        if is_varlen
        else (dq_stride_b, dq_stride_s, dq_stride_h, 1)
    )
    dq_type = T.StridedTensor(dq_shape, dq_strides, dtype)

    @T.prim_func
    def main(
        dQ_accum: T.Tensor([batch, heads, max_seq_q, dim], "float"),  # type: ignore
        cu_seq_q: T.Tensor([batch + 1], T.int32),  # type: ignore
        dQ: dq_type,  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(max_seq_q, block_M),
            heads,
            batch * num_dim_tiles,
            threads=threads,
        ) as (bx, by, bz):
            T.assume(dq_stride_s % 8 == 0)
            T.assume(dq_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(dq_stride_b % 8 == 0)
            batch_idx = bz // num_dim_tiles
            dim_tile = bz % num_dim_tiles
            begin_seq_q = T.alloc_var(T.int32)
            end_seq_q = T.alloc_var(T.int32)
            if is_varlen:
                begin_seq_q = cu_seq_q[batch_idx]
                end_seq_q = cu_seq_q[batch_idx + 1]
            else:
                begin_seq_q = batch_idx * seqlen_q
                end_seq_q = begin_seq_q + seqlen_q
            local_q_start = bx * block_M
            dim_start = dim_tile * block_D
            dQ_local = T.alloc_fragment([block_M, block_D], "float")
            dQ_cast = T.alloc_fragment([block_M, block_D], dtype)
            for i, j in T.Parallel(block_M, block_D):
                q_idx = local_q_start + i
                d_idx = dim_start + j
                if q_idx < (end_seq_q - begin_seq_q) and d_idx < dim:
                    T.copy(dQ_accum[batch_idx, by, q_idx, d_idx], dQ_local[i, j])
                else:
                    dQ_local[i, j] = 0.0
            T.copy(dQ_local, dQ_cast)
            for i, j in T.Parallel(block_M, block_D):
                q_idx = local_q_start + i
                d_idx = dim_start + j
                if q_idx < (end_seq_q - begin_seq_q) and d_idx < dim:
                    if is_varlen:
                        T.copy(dQ_cast[i, j], dQ[begin_seq_q + q_idx, by, d_idx])
                    else:
                        T.copy(dQ_cast[i, j], dQ[batch_idx, q_idx, by, d_idx])

    return main


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=False,
    compile_flags=JIT_COMPILE_FLAGS,
)
def compute_delta_ws(
    dim,
    is_varlen,
    is_bhsd=False,
    dtype="bfloat16",
    block_M=64,
    block_D=32,
    threads=256,
    use_strided_tensors=True,
):
    total_seq_q = T.dynamic("total_seq_q")
    heads = T.dynamic("heads")
    batch = T.dynamic("batch")
    seqlen_q = T.dynamic("seqlen_q")
    o_stride_b = T.dynamic("o_stride_b")
    o_stride_s = T.dynamic("o_stride_s")
    o_stride_h = T.dynamic("o_stride_h")
    do_stride_b = T.dynamic("do_stride_b")
    do_stride_s = T.dynamic("do_stride_s")
    do_stride_h = T.dynamic("do_stride_h")
    num_queries = total_seq_q if is_varlen else batch * seqlen_q
    warps_per_block = max(1, threads // 32)
    num_dim_tiles = (dim + 31) // 32
    q_shape = (total_seq_q, heads, dim) if is_varlen else (batch, seqlen_q, heads, dim)
    o_strides = (
        (o_stride_s, o_stride_h, 1)
        if is_varlen
        else (o_stride_b, o_stride_s, o_stride_h, 1)
    )
    do_strides = (
        (do_stride_s, do_stride_h, 1)
        if is_varlen
        else (do_stride_b, do_stride_s, do_stride_h, 1)
    )
    o_type = T.StridedTensor(q_shape, o_strides, dtype)
    do_type = T.StridedTensor(q_shape, do_strides, dtype)

    def q_elem(tensor, q_idx, head, dim_idx):
        if is_varlen:
            return tensor[q_idx, head, dim_idx]
        batch_idx = q_idx // seqlen_q
        local_q_idx = q_idx - batch_idx * seqlen_q
        return tensor[batch_idx, local_q_idx, head, dim_idx]

    @T.prim_func
    def main(
        Output: o_type,  # type: ignore
        dO: do_type,  # type: ignore
        Delta: T.Tensor([total_seq_q, heads], "float"),  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(num_queries, warps_per_block),
            heads,
            threads=threads,
        ) as (bx, by):
            T.assume(o_stride_s % 8 == 0)
            T.assume(o_stride_h % 8 == 0)
            T.assume(do_stride_s % 8 == 0)
            T.assume(do_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(o_stride_b % 8 == 0)
                T.assume(do_stride_b % 8 == 0)
            tid = T.get_thread_binding()
            warp_idx = tid // 32
            lane_idx = tid % 32
            q_idx = bx * warps_per_block + warp_idx
            delta_local = T.alloc_local([1], "float")
            delta_local[0] = 0.0
            for dim_tile in range(num_dim_tiles):
                d_idx = dim_tile * 32 + lane_idx
                if q_idx < num_queries and d_idx < dim:
                    delta_local[0] += tir.Cast(
                        "float32", q_elem(Output, q_idx, by, d_idx)
                    ) * tir.Cast("float32", q_elem(dO, q_idx, by, d_idx))
            delta_local[0] += T.shfl_xor(delta_local[0], 16)
            delta_local[0] += T.shfl_xor(delta_local[0], 8)
            delta_local[0] += T.shfl_xor(delta_local[0], 4)
            delta_local[0] += T.shfl_xor(delta_local[0], 2)
            delta_local[0] += T.shfl_xor(delta_local[0], 1)
            if q_idx < num_queries and lane_idx == 0:
                Delta[q_idx, by] = delta_local[0]

    return main


@tilelang.jit(
    out_idx=[],
    pass_configs=JIT_PASS_CONFIGS,
    verbose=False,
    compile_flags=JIT_COMPILE_FLAGS,
)
def reduce_kv_grads_ws(
    head_groups,
    dim,
    is_varlen,
    is_bhsd=False,
    block_T=8,
    block_D=32,
    threads=256,
    use_strided_tensors=True,
):
    batch = T.dynamic("batch")
    total_seq_kv = T.dynamic("total_seq_kv")
    seqlen_k = T.dynamic("seqlen_k")
    num_kv = total_seq_kv if is_varlen else batch * seqlen_k
    heads_kv = T.dynamic("heads_kv")
    dk_stride_b = T.dynamic("dk_stride_b")
    dk_stride_s = T.dynamic("dk_stride_s")
    dk_stride_h = T.dynamic("dk_stride_h")
    dv_stride_b = T.dynamic("dv_stride_b")
    dv_stride_s = T.dynamic("dv_stride_s")
    dv_stride_h = T.dynamic("dv_stride_h")
    kv_accum_shape = (
        (total_seq_kv, heads_kv * head_groups, dim)
        if is_varlen
        else (batch, seqlen_k, heads_kv * head_groups, dim)
    )
    kv_shape = (
        (total_seq_kv, heads_kv, dim) if is_varlen else (batch, seqlen_k, heads_kv, dim)
    )
    dk_strides = (
        (dk_stride_s, dk_stride_h, 1)
        if is_varlen
        else (dk_stride_b, dk_stride_s, dk_stride_h, 1)
    )
    dv_strides = (
        (dv_stride_s, dv_stride_h, 1)
        if is_varlen
        else (dv_stride_b, dv_stride_s, dv_stride_h, 1)
    )
    dk_type = T.StridedTensor(kv_shape, dk_strides, "float")
    dv_type = T.StridedTensor(kv_shape, dv_strides, "float")

    def kv_accum_elem(tensor, kv_idx, head, dim_idx):
        if is_varlen:
            return tensor[kv_idx, head, dim_idx]
        batch_idx = kv_idx // seqlen_k
        local_kv_idx = kv_idx - batch_idx * seqlen_k
        return tensor[batch_idx, local_kv_idx, head, dim_idx]

    def kv_grad_elem(tensor, kv_idx, head, dim_idx):
        if is_varlen:
            return tensor[kv_idx, head, dim_idx]
        batch_idx = kv_idx // seqlen_k
        local_kv_idx = kv_idx - batch_idx * seqlen_k
        return tensor[batch_idx, local_kv_idx, head, dim_idx]

    @T.prim_func
    def main(
        dK_accum: T.Tensor(kv_accum_shape, "float"),  # type: ignore
        dV_accum: T.Tensor(kv_accum_shape, "float"),  # type: ignore
        dK: dk_type,  # type: ignore
        dV: dv_type,  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(num_kv, block_T),
            heads_kv,
            T.ceildiv(dim, block_D),
            threads=threads,
        ) as (bx, by, bz):
            T.assume(dk_stride_s % 8 == 0)
            T.assume(dk_stride_h % 8 == 0)
            T.assume(dv_stride_s % 8 == 0)
            T.assume(dv_stride_h % 8 == 0)
            if not is_varlen:
                T.assume(dk_stride_b % 8 == 0)
                T.assume(dv_stride_b % 8 == 0)
            k_start = bx * block_T
            d_start = bz * block_D
            dK_local = T.alloc_fragment([block_T, block_D], "float")
            dV_local = T.alloc_fragment([block_T, block_D], "float")
            for i, j in T.Parallel(block_T, block_D):
                dK_local[i, j] = 0.0
                dV_local[i, j] = 0.0
            for g in range(head_groups):
                for i, j in T.Parallel(block_T, block_D):
                    kv_idx = k_start + i
                    d_idx = d_start + j
                    hq_idx = by * head_groups + g
                    if kv_idx < num_kv and d_idx < dim:
                        dK_local[i, j] += kv_accum_elem(dK_accum, kv_idx, hq_idx, d_idx)
                        dV_local[i, j] += kv_accum_elem(dV_accum, kv_idx, hq_idx, d_idx)
            for i, j in T.Parallel(block_T, block_D):
                kv_idx = k_start + i
                d_idx = d_start + j
                if kv_idx < num_kv and d_idx < dim:
                    T.copy(dK_local[i, j], kv_grad_elem(dK, kv_idx, by, d_idx))
                    T.copy(dV_local[i, j], kv_grad_elem(dV, kv_idx, by, d_idx))

    return main


def flashattn_varlen_bwd_interface(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    is_causal=False,
    smscale=None,
    dtype=None,
    block_M=64,
    block_N=64,
    threads=640,
    is_bhsd=False,
    _force_strided_tensors=True,
):
    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_varlen and (cu_seqlens_q is None or cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")

    if is_varlen:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("varlen inputs must have shape [total_seq, heads, dim]")
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("max_seqlen_q and max_seqlen_k are required for varlen")
        q_flat, k_flat, v_flat = q, k, v
        out_flat, dout_flat = out, dout
        total_seq_q, heads_q, dim = q_flat.shape
        total_seq_kv, heads_kv, _ = k_flat.shape
        out_expected_shape = q_flat.shape
        lse_expected_shape = (heads_q, total_seq_q)
    else:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "non-varlen inputs must have shape [batch, seqlen, heads, dim]"
            )
        if softmax_lse.ndim != 3:
            raise ValueError(
                "non-varlen softmax_lse must have shape [batch, heads, seqlen]"
            )
        batch = q.shape[0]
        if softmax_lse.shape[0] != batch:
            raise ValueError("q and softmax_lse must have the same batch size")
        _, seqlen_q, heads_q, dim = q.shape

        batch_k, seqlen_k, heads_kv, dim_k = k.shape
        expected_v_shape = (batch, seqlen_k, heads_kv, dim)
        if batch_k != batch:
            raise ValueError("q and k must have the same batch size for non-varlen")
        if dim_k != dim:
            raise ValueError("q and k must have the same head dimension")
        if v.shape != expected_v_shape:
            raise ValueError(
                f"v must match k layout for non-varlen backward, expected {expected_v_shape}, "
                f"got {tuple(v.shape)}"
            )
        total_seq_q = batch * seqlen_q
        total_seq_kv = batch * seqlen_k
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k
        q_flat, k_flat, v_flat = q, k, v
        out_flat, dout_flat = out, dout
        out_expected_shape = q.shape
        lse_expected_shape = (batch, heads_q, seqlen_q)
        if softmax_lse.shape != lse_expected_shape:
            raise ValueError(
                f"non-varlen BSHD softmax_lse must be shaped {lse_expected_shape}, "
                f"got {tuple(softmax_lse.shape)}"
            )

    if is_varlen and v_flat.shape != (total_seq_kv, heads_kv, dim):
        raise ValueError(
            "v must match k shape for the TileLang varlen backward interface"
        )
    if out.shape != out_expected_shape or dout.shape != out_expected_shape:
        raise ValueError("out and dout must have the same shape as q")
    if is_varlen and (cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1D tensors")
    for name, tensor in (
        ("q", q_flat),
        ("k", k_flat),
        ("v", v_flat),
        ("out", out_flat),
        ("dout", dout_flat),
    ):
        _check_attention_strides(name, tensor)
    if not softmax_lse.is_contiguous():
        raise ValueError("softmax_lse must be contiguous")
    use_strided_tensors = True

    kernel_dtype = dtype if dtype is not None else to_tilelang_dtype(q.dtype)
    heads_q_eq_heads_kv = heads_q == heads_kv
    if heads_q % heads_kv != 0:
        raise ValueError(
            f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv})"
        )
    batch = cu_seqlens_q.numel() - 1 if is_varlen else q.shape[0]
    num_blocks_kv = ceil_div(max_seqlen_k, block_N)
    max_seq_q_padded = ceil_div(max_seqlen_q, block_M) * block_M

    if softmax_lse.shape != lse_expected_shape:
        raise ValueError(
            f"softmax_lse must be shaped {lse_expected_shape}, got {tuple(softmax_lse.shape)}"
        )
    kernel_arg_byte_spans = [
        _tensor_cosize_bytes(q_flat),
        _tensor_cosize_bytes(k_flat),
        _tensor_cosize_bytes(v_flat),
        _tensor_cosize_bytes(out_flat),
        _tensor_cosize_bytes(dout_flat),
        _tensor_cosize_bytes(softmax_lse),
        _contiguous_cosize_bytes((total_seq_q, heads_q), torch.float32),
        _contiguous_cosize_bytes(
            (batch, heads_q, max_seq_q_padded, dim), torch.float32
        ),
    ]
    if not heads_q_eq_heads_kv:
        kv_accum_shape = (
            (total_seq_kv, heads_q, dim)
            if is_varlen
            else (batch, seqlen_k, heads_q, dim)
        )
        kv_grad_shape = (total_seq_kv, heads_kv, dim) if is_varlen else k.shape
        kernel_arg_byte_spans.extend(
            (
                _contiguous_cosize_bytes(kv_accum_shape, torch.float32),
                _contiguous_cosize_bytes(kv_grad_shape, torch.float32),
            )
        )
    enable_index_type_promotion = _needs_index_type_promotion(*kernel_arg_byte_spans)

    delta = torch.empty((total_seq_q, heads_q), device=q.device, dtype=torch.float32)
    compute_delta = _jit_for_index_type_promotion(
        compute_delta_ws, enable_index_type_promotion
    )
    compute_delta(
        dim,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        dtype=kernel_dtype,
        block_M=block_M,
        threads=threads,
        use_strided_tensors=use_strided_tensors,
    )(out_flat, dout_flat, delta)
    dQ_accum = torch.zeros(
        (batch, heads_q, max_seq_q_padded, dim), dtype=torch.float32, device=q.device
    )
    if heads_q_eq_heads_kv:
        dK_accum = torch.zeros_like(k)
        dV_accum = torch.zeros_like(v)
    else:
        kv_accum_shape = (
            (total_seq_kv, heads_q, dim)
            if is_varlen
            else (batch, seqlen_k, heads_q, dim)
        )
        dK_accum = torch.zeros(kv_accum_shape, dtype=torch.float32, device=q.device)
        dV_accum = torch.zeros(kv_accum_shape, dtype=torch.float32, device=q.device)
    debug = torch.empty([num_blocks_kv], device=q.device, dtype=torch.int32)
    if is_varlen:
        cu_seqlens_q_arg = cu_seqlens_q
        cu_seqlens_k_arg = cu_seqlens_k
    else:
        cu_seqlens_dummy = torch.empty((batch + 1,), device=q.device, dtype=torch.int32)
        cu_seqlens_q_arg = cu_seqlens_dummy
        cu_seqlens_k_arg = cu_seqlens_dummy

    flashattn_bwd = _jit_for_index_type_promotion(
        flashattn_bwd_ws, enable_index_type_promotion
    )
    kernel = flashattn_bwd(
        # num_blocks_kv=num_blocks_kv,
        dim=dim,
        is_causal=is_causal,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        heads_q_eq_heads_kv=heads_q_eq_heads_kv,
        block_M=block_M,
        block_N=block_N,
        smscale=smscale,
        threads=threads,
        dtype=kernel_dtype,
        use_strided_tensors=use_strided_tensors,
    )
    kernel(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dQ_accum,
        dK_accum,
        dV_accum,
        dout_flat,
        cu_seqlens_q_arg,
        cu_seqlens_k_arg,
        softmax_lse,
        delta,
        debug,
    )
    # kernel.show_source()
    dq_shape = (total_seq_q, heads_q, dim) if is_varlen else q.shape
    dq = torch.empty(dq_shape, dtype=q.dtype, device=q.device)
    pack_dq = _jit_for_index_type_promotion(
        pack_dq_from_accum_ws, enable_index_type_promotion
    )
    pack_dq(
        dim,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        dtype=kernel_dtype,
        use_strided_tensors=use_strided_tensors,
    )(dQ_accum, cu_seqlens_q_arg, dq)
    if heads_q_eq_heads_kv:
        dk, dv = dK_accum, dV_accum
    else:
        kv_shape = (total_seq_kv, heads_kv, dim) if is_varlen else k.shape
        dk = torch.empty(kv_shape, dtype=torch.float32, device=q.device)
        dv = torch.empty(kv_shape, dtype=torch.float32, device=q.device)
        reduce_kv_grads = _jit_for_index_type_promotion(
            reduce_kv_grads_ws, enable_index_type_promotion
        )
        reduce_kv_grads(
            heads_q // heads_kv,
            dim,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            use_strided_tensors=use_strided_tensors,
        )(dK_accum, dV_accum, dk, dv)
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)
    return dq, dk, dv
