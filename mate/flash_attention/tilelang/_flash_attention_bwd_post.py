# ruff: noqa
# type: ignore
import torch

from ._flash_attention_bwd_common import (
    JIT_COMPILE_FLAGS,
    JIT_PASS_CONFIGS,
    T,
    tilelang,
    tir,
)


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
    accum_dtype="float",
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
        dQ_accum: T.Tensor([batch, heads, max_seq_q, dim], accum_dtype),  # type: ignore
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
            dQ_local = T.alloc_fragment([block_M, block_D], accum_dtype)
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
