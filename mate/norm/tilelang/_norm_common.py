import torch
import tilelang
import tilelang.language as T


def _fp8_torch_max(dtype: torch.dtype) -> float:
    if dtype == torch.float8_e5m2:
        return 57344.0
    if dtype == torch.float8_e4m3fn:
        return 448.0
    raise RuntimeError("output dtype must be float8_e4m3fn or float8_e5m2.")


def _tl_fp8_dtype(dtype: torch.dtype):
    if dtype == torch.float8_e4m3fn:
        return T.float8_e4m3fn
    if dtype == torch.float8_e5m2:
        tl_dtype = getattr(T, "float8_e5m2", None)
        if tl_dtype is None:
            raise RuntimeError("TileLang does not expose T.float8_e5m2.")
        return tl_dtype
    raise RuntimeError("output dtype must be float8_e4m3fn or float8_e5m2.")


def _ceil_pow2_expr(x):
    bits = T.reinterpret(x, T.uint32)
    exp = T.cast(((bits - 1) >> 23) + 1 - 127, T.int32)
    return T.reinterpret(T.cast((exp + 127) << 23, T.uint32), T.float32)


_JIT_CONFIG = {
    "pass_configs": {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
    "compile_flags": [
        "-O3",
        "-fno-signed-zeros",
        "-mllvm",
        "-mtgpu-if-convert=1",
        "-mllvm",
        "-misched=mtgpu-max-ilp",
        "-mllvm",
        "-mtgpu-tiny-offset-hint=1",
        "-mllvm",
        "-misched-recompute-slotindex=1",
    ],
}


def _make_sum_squares_macro(
    full_groups,
    tile_width,
    values_per_thread: int,
    tail_elems,
    hidden_size,
    is_3d: bool,
):
    @T.macro
    def sum_squares(x, x_local, sum_sq, row, batch_idx, seq_idx, lane):
        for group in T.serial(full_groups):
            base = group * tile_width + lane * values_per_thread
            for v in T.vectorized(values_per_thread):
                if is_3d:
                    x_local[v] = x[batch_idx, seq_idx, base + v]
                else:
                    x_local[v] = x[row, base + v]
            for v in T.unroll(values_per_thread):
                value = T.cast(x_local[v], T.float32)
                sum_sq[0] += value * value

        if tail_elems != 0:
            base = full_groups * tile_width + lane * values_per_thread
            for v in T.unroll(values_per_thread):
                col = base + v
                if col < hidden_size:
                    if is_3d:
                        value = T.cast(x[batch_idx, seq_idx, col], T.float32)
                    else:
                        value = T.cast(x[row, col], T.float32)
                    sum_sq[0] += value * value

    return sum_squares


def _make_row_sum_reduce_macro(
    num_warps_per_row: int,
    row_reduce_shuffles: int,
):
    @T.macro
    def row_sum_reduce(
        value,
        warp_sum_shared,
        row_in_block,
        warp_lane,
        warp_in_row,
    ):
        for offset in T.unroll(5):
            value[0] += T.shfl_xor(value[0], 16 >> offset)

        if num_warps_per_row == 1:
            if warp_lane == 0:
                warp_sum_shared[row_in_block, 0] = value[0]
        else:
            if warp_lane == 0:
                warp_sum_shared[row_in_block, warp_in_row] = value[0]
            T.sync_threads()
            if warp_in_row == 0:
                if warp_lane < num_warps_per_row:
                    value[0] = warp_sum_shared[row_in_block, warp_lane]
                else:
                    value[0] = 0.0
                for offset in T.unroll(row_reduce_shuffles):
                    value[0] += T.shfl_xor(value[0], (num_warps_per_row // 2) >> offset)
                if warp_lane == 0:
                    warp_sum_shared[row_in_block, 0] = value[0]
            T.sync_threads()

    return row_sum_reduce
