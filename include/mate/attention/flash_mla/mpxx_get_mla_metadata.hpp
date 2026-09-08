#pragma once

#include <mutlass/fast_math.h>
#include <mutlass/mutlass.h>

#include "mpxx_params.hpp"

namespace mate::flash_mla {

// Return the effective KV length used by the metadata scheduler for one batch
// item.  The regular kernel materializes these values in shared memory.  The
// fallback kernel below uses the same calculation while avoiding a large
// per-block shared-memory allocation.
__device__ __forceinline__ int metadata_sequence_length(const GetDecodingMetadataParams& params, int batch_idx) {
  int cur_s_k = 0;
  if (params.topk_length_ptr != nullptr) {
    cur_s_k = max(__ldg(params.topk_length_ptr + batch_idx), 0);
    if (params.topk != -1) {
      // The FlashInfer adapter reuses this pointer for raw sequence lengths;
      // cap scheduled work to the sparse index capacity.
      cur_s_k = min(cur_s_k, params.topk);
    }
    if (cur_s_k == 0) {
      cur_s_k = 1;
    }
    if (params.extra_topk != -1 || params.extra_topk_length_ptr != nullptr) {
      cur_s_k = mutlass::ceil_div(cur_s_k, params.block_size_n) * params.block_size_n;
      cur_s_k += params.extra_topk_length_ptr != nullptr ? max(__ldg(params.extra_topk_length_ptr + batch_idx), 0)
                                                         : params.extra_topk;
    }
  } else {
    cur_s_k = params.topk == -1 ? __ldg(params.seqlens_k_ptr + batch_idx) : params.topk;
    if (params.topk != -1 && (params.extra_topk != -1 || params.extra_topk_length_ptr != nullptr)) {
      cur_s_k = mutlass::ceil_div(max(cur_s_k, 1), params.block_size_n) * params.block_size_n;
      cur_s_k += params.extra_topk_length_ptr != nullptr ? max(__ldg(params.extra_topk_length_ptr + batch_idx), 0)
                                                         : params.extra_topk;
    }
  }
  return cur_s_k;
}

__device__ __forceinline__ int metadata_num_blocks(const GetDecodingMetadataParams& params, int batch_idx) {
  const int cur_s_k        = metadata_sequence_length(params, batch_idx);
  const int last_token_idx = max(cur_s_k - 1, 0);
  return last_token_idx / params.block_size_n + 1;
}

__global__ void __launch_bounds__(32, 1) get_mla_metadata_kernel(const GetDecodingMetadataParams params) {
  int* seqlens_k_ptr               = params.seqlens_k_ptr;
  int* tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
  int* num_splits_ptr              = params.num_splits_ptr;
  int  batch_size                  = params.batch_size;
  int  block_size_n                = params.block_size_n;
  int  fixed_overhead_num_blocks   = params.fixed_overhead_num_blocks;
  int  num_mp_parts                = params.num_mp_parts;

  extern __shared__ int shared_mem[];
  int*                  num_blocks_shared      = shared_mem;                       // [batch_size]
  int*                  num_splits_shared      = shared_mem + batch_size;          // [batch_size+1]
  int*                  seqlens_k_shared       = shared_mem + batch_size * 2 + 1;  // [batch_size]
  int*                  first_block_idx_shared = shared_mem + batch_size * 3 + 1;  // [batch_size]
  int*                  last_block_idx_shared  = shared_mem + batch_size * 4 + 1;  // [batch_size]

  int total_num_blocks = 0;
  for (int i = threadIdx.x; i < batch_size; i += 32) {
    int cur_s_k = 0;
    if (params.topk_length_ptr != nullptr) {
      cur_s_k = max(__ldg(params.topk_length_ptr + i), 0);
      if (params.topk != -1) {
        // The FlashInfer adapter reuses this pointer for raw sequence lengths;
        // cap scheduled work to the sparse index capacity.
        cur_s_k = min(cur_s_k, params.topk);
      }
      if (cur_s_k == 0) {
        cur_s_k = 1;
      }
      if (params.extra_topk != -1 || params.extra_topk_length_ptr != nullptr) {
        cur_s_k = mutlass::ceil_div(cur_s_k, block_size_n) * block_size_n;
        cur_s_k += params.extra_topk_length_ptr != nullptr ? max(__ldg(params.extra_topk_length_ptr + i), 0)
                                                           : params.extra_topk;
      }
    } else {
      cur_s_k = params.topk == -1 ? __ldg(seqlens_k_ptr + i) : params.topk;
      if (params.topk != -1 && (params.extra_topk != -1 || params.extra_topk_length_ptr != nullptr)) {
        cur_s_k = mutlass::ceil_div(max(cur_s_k, 1), block_size_n) * block_size_n;
        cur_s_k += params.extra_topk_length_ptr != nullptr ? max(__ldg(params.extra_topk_length_ptr + i), 0)
                                                           : params.extra_topk;
      }
    }
    seqlens_k_shared[i]     = cur_s_k;
    int first_token_idx     = 0;
    int last_token_idx      = max(cur_s_k - 1, 0);
    int cur_first_block_idx = first_token_idx / block_size_n;
    int cur_last_block_idx  = last_token_idx / block_size_n;
    // NOTE Should attend to tokens [first_token_idx, last_token_idx], i.e. blocks [cur_first_block_idx,
    // cur_last_block_idx] NOTE Before clamping, first_token_idx <= last_token_idx always holds, so after clamping,
    // first_token_idx <= last_token_idx still holds. NOTE if seqlens_k is 0, then first_token_idx == last_token_idx ==
    // cur_first_block_idx == cur_last_block_idx == 0. So the sequence will have 1 block. We will correct this later in
    // this kernel.
    int num_blocks = cur_last_block_idx - cur_first_block_idx + 1;
    total_num_blocks += num_blocks + fixed_overhead_num_blocks;
    num_blocks_shared[i]      = num_blocks;
    first_block_idx_shared[i] = cur_first_block_idx;
    last_block_idx_shared[i]  = cur_last_block_idx;
  }
  for (int offset = 16; offset >= 1; offset /= 2) {
    total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    int payload = mutlass::ceil_div(total_num_blocks, num_mp_parts) + fixed_overhead_num_blocks;

    int now_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
    num_splits_shared[0] = 0;
    for (int i = 0; i < num_mp_parts; ++i) {
      int tile_scheduler_metadata0[4], tile_scheduler_metadata1;
      tile_scheduler_metadata0[0] = now_idx;
      tile_scheduler_metadata0[1] = now_block + first_block_idx_shared[now_idx];
      tile_scheduler_metadata1    = now_n_split_idx;
      int remain_payload          = payload;
      while (now_idx < batch_size) {
        int num_blocks        = num_blocks_shared[now_idx];
        int now_remain_blocks = num_blocks - now_block;
        if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
          cum_num_splits += now_n_split_idx + 1;
          num_splits_shared[now_idx + 1] = cum_num_splits;
          remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
          ++now_idx;
          now_block       = 0;
          now_n_split_idx = 0;
        } else {
          if (remain_payload - fixed_overhead_num_blocks > 0) {
            now_block += remain_payload - fixed_overhead_num_blocks;
            ++now_n_split_idx;
            remain_payload = 0;
          }
          break;
        }
      }
      tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1;
      tile_scheduler_metadata0[3] =
          now_block > 0 ? now_block + first_block_idx_shared[now_idx]
                        : (seqlens_k_shared[now_idx - 1] == 0 ? 0 : last_block_idx_shared[now_idx - 1] + 1);
      *reinterpret_cast<int4*>(tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize) =
          *reinterpret_cast<int4*>(tile_scheduler_metadata0);
      tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
    }
    // TODO: device assert under DEBUG macro
    // FLASH_DEVICE_ASSERT(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
  }
  __syncwarp();

  for (int i = threadIdx.x; i <= batch_size; i += 32) {
    num_splits_ptr[i] = num_splits_shared[i];
  }
}

// For batches that do not fit the original five arrays, a single block-count
// array is sufficient for the scheduler.  It keeps the hot metadata path in
// shared memory while reducing the temporary footprint by 5x.
__global__ void __launch_bounds__(32, 1)
    get_mla_metadata_kernel_compact_fallback(const GetDecodingMetadataParams params) {
  const int batch_size                = params.batch_size;
  const int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
  const int num_mp_parts              = params.num_mp_parts;

  extern __shared__ int num_blocks_shared[];
  int                   total_num_blocks = 0;
  for (int i = threadIdx.x; i < batch_size; i += 32) {
    const int num_blocks = metadata_num_blocks(params, i);
    num_blocks_shared[i] = num_blocks;
    total_num_blocks += num_blocks + fixed_overhead_num_blocks;
  }
  for (int offset = 16; offset >= 1; offset /= 2) {
    total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
  }
  __syncwarp();

  if (threadIdx.x != 0) {
    return;
  }

  const int payload = mutlass::ceil_div(total_num_blocks, num_mp_parts) + fixed_overhead_num_blocks;
  int       now_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
  params.num_splits_ptr[0] = 0;
  for (int i = 0; i < num_mp_parts; ++i) {
    int tile_scheduler_metadata0[4], tile_scheduler_metadata1;
    tile_scheduler_metadata0[0] = now_idx;
    tile_scheduler_metadata0[1] = now_block;
    tile_scheduler_metadata1    = now_n_split_idx;
    int remain_payload          = payload;
    while (now_idx < batch_size) {
      const int num_blocks        = num_blocks_shared[now_idx];
      const int now_remain_blocks = num_blocks - now_block;
      if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
        cum_num_splits += now_n_split_idx + 1;
        params.num_splits_ptr[now_idx + 1] = cum_num_splits;
        remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
        ++now_idx;
        now_block       = 0;
        now_n_split_idx = 0;
      } else {
        if (remain_payload - fixed_overhead_num_blocks > 0) {
          now_block += remain_payload - fixed_overhead_num_blocks;
          ++now_n_split_idx;
          remain_payload = 0;
        }
        break;
      }
    }
    tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1;
    if (now_block > 0) {
      tile_scheduler_metadata0[3] = now_block;
    } else if (now_idx > 0) {
      const int previous_s_k      = metadata_sequence_length(params, now_idx - 1);
      tile_scheduler_metadata0[3] = previous_s_k == 0 ? 0 : max(previous_s_k - 1, 0) / params.block_size_n + 1;
    } else {
      tile_scheduler_metadata0[3] = 0;
    }
    *reinterpret_cast<int4*>(params.tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize) =
        *reinterpret_cast<int4*>(tile_scheduler_metadata0);
    params.tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
  }
}

// Extremely large batches can exceed even the compact block-count array.  The
// global-memory fallback below handles that rare case without another launch
// failure.
// It is only selected after both the original and compact shared-memory paths
// are unavailable.
__global__ void __launch_bounds__(32, 1) get_mla_metadata_kernel_fallback(const GetDecodingMetadataParams params) {
  const int batch_size                = params.batch_size;
  const int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
  const int num_mp_parts              = params.num_mp_parts;

  // Reuse num_splits[1:] as a temporary global-memory block-count buffer.
  // Each entry is overwritten with its final cumulative split count as soon
  // as the scheduler consumes it, so no additional allocation is required.
  int total_num_blocks = 0;
  for (int i = threadIdx.x; i < batch_size; i += 32) {
    const int num_blocks         = metadata_num_blocks(params, i);
    params.num_splits_ptr[i + 1] = num_blocks;
    total_num_blocks += num_blocks + fixed_overhead_num_blocks;
  }
  for (int offset = 16; offset >= 1; offset /= 2) {
    total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
  }
  __syncwarp();

  if (threadIdx.x != 0) {
    return;
  }

  const int payload         = mutlass::ceil_div(total_num_blocks, num_mp_parts) + fixed_overhead_num_blocks;
  int       now_idx         = 0;
  int       now_block       = 0;
  int       now_n_split_idx = 0;
  int       cum_num_splits  = 0;
  params.num_splits_ptr[0]  = 0;

  for (int i = 0; i < num_mp_parts; ++i) {
    int tile_scheduler_metadata0[4];
    tile_scheduler_metadata0[0]        = now_idx;
    tile_scheduler_metadata0[1]        = now_block;
    tile_scheduler_metadata0[2]        = 0;
    tile_scheduler_metadata0[3]        = 0;
    const int tile_scheduler_metadata1 = now_n_split_idx;
    int       remain_payload           = payload;

    while (now_idx < batch_size) {
      const int num_blocks        = params.num_splits_ptr[now_idx + 1];
      const int now_remain_blocks = num_blocks - now_block;
      if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
        cum_num_splits += now_n_split_idx + 1;
        params.num_splits_ptr[now_idx + 1] = cum_num_splits;
        remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
        ++now_idx;
        now_block       = 0;
        now_n_split_idx = 0;
      } else {
        if (remain_payload - fixed_overhead_num_blocks > 0) {
          now_block += remain_payload - fixed_overhead_num_blocks;
          ++now_n_split_idx;
          remain_payload = 0;
        }
        break;
      }
    }

    tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1;
    if (now_block > 0) {
      tile_scheduler_metadata0[3] = now_block;
    } else if (now_idx > 0) {
      const int previous_s_k      = metadata_sequence_length(params, now_idx - 1);
      tile_scheduler_metadata0[3] = previous_s_k == 0 ? 0 : max(previous_s_k - 1, 0) / params.block_size_n + 1;
    } else {
      tile_scheduler_metadata0[3] = 0;
    }

    *reinterpret_cast<int4*>(params.tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize) =
        *reinterpret_cast<int4*>(tile_scheduler_metadata0);
    params.tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
  }
}

}  // namespace mate::flash_mla
