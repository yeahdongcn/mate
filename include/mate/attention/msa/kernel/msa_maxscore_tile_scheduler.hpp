#pragma once

#include <mutlass/fast_math.h>

#include <algorithm>
#include <cstdint>
#include <mute/tensor.hpp>

namespace mate::attention::msa {

template <class TileShape_, int HeadRatio_, bool ParallelKTiles_, bool IsVarlen_ = true, bool HasMetadata_ = false>
struct MsaMaxScoreTileScheduler {
  using TileShape = TileShape_;

  static constexpr int  TileQ          = mute::get<0>(TileShape{});
  static constexpr int  TileKV         = mute::get<1>(TileShape{});
  static constexpr int  HeadRatio      = HeadRatio_;
  static constexpr int  QTokensPerTile = TileQ / HeadRatio;
  static constexpr bool ParallelKTiles = ParallelKTiles_;
  static constexpr bool IsVarlen       = IsVarlen_;
  static constexpr bool HasMetadata    = HasMetadata_;
  static constexpr int  CtasPerMp      = ParallelKTiles ? 3 : 1;

  struct Arguments {
    int32_t const* ptr_cu_seqlens_q  = nullptr;
    int32_t const* ptr_cu_seqlens_k  = nullptr;
    int32_t const* ptr_schedule_meta = nullptr;  // [mp, {q_begin,q_len}]
    int            schedule_num_mps  = 0;
    int            uniform_q_len     = 0;
    int            uniform_k_len     = 0;
  };

  struct Params {
    int                 total_works;
    int                 total_q_works;
    int                 q_tiles;
    int                 k_tiles;
    int                 k_partitions;
    int                 num_kv_heads;
    int                 batch_size;
    int32_t const*      ptr_cu_seqlens_q;
    int32_t const*      ptr_cu_seqlens_k;
    int32_t const*      ptr_schedule_meta;
    int                 schedule_num_mps;
    int                 uniform_q_len;
    int                 uniform_k_len;
    mutlass::FastDivmod k_partitions_divmod;
    mutlass::FastDivmod q_tiles_divmod;
    mutlass::FastDivmod head_divmod;
  };

  struct WorkTileInfo {
    int work_idx;
    int q_work_idx;
    int batch_idx;
    int q_tile_idx;
    int k_partition_idx;
    int k_tile_idx;
    int head_kv;
    int q_batch_begin;
    int k_batch_begin;
    int q_len;
    int kv_len;
    int q_local_begin;
    int q_abs_begin;
    int q_count;
    int k_tile_begin;
    int valid_k_tiles;

    MUTLASS_DEVICE bool is_valid(Params const& params) const {
      return work_idx < params.total_works;
    }
  };

  // The metadata kernel assigns one contiguous range of logical Q-work items
  // to each MP slot.  Cache the decoded row once per CTA before entering the
  // persistent loop; reading the row from global memory on every iteration
  // would put the dispatch metadata back on the hot path.
  struct ScheduleInfo {
    int work_begin;
    int work_end;
    int work_stride;
    int active;
  };

  static MUTLASS_DEVICE mute::tuple<int, int, int> decode_q_work_coord(Params const& params, int q_work_idx) {
    int q_tile_idx = 0;
    int head_kv    = 0;
    int tile       = params.q_tiles_divmod.divmod(q_tile_idx, q_work_idx);
    int batch_idx  = params.head_divmod.divmod(head_kv, tile);
    return {q_tile_idx, head_kv, batch_idx};
  }

  static MUTLASS_DEVICE void fill_q_meta(Params const& params, WorkTileInfo& work_info) {
    if constexpr (IsVarlen) {
      work_info.q_batch_begin = params.ptr_cu_seqlens_q[work_info.batch_idx];
      int q_end               = params.ptr_cu_seqlens_q[work_info.batch_idx + 1];
      work_info.q_len         = q_end - work_info.q_batch_begin;
    } else {
      work_info.q_batch_begin = work_info.batch_idx * params.uniform_q_len;
      work_info.q_len         = params.uniform_q_len;
    }
    work_info.q_local_begin = work_info.q_tile_idx * QTokensPerTile;
    work_info.q_abs_begin   = work_info.q_batch_begin + work_info.q_local_begin;
    int q_count             = work_info.q_len - work_info.q_local_begin;
    q_count                 = q_count > 0 ? q_count : 0;
    work_info.q_count       = q_count < QTokensPerTile ? q_count : QTokensPerTile;
  }

  static MUTLASS_DEVICE void copy_q_meta(WorkTileInfo const& src, WorkTileInfo& dst) {
    dst.q_batch_begin = src.q_batch_begin;
    dst.q_len         = src.q_len;
    dst.q_local_begin = src.q_local_begin;
    dst.q_abs_begin   = src.q_abs_begin;
    dst.q_count       = src.q_count;
  }

  static MUTLASS_DEVICE void fill_k_meta(Params const& params, WorkTileInfo& work_info) {
    if constexpr (IsVarlen) {
      work_info.k_batch_begin = params.ptr_cu_seqlens_k[work_info.batch_idx];
      int k_end               = params.ptr_cu_seqlens_k[work_info.batch_idx + 1];
      work_info.kv_len        = k_end - work_info.k_batch_begin;
    } else {
      work_info.k_batch_begin = work_info.batch_idx * params.uniform_k_len;
      work_info.kv_len        = params.uniform_k_len;
    }
    work_info.valid_k_tiles = mutlass::ceil_div(work_info.kv_len, TileKV);
  }

  static MUTLASS_DEVICE void copy_k_meta(WorkTileInfo const& src, WorkTileInfo& dst) {
    dst.k_batch_begin = src.k_batch_begin;
    dst.kv_len        = src.kv_len;
    dst.valid_k_tiles = src.valid_k_tiles;
  }

  static MUTLASS_DEVICE WorkTileInfo make_q_work_tile(Params const& params, int work_idx) {
    WorkTileInfo work_info{};
    work_info.work_idx = work_idx;
    if (!work_info.is_valid(params)) {
      return work_info;
    }

    int q_work_idx      = work_idx;
    int k_partition_idx = 0;
    if constexpr (ParallelKTiles) {
      q_work_idx = params.k_partitions_divmod.divmod(k_partition_idx, work_idx);
    }
    auto [q_tile_idx, head_kv, batch_idx] = decode_q_work_coord(params, q_work_idx);
    work_info.q_work_idx                  = q_work_idx;
    work_info.batch_idx                   = batch_idx;
    work_info.q_tile_idx                  = q_tile_idx;
    work_info.k_partition_idx             = k_partition_idx;
    work_info.k_tile_idx                  = 0;
    work_info.head_kv                     = head_kv;
    work_info.k_tile_begin                = 0;
    fill_q_meta(params, work_info);
    fill_k_meta(params, work_info);
    return work_info;
  }

  static MUTLASS_DEVICE int schedule_mp_slot(Params const& params) {
    return static_cast<int>(blockIdx.x) / CtasPerMp;
  }

  static MUTLASS_DEVICE int schedule_cta_slot() {
    return static_cast<int>(blockIdx.x) % CtasPerMp;
  }

  static MUTLASS_DEVICE int schedule_partition_group(Params const& params, int mp_slot, int& row) {
    int partition_groups = (params.k_partitions + CtasPerMp - 1) / CtasPerMp;
    int rows_per_group   = params.schedule_num_mps / partition_groups;
    int extra_groups     = params.schedule_num_mps - rows_per_group * partition_groups;
    row                  = mp_slot;
    for (int group = 0; group < partition_groups; ++group) {
      int rows = rows_per_group + (group < extra_groups ? 1 : 0);
      if (row < rows) {
        return group;
      }
      row -= rows;
    }
    row = 0;
    return partition_groups;
  }

  static MUTLASS_DEVICE ScheduleInfo get_schedule_info(Params const& params) {
    ScheduleInfo info{};
    if constexpr (!HasMetadata) {
      info.work_begin  = static_cast<int>(blockIdx.x);
      info.work_end    = params.total_works;
      info.work_stride = static_cast<int>(gridDim.x);
      info.active      = info.work_begin < info.work_end;
      return info;
    } else {
      int mp_slot = schedule_mp_slot(params);
      if (mp_slot >= params.schedule_num_mps) {
        info.work_begin  = params.total_works;
        info.work_end    = params.total_works;
        info.work_stride = params.k_partitions;
        info.active      = 0;
        return info;
      }

      // Each row is {q_work_begin, q_work_length}; the partition group is
      // derived from this 1-D grid slot and is not stored in metadata.
      int row             = 0;
      int partition_group = schedule_partition_group(params, mp_slot, row);
      if (partition_group >= (params.k_partitions + CtasPerMp - 1) / CtasPerMp) {
        info.work_begin  = params.total_works;
        info.work_end    = params.total_works;
        info.work_stride = params.k_partitions;
        info.active      = 0;
        return info;
      }
      int q_begin      = params.ptr_schedule_meta[2 * mp_slot];
      int q_length     = params.ptr_schedule_meta[2 * mp_slot + 1];
      int partition    = partition_group * CtasPerMp + schedule_cta_slot();
      info.work_begin  = q_begin * params.k_partitions + partition;
      info.work_end    = (q_begin + q_length) * params.k_partitions;
      info.work_stride = params.k_partitions;
      info.active      = partition < params.k_partitions && q_length > 0;
      if (!info.active) {
        info.work_begin = params.total_works;
        info.work_end   = params.total_works;
      }
    }
    return info;
  }

  static MUTLASS_DEVICE WorkTileInfo make_next_q_work_tile(Params const& params, WorkTileInfo const& current_work) {
    WorkTileInfo work_info{};
    work_info.work_idx = current_work.work_idx + static_cast<int>(gridDim.x);
    if (!work_info.is_valid(params)) {
      return work_info;
    }

    int q_work_idx      = work_info.work_idx;
    int k_partition_idx = 0;
    if constexpr (ParallelKTiles) {
      q_work_idx = params.k_partitions_divmod.divmod(k_partition_idx, work_info.work_idx);
    }
    auto [q_tile_idx, head_kv, batch_idx] = decode_q_work_coord(params, q_work_idx);
    work_info.q_work_idx                  = q_work_idx;
    work_info.batch_idx                   = batch_idx;
    work_info.q_tile_idx                  = q_tile_idx;
    work_info.k_partition_idx             = k_partition_idx;
    work_info.k_tile_idx                  = 0;
    work_info.head_kv                     = head_kv;
    work_info.k_tile_begin                = 0;

    if (q_work_idx == current_work.q_work_idx) {
      copy_q_meta(current_work, work_info);
    } else {
      fill_q_meta(params, work_info);
    }
    if (batch_idx == current_work.batch_idx) {
      copy_k_meta(current_work, work_info);
    } else {
      fill_k_meta(params, work_info);
    }
    return work_info;
  }

  static MUTLASS_DEVICE WorkTileInfo make_next_q_work_tile(Params const&       params,
                                                           WorkTileInfo const& current_work,
                                                           ScheduleInfo const& schedule_info) {
    WorkTileInfo work_info{};
    work_info.work_idx = current_work.work_idx + schedule_info.work_stride;
    if (!schedule_info.active || work_info.work_idx >= schedule_info.work_end || !work_info.is_valid(params)) {
      return work_info;
    }

    // Metadata ranges advance by exactly k_partitions, so retain the decoded
    // partition and increment only the logical Q-work index.  Avoiding a
    // second divmod in the persistent loop keeps the range scheduler close to
    // the legacy next-work lowering.
    int q_work_idx                        = current_work.q_work_idx + 1;
    int k_partition_idx                   = current_work.k_partition_idx;
    auto [q_tile_idx, head_kv, batch_idx] = decode_q_work_coord(params, q_work_idx);
    work_info.q_work_idx                  = q_work_idx;
    work_info.batch_idx                   = batch_idx;
    work_info.q_tile_idx                  = q_tile_idx;
    work_info.k_partition_idx             = k_partition_idx;
    work_info.k_tile_idx                  = 0;
    work_info.head_kv                     = head_kv;
    work_info.k_tile_begin                = 0;

    if (q_work_idx == current_work.q_work_idx) {
      copy_q_meta(current_work, work_info);
    } else {
      fill_q_meta(params, work_info);
    }

    if (batch_idx == current_work.batch_idx) {
      copy_k_meta(current_work, work_info);
    } else {
      fill_k_meta(params, work_info);
    }
    return work_info;
  }

  static MUTLASS_DEVICE bool is_valid_q_tile(WorkTileInfo const& work_info) {
    return work_info.q_count > 0;
  }

  static MUTLASS_DEVICE bool is_valid_k_partition(WorkTileInfo const& work_info) {
    return work_info.k_partition_idx < work_info.valid_k_tiles;
  }

  static MUTLASS_DEVICE WorkTileInfo make_k_work_tile(WorkTileInfo const& q_work_info, int k_tile_idx) {
    WorkTileInfo work_info = q_work_info;
    work_info.k_tile_idx   = k_tile_idx;
    work_info.k_tile_begin = k_tile_idx * TileKV;
    return work_info;
  }

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args, int mp_count) {
    int uniform_q_len = args.uniform_q_len > 0 ? args.uniform_q_len : problem_size.max_seqlen_q;
    int uniform_k_len = args.uniform_k_len > 0 ? args.uniform_k_len : problem_size.max_seqlen_k;
    int q_tiles       = mutlass::ceil_div(problem_size.max_seqlen_q, QTokensPerTile);
    int k_tiles       = mutlass::ceil_div(problem_size.max_seqlen_k, TileKV);
    int total_q_works = q_tiles * problem_size.num_kv_heads * problem_size.batch_size;
    int k_partitions  = 1;
    if constexpr (ParallelKTiles) {
      int q_works      = std::max(total_q_works, 1);
      int target_works = std::max(mp_count, 1) * CtasPerMp;
      k_partitions     = std::min(k_tiles, std::max(1, (target_works + q_works / 2) / q_works));
    }
    int total_works = total_q_works * k_partitions;
    return {
        total_works,
        total_q_works,
        q_tiles,
        k_tiles,
        k_partitions,
        problem_size.num_kv_heads,
        problem_size.batch_size,
        args.ptr_cu_seqlens_q,
        args.ptr_cu_seqlens_k,
        args.ptr_schedule_meta,
        args.schedule_num_mps > 0 ? args.schedule_num_mps : mp_count,
        IsVarlen ? 0 : uniform_q_len,
        IsVarlen ? 0 : uniform_k_len,
        mutlass::FastDivmod(k_partitions),
        mutlass::FastDivmod(q_tiles),
        mutlass::FastDivmod(problem_size.num_kv_heads),
    };
  }

  static dim3 get_grid_shape(Params const& params, int mp_count) {
    int target_ctas = std::max(mp_count, 1) * CtasPerMp;
    // With per-MP schedule metadata, retain one fixed CTA group per MP slot;
    // each group advances through its assigned Q-work range. Without metadata
    // preserve the legacy persistent grid sizing.
    int grid_x = 0;
    if constexpr (HasMetadata) {
      grid_x = std::max(1, params.schedule_num_mps) * CtasPerMp;
    } else {
      grid_x = std::max(1, std::min(target_ctas, std::max(params.total_works, 1)));
    }
    return dim3(static_cast<uint32_t>(grid_x), 1, 1);
  }

  MUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params, ScheduleInfo const& schedule_info) const {
    if constexpr (!HasMetadata) {
      return make_q_work_tile(params, static_cast<int>(blockIdx.x));
    } else {
      if (!schedule_info.active || schedule_info.work_begin >= schedule_info.work_end) {
        WorkTileInfo invalid{};
        invalid.work_idx = params.total_works;
        return invalid;
      }
      return make_q_work_tile(params, schedule_info.work_begin);
    }
  }

  MUTLASS_DEVICE WorkTileInfo get_next_work(Params const&       params,
                                            WorkTileInfo const& current_work,
                                            ScheduleInfo const& schedule_info) const {
    if constexpr (!HasMetadata) {
      return make_next_q_work_tile(params, current_work);
    } else {
      return make_next_q_work_tile(params, current_work, schedule_info);
    }
  }
};

}  // namespace mate::attention::msa
