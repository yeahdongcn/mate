#pragma once

#include <cstdint>
#include <mute/tensor.hpp>
#include <mutlass/kernel_hardware_info.hpp>
#include <type_traits>

#include "mate/flat/kda/chunk_kda_metadata.hpp"
#include "mate/flat/kda/chunk_kda_problem.hpp"

namespace mate::flat::kda {

namespace detail {

MUTLASS_HOST_DEVICE constexpr int grouped_head_idx(int head_idx, int dst_heads, int src_heads) {
  return head_idx / (dst_heads / src_heads);
}

}  // namespace detail

template <bool IsVarlen_, class LogicalShape_, class IndexType_ = int64_t>
struct ChunkKdaWorkDesc {
  using LogicalShape             = LogicalShape_;
  using IndexType                = IndexType_;
  static constexpr bool IsVarlen = IsVarlen_;
  static constexpr int  kChunk   = LogicalShape::Chunk;
  static constexpr int  kHeadDim = LogicalShape::HeadDim;

  IndexType bos;
  int       seq_idx;
  int       head_idx;
  int       qk_head_idx;
  IndexType seq_len;
  int       n_chunks;
  int       chunk_base;
  IndexType workspace_base;
  int       value_start;

  MUTLASS_DEVICE bool is_valid() const {
    return seq_idx >= 0 && head_idx >= 0;
  }

  template <class Scheduler>
  MUTLASS_DEVICE bool is_valid(Scheduler const& scheduler) const {
    return is_valid() && seq_idx < scheduler.num_seqs && head_idx < scheduler.num_heads;
  }

  MUTLASS_DEVICE int64_t chunk_start(int chunk_idx) const {
    return int64_t(chunk_base + chunk_idx) * kChunk;
  }

  MUTLASS_DEVICE int actual_len(int chunk_idx) const {
    return min(kChunk, int(seq_len - chunk_start(chunk_idx)));
  }

  MUTLASS_DEVICE bool is_sequence_final_chunk(int chunk_idx) const {
    return chunk_base + chunk_idx + 1 == mutlass::ceil_div(seq_len, int64_t(kChunk));
  }

  MUTLASS_DEVICE int64_t tme_token(int chunk_idx) const {
    if constexpr (IsVarlen) {
      return bos + chunk_start(chunk_idx);
    } else {
      return chunk_start(chunk_idx);
    }
  }

  MUTLASS_DEVICE int tme_batch() const {
    if constexpr (IsVarlen) {
      return 0;
    } else {
      return seq_idx;
    }
  }

  MUTLASS_DEVICE int64_t token(int chunk_idx, int row) const {
    return bos + chunk_start(chunk_idx) + row;
  }

  MUTLASS_DEVICE constexpr int value_head_dim() const {
    return kHeadDim;
  }
};

template <class LogicalShape_, class CuSeqlensElement_, class SchedulePolicy_, class... Options_>
struct ChunkKdaPrepareTileScheduler {
  using LogicalShape     = LogicalShape_;
  using CuSeqlensElement = CuSeqlensElement_;
  using SchedulePolicy   = SchedulePolicy_;
  using ProblemShape     = ChunkKdaPrepareProblemShape<CuSeqlensElement>;
  static constexpr bool IsVarlen =
      mate::flat::find_option_t<mate::flat::Tag::IsVarlen, std::false_type, Options_...>::value;
  using WorkDesc = ChunkKdaWorkDesc<IsVarlen, LogicalShape>;
  static_assert(std::is_integral_v<CuSeqlensElement>);

  static constexpr int kChunk = LogicalShape::Chunk;
  // Occupancy targets are architecture policy, while work partitioning is
  // shared by the MP31 and MP32 kernel shells.
  static constexpr int CtasPerMp = SchedulePolicy::PrepareCtasPerMp;

  struct Params {
    int                              num_seqs;
    int                              num_heads;
    int                              heads_per_qk;
    int                              max_chunks;
    int                              total_works;
    int                              num_partitions;
    ChunkKdaPartitionMetadata const* metadata;
  };

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size) {
    int max_chunks  = mutlass::ceil_div(problem_size.T, kChunk);
    int work_tiles  = IsVarlen ? max_chunks + problem_size.N : problem_size.N * max_chunks;
    int total_works = work_tiles * problem_size.H;
    return Params{problem_size.N,
                  problem_size.H,
                  problem_size.H / problem_size.Hqk,
                  max_chunks,
                  total_works,
                  problem_size.num_partitions,
                  problem_size.metadata};
  }

  static dim3 get_grid_shape(Params const& params, int mp_count) {
    int target_ctas = (mp_count > 0 ? mp_count : 1) * CtasPerMp;
    int grid_x      = params.num_partitions > 0 ? params.num_partitions
                                                : (params.total_works < target_ctas ? params.total_works : target_ctas);
    if (grid_x < 1) {
      grid_x = 1;
    }
    return dim3(static_cast<uint32_t>(grid_x), 1, 1);
  }

  static dim3 get_grid_shape(Params const& params) {
    return get_grid_shape(params, mutlass::KernelHardwareInfo::query_device_multiprocessor_count());
  }

  static MUTLASS_DEVICE WorkDesc invalid_work(int head_idx = 0) {
    return WorkDesc{0, -1, head_idx, 0, 0, 0, 0, 0, 0};
  }

  struct WorkCursor {
    int     seq_idx;
    int     chunk_idx;
    int     head_idx;
    int     remaining;
    int     qk_head_idx;
    int     qk_head_end;
    int64_t bos;
    int64_t seq_len;
    int64_t workspace_base;
  };

  template <class ProblemSize>
  MUTLASS_DEVICE void refresh_cursor_metadata(Params const&      params,
                                              ProblemSize const& problem_size,
                                              WorkCursor&        cursor) const {
    cursor.qk_head_idx = cursor.head_idx / params.heads_per_qk;
    cursor.qk_head_end = (cursor.qk_head_idx + 1) * params.heads_per_qk;
    if (cursor.seq_idx < 0 || cursor.seq_idx >= params.num_seqs) {
      return;
    }
    if constexpr (IsVarlen) {
      int64_t eos           = int64_t(problem_size.cu_seqlens[cursor.seq_idx + 1]);
      cursor.bos            = int64_t(problem_size.cu_seqlens[cursor.seq_idx]);
      cursor.seq_len        = eos - cursor.bos;
      cursor.workspace_base = mutlass::ceil_div(cursor.bos, int64_t(kChunk)) + cursor.seq_idx + cursor.chunk_idx;
    } else {
      cursor.bos            = int64_t(cursor.seq_idx) * problem_size.T;
      cursor.seq_len        = problem_size.T;
      cursor.workspace_base = int64_t(cursor.seq_idx) * problem_size.workspace_chunks + cursor.chunk_idx;
    }
  }

  template <class ProblemSize>
  MUTLASS_DEVICE void normalize_cursor(Params const&      params,
                                       ProblemSize const& problem_size,
                                       WorkCursor&        cursor) const {
    if constexpr (IsVarlen) {
      while (cursor.remaining > 0 && cursor.seq_idx < params.num_seqs) {
        int64_t bos    = int64_t(problem_size.cu_seqlens[cursor.seq_idx]);
        int64_t eos    = int64_t(problem_size.cu_seqlens[cursor.seq_idx + 1]);
        int     chunks = mutlass::ceil_div(eos - bos, int64_t(kChunk));
        if (cursor.chunk_idx < chunks) {
          break;
        }
        cursor.chunk_idx = 0;
        cursor.head_idx  = 0;
        ++cursor.seq_idx;
      }
    }
  }

  template <class ProblemSize>
  MUTLASS_DEVICE WorkCursor get_initial_work(Params const& params, ProblemSize const& problem_size) const {
    if constexpr (IsVarlen) {
      if (params.num_seqs == 1) {
        // A single cu-seqlens range needs no partition metadata.  Read its
        // exact length on device and distribute (chunk, head) work directly
        // across the launched persistent grid.
        int64_t    bos        = int64_t(problem_size.cu_seqlens[0]);
        int64_t    eos        = int64_t(problem_size.cu_seqlens[1]);
        int        total      = mutlass::ceil_div(eos - bos, int64_t(kChunk)) * params.num_heads;
        int        partitions = int(gridDim.x);
        int        base       = total / partitions;
        int        remainder  = total - base * partitions;
        int        partition  = int(blockIdx.x);
        int        begin      = partition * base + (partition < remainder ? partition : remainder);
        int        count      = base + int(partition < remainder);
        int        chunk_idx  = begin / params.num_heads;
        int        head_idx   = begin - chunk_idx * params.num_heads;
        WorkCursor cursor{0, chunk_idx, head_idx, count, 0, 0, bos, eos - bos, 0};
        refresh_cursor_metadata(params, problem_size, cursor);
        return cursor;
      }
      ChunkKdaPartitionMetadata meta = params.metadata[int(blockIdx.x)];
      WorkCursor cursor{meta.begin_seq, meta.begin_chunk, meta.begin_head, meta.work_count, 0, 0, 0, 0, 0};
      normalize_cursor(params, problem_size, cursor);
      refresh_cursor_metadata(params, problem_size, cursor);
      return cursor;
    } else {
      // Bind the persistent work split to the grid that was actually
      // launched.  This prevents a stale host-side partition count from
      // changing the work stride when the CTA target is tuned.
      int        partitions = int(gridDim.x);
      int        total      = params.total_works;
      int        base       = total / partitions;
      int        remainder  = total - base * partitions;
      int        partition  = int(blockIdx.x);
      int        begin      = partition * base + (partition < remainder ? partition : remainder);
      int        count      = base + int(partition < remainder);
      int        head_idx   = begin % params.num_heads;
      int        chunk_seq  = begin / params.num_heads;
      int        chunk_idx  = chunk_seq % params.max_chunks;
      int        seq_idx    = chunk_seq / params.max_chunks;
      WorkCursor cursor{seq_idx, chunk_idx, head_idx, count, 0, 0, 0, 0, 0};
      refresh_cursor_metadata(params, problem_size, cursor);
      return cursor;
    }
  }

  template <class ProblemSize>
  MUTLASS_DEVICE WorkDesc get_work(Params const&      params,
                                   ProblemSize const& problem_size,
                                   WorkCursor const&  cursor) const {
    if (cursor.remaining <= 0 || cursor.seq_idx < 0 || cursor.seq_idx >= params.num_seqs) {
      return invalid_work(cursor.head_idx);
    }
    return WorkDesc{cursor.bos,
                    cursor.seq_idx,
                    cursor.head_idx,
                    cursor.qk_head_idx,
                    cursor.seq_len,
                    1,
                    cursor.chunk_idx,
                    cursor.workspace_base,
                    0};
  }

  template <class ProblemSize>
  MUTLASS_DEVICE void advance(Params const& params, ProblemSize const& problem_size, WorkCursor& cursor) const {
    --cursor.remaining;
    if (cursor.remaining <= 0) {
      return;
    }
    if (++cursor.head_idx < params.num_heads) {
      if (cursor.head_idx == cursor.qk_head_end) {
        ++cursor.qk_head_idx;
        cursor.qk_head_end += params.heads_per_qk;
      }
      return;
    }
    cursor.head_idx    = 0;
    cursor.qk_head_idx = 0;
    cursor.qk_head_end = params.heads_per_qk;
    int chunks         = params.max_chunks;
    if constexpr (IsVarlen) {
      int64_t bos = int64_t(problem_size.cu_seqlens[cursor.seq_idx]);
      int64_t eos = int64_t(problem_size.cu_seqlens[cursor.seq_idx + 1]);
      chunks      = mutlass::ceil_div(eos - bos, int64_t(kChunk));
    }
    if (++cursor.chunk_idx < chunks) {
      ++cursor.workspace_base;
      return;
    }
    cursor.chunk_idx = 0;
    ++cursor.seq_idx;
    normalize_cursor(params, problem_size, cursor);
    refresh_cursor_metadata(params, problem_size, cursor);
  }
};

template <class LogicalShape_, class CuSeqlensElement_, class SchedulePolicy_, class... Options_>
struct ChunkKdaRecurrenceTileScheduler {
  using LogicalShape     = LogicalShape_;
  using CuSeqlensElement = CuSeqlensElement_;
  using SchedulePolicy   = SchedulePolicy_;
  using ProblemShape     = ChunkKdaProblemShape<CuSeqlensElement>;
  static constexpr bool IsVarlen =
      mate::flat::find_option_t<mate::flat::Tag::IsVarlen, std::false_type, Options_...>::value;
  using WorkIndex = std::conditional_t<IsVarlen, int64_t, int32_t>;
  using WorkDesc  = ChunkKdaWorkDesc<IsVarlen, LogicalShape, WorkIndex>;
  static_assert(std::is_integral_v<CuSeqlensElement>);

  static constexpr int kChunk      = LogicalShape::Chunk;
  static constexpr int kHeadDim    = LogicalShape::HeadDim;
  static constexpr int kValueDim   = LogicalShape::ValueDim;
  static constexpr int kValueTile  = SchedulePolicy::RecurrenceValueTile;
  static constexpr int kValueTiles = kValueDim / kValueTile;
  // The value slice and occupancy target are supplied by the architecture
  // policy; the recurrent work traversal itself is architecture-independent.
  static constexpr int CtasPerMp = SchedulePolicy::RecurrenceCtasPerMp;
  static_assert(kHeadDim == 128);
  static_assert(kValueDim % kValueTile == 0);

  struct Params {
    int num_seqs;
    int num_heads;
    int total_works;
  };

  int  seq_idx;
  int  head_idx;
  int  num_seqs;
  int  num_heads;
  int  work_idx;
  bool scheduled = false;

  MUTLASS_DEVICE explicit ChunkKdaRecurrenceTileScheduler(Params const& params)
      : seq_idx(0), head_idx(0), num_seqs(params.num_seqs), num_heads(params.num_heads), work_idx(int(blockIdx.x)) {
    if constexpr (IsVarlen) {
      seq_idx  = int(blockIdx.x);
      head_idx = int(blockIdx.y);
    }
  }

  template <class ProblemSize>
  static MUTLASS_HOST_DEVICE Params to_underlying_arguments(ProblemSize const& problem_size) {
    return Params{problem_size.N, problem_size.H, problem_size.N * problem_size.H * kValueTiles};
  }

  static dim3 get_grid_shape(Params const& params, int mp_count) {
    if constexpr (IsVarlen) {
      return dim3(params.num_seqs, params.num_heads, kValueTiles);
    } else {
      int target_ctas = (mp_count > 0 ? mp_count : 1) * CtasPerMp;
      int grid_x      = params.total_works < target_ctas ? params.total_works : target_ctas;
      return dim3(static_cast<uint32_t>(grid_x > 0 ? grid_x : 1), 1, 1);
    }
  }

  static dim3 get_grid_shape(Params const& params) {
    return get_grid_shape(params, mutlass::KernelHardwareInfo::query_device_multiprocessor_count());
  }

  template <class ProblemSize>
  MUTLASS_DEVICE WorkDesc get_next_work(ProblemSize const& problem_size) {
    int value_start;
    if constexpr (IsVarlen) {
      if (scheduled) {
        return WorkDesc{0, -1, head_idx, 0, 0, 0, 0, 0, 0};
      }
      scheduled   = true;
      value_start = int(blockIdx.z) * kValueTile;
    } else {
      int current_work = work_idx;
      if (current_work >= num_seqs * num_heads * kValueTiles) {
        return WorkDesc{0, -1, 0, 0, 0, 0, 0, 0, 0};
      }
      work_idx += int(gridDim.x);
      int head_seq = current_work / kValueTiles;
      value_start  = (current_work - head_seq * kValueTiles) * kValueTile;
      head_idx     = head_seq % num_heads;
      seq_idx      = head_seq / num_heads;
    }

    int64_t bos;
    int64_t eos;
    int64_t workspace_base;
    if constexpr (IsVarlen) {
      bos            = int64_t(problem_size.cu_seqlens[seq_idx]);
      eos            = int64_t(problem_size.cu_seqlens[seq_idx + 1]);
      workspace_base = mutlass::ceil_div(bos, int64_t(kChunk)) + seq_idx;
    } else {
      bos            = int64_t(seq_idx) * problem_size.T;
      eos            = bos + problem_size.T;
      workspace_base = int64_t(seq_idx) * problem_size.workspace_chunks;
    }
    int64_t seq_len = eos - bos;
    using IndexType = typename WorkDesc::IndexType;
    return WorkDesc{
        static_cast<IndexType>(bos),
        seq_idx,
        head_idx,
        detail::grouped_head_idx(head_idx, problem_size.H, problem_size.Hqk),
        static_cast<IndexType>(seq_len),
        mutlass::ceil_div(seq_len, int64_t(kChunk)),
        0,
        static_cast<IndexType>(workspace_base),
        value_start,
    };
  }
};

}  // namespace mate::flat::kda
