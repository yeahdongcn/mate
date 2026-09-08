#pragma once

#include <mutlass/mutlass.h>

#include <cstddef>
#include <cstdint>
#include <mute/tensor.hpp>

namespace mate::flat::kda {

struct ChunkKdaPartitionMetadata;

using ChunkKdaQkStride = mute::Stride<int64_t, mute::_1, mute::Stride<int64_t, int64_t>>;
using ChunkKdaVoStride = mute::Stride<mute::_1, int64_t, mute::Stride<int64_t, int64_t>>;

template <int Chunk_, int HeadDim_, int ValueDim_ = HeadDim_>
struct ChunkKdaLogicalShape {
  static constexpr int Chunk    = Chunk_;
  static constexpr int HeadDim  = HeadDim_;
  static constexpr int ValueDim = ValueDim_;

  static_assert(Chunk > 0);
  static_assert(HeadDim > 0);
  static_assert(ValueDim > 0);
};

template <class CuSeqlensElement_>
struct ChunkKdaProblemShape {
  using CuSeqlensElement = CuSeqlensElement_;

  int                     B;
  int                     T;
  int                     H;
  int                     Hqk;
  int                     N;
  CuSeqlensElement const* cu_seqlens;
  int                     workspace_chunks;

  template <int Chunk>
  MUTLASS_HOST_DEVICE int max_chunks() const {
    static_assert(Chunk > 0);
    return mutlass::ceil_div(T, Chunk);
  }
};

template <class CuSeqlensElement_>
struct ChunkKdaPrepareProblemShape : ChunkKdaProblemShape<CuSeqlensElement_> {
  int                              num_partitions = 0;
  ChunkKdaPartitionMetadata const* metadata       = nullptr;
};

template <int         PrepareCtasPerMp_,
          int         RecurrenceCtasPerMp_,
          int         RecurrenceValueTile_,
          int         MetadataThreads_,
          std::size_t MetadataSharedLimit_>
struct ChunkKdaPrefillSchedulePolicy {
  static constexpr int         PrepareCtasPerMp    = PrepareCtasPerMp_;
  static constexpr int         RecurrenceCtasPerMp = RecurrenceCtasPerMp_;
  static constexpr int         RecurrenceValueTile = RecurrenceValueTile_;
  static constexpr int         MetadataThreads     = MetadataThreads_;
  static constexpr std::size_t MetadataSharedLimit = MetadataSharedLimit_;

  static_assert(PrepareCtasPerMp > 0);
  static_assert(RecurrenceCtasPerMp > 0);
  static_assert(RecurrenceValueTile > 0);
  static_assert(MetadataThreads > 0);
  static_assert(MetadataSharedLimit > 0);
};

}  // namespace mate::flat::kda
