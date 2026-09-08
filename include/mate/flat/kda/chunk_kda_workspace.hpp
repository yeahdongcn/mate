#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>

namespace mate::flat::kda {

// The prepare kernel writes this record layout and the recurrence kernel reads
// it.  Keep the layout in one place: changing one tile size or
// pointer order must not silently change the K1/K2 ABI.
template <class Element_, int Chunk_ = 16, int HeadDim_ = 128>
struct ChunkKdaWorkspace {
  using Element = Element_;

  static constexpr int kChunk   = Chunk_;
  static constexpr int kHeadDim = HeadDim_;

  static constexpr int64_t DecayedElements   = int64_t(2 * kChunk) * kHeadDim;
  static constexpr int64_t KRestoredElements = int64_t(kChunk) * kHeadDim;
  static constexpr int64_t InverseElements   = int64_t(kChunk) * kChunk;
  static constexpr int64_t PElements         = int64_t(kChunk) * kChunk;
  static constexpr int64_t ScaleElements     = kHeadDim;

  using DecayedLayout = decltype(mute::make_layout(
      mute::make_shape(mute::Int<2 * kChunk>{}, mute::Int<kHeadDim>{}, int32_t{}),
      mute::make_stride(mute::Int<kHeadDim>{}, mute::_1{}, mute::Int<2 * kChunk * kHeadDim>{})));
  using KRestoredLayout =
      decltype(mute::make_layout(mute::make_shape(mute::Int<kHeadDim>{}, mute::Int<kChunk>{}, int32_t{}),
                                 mute::make_stride(mute::_1{}, mute::Int<kHeadDim>{}, mute::Int<kChunk * kHeadDim>{})));
  using MatrixLayout =
      decltype(mute::make_layout(mute::make_shape(mute::Int<kChunk>{}, mute::Int<kChunk>{}, int32_t{}),
                                 mute::make_stride(mute::Int<kChunk>{}, mute::_1{}, mute::Int<kChunk * kChunk>{})));

  struct MutablePointers {
    Element* decayed;
    Element* k_restored;
    Element* inverse;
    Element* p;
    float*   total;
  };

  struct ConstPointers {
    Element const* decayed;
    Element const* k_restored;
    Element const* inverse;
    Element const* p;
    float const*   total;
  };

  template <bool IsVarlen, class ProblemShape>
  MUTLASS_HOST_DEVICE static int record_count(ProblemShape const& problem_shape) {
    if constexpr (IsVarlen) {
      return problem_shape.template max_chunks<kChunk>() + problem_shape.N;
    } else {
      return problem_shape.N * problem_shape.workspace_chunks;
    }
  }

  template <class Pointer>
  MUTLASS_HOST_DEVICE static auto make_decayed_tensor(Pointer pointer, int records) {
    return mute::make_tensor(
        mute::make_gmem_ptr(pointer),
        mute::make_layout(mute::make_shape(mute::Int<2 * kChunk>{}, mute::Int<kHeadDim>{}, records),
                          mute::make_stride(mute::Int<kHeadDim>{}, mute::_1{}, mute::Int<2 * kChunk * kHeadDim>{})));
  }

  template <class Pointer>
  MUTLASS_HOST_DEVICE static auto make_k_restored_tensor(Pointer pointer, int records) {
    return mute::make_tensor(
        mute::make_gmem_ptr(pointer),
        mute::make_layout(mute::make_shape(mute::Int<kHeadDim>{}, mute::Int<kChunk>{}, records),
                          mute::make_stride(mute::_1{}, mute::Int<kHeadDim>{}, mute::Int<kChunk * kHeadDim>{})));
  }

  template <class Pointer>
  MUTLASS_HOST_DEVICE static auto make_matrix_tensor(Pointer pointer, int records) {
    return mute::make_tensor(
        mute::make_gmem_ptr(pointer),
        mute::make_layout(mute::make_shape(mute::Int<kChunk>{}, mute::Int<kChunk>{}, records),
                          mute::make_stride(mute::Int<kChunk>{}, mute::_1{}, mute::Int<kChunk * kChunk>{})));
  }

  // Workspace records are allocated per value head (the second dimension of
  // the Python workspace tensors), so use the value-head index rather than
  // the grouped Q/K head index stored in WorkDesc::qk_head_idx.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_HOST_DEVICE static typename WorkDesc::IndexType record(ProblemSize const& problem_size,
                                                                 WorkDesc const&    work_desc,
                                                                 int                chunk_idx) {
    using IndexType = typename WorkDesc::IndexType;
    return (IndexType(work_desc.workspace_base) + IndexType(chunk_idx)) * IndexType(problem_size.H) +
           IndexType(work_desc.head_idx);
  }
};

}  // namespace mate::flat::kda
