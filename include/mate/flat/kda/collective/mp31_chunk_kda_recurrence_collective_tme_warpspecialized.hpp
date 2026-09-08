#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <cstdint>
#include <limits>
#include <mute/arch/copy_mp31.hpp>
#include <mute/swizzle_layout.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/pipeline.hpp>
#include <type_traits>

#include "mate/common/layout.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/flat/collective/flat_collective_inverse_nxn.hpp"
#include "mate/flat/collective/flat_collective_load_tme.hpp"
#include "mate/flat/collective/flat_collective_mma_tme_sqmma.hpp"
#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/named_barrier.hpp"
#include "mate/flat/simd_helper.hpp"

#ifndef INLINE_LAMBDA
#define INLINE_LAMBDA __attribute__((always_inline))
#endif

namespace mate::flat::kda {

namespace recurrence {

using namespace mute;

template <bool UsePermute, class TiledMma, class PermuteTile>
struct MaybePermutedSqmma {
  using Type = TiledMma;
};

template <class TiledMma, class PermuteTile>
struct MaybePermutedSqmma<true, TiledMma, PermuteTile> {
  using Type = decltype(mate::convert_to_permuted_sqmma(TiledMma{}, PermuteTile{}));
};

template <class Components_, class TileShape_, class StateElement_, class StrideV_, class StrideO_>
struct Mp31ChunkKdaRecurrenceCollectiveTmeWarpSpecialized {
  using ArchTag                     = mutlass::arch::Mp31;
  using Components                  = Components_;
  using Element                     = typename Components::Element;
  using StateElement                = StateElement_;
  using TileShape                   = TileShape_;
  using StrideQK                    = ChunkKdaQkStride;
  using StrideV                     = StrideV_;
  using StrideO                     = StrideO_;
  static constexpr bool HasStateIn  = Components::HasStateIn;
  static constexpr bool HasStateOut = Components::HasStateOut;
  static constexpr bool IsVarlen    = Components::IsVarlen;

  static constexpr int kChunk    = Components::kChunk;
  static constexpr int kHeadDim  = Components::kHeadDim;
  static constexpr int kValueDim = Components::kRecurrenceValueTile;
  using Workspace                = typename Components::Workspace;
  static_assert(decltype(get<0>(TileShape{}))::value == kChunk);
  static_assert(decltype(get<1>(TileShape{}))::value == kChunk);
  static_assert(decltype(get<2>(TileShape{}))::value == kHeadDim);
  static_assert(kChunk % 2 == 0);
  // Use a 32-column SQMMA atom so the two state squads own disjoint halves
  // of the 64-column value tile.  With MaxInstructionN=64 the builder
  // replicates the complete C tile in both squads, wasting compute and
  // making state shared-memory stores racy.
  static constexpr int kStateMmaN  = kValueDim < 64 ? kValueDim : 32;
  static constexpr int kOutputMmaN = kValueDim < 64 ? kValueDim : 64;
  static_assert(decltype(get<0>(TileShape{}))::value == decltype(get<1>(TileShape{}))::value);
  static_assert(kHeadDim == 128);

  // K2 owns the recurrent traversal.  Keep the TME/SQMMA stage count tied to
  // the GEMM builder's stage policy so staged layouts cannot silently drift.
  using MmaStageCount                     = mutlass::gemm::collective::StageCount<2>;
  static constexpr int TmeStages          = MmaStageCount::value;
  static constexpr int NumProducerSquads  = 1;
  static constexpr int NumStateSquads     = 2;
  static constexpr int NumOutputSquads    = 1;
  static constexpr int NumProducerThreads = NumProducerSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumStateThreads    = NumStateSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumOutputThreads   = NumOutputSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumProducerWarps   = NumProducerSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumDataLoaderWarps = NumProducerWarps;
  static constexpr int NumStateWarps      = NumStateSquads * mutlass::NumWarpsPerWarpSquad;
  static_assert(Components::kValueDim == NumStateSquads * kValueDim);
  static constexpr int NumOutputWarps = NumOutputSquads * mutlass::NumWarpsPerWarpSquad;
  // The concatenated decayed tile is laid out as
  //   K[0:k/2], Q[0:k/2], K[k/2:k], Q[k/2:k].
  // With the singleton 128-thread DS atom, output warps 0/1 own all K rows
  // and output warps 2/3 own all Q rows.  This is the register ownership used
  // by the output squad when forming residual = V - K@State.
  static constexpr int NumKOutputWarpsPerSquad = mutlass::NumWarpsPerWarpSquad / 2;
  static_assert(mutlass::NumWarpsPerWarpSquad % 2 == 0);
  static constexpr int NumLogicalBarriers = static_cast<int>(RecurrenceBarrier::NumBarriers);
  static constexpr int NumStageBarriers   = TmeStages * NumLogicalBarriers;
  static constexpr int SmemAlignmentBytes = 256;
  static constexpr int VecSize            = 4;

  using TileShapeDS     = Shape<Int<2 * kChunk>, Int<kValueDim>, Int<kHeadDim>>;
  using TileShapeKU     = Shape<Int<kHeadDim>, Int<kValueDim>, Int<kChunk>>;
  using TileShapeQS     = Shape<Int<kChunk>, Int<kValueDim>, Int<kHeadDim>>;
  using TileShapePU     = Shape<Int<kChunk>, Int<kValueDim>, Int<kChunk>>;
  using TileShapeV      = Shape<Int<kChunk>, Int<kChunk>, Int<kValueDim>>;
  using StateAtomLayout = Layout<Shape<_1, Int<NumStateSquads>, _1>>;
  using RowAtomLayout   = Layout<Shape<_1, _1, _1>>;

  using CollectiveMmaDS = typename collective::Mp31TmeSqmmaCollective<Element,
                                                                      StrideQK,
                                                                      StrideQK,
                                                                      TileShapeDS,
                                                                      Int<2 * kChunk>,
                                                                      Int<kOutputMmaN>,
                                                                      RowAtomLayout,
                                                                      TmeStages>::CollectiveOp;

  using CollectiveMmaKU = typename collective::Mp31TmeSqmmaCollective<Element,
                                                                      StrideV,
                                                                      StrideV,
                                                                      TileShapeKU,
                                                                      Int<kHeadDim>,
                                                                      Int<kStateMmaN>,
                                                                      StateAtomLayout,
                                                                      TmeStages>::CollectiveOp;
  using CollectiveMmaQS = typename collective::Mp31TmeSqmmaCollective<Element,
                                                                      StrideQK,
                                                                      StrideV,
                                                                      TileShapeQS,
                                                                      Int<kChunk>,
                                                                      Int<kOutputMmaN>,
                                                                      RowAtomLayout,
                                                                      TmeStages>::CollectiveOp;
  using CollectiveMmaPU = typename collective::Mp31TmeSqmmaCollective<Element,
                                                                      StrideQK,
                                                                      StrideV,
                                                                      TileShapePU,
                                                                      Int<kChunk>,
                                                                      Int<kOutputMmaN>,
                                                                      RowAtomLayout,
                                                                      TmeStages>::CollectiveOp;

  using TiledMmaDS = typename CollectiveMmaDS::TiledMma;
  using TiledMmaKU = typename CollectiveMmaKU::TiledMma;
  using TiledMmaQS = typename CollectiveMmaQS::TiledMma;
  using TiledMmaPU = typename CollectiveMmaPU::TiledMma;
  static_assert(TmeStages == CollectiveMmaDS::DispatchPolicy::Stages);
  static_assert(decltype(size(TiledMmaDS{}))::value == NumOutputThreads);
  static_assert(decltype(size(TiledMmaKU{}))::value == NumStateThreads);
  static_assert(decltype(size(TiledMmaQS{}))::value == NumOutputThreads);
  static_assert(decltype(size(TiledMmaPU{}))::value == NumOutputThreads);

  using SmemLayoutDecayed =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaDS::SmemLayoutA{}, Int<TmeStages>{}));
  using SmemLayoutDecayedTile = decltype(take<0, 2>(SmemLayoutDecayed{}));
  using SmemLayoutKRestored =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaKU::SmemLayoutA{}, Int<TmeStages>{}));
  using SmemLayoutKRestoredTile = decltype(take<0, 2>(SmemLayoutKRestored{}));
  // Keep U in the output collective's B layout.  The state collective uses a
  // 32-column atom to split the two state squads, while PU/QS still require a
  // 64-column atom on MP31.  Logical tensor indexing is shared by both paths;
  // each MMA partitions the same U view according to its own operand layout.
  using SmemLayoutUState = typename CollectiveMmaKU::SmemLayoutB;
  using SmemLayoutU      = typename CollectiveMmaPU::SmemLayoutB;
  using SmemLayoutV      = decltype(mate::unstage_smem_layout(SmemLayoutUState{}, Int<TmeStages>{}));
  // The recurrent state is one live snapshot, not a double-buffered TME
  // operand.  StateCommitted also orders reuse of the U stage: the state
  // squad publishes the next state only after completing the prior U update.
  using SmemLayoutState    = decltype(take<0, 2>(typename CollectiveMmaDS::SmemLayoutB{}));
  using SmemLayoutInverseA = typename CollectiveMmaPU::SmemLayoutA;
  using SmemLayoutP        = SmemLayoutInverseA;
  using SmemLayoutPTile    = decltype(take<0, 2>(SmemLayoutP{}));

  static constexpr int R2SVectorBits    = kValueDim == 32 ? 16 : 128;
  static constexpr int R2SMmaAtomN      = 8;
  static constexpr int R2SFragmentSize  = R2SVectorBits / sizeof_bits_v<Element>;
  static constexpr int R2SGranularity   = R2SMmaAtomN * R2SFragmentSize;
  static constexpr int SmemTileNForR2S  = decltype(size<0>(SmemLayoutU{}))::value;
  static constexpr int R2SPermuteRepeat = kValueDim == 32 ? 1 : SmemTileNForR2S / R2SGranularity;
  static_assert(kValueDim == 32 || SmemTileNForR2S % R2SGranularity == 0);
  static constexpr int VPermuteMmaAtomN = 8;
  static constexpr int VPermuteRepeat   = kValueDim / VPermuteMmaAtomN;
  static_assert(kValueDim % VPermuteMmaAtomN == 0);
  static constexpr int VPermuteBlockCols  = VPermuteRepeat < VPermuteMmaAtomN ? VPermuteRepeat : VPermuteMmaAtomN;
  static constexpr int GmemVPermuteBlocks = VPermuteRepeat / VPermuteBlockCols;

  using R2SFragmentType   = mute::uint_bit_t<R2SVectorBits>;
  using UMmaPermuteTile   = decltype(filter(
      make_ordered_layout(Shape<Int<R2SMmaAtomN>, Int<R2SFragmentSize>, Int<R2SPermuteRepeat>>{}, Step<_2, _1, _3>{})));
  using PermuteTileForU   = Tile<Underscore, UMmaPermuteTile, Underscore>;
  using PermuteTiledMmaPU = typename MaybePermutedSqmma<kValueDim != 32, TiledMmaPU, PermuteTileForU>::Type;
  // SQMMA fragments enumerate V in 8x8-transposed blocks. Compose this
  // internal-to-natural mapping into gmem views so vectorized copies/casts stay unchanged.
  using PermutedInternalToNaturalVLayout = decltype(filter(make_ordered_layout(
      Shape<Int<VPermuteMmaAtomN>, Int<VPermuteBlockCols>, Int<GmemVPermuteBlocks>>{}, Step<_2, _1, _3>{})));
  using NaturalVLayout                   = Layout<Shape<Int<kValueDim>>, Stride<_1>>;
  using InternalToNaturalVLayout =
      std::conditional_t<kValueDim == 32, NaturalVLayout, PermutedInternalToNaturalVLayout>;
  using R2SCopyAtom        = Copy_Atom<UniversalCopy<R2SFragmentType>, Element>;
  using R2STiledCopy       = decltype(make_tiled_copy_C(R2SCopyAtom{}, PermuteTiledMmaPU{}));
  using StateGmemCopyAtom  = Copy_Atom<UniversalCopy<StateElement>, StateElement>;
  using StateGmemTiledCopy = decltype(make_tiled_copy_C(StateGmemCopyAtom{}, TiledMmaKU{}));

  // Q@State is produced and consumed entirely by the output squad before it
  // advances to the next chunk.  It therefore needs one bridge tile, not a
  // TME-staged allocation.
  using SmemLayoutQStateNatural =
      decltype(make_layout(make_shape(Int<kChunk>{}, Int<kValueDim>{}), make_stride(Int<kValueDim>{}, _1{})));
  using SmemLayoutQState = decltype(composition(SmemLayoutQStateNatural{}, make_tile(_, InternalToNaturalVLayout{})));

  static_assert(cosize_v<SmemLayoutDecayed> == 2 * kChunk * kHeadDim * TmeStages);
  static_assert(cosize_v<SmemLayoutQState> == kChunk * kValueDim);
  static_assert(cosize_v<SmemLayoutP> == kChunk * kChunk * TmeStages);
  static_assert(cosize_v<SmemLayoutState> >= kHeadDim * kValueDim);
  using TME_V = typename CollectiveMmaKU::Params::TME_B;

  using TMEWorkspaceDecayed   = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), typename Workspace::DecayedLayout{}),
      SmemLayoutDecayedTile{}));
  using TMEWorkspaceKRestored = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), typename Workspace::KRestoredLayout{}),
      SmemLayoutKRestoredTile{}));
  using TMEWorkspaceMatrix    = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), typename Workspace::MatrixLayout{}),
      SmemLayoutPTile{}));
  using PipelineQ             = mutlass::Mp31PipelineTmeAsync<TmeStages>;
  using PipelineK             = mutlass::Mp31PipelineTmeAsync<TmeStages>;
  using PipelineG             = mutlass::Mp31PipelineTmeAsync<TmeStages>;
  using PipelineV             = mutlass::Mp31PipelineTmeAsync<TmeStages>;
  using LoadV                 = collective::
      CollectiveLoadTme<collective::LoadKind::kV, PipelineV, Element, SmemLayoutV, TME_V, SmemAlignmentBytes>;
  using PipelineQParams = typename PipelineQ::Params;
  using PipelineKParams = typename PipelineK::Params;
  using PipelineGParams = typename PipelineG::Params;
  using PipelineVParams = typename PipelineV::Params;
  using PipelineQState  = typename PipelineQ::PipelineState;
  using PipelineKState  = typename PipelineK::PipelineState;
  using PipelineGState  = typename PipelineG::PipelineState;
  using PipelineVState  = typename PipelineV::PipelineState;

  template <class StorageElement, class SmemLayout>
  using SmemArray = array_aligned<StorageElement, cosize_v<SmemLayout>, SmemAlignmentBytes>;

  struct SharedStorage {
    SmemArray<Element, SmemLayoutDecayed>   smem_decayed;
    typename LoadV::SharedStorage           smem_v_raw;
    SmemArray<Element, SmemLayoutKRestored> smem_k_restored;
    // U is already a builder-staged GEMM operand (the stage is part of
    // CollectiveMmaKU::SmemLayoutB); keep one allocation and select the live
    // stage in the tensor view.
    SmemArray<Element, SmemLayoutU>        smem_u;
    SmemArray<Element, SmemLayoutInverseA> smem_inverse;
    SmemArray<Element, SmemLayoutP>        smem_p;
    SmemArray<float, SmemLayoutQState>     smem_q_state;
    SmemArray<Element, SmemLayoutState>    smem_state;
  };

  static constexpr int     SharedStorageSize  = sizeof(SharedStorage);
  static constexpr int64_t KTileBytes         = int64_t(kChunk) * kHeadDim * sizeof(Element);
  static constexpr int64_t VTileBytes         = int64_t(kChunk) * kValueDim * sizeof(Element);
  static constexpr int64_t KRestoredTileBytes = KTileBytes;
  static constexpr int64_t ChunkMatrixBytes   = int64_t(kChunk) * kChunk * sizeof(Element);

  struct Arguments {
    Element const*                    ptr_v;
    void const*                       ptr_initial_state;
    Element*                          ptr_out;
    void*                             ptr_final_state;
    int32_t const*                    ptr_state_indices;
    StrideV                           stride_v;
    StrideO                           stride_out;
    typename Workspace::ConstPointers workspace;
  };

  struct Params {
    Element const*        ptr_v;
    void const*           ptr_initial_state;
    Element*              ptr_out;
    void*                 ptr_final_state;
    int32_t const*        ptr_state_indices;
    StrideV               stride_v;
    StrideO               stride_out;
    TMEWorkspaceDecayed   tme_workspace_decayed;
    TMEWorkspaceKRestored tme_workspace_k_restored;
    TMEWorkspaceMatrix    tme_workspace_inverse;
    TMEWorkspaceMatrix    tme_workspace_p;
    float const*          ptr_workspace_total;
    TME_V                 tme_v;
  };

  struct alignas(1) BarrierStorage {
    uint8_t stage[NumStageBarriers];
    uint8_t pipeline_q[PipelineQ::NumBarriers];
    uint8_t pipeline_k[PipelineK::NumBarriers];
    uint8_t pipeline_g[PipelineG::NumBarriers];
    uint8_t pipeline_v[PipelineV::NumBarriers];
  };

  struct Pipeline {
    PipelineQ      q;
    PipelineK      k;
    PipelineG      g;
    PipelineV      v;
    PipelineQState q_read;
    PipelineQState q_write;
    PipelineKState k_read;
    PipelineKState k_write;
    PipelineGState g_read;
    PipelineGState g_write;
    PipelineVState v_read;
    PipelineVState v_write;

    static MUTLASS_DEVICE PipelineQParams make_q_params() {
      PipelineQParams params;
      params.transaction_bytes = uint32_t(2 * KTileBytes);
      params.num_consumers     = NumOutputWarps;
      params.num_producers     = 1;
      return params;
    }

    static MUTLASS_DEVICE PipelineKParams make_k_params() {
      PipelineKParams params;
      params.transaction_bytes = uint32_t(KRestoredTileBytes);
      params.num_consumers     = NumStateWarps;
      params.num_producers     = 1;
      return params;
    }

    static MUTLASS_DEVICE PipelineGParams make_g_params() {
      PipelineGParams params;
      params.transaction_bytes = uint32_t(2 * ChunkMatrixBytes);
      params.num_consumers     = NumOutputWarps;
      params.num_producers     = 1;
      return params;
    }

    static MUTLASS_DEVICE PipelineVParams make_v_params() {
      PipelineVParams params;
      params.transaction_bytes = uint32_t(VTileBytes);
      params.num_consumers     = NumOutputWarps;
      params.num_producers     = 1;
      return params;
    }

    MUTLASS_DEVICE explicit Pipeline(BarrierStorage* storage)
        : q(make_q_params(), reinterpret_cast<uint64_t>(&storage->pipeline_q), 1),
          k(make_k_params(), reinterpret_cast<uint64_t>(&storage->pipeline_k), 1),
          g(make_g_params(), reinterpret_cast<uint64_t>(&storage->pipeline_g), 1),
          v(make_v_params(), reinterpret_cast<uint64_t>(&storage->pipeline_v), 1),
          q_read{},
          q_write(mutlass::make_producer_start_state<PipelineQ>()),
          k_read{},
          k_write(mutlass::make_producer_start_state<PipelineK>()),
          g_read{},
          g_write(mutlass::make_producer_start_state<PipelineG>()),
          v_read{},
          v_write(mutlass::make_producer_start_state<PipelineV>()) {
    }
  };

  template <class Params_, class ProblemSize, class WorkDesc, class Pipeline_>
  MUTLASS_DEVICE static void issue_workspace_tme(Params_ const&     params,
                                                 ProblemSize const& problem_size,
                                                 WorkDesc const&    work_desc,
                                                 SharedStorage&     shared_storage,
                                                 Pipeline_&         pipeline,
                                                 int                chunk_idx,
                                                 int                producer_warp) {
    if (mutlass::canonical_lane_idx() != 0) {
      return;
    }

    int64_t record  = Workspace::record(problem_size, work_desc, chunk_idx);
    int32_t records = Components::workspace_records(problem_size);
    int32_t outer   = records * problem_size.H;

    if (producer_warp == 0) {
      pipeline.q.producer_acquire(pipeline.q_write);
      uint32_t barrier_id = pipeline.q.producer_get_barrier_id(pipeline.q_write);
      auto     g_decayed =
          params.tme_workspace_decayed.get_tme_tensor(make_shape(Int<2 * kChunk>{}, Int<kHeadDim>{}, outer));
      auto s_decayed = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{});
      auto cta_tme   = params.tme_workspace_decayed.get_slice(_0{});
      auto tDgD      = group_modes<0, 3>(cta_tme.partition_S(g_decayed));
      auto tDsD      = group_modes<0, 3>(cta_tme.partition_D(s_decayed));
      copy(params.tme_workspace_decayed.with(barrier_id), tDgD(_, int32_t(record)), tDsD(_, pipeline.q_write.index()));
      ++pipeline.q_write;
    } else if (producer_warp == 1) {
      pipeline.k.producer_acquire(pipeline.k_write);
      uint32_t barrier_id = pipeline.k.producer_get_barrier_id(pipeline.k_write);
      auto     g_kr = params.tme_workspace_k_restored.get_tme_tensor(make_shape(Int<kHeadDim>{}, Int<kChunk>{}, outer));
      auto     s_kr = make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestored{});
      auto     cta_tme_kr = params.tme_workspace_k_restored.get_slice(_0{});
      auto     tKRgKR     = group_modes<0, 3>(cta_tme_kr.partition_S(g_kr));
      auto     tKRsKR     = group_modes<0, 3>(cta_tme_kr.partition_D(s_kr));
      copy(params.tme_workspace_k_restored.with(barrier_id),
           tKRgKR(_, int32_t(record)),
           tKRsKR(_, pipeline.k_write.index()));
      ++pipeline.k_write;
    } else if (producer_warp == 3) {
      pipeline.g.producer_acquire(pipeline.g_write);
      uint32_t barrier_id = pipeline.g.producer_get_barrier_id(pipeline.g_write);
      auto     g_inverse = params.tme_workspace_inverse.get_tme_tensor(make_shape(Int<kChunk>{}, Int<kChunk>{}, outer));
      auto     g_p       = params.tme_workspace_p.get_tme_tensor(make_shape(Int<kChunk>{}, Int<kChunk>{}, outer));
      auto     s_inverse = make_tensor(make_smem_ptr(shared_storage.smem_inverse.data()), SmemLayoutInverseA{});
      auto     s_p       = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
      auto     cta_tme_inverse = params.tme_workspace_inverse.get_slice(_0{});
      auto     tIgI            = group_modes<0, 3>(cta_tme_inverse.partition_S(g_inverse));
      auto     tIsI            = group_modes<0, 3>(cta_tme_inverse.partition_D(s_inverse));
      auto     cta_tme_p       = params.tme_workspace_p.get_slice(_0{});
      auto     tPgP            = group_modes<0, 3>(cta_tme_p.partition_S(g_p));
      auto     tPsP            = group_modes<0, 3>(cta_tme_p.partition_D(s_p));
      copy(params.tme_workspace_inverse.with(barrier_id), tIgI(_, int32_t(record)), tIsI(_, pipeline.g_write.index()));
      copy(params.tme_workspace_p.with(barrier_id), tPgP(_, int32_t(record)), tPsP(_, pipeline.g_write.index()));
      ++pipeline.g_write;
    }
  }

  static MUTLASS_DEVICE constexpr uint32_t stage_barrier_id(RecurrenceBarrier barrier, int stage) {
    return uint32_t(stage * NumLogicalBarriers + static_cast<int>(barrier));
  }

  static MUTLASS_DEVICE constexpr bool is_k_decayed_row(int physical_row) {
    return (physical_row % kChunk) < (kChunk / 2);
  }

  static MUTLASS_DEVICE constexpr int k_state_row(int physical_row) {
    return (physical_row / kChunk) * (kChunk / 2) + (physical_row % kChunk);
  }

  static MUTLASS_DEVICE constexpr int q_state_row(int physical_row) {
    return (physical_row / kChunk) * (kChunk / 2) + (physical_row % kChunk) - (kChunk / 2);
  }

  static MUTLASS_DEVICE void stage_barrier_arrive(RecurrenceBarrier barrier, int stage) {
    named_barrier_arrive(stage_barrier_id(barrier, stage));
  }

  static MUTLASS_DEVICE void stage_barrier_wait(RecurrenceBarrier barrier, int stage, uint32_t phase) {
    named_barrier_wait(stage_barrier_id(barrier, stage), phase);
  }

  static MUTLASS_DEVICE void init_stage_barriers(BarrierStorage* barrier_storage, int tid) {
    (void)barrier_storage;
    static constexpr uint32_t StageBarrierArriveCount[NumLogicalBarriers] = {
        NumOutputWarps,  // UReady: output squad finishes inverse(residual)
        NumStateWarps,   // StateCommitted: state squads publish S_i
        NumOutputWarps,  // ResidualReady: output squad finishes V-K@State stores
    };
    if (tid == 0) {
      MUTLASS_PRAGMA_UNROLL
      for (int stage = 0; stage < TmeStages; ++stage) {
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NumLogicalBarriers; ++i) {
          named_barrier_init(stage * NumLogicalBarriers + i, StageBarrierArriveCount[i]);
        }
      }
    }
  }

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    auto params_v = CollectiveMmaKU::to_underlying_arguments(
        make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}, problem_size.T, make_shape(problem_size.H, problem_size.B)),
        typename CollectiveMmaKU::Arguments{
            args.ptr_v,
            args.stride_v,
            args.ptr_v,
            args.stride_v,
        },
        nullptr);
    int32_t records = Components::workspace_records(problem_size);
    int32_t outer   = records * problem_size.H;

    auto workspace_decayed = Workspace::make_decayed_tensor(args.workspace.decayed, outer);
    auto workspace_kr      = Workspace::make_k_restored_tensor(args.workspace.k_restored, outer);
    auto workspace_inverse = Workspace::make_matrix_tensor(args.workspace.inverse, outer);
    auto workspace_p       = Workspace::make_matrix_tensor(args.workspace.p, outer);
    return Params{
        args.ptr_v,
        args.ptr_initial_state,
        args.ptr_out,
        args.ptr_final_state,
        args.ptr_state_indices,
        args.stride_v,
        args.stride_out,
        make_tme_copy(MP31_TME_LOAD{}, workspace_decayed, SmemLayoutDecayedTile{}),
        make_tme_copy(MP31_TME_LOAD{}, workspace_kr, SmemLayoutKRestoredTile{}),
        make_tme_copy(MP31_TME_LOAD{}, workspace_inverse, SmemLayoutPTile{}),
        make_tme_copy(MP31_TME_LOAD{}, workspace_p, SmemLayoutPTile{}),
        args.workspace.total,
        params_v.tme_load_b,
    };
  }

  template <class TensorStateSmem, class TensorState>
  static MUTLASS_DEVICE void commit_state(TensorStateSmem& sState, TensorState const& rState, int local_tid) {
    static_assert(kHeadDim == 128 && kValueDim == 64 && kStateMmaN == 32);
    static_assert(decltype(size(rState))::value == 32);

    // A canonical 128x32 KU accumulator gives each lane four V values spaced
    // by eight for one K row.  The four lanes in the same lane%8 group own
    // four consecutive K rows.  Transpose that 4x4 register tile in-warp so
    // every lane can issue one contiguous 64-bit store along State's K axis.
    int const lane          = local_tid % mutlass::NumThreadsPerWarp;
    int const warp_in_squad = local_tid / mutlass::NumThreadsPerWarp % mutlass::NumWarpsPerWarpSquad;
    int const state_squad   = local_tid / mutlass::NumThreadsPerWarpSquad;
    int const value_col     = state_squad * kStateMmaN + lane;

    using FloatVec   = mutlass::Array<float, 4>;
    using ElementVec = mutlass::Array<Element, 4>;
    mutlass::NumericArrayConverter<Element, float, 4, mutlass::FloatRoundStyle::round_to_nearest> convert_state;

    MUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < kHeadDim / 16; ++k_block) {
      int const base = k_block * 4;
      float     x0   = rState(base + 0);
      float     x1   = rState(base + 1);
      float     x2   = rState(base + 2);
      float     x3   = rState(base + 3);

      bool const  group_bit0 = (lane & 8) != 0;
      float const recv01     = __shfl_xor_sync(uint32_t(-1), group_bit0 ? x0 : x1, 8);
      float const recv23     = __shfl_xor_sync(uint32_t(-1), group_bit0 ? x2 : x3, 8);
      if (group_bit0) {
        x0 = recv01;
        x2 = recv23;
      } else {
        x1 = recv01;
        x3 = recv23;
      }

      // After the XOR-8 stage, the XOR-16 stage only swaps the two BF16
      // pairs.  Pack each pair so one 32-bit shuffle handles both values.
      ElementVec     packed       = convert_state(FloatVec{x0, x1, x2, x3});
      uint32_t*      packed_words = reinterpret_cast<uint32_t*>(&packed);
      uint32_t       pair01       = packed_words[0];
      uint32_t       pair23       = packed_words[1];
      bool const     group_bit1   = (lane & 16) != 0;
      uint32_t const recv_pair    = __shfl_xor_sync(uint32_t(-1), group_bit1 ? pair01 : pair23, 16);
      if (group_bit1) {
        pair01 = recv_pair;
      } else {
        pair23 = recv_pair;
      }
      packed_words[0]                                                      = pair01;
      packed_words[1]                                                      = pair23;
      int const k_base                                                     = warp_in_squad * 4 + k_block * 16;
      *reinterpret_cast<uint64_t*>(&sState(make_coord(value_col, k_base))) = reinterpret_cast<uint64_t const&>(packed);
    }
  }

  template <class ProblemSize, class WorkDesc, class TensorState>
  static MUTLASS_DEVICE void kv_load(Params const&      params,
                                     ProblemSize const& problem_size,
                                     WorkDesc const&    work_desc,
                                     TensorState&       rState,
                                     int                local_tid) {
    if constexpr (HasStateIn) {
      if constexpr (kValueDim != kHeadDim) {
        TiledMmaKU tiled_mma_ku;
        auto       thr_mma_ku   = tiled_mma_ku.get_thread_slice(local_tid);
        auto       cState       = make_identity_tensor(make_shape(Int<kHeadDim>{}, Int<kValueDim>{}));
        auto       tCcState     = thr_mma_ku.partition_C(cState);
        int32_t    state_idx    = params.ptr_state_indices[work_desc.seq_idx];
        int64_t    state_offset = (int64_t(state_idx) * problem_size.H + work_desc.head_idx) * kHeadDim * kHeadDim +
                               int64_t(work_desc.value_start) * kHeadDim;
        auto state_ptr = static_cast<StateElement const*>(params.ptr_initial_state) + state_offset;
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(rState); ++i) {
          int row   = int(get<0>(tCcState(i)));
          int col   = int(InternalToNaturalVLayout{}(get<1>(tCcState(i))));
          rState(i) = static_cast<float>(state_ptr[col * kHeadDim + row]);
        }
        return;
      }
      StateGmemTiledCopy tiled_copy_state;
      auto               thr_copy_state = tiled_copy_state.get_thread_slice(local_tid);
      int32_t            state_idx      = params.ptr_state_indices[work_desc.seq_idx];
      int64_t state_offset = (int64_t(state_idx) * problem_size.H + work_desc.head_idx) * kHeadDim * kHeadDim +
                             int64_t(work_desc.value_start) * kHeadDim;
      auto gStateNatural =
          make_tensor(make_gmem_ptr(static_cast<StateElement const*>(params.ptr_initial_state) + state_offset),
                      make_shape(Int<kHeadDim>{}, Int<kValueDim>{}),
                      make_stride(_1{}, Int<kHeadDim>{}));
      auto   gState    = gStateNatural;
      Tensor tSgState  = thr_copy_state.partition_S(gState);
      Tensor rStateCvt = make_fragment_like<StateElement>(rState);
      Tensor tSrState  = thr_copy_state.retile_D(rStateCvt);

      copy(tiled_copy_state, tSgState, tSrState);

      using StateVec    = mutlass::Array<StateElement, VecSize>;
      using FloatVec    = mutlass::Array<float, VecSize>;
      auto rStateCvtVec = recast<StateVec>(rStateCvt);
      auto rStateVec    = recast<FloatVec>(rState);

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rStateVec); ++i) {
        rStateVec(i) = mutlass::NumericArrayConverter<float, StateElement, VecSize>{}(rStateCvtVec(i));
      }
    } else {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rState); ++i) {
        rState(i) = 0.0f;
      }
    }
  }

  template <class ProblemSize, class WorkDesc, class TensorState>
  static MUTLASS_DEVICE void kv_store(Params const&      params,
                                      ProblemSize const& problem_size,
                                      WorkDesc const&    work_desc,
                                      TensorState const& rState,
                                      int                local_tid) {
    if constexpr (HasStateOut) {
      if constexpr (kValueDim != kHeadDim) {
        TiledMmaKU tiled_mma_ku;
        auto       thr_mma_ku   = tiled_mma_ku.get_thread_slice(local_tid);
        auto       cState       = make_identity_tensor(make_shape(Int<kHeadDim>{}, Int<kValueDim>{}));
        auto       tCcState     = thr_mma_ku.partition_C(cState);
        int32_t    state_idx    = params.ptr_state_indices[work_desc.seq_idx];
        int64_t    state_offset = (int64_t(state_idx) * problem_size.H + work_desc.head_idx) * kHeadDim * kHeadDim +
                               int64_t(work_desc.value_start) * kHeadDim;
        auto state_ptr = static_cast<StateElement*>(params.ptr_final_state) + state_offset;
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(rState); ++i) {
          int row                         = int(get<0>(tCcState(i)));
          int col                         = int(InternalToNaturalVLayout{}(get<1>(tCcState(i))));
          state_ptr[col * kHeadDim + row] = static_cast<StateElement>(rState(i));
        }
        return;
      }
      StateGmemTiledCopy tiled_copy_state;
      auto               thr_copy_state = tiled_copy_state.get_thread_slice(local_tid);
      int32_t            state_idx      = params.ptr_state_indices[work_desc.seq_idx];
      int64_t state_offset = (int64_t(state_idx) * problem_size.H + work_desc.head_idx) * kHeadDim * kHeadDim +
                             int64_t(work_desc.value_start) * kHeadDim;
      auto gStateNatural = make_tensor(make_gmem_ptr(static_cast<StateElement*>(params.ptr_final_state) + state_offset),
                                       make_shape(Int<kHeadDim>{}, Int<kValueDim>{}),
                                       make_stride(_1{}, Int<kHeadDim>{}));
      auto gState        = gStateNatural;
      Tensor rStateCvt   = make_fragment_like<StateElement>(rState);
      Tensor tSrState    = thr_copy_state.retile_S(rStateCvt);
      Tensor tDgState    = thr_copy_state.partition_D(gState);
      using FloatVec     = mutlass::Array<float, VecSize>;
      using StateVec     = mutlass::Array<StateElement, VecSize>;
      auto rStateVec     = recast<FloatVec const>(rState);
      auto rStateCvtVec  = recast<StateVec>(rStateCvt);

      mutlass::NumericArrayConverter<StateElement, float, VecSize, mutlass::FloatRoundStyle::round_to_nearest>
          convert_state;
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rStateCvtVec); ++i) {
        rStateCvtVec(i) = convert_state(rStateVec(i));
      }
      copy(tiled_copy_state, tSrState, tDgState);
    }
  }

  template <class TensorState, class ProblemSize, class WorkDesc>
  static MUTLASS_DEVICE void scale_state(TensorState&       rState,
                                         Params const&      params,
                                         ProblemSize const& problem_size,
                                         WorkDesc const&    work_desc,
                                         int                chunk_idx,
                                         int                local_tid) {
    using IndexType     = typename WorkDesc::IndexType;
    IndexType    record = Workspace::record(problem_size, work_desc, chunk_idx);
    float const* total  = params.ptr_workspace_total + record * IndexType(kHeadDim);

    TiledMmaKU tiled_mma_ku;
    auto       thr_mma_ku = tiled_mma_ku.get_thread_slice(local_tid);
    auto       cState     = make_identity_tensor(make_shape(Int<kHeadDim>{}, Int<kValueDim>{}));
    auto       tCcState   = thr_mma_ku.partition_C(cState);
    using StateVec        = mutlass::Array<float, VecSize>;
    auto rStateVec        = recast<StateVec>(rState);

    MUTLASS_PRAGMA_UNROLL
    for (int vec_idx = 0; vec_idx < size(rStateVec); ++vec_idx) {
      int   col_k  = int(get<0>(tCcState(vec_idx * VecSize)));
      float scale  = total[col_k];
      auto& values = reinterpret_cast<float4&>(rStateVec(vec_idx));
      values       = simd::vmul(values, scale);
    }
  }

  static MUTLASS_DEVICE void inverse_residual(
      SharedStorage& shared_storage, int stage, int inverse_stage, int local_tid, uint32_t phase) {
    auto sInverse =
        make_tensor(make_smem_ptr(shared_storage.smem_inverse.data()), SmemLayoutInverseA{})(_, _, inverse_stage);
    auto sU   = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);
    auto sUMn = make_tensor(sU.data(), mate::select_layout<1, 0>(sU.layout()));

    TiledMmaPU tiled_mma_pu;
    auto       thr_mma_pu = tiled_mma_pu.get_thread_slice(local_tid);
    auto       acc_u      = partition_fragment_C(tiled_mma_pu, make_shape(Int<kChunk>{}, Int<kValueDim>{}));
    clear(acc_u);
    auto tSsInverse = thr_mma_pu.partition_A(sInverse);
    auto tSsU       = thr_mma_pu.partition_B(sU);
    auto tSrInverse = thr_mma_pu.make_fragment_A(tSsInverse);
    auto tSrU       = thr_mma_pu.make_fragment_B(tSsU);
    gemm(tiled_mma_pu, acc_u, tSrInverse, tSrU, acc_u);
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait();

    R2STiledCopy tiled_copy_r2s;
    auto         thr_copy_r2s = tiled_copy_r2s.get_thread_slice(local_tid);
    Tensor       tUsU         = thr_copy_r2s.partition_D(as_position_independent_swizzle_tensor(sUMn));
    Tensor       accum_cvt    = make_fragment_like<Element>(acc_u);
    Tensor       tUrU         = thr_copy_r2s.retile_S(accum_cvt);
    Tensor       tCvt_frg     = recast<mutlass::Array<Element, R2SFragmentSize>>(accum_cvt);
    Tensor       tAcc_frg     = recast<mutlass::Array<float, R2SFragmentSize>>(acc_u);

    MUTE_UNROLL
    for (int i = 0; i < size(tCvt_frg); ++i) {
      tCvt_frg(i) =
          mutlass::NumericArrayConverter<Element, float, R2SFragmentSize, mutlass::FloatRoundStyle::round_to_nearest>{}(
              tAcc_frg(i));
    }
    copy(tiled_copy_r2s, tUrU, tUsU);
    stage_barrier_arrive(RecurrenceBarrier::UReady, stage);
  }

  template <class TensorState>
  static MUTLASS_DEVICE void issue_state_update(
      TensorState& rState, SharedStorage& shared_storage, int stage, int k_stage, int local_tid, uint32_t phase) {
    auto sKRestored =
        make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestored{})(_, _, k_stage);
    auto sU = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);

    TiledMmaKU tiled_mma_ku;
    auto       thr_mma_ku = tiled_mma_ku.get_thread_slice(local_tid);

    auto tSsKRestored = thr_mma_ku.partition_A(sKRestored);
    auto tSsU         = thr_mma_ku.partition_B(sU);
    auto tSrKRestored = thr_mma_ku.make_fragment_A(tSsKRestored);
    auto tSrU         = thr_mma_ku.make_fragment_B(tSsU);

    stage_barrier_wait(RecurrenceBarrier::UReady, stage, phase);
    gemm(tiled_mma_ku, rState, tSrKRestored, tSrU, rState);
    mate::warpsquad_commit_batch();
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void load(SharedStorage&     shared_storage,
                                  Params const&      params,
                                  ProblemSize const& problem_size,
                                  WorkDesc const&    work_desc,
                                  Pipeline&          pipeline,
                                  int                local_tid) {
    if (work_desc.n_chunks <= 0) {
      return;
    }

    int producer_warp = local_tid / mutlass::NumThreadsPerWarp;
    MUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx) {
      if (producer_warp == 2) {
        LoadV load_v(params.tme_v, pipeline.v, shared_storage.smem_v_raw);
        auto  v_src_dst = load_v.partition_SD(problem_size, TileShapeV{}, work_desc);
        if (mutlass::canonical_lane_idx() == 0) {
          pipeline.v.producer_acquire(pipeline.v_write);
        }
        __syncwarp();
        load_v.template step<false>(v_src_dst, chunk_idx, pipeline.v_write);
      } else {
        issue_workspace_tme(params, problem_size, work_desc, shared_storage, pipeline, chunk_idx, producer_warp);
      }
    }
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void kda_recurrence_producer(SharedStorage&     shared_storage,
                                                     Params const&      params,
                                                     ProblemSize const& problem_size,
                                                     WorkDesc const&    work_desc,
                                                     Pipeline&          pipeline,
                                                     int                local_tid) {
    load(shared_storage, params, problem_size, work_desc, pipeline, local_tid);
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void kda_recurrence_state(SharedStorage&     shared_storage,
                                                  Params const&      params,
                                                  ProblemSize const& problem_size,
                                                  WorkDesc const&    work_desc,
                                                  Pipeline&          pipeline,
                                                  int                local_tid) {
    auto       sState = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutState{});
    TiledMmaKU tiled_mma_ku;
    // Keep the recurrent state fragment live across the chunk loop.  The
    // state squad owns only 32 V columns with the N=32 atom, so this costs
    // half the registers of the old replicated N=64 state MMA while avoiding
    // a scalar shared-memory round trip on every chunk.
    auto rState = partition_fragment_C(tiled_mma_ku, take<0, 2>(TileShapeKU{}));
    kv_load(params, problem_size, work_desc, rState, local_tid);

    MUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx) {
      // Publish exactly the snapshot consumed by this chunk.  Keeping the
      // publication at the loop head avoids a final-chunk predicate.
      commit_state(sState, rState, local_tid);
      stage_barrier_arrive(RecurrenceBarrier::StateCommitted, pipeline.k_read.index());
      // State only consumes K_restored and the U tile produced by output;
      // q/decayed and V are private to the output squad.
      // Reapply the chunk total while K_restored is still in flight.  This
      // work is independent of the K TME completion and hides its latency.
      scale_state(rState, params, problem_size, work_desc, chunk_idx, local_tid);
      pipeline.k.consumer_wait(pipeline.k_read);
      int      stage   = pipeline.k_read.index();
      int      k_stage = pipeline.k_read.index();
      uint32_t phase   = pipeline.k_read.phase();

      // Accumulate K_restored^T @ U.  issue_state_update waits for output's
      // UReady barrier, so the state snapshot above is no longer being read.
      issue_state_update(rState, shared_storage, stage, k_stage, local_tid, phase);
      mate::warpsquad_wait<0>();

      pipeline.k.consumer_release(pipeline.k_read);
      ++pipeline.k_read;
    }

    kv_store(params, problem_size, work_desc, rState, local_tid);
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void kda_recurrence_output(SharedStorage&     shared_storage,
                                                   Params const&      params,
                                                   ProblemSize const& problem_size,
                                                   WorkDesc const&    work_desc,
                                                   Pipeline&          pipeline,
                                                   int                local_tid) {
    MUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx) {
      pipeline.q.consumer_wait(pipeline.q_read);
      int      stage = pipeline.q_read.index();
      uint32_t phase = pipeline.q_read.phase();

      {
        // On stage reuse, the state squad reaches StateCommitted only after
        // finishing the previous K_restored^T@U update for this stage.
        stage_barrier_wait(RecurrenceBarrier::StateCommitted, stage, phase);

        auto sDecayed =
            make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{})(_, _, stage);
        auto sState = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutState{});

        // DS is an output-squad MMA.  Its K rows are consumed immediately for
        // the residual; its Q rows are scattered into the small QS-layout bridge
        // because DS and QS intentionally have different warp ownership.
        // For the singleton DS atom, physical_row = 4*warp + (lane>>3) +
        // 16*(slot>>3); hence the first two warps are K and the last two Q.
        TiledMmaDS tiled_mma_ds;
        auto       thr_mma_ds = tiled_mma_ds.get_thread_slice(local_tid);
        auto       tSsDecayed = thr_mma_ds.partition_A(sDecayed);
        auto       tSsState   = thr_mma_ds.partition_B(sState);
        auto       tSrDecayed = thr_mma_ds.make_fragment_A(tSsDecayed);
        auto       tSrState   = thr_mma_ds.make_fragment_B(tSsState);
        auto       acc_ds     = partition_fragment_C(tiled_mma_ds, take<0, 2>(TileShapeDS{}));
        clear(acc_ds);
        gemm(tiled_mma_ds, acc_ds, tSrDecayed, tSrState, acc_ds);
        mate::warpsquad_commit_batch();
        mate::warpsquad_wait<0>();

        // Decayed is dead as soon as the DS SQMMA completes.  Release its TME
        // stage before the residual/QState scatter so warp 0 can prefetch the
        // next chunk while output still consumes V and the register fragment.
        pipeline.q.consumer_release(pipeline.q_read);
        ++pipeline.q_read;

        // V is not consumed by Decayed@State.  Defer its wait until the DS
        // fragment is ready so producer warp 2 can overlap the V TME load with
        // both DS SQMMA instructions.
        pipeline.v.consumer_wait(pipeline.v_read);
        auto sU      = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);
        auto sQState = make_tensor(make_smem_ptr(shared_storage.smem_q_state.data()), SmemLayoutQState{});
        int  v_stage = pipeline.v_read.index();
        auto sV      = make_tensor(make_smem_ptr(shared_storage.smem_v_raw.data()), SmemLayoutV{})(_, _, v_stage);

        auto cDS           = make_identity_tensor(take<0, 2>(TileShapeDS{}));
        auto tDcD          = thr_mma_ds.partition_C(cDS);
        int  warp_in_squad = (local_tid / mutlass::NumThreadsPerWarp) % mutlass::NumWarpsPerWarpSquad;
        bool is_k_warp     = warp_in_squad < NumKOutputWarpsPerSquad;
        if (is_k_warp) {
          using AccumResidualVec = mutlass::Array<float, R2SFragmentSize>;
          using ElementVec       = mutlass::Array<Element, R2SFragmentSize>;
          auto acc_ds_vec        = recast<AccumResidualVec const>(acc_ds);
          MUTLASS_PRAGMA_UNROLL
          for (int vec_idx = 0; vec_idx < size(acc_ds_vec); ++vec_idx) {
            int              base            = vec_idx * R2SFragmentSize;
            int              physical_row    = int(get<0>(tDcD(base)));
            int              col             = int(get<1>(tDcD(base)));
            int              state_row       = (physical_row / kChunk) * (kChunk / 2) + (physical_row % (kChunk / 2));
            int              internal_col    = InternalToNaturalVLayout{}(col);
            ElementVec       v               = *reinterpret_cast<ElementVec const*>(&sV(internal_col, state_row));
            AccumResidualVec residual        = mutlass::NumericArrayConverter<float, Element, R2SFragmentSize>{}(v);
            auto*            residual_float4 = reinterpret_cast<float4*>(&residual);
            auto*            acc_ds_float4   = reinterpret_cast<float4 const*>(&acc_ds_vec(vec_idx));
            MUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < R2SFragmentSize / 4; ++i) {
              residual_float4[i] = simd::vsub(residual_float4[i], acc_ds_float4[i]);
            }
            ElementVec packed = mutlass::
                NumericArrayConverter<Element, float, R2SFragmentSize, mutlass::FloatRoundStyle::round_to_nearest>{}(
                    residual);
            *reinterpret_cast<ElementVec*>(&sU(internal_col, state_row)) = packed;
          }
        } else {
          using QStateVec = mutlass::Array<float, R2SFragmentSize>;
          auto acc_ds_vec = recast<QStateVec const>(acc_ds);
          MUTLASS_PRAGMA_UNROLL
          for (int vec_idx = 0; vec_idx < size(acc_ds_vec); ++vec_idx) {
            int base         = vec_idx * R2SFragmentSize;
            int physical_row = int(get<0>(tDcD(base)));
            int col          = int(get<1>(tDcD(base)));
            int q_row        = (physical_row / kChunk) * (kChunk / 2) + (physical_row % kChunk) - (kChunk / 2);
            *reinterpret_cast<QStateVec*>(&sQState(q_row, col)) = acc_ds_vec(vec_idx);
          }
        }
        pipeline.v.consumer_release(pipeline.v_read);
        ++pipeline.v_read;
        stage_barrier_arrive(RecurrenceBarrier::ResidualReady, stage);
        stage_barrier_wait(RecurrenceBarrier::ResidualReady, stage, phase);
      }

      {
        pipeline.g.consumer_wait(pipeline.g_read);
        int  g_stage = pipeline.g_read.index();
        auto sP      = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{})(_, _, g_stage);
        auto sU      = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);
        auto sQState = make_tensor(make_smem_ptr(shared_storage.smem_q_state.data()), SmemLayoutQState{});
        // Inverse(residual) remains in the output squad.  It overwrites the
        // residual in smem_u with U and signals the state squads to consume it.
        inverse_residual(shared_storage, stage, g_stage, local_tid, phase);
        stage_barrier_wait(RecurrenceBarrier::UReady, stage, phase);

        TiledMmaPU tiled_mma_pu;
        TiledMmaQS tiled_mma_qs;
        auto       thr_mma_pu = tiled_mma_pu.get_thread_slice(local_tid);
        auto       thr_mma_qs = tiled_mma_qs.get_thread_slice(local_tid);
        auto       tSsP       = thr_mma_pu.partition_A(sP);
        auto       tSsU       = thr_mma_pu.partition_B(sU);
        auto       tSrP       = thr_mma_pu.make_fragment_A(tSsP);
        auto       tSrU       = thr_mma_pu.make_fragment_B(tSsU);
        auto       acc_output = partition_fragment_C(tiled_mma_qs, take<0, 2>(TileShapeQS{}));
        clear(acc_output);
        gemm(tiled_mma_pu, acc_output, tSrP, tSrU, acc_output);
        mate::warpsquad_commit_batch();
        mate::warpsquad_wait<0>();
        pipeline.g.consumer_release(pipeline.g_read);
        ++pipeline.g_read;
        auto cO   = make_identity_tensor(take<0, 2>(TileShapeQS{}));
        auto tOcO = thr_mma_qs.partition_C(cO);
        static_assert(decltype(size(acc_output))::value == R2SFragmentSize);
        using QStateVec             = mutlass::Array<float, R2SFragmentSize>;
        auto      acc_output_vec    = recast<QStateVec>(acc_output);
        int       q_row             = int(get<0>(tOcO(0)));
        int       q_col             = int(get<1>(tOcO(0)));
        QStateVec q_state_vec       = *reinterpret_cast<QStateVec const*>(&sQState(q_row, q_col));
        auto*     acc_output_float4 = reinterpret_cast<float4*>(&acc_output_vec(0));
        auto*     q_state_float4    = reinterpret_cast<float4*>(&q_state_vec);
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < R2SFragmentSize / 4; ++i) {
          acc_output_float4[i] = simd::vadd(acc_output_float4[i], q_state_float4[i]);
        }

        int     actual_len = work_desc.actual_len(chunk_idx);
        int64_t token_base = work_desc.tme_token(chunk_idx);
        auto    out_outer  = get<2>(params.stride_out);
        static_assert(decltype(size(acc_output))::value == R2SFragmentSize);
        int row = int(get<0>(tOcO(0)));
        if (row < actual_len) {
          using AccumOutputVec         = mutlass::Array<float, R2SFragmentSize>;
          using ElementOutVec          = mutlass::Array<Element, R2SFragmentSize>;
          auto          acc_output_vec = recast<AccumOutputVec const>(acc_output);
          ElementOutVec packed         = mutlass::
              NumericArrayConverter<Element, float, R2SFragmentSize, mutlass::FloatRoundStyle::round_to_nearest>{}(
                  acc_output_vec(0));
          int     internal_col = int(get<1>(tOcO(0)));
          int     natural_col  = InternalToNaturalVLayout{}(internal_col) + work_desc.value_start;
          int64_t offset       = int64_t(natural_col) + (token_base + row) * int64_t(get<1>(params.stride_out)) +
                           int64_t(work_desc.head_idx) * int64_t(get<0>(out_outer)) +
                           int64_t(work_desc.tme_batch()) * int64_t(get<1>(out_outer));
          *reinterpret_cast<ElementOutVec*>(params.ptr_out + offset) = packed;
        }
      }
    }
  }
};

}  // namespace recurrence

}  // namespace mate::flat::kda
