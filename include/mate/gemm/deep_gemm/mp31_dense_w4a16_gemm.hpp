#pragma once

#include <musa_burst.h>
#include <mutlass/bfloat16.h>
#include <mutlass/half.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/algorithm/clear.hpp>
#include <mute/atom/copy_atom.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/gemm/deep_gemm/gemm_type.hpp"
#include "mate/gemm/deep_gemm/mp31_scheduler.hpp"

namespace mate::gemm::dense_w4a16_gemm {

using namespace mute;

// Stored weight nibbles are signed INT4 whose value is `unsigned_nibble - kWeightNibbleBias`, so the
// atom's signed INT4 product already carries the shared bias of every group. One signed INT4 operand
// therefore covers the symmetric case, where the dequant is exactly `scale * q` because the zero
// point equals the bias, and the asymmetric case, where the remaining offset
// `(kWeightNibbleBias - zero_point)` is a per-group rank-1 correction on the reduction of A.
inline constexpr int kWeightNibbleBias = 8;

// Act order (g_idx) permutes K, which smem load cannot express: a K tile is copied contiguously with
// no per-element index to hook. The permutation must be materialized into the packed B operand
// before launch; this type is the single device-side marker of that contract.
struct GidxKPermuteHook {
  static constexpr bool     IsIdentityOnly = true;
  MUTLASS_DEVICE static int apply(int k_index) {
    return k_index;
  }
};

// fp16 activations, signed INT4 weights, one fp16 scale per K128 weight group. Zero points are
// optional: without them the kernel is the symmetric fast path, which computes no correction.
template <class ElementA_,
          class ElementB_,
          class ElementScaleB_,
          class ElementZeroB_,
          class ElementRowSum_,
          class ElementD_,
          class TileShape_,
          uint32_t Stages_,
          bool     ZeroPoints_,
          class GidxPermute_ = GidxKPermuteHook>
struct Mp31DenseW4A16Gemm {
  using ElementA      = ElementA_;
  using ElementB      = ElementB_;
  using ElementScaleB = ElementScaleB_;
  using ElementZeroB  = ElementZeroB_;
  using ElementRowSum = ElementRowSum_;
  using ElementMmaA   = ElementA_;
  using ElementD      = ElementD_;
  using TileShape     = TileShape_;
  using GidxPermute   = GidxPermute_;

  static constexpr bool IsZeroPoints = ZeroPoints_;

  // Act order is a property of the packed operand, not of this kernel's indexing: the K extent of a
  // tile is copied contiguously, so a permutation has to be materialized before launch.
  static_assert(GidxPermute::IsIdentityOnly, "Act order (g_idx) must be applied to the packed weights");

  using StrideA      = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideB      = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideScaleB = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideZeroB  = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideRowSum = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideD      = decltype(make_stride(int64_t{}, _1{}, int64_t{}));

  static constexpr int BlockM             = size<0>(TileShape{});
  static constexpr int BlockN             = size<1>(TileShape{});
  static constexpr int BlockK             = size<2>(TileShape{});
  static constexpr int ScaleBlockK        = 128;
  static constexpr int ScaleBlocksPerTile = BlockK / ScaleBlockK;
  static constexpr int TmeScaleKBlocks    = 8;
  static constexpr int SqmmasPerScale     = 2;
  static constexpr int Stages             = int(Stages_);

  static_assert(BlockK == 256, "This kernel requires a K256 tile");
  static_assert(ScaleBlocksPerTile == 2, "Each K256 mainloop tile must contain two K128 weight-scale blocks");
  static_assert(TmeScaleKBlocks * int(sizeof(ElementScaleB)) >= 16,
                "Scale-B TME loads require at least 16 contiguous bytes");
  static_assert(TmeScaleKBlocks * int(sizeof(ElementZeroB)) >= 16,
                "Zero-B TME loads require at least 16 contiguous bytes");
  static_assert(Stages >= 2 && Stages <= 3, "The W4A16 K256 pipeline supports 2 or 3 stages");
  static_assert(is_same_v<ElementA, mutlass::half_t>, "The A operand must use FP16 elements end to end");
  static_assert(sizeof_bits_v<ElementMmaA> == 16, "The A operand must use FP16 elements end to end");
  static_assert(is_same_v<ElementB, int4_t>, "The B operand must use signed INT4 elements");
  static_assert(is_same_v<ElementScaleB, mutlass::half_t>, "B scales must use FP16 elements");
  static_assert(!IsZeroPoints || is_same_v<ElementZeroB, mutlass::half_t>,
                "Zero-point corrections must use FP16 elements");
  static_assert(is_same_v<ElementRowSum, float>, "The A reduction must accumulate in FP32");

  static constexpr TCE::Major SqmmaMajorA = TCE::Major::K;
  static constexpr TCE::Major SqmmaMajorB = TCE::Major::K;

  using SqmmaOp =
      decltype(MP31::SQMMA::
                   ss_op_selector<ElementMmaA, ElementB, float, TileShape, SqmmaMajorA, SqmmaMajorB, _32, _128>());
  using AtomLayout = decltype(make_layout(Shape<_1, _2, _1>{}, LayoutRight{}));
  using TiledMma   = decltype(make_tiled_mma(SqmmaOp{}, AtomLayout{}));

  using SmemLayoutAtomMmaA =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_A<SqmmaMajorA, ElementMmaA, SqmmaOp, TileShape>());
  using SmemLayoutAtomB =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_B<SqmmaMajorB, ElementB, SqmmaOp, TileShape>());
  using SmemLayoutMmaA =
      decltype(tile_to_shape(SmemLayoutAtomMmaA{}, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<Stages>{})));
  using SmemLayoutB =
      decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<BlockN>{}, Int<BlockK>{}, Int<Stages>{})));

  using SmemLayoutA =
      Layout<Shape<Int<BlockM>, Int<BlockK>, Int<Stages>>, Stride<Int<BlockK>, _1, Int<BlockM * BlockK>>>;

  static constexpr TME::CacheHint TmeAInnerHint      = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeAOuterHint      = TME::CacheHint::CACHE_PERSIST;
  static constexpr TME::CacheHint TmeBInnerHint      = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeBOuterHint      = TME::CacheHint::CACHE_NONE;
  static constexpr TME::CacheHint TmeScaleBInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeScaleBOuterHint = TME::CacheHint::CACHE_PERSIST;

  using TME_A = decltype(make_tme_copy<TmeAInnerHint, TmeAOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), repeat_like(StrideA{}, int64_t(0)), StrideA{}),
      take<0, 2>(SmemLayoutA{})));

  using TME_B = decltype(make_tme_copy<TmeBInnerHint, TmeBOuterHint, uint8_t>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr<ElementB>(nullptr), repeat_like(StrideB{}, int64_t(0)), StrideB{}),
      take<0, 2>(SmemLayoutB{}),
      make_shape(Int<BlockN>{}, Int<BlockK>{})));

  static constexpr int      SmemAlignmentBytes           = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor   = 1;
  static constexpr uint32_t Num1DBlocksPerSchedulerGroup = 4;

  static constexpr uint32_t NumProducerWarpSquads  = 1;
  static constexpr uint32_t NumCasterWarpSquads    = 1;
  static constexpr uint32_t NumConsumerWarpSquads  = 2;
  static constexpr uint32_t NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr uint32_t NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  static constexpr uint32_t WarpsPerWarpSquad      = NumThreadsPerWarpSquad / NumThreadsPerWarp;
  static constexpr uint32_t NumCasterThreads       = NumCasterWarpSquads * NumThreadsPerWarpSquad;
  static constexpr uint32_t MaxThreadsPerBlock =
      (NumProducerWarpSquads + NumCasterWarpSquads + NumConsumerWarpSquads) * NumThreadsPerWarpSquad;

  static_assert(int(NumConsumerWarpSquads * NumThreadsPerWarpSquad) == int(size(TiledMma{})),
                "The MMA thread layout must cover both consumer warp squads");

  static constexpr int MaxBarriersPerStage = 2;
  using PipelineBarrierRatio =
      mutlass::Mp31PipelineWarpSpecializedBarrierRatio<MaxBarriersPerStage, 0, Stages, Stages, Stages>;
  static constexpr int BarriersPerStage = PipelineBarrierRatio::value;
  static_assert(BarriersPerStage > 0, "A, B, cast, and Scale-B pipelines exceed the MP31 async-barrier budget");

  using PipelineLoadA       = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<Stages, BarriersPerStage>;
  using PipelineLoadB       = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<Stages, BarriersPerStage>;
  using PipelineCastA       = mutlass::Mp31PipelineAsyncWarpSpecialized<Stages, BarriersPerStage>;
  using PipelineLoadAParams = typename PipelineLoadA::Params;
  using PipelineLoadBParams = typename PipelineLoadB::Params;
  using PipelineCastAParams = typename PipelineCastA::Params;
  using PipelineLoadAState  = typename PipelineLoadA::PipelineState;
  using PipelineLoadBState  = typename PipelineLoadB::PipelineState;
  using PipelineCastAState  = typename PipelineCastA::PipelineState;

  static constexpr int TmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  static constexpr int TmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);
  static constexpr int MaxSharedStorageBytes = 192 * 1024;
  // Per-m operand of the scale step: the reduction of A over each K128 weight group, produced by the
  // A prepare stage, so it is staged on the cast pipeline rather than fetched from global memory.
  using SmemLayoutRowSum = Layout<Shape<Int<BlockM>, Int<ScaleBlocksPerTile>, Int<Stages>>,
                                  Stride<Int<ScaleBlocksPerTile>, _1, Int<BlockM * ScaleBlocksPerTile>>>;
  using SmemLayoutScaleB = Layout<Shape<Int<BlockN>, Int<TmeScaleKBlocks>, Int<Stages>>,
                                  Stride<Int<TmeScaleKBlocks>, _1, Int<BlockN * TmeScaleKBlocks>>>;
  using TME_ScaleB       = decltype(make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementScaleB const*>(nullptr)),
                  make_shape(int64_t{}, int64_t{}, int64_t{}),
                  make_stride(int64_t{}, _1{}, int64_t{})),
      take<0, 2>(SmemLayoutScaleB{}),
      make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{})));
  static constexpr int TmeTransactionBytesScaleB =
      mutlass::bits_to_bytes(BlockN * TmeScaleKBlocks * sizeof_bits_v<ElementScaleB>);
  using SmemLayoutZeroB = SmemLayoutScaleB;
  using TME_ZeroB       = decltype(make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementZeroB const*>(nullptr)),
                  make_shape(int64_t{}, int64_t{}, int64_t{}),
                  make_stride(int64_t{}, _1{}, int64_t{})),
      take<0, 2>(SmemLayoutZeroB{}),
      make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{})));
  static constexpr int TmeTransactionBytesZeroB =
      IsZeroPoints ? mutlass::bits_to_bytes(BlockN * TmeScaleKBlocks * sizeof_bits_v<ElementZeroB>) : 0;

  using Scheduler = mate::deep_gemm::detail::
      Mp31PersistentTileScheduler<mate::deep_gemm::GemmType::Normal, BlockM, BlockN, Num1DBlocksPerSchedulerGroup>;
  using WorkTileInfo = typename Scheduler::WorkTileInfo;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineLoadA[PipelineLoadA::NumBarriers];
    uint8_t PipelineLoadB[PipelineLoadB::NumBarriers];
    uint8_t PipelineCastA[PipelineCastA::NumBarriers];
  };
  static_assert(sizeof(BarrierStorage) + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                    mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "Async barrier storage exceeds the MP31 hardware limit");

  struct SharedStorage {
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementMmaA, cosize_v<SmemLayoutMmaA>> smem_mma_a;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementB, cosize_v<SmemLayoutB>> smem_b;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementRowSum, cosize_v<SmemLayoutRowSum>> smem_row_sum;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementScaleB, cosize_v<SmemLayoutScaleB>> smem_scale_b;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementZeroB, cosize_v<SmemLayoutZeroB>> smem_zero_b;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementA, cosize_v<SmemLayoutA>> smem_a;
  };

  struct Arguments {
    ElementA const*      ptr_a       = nullptr;
    ElementB const*      ptr_b       = nullptr;
    ElementScaleB const* ptr_scale_b = nullptr;
    // Per-(n, K128 group) `(kWeightNibbleBias - zero_point) * scale`, already folded into the
    // magnitude of the scale it corrects. Unused when IsZeroPoints is false.
    ElementZeroB const* ptr_zero_b = nullptr;
    ElementD*           ptr_d      = nullptr;

    StrideA      stride_a;
    StrideB      stride_b;
    StrideScaleB stride_scale_b;
    StrideZeroB  stride_zero_b;
    StrideD      stride_d;

    int m       = 0;
    int n       = 0;
    int k       = 0;
    int num_mps = 0;
  };

  struct Params {
    TME_A      tme_a;
    TME_B      tme_b;
    TME_ScaleB tme_scale_b;
    TME_ZeroB  tme_zero_b;
    ElementD*  ptr_d;

    StrideD stride_d;

    int m;
    int n;
    int k;
    int num_mps;
    int k_blocks;
    int scale_k_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    const int k_blocks       = ceil_div(args.k, BlockK);
    const int scale_k_blocks = k_blocks * ScaleBlocksPerTile;

    auto  gA    = make_tensor(make_gmem_ptr(args.ptr_a), make_shape(args.m, args.k, 1), args.stride_a);
    TME_A tme_a = make_tme_copy<TmeAInnerHint, TmeAOuterHint>(MP31_TME_LOAD{}, gA, take<0, 2>(SmemLayoutA{}));

    auto gB = make_tensor(
        make_gmem_ptr<ElementB>(static_cast<void const*>(args.ptr_b)), make_shape(args.n, args.k, 1), args.stride_b);
    TME_B tme_b = make_tme_copy<TmeBInnerHint, TmeBOuterHint, uint8_t>(
        MP31_TME_LOAD{}, gB, take<0, 2>(SmemLayoutB{}), make_shape(Int<BlockN>{}, Int<BlockK>{}));

    auto       scale_b_layout = make_layout(make_shape(args.n, scale_k_blocks, 1), args.stride_scale_b);
    auto       gScaleB        = make_tensor(make_gmem_ptr(args.ptr_scale_b), scale_b_layout);
    TME_ScaleB tme_scale_b    = make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
        MP31_TME_LOAD{}, gScaleB, take<0, 2>(SmemLayoutScaleB{}), make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}));

    auto      zero_b_layout = make_layout(make_shape(args.n, scale_k_blocks, 1), args.stride_zero_b);
    auto      gZeroB        = make_tensor(make_gmem_ptr(args.ptr_zero_b), zero_b_layout);
    TME_ZeroB tme_zero_b    = make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
        MP31_TME_LOAD{}, gZeroB, take<0, 2>(SmemLayoutZeroB{}), make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}));

    return Params{
        tme_a,
        tme_b,
        tme_scale_b,
        tme_zero_b,
        args.ptr_d,
        args.stride_d,
        args.m,
        args.n,
        args.k,
        args.num_mps,
        k_blocks,
        scale_k_blocks,
    };
  }

  static dim3 get_grid_shape_persistent(Params const& params) {
    const int num_mps = params.num_mps > 0 ? params.num_mps : 1;
    return dim3(static_cast<uint32_t>(num_mps), 1u, 1u);
  }

  MUTLASS_DEVICE
  static void load(Params const&       params,
                   SharedStorage&      shared_storage,
                   PipelineLoadA&      pipeline_load_a,
                   PipelineLoadB&      pipeline_load_b,
                   PipelineLoadAState& pipe_a_write,
                   PipelineLoadBState& pipe_b_write,
                   WorkTileInfo const& work_tile,
                   int                 warp_idx_in_squad) {
    const bool is_tme_issue_warp = warp_idx_in_squad == 0;

    Tensor sA      = make_tensor(make_smem_ptr(shared_storage.smem_a.begin()), SmemLayoutA{});
    Tensor sB      = make_tensor(make_smem_ptr(shared_storage.smem_b.begin()), SmemLayoutB{});
    Tensor sScaleB = make_tensor(make_smem_ptr(shared_storage.smem_scale_b.begin()), SmemLayoutScaleB{});
    Tensor sZeroB  = make_tensor(make_smem_ptr(shared_storage.smem_zero_b.begin()), SmemLayoutZeroB{});

    auto cta_tme_a       = params.tme_a.get_slice(0);
    auto cta_tme_b       = params.tme_b.get_slice(0);
    auto cta_tme_scale_b = params.tme_scale_b.get_slice(0);
    auto cta_tme_zero_b  = params.tme_zero_b.get_slice(0);

    const uint32_t m_block_idx = static_cast<uint32_t>(work_tile.M_idx);
    const uint32_t n_block_idx = static_cast<uint32_t>(work_tile.N_idx);

    auto mA  = params.tme_a.get_tme_tensor(make_shape(params.m, params.k, 1))(_, _, _0{});
    auto gA_ = local_tile(mA, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(_, _));

    auto mB  = params.tme_b.get_tme_tensor(make_shape(params.n, params.k, 1))(_, _, _0{});
    auto gB_ = local_tile(mB, make_shape(Int<BlockN>{}, Int<BlockK>{}), make_coord(_, _));

    Tensor gA = gA_(_, _, m_block_idx, _);
    Tensor gB = gB_(_, _, n_block_idx, _);

    Tensor tAgA = cta_tme_a.partition_S(gA);
    Tensor tAsA = cta_tme_a.partition_D(sA);
    Tensor tBgB = cta_tme_b.partition_S(gB);
    Tensor tBsB = cta_tme_b.partition_D(sB);

    auto   mScaleB   = params.tme_scale_b.get_tme_tensor(make_shape(params.n, params.scale_k_blocks, 1))(_, _, _0{});
    Tensor tSsScaleB = cta_tme_scale_b.partition_D(sScaleB);

    auto   mZeroB   = params.tme_zero_b.get_tme_tensor(make_shape(params.n, params.scale_k_blocks, 1))(_, _, _0{});
    Tensor tSsZeroB = cta_tme_zero_b.partition_D(sZeroB);

    MUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = 0; k_block < params.k_blocks; ++k_block) {
      if (is_tme_issue_warp) {
        pipeline_load_a.producer_acquire(pipe_a_write);
        const uint32_t barrier_id_a = pipeline_load_a.producer_get_barrier_id(pipe_a_write);
        copy(params.tme_a.with(barrier_id_a), tAgA(_, _, _, k_block), tAsA(_, _, _, int(pipe_a_write.index())));

        pipeline_load_b.producer_acquire(pipe_b_write);
        const uint32_t barrier_id_b = pipeline_load_b.producer_get_barrier_id(pipe_b_write);
        auto           mScaleBAtK   = domain_offset(make_coord(_0{}, k_block * ScaleBlocksPerTile), mScaleB);
        Tensor         gScaleB =
            local_tile(mScaleBAtK, make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}), make_coord(n_block_idx, _0{}));
        Tensor tSgScaleB = cta_tme_scale_b.partition_S(gScaleB);
        copy(params.tme_scale_b.with(barrier_id_b), tSgScaleB, tSsScaleB(_, _, _, int(pipe_b_write.index())));

        if constexpr (IsZeroPoints) {
          auto   mZeroBAtK = domain_offset(make_coord(_0{}, k_block * ScaleBlocksPerTile), mZeroB);
          Tensor gZeroB =
              local_tile(mZeroBAtK, make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}), make_coord(n_block_idx, _0{}));
          Tensor tSgZeroB = cta_tme_zero_b.partition_S(gZeroB);
          copy(params.tme_zero_b.with(barrier_id_b), tSgZeroB, tSsZeroB(_, _, _, int(pipe_b_write.index())));
        }

        copy(params.tme_b.with(barrier_id_b), tBgB(_, _, _, k_block), tBsB(_, _, _, int(pipe_b_write.index())));

        ++pipe_a_write;
        ++pipe_b_write;
      }
    }
  }

  // A is fp16 end to end, so this stage only restages the tile into the MMA smem layout and produces
  // the per-K128 reductions of A. The reductions ride the cast pipeline, so the consumer observes them
  // for the same stage as the tile they belong to.
  MUTLASS_DEVICE
  static void prepare_a(Params const&       params,
                        SharedStorage&      shared_storage,
                        PipelineLoadA&      pipeline_load_a,
                        PipelineCastA&      pipeline_cast_a,
                        PipelineLoadAState& pipe_a_read,
                        PipelineCastAState& pipe_cast_a_write,
                        int                 caster_thread_idx) {
    Tensor sA      = make_tensor(make_smem_ptr(shared_storage.smem_a.begin()), SmemLayoutA{});
    Tensor sMmaA   = make_tensor(make_smem_ptr(shared_storage.smem_mma_a.begin()), SmemLayoutMmaA{});
    Tensor sRowSum = make_tensor(make_smem_ptr(shared_storage.smem_row_sum.begin()), SmemLayoutRowSum{});

    using StoreVector               = uint128_t;
    constexpr int ElementsPerVector = sizeof(StoreVector) / sizeof(ElementA);
    constexpr int ThreadsPerRow     = int(NumCasterThreads) / BlockM;
    constexpr int VectorsPerRow     = BlockK / ElementsPerVector;
    constexpr int VectorsPerThread  = VectorsPerRow / ThreadsPerRow;
    constexpr int ElementsPerThread = VectorsPerThread * ElementsPerVector;
    constexpr int ThreadsPerGroup   = ScaleBlockK / ElementsPerThread;
    static_assert(is_same_v<ElementA, ElementMmaA>, "The A prepare stage performs no type conversion");
    static_assert(int(NumCasterThreads) % BlockM == 0);
    static_assert(VectorsPerRow % ThreadsPerRow == 0);
    static_assert(sizeof(StoreVector) / sizeof(ElementMmaA) == ElementsPerVector);
    // A thread run that straddles a K128 boundary would need two reductions in one accumulator.
    static_assert(ScaleBlockK < ElementsPerThread || ScaleBlockK % ElementsPerThread == 0);
    static_assert(ThreadsPerRow % ThreadsPerGroup == 0, "The threads of a row must tile the K128 groups");
    static_assert(ScaleBlocksPerTile == ThreadsPerRow / ThreadsPerGroup,
                  "The row reduction must produce exactly one value per K128 weight group");

    using StoreCopyAtom     = Copy_Atom<UniversalCopy<StoreVector>, ElementMmaA>;
    using StoreThreadLayout = Layout<Shape<Int<BlockM>, Int<ThreadsPerRow>>, Stride<Int<ThreadsPerRow>, _1>>;
    using StoreValueLayout  = Layout<Shape<_1, Int<ElementsPerThread>>>;
    using StoreTiledCopy    = decltype(make_tiled_copy(StoreCopyAtom{}, StoreThreadLayout{}, StoreValueLayout{}));

    const int m_idx               = caster_thread_idx / ThreadsPerRow;
    const int thread_idx_in_row   = caster_thread_idx % ThreadsPerRow;
    const int first_vector_in_row = thread_idx_in_row * VectorsPerThread;

    StoreTiledCopy tiled_copy_store;
    auto           thr_copy_store = tiled_copy_store.get_thread_slice(caster_thread_idx);

    Tensor rA                  = make_tensor<StoreVector>(Shape<Int<VectorsPerThread>>{});
    Tensor rAArray             = recast<ElementMmaA>(rA);
    auto   tCsMmaAStorePattern = thr_copy_store.partition_D(sMmaA(_, _, _0{}));
    auto   tCrMmaAStore        = make_fragment_like<ElementMmaA>(tCsMmaAStorePattern);
    Tensor rMmaAArray          = recast<StoreVector>(tCrMmaAStore);
    MUTE_STATIC_ASSERT_V(size(rMmaAArray) == Int<VectorsPerThread>{});

    MUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = 0; k_block < params.k_blocks; ++k_block) {
      pipeline_load_a.consumer_wait(pipe_a_read);

      Tensor sAVec = recast<StoreVector>(sA(_, _, int(pipe_a_read.index())));
      MUTE_UNROLL
      for (int vector_idx = 0; vector_idx < VectorsPerThread; ++vector_idx) {
        UniversalCopy<StoreVector>::copy(sAVec(m_idx, first_vector_in_row + vector_idx), rA(vector_idx));
      }

      __syncwarp();
      pipeline_load_a.consumer_release(pipe_a_read);
      ++pipe_a_read;

      MUTE_UNROLL
      for (int vector_idx = 0; vector_idx < VectorsPerThread; ++vector_idx) {
        rMmaAArray(vector_idx) = rA(vector_idx);
      }

      float row_sum = 0.0f;
      MUTE_UNROLL
      for (int element_idx = 0; element_idx < ElementsPerThread; ++element_idx) {
        row_sum += float(rAArray(element_idx));
      }
      MUTLASS_PRAGMA_UNROLL
      for (int offset = 1; offset < ThreadsPerGroup; offset <<= 1) {
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
      }

      pipeline_cast_a.producer_acquire(pipe_cast_a_write);

      const int stage        = int(pipe_cast_a_write.index());
      auto      tCsMmaAStore = thr_copy_store.partition_D(sMmaA(_, _, stage));
      copy(tiled_copy_store, tCrMmaAStore, tCsMmaAStore);

      if (thread_idx_in_row % ThreadsPerGroup == 0) {
        sRowSum(m_idx, thread_idx_in_row / ThreadsPerGroup, stage) = row_sum;
      }

      __syncwarp();
      pipeline_cast_a.producer_commit(pipe_cast_a_write);
      ++pipe_cast_a_write;
    }
  }

  template <class AccumTensor>
  MUTLASS_DEVICE static void epilogue(Params const&       params,
                                      AccumTensor const&  rAcc,
                                      WorkTileInfo const& work_tile,
                                      int                 consumer_thread_idx) {
    TiledMma tiled_mma;
    auto     thr_mma = tiled_mma.get_thread_slice(consumer_thread_idx);

    auto mD  = make_tensor(make_gmem_ptr(params.ptr_d), make_shape(params.m, params.n, 1), params.stride_d);
    auto gD_ = local_tile(mD, make_shape(Int<BlockM>{}, Int<BlockN>{}, _1{}), make_coord(_, _, _));
    auto gD  = gD_(_, _, _0{}, int(work_tile.M_idx), int(work_tile.N_idx), _0{});

    auto tCgD = thr_mma.partition_C(gD);
    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCgD));

    const uint32_t m0 = uint32_t(work_tile.M_idx) * uint32_t(BlockM);
    const uint32_t n0 = uint32_t(work_tile.N_idx) * uint32_t(BlockN);
    if (m0 + uint32_t(BlockM) <= uint32_t(params.m) && n0 + uint32_t(BlockN) <= uint32_t(params.n)) {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rAcc); ++i) {
        tCgD(i) = ElementD(rAcc(i));
      }
    } else {
      auto cD   = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
      auto tCcD = thr_mma.partition_C(cD);

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rAcc); ++i) {
        const auto     coord   = tCcD(i);
        const uint32_t coord_m = m0 + uint32_t(get<0>(coord));
        const uint32_t coord_n = n0 + uint32_t(get<1>(coord));
        if (coord_m < uint32_t(params.m) && coord_n < uint32_t(params.n)) {
          tCgD(i) = ElementD(rAcc(i));
        }
      }
    }
  }

  // One packed word carries a 16-bit group value for each K128 weight block of the K256 tile: the low
  // half is the first block, the high half the second.
  template <class ElementPacked, int ScaleIndex, class PackedTensor>
  MUTLASS_DEVICE static float4 unpack_packed_bst4(PackedTensor const& packed, int i) {
    constexpr int Shift = ScaleIndex * 16;
    return make_float4(float(ElementPacked::bitcast(uint16_t(packed(i + 0) >> Shift))),
                       float(ElementPacked::bitcast(uint16_t(packed(i + 1) >> Shift))),
                       float(ElementPacked::bitcast(uint16_t(packed(i + 2) >> Shift))),
                       float(ElementPacked::bitcast(uint16_t(packed(i + 3) >> Shift))));
  }

  template <int ScaleIndex, class ScaleTensor, class PackedScaleBTensor>
  MUTLASS_DEVICE static void unpack_scale_bst4(ScaleTensor& scale, PackedScaleBTensor const& scale_b_packed) {
    static_assert(ScaleIndex == 0 || ScaleIndex == 1);
    static_assert(size(ScaleTensor{}) % 4 == 0, "BST4 scaling requires groups of four accumulators");

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(scale); i += 4) {
      const float4 scale_vec = unpack_packed_bst4<ElementScaleB, ScaleIndex>(scale_b_packed, i);
      scale(i + 0)           = scale_vec.x;
      scale(i + 1)           = scale_vec.y;
      scale(i + 2)           = scale_vec.z;
      scale(i + 3)           = scale_vec.w;
    }
  }

  // Rank-1 term of the asymmetric dequant, `(kWeightNibbleBias - zero_point) * reduce_k(a)`, for the
  // K128 group selected by ScaleIndex. Adding it to the scaled accumulator of every output column is
  // exact, because the atom already produced the signed INT4 product of the same group.
  template <int ScaleIndex, class CorrTensor, class RowSumTensor, class PackedZeroBTensor>
  MUTLASS_DEVICE static void calculate_correction_bst4(CorrTensor&              corr,
                                                       RowSumTensor const&      row_sum,
                                                       PackedZeroBTensor const& zero_b_packed) {
    static_assert(ScaleIndex == 0 || ScaleIndex == 1);
    static_assert(size(CorrTensor{}) % 4 == 0, "BST4 scaling requires groups of four accumulators");

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(corr); i += 4) {
      const float4 zero_vec = unpack_packed_bst4<ElementZeroB, ScaleIndex>(zero_b_packed, i);
      corr(i + 0)           = row_sum(i + 0) * zero_vec.x;
      corr(i + 1)           = row_sum(i + 1) * zero_vec.y;
      corr(i + 2)           = row_sum(i + 2) * zero_vec.z;
      corr(i + 3)           = row_sum(i + 3) * zero_vec.w;
    }
  }

  template <class AccumTensor, class ScaleTensor, class CorrTensor>
  MUTLASS_DEVICE static void scale_accumulate_bst4(AccumTensor&       accum,
                                                   AccumTensor const& accum_temp,
                                                   ScaleTensor const& scale,
                                                   CorrTensor const&  correction) {
    static_assert(size(AccumTensor{}) % 4 == 0, "BST4 scaling requires groups of four accumulators");

    auto accum_bst4      = recast<f4>(accum);
    auto accum_temp_bst4 = recast<f4>(accum_temp);
    auto scale_bst4      = recast<f4>(scale);
    auto corr_bst4       = recast<f4>(correction);

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_bst4); ++i) {
      if constexpr (IsZeroPoints) {
        accum_bst4(i) += accum_temp_bst4(i) * scale_bst4(i) + corr_bst4(i);
      } else {
        accum_bst4(i) += accum_temp_bst4(i) * scale_bst4(i);
      }
    }
  }

  MUTLASS_DEVICE
  static auto compute(Params const&       params,
                      SharedStorage&      shared_storage,
                      PipelineLoadB&      pipeline_load_b,
                      PipelineCastA&      pipeline_cast_a,
                      PipelineLoadBState& pipe_b_read,
                      PipelineLoadBState& pipe_b_prefetch,
                      PipelineCastAState& pipe_cast_a_read,
                      WorkTileInfo const& work_tile,
                      int                 consumer_thread_idx) {
    TiledMma tiled_mma;
    auto     thr_mma = tiled_mma.get_thread_slice(consumer_thread_idx);

    Tensor sMmaA   = make_tensor(make_smem_ptr(shared_storage.smem_mma_a.begin()), SmemLayoutMmaA{});
    Tensor sB      = make_tensor(make_smem_ptr(shared_storage.smem_b.begin()), SmemLayoutB{});
    Tensor sRowSum = make_tensor(make_smem_ptr(shared_storage.smem_row_sum.begin()), SmemLayoutRowSum{});
    Tensor sScaleB = make_tensor(make_smem_ptr(shared_storage.smem_scale_b.begin()), SmemLayoutScaleB{});
    Tensor sZeroB  = make_tensor(make_smem_ptr(shared_storage.smem_zero_b.begin()), SmemLayoutZeroB{});

    auto rAcc           = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    auto rAccTempFirst  = make_fragment_like(rAcc);
    auto rAccTempSecond = make_fragment_like(rAcc);
    auto rScaleFirst    = make_fragment_like(rAcc);
    auto rScaleSecond   = make_fragment_like(rAcc);
    auto rCorrFirst     = make_fragment_like(rAcc);
    auto rCorrSecond    = make_fragment_like(rAcc);
    fill(rAcc, float(0));

    Tensor tCsA = thr_mma.partition_A(sMmaA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    MUTE_STATIC_ASSERT_V(size<2>(tCrA) == _4{});
    MUTE_STATIC_ASSERT_V(size<2>(tCrB) == _4{});

    using RowSumViewAsCLayout = Layout<Shape<Shape<_1, Int<BlockM>>, Int<BlockN>>, Stride<Stride<_0, _1>, _0>>;
    using ScaleBViewAsCLayout = Layout<Shape<Int<BlockM>, Shape<_1, Int<BlockN>>>, Stride<_0, Stride<_0, _1>>>;

    // The A reduction is published on the cast pipeline together with the tile it belongs to, so it is
    // read per stage rather than software-prefetched like the weight streams.
    Tensor tCsRowSumTemplate   = thr_mma.partition_C(sRowSum(_, _0{}, _0{}).compose(RowSumViewAsCLayout{}));
    Tensor tCrRowSumFirst      = make_tensor_like<ElementRowSum>(tCsRowSumTemplate);
    Tensor tCrRowSumSecond     = make_tensor_like<ElementRowSum>(tCsRowSumTemplate);
    auto   tCrRowSumFirstFlat  = filter_zeros(tCrRowSumFirst);
    auto   tCrRowSumSecondFlat = filter_zeros(tCrRowSumSecond);

    Tensor tCsScaleBTemplate       = thr_mma.partition_C(sScaleB(_, _0{}, _0{}).compose(ScaleBViewAsCLayout{}));
    Tensor tCrScaleBPacked         = make_tensor_like<uint32_t>(tCsScaleBTemplate);
    Tensor tCrScaleBNextPacked     = make_tensor_like<uint32_t>(tCsScaleBTemplate);
    auto   tCrScaleBPackedFlat     = filter_zeros(tCrScaleBPacked);
    auto   tCrScaleBNextPackedFlat = filter_zeros(tCrScaleBNextPacked);

    Tensor tCsZeroBTemplate       = thr_mma.partition_C(sZeroB(_, _0{}, _0{}).compose(ScaleBViewAsCLayout{}));
    Tensor tCrZeroBPacked         = make_tensor_like<uint32_t>(tCsZeroBTemplate);
    Tensor tCrZeroBNextPacked     = make_tensor_like<uint32_t>(tCsZeroBTemplate);
    auto   tCrZeroBPackedFlat     = filter_zeros(tCrZeroBPacked);
    auto   tCrZeroBNextPackedFlat = filter_zeros(tCrZeroBNextPacked);

    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCrRowSumFirst));
    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCrScaleBPacked));
    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCrZeroBPacked));

    pipeline_load_b.consumer_wait(pipe_b_prefetch);
    const int initial_scale_stage = int(pipe_b_prefetch.index());
    Tensor tCsScaleBFirst = thr_mma.partition_C(sScaleB(_, _0{}, initial_scale_stage).compose(ScaleBViewAsCLayout{}));
    auto   tCsScaleBFirstFlat = filter_zeros(tCsScaleBFirst);
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCrScaleBPackedFlat); ++i) {
      UniversalCopy<uint32_t>::copy(reinterpret_cast<uint32_t const&>(tCsScaleBFirstFlat(i)), tCrScaleBPackedFlat(i));
    }
    if constexpr (IsZeroPoints) {
      Tensor tCsZeroBFirst = thr_mma.partition_C(sZeroB(_, _0{}, initial_scale_stage).compose(ScaleBViewAsCLayout{}));
      auto   tCsZeroBFirstFlat = filter_zeros(tCsZeroBFirst);
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrZeroBPackedFlat); ++i) {
        UniversalCopy<uint32_t>::copy(reinterpret_cast<uint32_t const&>(tCsZeroBFirstFlat(i)), tCrZeroBPackedFlat(i));
      }
    }
    ++pipe_b_prefetch;

    MUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = 0; k_block < params.k_blocks; ++k_block) {
      if (k_block < params.k_blocks - 1) {
        pipeline_load_b.consumer_wait(pipe_b_prefetch);
        const int next_scale_stage = int(pipe_b_prefetch.index());
        Tensor    tCsScaleBNextFirst =
            thr_mma.partition_C(sScaleB(_, _0{}, next_scale_stage).compose(ScaleBViewAsCLayout{}));
        auto tCsScaleBNextFirstFlat = filter_zeros(tCsScaleBNextFirst);
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleBNextPackedFlat); ++i) {
          UniversalCopy<uint32_t>::copy(reinterpret_cast<uint32_t const&>(tCsScaleBNextFirstFlat(i)),
                                        tCrScaleBNextPackedFlat(i));
        }
        if constexpr (IsZeroPoints) {
          Tensor tCsZeroBNextFirst =
              thr_mma.partition_C(sZeroB(_, _0{}, next_scale_stage).compose(ScaleBViewAsCLayout{}));
          auto tCsZeroBNextFirstFlat = filter_zeros(tCsZeroBNextFirst);
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrZeroBNextPackedFlat); ++i) {
            UniversalCopy<uint32_t>::copy(reinterpret_cast<uint32_t const&>(tCsZeroBNextFirstFlat(i)),
                                          tCrZeroBNextPackedFlat(i));
          }
        }
        ++pipe_b_prefetch;
      }

      pipeline_cast_a.consumer_wait(pipe_cast_a_read);
      pipeline_load_b.consumer_wait(pipe_b_read);

      const int stage_a = int(pipe_cast_a_read.index());
      const int stage_b = int(pipe_b_read.index());

      // The reduction of the A tile stages currently in the MMA buffer, one pair per K128 group.
      if constexpr (IsZeroPoints) {
        Tensor tCsRowSumFirst      = thr_mma.partition_C(sRowSum(_, _0{}, stage_a).compose(RowSumViewAsCLayout{}));
        Tensor tCsRowSumSecond     = thr_mma.partition_C(sRowSum(_, _1{}, stage_a).compose(RowSumViewAsCLayout{}));
        auto   tCsRowSumFirstFlat  = filter_zeros(tCsRowSumFirst);
        auto   tCsRowSumSecondFlat = filter_zeros(tCsRowSumSecond);
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrRowSumFirstFlat); ++i) {
          UniversalCopy<ElementRowSum>::copy(tCsRowSumFirstFlat(i), tCrRowSumFirstFlat(i));
          UniversalCopy<ElementRowSum>::copy(tCsRowSumSecondFlat(i), tCrRowSumSecondFlat(i));
        }
      }

      tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
      MUTLASS_PRAGMA_UNROLL
      for (int mma_k = 0; mma_k < SqmmasPerScale; ++mma_k) {
        mute::gemm(tiled_mma, tCrA(_, _, mma_k, stage_a), tCrB(_, _, mma_k, stage_b), rAccTempFirst);
        tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
      }
      ::mate::warpsquad_commit_batch();

      tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
      MUTLASS_PRAGMA_UNROLL
      for (int mma_k = SqmmasPerScale; mma_k < ScaleBlocksPerTile * SqmmasPerScale; ++mma_k) {
        mute::gemm(tiled_mma, tCrA(_, _, mma_k, stage_a), tCrB(_, _, mma_k, stage_b), rAccTempSecond);
        tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
      }
      ::mate::warpsquad_commit_batch();

      unpack_scale_bst4<0>(rScaleFirst, tCrScaleBPacked);
      unpack_scale_bst4<1>(rScaleSecond, tCrScaleBPacked);
      if constexpr (IsZeroPoints) {
        calculate_correction_bst4<0>(rCorrFirst, tCrRowSumFirst, tCrZeroBPacked);
        calculate_correction_bst4<1>(rCorrSecond, tCrRowSumSecond, tCrZeroBPacked);
      }
      if (k_block < params.k_blocks - 1) {
        copy(tCrScaleBNextPackedFlat, tCrScaleBPackedFlat);
        if constexpr (IsZeroPoints) {
          copy(tCrZeroBNextPackedFlat, tCrZeroBPackedFlat);
        }
      }
      ::mate::warpsquad_wait<1>();
      scale_accumulate_bst4(rAcc, rAccTempFirst, rScaleFirst, rCorrFirst);
      ::mate::warpsquad_wait<0>();
      scale_accumulate_bst4(rAcc, rAccTempSecond, rScaleSecond, rCorrSecond);

      pipeline_cast_a.consumer_release(pipe_cast_a_read);
      pipeline_load_b.consumer_release(pipe_b_read);
      ++pipe_cast_a_read;
      ++pipe_b_read;
    }

    return rAcc;
  }

  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static_assert(SharedStorageSize <= MaxSharedStorageBytes, "Shared storage exceeds the MP31 per-block limit");

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineLoadAParams pipeline_a_params;
    pipeline_a_params.transaction_bytes = TmeTransactionBytesA;
    pipeline_a_params.num_consumers     = NumCasterWarpSquads * WarpsPerWarpSquad;
    pipeline_a_params.num_producers     = 1;
    PipelineLoadA pipeline_load_a(pipeline_a_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineLoadA));

    PipelineLoadBParams pipeline_b_params;
    pipeline_b_params.transaction_bytes = TmeTransactionBytesB + TmeTransactionBytesScaleB + TmeTransactionBytesZeroB;
    pipeline_b_params.num_consumers     = NumConsumerWarpSquads * WarpsPerWarpSquad;
    pipeline_b_params.num_producers     = 1;
    PipelineLoadB pipeline_load_b(pipeline_b_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineLoadB));

    PipelineCastAParams pipeline_cast_a_params;
    pipeline_cast_a_params.producer_arv_count = NumCasterWarpSquads * WarpsPerWarpSquad;
    pipeline_cast_a_params.consumer_arv_count = NumConsumerWarpSquads * WarpsPerWarpSquad;
    PipelineCastA pipeline_cast_a(pipeline_cast_a_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineCastA));

    const int squad_idx           = mutlass::canonical_warp_squad_idx();
    const int warp_idx_in_squad   = mutlass::canonical_warp_idx_sync() % int(WarpsPerWarpSquad);
    const int thread_idx_in_squad = int(threadIdx.x) % int(NumThreadsPerWarpSquad);

    constexpr int FirstCasterSquad   = int(NumProducerWarpSquads);
    constexpr int FirstConsumerSquad = int(NumProducerWarpSquads + NumCasterWarpSquads);
    constexpr int TotalWarpSquads    = int(NumProducerWarpSquads + NumCasterWarpSquads + NumConsumerWarpSquads);

    const bool is_producer_squad = squad_idx == 0;
    const bool is_caster_squad   = squad_idx >= FirstCasterSquad && squad_idx < FirstConsumerSquad;
    const bool is_consumer_squad = squad_idx >= FirstConsumerSquad && squad_idx < TotalWarpSquads;
    const int  caster_thread_idx = (squad_idx - FirstCasterSquad) * int(NumThreadsPerWarpSquad) + thread_idx_in_squad;
    const int  consumer_thread_idx =
        (squad_idx - FirstConsumerSquad) * int(NumThreadsPerWarpSquad) + thread_idx_in_squad;

    PipelineLoadAState pipe_a_write      = mutlass::make_producer_start_state_warpspecialized<PipelineLoadA>();
    PipelineLoadBState pipe_b_write      = mutlass::make_producer_start_state_warpspecialized<PipelineLoadB>();
    PipelineCastAState pipe_cast_a_write = mutlass::make_producer_start_state_warpspecialized<PipelineCastA>();

    Scheduler scheduler(static_cast<uint32_t>(params.m), static_cast<uint32_t>(params.n), 1u, nullptr, 0u);

    __syncthreads();

    auto work_tile = scheduler.initial_work_tile_info();

    if (is_producer_squad) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        load(params,
             shared_storage,
             pipeline_load_a,
             pipeline_load_b,
             pipe_a_write,
             pipe_b_write,
             work_tile,
             warp_idx_in_squad);
        scheduler.advance_to_next_work();
        work_tile = scheduler.get_work_tile_info();
      }
      return;
    }

    PipelineLoadAState pipe_a_read;
    PipelineLoadBState pipe_b_read;
    PipelineLoadBState pipe_b_prefetch;
    PipelineCastAState pipe_cast_a_read;

    if (is_caster_squad) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        prepare_a(params,
                  shared_storage,
                  pipeline_load_a,
                  pipeline_cast_a,
                  pipe_a_read,
                  pipe_cast_a_write,
                  caster_thread_idx);
        scheduler.advance_to_next_work();
        work_tile = scheduler.get_work_tile_info();
      }
      return;
    }

    if (is_consumer_squad) {
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        auto rAcc = compute(params,
                            shared_storage,
                            pipeline_load_b,
                            pipeline_cast_a,
                            pipe_b_read,
                            pipe_b_prefetch,
                            pipe_cast_a_read,
                            work_tile,
                            consumer_thread_idx);
        epilogue(params, rAcc, work_tile, consumer_thread_idx);

        scheduler.advance_to_next_work();
        work_tile = scheduler.get_work_tile_info();
      }
    }
  }
};

}  // namespace mate::gemm::dense_w4a16_gemm
