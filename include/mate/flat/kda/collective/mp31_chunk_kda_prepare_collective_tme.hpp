#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <cstdint>
#include <limits>
#include <mute/arch/copy_mp31_tme.hpp>
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

// The MP31 frontend exposes the correctly-rounded reciprocal-square-root
// builtin in BuiltinsMTGPU.def but does not declare it in the public math
// headers.  Declare the device entry explicitly so mcc can select the native
// instruction without the generic rsqrtf edge-case wrapper.
extern "C" __device__ float __musa_rsqrt_rn_f(float);

namespace mate::flat::kda {

namespace prepare {

using namespace mute;

static constexpr float kLog2E = 1.4426950408889634074f;

template <class Components_, class TileShape_, class StrideQ_, class StrideK_, class StrideG_>
struct Mp31ChunkKdaPrepareCollectiveTme {
  using ArchTag                       = mutlass::arch::Mp31;
  using Components                    = Components_;
  using Element                       = typename Components::Element;
  using TileShape                     = TileShape_;
  using StrideQ                       = StrideQ_;
  using StrideK                       = StrideK_;
  using StrideV                       = ChunkKdaVoStride;
  using StrideG                       = StrideG_;
  static constexpr bool HasGateParams = Components::HasGateParams;
  static constexpr bool NormalizeQK   = Components::NormalizeQK;
  static constexpr bool IsVarlen      = Components::IsVarlen;

  static constexpr int kChunk    = Components::kChunk;
  static constexpr int kHeadDim  = Components::kHeadDim;
  static constexpr int kValueDim = Components::kValueDim;
  using Workspace                = typename Components::Workspace;
  static_assert(decltype(get<0>(TileShape{}))::value == kChunk);
  static_assert(decltype(get<1>(TileShape{}))::value == kChunk);
  static_assert(decltype(get<2>(TileShape{}))::value == kHeadDim);
  static constexpr int kValueMmaN = kValueDim < 64 ? kValueDim : 64;
  static_assert(decltype(get<0>(TileShape{}))::value == decltype(get<1>(TileShape{}))::value);
  static_assert(kChunk == 16, "the 32x32 SQMMA warp-role permutation requires chunk size 16");

  // Keep the four SQMMA consumers on a physical warp-squad boundary.  MP31
  // launches SQMMA CTAs in complete 128-thread squads.  Three producer warps
  // are active (Q, K and G); the fourth producer warp performs workspace
  // stores.
  static constexpr int NumProducerWarps   = 3;
  static constexpr int NumConsumerSquads  = 1;
  static constexpr int NumProducerThreads = mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumConsumerThreads = NumConsumerSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumConsumerWarps   = NumConsumerSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int QkProducerWarp     = 0;
  static constexpr int StoreProducerWarp  = 1;
  static constexpr int GateProducerWarp   = 2;
  static_assert(NumConsumerThreads == kHeadDim);
  static_assert(NumConsumerWarps == 4);
  static constexpr int NumStageBarriers   = static_cast<int>(PrepareBarrier::NumBarriers);
  static constexpr int SmemAlignmentBytes = 256;
  static constexpr int VecSize            = 4;

  // The MUTLASS builder requires at least two stages, but K1 only needs one
  // physical Q/K stage: gate/normalization, inverse and stores hide the next
  // load, while the smaller storage raises persistent CTA occupancy.
  using BuilderStageCount             = mutlass::gemm::collective::StageCount<2>;
  static constexpr int PipelineStages = 1;
  using TileShapeQK                   = Shape<Int<2 * kChunk>, Int<2 * kChunk>, Int<kHeadDim>>;
  using TileShapeKU                   = Shape<Int<kHeadDim>, Int<kValueDim>, Int<kChunk>>;
  using TileShapeQS                   = Shape<Int<kChunk>, Int<kValueDim>, Int<kHeadDim>>;
  using CollectiveMmaQK =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideQ,
                                                            2,
                                                            Element,
                                                            StrideK,
                                                            2,
                                                            float,
                                                            TileShapeQK,
                                                            Shape<_1, _1, _1>,
                                                            BuilderStageCount,
                                                            mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;
  static constexpr int BuilderStages = CollectiveMmaQK::DispatchPolicy::Stages;
  static_assert(BuilderStages == BuilderStageCount::value);
  using StateAtomLayout = Layout<Shape<_1, _1, _1>>;
  using RowAtomLayout   = Layout<Shape<_1, _1, _1>>;

  using CollectiveMmaKU = typename collective::Mp31TmeSqmmaCollective<Element,
                                                                      StrideV,
                                                                      StrideV,
                                                                      TileShapeKU,
                                                                      Int<kHeadDim>,
                                                                      Int<kValueMmaN>,
                                                                      StateAtomLayout,
                                                                      BuilderStages>::CollectiveOp;
  using CollectiveMmaQS = typename collective::Mp31TmeSqmmaCollective<Element,
                                                                      StrideQ,
                                                                      StrideQ,
                                                                      TileShapeQS,
                                                                      Int<kChunk>,
                                                                      Int<kValueMmaN>,
                                                                      RowAtomLayout,
                                                                      BuilderStages>::CollectiveOp;
  using CollectiveMmaKS = typename collective::Mp31TmeSqmmaCollective<Element,
                                                                      StrideK,
                                                                      StrideK,
                                                                      TileShapeQS,
                                                                      Int<kChunk>,
                                                                      Int<kValueMmaN>,
                                                                      RowAtomLayout,
                                                                      BuilderStages>::CollectiveOp;
  using TiledMmaQK      = typename CollectiveMmaQK::TiledMma;
  static_assert(decltype(size(TiledMmaQK{}))::value == NumConsumerThreads);

  using SmemLayoutQ =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaQS::SmemLayoutA{}, Int<PipelineStages>{}));
  using SmemLayoutK =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaKS::SmemLayoutA{}, Int<PipelineStages>{}));
  using SmemLayoutG = SmemLayoutQ;
  using SmemLayoutDecayed =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutA{}, Int<PipelineStages>{}));
  using SmemLayoutKRestored =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaKU::SmemLayoutA{}, Int<PipelineStages>{}));
  using SmemLayoutKInverse =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutB{}, Int<PipelineStages>{}));
  using CollectiveInverseNxN = collective::CollectiveLowerTriangularInverseNxN<Element, kChunk>;

  using SmemLayoutGCumsum = decltype(as_position_independent_swizzle_layout(take<0, 2>(SmemLayoutQ{})));

  using TME_Q = typename CollectiveMmaQS::Params::TME_A;
  using TME_K = typename CollectiveMmaKS::Params::TME_A;
  // Q and G use the same [T, D, head, batch] descriptor layout in the public
  // KDA ABI.  Reusing the Q descriptor type avoids carrying a second builder
  // object through every persistent work item.
  using TME_G = TME_Q;

  // Workspace stores are issued by the second producer warp.
  // Keep their descriptors separate from the input-load descriptors: TME
  // load/store have different copy traits even when the logical tile layout
  // is identical.
  using TMEWorkspaceDecayed   = decltype(make_tme_copy(
      MP31_TME_STORE{},
      make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), typename Workspace::DecayedLayout{}),
      take<0, 2>(SmemLayoutDecayed{})));
  using TMEWorkspaceKRestored = decltype(make_tme_copy(
      MP31_TME_STORE{},
      make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), typename Workspace::KRestoredLayout{}),
      take<0, 2>(SmemLayoutKRestored{})));
  using PipelineQK            = mutlass::Mp31PipelineTmeAsync<PipelineStages>;
  using PipelineG             = mutlass::Mp31PipelineTmeAsync<PipelineStages>;
  using LoadQ                 = collective::
      CollectiveLoadTme<collective::LoadKind::kQ, PipelineQK, Element, SmemLayoutQ, TME_Q, SmemAlignmentBytes>;
  using LoadK = collective::
      CollectiveLoadTme<collective::LoadKind::kK, PipelineQK, Element, SmemLayoutK, TME_K, SmemAlignmentBytes>;
  using LoadG = collective::
      CollectiveLoadTme<collective::LoadKind::kG, PipelineG, Element, SmemLayoutG, TME_G, SmemAlignmentBytes>;
  using PipelineQKParams = typename PipelineQK::Params;
  using PipelineQKState  = typename PipelineQK::PipelineState;
  using PipelineGParams  = typename PipelineG::Params;

  template <class StorageElement, class SmemLayout>
  using SmemArray = array_aligned<StorageElement, cosize_v<SmemLayout>, SmemAlignmentBytes>;

  struct SharedStorage {
    typename LoadQ::SharedStorage smem_q_raw;
    typename LoadK::SharedStorage smem_k_raw;
    // The raw G tile is consumed completely by the prefix pass before
    // scale_normalized_qk writes K-restored.  Reusing that 4 KiB stage keeps
    // the CTA below the MP31 shared-memory allocation bucket boundary.
    union {
      typename LoadG::SharedStorage           smem_g_raw;
      SmemArray<Element, SmemLayoutKRestored> smem_k_restored;
    };
    SmemArray<Element, SmemLayoutDecayed> smem_decayed;
    // scale_normalized_qk reads each gate-prefix element before the
    // subwarp norm reduction, then writes K-inverse.  The reduction is the
    // phase boundary, so the gate-prefix and inverse tiles can share bytes.
    union {
      SmemArray<Element, SmemLayoutKInverse> smem_k_inverse;
      SmemArray<float, SmemLayoutGCumsum>    smem_g_cumsum;
    };
    typename CollectiveInverseNxN::SharedStorage smem_inverse_workspace;
  };

  static constexpr int     SharedStorageSize = sizeof(SharedStorage);
  static constexpr int64_t KTileBytes        = int64_t(kChunk) * kHeadDim * sizeof(Element);
  static constexpr int64_t QKTileBytes       = 2 * KTileBytes;

  struct Params;

  struct Arguments {
    Element const*                      ptr_q;
    Element const*                      ptr_k;
    Element const*                      ptr_g;
    Element const*                      ptr_beta;
    float const*                        ptr_A_log;
    float const*                        ptr_dt_bias;
    StrideQ                             stride_q;
    StrideK                             stride_k;
    StrideG                             stride_g;
    float                               scale;
    float                               lower_bound;
    typename Workspace::MutablePointers workspace;
  };

  struct Params {
    Element const*                      ptr_q;
    Element const*                      ptr_k;
    Element const*                      ptr_g;
    Element const*                      ptr_beta;
    float const*                        ptr_A_log;
    float const*                        ptr_dt_bias;
    StrideQ                             stride_q;
    StrideK                             stride_k;
    StrideG                             stride_g;
    float                               scale;
    float                               lower_bound;
    typename Workspace::MutablePointers workspace;
    TME_Q                               tme_q;
    TME_K                               tme_k;
    TME_G                               tme_g;
    TMEWorkspaceDecayed                 tme_workspace_decayed;
    TMEWorkspaceKRestored               tme_workspace_k_restored;
  };

  struct Pipeline;

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void store_workspace_tme(Params const&      params,
                                                 ProblemSize const& problem_size,
                                                 WorkDesc const&    work_desc,
                                                 SharedStorage&     shared_storage,
                                                 Pipeline&          pipeline,
                                                 int                thread_idx) {
    int producer_warp = thread_idx / mutlass::NumThreadsPerWarp;
    if (producer_warp != StoreProducerWarp || work_desc.n_chunks <= 0) {
      return;
    }
    int lane = thread_idx % mutlass::NumThreadsPerWarp;

    int32_t records = Components::workspace_records(problem_size);
    int32_t outer   = records * problem_size.H;

    auto issue_decayed = [&](int record, int stage) INLINE_LAMBDA {
      auto g_decayed =
          params.tme_workspace_decayed.get_tme_tensor(make_shape(Int<2 * kChunk>{}, Int<kHeadDim>{}, outer));
      auto s_decayed = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{});
      auto cta_tme   = params.tme_workspace_decayed.get_slice(_0{});
      auto tS        = group_modes<0, 3>(cta_tme.partition_S(s_decayed));
      auto tG        = group_modes<0, 3>(cta_tme.partition_D(g_decayed));
      copy(params.tme_workspace_decayed, tS(_, stage), tG(_, int32_t(record)));
      tme_store_arrive();
    };
    auto issue_kr = [&](int record, int stage) INLINE_LAMBDA {
      auto g_kr    = params.tme_workspace_k_restored.get_tme_tensor(make_shape(Int<kHeadDim>{}, Int<kChunk>{}, outer));
      auto s_kr    = make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestored{});
      auto cta_tme = params.tme_workspace_k_restored.get_slice(_0{});
      auto tS      = group_modes<0, 3>(cta_tme.partition_S(s_kr));
      auto tG      = group_modes<0, 3>(cta_tme.partition_D(g_kr));
      copy(params.tme_workspace_k_restored, tS(_, stage), tG(_, int32_t(record)));
      tme_store_arrive();
    };
    for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx) {
      int      record = Workspace::record(problem_size, work_desc, chunk_idx);
      int      stage  = pipeline.store_stage.index();
      uint32_t phase  = pipeline.store_stage.phase();

      named_barrier_wait(PrepareBarrier::WorkspaceReady, phase);
      if (lane == 0) {
        issue_decayed(record, stage);
        issue_kr(record, stage);
      }

      if (lane == 0) {
        tme_store_wait();
        named_barrier_arrive(PrepareBarrier::WorkspaceStoreDone);
      }
      ++pipeline.store_stage;
      (void)pipeline;
    }
  }

  struct alignas(1) BarrierStorage {
    uint8_t stage[NumStageBarriers];
    uint8_t pipeline_qk[PipelineQK::NumBarriers];
    uint8_t pipeline_g[PipelineG::NumBarriers];
  };

  struct Pipeline {
    PipelineQK      qk;
    PipelineG       g;
    PipelineQKState qk_read;
    PipelineQKState qk_write;
    PipelineQKState store_stage;

    static MUTLASS_DEVICE PipelineQKParams make_qk_params() {
      PipelineQKParams params;
      params.transaction_bytes = uint32_t(QKTileBytes);
      params.num_consumers     = NumConsumerWarps;
      params.num_producers     = 1;
      return params;
    }

    static MUTLASS_DEVICE PipelineGParams make_g_params() {
      PipelineGParams params;
      params.transaction_bytes = uint32_t(KTileBytes);
      params.num_consumers     = NumConsumerWarps;
      params.num_producers     = 1;
      return params;
    }

    MUTLASS_DEVICE explicit Pipeline(BarrierStorage* barrier_storage)
        : qk(make_qk_params(), reinterpret_cast<uint64_t>(&barrier_storage->pipeline_qk), 1),
          g(make_g_params(), reinterpret_cast<uint64_t>(&barrier_storage->pipeline_g), 1),
          qk_read{},
          qk_write(mutlass::make_producer_start_state<PipelineQK>()),
          store_stage{} {
    }
  };

  static MUTLASS_DEVICE void init_stage_barriers(BarrierStorage* barrier_storage, int tid) {
    (void)barrier_storage;
    static constexpr uint32_t StageBarrierArriveCount[NumStageBarriers] = {
        NumConsumerWarps,  // GateTotalsReady
        NumConsumerWarps,  // WorkspaceReady
        NumConsumerWarps,  // OperandsReady
        2,                 // InverseSmemReady
        1,                 // WorkspaceStoreDone
    };
    if (tid == 0) {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < NumStageBarriers; ++i) {
        named_barrier_init(i, StageBarrierArriveCount[i]);
      }
    }
  }

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    auto params_q = CollectiveMmaQS::to_underlying_arguments(
        make_shape(problem_size.T, problem_size.T, Int<kHeadDim>{}, make_shape(problem_size.Hqk, problem_size.B)),
        typename CollectiveMmaQS::Arguments{
            args.ptr_q,
            args.stride_q,
            args.ptr_q,
            args.stride_q,
        },
        nullptr);
    auto params_k = CollectiveMmaKS::to_underlying_arguments(
        make_shape(problem_size.T, problem_size.T, Int<kHeadDim>{}, make_shape(problem_size.Hqk, problem_size.B)),
        typename CollectiveMmaKS::Arguments{
            args.ptr_k,
            args.stride_k,
            args.ptr_k,
            args.stride_k,
        },
        nullptr);
    auto params_g = CollectiveMmaQS::to_underlying_arguments(
        make_shape(problem_size.T, problem_size.T, Int<kHeadDim>{}, make_shape(problem_size.H, problem_size.B)),
        typename CollectiveMmaQS::Arguments{
            args.ptr_g,
            args.stride_g,
            args.ptr_g,
            args.stride_g,
        },
        nullptr);
    int32_t records           = Components::workspace_records(problem_size);
    int32_t outer             = records * problem_size.H;
    auto    workspace_decayed = Workspace::make_decayed_tensor(args.workspace.decayed, outer);
    auto    workspace_kr      = Workspace::make_k_restored_tensor(args.workspace.k_restored, outer);
    return Params{
        args.ptr_q,
        args.ptr_k,
        args.ptr_g,
        args.ptr_beta,
        args.ptr_A_log,
        args.ptr_dt_bias,
        args.stride_q,
        args.stride_k,
        args.stride_g,
        args.scale,
        args.lower_bound,
        args.workspace,
        params_q.tme_load_a,
        params_k.tme_load_a,
        params_g.tme_load_a,
        make_tme_copy(MP31_TME_STORE{}, workspace_decayed, take<0, 2>(SmemLayoutDecayed{})),
        make_tme_copy(MP31_TME_STORE{}, workspace_kr, take<0, 2>(SmemLayoutKRestored{})),
    };
  }

  static MUTLASS_DEVICE float sigmoid_fast(float x) {
    return 1.0f / (1.0f + ::mate::flat::detail::fast_exp2(-x * kLog2E));
  }

  static MUTLASS_DEVICE Element inactive_gate_raw_element() {
    if constexpr (HasGateParams) {
      return -std::numeric_limits<Element>::infinity();
    } else {
      return Element(0.0f);
    }
  }

  template <class TensorQK>
  static MUTLASS_DEVICE float4 load_qk_vec4(TensorQK&& sQk, int row, int col_base) {
    return simd::load_packed4_as_float4(&sQk(make_coord(row, col_base)));
  }

  template <class TensorQK>
  static MUTLASS_DEVICE void store_qk_vec4(TensorQK&& sQk, int row, int col_base, float4 value) {
    simd::store_float4_to_packed4_rn(&sQk(make_coord(row, col_base)), value);
  }

  template <int SubWarpSize>
  static MUTLASS_DEVICE void subwarp_reduce_sum_pair(float& lhs, float& rhs) {
    MUTE_UNROLL
    for (int offset = SubWarpSize / 2; offset > 0; offset >>= 1) {
      float lhs_other = __shfl_xor_sync(uint32_t(-1), lhs, offset, SubWarpSize);
      float rhs_other = __shfl_xor_sync(uint32_t(-1), rhs, offset, SubWarpSize);
      lhs += lhs_other;
      rhs += rhs_other;
    }
  }

  template <class ProblemSize, class WorkDesc>
  static MUTLASS_DEVICE float load_beta_register(Params const&      params,
                                                 ProblemSize const& problem_size,
                                                 WorkDesc const&    work_desc,
                                                 int                chunk_idx,
                                                 int                lane,
                                                 bool               is_not_full_chunk) {
    bool    token_valid = lane < kChunk && (!is_not_full_chunk || lane < work_desc.actual_len(chunk_idx));
    int64_t offset =
        (int64_t(work_desc.tme_batch()) * problem_size.T + int64_t(work_desc.tme_token(chunk_idx) + lane)) *
            problem_size.H +
        int64_t(work_desc.head_idx);
    float raw_beta =
        token_valid ? static_cast<float>(params.ptr_beta[offset]) : -std::numeric_limits<float>::infinity();
    return sigmoid_fast(raw_beta);
  }

  template <class TensorK,
            class TensorQ,
            class TensorG,
            class TensorDecayed,
            class TensorKInverse,
            class TensorKRestored>
  static MUTLASS_DEVICE void scale_normalized_qk(TensorK&&         sK,
                                                 TensorQ&&         sQ,
                                                 TensorG&&         sGCumsum,
                                                 TensorDecayed&&   sDecayed,
                                                 TensorKInverse&&  sKInverse,
                                                 TensorKRestored&& sKRestored,
                                                 float const*      gTotal,
                                                 float             q_scale,
                                                 int               local_tid,
                                                 int               actual_len,
                                                 bool              is_not_full_chunk) {
    // Six resident prepare CTAs need the normalization live set to stay
    // below the MP31 register-allocation cliff.  Splitting each row across a
    // half warp halves the q/k/inverse vector arrays held by each thread.
    constexpr int ThreadsPerRow     = 16;
    constexpr int RowsPerWarp       = mutlass::NumThreadsPerWarp / ThreadsPerRow;
    constexpr int ElementsPerThread = kHeadDim / ThreadsPerRow;
    constexpr int VectorsPerThread  = ElementsPerThread / VecSize;
    constexpr int RowsPerWave       = mutlass::NumWarpsPerWarpSquad * RowsPerWarp;
    static_assert(kChunk % RowsPerWave == 0);
    int warp_idx      = local_tid / mutlass::NumThreadsPerWarp;
    int lane_idx      = local_tid % mutlass::NumThreadsPerWarp;
    int row_in_warp   = lane_idx / ThreadsPerRow;
    int lane_in_row   = lane_idx % ThreadsPerRow;
    int thread_col_lo = lane_in_row * ElementsPerThread;

    MUTE_UNROLL
    for (int row_base = 0; row_base < kChunk; row_base += RowsPerWave) {
      int    row          = row_base + warp_idx * RowsPerWarp + row_in_warp;
      bool   row_valid    = !is_not_full_chunk || row < actual_len;
      float  thread_sum_q = 0.0f;
      float  thread_sum_k = 0.0f;
      float4 q_decayed_reg[VectorsPerThread];
      float4 k_decayed_reg[VectorsPerThread];
      float4 k_inverse_reg[VectorsPerThread];

      MUTE_UNROLL
      for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
        int vec_slot = (vec_iter + row_in_warp) % VectorsPerThread;
        int col_base = thread_col_lo + vec_slot * VecSize;
        if (row_valid) {
          float4 gate_arg = simd::load_float4(&sGCumsum(make_coord(row, col_base)));
          float4 gate_vec = simd::fast_exp2(gate_arg);
          float4 inv_gate = simd::vrcp(gate_vec);
          float4 k_vec    = load_qk_vec4(sK, row, col_base);
          float4 q_vec    = load_qk_vec4(sQ, row, col_base);
          if constexpr (NormalizeQK) {
            simd::dot4(k_vec, k_vec, thread_sum_k);
            simd::dot4(q_vec, q_vec, thread_sum_q);
          }
          k_decayed_reg[vec_iter] = simd::vmul(k_vec, gate_vec);
          k_inverse_reg[vec_iter] = simd::vmul(k_vec, inv_gate);
          q_decayed_reg[vec_iter] = simd::vmul(q_vec, gate_vec);
        } else {
          k_decayed_reg[vec_iter] = simd::splat4(0.0f);
          k_inverse_reg[vec_iter] = simd::splat4(0.0f);
          q_decayed_reg[vec_iter] = simd::splat4(0.0f);
        }
      }

      float inv_norm_q = 1.0f;
      float inv_norm_k = 1.0f;
      if constexpr (NormalizeQK) {
        subwarp_reduce_sum_pair<ThreadsPerRow>(thread_sum_k, thread_sum_q);
        inv_norm_k = __musa_rsqrt_rn_f(thread_sum_k + 1.0e-6f);
        inv_norm_q = __musa_rsqrt_rn_f(thread_sum_q + 1.0e-6f);
      }

      MUTE_UNROLL
      for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
        int vec_slot = (vec_iter + row_in_warp) % VectorsPerThread;
        int col_base = thread_col_lo + vec_slot * VecSize;
        if constexpr (NormalizeQK) {
          k_decayed_reg[vec_iter] = simd::vmul(k_decayed_reg[vec_iter], inv_norm_k);
          k_inverse_reg[vec_iter] = simd::vmul(k_inverse_reg[vec_iter], inv_norm_k);
        }
        q_decayed_reg[vec_iter] = simd::vmul(q_decayed_reg[vec_iter], inv_norm_q * q_scale);
        float4 k_restored       = simd::vmul(k_inverse_reg[vec_iter], simd::load_float4(gTotal + col_base));
        int    k_row            = row < kChunk / 2 ? row : row + kChunk / 2;
        int    q_row            = row < kChunk / 2 ? row + kChunk / 2 : row + kChunk;
        store_qk_vec4(sDecayed, k_row, col_base, k_decayed_reg[vec_iter]);
        store_qk_vec4(sDecayed, q_row, col_base, q_decayed_reg[vec_iter]);
        store_qk_vec4(sKInverse, row, col_base, k_inverse_reg[vec_iter]);
        store_qk_vec4(sKRestored, col_base, row, k_restored);
      }
    }
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void compute_consumer_operands(SharedStorage&     shared_storage,
                                                       Params const&      params,
                                                       ProblemSize const& problem_size,
                                                       WorkDesc const&    work_desc,
                                                       Pipeline&          pipeline,
                                                       int                local_tid) {
    if (work_desc.n_chunks <= 0) {
      return;
    }

    auto sGCumsum = make_tensor(make_smem_ptr(shared_storage.smem_g_cumsum.data()), SmemLayoutGCumsum{});
    MUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx) {
      // The G producer fills a shared tile while this squad handles the
      // previous chunk's inverse/P work.  Waiting only lane 0 mirrors the Q/K
      // pipeline protocol; the warp sync makes the tile visible to all lanes.
      int lane = local_tid % mutlass::NumThreadsPerWarp;
      if (lane == 0) {
        pipeline.g.consumer_wait(pipeline.qk_read);
      }
      __syncwarp();
      int  g_stage = pipeline.qk_read.index();
      auto sGRaw   = make_tensor(make_smem_ptr(shared_storage.smem_g_raw.data()), SmemLayoutG{})(_, _, g_stage);

      // Each consumer thread owns one gate column and computes its prefix in
      // registers.  G now comes from the TME-filled shared tile rather than
      // issuing one strided global load per row.
      float a_log_exp = 1.0f;
      if constexpr (HasGateParams) {
        float warp_a_log_exp = 0.0f;
        if (lane == 0) {
          warp_a_log_exp = ::mate::flat::detail::fast_exp2(params.ptr_A_log[work_desc.head_idx] * kLog2E);
        }
        a_log_exp = __shfl_sync(uint32_t(-1), warp_a_log_exp, 0, mutlass::NumThreadsPerWarp);
      }
      int   actual_len = work_desc.actual_len(chunk_idx);
      int   col        = local_tid;
      float dt         = 0.0f;
      if constexpr (HasGateParams) {
        dt = params.ptr_dt_bias[int64_t(work_desc.head_idx) * kHeadDim + col];
      }
      float cumsum = 0.0f;
      if constexpr (HasGateParams) {
        float gate_scale = -a_log_exp * kLog2E;
        float gate_coeff = params.lower_bound * kLog2E;
        MUTLASS_PRAGMA_UNROLL
        for (int row_base = 0; row_base < kChunk; row_base += VecSize) {
          auto load_raw_g = [&](int row_offset) INLINE_LAMBDA {
            return row_base + row_offset < actual_len
                       ? static_cast<float>(sGRaw(make_coord(row_base + row_offset, col)))
                       : static_cast<float>(inactive_gate_raw_element());
          };
          float4       raw_g       = make_float4(load_raw_g(0), load_raw_g(1), load_raw_g(2), load_raw_g(3));
          float4       exp_arg     = simd::vmul(simd::vadd(raw_g, dt), gate_scale);
          float4       gate        = simd::vmul(simd::vrcp(simd::vadd(simd::fast_exp2(exp_arg), 1.0f)), gate_coeff);
          float const* gate_values = reinterpret_cast<float const*>(&gate);
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < VecSize; ++i) {
            int row = row_base + i;
            cumsum += gate_values[i];
            sGCumsum(make_coord(row, col)) = cumsum;
          }
        }
      } else {
        MUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kChunk; ++row) {
          float raw_g = row < actual_len ? static_cast<float>(sGRaw(make_coord(row, col)))
                                         : static_cast<float>(inactive_gate_raw_element());
          cumsum += raw_g;
          sGCumsum(make_coord(row, col)) = cumsum;
        }
      }
      // Invalid rows contribute exactly zero, so the final prefix is already
      // the valid chunk total.  Store it directly to the workspace from its
      // owning column thread; row-owned normalization reloads it from cache.
      float  total  = ::mate::flat::detail::fast_exp2(cumsum);
      int    record = Workspace::record(problem_size, work_desc, chunk_idx);
      float* gTotal = params.workspace.total + int64_t(record) * kHeadDim;
      gTotal[col]   = total;
      __threadfence_block();
      // Publish the column-owned cumsums before row-owned normalization.
      named_barrier_arrive_and_wait(PrepareBarrier::GateTotalsReady);
      if (lane == 0) {
        pipeline.g.consumer_release(pipeline.qk_read);
      }

      if (lane == 0) {
        pipeline.qk.consumer_wait(pipeline.qk_read);
      }
      __syncwarp();
      int      qk_stage = pipeline.qk_read.index();
      uint32_t phase    = pipeline.qk_read.phase();
      auto     sKRaw    = make_tensor(make_smem_ptr(shared_storage.smem_k_raw.data()), SmemLayoutK{})(_, _, qk_stage);
      auto     sQRaw    = make_tensor(make_smem_ptr(shared_storage.smem_q_raw.data()), SmemLayoutQ{})(_, _, qk_stage);
      auto     sDecayed =
          make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{})(_, _, qk_stage);
      auto sKInverse =
          make_tensor(make_smem_ptr(shared_storage.smem_k_inverse.data()), SmemLayoutKInverse{})(_, _, qk_stage);
      auto sKRestored =
          make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestored{})(_, _, qk_stage);
      bool is_not_full_chunk = actual_len < kChunk;
      scale_normalized_qk(sKRaw,
                          sQRaw,
                          sGCumsum,
                          sDecayed,
                          sKInverse,
                          sKRestored,
                          gTotal,
                          params.scale,
                          local_tid,
                          actual_len,
                          is_not_full_chunk);

      named_barrier_arrive_and_wait(PrepareBarrier::OperandsReady);
      // Raw Q/K are no longer live.  Release their shared pipeline stage
      // before inverse/P and workspace stores so the producer can refill it.
      __syncwarp();
      if (lane == 0) {
        pipeline.qk.consumer_release(pipeline.qk_read);
      }
      ++pipeline.qk_read;
      named_barrier_arrive(PrepareBarrier::WorkspaceReady);

      compute_inverse_and_p(
          shared_storage, params, problem_size, work_desc, chunk_idx, qk_stage, local_tid, is_not_full_chunk);
      named_barrier_wait(PrepareBarrier::WorkspaceStoreDone, phase);
    }
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void compute_inverse_and_p(SharedStorage&     shared_storage,
                                                   Params const&      params,
                                                   ProblemSize const& problem_size,
                                                   WorkDesc const&    work_desc,
                                                   int                chunk_idx,
                                                   int                stage,
                                                   int                local_tid,
                                                   bool               is_not_full_chunk) {
    auto sDecayed = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{})(_, _, stage);
    auto sKInverse =
        make_tensor(make_smem_ptr(shared_storage.smem_k_inverse.data()), SmemLayoutKInverse{})(_, _, stage);

    TiledMmaQK tiled_mma;
    auto       thr_mma = tiled_mma.get_thread_slice(local_tid);
    auto       acc     = partition_fragment_C(tiled_mma, make_shape(Int<2 * kChunk>{}, Int<2 * kChunk>{}));
    static_assert(decltype(size(acc))::value == 8);
    clear(acc);

    auto tSsDecayed  = thr_mma.partition_A(sDecayed);
    auto tSsKInverse = thr_mma.partition_B(sKInverse);
    auto tSrDecayed  = thr_mma.make_fragment_A(tSsDecayed);
    auto tSrKInverse = thr_mma.make_fragment_B(tSsKInverse);

    // Only K-inverse rows 0..15 are initialized.  They produce the valid C
    // columns 0..15; columns 16..31 are intentionally computed and discarded.
    gemm(tiled_mma, acc, tSrDecayed, tSrKInverse, acc);
    mate::warpsquad_commit_batch();

    int   warp_idx  = local_tid / mutlass::NumThreadsPerWarp;
    float beta_lane = 0.0f;
    if (warp_idx < 2) {
      beta_lane = load_beta_register(
          params, problem_size, work_desc, chunk_idx, local_tid % mutlass::NumThreadsPerWarp, is_not_full_chunk);
    }

    mate::warpsquad_wait<0>();
    CollectiveInverseNxN inverse_nxn;
    int64_t              record      = Workspace::record(problem_size, work_desc, chunk_idx);
    Element*             gInverseOut = params.workspace.inverse + record * Workspace::InverseElements;
    Element*             gP          = params.workspace.p + record * Workspace::PElements;
    auto                 matrix      = make_tensor(make_smem_ptr(shared_storage.smem_inverse_workspace.matrix),
                              typename CollectiveInverseNxN::SmemLayout{});
    auto                 cC          = make_identity_tensor(make_shape(Int<2 * kChunk>{}, Int<2 * kChunk>{}));
    auto                 tCc         = thr_mma.partition_C(cC);
    // MP31 32x32 C ownership pairs physical rows r and r+16 in each
    // warp.  The decayed row permutation therefore gives warp 0/1 only
    // K rows and warp 2/3 only Q rows: KKT and P never coexist in one
    // thread's accumulator fragment.  acc[0..3] and acc[4..7] are scatter
    // values (columns differ by 8), not contiguous matrix vectors.
    if (warp_idx < 2) {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc) / VecSize; ++i) {
        int   base                        = i * VecSize;
        int   row                         = int(get<0>(tCc(base)));
        int   col_lo                      = int(get<1>(tCc(base)));
        int   col_hi                      = int(get<1>(tCc(base + 1)));
        int   k_row                       = row < kChunk / 2 ? row : row - kChunk / 2;
        float beta_row                    = __shfl_sync(uint32_t(-1), beta_lane, k_row, mutlass::NumThreadsPerWarp);
        matrix(make_coord(k_row, col_lo)) = k_row == col_lo ? 1.0f : (k_row > col_lo ? acc(base) * beta_row : 0.0f);
        matrix(make_coord(k_row, col_hi)) = k_row == col_hi ? 1.0f : (k_row > col_hi ? acc(base + 1) * beta_row : 0.0f);
      }

      inverse_nxn.solve_columns(
          shared_storage.smem_inverse_workspace,
          gInverseOut,
          beta_lane,
          []() { named_barrier_arrive_and_wait(PrepareBarrier::InverseSmemReady); },
          local_tid);
    } else {
      int actual_len = work_desc.actual_len(chunk_idx);
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc) / VecSize; ++i) {
        int     base   = i * VecSize;
        int     row    = int(get<0>(tCc(base)));
        int     col_lo = int(get<1>(tCc(base)));
        int     col_hi = int(get<1>(tCc(base + 1)));
        int     p_row  = row < kChunk ? row - kChunk / 2 : row - kChunk;
        float4  values = *reinterpret_cast<float4 const*>(&acc(base));
        Element packed[VecSize];
        simd::store_float4_to_packed4_rn(packed, values);
        gP[p_row * kChunk + col_lo] = p_row < actual_len && p_row >= col_lo ? packed[0] : Element(0.0f);
        gP[p_row * kChunk + col_hi] = p_row < actual_len && p_row >= col_hi ? packed[1] : Element(0.0f);
      }
    }
  }
};

}  // namespace prepare

}  // namespace mate::flat::kda
