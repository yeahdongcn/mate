#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/pipeline/pipeline.hpp>
#include <type_traits>

#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

template <class CollectivePrepare_, class TileScheduler_>
struct ChunkKdaPrepareKernel {
  using CollectivePrepare = CollectivePrepare_;
  using TileScheduler     = TileScheduler_;
  using Components        = typename CollectivePrepare::Components;
  using ArchTag           = typename CollectivePrepare::ArchTag;
  using CuSeqlensElement  = typename TileScheduler::CuSeqlensElement;
  using ProblemSize       = typename Components::PrepareProblemShape;
  using SharedStorage     = typename CollectivePrepare::SharedStorage;
  using BarrierStorage    = typename CollectivePrepare::BarrierStorage;
  using Pipeline          = typename CollectivePrepare::Pipeline;

  static_assert(std::is_same_v<ProblemSize, typename TileScheduler::ProblemShape>);

  static constexpr int      SharedStorageSize          = CollectivePrepare::SharedStorageSize;
  static constexpr int      SmemAlignmentBytes         = CollectivePrepare::SmemAlignmentBytes;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int      MaxThreadsPerBlock =
      CollectivePrepare::NumProducerThreads + CollectivePrepare::NumConsumerThreads;

  struct Arguments {
    ProblemSize                           problem_size;
    typename CollectivePrepare::Arguments prepare;
  };

  struct Params {
    ProblemSize                        problem_size;
    typename CollectivePrepare::Params prepare;
    typename TileScheduler::Params     scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{
        args.problem_size,
        CollectivePrepare::to_underlying_arguments(args.problem_size, args.prepare),
        TileScheduler::to_underlying_arguments(args.problem_size),
    };
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    int thread_idx = int(threadIdx.x);
    int warp_idx   = mutlass::canonical_warp_idx();

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    TileScheduler tile_scheduler;

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);
    CollectivePrepare::init_stage_barriers(barrier_storage, int(threadIdx.x));
    Pipeline pipeline(barrier_storage);

    __syncthreads();

    auto work_cursor = tile_scheduler.get_initial_work(params.scheduler, params.problem_size);
    while (work_cursor.remaining > 0) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
      __musa_loop_transparent_outermost();
#endif
      auto work_desc = tile_scheduler.get_work(params.scheduler, params.problem_size, work_cursor);

      if (warp_idx < CollectivePrepare::NumConsumerWarps) {
        CollectivePrepare::compute_consumer_operands(
            shared_storage, params.prepare, params.problem_size, work_desc, pipeline, thread_idx);
      } else {
        int producer_warp = warp_idx - CollectivePrepare::NumConsumerWarps;
        if (producer_warp == CollectivePrepare::QkProducerWarp) {
          if (work_desc.n_chunks > 0) {
            typename CollectivePrepare::LoadQ load_q(params.prepare.tme_q, pipeline.qk, shared_storage.smem_q_raw);
            typename CollectivePrepare::LoadK load_k(params.prepare.tme_k, pipeline.qk, shared_storage.smem_k_raw);
            auto                              q_src_dst =
                load_q.partition_SD(params.problem_size, typename CollectivePrepare::TileShape{}, work_desc);
            auto k_src_dst =
                load_k.partition_SD(params.problem_size, typename CollectivePrepare::TileShape{}, work_desc);
            for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx) {
              if (mutlass::canonical_lane_idx() == 0) {
                pipeline.qk.producer_acquire(pipeline.qk_write);
              }
              auto q_write = pipeline.qk_write;
              auto k_write = pipeline.qk_write;
              load_q.template step<false>(q_src_dst, chunk_idx, q_write);
              load_k.template step<false>(k_src_dst, chunk_idx, k_write);
              if (mutlass::canonical_lane_idx() == 0) {
                ++pipeline.qk_write;
              }
            }
          }
        } else if (producer_warp == CollectivePrepare::GateProducerWarp) {
          if (work_desc.n_chunks > 0) {
            typename CollectivePrepare::LoadG load_g(params.prepare.tme_g, pipeline.g, shared_storage.smem_g_raw);
            auto                              g_src_dst =
                load_g.partition_SD(params.problem_size, typename CollectivePrepare::TileShape{}, work_desc);
            for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx) {
              if (mutlass::canonical_lane_idx() == 0) {
                pipeline.g.producer_acquire(pipeline.qk_write);
              }
              auto g_write = pipeline.qk_write;
              load_g.template step<false>(g_src_dst, chunk_idx, g_write);
              if (mutlass::canonical_lane_idx() == 0) {
                ++pipeline.qk_write;
              }
            }
          }
        } else if (producer_warp == CollectivePrepare::StoreProducerWarp) {
          CollectivePrepare::store_workspace_tme(params.prepare,
                                                 params.problem_size,
                                                 work_desc,
                                                 shared_storage,
                                                 pipeline,
                                                 thread_idx - CollectivePrepare::NumConsumerThreads);
        }
      }
      tile_scheduler.advance(params.scheduler, params.problem_size, work_cursor);
    }
  }
};

}  // namespace mate::flat::kda
