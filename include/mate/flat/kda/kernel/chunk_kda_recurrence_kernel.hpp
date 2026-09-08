#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/pipeline/pipeline.hpp>
#include <type_traits>

#include "mate/flat/flat_options.hpp"

namespace mate::flat::kda {

template <class CollectiveRecurrence_, class TileScheduler_>
struct ChunkKdaRecurrenceKernel {
  using CollectiveRecurrence = CollectiveRecurrence_;
  using TileScheduler        = TileScheduler_;
  using Components           = typename CollectiveRecurrence::Components;
  using ArchTag              = typename CollectiveRecurrence::ArchTag;
  using CuSeqlensElement     = typename TileScheduler::CuSeqlensElement;
  using ProblemSize          = typename Components::ProblemShape;
  using SharedStorage        = typename CollectiveRecurrence::SharedStorage;
  using BarrierStorage       = typename CollectiveRecurrence::BarrierStorage;
  using Pipeline             = typename CollectiveRecurrence::Pipeline;

  static_assert(std::is_same_v<ProblemSize, typename TileScheduler::ProblemShape>);

  static constexpr int      SharedStorageSize          = CollectiveRecurrence::SharedStorageSize;
  static constexpr int      SmemAlignmentBytes         = CollectiveRecurrence::SmemAlignmentBytes;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 3;
  static constexpr int      MaxThreadsPerBlock         = CollectiveRecurrence::NumProducerThreads +
                                            CollectiveRecurrence::NumStateThreads +
                                            CollectiveRecurrence::NumOutputThreads;

  struct Arguments {
    ProblemSize                              problem_size;
    typename CollectiveRecurrence::Arguments recurrence;
  };

  struct Params {
    ProblemSize                           problem_size;
    typename CollectiveRecurrence::Params recurrence;
    typename TileScheduler::Params        scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{
        args.problem_size,
        CollectiveRecurrence::to_underlying_arguments(args.problem_size, args.recurrence),
        TileScheduler::to_underlying_arguments(args.problem_size),
    };
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    enum class WarpSpecRole {
      Producer = 0,  // decayed/K_restored/V/inverse/P TME producers
      Output   = 1,  // Decayed@State, residual inverse, and P epilogue
      State0   = 2,  // recurrent update for Dv[0:32]
      State1   = 3,  // recurrent update for Dv[32:64]
    };

    int  thread_idx    = int(threadIdx.x);
    int  warp_spec_idx = mutlass::canonical_warp_squad_idx();
    auto warp_role     = WarpSpecRole(warp_spec_idx);

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    TileScheduler tile_scheduler(params.scheduler);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);
    CollectiveRecurrence::init_stage_barriers(barrier_storage, int(threadIdx.x));
    Pipeline pipeline(barrier_storage);

    __syncthreads();

    if (warp_role == WarpSpecRole::Producer) {
      for (auto work_desc = tile_scheduler.get_next_work(params.problem_size); work_desc.is_valid(tile_scheduler);
           work_desc      = tile_scheduler.get_next_work(params.problem_size)) {
        __musa_loop_transparent_outermost();
        CollectiveRecurrence::kda_recurrence_producer(
            shared_storage, params.recurrence, params.problem_size, work_desc, pipeline, thread_idx);
      }
    } else if (warp_role == WarpSpecRole::Output) {
      for (auto work_desc = tile_scheduler.get_next_work(params.problem_size); work_desc.is_valid(tile_scheduler);
           work_desc      = tile_scheduler.get_next_work(params.problem_size)) {
        __musa_loop_transparent_outermost();
        CollectiveRecurrence::kda_recurrence_output(shared_storage,
                                                    params.recurrence,
                                                    params.problem_size,
                                                    work_desc,
                                                    pipeline,
                                                    thread_idx - CollectiveRecurrence::NumProducerThreads);
      }
    } else if (warp_role == WarpSpecRole::State0 || warp_role == WarpSpecRole::State1) {
      for (auto work_desc = tile_scheduler.get_next_work(params.problem_size); work_desc.is_valid(tile_scheduler);
           work_desc      = tile_scheduler.get_next_work(params.problem_size)) {
        __musa_loop_transparent_outermost();
        CollectiveRecurrence::kda_recurrence_state(
            shared_storage,
            params.recurrence,
            params.problem_size,
            work_desc,
            pipeline,
            thread_idx - CollectiveRecurrence::NumProducerThreads - CollectiveRecurrence::NumOutputThreads);
      }
    }
  }
};

}  // namespace mate::flat::kda
