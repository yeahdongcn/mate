#pragma once

#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/chunk_kda_mp31_policy.hpp"
#include "mate/flat/kda/collective/mp31_chunk_kda_prepare_collective_tme.hpp"
#include "mate/flat/kda/kernel/chunk_kda_metadata_kernel.hpp"
#include "mate/flat/kda/kernel/chunk_kda_prepare_kernel.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

template <class Element_,
          class CuSeqlensElement_,
          class LogicalShape_,
          class TileShape_,
          class StrideQ_,
          class StrideK_,
          class StrideG_,
          class... Options_>
struct Mp31ChunkKdaPrepareBuilder {
  using Components =
      ChunkKdaPrefillComponents<Element_, CuSeqlensElement_, LogicalShape_, ChunkKdaMp31SchedulePolicy, Options_...>;
  using CollectivePrepare =
      prepare::Mp31ChunkKdaPrepareCollectiveTme<Components, TileShape_, StrideQ_, StrideK_, StrideG_>;
  using TileScheduler = typename Components::PrepareTileScheduler;
  using Kernel        = ChunkKdaPrepareKernel<CollectivePrepare, TileScheduler>;
};

}  // namespace mate::flat::kda
