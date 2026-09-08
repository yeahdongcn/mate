#pragma once

#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/chunk_kda_mp31_policy.hpp"
#include "mate/flat/kda/collective/mp31_chunk_kda_recurrence_collective_tme_warpspecialized.hpp"
#include "mate/flat/kda/kernel/chunk_kda_recurrence_kernel.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

template <class Element_,
          class StateElement_,
          class CuSeqlensElement_,
          class LogicalShape_,
          class TileShape_,
          class StrideV_,
          class StrideO_,
          class... Options_>
struct Mp31ChunkKdaRecurrenceBuilder {
  using Components =
      ChunkKdaPrefillComponents<Element_, CuSeqlensElement_, LogicalShape_, ChunkKdaMp31SchedulePolicy, Options_...>;
  using CollectiveRecurrence = recurrence::
      Mp31ChunkKdaRecurrenceCollectiveTmeWarpSpecialized<Components, TileShape_, StateElement_, StrideV_, StrideO_>;
  using TileScheduler = typename Components::RecurrenceTileScheduler;
  using Kernel        = ChunkKdaRecurrenceKernel<CollectiveRecurrence, TileScheduler>;
};

}  // namespace mate::flat::kda
