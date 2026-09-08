#pragma once

#include "mate/flat/kda/chunk_kda_problem.hpp"

namespace mate::flat::kda {

// MP31 occupancy/resource choices.  They are injected into the common work
// traversal instead of being treated as KDA semantics.
// prepare CTAs/MP=6, recurrence CTAs/MP=3, value tile=64,
// metadata threads=512, metadata dynamic smem limit=192 KiB.
struct ChunkKdaMp31SchedulePolicy : ChunkKdaPrefillSchedulePolicy<6, 3, 64, 512, 192 * 1024> {};

}  // namespace mate::flat::kda
