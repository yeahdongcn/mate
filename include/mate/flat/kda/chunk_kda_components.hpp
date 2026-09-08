#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <type_traits>

#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/chunk_kda_problem.hpp"
#include "mate/flat/kda/chunk_kda_workspace.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

// Architecture collectives provide their own SQMMA/TME layouts and pipelines,
// but K1 and K2 must agree on this problem/workspace/scheduling contract.
// Keeping the contract free of an ArchTag lets MP31 and MP32 builders reuse it
// without pretending that their collectives are interchangeable.
template <class Element_, class CuSeqlensElement_, class LogicalShape_, class SchedulePolicy_, class... Options_>
struct ChunkKdaPrefillComponents {
  using Element             = Element_;
  using CuSeqlensElement    = CuSeqlensElement_;
  using LogicalShape        = LogicalShape_;
  using SchedulePolicy      = SchedulePolicy_;
  using ProblemShape        = ChunkKdaProblemShape<CuSeqlensElement>;
  using PrepareProblemShape = ChunkKdaPrepareProblemShape<CuSeqlensElement>;

  static constexpr bool HasStateIn =
      mate::flat::find_option_t<mate::flat::Tag::HasStateIn, std::false_type, Options_...>::value;
  static constexpr bool HasStateOut =
      mate::flat::find_option_t<mate::flat::Tag::HasStateOut, std::true_type, Options_...>::value;
  static constexpr bool HasGateParams =
      mate::flat::find_option_t<mate::flat::Tag::HasGateParams, std::true_type, Options_...>::value;
  static constexpr bool IsVarlen =
      mate::flat::find_option_t<mate::flat::Tag::IsVarlen, std::false_type, Options_...>::value;
  static constexpr bool NormalizeQK =
      mate::flat::find_option_t<mate::flat::Tag::NormalizeQK, std::true_type, Options_...>::value;

  static constexpr int kChunk               = LogicalShape::Chunk;
  static constexpr int kHeadDim             = LogicalShape::HeadDim;
  static constexpr int kValueDim            = LogicalShape::ValueDim;
  static constexpr int kRecurrenceValueTile = SchedulePolicy::RecurrenceValueTile;

  static_assert(std::is_integral_v<CuSeqlensElement>);
  static_assert(kValueDim % kRecurrenceValueTile == 0);

  using Workspace = ChunkKdaWorkspace<Element, kChunk, kHeadDim>;
  using PrepareTileScheduler =
      ChunkKdaPrepareTileScheduler<LogicalShape, CuSeqlensElement, SchedulePolicy, Options_...>;
  using RecurrenceTileScheduler =
      ChunkKdaRecurrenceTileScheduler<LogicalShape, CuSeqlensElement, SchedulePolicy, Options_...>;

  MUTLASS_HOST_DEVICE static int workspace_records(ProblemShape const& problem_shape) {
    return Workspace::template record_count<IsVarlen>(problem_shape);
  }

  MUTLASS_HOST_DEVICE static int prepare_upper_bound_works(ProblemShape const& problem_shape) {
    return workspace_records(problem_shape) * problem_shape.H;
  }

  MUTLASS_HOST_DEVICE static int prepare_partition_count(ProblemShape const& problem_shape, int mp_count) {
    int target_ctas = (mp_count > 0 ? mp_count : 1) * SchedulePolicy::PrepareCtasPerMp;
    int total_works = prepare_upper_bound_works(problem_shape);
    int partitions  = total_works < target_ctas ? total_works : target_ctas;
    return partitions > 0 ? partitions : 1;
  }
};

}  // namespace mate::flat::kda
