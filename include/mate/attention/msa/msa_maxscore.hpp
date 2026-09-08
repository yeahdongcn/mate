#pragma once

#include "mate/attention/msa/collective/mp31_msa_maxscore_collective_tme_warpspecialized.hpp"
#include "mate/attention/msa/kernel/msa_maxscore_kernel_tme_warpspecialized.hpp"
#include "mate/attention/msa/kernel/msa_maxscore_metadata_kernel.hpp"
#include "mate/attention/msa/kernel/msa_maxscore_tile_scheduler.hpp"
#include "mate/attention/msa/msa_options.hpp"

namespace mate::attention::msa {

template <class Element_, class TileShape_, int HeadRatio_, class... Options_>
struct MsaMaxScoreBuilder {
  using Element                        = Element_;
  using TileShape                      = TileShape_;
  static constexpr int  HeadRatio      = HeadRatio_;
  static constexpr bool ParallelKTiles = find_option_t<Tag::ParallelKTiles, std::false_type, Options_...>::value;
  static constexpr bool IsVarlen       = find_option_t<Tag::IsVarlen, std::true_type, Options_...>::value;
  static constexpr bool HasMetadata    = find_option_t<Tag::HasMetadata, std::false_type, Options_...>::value;
  using CollectiveMainloop =
      collective::Mp31MsaMaxScoreCollectiveTmeWarpSpecialized<Element, TileShape, HeadRatio, Options_...>;
  using TileScheduler = MsaMaxScoreTileScheduler<TileShape, HeadRatio, ParallelKTiles, IsVarlen, HasMetadata>;
  using Kernel        = MsaMaxScoreKernelTmeWarpSpecialized<CollectiveMainloop, TileScheduler>;
};

}  // namespace mate::attention::msa
