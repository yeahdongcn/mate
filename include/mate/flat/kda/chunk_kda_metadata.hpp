#pragma once

#include <cstdint>

namespace mate::flat::kda {

// One contiguous compact-work range per persistent prepare CTA.  This record
// is part of the architecture-independent scheduler ABI.
struct alignas(16) ChunkKdaPartitionMetadata {
  int32_t begin_seq;
  int32_t begin_chunk;
  int32_t begin_head;
  int32_t work_count;
};

}  // namespace mate::flat::kda
