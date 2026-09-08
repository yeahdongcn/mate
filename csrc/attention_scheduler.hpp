#pragma once

#include <musa_runtime.h>

#include <cstddef>

#include "mate/attention/flash_mla/mpxx_params.hpp"

void run_get_mla_metadata_kernel(mate::flash_mla::GetDecodingMetadataParams& params,
                                 musaStream_t                                stream,
                                 size_t                                      max_shared_memory_per_block);
