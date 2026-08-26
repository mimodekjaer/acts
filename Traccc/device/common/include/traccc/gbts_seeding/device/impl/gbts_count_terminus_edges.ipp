/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_count_terminus_edges(
    const thread_id_t& thread_id,
    const gbts_count_terminus_edges_payload& payload) {
  const vecmem::device_vector<const int2> d_outgoing_paths(
      payload.outgoing_paths);
  vecmem::device_vector<unsigned int> d_row_sizes(payload.row_sizes);

  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int blockDimX = thread_id.getBlockDimX();
  const unsigned int gridDimX = thread_id.getGridDimX();

  for (unsigned int globalIndex = globalIdx;
       globalIndex < payload.nConnectedEdges;
       globalIndex += blockDimX * gridDimX) {
    const int2 out_paths = d_outgoing_paths[globalIndex];
    if (out_paths.y == -1) {
      d_row_sizes[globalIndex] = 0u;
      continue;
    }
    d_row_sizes[globalIndex] = 1u + static_cast<unsigned int>(out_paths.x);
    vecmem::device_atomic_ref<unsigned int>(*payload.nTerminusEdgesCounter)
        .fetch_add(1u);
  }
}

}  // namespace traccc::device
