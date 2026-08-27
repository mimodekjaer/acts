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
  const vecmem::device_vector<const unsigned char> d_has_parent(
      payload.has_parent);
  vecmem::device_vector<unsigned int> d_row_sizes(payload.row_sizes);
  vecmem::device_vector<unsigned long long int> d_edge_bids(payload.edge_bids);
  vecmem::device_vector<unsigned long long int> d_hit_bids(payload.hit_bids);
  vecmem::device_vector<char> d_seed_ambiguity(payload.seed_ambiguity);

  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int blockDimX = thread_id.getBlockDimX();
  const unsigned int gridDimX = thread_id.getGridDimX();

  const unsigned int nConnectedEdges =
      (*payload.d_nConnectedEdges < payload.nConnectedEdges)
          ? *payload.d_nConnectedEdges
          : payload.nConnectedEdges;
  // Only the bids of the edges present are used (halves laid out with the
  // capacity as stride).
  for (unsigned int globalIndex = globalIdx; globalIndex < nConnectedEdges;
       globalIndex += blockDimX * gridDimX) {
    d_edge_bids[globalIndex] = 0ull;
    d_edge_bids[payload.nConnectedEdges + globalIndex] = 0ull;
  }
  for (unsigned int globalIndex = globalIdx; globalIndex < d_hit_bids.size();
       globalIndex += blockDimX * gridDimX) {
    d_hit_bids[globalIndex] = 0ull;
  }
  for (unsigned int globalIndex = globalIdx; globalIndex < payload.nRows;
       globalIndex += blockDimX * gridDimX) {
    d_seed_ambiguity[globalIndex] = 0;
  }

  // Row sizes beyond the edge count stay zero so the scan over the whole
  // capacity ends with the total row count.
  for (unsigned int globalIndex = globalIdx + nConnectedEdges;
       globalIndex < payload.nConnectedEdges;
       globalIndex += blockDimX * gridDimX) {
    d_row_sizes[globalIndex] = 0u;
  }
  for (unsigned int globalIndex = globalIdx; globalIndex < nConnectedEdges;
       globalIndex += blockDimX * gridDimX) {
    const int2 out_paths = d_outgoing_paths[globalIndex];
    if ((out_paths.y == -1) || (d_has_parent[globalIndex] != 0u)) {
      d_row_sizes[globalIndex] = 0u;
      continue;
    }
    d_row_sizes[globalIndex] = 1u + static_cast<unsigned int>(out_paths.x);
  }
}

}  // namespace traccc::device
