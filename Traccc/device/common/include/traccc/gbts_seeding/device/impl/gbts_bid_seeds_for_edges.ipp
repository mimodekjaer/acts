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
#include "traccc/gbts_seeding/device/details/gbts_create_seed_candidate.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_bid_seeds_for_edges(
    const thread_id_t& thread_id,
    const gbts_bid_seeds_for_edges_payload& payload) {
  const vecmem::device_vector<const int2> d_seed_proposals(
      payload.seed_proposals);

  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int blockDimX = thread_id.getBlockDimX();
  const unsigned int gridDimX = thread_id.getGridDimX();

  for (unsigned int prop_idx = globalIdx; prop_idx < payload.nRows;
       prop_idx += blockDimX * gridDimX) {
    const int2 prop = d_seed_proposals[prop_idx];
    if (prop.y < 0) {
      continue;
    }

    // Depth 1: bid for the terminus edge only, as the fitting kernel used to.
    details::gbts_create_seed_candidate(
        prop.x, prop.y, prop_idx, payload.seed_ambiguity,
        payload.seed_proposals, payload.edge_bids, payload.path_store, 1);
  }
}

}  // namespace traccc::device
