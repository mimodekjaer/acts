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
TRACCC_HOST_DEVICE inline void gbts_link_graph_edges(
    const thread_id_t& thread_id,
    const gbts_link_graph_edges_payload& payload) {
  const vecmem::device_vector<const uint2> d_edge_nodes(payload.edge_nodes);
  vecmem::device_vector<unsigned int> d_bucket_outer(payload.bucket_outer);
  vecmem::device_vector<unsigned int> d_bucket_edge(payload.bucket_edge);
  vecmem::device_vector<unsigned int> d_num_outgoing_edges(
      payload.num_outgoing_edges);

  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int blockDimX = thread_id.getBlockDimX();
  const unsigned int gridDimX = thread_id.getGridDimX();

  for (unsigned int globalIndex = globalIdx; globalIndex < payload.nEdges;
       globalIndex += blockDimX * gridDimX) {
    const uint2 nodes = d_edge_nodes[globalIndex];
    const unsigned int pos = vecmem::device_atomic_ref<unsigned int>(
                                 d_num_outgoing_edges[nodes.y])
                                 .fetch_sub(1u);
    d_bucket_outer[pos - 1u] = nodes.x;
    d_bucket_edge[pos - 1u] = globalIndex;
  }
}

}  // namespace traccc::device
