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
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Global Event Data) Payload for the @c traccc::device::gbts_link_graph_edges
/// function
struct gbts_link_graph_edges_payload {
  /// Number of edges produced earlier
  unsigned int nEdges;
  /// (src, dst) node indices per edge
  vecmem::data::vector_view<const uint2> edge_nodes;
  /// Output: per inner-node bucket, the outer node index of every edge
  /// entering that bucket (bucket order is race-assigned; only the
  /// multiset matters, gbts_sort_graph_edges ranks it)
  vecmem::data::vector_view<unsigned int> bucket_outer;
  /// Output: the edge index matching each @c bucket_outer entry
  vecmem::data::vector_view<unsigned int> bucket_edge;
  /// In/out: per-node prefix-sum / write cursor of incoming edges
  vecmem::data::vector_view<unsigned int> num_outgoing_edges;
};

/// @brief Scatter each edge's outer node index into its inner node's bucket.
///
/// One thread per edge atomically decrements the per-inner-node cursor and
/// records the edge's (outer node index, edge index) at the returned slot.  After this
/// kernel the count buffer holds the bucket begin offsets, and
/// gbts_sort_graph_edges can rank every edge inside its bucket.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_link_graph_edges(
    const thread_id_t& thread_id, const gbts_link_graph_edges_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_link_graph_edges.ipp"
