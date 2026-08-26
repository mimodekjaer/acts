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
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_match_graph_edges function
struct gbts_match_graph_edges_payload {
  /// Number of edges to match
  unsigned int nEdges;
  /// Maximum number of neighbours retained per edge
  unsigned int nMaxNei;
  /// Edge-matching pair cuts
  traccc::gbts_match_graph_edges_params gbts_match_graph_edges_params;
  /// Packed per-edge [exp_eta, curv, phi_z, phi_w], in canonical edge
  /// order (from gbts_sort_graph_edges)
  vecmem::data::vector_view<const short4> edge_params;
  /// (outer, inner) node indices per edge, in canonical edge order
  vecmem::data::vector_view<const uint2> edge_nodes;
  /// Per-node [begin, end) canonical edge range of the edges entering the
  /// node (used to locate candidates)
  vecmem::data::vector_view<const unsigned int> num_outgoing_edges;
  /// Output: number of accepted neighbours per edge (0..nMaxNei)
  vecmem::data::vector_view<unsigned char> num_neighbours;
  /// Output: neighbour edge indices, nMaxNei entries per edge (flat)
  vecmem::data::vector_view<unsigned int> neighbours;
  /// Output: per-edge "kept" flag, later compacted into a re-index
  vecmem::data::vector_view<int> reIndexer;
  /// class for decoding short4 edge params to float4, bool
  edge_params_converter edge_params_decoder;
  /// In/out: global atomic counter of total accepted connections
  unsigned int* nConnectionsCounter;
};

/// @brief For each edge, find compatible neighbour edges sharing its outer
/// node.
///
/// One thread per edge pair-tests the edge against every edge entering its
/// outer node (a contiguous canonical index range) using the packed edge
/// parameters, recording up to nMaxNei accepted neighbours in canonical
/// order, marking the edge as "kept", and atomically incrementing the
/// connection counter.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_match_graph_edges(
    const thread_id_t& thread_id,
    const gbts_match_graph_edges_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_match_graph_edges.ipp"
