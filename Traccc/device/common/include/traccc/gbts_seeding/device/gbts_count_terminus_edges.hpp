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

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_count_terminus_edges function
struct gbts_count_terminus_edges_payload {
  /// Number of edges in the compacted graph
  unsigned int nConnectedEdges;
  /// Per-edge (subtree row count, terminus flag) from CCA
  vecmem::data::vector_view<const int2> outgoing_paths;
  /// Output: per-edge number of path-store rows owned by the edge (1 + its
  /// subtree row count for a terminus edge, 0 otherwise). The kernel
  /// launcher turns this into an inclusive prefix sum in place, so the rows
  /// of terminus edge e are [row_sizes[e] - (1 + subtree), row_sizes[e]) and
  /// the last entry is the total path-store size.
  vecmem::data::vector_view<unsigned int> row_sizes;
  /// In/out: global atomic count of terminus edges
  unsigned int* nTerminusEdgesCounter;
};

/// @brief Count terminus edges and lay out the path store.
///
/// Each thread inspects one edge; a terminus edge (a settled edge long
/// enough to seed) claims 1 + subtree rows of the path store, everything
/// else claims none. The launcher's prefix sum then gives every terminus a
/// contiguous, deterministic row range that gbts_fill_path_store fills.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_count_terminus_edges(
    const thread_id_t& thread_id,
    const gbts_count_terminus_edges_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_count_terminus_edges.ipp"
