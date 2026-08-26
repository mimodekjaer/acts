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

/// (Global Event Data) Payload for the @c traccc::device::gbts_fill_path_store
/// function
struct gbts_fill_path_store_payload {
  /// Number of path-store rows
  unsigned int nRows;
  /// Number of edges in the compacted graph
  unsigned int nConnectedEdges;
  /// Maximum number of neighbours retained per edge
  unsigned int max_num_neighbours;
  /// Output: per-row (edge index, parent row or -1) entries
  vecmem::data::vector_view<int2> path_store;
  /// Compacted graph (read for per-edge neighbour lookup)
  vecmem::data::vector_view<const unsigned int> output_graph;
  /// Per-edge CCA level array
  vecmem::data::vector_view<const unsigned char> levels;
  /// Per-edge (subtree row count, terminus flag) from CCA
  vecmem::data::vector_view<const int2> outgoing_paths;
  /// Inclusive prefix sum of the per-edge row counts (see
  /// gbts_count_terminus_edges_payload)
  vecmem::data::vector_view<const unsigned int> row_sizes;
};

/// @brief Enumerate every path below every terminus edge into the path
/// store.
///
/// The rows of a terminus edge hold its subtree in preorder: the edge
/// itself, then for each child (a neighbour exactly one level down, in
/// neighbour-list order) the child's own subtree, whose size CCA already
/// computed. One thread per row locates its terminus by binary search on
/// the row prefix sums and descends the preorder layout to the edge sitting
/// at its row, writing (edge, parent row). The row of an edge is therefore
/// a pure function of the graph: no atomics, no shared frontier, no
/// truncation, and full parallelism over rows.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_fill_path_store(
    const thread_id_t& thread_id, const gbts_fill_path_store_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_fill_path_store.ipp"
