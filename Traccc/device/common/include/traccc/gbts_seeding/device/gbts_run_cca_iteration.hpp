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

// System include(s).
#include <cstdint>

namespace traccc::device {

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_run_cca_iteration function
/// Size of the fused-CCA scratch (gbts_run_cca_iteration_payload::
/// active_counters): per-iteration active counts, the row count, and
/// per-block partial sums of the row-size scan
inline constexpr unsigned int gbts_run_cca_max_blocks = 4096u;
inline constexpr unsigned int gbts_run_cca_row_count_slot =
    gbts_consts::max_cca_iter + 2u;
inline constexpr unsigned int gbts_run_cca_scratch_size =
    gbts_run_cca_row_count_slot + 1u + gbts_run_cca_max_blocks;

struct gbts_run_cca_iteration_payload {
  /// Capacity of the compacted graph (maximum number of edges)
  unsigned int nConnectedEdges;
  /// Device-side number of edges in the compacted graph (clamped to
  /// nConnectedEdges by the kernels)
  const unsigned int* d_nConnectedEdges;
  /// Maximum number of neighbours retained per edge
  unsigned int max_num_neighbours;
  /// Minimum path length required for an edge to be considered active
  unsigned char minLevel;
  /// Compacted graph from gbts_compress_graph
  vecmem::data::vector_view<const unsigned int> output_graph;
  /// In/out: per-edge level ping-pong buffer (2 * nConnectedEdges bytes)
  vecmem::data::vector_view<unsigned char> levels;
  /// In/out: per-edge active-flag (holds the next iter index, or -1
  /// once the edge is no longer active).
  vecmem::data::vector_view<char> active_edges;
  /// Output: per-edge (number of path-store rows below the edge, terminus
  /// flag: 0 = candidate seed root, -1 = not a root)
  vecmem::data::vector_view<int2> outgoing_paths;
  /// Iteration index (0-based)
  unsigned char iter;
  /// Scratch for fused implementations: gbts_consts::max_cca_iter + 1
  /// counters of the edges still active after each iteration (unused by the
  /// per-iteration function)
  unsigned int* active_counters;
  /// Output of fused implementations: number of edges that did not fit the
  /// resident cooperative grid and were dropped (deferred warning)
  unsigned int* dropped_counter;
};

/// @brief One iteration of the cellular-automaton "longest path" relaxation.
///
/// Threads cooperatively process the current active-edge list, propagate
/// levels along the compact graph, and write the next iteration's active set
/// into the opposite ping-pong buffer (selected by iter parity).  The
/// final block to finish records the longest outgoing path summary per edge.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_run_cca_iteration(
    const thread_id_t& thread_id,
    const gbts_run_cca_iteration_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_run_cca_iteration.ipp"
