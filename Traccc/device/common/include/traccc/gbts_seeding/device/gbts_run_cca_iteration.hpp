/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

/// Scratch of the CCA sweeps (gbts_run_cca_iteration_payload::counters):
/// [0, max_cca_iter] the number of blocks that changed something in sweep i,
/// [gbts_run_cca_row_count_slot] the path-store row count published by the
/// terminus step. Must be zero at the start of the event.
inline constexpr unsigned int gbts_run_cca_row_count_slot =
    gbts_consts::max_cca_iter + 1u;
/// Value of gbts_run_cca_iteration_payload::iter selecting the finishing
/// pass
inline constexpr unsigned char gbts_run_cca_finish_pass = 0xFFu;
inline constexpr unsigned int gbts_run_cca_scratch_size =
    gbts_run_cca_row_count_slot + 1u;

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_run_cca_iteration function
struct gbts_run_cca_iteration_payload {
  /// Capacity of the compacted graph (maximum number of edges)
  unsigned int nConnectedEdges;
  /// Device-side number of edges in the compacted graph (clamped to
  /// nConnectedEdges by the kernels)
  const unsigned int* d_nConnectedEdges;
  /// Maximum number of neighbours retained per edge
  unsigned int max_num_neighbours;
  /// Minimum level (path length in edges) of a seed root
  unsigned char minLevel;
  /// Compacted graph from gbts_compress_graph
  vecmem::data::vector_view<const unsigned int> output_graph;
  /// In/out: per-edge level (longest path in edges, capped at
  /// max_cca_iter + 1 which marks an edge that does not settle);
  /// initialised to 1 by gbts_compress_graph
  vecmem::data::vector_view<unsigned char> levels;
  /// Output: per-edge (number of path-store rows below the edge, terminus
  /// flag: 0 = candidate seed root, -1 = not a root because the path is too
  /// short or the edge did not settle)
  vecmem::data::vector_view<int2> outgoing_paths;
  /// Output: per-edge "has a settled parent" mark (idempotent 1-writes),
  /// zero-initialised by gbts_compress_graph. An edge is a path root iff
  /// its terminus flag is 0 AND it has no parent mark.
  vecmem::data::vector_view<unsigned char> has_parent;
  /// Launch index: [0, max_cca_iter] are the relaxation sweeps (a sweep
  /// after a sweep without changes returns immediately),
  /// gbts_run_cca_finish_pass is the finishing pass (parent marks, terminus
  /// flags)
  unsigned char iter;
  /// Scratch, see gbts_run_cca_scratch_size (zeroed per event)
  unsigned int* counters;
  /// Per-edge (neighbour count, first three neighbours) from
  /// gbts_compress_graph
  vecmem::data::vector_view<const uint4> nei_cache;
};

/// (Shared Event Data) Payload for the @c
/// traccc::device::gbts_run_cca_iteration function
struct gbts_run_cca_iteration_shared_payload {
  /// One unsigned int: the block's change flag
  vecmem::data::vector_view<unsigned int> changed;
};

/// @brief One sweep of the deterministic longest-path relaxation (the
/// "CCA"), or its finishing pass.
///
/// Sweeps 0 .. max_cca_iter: every edge recomputes, in place, its level
/// (1 + the maximum level of its neighbours, capped at max_cca_iter + 1 =
/// "does not settle") and its subtree row count (the sum of 1 + count over
/// the neighbours one level below). A sweep after a sweep without any
/// change returns at once. The fixed point is unique - the longest path
/// lengths and the counts they imply - so the result does not depend on the
/// schedule; the levels of level L settle after sweep L - 1 and the counts
/// one sweep later, so max_cca_iter + 1 sweeps always suffice.
/// Finishing pass (iter == gbts_run_cca_finish_pass): settled edges
/// (level <= max_cca_iter) mark their neighbours as having a parent and get
/// their terminus flag; edges that do not settle are excluded.
///
/// @param[in] thread_id      Thread identifier for the kernel launch
/// @param[in] barrier        Block-wide barrier
/// @param[in] payload        The global memory payload
/// @param[in] shared_payload The shared memory payload
///
template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_run_cca_iteration(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_run_cca_iteration_payload& payload,
    const gbts_run_cca_iteration_shared_payload& shared_payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_run_cca_iteration.ipp"
