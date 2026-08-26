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

// System include(s).
#include <cstdint>

namespace traccc::device {

/// (Global Event Data) Payload for the @c traccc::device::gbts_fit_segments
/// function
struct gbts_fit_segments_payload {
  /// Number of path-store rows
  unsigned int nRows;
  /// Maximum number of neighbours retained per edge
  unsigned int max_num_neighbours;
  /// Minimum number of edges a path must have to be fit
  unsigned char minLevel;
  /// Reduced (x, y, z, r) per original spacepoint
  vecmem::data::vector_view<const float4> reducedSP;
  /// Compacted graph from gbts_compress_graph
  vecmem::data::vector_view<const unsigned int> output_graph;
  /// Per-row (edge index, parent row or -1) entries
  vecmem::data::vector_view<const int2> path_store;
  /// Output: per-row (quality, row) for accepted paths; rows that do not
  /// become a proposal keep their pre-filled {-1, -1}
  vecmem::data::vector_view<int2> seed_proposals;
  /// In/out: global atomic count of accepted seed proposals
  unsigned int* nPropsCounter;
  /// Curvature / pT / chi-squared cut parameters
  traccc::gbts_fit_segments_params gbts_fit_segments_params;
  /// Maximum |z0| at the beamline for extrapolation cuts
  float max_z0;
};

/// @brief Fit each candidate path and emit seed proposals that pass quality
/// cuts.
///
/// One thread per path-store row walks backwards from the row's edge,
/// gathers the involved spacepoints, runs the helix / chi-squared fit, and
/// on success writes the proposal at the row itself. Path-store rows are a
/// deterministic function of the graph, so the row doubles as the
/// reproducible tie-break of every later bid.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload     The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_fit_segments(
    const thread_id_t& thread_id, const gbts_fit_segments_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_fit_segments.ipp"
