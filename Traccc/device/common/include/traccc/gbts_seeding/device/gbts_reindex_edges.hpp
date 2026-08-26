/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

/// (Global Event Data) Payload for the @c traccc::device::gbts_reindex_edges
/// function
struct gbts_reindex_edges_payload {
  /// Capacity of the edge buffers (length of the scan)
  unsigned int nEdgesMax;
  /// Number of original edges, on the device
  const unsigned int* nEdges;
  /// Per-edge "kept" flag (0/1)
  vecmem::data::vector_view<const unsigned char> kept;
  /// Output: inclusive prefix sum of the kept flags (written by the kernel
  /// launcher); the compact index of a kept edge e is reIndexer[e] - 1, and
  /// entry nEdges - 1 is the total.
  vecmem::data::vector_view<int> reIndexer;
  /// Output: number of kept edges (reIndexer[nEdges - 1])
  unsigned int* nConnectedEdges;
};

/// @brief Store the number of kept edges after the scan (one thread).
TRACCC_HOST_DEVICE inline void gbts_reindex_edges_finish(
    const gbts_reindex_edges_payload& payload) {
  const vecmem::device_vector<const int> d_reIndexer(payload.reIndexer);
  const unsigned int nEdges = *payload.nEdges;
  *payload.nConnectedEdges =
      (nEdges > 0u) ? static_cast<unsigned int>(d_reIndexer[nEdges - 1u]) : 0u;
}

/// @brief Turn the per-edge "kept" flags into compact indices.
///
/// Implemented entirely by the backend launcher as an in-place inclusive
/// scan, so the compact order is the canonical edge order restricted to the
/// kept edges -- deterministic and locality preserving.

}  // namespace traccc::device
