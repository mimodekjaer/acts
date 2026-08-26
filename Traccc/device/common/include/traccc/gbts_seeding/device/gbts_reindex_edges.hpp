/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Global Event Data) Payload for the @c traccc::device::gbts_reindex_edges
/// function
struct gbts_reindex_edges_payload {
  /// Number of original edges
  unsigned int nEdges;
  /// In/out: per-edge "kept" flag (0/1) in, inclusive prefix sum out. The
  /// kernel launcher runs the scan in place; the compact index of a kept
  /// edge e is then reIndexer[e] - 1, and the last entry is the total.
  vecmem::data::vector_view<int> reIndexer;
};

/// @brief Turn the per-edge "kept" flags into compact indices.
///
/// Implemented entirely by the backend launcher as an in-place inclusive
/// scan, so the compact order is the canonical edge order restricted to the
/// kept edges -- deterministic and locality preserving.

}  // namespace traccc::device
