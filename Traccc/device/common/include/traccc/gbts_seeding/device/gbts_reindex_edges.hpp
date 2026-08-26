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
  /// Capacity of the edge buffers (length of the scan; the flags beyond the
  /// edge count are zero, so the last scan entry is the kept-edge count)
  unsigned int nEdgesMax;
  /// Per-edge "kept" flag (0/1)
  vecmem::data::vector_view<const unsigned char> kept;
  /// Output: inclusive prefix sum of the kept flags (written by the kernel
  /// launcher); the compact index of a kept edge e is reIndexer[e] - 1, and
  /// the last entry is the total.
  vecmem::data::vector_view<int> reIndexer;
};

/// @brief Turn the per-edge "kept" flags into compact indices.
///
/// Implemented entirely by the backend launcher as an in-place inclusive
/// scan, so the compact order is the canonical edge order restricted to the
/// kept edges -- deterministic and locality preserving.

}  // namespace traccc::device
