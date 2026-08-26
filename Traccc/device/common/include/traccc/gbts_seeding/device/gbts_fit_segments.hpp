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

/// The Kalman-filter helpers of the segment fit live in the implementation
/// file (details::edgeState, details::gbts_kalman_update); the fit itself is
/// run by gbts_fill_path_store on each path-store row.

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_fit_segments.ipp"
