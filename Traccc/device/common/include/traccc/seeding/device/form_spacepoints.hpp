/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/geometry/detector.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// Function flagging the measurements that can be turned into spacepoints
///
/// The flags are turned into (inclusive) prefix sums by the algorithm, which
/// gives every spacepoint a deterministic position in the output collection.
///
/// @param[in]  globalIndex  The index of the current thread
/// @param[in]  measurements All measurements in an event
/// @param[out] flags        1 for measurements producing a spacepoint, 0
///                          otherwise
///
TRACCC_HOST_DEVICE inline void flag_spacepoint_measurements(
    global_index_t globalIndex,
    const edm::measurement_collection::const_view& measurements,
    vecmem::data::vector_view<unsigned int> flags);

/// Function for creating 3D spacepoints out of 2D measurements
///
/// @param[in]  globalIndex   The index of the current thread
/// @param[in]  det_view      The view of the detector
/// @param[in]  measurements  All measurements in an event
/// @param[in]  offsets       Inclusive prefix sums of the measurement flags
/// @param[out] spacepoints   All spacepoints in the event
///
template <typename detector_t>
TRACCC_HOST_DEVICE inline void form_spacepoints(
    global_index_t globalIndex, typename detector_t::view det_view,
    const edm::measurement_collection::const_view& measurements,
    const vecmem::data::vector_view<const unsigned int>& offsets,
    edm::spacepoint_collection::view spacepoints);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/form_spacepoints.ipp"
