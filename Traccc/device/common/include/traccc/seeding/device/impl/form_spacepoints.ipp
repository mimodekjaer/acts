/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/detail/spacepoint_formation.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

// System include(s).
#include <cassert>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void flag_spacepoint_measurements(
    const global_index_t globalIndex,
    const edm::measurement_collection::const_view& measurements_view,
    vecmem::data::vector_view<unsigned int> flags_view) {
  const edm::measurement_collection::const_device measurements(
      measurements_view);
  if (globalIndex >= measurements.size()) {
    return;
  }
  vecmem::device_vector<unsigned int> flags(flags_view);
  flags.at(globalIndex) =
      traccc::details::is_valid_measurement(measurements.at(globalIndex)) ? 1u
                                                                          : 0u;
}

template <typename detector_t>
TRACCC_HOST_DEVICE inline void form_spacepoints(
    const global_index_t globalIndex, typename detector_t::view det_view,
    const edm::measurement_collection::const_view& measurements_view,
    const vecmem::data::vector_view<const unsigned int>& offsets_view,
    edm::spacepoint_collection::view spacepoints_view) {
  // Set up the input container(s).
  const edm::measurement_collection::const_device measurements(
      measurements_view);

  // Check if anything needs to be done
  if (globalIndex >= measurements.size()) {
    return;
  }

  const edm::measurement meas = measurements.at(globalIndex);
  if (!traccc::details::is_valid_measurement(meas)) {
    return;
  }

  // Create the tracking geometry
  typename detector_t::device det(det_view);

  // Set up the output container(s).
  edm::spacepoint_collection::device spacepoints(spacepoints_view);
  const vecmem::device_vector<const unsigned int> offsets(offsets_view);

  // The position of the spacepoint is given by the (inclusive) prefix sum of
  // the measurement flags. This makes the output order deterministic.
  const edm::spacepoint_collection::device::size_type i =
      offsets.at(globalIndex) - 1u;
  assert(i < spacepoints.capacity());
  edm::spacepoint sp = spacepoints.at(i);
  traccc::details::fill_pixel_spacepoint(sp, det, meas);
  sp.measurement_index_1() = globalIndex;
  sp.measurement_index_2() =
      edm::spacepoint_collection::device::INVALID_MEASUREMENT_INDEX;
}

}  // namespace traccc::device
