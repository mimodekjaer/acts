/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void fill_measurement_cluster_keys(
    const global_index_t globalIndex,
    const edm::measurement_collection::const_view& measurements_view,
    vecmem::data::vector_view<measurement_cluster_key_t> keys_view,
    vecmem::data::vector_view<unsigned int> indices_view) {
  const edm::measurement_collection::const_device measurements{
      measurements_view};
  if (globalIndex >= measurements.size()) {
    return;
  }
  vecmem::device_vector<measurement_cluster_key_t> keys{keys_view};
  vecmem::device_vector<unsigned int> indices{indices_view};
  keys.at(globalIndex) = measurements.identifier().at(globalIndex);
  indices.at(globalIndex) = globalIndex;
}

TRACCC_HOST_DEVICE inline void gather_measurement_surface_keys(
    const global_index_t globalIndex,
    const edm::measurement_collection::const_view& measurements_view,
    const vecmem::data::vector_view<const unsigned int>& indices_view,
    vecmem::data::vector_view<measurement_surface_key_t> keys_view) {
  const edm::measurement_collection::const_device measurements{
      measurements_view};
  if (globalIndex >= measurements.size()) {
    return;
  }
  const vecmem::device_vector<const unsigned int> indices{indices_view};
  vecmem::device_vector<measurement_surface_key_t> keys{keys_view};
  static_assert(sizeof(measurement_surface_key_t) >=
                sizeof(detray::geometry::identifier::value_t));
  keys.at(globalIndex) = static_cast<measurement_surface_key_t>(
      measurements.surface_link().at(indices.at(globalIndex)).value());
}

TRACCC_HOST_DEVICE inline void fill_sorted_measurements(
    const global_index_t globalIndex,
    const edm::measurement_collection::const_view& input_view,
    edm::measurement_collection::view output_view,
    const vecmem::data::vector_view<const unsigned int>& sorted_indices_view) {
  const edm::measurement_collection::const_device input{input_view};
  if (globalIndex >= input.size()) {
    return;
  }
  edm::measurement_collection::device output{output_view};
  const vecmem::device_vector<const unsigned int> sorted_indices{
      sorted_indices_view};
  auto out = output.at(globalIndex);
  out = input.at(sorted_indices.at(globalIndex));
  out.identifier() = globalIndex;
  out.cluster_index() = globalIndex;
}

}  // namespace traccc::device
