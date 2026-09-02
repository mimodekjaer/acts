/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/edm/measurement_collection.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

/// Key type used for sorting measurements by their surface identifier
using measurement_surface_key_t = std::uint64_t;
/// Key type used for the deterministic ordering of measurements within a
/// surface (the index of the first cell of the cluster)
using measurement_cluster_key_t = unsigned int;

/// Comparison functor giving measurements a deterministic total order
///
/// Measurements are ordered by their surface identifier first, and by their
/// (unique, deterministic) cluster key second. Since no two measurements share
/// both keys, any correct sorting algorithm produces the same permutation.
///
class measurement_order_sorter {
 public:
  /// Constructor, capturing the (unsorted) measurements
  explicit measurement_order_sorter(
      const edm::measurement_collection::const_view& measurements)
      : m_measurements(measurements) {}

  /// Index comparison operator
  TRACCC_HOST_DEVICE bool operator()(unsigned int lhs, unsigned int rhs) const {
    const edm::measurement_collection::const_device measurements{
        m_measurements};
    const auto lhs_surface = measurements.surface_link().at(lhs);
    const auto rhs_surface = measurements.surface_link().at(rhs);
    if (lhs_surface != rhs_surface) {
      return lhs_surface < rhs_surface;
    }
    return measurements.identifier().at(lhs) < measurements.identifier().at(rhs);
  }

 private:
  /// The (unsorted) measurements
  edm::measurement_collection::const_view m_measurements;

};  // class measurement_order_sorter

/// Fill the primary sort keys and the identity index sequence
///
/// The primary key of a measurement is the (unique, deterministic) identifier
/// that the clusterization assigned to it: the index of the first cell of its
/// cluster in the (sorted) cell collection. Sorting by this key first, and
/// then stably by the surface identifier, produces a fully deterministic
/// measurement order, independent of the (non-deterministic) order in which
/// the clusterization kernel wrote the measurements.
///
/// @param[in]  globalIndex  The index of the current thread
/// @param[in]  measurements The (unsorted) measurements
/// @param[out] keys         The cluster keys of the measurements
/// @param[out] indices      The identity index sequence
///
TRACCC_HOST_DEVICE inline void fill_measurement_cluster_keys(
    global_index_t globalIndex,
    const edm::measurement_collection::const_view& measurements,
    vecmem::data::vector_view<measurement_cluster_key_t> keys,
    vecmem::data::vector_view<unsigned int> indices);

/// Gather the surface identifier keys of the measurements in the order given
/// by the (partially sorted) index sequence
///
/// @param[in]  globalIndex  The index of the current thread
/// @param[in]  measurements The (unsorted) measurements
/// @param[in]  indices      The (partially sorted) measurement indices
/// @param[out] keys         The surface keys, in the order of @c indices
///
TRACCC_HOST_DEVICE inline void gather_measurement_surface_keys(
    global_index_t globalIndex,
    const edm::measurement_collection::const_view& measurements,
    const vecmem::data::vector_view<const unsigned int>& indices,
    vecmem::data::vector_view<measurement_surface_key_t> keys);

/// Fill the output collection with the sorted measurements
///
/// The identifier and cluster index of every output measurement is set to its
/// position in the sorted collection, mirroring the host algorithms.
///
/// @param[in]  globalIndex    The index of the current thread
/// @param[in]  input          The unsorted measurements
/// @param[out] output         The sorted measurements
/// @param[in]  sorted_indices The sorted measurement indices
///
TRACCC_HOST_DEVICE inline void fill_sorted_measurements(
    global_index_t globalIndex,
    const edm::measurement_collection::const_view& input,
    edm::measurement_collection::view output,
    const vecmem::data::vector_view<const unsigned int>& sorted_indices);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/measurement_sorting.ipp"
