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
    return measurements.identifier().at(lhs) <
           measurements.identifier().at(rhs);
  }

 private:
  /// The (unsorted) measurements
  edm::measurement_collection::const_view m_measurements;

};  // class measurement_order_sorter

/// Fill the surface identifier sort keys and the identity index sequence
///
/// The clusterization writes the measurements in a deterministic order (that
/// of the first cells of their clusters), so a stable sort by the surface
/// identifier alone yields a deterministic measurement order.
///
/// @param[in]  globalIndex  The index of the current thread
/// @param[in]  measurements The (unsorted) measurements
/// @param[out] keys         The surface keys of the measurements
/// @param[out] indices      The identity index sequence
///
TRACCC_HOST_DEVICE inline void fill_measurement_surface_keys(
    global_index_t globalIndex,
    const edm::measurement_collection::const_view& measurements,
    vecmem::data::vector_view<measurement_surface_key_t> keys,
    vecmem::data::vector_view<unsigned int> indices);

/// Flag the measurement collection if it is not sorted by surface identifier
///
/// Measurements are frequently already sorted by surface (e.g. when the cells
/// were read grouped by surface), in which case the sorting can be replaced
/// by a plain copy.
///
/// @param[in]  globalIndex  The index of the current thread
/// @param[in]  measurements The measurements to check
/// @param[out] unsorted     Set to 1 if any two neighbouring measurements are
///                          out of order (must be initialised to 0)
///
TRACCC_HOST_DEVICE inline void flag_unsorted_measurements(
    global_index_t globalIndex,
    const edm::measurement_collection::const_view& measurements,
    vecmem::data::vector_view<unsigned int> unsorted);

/// Copy the (already sorted) measurements to the output collection
///
/// The identifier and cluster index of every output measurement is set to its
/// position, exactly like @c fill_sorted_measurements does.
///
/// @param[in]  globalIndex The index of the current thread
/// @param[in]  input       The input measurements
/// @param[out] output      The output measurements
///
TRACCC_HOST_DEVICE inline void copy_measurements(
    global_index_t globalIndex,
    const edm::measurement_collection::const_view& input,
    edm::measurement_collection::view output);

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
