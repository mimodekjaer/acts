/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/geometry/detector_conditions_description.hpp"
#include "traccc/geometry/detector_design_description.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// Function creating one measurement for every cluster
///
/// Runs one thread per cell. Threads of "root" cells (the first cell of a
/// cluster, flagged by @c ccl_kernel) walk the cluster's linked list and
/// compute the cluster properties. The output position of the measurement
/// is given by the inclusive prefix sum of the root flags, so the output
/// order is deterministic (the order of the clusters' first cells).
///
/// @param[in]  globalIndex     The index of the current thread (cell)
/// @param[in]  cfg             The clustering configuration
/// @param[in]  cells_view      Collection of cells
/// @param[in]  det_descr_view  Detector description
/// @param[in]  det_cond_view   Detector conditions
/// @param[in]  cluster_prefix_view Inclusive prefix sums of the root flags
/// @param[in]  next_cell_view  Cluster linked lists (from @c ccl_kernel)
/// @param[out] measurements_view Collection of measurements (its size must
///                             already be set to the number of clusters)
/// @param[out] disjoint_set_view Optional (may be empty): the measurement
///                             index for every cell
/// @param[out] cluster_size_view Optional (may be empty): the number of
///                             cells for every measurement
///
TRACCC_HOST_DEVICE inline void aggregate_clusters(
    global_index_t globalIndex, const clustering_config& cfg,
    const edm::silicon_cell_collection::const_view& cells_view,
    const detector_design_description::const_view& det_descr_view,
    const detector_conditions_description::const_view& det_cond_view,
    const vecmem::data::vector_view<const unsigned int>& cluster_prefix_view,
    const vecmem::data::vector_view<const unsigned int>& next_cell_view,
    edm::measurement_collection::view measurements_view,
    vecmem::data::vector_view<unsigned int> disjoint_set_view,
    vecmem::data::vector_view<unsigned int> cluster_size_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/aggregate_cluster.ipp"
