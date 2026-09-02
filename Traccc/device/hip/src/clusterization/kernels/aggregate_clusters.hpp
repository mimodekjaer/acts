/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/geometry/detector_conditions_description.hpp"
#include "traccc/geometry/detector_design_description.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::hip::kernels {

/// HIP kernel for running @c traccc::device::aggregate_clusters
__global__ void aggregate_clusters(
    const clustering_config cfg,
    const edm::silicon_cell_collection::const_view cells_view,
    const detector_design_description::const_view det_descr_view,
    const detector_conditions_description::const_view det_cond_view,
    const vecmem::data::vector_view<const unsigned int> cluster_prefix_view,
    const vecmem::data::vector_view<const unsigned int> next_cell_view,
    edm::measurement_collection::view measurements_view,
    vecmem::data::vector_view<unsigned int> disjoint_set_view,
    vecmem::data::vector_view<unsigned int> cluster_size_view);

}  // namespace traccc::hip::kernels
