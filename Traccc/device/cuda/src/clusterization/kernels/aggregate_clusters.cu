/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "aggregate_clusters.cuh"

// Project include(s).
#include "traccc/clusterization/device/aggregate_cluster.hpp"

namespace traccc::cuda::kernels {

__global__ void aggregate_clusters(
    const clustering_config cfg,
    const edm::silicon_cell_collection::const_view cells_view,
    const detector_design_description::const_view det_descr_view,
    const detector_conditions_description::const_view det_cond_view,
    const vecmem::data::vector_view<const unsigned int> cluster_prefix_view,
    const vecmem::data::vector_view<const unsigned int> next_cell_view,
    edm::measurement_collection::view measurements_view,
    vecmem::data::vector_view<unsigned int> disjoint_set_view,
    vecmem::data::vector_view<unsigned int> cluster_size_view) {
  device::aggregate_clusters(details::global_index1(), cfg, cells_view,
                             det_descr_view, det_cond_view, cluster_prefix_view,
                             next_cell_view, measurements_view,
                             disjoint_set_view, cluster_size_view);
}

}  // namespace traccc::cuda::kernels
