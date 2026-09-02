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
#include "traccc/definitions/hints.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function which reads raw detector cells and turns them into clusters
///
/// Every block of threads processes one partition of the cell collection. The
/// partitions are found with a parallel search (see the implementation), and
/// tile the cell collection without gaps or overlaps. Within a partition a
/// FastSV algorithm labels the cells, and the result is written out as
///
///  - @c cluster_flags: 1 for the first ("root") cell of every cluster, 0
///    otherwise. An inclusive prefix sum of this array gives every cluster a
///    deterministic output position (see @c aggregate_clusters);
///  - @c next_cell: for every cell the index of the next cell of the same
///    cluster (in cell order), or @c details::INVALID_CELL for the last one.
///
/// @param[in] cfg clustering configuration
/// @param[in] thread_id a thread identifier object
/// @param[in] cells_view collection of cells
/// @param[in] partition_start shared memory variable for the partition start
/// @param[in] partition_end shared memory variable for the partition end
/// @param[in] f_view array of "parent" indices for all cells in this
///                   partition (shared memory)
/// @param[in] gf_view array of "grandparent" indices for all cells in this
///                    partition (shared memory)
/// @param[in] f_backup_view global memory alternative to @c f_view
/// @param[in] gf_backup_view global memory alternative to @c gf_view
/// @param[in] adjc_backup_view global memory alternative to the adjacency
///                             counts
/// @param[in] adjv_backup_view global memory alternative to the adjacency
///                             vectors
/// @param[in] backup_mutex mutex protecting the global memory backups
/// @param[in] barrier a barrier object for block synchronisation
/// @param[out] cluster_flags_view root cell flags (one per cell)
/// @param[out] next_cell_view cluster linked lists (one entry per cell)
///
template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void ccl_kernel(
    const clustering_config cfg, const thread_id_t& thread_id,
    const edm::silicon_cell_collection::const_view& cells_view,
    unsigned int& partition_start, unsigned int& partition_end,
    vecmem::data::vector_view<details::index_t> f_view,
    vecmem::data::vector_view<details::index_t> gf_view,
    vecmem::data::vector_view<details::fallback_index_t> f_backup_view,
    vecmem::data::vector_view<details::fallback_index_t> gf_backup_view,
    vecmem::data::vector_view<unsigned char> adjc_backup_view,
    vecmem::data::vector_view<details::fallback_index_t> adjv_backup_view,
    vecmem::device_atomic_ref<uint32_t> backup_mutex, const barrier_t& barrier,
    vecmem::data::vector_view<unsigned int> cluster_flags_view,
    vecmem::data::vector_view<unsigned int> next_cell_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/ccl_kernel.ipp"
