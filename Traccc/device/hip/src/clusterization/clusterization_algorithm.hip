/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../sanity/contiguous_on.hpp"
#include "../sanity/ordered_on.hpp"
#include "../utils/hip_error_handling.hpp"
#include "../utils/utils.hpp"
#include "./kernels/aggregate_clusters.hpp"
#include "./kernels/ccl_kernel.hpp"
#include "./kernels/reify_cluster_data.hpp"
#include "traccc/hip/clusterization/clusterization_algorithm.hpp"

// Project include(s).
#include "traccc/utils/projections.hpp"
#include "traccc/utils/relations.hpp"

// HIP include(s).
#include <hip/hip_runtime.h>

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// System include(s).
#include <memory_resource>

// System include(s).
#include <stdexcept>

namespace traccc::hip {

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, const vecmem::copy& copy,
    const stream_wrapper& str, const config_type& config,
    std::unique_ptr<const Logger> logger)
    : device::clusterization_algorithm(mr, copy, config, std::move(logger)),
      hip::algorithm_base(str) {}

bool clusterization_algorithm::input_is_contiguous(
    const edm::silicon_cell_collection::const_view& cells) const {
  return is_contiguous_on<edm::silicon_cell_collection::const_device>(
      cell_module_projection(), mr().main, copy(), stream(), cells);
}

bool clusterization_algorithm::input_is_sorted(
    const edm::silicon_cell_collection::const_view& cells) const {
  return is_ordered_on<edm::silicon_cell_collection::const_device>(
      channel0_major_cell_order_relation(), mr().main, copy(), stream(), cells);
}

void clusterization_algorithm::sort_cells_kernel(
    const unsigned int, const edm::silicon_cell_collection::const_view&,
    edm::silicon_cell_collection::view&,
    vecmem::data::vector_view<unsigned int>&, const bool) const {
  throw std::runtime_error(
      "Cell sorting is not yet implemented for the HIP backend.");
}

void clusterization_algorithm::ccl_kernel(
    const ccl_kernel_payload& payload) const {
  hipStream_t hip_stream = details::get_stream(stream());

  // Run the CCL kernel, one block per partition.
  const unsigned int num_blocks =
      (payload.n_cells + (payload.config.target_partition_size()) - 1) /
      payload.config.target_partition_size();
  hipLaunchKernelGGL(kernels::ccl_kernel, dim3{num_blocks},
                     dim3{payload.config.threads_per_partition},
                     2 * payload.config.max_partition_size() *
                         sizeof(device::details::index_t),
                     hip_stream, payload.config, payload.cells,
                     payload.f_backup, payload.gf_backup, payload.adjc_backup,
                     payload.adjv_backup, payload.backup_mutex,
                     payload.cluster_flags, payload.next_cell);
  TRACCC_HIP_ERROR_CHECK(hipGetLastError());

  // Turn the root flags into inclusive prefix sums, in place.
#if defined(__HIP_PLATFORM_AMD__)
  auto policy =
      thrust::hip::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(hip_stream);
#elif defined(__HIP_PLATFORM_NVIDIA__)
  auto policy =
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(hip_stream);
#else
#error "Unrecognised HIP platform."
#endif
  thrust::inclusive_scan(policy, payload.cluster_flags.ptr(),
                         payload.cluster_flags.ptr() + payload.n_cells,
                         payload.cluster_flags.ptr());

  // The last prefix sum is the number of measurements. Copy it into the size
  // of the output buffer (device-to-device, in stream order).
  copy()(vecmem::data::vector_view<const char>{
             static_cast<vecmem::data::vector_view<const char>::size_type>(
                 sizeof(unsigned int)),
             reinterpret_cast<const char*>(payload.cluster_flags.ptr() +
                                           payload.n_cells - 1u)},
         payload.measurements.size())
      ->ignore();

  // Create the measurements, one thread per cell.
  const unsigned int agg_threads = warp_size() * 8u;
  const unsigned int agg_blocks =
      (payload.n_cells + agg_threads - 1) / agg_threads;
  hipLaunchKernelGGL(kernels::aggregate_clusters, dim3{agg_blocks},
                     dim3{agg_threads}, 0, hip_stream, payload.config,
                     payload.cells, payload.det_descr, payload.det_cond,
                     vecmem::data::vector_view<const unsigned int>(
                         payload.cluster_flags),
                     vecmem::data::vector_view<const unsigned int>(
                         payload.next_cell),
                     vecmem::get_data(payload.measurements),
                     payload.disjoint_set, payload.cluster_sizes);
  TRACCC_HIP_ERROR_CHECK(hipGetLastError());

  // With sorting enabled, the kernels read from sorted cell and permutation
  // map buffers that the base class destroys without any further
  // synchronization, so the kernels must finish before returning.
  if (payload.config.sort_cells) {
    stream().synchronize();
  }
}

void clusterization_algorithm::cluster_maker_kernel(
    unsigned int num_cells,
    const vecmem::data::vector_view<unsigned int>& disjoint_set,
    edm::silicon_cluster_collection::view& cluster_data,
    const vecmem::data::vector_view<const unsigned int>& permutation_map_view)
    const {
  const unsigned int num_threads = warp_size() * 16u;
  const unsigned int num_blocks = (num_cells + num_threads - 1) / num_threads;
  hipLaunchKernelGGL(kernels::reify_cluster_data, num_blocks, num_threads, 0,
                     details::get_stream(stream()), disjoint_set,
                     permutation_map_view, cluster_data);
  TRACCC_HIP_ERROR_CHECK(hipGetLastError());
  // The base class destroys the input buffers right after this call, so
  // the kernel must finish before returning.
  stream().synchronize();
}

}  // namespace traccc::hip
