/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "../utils/hip_error_handling.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "traccc/hip/clusterization/measurement_sorting_algorithm.hpp"

// Project include(s).
#include "traccc/clusterization/device/measurement_sorting.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// HIP include(s).
#include <hip/hip_runtime.h>

// System include(s).
#include <memory_resource>

namespace traccc::hip {
namespace kernels {

/// Kernel wrapping @c traccc::device::fill_measurement_cluster_keys
__global__ void fill_measurement_cluster_keys(
    const edm::measurement_collection::const_view measurements,
    vecmem::data::vector_view<device::measurement_cluster_key_t> keys,
    vecmem::data::vector_view<unsigned int> indices) {
  device::fill_measurement_cluster_keys(
      details::thread_id1{}.getGlobalThreadId(), measurements, keys, indices);
}

/// Kernel wrapping @c traccc::device::gather_measurement_surface_keys
__global__ void gather_measurement_surface_keys(
    const edm::measurement_collection::const_view measurements,
    const vecmem::data::vector_view<const unsigned int> indices,
    vecmem::data::vector_view<device::measurement_surface_key_t> keys) {
  device::gather_measurement_surface_keys(
      details::thread_id1{}.getGlobalThreadId(), measurements, indices, keys);
}

/// Kernel wrapping @c traccc::device::fill_sorted_measurements
__global__ void fill_sorted_measurements(
    const edm::measurement_collection::const_view input,
    edm::measurement_collection::view output,
    const vecmem::data::vector_view<const unsigned int> sorted_indices) {
  device::fill_sorted_measurements(details::thread_id1{}.getGlobalThreadId(),
                                   input, output, sorted_indices);
}

}  // namespace kernels

measurement_sorting_algorithm::measurement_sorting_algorithm(
    const traccc::memory_resource& mr, const vecmem::copy& copy,
    const stream_wrapper& str, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr{mr}, m_copy{copy}, m_stream{str} {}

measurement_sorting_algorithm::output_type
measurement_sorting_algorithm::operator()(
    const edm::measurement_collection::const_view& measurements_view) const {
  // Exit early if there are no measurements.
  if (measurements_view.capacity() == 0) {
    return {};
  }

  // Get the number of measurements. In an asynchronous way if possible.
  // (The radix sorts need to know the number of elements on the host.)
  edm::measurement_collection::const_view::size_type n_measurements = 0u;
  if (m_mr.host) {
    const vecmem::async_size size =
        m_copy.get().get_size(measurements_view, *(m_mr.host));
    n_measurements = size.get();
  } else {
    n_measurements = m_copy.get().get_size(measurements_view);
  }

  // Create the output buffer, sized exactly for the measurements. It is not
  // resizable, so that its size is known on the host without a device
  // synchronisation.
  output_type result{n_measurements, m_mr.main};
  m_copy.get().setup(result)->ignore();
  if (n_measurements == 0) {
    return result;
  }

  // Get a convenience variable for the stream that we'll be using.
  hipStream_t stream = details::get_stream(m_stream);
  // Set up the Thrust execution policy. Unfortunately this has to be done in
  // a backend specify way. :-(
#if defined(__HIP_PLATFORM_AMD__)
  auto policy =
      thrust::hip::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
          .on(stream);
#elif defined(__HIP_PLATFORM_NVIDIA__)
  auto policy =
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
          .on(stream);
#else
#error "Unrecognised HIP platform."
#endif

  // Sorting keys and index sequence.
  vecmem::data::vector_buffer<device::measurement_cluster_key_t> cluster_keys(
      n_measurements, m_mr.main);
  vecmem::data::vector_buffer<device::measurement_surface_key_t> surface_keys(
      n_measurements, m_mr.main);
  vecmem::data::vector_buffer<unsigned int> indices(n_measurements, m_mr.main);
  m_copy.get().setup(cluster_keys)->ignore();
  m_copy.get().setup(surface_keys)->ignore();
  m_copy.get().setup(indices)->ignore();

  static constexpr unsigned int BLOCK_SIZE = 256;
  const unsigned int n_blocks = (n_measurements + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // First sort by the deterministic cluster key. This removes the effect of
  // the non-deterministic order in which the clusterization wrote the
  // measurements.
  kernels::fill_measurement_cluster_keys<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
      measurements_view, cluster_keys, indices);
  TRACCC_HIP_ERROR_CHECK(hipGetLastError());
  thrust::stable_sort_by_key(policy, cluster_keys.ptr(),
                             cluster_keys.ptr() + n_measurements,
                             indices.ptr());

  // Then (stably) sort by the surface identifier. Both sorts use radix sort
  // on primitive keys, which is deterministic.
  kernels::gather_measurement_surface_keys<<<n_blocks, BLOCK_SIZE, 0,
                                             stream>>>(
      measurements_view, indices, surface_keys);
  TRACCC_HIP_ERROR_CHECK(hipGetLastError());
  thrust::stable_sort_by_key(policy, surface_keys.ptr(),
                             surface_keys.ptr() + n_measurements,
                             indices.ptr());

  // Fill the output with the sorted measurements.
  kernels::fill_sorted_measurements<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
      measurements_view, result, indices);
  TRACCC_HIP_ERROR_CHECK(hipGetLastError());

  // Return the sorted buffer.
  return result;
}

}  // namespace traccc::hip
