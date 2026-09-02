/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"

// Project include(s).
#include "traccc/clusterization/device/measurement_sorting.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/utils/copy.hpp>

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// System include(s).
#include <memory_resource>

namespace traccc::cuda {
namespace kernels {

/// Kernel wrapping @c traccc::device::fill_measurement_surface_keys
__global__ void fill_measurement_surface_keys(
    const edm::measurement_collection::const_view measurements,
    vecmem::data::vector_view<device::measurement_surface_key_t> keys,
    vecmem::data::vector_view<unsigned int> indices) {
  device::fill_measurement_surface_keys(details::global_index1(), measurements, keys, indices);
}

/// Kernel wrapping @c traccc::device::flag_unsorted_measurements
__global__ void flag_unsorted_measurements(
    const edm::measurement_collection::const_view measurements,
    vecmem::data::vector_view<unsigned int> unsorted) {
  device::flag_unsorted_measurements(details::global_index1(), measurements, unsorted);
}

/// Kernel wrapping @c traccc::device::copy_measurements
__global__ void copy_measurements(
    const edm::measurement_collection::const_view input,
    edm::measurement_collection::view output) {
  device::copy_measurements(details::global_index1(), input, output);
}

/// Kernel wrapping @c traccc::device::fill_sorted_measurements
__global__ void fill_sorted_measurements(
    const edm::measurement_collection::const_view input,
    edm::measurement_collection::view output,
    const vecmem::data::vector_view<const unsigned int> sorted_indices) {
  device::fill_sorted_measurements(details::global_index1(), input, output,
                                   sorted_indices);
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

  // Get a convenience variable for the stream that we'll be using.
  cudaStream_t stream = details::get_stream(m_stream);

  // Check on the device whether the measurements are already sorted by
  // surface identifier. (Frequently the case, e.g. when the cells were read
  // grouped by surface.) The flag is read together with the size below.
  static constexpr unsigned int BLOCK_SIZE = 256;
  vecmem::data::vector_buffer<unsigned int> unsorted_flag(1u, m_mr.main);
  m_copy.get().setup(unsorted_flag)->ignore();
  m_copy.get().memset(unsorted_flag, 0)->ignore();
  {
    const unsigned int n_blocks =
        (measurements_view.capacity() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernels::flag_unsorted_measurements<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
        measurements_view, unsorted_flag);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
  }

  // Get the number of measurements and the sortedness flag. In an
  // asynchronous way if possible, with a single synchronisation.
  vecmem::vector<unsigned int> unsorted_flag_host(
      1u, (m_mr.host != nullptr) ? m_mr.host : std::pmr::get_default_resource());
  edm::measurement_collection::const_view::size_type n_measurements = 0u;
  if (m_mr.host) {
    const vecmem::async_size size =
        m_copy.get().get_size(measurements_view, *(m_mr.host));
    m_copy.get()(unsorted_flag, unsorted_flag_host)->wait();
    n_measurements = size.get();
  } else {
    n_measurements = m_copy.get().get_size(measurements_view);
    m_copy.get()(unsorted_flag, unsorted_flag_host)->wait();
  }
  const bool already_sorted = (unsorted_flag_host.at(0) == 0u);

  // Create the output buffer, sized exactly for the measurements. It is not
  // resizable, so that its size is known on the host without a device
  // synchronisation.
  output_type result{n_measurements, m_mr.main};
  m_copy.get().setup(result)->ignore();
  if (n_measurements == 0) {
    return result;
  }

  const unsigned int n_blocks = (n_measurements + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Fast path: the input is already sorted, so just copy it.
  if (already_sorted) {
    kernels::copy_measurements<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
        measurements_view, result);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    return result;
  }

  // Set up the Thrust execution policy.
  auto policy =
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
          .on(stream);

  // Sorting keys and index sequence.
  vecmem::data::vector_buffer<device::measurement_surface_key_t> surface_keys(
      n_measurements, m_mr.main);
  vecmem::data::vector_buffer<unsigned int> indices(n_measurements, m_mr.main);
  m_copy.get().setup(surface_keys)->ignore();
  m_copy.get().setup(indices)->ignore();


  // Sort the measurement indices by the surface identifier, using a stable
  // radix sort on primitive keys. The clusterization writes the measurements
  // in a deterministic order (that of their first cells), so the stable
  // sort produces a deterministic result.
  kernels::fill_measurement_surface_keys<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
      measurements_view, surface_keys, indices);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
  thrust::stable_sort_by_key(policy, surface_keys.ptr(),
                             surface_keys.ptr() + n_measurements,
                             indices.ptr());

  // Fill the output with the sorted measurements.
  kernels::fill_sorted_measurements<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
      measurements_view, result, indices);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

  // Return the sorted buffer.
  return result;
}

}  // namespace traccc::cuda
