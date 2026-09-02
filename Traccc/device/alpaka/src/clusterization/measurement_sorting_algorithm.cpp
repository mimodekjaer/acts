/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"

#include "../utils/get_queue.hpp"
#include "../utils/parallel_algorithms.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/clusterization/device/measurement_sorting.hpp"
#include "traccc/clusterization/device/sorting_index_filler.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <memory_resource>

namespace traccc::alpaka {
namespace kernels {

/// Kernel filling the output buffer with sorted measurements.
struct fill_sorted_measurements {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, const edm::measurement_collection::const_view input_view,
      edm::measurement_collection::view output_view,
      const vecmem::data::vector_view<const unsigned int> sorted_indices_view)
      const {
    device::fill_sorted_measurements(
        details::thread_id1{acc}.getGlobalThreadId(), input_view, output_view,
        sorted_indices_view);
  }
};  // struct fill_sorted_measurements

/// Kernel flagging an unsorted measurement collection.
struct flag_unsorted_measurements {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, const edm::measurement_collection::const_view input_view,
      vecmem::data::vector_view<unsigned int> unsorted_view) const {
    device::flag_unsorted_measurements(
        details::thread_id1{acc}.getGlobalThreadId(), input_view,
        unsorted_view);
  }
};  // struct flag_unsorted_measurements

/// Kernel copying an already sorted measurement collection.
struct copy_measurements {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, const edm::measurement_collection::const_view input_view,
      edm::measurement_collection::view output_view) const {
    device::copy_measurements(details::thread_id1{acc}.getGlobalThreadId(),
                              input_view, output_view);
  }
};  // struct copy_measurements

}  // namespace kernels

measurement_sorting_algorithm::measurement_sorting_algorithm(
    const traccc::memory_resource& mr, const vecmem::copy& copy, queue& q,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr{mr}, m_copy{copy}, m_queue{q} {}

measurement_sorting_algorithm::output_type
measurement_sorting_algorithm::operator()(
    const edm::measurement_collection::const_view& measurements_view) const {
  // Exit early if there are no measurements.
  if (measurements_view.capacity() == 0) {
    return {};
  }

  auto queue = details::get_queue(m_queue);
  static constexpr unsigned int BLOCK_SIZE = 256;

  // Check on the device whether the measurements are already sorted by
  // surface identifier. (Frequently the case, e.g. when the cells were read
  // grouped by surface.) The flag is read together with the size below.
  vecmem::data::vector_buffer<unsigned int> unsorted_flag(1u, m_mr.main);
  m_copy.get().setup(unsorted_flag)->ignore();
  m_copy.get().memset(unsorted_flag, 0)->ignore();
  {
    const unsigned int n_blocks =
        (measurements_view.capacity() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ::alpaka::exec<Acc>(queue, makeWorkDiv<Acc>(n_blocks, BLOCK_SIZE),
                        kernels::flag_unsorted_measurements{},
                        measurements_view, vecmem::get_data(unsorted_flag));
  }

  // Get the number of measurements and the sortedness flag. In an
  // asynchronous way if possible.
  vecmem::vector<unsigned int> unsorted_flag_host(
      1u,
      (m_mr.host != nullptr) ? m_mr.host : std::pmr::get_default_resource());
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

  // Fast path: the input is already sorted, so just copy it.
  const unsigned int n_blocks = (n_measurements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (already_sorted) {
    ::alpaka::exec<Acc>(queue, makeWorkDiv<Acc>(n_blocks, BLOCK_SIZE),
                        kernels::copy_measurements{}, measurements_view,
                        vecmem::get_data(result));
    return result;
  }

  // Create a vector of measurement indices, which would be sorted.
  vecmem::data::vector_buffer<unsigned int> indices(n_measurements, m_mr.main);
  m_copy.get().setup(indices)->wait();
  details::for_each(queue, m_mr, indices.ptr(), indices.ptr() + n_measurements,
                    device::sorting_index_filler{indices});

  // Sort the indices with a deterministic total order (surface identifier,
  // then the cluster key stored in the measurement identifier).
  details::sort(queue, m_mr, indices.ptr(), indices.ptr() + n_measurements,
                device::measurement_order_sorter{measurements_view});

  // Fill the output with the sorted measurements.
  auto workDiv = makeWorkDiv<Acc>(n_blocks, BLOCK_SIZE);
  ::alpaka::exec<Acc>(queue, workDiv, kernels::fill_sorted_measurements{},
                      measurements_view, vecmem::get_data(result),
                      vecmem::get_data(indices));

  return result;
}

}  // namespace traccc::alpaka
