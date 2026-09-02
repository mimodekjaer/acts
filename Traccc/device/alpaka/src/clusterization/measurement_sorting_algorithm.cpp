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

  // Get the number of measurements. In an asynchronous way if possible.
  edm::measurement_collection::const_view::size_type n_measurements = 0u;
  if (m_mr.host) {
    const vecmem::async_size size =
        m_copy.get().get_size(measurements_view, *(m_mr.host));
    n_measurements = size.get();
  } else {
    n_measurements = m_copy.get().get_size(measurements_view);
  }

  // Create the output buffer, sized exactly for the measurements.
  output_type result{n_measurements, m_mr.main,
                     vecmem::data::buffer_type::resizable};
  m_copy.get().setup(result)->ignore();
  if (n_measurements == 0) {
    return result;
  }
  m_copy.get()(measurements_view.size(), result.size())->ignore();

  auto queue = details::get_queue(m_queue);

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
  static constexpr unsigned int BLOCK_SIZE = 256;
  const unsigned int n_blocks = (n_measurements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  auto workDiv = makeWorkDiv<Acc>(n_blocks, BLOCK_SIZE);
  ::alpaka::exec<Acc>(queue, workDiv, kernels::fill_sorted_measurements{},
                      measurements_view, vecmem::get_data(result),
                      vecmem::get_data(indices));

  return result;
}

}  // namespace traccc::alpaka
