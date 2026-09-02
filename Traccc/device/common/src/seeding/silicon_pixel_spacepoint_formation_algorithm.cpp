/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/seeding/device/silicon_pixel_spacepoint_formation_algorithm.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>

namespace traccc::device {

silicon_pixel_spacepoint_formation_algorithm::
    silicon_pixel_spacepoint_formation_algorithm(
        const traccc::memory_resource& mr, const vecmem::copy& copy,
        std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), algorithm_base(mr, copy) {}

auto silicon_pixel_spacepoint_formation_algorithm::operator()(
    const detector_buffer& det,
    const edm::measurement_collection::const_view& measurements) const
    -> output_type {
  // Get the number of measurements. In an asynchronous way if possible.
  edm::measurement_collection::const_view::size_type n_measurements = 0u;
  if (mr().host) {
    vecmem::async_size size = copy().get_size(measurements, *(mr().host));
    // Here we could give control back to the caller, once our code allows
    // for it. (coroutines...)
    n_measurements = size.get();
  } else {
    n_measurements = copy().get_size(measurements);
  }

  // If there are no measurements, return right away.
  if (n_measurements == 0) {
    return {};
  }

  // Create the result buffer.
  edm::spacepoint_collection::buffer spacepoints(
      n_measurements, mr().main, vecmem::data::buffer_type::resizable);
  copy().setup(spacepoints)->ignore();

  // Scratch buffer for the measurement flags / prefix sums.
  vecmem::data::vector_buffer<unsigned int> offsets(n_measurements, mr().main);
  copy().setup(offsets)->ignore();

  // Launch the spacepoint formation kernel(s).
  form_spacepoints_kernel(
      {n_measurements, det, measurements, offsets, spacepoints});

  // Return the reconstructed spacepoints.
  return spacepoints;
}

}  // namespace traccc::device
