/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"

#include "../utils/barrier.hpp"
#include "../utils/get_queue.hpp"
#include "../utils/parallel_algorithms.hpp"
#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/geometry/detector.hpp"
#include "traccc/seeding/device/form_spacepoints.hpp"

namespace traccc::alpaka {
namespace kernels {

/// Kernel wrapping @c device::flag_spacepoint_measurements
struct flag_spacepoint_measurements {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, edm::measurement_collection::const_view measurements,
      vecmem::data::vector_view<unsigned int> flags) const {
    auto const globalThreadIdx =
        ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
    device::flag_spacepoint_measurements(globalThreadIdx, measurements, flags);
  }
};

/// Kernel wrapping @c device::form_spacepoints
template <typename detector_t>
struct form_spacepoints {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, const typename detector_t::view* detector,
      edm::measurement_collection::const_view measurements,
      vecmem::data::vector_view<const unsigned int> offsets,
      edm::spacepoint_collection::view spacepoints) const {
    auto const globalThreadIdx =
        ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
    device::form_spacepoints<detector_t>(globalThreadIdx, *detector,
                                         measurements, offsets, spacepoints);
  }
};

}  // namespace kernels

silicon_pixel_spacepoint_formation_algorithm::
    silicon_pixel_spacepoint_formation_algorithm(
        const traccc::memory_resource& mr, const vecmem::copy& copy,
        alpaka::queue& q, std::unique_ptr<const Logger> logger)
    : device::silicon_pixel_spacepoint_formation_algorithm(mr, copy,
                                                           std::move(logger)),
      alpaka::algorithm_base(q) {}

void silicon_pixel_spacepoint_formation_algorithm::form_spacepoints_kernel(
    const form_spacepoints_kernel_payload& payload) const {
  const unsigned int n_threads = warp_size() * 8;
  const unsigned int n_blocks =
      (payload.n_measurements + n_threads - 1) / n_threads;
  auto queue = details::get_queue(this->queue());

  // Flag the measurements that produce spacepoints.
  ::alpaka::exec<Acc>(queue, makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::flag_spacepoint_measurements{},
                      payload.measurements, payload.offsets);

  // Turn the flags into inclusive prefix sums, in place.
  details::inclusive_scan(queue, mr(), payload.offsets.ptr(),
                          payload.offsets.ptr() + payload.n_measurements,
                          payload.offsets.ptr());

  // The last prefix sum is the number of spacepoints. Copy it into the size
  // of the output buffer.
  copy()(vecmem::data::vector_view<const char>{
             static_cast<vecmem::data::vector_view<const char>::size_type>(
                 sizeof(unsigned int)),
             reinterpret_cast<const char*>(payload.offsets.ptr() +
                                           payload.n_measurements - 1u)},
         payload.spacepoints.size())
      ->wait();

  // Form the spacepoints.
  detector_buffer_visitor<detector_type_list>(
      payload.detector, [&]<typename detector_traits_t>(
                            const typename detector_traits_t::view& det) {
        // Put the detector view into device memory.
        vecmem::data::vector_buffer<typename detector_traits_t::view>
            device_det(1u, mr().main);
        copy().setup(device_det)->wait();
        copy()(
            vecmem::data::vector_view<const typename detector_traits_t::view>(
                1u, &det),
            device_det)
            ->wait();
        // Launch the spacepoint formation kernel.
        ::alpaka::exec<Acc>(
            queue, makeWorkDiv<Acc>(n_blocks, n_threads),
            kernels::form_spacepoints<detector_traits_t>{}, device_det.ptr(),
            payload.measurements,
            vecmem::data::vector_view<const unsigned int>(payload.offsets),
            vecmem::get_data(payload.spacepoints));
      });
}

}  // namespace traccc::alpaka
