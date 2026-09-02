/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"

// Project include(s).
#include "traccc/geometry/detector.hpp"
#include "traccc/seeding/device/form_spacepoints.hpp"

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// System include(s).
#include <memory_resource>

namespace traccc::cuda {
namespace kernels {

/// Kernel wrapping @c device::flag_spacepoint_measurements
__global__ void flag_spacepoint_measurements(
    edm::measurement_collection::const_view measurements,
    vecmem::data::vector_view<unsigned int> flags) {
  device::flag_spacepoint_measurements(details::global_index1(), measurements,
                                       flags);
}

/// Kernel wrapping @c device::form_spacepoints
template <typename detector_t>
__global__ void __launch_bounds__(1024, 1)
    form_spacepoints(typename detector_t::view detector,
                     edm::measurement_collection::const_view measurements,
                     vecmem::data::vector_view<const unsigned int> offsets,
                     edm::spacepoint_collection::view spacepoints)
  requires(traccc::is_detector_traits<detector_t>)
{
  device::form_spacepoints<detector_t>(details::global_index1(), detector,
                                       measurements, offsets, spacepoints);
}

}  // namespace kernels

silicon_pixel_spacepoint_formation_algorithm::
    silicon_pixel_spacepoint_formation_algorithm(
        const traccc::memory_resource& mr, const vecmem::copy& copy,
        const stream_wrapper& str, std::unique_ptr<const Logger> logger)
    : device::silicon_pixel_spacepoint_formation_algorithm(mr, copy,
                                                           std::move(logger)),
      cuda::algorithm_base(str) {}

void silicon_pixel_spacepoint_formation_algorithm::form_spacepoints_kernel(
    const form_spacepoints_kernel_payload& payload) const {
  cudaStream_t cuda_stream = details::get_stream(stream());
  const unsigned int n_threads = warp_size() * 8;
  const unsigned int n_blocks =
      (payload.n_measurements + n_threads - 1) / n_threads;

  // Flag the measurements that produce spacepoints.
  kernels::flag_spacepoint_measurements<<<n_blocks, n_threads, 0,
                                          cuda_stream>>>(payload.measurements,
                                                         payload.offsets);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

  // Turn the flags into inclusive prefix sums, in place.
  auto policy =
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(cuda_stream);
  thrust::inclusive_scan(policy, payload.offsets.ptr(),
                         payload.offsets.ptr() + payload.n_measurements,
                         payload.offsets.ptr());

  // The last prefix sum is the number of spacepoints. Copy it into the size
  // of the output buffer (device-to-device, in stream order).
  copy()(vecmem::data::vector_view<const char>{
             static_cast<vecmem::data::vector_view<const char>::size_type>(
                 sizeof(unsigned int)),
             reinterpret_cast<const char*>(payload.offsets.ptr() +
                                           payload.n_measurements - 1u)},
         payload.spacepoints.size())
      ->ignore();

  // Form the spacepoints.
  detector_buffer_visitor<detector_type_list>(
      payload.detector, [&]<typename detector_traits_t>(
                            const typename detector_traits_t::view& det) {
        kernels::form_spacepoints<detector_traits_t>
            <<<n_blocks, n_threads, 0, cuda_stream>>>(
                det, payload.measurements, payload.offsets,
                payload.spacepoints);
      });
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
