/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// Block size used by @c traccc::device::gbts_find_minmax_radius (one block
/// per eta bin; must be a power of two, it sizes the shared reduction
/// buffers)
inline constexpr unsigned int gbts_find_minmax_radius_block_size = 256u;

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_find_minmax_radius function
struct gbts_find_minmax_radius_payload {
  /// Number of eta bins
  unsigned int nEtaBins;
  /// Per-eta (begin, end) node range, as 2*nEtaBins flat ints
  vecmem::data::vector_view<const unsigned int> eta_bin_views;
  /// Per-node (tau_min, tau_max, r, z) (only r is read here)
  vecmem::data::vector_view<const float4> node_params;
  /// Output: per-eta (rmin, rmax) pair, flat (2*nEtaBins floats)
  vecmem::data::vector_view<float> bin_rads;
};

/// (Shared Event Data) Payload for the @c
/// traccc::device::gbts_find_minmax_radius function
struct gbts_find_minmax_radius_shared_payload {
  /// Per-thread partial minimum radius
  /// (gbts_find_minmax_radius_block_size floats)
  vecmem::data::vector_view<float> shared_min;
  /// Per-thread partial maximum radius
  /// (gbts_find_minmax_radius_block_size floats)
  vecmem::data::vector_view<float> shared_max;
};

/// @brief Compute the per-eta-bin minimum and maximum radius.
///
/// One block per eta bin: the threads stride over the bin's node range,
/// then the partial results are tree-reduced in shared memory and thread 0
/// writes the (rmin, rmax) pair into the output; the host uses these to
/// estimate the maximum delta-R for each bin pair.
///
/// @param[in]     thread_id      Thread identifier for the kernel launch
/// @param[in]     barrier        Block-level barrier
/// @param[in]     payload        The global memory payload
/// @param[in,out] shared_payload The shared memory payload
///
template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_find_minmax_radius(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_find_minmax_radius_payload& payload,
    const gbts_find_minmax_radius_shared_payload& shared_payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_find_minmax_radius.ipp"
