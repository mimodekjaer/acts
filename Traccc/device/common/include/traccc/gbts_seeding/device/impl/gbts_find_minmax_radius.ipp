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
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_find_minmax_radius(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_find_minmax_radius_payload& payload,
    const gbts_find_minmax_radius_shared_payload& shared_payload) {
  const vecmem::device_vector<const unsigned int> d_eta_bin_views(
      payload.eta_bin_views);
  const vecmem::device_vector<const float4> d_node_params(payload.node_params);
  vecmem::device_vector<float> d_bin_rads(payload.bin_rads);
  vecmem::device_vector<float> shared_min(shared_payload.shared_min);
  vecmem::device_vector<float> shared_max(shared_payload.shared_max);

  // One block per eta bin (block-uniform exit, so safe before the barriers).
  const unsigned int bin = thread_id.getBlockIdX();
  if (bin >= payload.nEtaBins) {
    return;
  }
  const unsigned int threadIndex = thread_id.getLocalThreadIdX();
  const unsigned int blockSize = thread_id.getBlockDimX();

  const unsigned int node_start = d_eta_bin_views[2u * bin];
  const unsigned int node_end = d_eta_bin_views[2u * bin + 1u];

  float min_r = 1e8f;
  float max_r = -1e8f;

  for (unsigned int node_idx = node_start + threadIndex; node_idx < node_end;
       node_idx += blockSize) {
    const float r = d_node_params[node_idx].z;
    max_r = fmaxf(r, max_r);
    min_r = fminf(r, min_r);
  }

  shared_min[threadIndex] = min_r;
  shared_max[threadIndex] = max_r;
  barrier.blockBarrier();

  // Tree reduction (blockSize is a power of two).
  for (unsigned int stride = blockSize / 2u; stride > 0u; stride /= 2u) {
    if (threadIndex < stride) {
      shared_min[threadIndex] =
          fminf(shared_min[threadIndex], shared_min[threadIndex + stride]);
      shared_max[threadIndex] =
          fmaxf(shared_max[threadIndex], shared_max[threadIndex + stride]);
    }
    barrier.blockBarrier();
  }

  if (threadIndex == 0u) {
    d_bin_rads[2u * bin] = shared_min[0];
    d_bin_rads[2u * bin + 1u] = shared_max[0];
  }
}

}  // namespace traccc::device
