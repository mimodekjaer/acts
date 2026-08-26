/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/device/gbts_bin_spacepoints.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_sort_nodes(
    const thread_id_t& thread_id, const gbts_sort_nodes_payload& payload) {
  const vecmem::device_vector<const float4> d_reducedSP(payload.reducedSP);
  const vecmem::device_vector<const gbts_sort_key_t> d_sort_keys(
      payload.sort_keys);
  const vecmem::device_vector<const unsigned int> d_sort_values(
      payload.sort_values);
  vecmem::device_vector<float4> d_node_params(payload.node_params);
  vecmem::device_vector<float> d_node_phi(payload.node_phi);
  vecmem::device_vector<unsigned int> d_node_index(payload.node_index);
  const vecmem::device_vector<const float> d_tau_lut(payload.tau_lut);

  const gbts_sort_nodes_params& ap = payload.gbts_sort_nodes_params;

  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int blockDimX = thread_id.getBlockDimX();
  const unsigned int gridDimX = thread_id.getGridDimX();

  const unsigned int nNodes = *payload.nNodes;
  for (unsigned int globalIndex = globalIdx; globalIndex < nNodes;
       globalIndex += blockDimX * gridDimX) {
    const unsigned int srcIdx = d_sort_values[globalIndex];
    const float4 sp = d_reducedSP[srcIdx];

    const float Phi = math::atan2(sp.y, sp.x);
    const float r = math::sqrt(sp.x * sp.x + sp.y * sp.y);
    const float z = sp.z;

    // Default to the full |tau| acceptance for nodes that carry no usable
    // cluster width (sp.w <= 0); the per-edge cuts then rely on these
    // bounds.
    float min_tau = 0.0f;
    float max_tau = ap.maxTau;

    if (sp.w > 0) {  // type 0 only
      if (ap.useTauLUT) {
        // LUT is laid out as [w_bin_edge, min_tau_0, max_tau_0,
        // min_tau_1, max_tau_1] per bin.
        const int tau_bin =
            5 * static_cast<int>(math::floor(ap.tau_lut_inv_bin * sp.w) - 1.0f);
        if (tau_bin > -1 && tau_bin < static_cast<int>(ap.tauLutSize)) {
          min_tau = d_tau_lut[static_cast<unsigned int>(tau_bin) + 1u];
          max_tau = d_tau_lut[static_cast<unsigned int>(tau_bin) + 2u];
        }
        if (max_tau < 0.0f) {
          max_tau = ap.maxTau;
        }
        if (min_tau < 0.0f) {
          min_tau = 0.0f;
        }
      } else {
        // linear fit + correction for short clusters
        min_tau = ap.tMin_slope * (sp.w - ap.offset);
        max_tau = ap.tMax_min + ap.tMax_correction / (sp.w + ap.offset) +
                  ap.tMax_slope * (sp.w - ap.offset);
      }
    }

    // The keys order the nodes by quantised phi; inside a run of equal keys
    // the exact (phi, spacepoint index) rank decides the slot, so the nodes
    // of an eta bin end up exactly sorted by phi (deterministically).
    unsigned int pos = globalIndex;
    const gbts_sort_key_t key = d_sort_keys[globalIndex];
    const bool in_run =
        ((globalIndex > 0u) && (d_sort_keys[globalIndex - 1u] == key)) ||
        ((globalIndex + 1u < nNodes) && (d_sort_keys[globalIndex + 1u] == key));
    if (in_run) {
      unsigned int start = globalIndex;
      while ((start > 0u) && (d_sort_keys[start - 1u] == key)) {
        --start;
      }
      unsigned int end = globalIndex + 1u;
      while ((end < nNodes) && (d_sort_keys[end] == key)) {
        ++end;
      }
      unsigned int rank = 0u;
      for (unsigned int j = start; j < end; j++) {
        if (j == globalIndex) {
          continue;
        }
        const unsigned int otherIdx = d_sort_values[j];
        const float4 other = d_reducedSP[otherIdx];
        const float otherPhi = math::atan2(other.y, other.x);
        if ((otherPhi < Phi) || ((otherPhi == Phi) && (otherIdx < srcIdx))) {
          ++rank;
        }
      }
      pos = start + rank;
    }
    d_node_params[pos] = float4{min_tau, max_tau, r, z};
    d_node_phi[pos] = Phi;
    d_node_index[pos] = srcIdx;
  }
}

}  // namespace traccc::device
