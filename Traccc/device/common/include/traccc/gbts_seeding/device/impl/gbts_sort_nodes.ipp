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
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_sort_nodes(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_sort_nodes_payload& payload,
    const gbts_sort_nodes_shared_payload& shared_payload) {
  const vecmem::device_vector<const float4> d_reducedSP(payload.reducedSP);
  const vecmem::device_vector<const gbts_sort_key_t> d_sort_keys(
      payload.sort_keys);
  vecmem::device_vector<float4> d_node_params(payload.node_params);
  vecmem::device_vector<float> d_node_phi(payload.node_phi);
  vecmem::device_vector<unsigned int> d_node_index(payload.node_index);
  vecmem::device_vector<unsigned int> d_bin_rads_bits(payload.bin_rads_bits);
  vecmem::device_vector<unsigned int> d_eta_bin_views(payload.eta_bin_views);
  const vecmem::device_vector<const float> d_tau_lut(payload.tau_lut);
  vecmem::device_vector<unsigned int> shared_min(shared_payload.min_bits);
  vecmem::device_vector<unsigned int> shared_max(shared_payload.max_bits);

  const gbts_sort_nodes_params& ap = payload.gbts_sort_nodes_params;

  const unsigned int threadIndex = thread_id.getLocalThreadIdX();
  const unsigned int blockSize = thread_id.getBlockDimX();
  const unsigned int stride = blockSize * thread_id.getGridDimX();
  const unsigned int nKeys = payload.nKeys;
  const unsigned int nEtaBins = payload.nEtaBins;
  // Eta bin of a key slot; the rejected keys (sorted last) get nEtaBins.
  auto bin_of = [&](const unsigned int i) -> unsigned int {
    const gbts_sort_key_t k = d_sort_keys[i];
    return (k == gbts_sort_key_rejected)
               ? nEtaBins
               : (gbts_sort_key_bin_phi(k) >> gbts_sort_key_phi_bits);
  };

  // Block-uniform loop over the capacity (the barriers below must be reached
  // by every thread).
  for (unsigned int base = thread_id.getBlockIdX() * blockSize; base < nKeys;
       base += stride) {
    const unsigned int globalIndex = base + threadIndex;

    // Eta-bin boundaries: the thread whose bin differs from the previous
    // slot's bin writes the ranges. Empty bins between the two get an empty
    // range; the first rejected key marks the node count.
    if (globalIndex < nKeys) {
      const unsigned int cur = bin_of(globalIndex);
      const unsigned int prev =
          (globalIndex == 0u) ? 0u : bin_of(globalIndex - 1u);
      if ((globalIndex == 0u) || (cur != prev)) {
        const unsigned int first_begin = (globalIndex == 0u) ? 0u : prev + 1u;
        for (unsigned int b = first_begin; (b <= cur) && (b < nEtaBins); b++) {
          d_eta_bin_views[2u * b] = globalIndex;
        }
        const unsigned int first_end = (globalIndex == 0u) ? 0u : prev;
        for (unsigned int b = first_end; (b < cur) && (b < nEtaBins); b++) {
          d_eta_bin_views[2u * b + 1u] = globalIndex;
        }
        if (cur == nEtaBins) {
          *payload.nNodes = globalIndex;
        }
      }
      if ((globalIndex + 1u == nKeys) && (cur < nEtaBins)) {
        // No rejected key at all: close the last bins at the capacity.
        for (unsigned int b = cur; b < nEtaBins; b++) {
          d_eta_bin_views[2u * b + 1u] = nKeys;
          if (b > cur) {
            d_eta_bin_views[2u * b] = nKeys;
          }
        }
        *payload.nNodes = nKeys;
      }
    }

    // Nodes of this block: the slots before the first rejected key.
    const unsigned int last =
        (base + blockSize <= nKeys) ? base + blockSize - 1u : nKeys - 1u;
    const unsigned int bin_first = bin_of(base);
    const bool active =
        (globalIndex < nKeys) && (bin_of(globalIndex) < nEtaBins);
    unsigned int bin_last = bin_of(last);
    if (bin_last >= nEtaBins) {
      bin_last = nEtaBins - 1u;
    }
    const unsigned int n_bins =
        (bin_first < nEtaBins) ? bin_last - bin_first + 1u : 0u;
    // Shared reduction only when the spanned bins fit the scratch arrays.
    const bool use_shared = n_bins <= blockSize;
    if (use_shared && (threadIndex < n_bins)) {
      shared_min[threadIndex] = gbts_float_bits(1e8f);
      shared_max[threadIndex] = gbts_float_bits(0.0f);
    }
    barrier.blockBarrier();

    unsigned int bin = 0u;
    unsigned int r_bits = 0u;
    if (active) {
      const gbts_sort_key_t key = d_sort_keys[globalIndex];
      const unsigned int srcIdx = gbts_sort_key_index(key);
      const unsigned int bin_phi = gbts_sort_key_bin_phi(key);
      bin = bin_phi >> gbts_sort_key_phi_bits;
      const float4 sp = d_reducedSP[srcIdx];

      const float Phi = math::atan2(sp.y, sp.x);
      const float r = math::sqrt(sp.x * sp.x + sp.y * sp.y);
      const float z = sp.z;
      r_bits = gbts_float_bits(r);

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
              5 *
              static_cast<int>(math::floor(ap.tau_lut_inv_bin * sp.w) - 1.0f);
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

      // The keys order the nodes by quantised phi; inside a run of equal
      // (eta bin, quantised phi) the exact (phi, r, z, width, spacepoint
      // index) rank decides the slot, so the nodes of an eta bin end up
      // exactly sorted by phi. The intrinsic node data come before the
      // spacepoint index on purpose: the spacepoint order produced by the
      // upstream (GPU) clusterization is not reproducible run to run, so an
      // index tie-break would make the node order - and through it every
      // index-based tie-break downstream - schedule dependent. Only nodes
      // with identical parameters still fall back to the index, and those
      // are interchangeable for the seeding.
      unsigned int pos = globalIndex;
      const bool in_run =
          ((globalIndex > 0u) &&
           (gbts_sort_key_bin_phi(d_sort_keys[globalIndex - 1u]) == bin_phi)) ||
          ((globalIndex + 1u < nKeys) &&
           (gbts_sort_key_bin_phi(d_sort_keys[globalIndex + 1u]) == bin_phi));
      if (in_run) {
        unsigned int start = globalIndex;
        while ((start > 0u) &&
               (gbts_sort_key_bin_phi(d_sort_keys[start - 1u]) == bin_phi)) {
          --start;
        }
        unsigned int end = globalIndex + 1u;
        while ((end < nKeys) &&
               (gbts_sort_key_bin_phi(d_sort_keys[end]) == bin_phi)) {
          ++end;
        }
        unsigned int rank = 0u;
        for (unsigned int j = start; j < end; j++) {
          if (j == globalIndex) {
            continue;
          }
          const unsigned int otherIdx = gbts_sort_key_index(d_sort_keys[j]);
          const float4 other = d_reducedSP[otherIdx];
          const float otherPhi = math::atan2(other.y, other.x);
          bool before = otherPhi < Phi;
          if (otherPhi == Phi) {
            const float otherR =
                math::sqrt(other.x * other.x + other.y * other.y);
            before = (otherR < r) ||
                     ((otherR == r) &&
                      ((other.z < z) ||
                       ((other.z == z) &&
                        ((other.w < sp.w) ||
                         ((other.w == sp.w) && (otherIdx < srcIdx))))));
          }
          if (before) {
            ++rank;
          }
        }
        pos = start + rank;
      }
      d_node_params[pos] = float4{min_tau, max_tau, r, z};
      d_node_phi[pos] = Phi;
      d_node_index[pos] = srcIdx;

      if (use_shared) {
        vecmem::device_atomic_ref<unsigned int,
                                  vecmem::device_address_space::local>(
            shared_min[bin - bin_first])
            .fetch_min(r_bits);
        vecmem::device_atomic_ref<unsigned int,
                                  vecmem::device_address_space::local>(
            shared_max[bin - bin_first])
            .fetch_max(r_bits);
      } else {
        vecmem::device_atomic_ref<unsigned int>(d_bin_rads_bits[2u * bin])
            .fetch_min(r_bits);
        vecmem::device_atomic_ref<unsigned int>(d_bin_rads_bits[2u * bin + 1u])
            .fetch_max(r_bits);
      }
    }
    barrier.blockBarrier();
    // Merge the block's per-bin ranges into the global ones.
    if (use_shared && (threadIndex < n_bins)) {
      const unsigned int lo = shared_min[threadIndex];
      const unsigned int hi = shared_max[threadIndex];
      if (lo <= hi) {  // the bin got at least one node of this block
        vecmem::device_atomic_ref<unsigned int>(
            d_bin_rads_bits[2u * (bin_first + threadIndex)])
            .fetch_min(lo);
        vecmem::device_atomic_ref<unsigned int>(
            d_bin_rads_bits[2u * (bin_first + threadIndex) + 1u])
            .fetch_max(hi);
      }
    }
    // The shared arrays are rewritten by the next iteration.
    barrier.blockBarrier();
  }
}

}  // namespace traccc::device
