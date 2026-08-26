/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

// Detray include(s).
#include <detray/geometry/identifier.hpp>

// System include(s).
#include <array>
#include <climits>
#include <utility>

namespace traccc::device {

namespace detail {

/// Bin one spacepoint: returns false when it is rejected, otherwise writes
/// its reduced parameters, bumps its eta-bin count and returns its node
/// sort key.
template <typename spacepoints_t, typename measurements_t>
TRACCC_HOST_DEVICE inline bool gbts_bin_one_spacepoint(
    const gbts_bin_spacepoints_payload& payload,
    const spacepoints_t& spacepoints, const measurements_t& measurements,
    const vecmem::device_vector<const short>& volumeToLayerMap,
    const vecmem::device_vector<const std::pair<unsigned int, unsigned int>>&
        surfaceToLayerMap,
    const vecmem::device_vector<const char>& layerType,
    const vecmem::device_vector<const std::pair<unsigned int, unsigned int>>&
        d_layer_info,
    const vecmem::device_vector<const std::pair<float, float>>& d_layer_geo,
    vecmem::device_vector<float4>& reducedSP,
    vecmem::device_vector<unsigned int>& d_eta_node_counter,
    const unsigned int globalIndex, gbts_sort_key_t& key) {
  // --- Stage 1: layer assignment -----------
  const auto spacepoint = spacepoints.at(globalIndex);
  const auto measurement = measurements.at(spacepoint.measurement_index_1());

  const detray::geometry::identifier geo_id = measurement.surface_link();
  const unsigned int volume = geo_id.volume();
  const short begin_or_bin =
      (volume < payload.volumeMapSize) ? volumeToLayerMap[volume] : SHRT_MAX;

  if (begin_or_bin == SHRT_MAX) {
    reducedSP[globalIndex].w = -CHAR_MAX - 1;
    return false;
  }
  unsigned int layerIdx = 0u;
  if (begin_or_bin < 0) {
    const unsigned int surface_index =
        static_cast<unsigned int>(geo_id.index());

    // The volume's block: a header (count) then the surfaces sorted by
    // surface index -> binary search.
    const unsigned int block =
        static_cast<unsigned int>(-1 * (begin_or_bin + 1));
    unsigned int lo = block + 1u;
    unsigned int hi = lo + surfaceToLayerMap[block].first;
    if (hi > payload.surfaceMapSize) {
      hi = static_cast<unsigned int>(payload.surfaceMapSize);
    }
    while (lo < hi) {
      const unsigned int mid = lo + (hi - lo) / 2u;
      const std::pair<unsigned int, unsigned int> surfaceBinPair =
          surfaceToLayerMap[mid];
      if (surfaceBinPair.first == surface_index) {
        layerIdx = surfaceBinPair.second;
        break;
      }
      if (surfaceBinPair.first < surface_index) {
        lo = mid + 1u;
      } else {
        hi = mid;
      }
    }
  } else {
    layerIdx = static_cast<unsigned int>(begin_or_bin);
  }
  float cluster_diameter = measurement.diameter();
  const int type = static_cast<int>(layerType[layerIdx]);
  if (type == 1 &&
      cluster_diameter >
          payload.gbts_count_spacepoints_by_layer_params.type1_max_width) {
    reducedSP[globalIndex].w = -CHAR_MAX - 1;
    return false;
  }
  cluster_diameter =
      (payload.gbts_count_spacepoints_by_layer_params.doTauCut && type != 0)
          ? static_cast<float>(-1 * type)
          : cluster_diameter;

  const std::array<float, 3u> pos = spacepoint.global();
  reducedSP[globalIndex] = float4{pos[0], pos[1], pos[2], cluster_diameter};
  // global x, y, z, and cluster diameter

  // --- Stage 2: node_eta_binning -----------
  const std::pair<unsigned int, unsigned int> layerInfo =
      d_layer_info[layerIdx];
  const unsigned int bin0 = layerInfo.first;
  const unsigned int num_eta_bins = layerInfo.second;
  unsigned int eta_index;
  if (num_eta_bins == 1u) {
    eta_index = bin0;
  } else {
    const std::pair<float, float> layerGeo = d_layer_geo[layerIdx];
    const float min_eta = layerGeo.first;
    const float eta_bin_width = layerGeo.second;
    const float r = math::sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
    const float t1 = pos[2] / r;
    const float eta = -math::log(math::sqrt(1.0f + t1 * t1) - t1);
    const unsigned int binIdx = static_cast<unsigned int>(
        math::max(0.0f, math::min((eta - min_eta) / eta_bin_width,
                                  static_cast<float>(num_eta_bins - 1u))));
    eta_index = bin0 + binIdx;
  }
  vecmem::device_atomic_ref<unsigned int>(d_eta_node_counter[eta_index])
      .fetch_add(1u);

  // --- Stage 3: node_sort_key -----------
  // Concatenate the eta bin, order-preserving phi bits, and the spacepoint
  // index into a single 64-bit integer, used to sort the nodes.
  const float Phi = math::atan2(pos[1], pos[0]);
  key = (eta_index << gbts_sort_key_eta_shift) | gbts_quantised_phi(Phi);
  return true;
}

}  // namespace detail

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_bin_spacepoints(
    const thread_id_t& thread_id, const gbts_bin_spacepoints_payload& payload) {
  const traccc::edm::spacepoint_collection::const_device spacepoints(
      payload.spacepoints);
  const edm::measurement_collection::const_device measurements(
      payload.measurements);
  const vecmem::device_vector<const short> volumeToLayerMap(
      payload.volumeToLayerMap);
  const vecmem::device_vector<const std::pair<unsigned int, unsigned int>>
      surfaceToLayerMap(payload.surfaceToLayerMap);
  const vecmem::device_vector<const char> layerType(payload.layerType);
  const vecmem::device_vector<const std::pair<unsigned int, unsigned int>>
      d_layer_info(payload.layer_info);
  const vecmem::device_vector<const std::pair<float, float>> d_layer_geo(
      payload.layer_geo);

  vecmem::device_vector<float4> reducedSP(payload.reducedSP);
  vecmem::device_vector<unsigned int> d_eta_node_counter(
      payload.eta_node_counter);
  vecmem::device_vector<gbts_sort_key_t> d_sort_keys(payload.sort_keys);
  vecmem::device_vector<unsigned int> d_sort_values(payload.sort_values);

  const unsigned int nSp = spacepoints.size();
  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int stride =
      thread_id.getBlockDimX() * thread_id.getGridDimX();

  // Every key slot of the capacity gets written: the unused tail sorts last.
  for (unsigned int globalIndex = globalIdx; globalIndex < payload.nSp;
       globalIndex += stride) {
    gbts_sort_key_t key = gbts_sort_key_rejected;
    if (globalIndex < nSp) {
      gbts_sort_key_t node_key = 0u;
      if (detail::gbts_bin_one_spacepoint(
              payload, spacepoints, measurements, volumeToLayerMap,
              surfaceToLayerMap, layerType, d_layer_info, d_layer_geo,
              reducedSP, d_eta_node_counter, globalIndex, node_key)) {
        key = node_key;
      }
    }
    d_sort_keys[globalIndex] = key;
    d_sort_values[globalIndex] = globalIndex;
  }
}

}  // namespace traccc::device
