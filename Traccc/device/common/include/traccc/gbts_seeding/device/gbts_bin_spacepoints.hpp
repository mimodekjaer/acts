/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <bit>
#include <utility>

namespace traccc::device {

/// Type of the node sort key
using gbts_sort_key_t = unsigned int;

/// Node sort key layout, from the least significant bit up: the quantised
/// phi, then the eta bin. Sorting the keys groups the nodes by eta bin and
/// orders them by phi up to the quantisation; the keys are written in
/// spacepoint order, so a stable sort breaks ties by spacepoint index, and
/// gbts_sort_nodes restores the exact (phi, spacepoint index) order inside
/// every run of equal keys.
inline constexpr unsigned int gbts_sort_key_phi_bits = 20u;
inline constexpr unsigned int gbts_sort_key_eta_bits =
    32u - gbts_sort_key_phi_bits;
inline constexpr unsigned int gbts_sort_key_eta_shift = gbts_sort_key_phi_bits;
/// Largest number of eta bins the key's eta field can hold (one value is
/// reserved for the "rejected" key)
inline constexpr unsigned int gbts_sort_key_max_eta_bins =
    (1u << gbts_sort_key_eta_bits) - 1u;
/// Key of a rejected spacepoint / unused slot: sorts after every node key
inline constexpr gbts_sort_key_t gbts_sort_key_rejected = ~0u;

/// Quantised phi (monotone in phi): the low gbts_sort_key_phi_bits key bits
TRACCC_HOST_DEVICE inline unsigned int gbts_quantised_phi(const float phi) {
  constexpr float scale =
      static_cast<float>(1u << gbts_sort_key_phi_bits) / TWO_PI_F;
  const float q = (phi + PI_F) * scale;
  if (q <= 0.0f) {
    return 0u;
  }
  const unsigned int qi = static_cast<unsigned int>(q);
  constexpr unsigned int max_q = (1u << gbts_sort_key_phi_bits) - 1u;
  return (qi < max_q) ? qi : max_q;
}

/// (Global Event Data) Payload for the @c traccc::device::gbts_bin_spacepoints
/// function
///
/// Each spacepoint is read once: it is assigned a GBTS layer (or rejected),
/// its reduced parameters are written, its eta bin's node count is bumped,
/// and its node sort key is appended to the compacted key array, all in a
/// single pass.
struct gbts_bin_spacepoints_payload {
  /// Capacity of the spacepoint collection (the event's spacepoint count is
  /// read on the device); also the size of the key / value arrays
  unsigned int nSp;
  /// Number of eta bins
  unsigned int nEtaBins;
  /// All spacepoints in the event
  edm::spacepoint_collection::const_view spacepoints;
  /// All measurements in the event (used to look up surface IDs)
  edm::measurement_collection::const_view measurements;
  /// Map from detector volume index to GBTS layer index
  vecmem::data::vector_view<const short> volumeToLayerMap;
  /// Map from (volume, surface) pair to GBTS layer index (optional)
  vecmem::data::vector_view<const std::pair<unsigned int, unsigned int>>
      surfaceToLayerMap;
  /// Per-layer type code (barrel/endcap/etc.) used for cluster-width cuts
  vecmem::data::vector_view<const char> layerType;
  /// Per-layer (first eta bin, number of eta bins) pair
  vecmem::data::vector_view<const std::pair<unsigned int, unsigned int>>
      layer_info;
  /// Per-layer geometry pair used to compute eta (e.g. (rmin, zmax))
  vecmem::data::vector_view<const std::pair<float, float>> layer_geo;
  /// Output: reduced (x, y, z, cluster width) per spacepoint after filtering
  vecmem::data::vector_view<float4> reducedSP;
  /// Output: one node sort key per spacepoint slot (nSp entries),
  /// (eta bin << gbts_sort_key_eta_shift) | quantised phi for accepted
  /// spacepoints and gbts_sort_key_rejected for rejected / unused slots, so
  /// that after a stable sort the nodes come first, grouped by eta bin and
  /// ordered by (quantised phi, spacepoint index).
  vecmem::data::vector_view<gbts_sort_key_t> sort_keys;
  /// Output: the spacepoint index matching each @c sort_keys slot;
  /// sorted alongside @c sort_keys by the gbts_sort_nodes launcher
  vecmem::data::vector_view<unsigned int> sort_values;
  /// Size of the volume-to-layer map (for bounds checking)
  unsigned long int volumeMapSize;
  /// Size of the surface-to-layer map (for bounds checking)
  unsigned long int surfaceMapSize;
  /// Parameters for SP filtering (passed through from config, used for tau
  /// cut if enabled)
  traccc::gbts_count_spacepoints_by_layer_params
      gbts_count_spacepoints_by_layer_params;
};

/// @brief Per-spacepoint binning kernel: look up the GBTS layer via the
/// volume / surface map, optionally apply a cluster-width cut, and on
/// acceptance write the reduced (x, y, z, width) tuple, bump the node's eta
/// bin count and write its node sort key -- all from a single read of the
/// source spacepoint. Keys are written in spacepoint order (no atomics);
/// rejected spacepoints and unused slots get gbts_sort_key_rejected.
///
/// Precondition (checked by make_nodes): n_eta_bins <=
/// gbts_sort_key_max_eta_bins, so the eta field of every key fits.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_bin_spacepoints(
    const thread_id_t& thread_id, const gbts_bin_spacepoints_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_bin_spacepoints.ipp"
