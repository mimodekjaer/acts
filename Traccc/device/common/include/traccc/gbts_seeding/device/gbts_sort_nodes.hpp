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
#include "traccc/gbts_seeding/device/gbts_bin_spacepoints.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Global Event Data) Payload for the @c traccc::device::gbts_sort_nodes
/// function
struct gbts_sort_nodes_payload {
  /// Number of key / value slots to sort (the spacepoint capacity)
  unsigned int nKeys;
  /// Number of eta bins (bounds the significant key bits)
  unsigned int nEtaBins;
  /// Output: number of GBTS nodes (accepted spacepoints)
  unsigned int* nNodes;
  /// Output: per eta bin (begin, end) node ranges, flat
  vecmem::data::vector_view<unsigned int> eta_bin_views;
  /// In/out: per eta bin (min r, max r) as float bits, initialised by
  /// gbts_bin_spacepoints to (1e8, 0) and accumulated here
  vecmem::data::vector_view<unsigned int> bin_rads_bits;
  /// Reduced (x, y, z, cluster width) per spacepoint, in original order
  vecmem::data::vector_view<const float4> reducedSP;
  /// In/out: the node sort keys from gbts_bin_spacepoints (the spacepoint
  /// index in the low bits); gbts_sort_node_keys_kernel sorts them in place
  vecmem::data::vector_view<gbts_sort_key_t> sort_keys;
  /// Output: per-node (tau_min, tau_max, r, z), written in sorted order
  vecmem::data::vector_view<float4> node_params;
  /// Output: per-node phi, written in sorted order
  vecmem::data::vector_view<float> node_phi;
  /// Output: per-sorted-slot original spacepoint index
  vecmem::data::vector_view<unsigned int> node_index;
  /// Optional tau lookup table (used iff gbts_sort_nodes_params.useTauLUT)
  vecmem::data::vector_view<const float> tau_lut;
  /// Tau-prediction cuts read by @c device::gbts_sort_nodes
  traccc::gbts_sort_nodes_params gbts_sort_nodes_params;
};

/// Block size of the gbts_sort_nodes kernel (also the size of the shared
/// min / max arrays)
inline constexpr unsigned int gbts_sort_nodes_block_size = 256u;

/// (Shared Event Data) Payload for the @c traccc::device::gbts_sort_nodes
/// function
struct gbts_sort_nodes_shared_payload {
  /// Per-block min radius bits of the eta bins the block spans
  vecmem::data::vector_view<unsigned int> min_bits;
  /// Per-block max radius bits of the eta bins the block spans
  vecmem::data::vector_view<unsigned int> max_bits;
};

/// @brief Gather nodes into their (eta bin, phi, spacepoint)-sorted slots,
/// pack their geometry tuple and accumulate the per-eta-bin radius range.
///
/// The keys were sorted on their (eta bin, quantised phi) bits (stable in
/// the spacepoint index); thread i reads the spacepoint index of key i and
/// writes that spacepoint's node data at rank i, except inside a run of
/// equal (eta bin, quantised phi) where the exact (phi, index) rank is used
/// -- no atomics, deterministic node order.
///
/// The per-eta-bin node ranges and the node count come from the key
/// boundaries: a thread whose key starts a new eta bin (or is the first
/// rejected key) writes the begin of its bin, the end of the previous one and
/// the ranges of the empty bins in between.
///
/// The (min r, max r) of every eta bin is reduced in shared memory per block
/// (consecutive sorted nodes share their bins) and merged with a few global
/// atomic min / max per block.
///
/// @param[in] thread_id      Thread identifier for the kernel launch
/// @param[in] barrier        Block-wide barrier
/// @param[in] payload        The global memory payload
/// @param[in] shared_payload The shared memory payload
///
template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_sort_nodes(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_sort_nodes_payload& payload,
    const gbts_sort_nodes_shared_payload& shared_payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_sort_nodes.ipp"
