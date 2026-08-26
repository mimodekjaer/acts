/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/device/gbts_bin_spacepoints.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// System include(s).
#include <cstring>

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// Bit pattern of a float (for atomic min / max on non-negative floats)
TRACCC_HOST_DEVICE inline unsigned int gbts_float_bits(const float f) {
  static_assert(sizeof(float) == sizeof(unsigned int));
  unsigned int bits = 0u;
  std::memcpy(&bits, &f, sizeof(float));
  return bits;
}

/// Block size of the (single block) gbts_build_edge_work_list kernel; must
/// be a power of two.
inline constexpr unsigned int gbts_build_edge_work_list_block_size = 1024u;

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_build_edge_work_list function
struct gbts_build_edge_work_list_payload {
  /// Number of eta bins
  unsigned int nEtaBins;
  /// Number of bin pairs
  unsigned int nBinPairs;
  /// Chunk size of the inner bin of a graph-making work item
  unsigned int chunkSize;
  /// The sorted node keys (rejected keys last), nKeys entries
  vecmem::data::vector_view<const gbts_sort_key_t> sort_keys;
  /// Per bin pair: (bin1, bin2)
  vecmem::data::vector_view<const uint2> bin_pairs;
  /// Output: per eta bin (begin, end) node range, flat
  vecmem::data::vector_view<unsigned int> eta_bin_views;
  /// Output: per bin pair the first graph-making work item of the pair
  /// (nBinPairs + 1 entries; the last one is the number of work items)
  vecmem::data::vector_view<unsigned int> pair_work_begin;
  /// Output: per work item (bin pair, chunk of the inner bin); sized for the
  /// host-side upper bound of the work item count
  vecmem::data::vector_view<uint2> work_items;
  /// Output: per eta bin (min r, max r) as float bits, initialised to
  /// (1e8, 0) so that gbts_sort_nodes can accumulate them with atomic
  /// min / max
  vecmem::data::vector_view<unsigned int> bin_rads_bits;
  /// Output: total number of nodes
  unsigned int* nNodes;
  /// Output: total number of graph-making work items
  unsigned int* nWork;
};

/// (Shared Event Data) Payload for the @c
/// traccc::device::gbts_build_edge_work_list function
struct gbts_build_edge_work_list_shared_payload {
  /// Block scan scratch, gbts_build_edge_work_list_block_size entries
  vecmem::data::vector_view<unsigned int> scratch;
};

/// @brief Turn the per-eta-bin node counts into node ranges and lay out the
/// graph-making work items, entirely on the device (single block).
///
/// The node ranges come from binary searches of the sorted keys (no node
/// counting is needed); the work item offsets from a block scan of the
/// per-pair chunk counts. A work item is one
/// (bin pair, chunkSize-sized chunk of the pair's inner bin); pairs with an
/// empty bin get no work items.
///
/// @param[in] thread_id      Thread identifier (one block)
/// @param[in] barrier        Block-wide barrier
/// @param[in,out] payload    The global memory payload
/// @param[in] shared_payload The shared memory payload
///
template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_build_edge_work_list(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_build_edge_work_list_payload& payload,
    const gbts_build_edge_work_list_shared_payload& shared_payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_build_edge_work_list.ipp"
