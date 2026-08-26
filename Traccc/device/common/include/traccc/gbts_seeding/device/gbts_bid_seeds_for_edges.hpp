/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_bid_seeds_for_edges function
struct gbts_bid_seeds_for_edges_payload {
  /// Number of path-store rows
  unsigned int nRows;
  /// In/out: per-row (quality, row) proposal, {-1, -1} for rows that hold
  /// no proposal
  vecmem::data::vector_view<int2> seed_proposals;
  /// Output: per-seed-proposal ambiguity tag (multi-bid resolution flag)
  vecmem::data::vector_view<char> seed_ambiguity;
  /// In/out: per-edge highest-bidder seed proposal (packed 64-bit)
  vecmem::data::vector_view<unsigned long long int> edge_bids;
  /// Per-path (edge index, parent path-store index or -1) entries
  vecmem::data::vector_view<const int2> path_store;
};

/// @brief Have every seed proposal bid for its terminus edge.
///
/// Proposals are indexed by their path-store row, which is a deterministic
/// function of the graph; that index is the low half of every bid key in
/// the pipeline, so the disambiguation is reproducible without any sorting.
///
/// One thread per row places the single-edge bid that used to be folded
/// into the fitting kernel: proposals that lose their terminus edge are
/// tagged ambiguous and dropped by the first bidding round.
///
/// @param[in]     thread_id Thread identifier for the kernel launch
/// @param[in,out] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_bid_seeds_for_edges(
    const thread_id_t& thread_id,
    const gbts_bid_seeds_for_edges_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_bid_seeds_for_edges.ipp"
