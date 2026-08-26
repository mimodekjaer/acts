/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/gbts_seeding/device/gbts_bid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_rebid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reset_edge_bids.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// Payload of the whole seed-vs-edge bidding sequence: the initial bid
/// (gbts_bid_seeds_for_edges) followed by nRounds rounds of
/// gbts_rebid_seeds_for_edges + gbts_reset_edge_bids. The edge bids are
/// double buffered: the initial bid uses half 0, round r uses half
/// (r + 1) % 2 and its reset zeroes the other half for the next round.
struct gbts_seed_bidding_payload {
  /// Number of path-store rows (proposals are indexed by row)
  unsigned int nRows;
  /// Number of connected edges (size of one edge-bid buffer)
  unsigned int nConnectedEdges;
  /// Number of rebid / reset rounds
  unsigned int nRounds;
  /// Per-path (edge index, parent path-store index or -1) entries
  vecmem::data::vector_view<const int2> path_store;
  /// In/out: per-seed-proposal (quality, path-store index)
  vecmem::data::vector_view<int2> seed_proposals;
  /// In/out: per-seed-proposal ambiguity tag
  vecmem::data::vector_view<char> seed_ambiguity;
  /// In/out: per-edge highest-bidder seed proposal, 2 * nConnectedEdges
  /// entries (both halves zeroed on input)
  vecmem::data::vector_view<unsigned long long int> edge_bids;
  /// In/out: global atomic counter of rejected proposals
  unsigned int* nRejectedPropsCounter;
};

/// One half of the double-buffered edge bids
TRACCC_HOST_DEVICE inline vecmem::data::vector_view<unsigned long long int>
gbts_edge_bids_half(const gbts_seed_bidding_payload& p,
                    const unsigned int half) {
  return vecmem::data::vector_view<unsigned long long int>(
      p.nConnectedEdges, p.edge_bids.ptr() + half * p.nConnectedEdges);
}

/// Payload of the initial bid
TRACCC_HOST_DEVICE inline gbts_bid_seeds_for_edges_payload
gbts_make_bid_seeds_for_edges_payload(const gbts_seed_bidding_payload& p) {
  return {p.nRows, p.seed_proposals, p.seed_ambiguity,
          gbts_edge_bids_half(p, 0u), p.path_store};
}

/// Payload of the rebid of round @c round
TRACCC_HOST_DEVICE inline gbts_rebid_seeds_for_edges_payload
gbts_make_rebid_seeds_for_edges_payload(const gbts_seed_bidding_payload& p,
                                        const unsigned int round) {
  return {p.nRows,          p.path_store,
          p.seed_proposals, gbts_edge_bids_half(p, (round + 1u) % 2u),
          p.seed_ambiguity, p.nRejectedPropsCounter,
          round == 0u};
}

/// Payload of the reset of round @c round
TRACCC_HOST_DEVICE inline gbts_reset_edge_bids_payload
gbts_make_reset_edge_bids_payload(const gbts_seed_bidding_payload& p,
                                  const unsigned int round) {
  const unsigned int half = (round + 1u) % 2u;
  return {p.nRows,
          p.nConnectedEdges,
          p.path_store,
          p.seed_proposals,
          gbts_edge_bids_half(p, half),
          gbts_edge_bids_half(p, 1u - half),
          p.seed_ambiguity,
          p.nRejectedPropsCounter};
}

}  // namespace traccc::device
