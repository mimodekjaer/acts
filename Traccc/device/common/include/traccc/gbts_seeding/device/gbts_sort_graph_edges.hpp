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
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// Buckets with at most this many edges are handled one thread per edge by
/// @c traccc::device::gbts_sort_graph_edges_small; larger ones by a block
/// in @c traccc::device::gbts_sort_graph_edges_large.
inline constexpr unsigned int gbts_sort_graph_edges_small_bucket = 32u;
/// Block size of @c traccc::device::gbts_sort_graph_edges_large
inline constexpr unsigned int gbts_sort_graph_edges_block_size = 128u;
/// Shared-memory cache capacity (bucket entries) of @c
/// traccc::device::gbts_sort_graph_edges_large; larger buckets are ranked
/// straight from global memory
inline constexpr unsigned int gbts_sort_graph_edges_cache_size = 1024u;

/// (Global Event Data) Payload for the @c traccc::device::gbts_sort_graph_edges
/// functions
struct gbts_sort_graph_edges_payload {
  /// Number of edges produced by gbts_make_graph_edges
  unsigned int nEdges;
  /// Number of nodes (= number of inner-node buckets)
  unsigned int nNodes;
  /// (outer, inner) node indices per edge, in race-assigned edge order
  vecmem::data::vector_view<const uint2> edge_nodes;
  /// Packed edge parameters per edge, in race-assigned edge order
  vecmem::data::vector_view<const short4> edge_params;
  /// Per-node bucket [begin, end) offsets (from gbts_link_graph_edges)
  vecmem::data::vector_view<const unsigned int> num_outgoing_edges;
  /// Outer node index of every edge, grouped by inner-node bucket
  vecmem::data::vector_view<const unsigned int> bucket_outer;
  /// The edge index matching each @c bucket_outer entry
  vecmem::data::vector_view<const unsigned int> bucket_edge;
  /// Output: (outer, inner) node indices in canonical edge order
  vecmem::data::vector_view<uint2> edge_nodes_sorted;
  /// Output: packed edge parameters in canonical edge order
  vecmem::data::vector_view<short4> edge_params_sorted;
};

/// (Shared Event Data) Payload for the @c
/// traccc::device::gbts_sort_graph_edges_large function
struct gbts_sort_graph_edges_shared_payload {
  /// Outer node indices of the bucket being ranked
  /// (gbts_sort_graph_edges_cache_size entries)
  vecmem::data::vector_view<unsigned int> bucket_cache;
  /// Per-thread "this node has a large bucket" flags
  /// (gbts_sort_graph_edges_block_size entries)
  vecmem::data::vector_view<unsigned int> large_flags;
};

/// @brief Gather the edges into a canonical, deterministic order.
///
/// The canonical index of an edge is its bucket begin offset plus its rank
/// by outer node index inside its inner node's bucket. (inner, outer) pairs
/// are unique, so the ranks are unique and the resulting order is a pure
/// function of the edge set, independent of the race-assigned indices
/// handed out by gbts_make_graph_edges. Each bucket then occupies the
/// contiguous canonical range [begin, end). No atomics.
///
/// Bucket sizes are very skewed (most nodes receive a handful of edges, a
/// few receive hundreds), so the work is split in two launches:
///  - @c gbts_sort_graph_edges_small: one thread per edge, for edges whose
///    bucket holds at most gbts_sort_graph_edges_small_bucket entries;
///  - @c gbts_sort_graph_edges_large: blocks stride over the nodes
///    block_size at a time (one node per thread to spot the large buckets),
///    and each larger bucket is staged in shared memory and ranked by the
///    whole block.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_sort_graph_edges_small(
    const thread_id_t& thread_id, const gbts_sort_graph_edges_payload& payload);

/// @brief Large-bucket half of the canonical edge ordering; see @c
/// traccc::device::gbts_sort_graph_edges_small.
///
/// @param[in]     thread_id      Thread identifier for the kernel launch
/// @param[in]     barrier        Block-level barrier
/// @param[in]     payload        The global memory payload
/// @param[in,out] shared_payload The shared memory payload
///
template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_sort_graph_edges_large(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_sort_graph_edges_payload& payload,
    const gbts_sort_graph_edges_shared_payload& shared_payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_sort_graph_edges.ipp"
