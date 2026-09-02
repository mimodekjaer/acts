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
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// Largest launch grid of gbts_make_graph_edges (the blocks stride over the
/// work items)
inline constexpr unsigned int gbts_make_graph_edges_max_blocks = 4096u;
/// Grid cap of the fill pass (one block per replayed work item; items
/// beyond the scratch are on the overflow list anyway)
inline constexpr unsigned int gbts_make_graph_edges_max_fill_blocks = 16384u;

/// (Global Event Data) Payload for the @c traccc::device::gbts_make_graph_edges
/// function (shared by the counting and the filling pass).
struct gbts_make_graph_edges_payload {
  /// Upper bound on the number of work items (sizes the edge_counts buffer
  /// and the launch grid)
  unsigned int nWorkMax;
  /// Number of work items, on the device (from gbts_build_edge_work_list)
  const unsigned int* nWork;
  /// In/out (fill pass only): next overflow item to process; the blocks
  /// grab the re-walked items from it (must be zero at launch)
  unsigned int* work_cursor;
  /// Per work item: (bin-pair index, chunk index within bin 1)
  vecmem::data::vector_view<const uint2> work_items;
  /// Per bin pair: first work item of the pair (nPairs + 1 entries, equal to
  /// the next entry when the pair has no work)
  vecmem::data::vector_view<const unsigned int> pair_work_begin;
  /// Per bin pair: (bin1, bin2), sorted by (bin1, bin2)
  vecmem::data::vector_view<const uint2> bin_pairs;
  /// Per bin pair: index of the first pair with the same bin1
  vecmem::data::vector_view<const unsigned int> pair_group_begin;
  /// Per eta bin: (begin, end) node ranges, flat
  vecmem::data::vector_view<const unsigned int> eta_bin_views;
  /// Per eta bin: (min r, max r), flat
  vecmem::data::vector_view<const float> bin_rads;
  /// Per-node (tau_min, tau_max, r, z)
  vecmem::data::vector_view<const float4> node_params;
  /// Per-node phi (sorted ascending within every eta bin)
  vecmem::data::vector_view<const float> node_phi;
  /// Per-bin-pair delta-phi window parameters
  traccc::gbts_dphi_window_params gbts_dphi_window_params;
  /// Edge-making geometric / kinematic cuts
  traccc::gbts_make_graph_edges_params gbts_make_graph_edges_params;
  /// class for compressing edge params to short4
  edge_params_converter edge_params_maker;
  /// Count pass output / fill pass input: edges per (work item, thread)
  vecmem::data::vector_view<unsigned int> edge_counts;
  /// Count pass output: edges per inner node written at [node + 1]; after
  /// the inclusive scan run by the launcher [node] / [node + 1] are the
  /// begin / end of the node's edge bucket and [nNodes] is the edge count
  vecmem::data::vector_view<unsigned int> num_outgoing_edges;
  /// Fill pass output: (outer node, inner node) per edge
  vecmem::data::vector_view<uint2> edge_nodes;
  /// Fill pass output: packed per-edge [eta, curv, phi_z, phi_w] used by
  /// matching
  vecmem::data::vector_view<short4> edge_params;
  /// Count pass output / fill pass input: the accepted outer nodes of every
  /// (work item, thread), up to gbts_make_graph_edges_scratch_edges each,
  /// slot-major: [(work * K + k) * blockSize + thread]; sized for
  /// scratch_work_items work items
  vecmem::data::vector_view<unsigned int> edge_scratch;
  /// Number of work items covered by edge_scratch / block_overflow
  unsigned int scratch_work_items;
  /// Count pass output / fill pass input: per work item, 1 when a thread
  /// of the block accepted more edges than fit into edge_scratch (the fill
  /// pass then re-walks the outer nodes for the whole block)
  vecmem::data::vector_view<unsigned char> block_overflow;
  /// Count pass output / fill pass input: the work items whose edges are not
  /// (completely) recorded in the scratch and must be re-walked by the fill
  /// pass (nWorkMax entries, arbitrary order)
  vecmem::data::vector_view<unsigned int> overflow_items;
  /// Count pass output: number of entries of @c overflow_items
  unsigned int* n_overflow;
  /// Count pass output / fill pass input: per work item (chunk begin, chunk
  /// size, delta-phi window bits, unused), so the replay needs one load
  /// instead of the dependent chain work item -> pair -> bins -> radii
  vecmem::data::vector_view<float4> item_info;
  /// Fill pass output: per-edge "kept" flag, initialised to 0 (later set by
  /// gbts_match_graph_edges)
  vecmem::data::vector_view<unsigned char> reindexer;
  /// Capacity of the edge buffers; edges beyond it are dropped (fill pass)
  unsigned int nEdgesMax;
  /// Output (fill pass): number of edges written, min(total, nEdgesMax)
  unsigned int* nEdges;
  /// Output (fill pass): number of edges found (before the cap)
  unsigned int* nEdgesTotal;
};

/// (Shared Event Data) Payload for the @c traccc::device::gbts_make_graph_edges
/// function
struct gbts_make_graph_edges_shared_payload {
  /// gbts_make_graph_edges_scratch_size unsigned ints, see the slot
  /// constants below
  vecmem::data::vector_view<unsigned int> work_slot;
};

/// Number of entries of gbts_make_graph_edges_shared_payload::work_slot
inline constexpr unsigned int gbts_make_graph_edges_scratch_size = 16u;
/// Slots of the shared work_slot array: the work item grabbed by the block
/// (fill pass) and the block-wide maximum edge count (count pass). Each use
/// alternates between its slot and the next one, so a slot is never
/// rewritten before every thread of the block has read it.
inline constexpr unsigned int gbts_make_graph_edges_slot_work = 0u;
inline constexpr unsigned int gbts_make_graph_edges_slot_max_count = 8u;

/// Accepted outer nodes recorded per (work item, thread) by the count pass
inline constexpr unsigned int gbts_make_graph_edges_scratch_edges = 16u;

/// Maximum number of work items covered by the edge scratch (memory cap:
/// scratch_edges * block size * 4 bytes per item)
inline constexpr unsigned int gbts_make_graph_edges_max_scratch_items = 16384u;
// A replayable item is only ever replayed by the block of the same index.
static_assert(gbts_make_graph_edges_max_fill_blocks >=
                  gbts_make_graph_edges_max_scratch_items,
              "every item with a scratch record needs a fill block");

/// @brief Create candidate edges between node pairs in compatible eta bins.
///
/// The blocks stride over the work items; a work item is a bin pair and one
/// gbts_consts::node_buffer_length-sized chunk of the pair's inner bin. Every
/// thread owns one inner node of the chunk and walks the phi-sorted outer bin
/// directly in global memory, testing the nodes inside its delta-phi window
/// against the geometric and kinematic cuts.
///
/// The function is run twice with the same decomposition:
/// - @c fill == false counts the edges of every thread (edge_counts) and
///   accumulates them per inner node (num_outgoing_edges[node + 1]);
/// - @c fill == true writes the edges. The write cursor of a thread is the
///   scanned bucket begin of its inner node plus the counts of the preceding
///   pairs of the same inner bin, so the edges come out directly in the
///   canonical (inner node bucket, outer node ascending) order without any
///   atomics: pairs are sorted by (bin1, bin2), outer bins are visited in
///   ascending node index and so are the nodes inside them.
///
/// @tparam fill                  false: counting pass, true: filling pass
/// @param[in] thread_id          Thread/block identifier (one block/work item)
/// @param[in] barrier            Block-wide barrier
/// @param[in,out] payload        The global memory payload
/// @param[in,out] shared_payload The shared memory payload
///
template <bool fill, concepts::thread_id1 thread_id_t,
          concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_make_graph_edges(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_make_graph_edges_payload& payload,
    const gbts_make_graph_edges_shared_payload& shared_payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_make_graph_edges.ipp"
