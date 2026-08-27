/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/algorithm_base.hpp"
#include "traccc/gbts_seeding/device/gbts_bid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_bid_seeds_for_hits.hpp"
#include "traccc/gbts_seeding/device/gbts_bin_spacepoints.hpp"
#include "traccc/gbts_seeding/device/gbts_build_edge_work_list.hpp"
#include "traccc/gbts_seeding/device/gbts_compress_graph.hpp"
#include "traccc/gbts_seeding/device/gbts_convert_seeds.hpp"
#include "traccc/gbts_seeding/device/gbts_count_terminus_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_fill_path_store.hpp"
#include "traccc/gbts_seeding/device/gbts_fit_segments.hpp"
#include "traccc/gbts_seeding/device/gbts_make_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_match_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_rebid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reindex_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reset_edge_bids.hpp"
#include "traccc/gbts_seeding/device/gbts_run_cca_iteration.hpp"
#include "traccc/gbts_seeding/device/gbts_seed_bidding.hpp"
#include "traccc/gbts_seeding/device/gbts_sort_nodes.hpp"

// Project include(s).
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace traccc::device {

/// @brief Main algorithm for performing GBTS seeding on a device
/// (backend-agnostic).
///
/// The algorithm orchestrates the sequence of kernel launches and host-side
/// synchronisations.  Backend-specific subclasses are responsible for
/// implementing the individual kernel launchers.
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class gbts_seeding_algorithm
    : public algorithm<edm::seed_collection::buffer(
          const edm::spacepoint_collection::const_view&,
          const edm::measurement_collection::const_view&)>,
      public messaging,
      public algorithm_base {
 public:
  /// Constructor for the GBTS seed finding algorithm
  ///
  /// @param cfg The GBTS seed finding configuration
  /// @param mr The memory resource(s) to use in the algorithm
  /// @param copy The copy object to use for copying data between device
  ///             and host memory blocks
  /// @param logger The logger instance to use
  ///
  gbts_seeding_algorithm(
      const gbts_seedfinder_config& cfg, const memory_resource& mr,
      const vecmem::copy& copy,
      std::unique_ptr<const Logger> logger = getDummyLogger().clone());

  /// Destructor
  virtual ~gbts_seeding_algorithm() = default;

  /// Operator executing the algorithm.
  ///
  /// @param spacepoints is a view of all spacepoints in the event
  /// @param measurements is a view of all measurements in the event
  /// @return the buffer of track seeds reconstructed from the spacepoints
  ///
  output_type operator()(
      const edm::spacepoint_collection::const_view& spacepoints,
      const edm::measurement_collection::const_view& measurements)
      const override;

 protected:
  /// @name Kernel launchers (to be implemented by backends)
  ///
  /// Each launcher receives the payload of the device function it runs;
  /// the payload types are defined in the per-function device headers.
  /// @{

  /// Spacepoint-binning kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_bin_spacepoints_kernel(
      const gbts_bin_spacepoints_payload& payload) const = 0;

  /// Node sorting kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_sort_nodes_kernel(
      const gbts_sort_nodes_payload& payload) const = 0;

  /// Node key sorting launcher: sorts (sort_keys, sort_values) of the
  /// payload in place (a library sort, no kernel of our own)
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_sort_node_keys_kernel(
      const gbts_sort_nodes_payload& payload) const = 0;

  /// Edge work-list building kernel launcher (single block)
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_build_edge_work_list_kernel(
      const gbts_build_edge_work_list_payload& payload) const = 0;

  /// Graph edge-counting kernel launcher (gbts_make_graph_edges<false>)
  ///
  /// The implementation runs an inclusive prefix sum over
  /// payload.num_outgoing_edges after the kernel.
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_count_graph_edges_kernel(
      const gbts_make_graph_edges_payload& payload) const = 0;

  /// Graph edge-making kernel launcher (gbts_make_graph_edges<true>)
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_make_graph_edges_kernel(
      const gbts_make_graph_edges_payload& payload) const = 0;

  /// Graph edge-matching kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_match_graph_edges_kernel(
      const gbts_match_graph_edges_payload& payload) const = 0;

  /// Edge re-indexing launcher (in-place inclusive scan of the kept flags)
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_reindex_edges_kernel(
      const gbts_reindex_edges_payload& payload) const = 0;

  /// Graph compression kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_compress_graph_kernel(
      const gbts_compress_graph_payload& payload) const = 0;

  /// CCA (connected-components iteration) kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_run_cca_iteration_kernel(
      const gbts_run_cca_iteration_payload& payload) const = 0;

  /// Launcher of the complete CCA (gbts_consts::max_cca_iter iterations)
  ///
  /// The default implementation launches gbts_run_cca_iteration_kernel once
  /// per iteration; backends may fuse the iterations into one kernel.
  ///
  /// @param payload The payload of the first iteration (payload.iter == 0)
  ///
  virtual void gbts_run_cca_kernel(
      const gbts_run_cca_iteration_payload& payload) const;

  /// Launcher of the CCA followed by the terminus-edge counting and the
  /// inclusive scan of the row sizes (see gbts_count_terminus_edges_kernel)
  ///
  /// The default implementation calls gbts_run_cca_kernel and
  /// gbts_count_terminus_edges_kernel; backends may fuse everything.
  ///
  /// @param cca      The payload of the first CCA iteration
  /// @param terminus The payload of the terminus counting
  ///
  virtual void gbts_run_cca_and_count_kernel(
      const gbts_run_cca_iteration_payload& cca,
      const gbts_count_terminus_edges_payload& terminus) const;

  /// Launcher of the complete seed-vs-edge bidding sequence (initial bid +
  /// payload.nRounds rebid / reset rounds)
  ///
  /// The default implementation launches the three per-round kernels;
  /// backends may fuse the sequence into one kernel.
  ///
  /// @param payload The payload of the bidding sequence
  ///
  virtual void gbts_bid_seeds_kernel(
      const gbts_seed_bidding_payload& payload) const;

  /// Launcher of the whole seed finishing sequence: bidding (see
  /// gbts_bid_seeds_kernel), then gbts_bid_seeds_for_hits, then
  /// gbts_convert_seeds
  ///
  /// The default implementation launches the three steps separately;
  /// backends may fuse them into one kernel.
  ///
  /// @param bidding The payload of the bidding sequence
  /// @param hits    The payload of the hit bidding
  /// @param convert The payload of the seed conversion
  ///
  virtual void gbts_finish_seeds_kernel(
      const gbts_seed_bidding_payload& bidding,
      const gbts_bid_seeds_for_hits_payload& hits,
      const gbts_convert_seeds_payload& convert) const;

  /// Terminus-edge counting / path-store layout kernel launcher
  ///
  /// The implementation runs an inclusive prefix sum over the per-edge row
  /// counts after the kernel.
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_count_terminus_edges_kernel(
      const gbts_count_terminus_edges_payload& payload) const = 0;

  /// Path-store-filling kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_fill_path_store_kernel(
      const gbts_fill_path_store_payload& payload) const = 0;

  /// Initial edge-bid kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_bid_seeds_for_edges_kernel(
      const gbts_bid_seeds_for_edges_payload& payload) const = 0;

  /// Edge-bid reset kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_reset_edge_bids_kernel(
      const gbts_reset_edge_bids_payload& payload) const = 0;

  /// Edge re-bid kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_rebid_seeds_for_edges_kernel(
      const gbts_rebid_seeds_for_edges_payload& payload) const = 0;

  /// Seeds-bid-for-hits kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_bid_seeds_for_hits_kernel(
      const gbts_bid_seeds_for_hits_payload& payload) const = 0;

  /// GBTS seed conversion kernel launcher
  ///
  /// @param payload The payload for the kernel
  ///
  virtual void gbts_convert_seeds_kernel(
      const gbts_convert_seeds_payload& payload) const = 0;

  /// @}

 private:
  /// @name Pipeline stages
  ///
  /// The pipeline is split into three stage methods so that each stage's
  /// transient device buffers are local and freed as soon as the stage
  /// returns; only the cross-stage handles below survive between stages.
  /// @{

  /// Outputs of the node-making stage that are consumed downstream.
  struct node_making_output {
    /// Reduced (x, y, z, w) per original spacepoint (used by seed
    /// extraction)
    vecmem::data::vector_buffer<float4> reducedSP;
    /// Per-node (tau_min, tau_max, r, z) (used by graph making)
    vecmem::data::vector_buffer<float4> node_params;
    /// Per-node phi (used by graph making)
    vecmem::data::vector_buffer<float> node_phi;
    /// Per-sorted-slot original spacepoint index (used by graph making)
    vecmem::data::vector_buffer<unsigned int> node_index;
    /// Per-eta (rmin, rmax) pair, device (used by graph making)
    vecmem::data::vector_buffer<float> bin_rads;
    /// Per-eta (begin, end) node ranges, device (used by graph making)
    vecmem::data::vector_buffer<unsigned int> eta_bin_views_buf;
    /// Per bin pair: first graph-making work item (nBinPairs + 1 entries)
    vecmem::data::vector_buffer<unsigned int> pair_work_begin_buf;
    /// Per graph-making work item: (bin pair, chunk)
    vecmem::data::vector_buffer<uint2> work_items_buf;
    /// Upper bound of the work item count (size of work_items_buf)
    unsigned int nWorkMax = 0;
    /// Capacity of the spacepoint collection (upper bound of the node count)
    unsigned int nSp = 0;
    /// Device copy of the packed static tables (see m_static_blob)
    vecmem::data::vector_buffer<unsigned char> static_blob;
  };

  /// Outputs of the graph-making stage that are consumed by seed extraction.
  struct graph_making_output {
    /// Compacted, row-major graph
    vecmem::data::vector_buffer<unsigned int> output_graph;
    /// CCA levels (2 * nConnectedEdges, initialised to 1)
    vecmem::data::vector_buffer<unsigned char> levels;
    /// Per-edge "has a settled parent" CCA mark (zero-initialised)
    vecmem::data::vector_buffer<unsigned char> has_parent;
    /// Capacity of the compacted graph (sizes output_graph / levels)
    unsigned int nConnectedEdgesMax = 0;
    /// Number of edges that survived re-indexing, on the device (capped at
    /// nConnectedEdgesMax; the second CCA levels buffer starts at this
    /// offset)
    const unsigned int* d_nConnectedEdges = nullptr;
    /// Number of edges that survived re-indexing (0 == nothing to do)
  };

  /// Stage 1: count, bin, sort and characterise nodes. Fully asynchronous:
  /// the node count and the graph-making work list live on the device.
  node_making_output make_nodes(
      const edm::spacepoint_collection::const_view& spacepoints,
      const edm::measurement_collection::const_view& measurements,
      vecmem::data::vector_buffer<unsigned int>& zero_buf) const;

  /// Stage 2: build, link, match and compress the edge graph. The per-node
  /// buffers are taken by value so they are released when this stage returns.
  graph_making_output create_edges(
      vecmem::data::vector_buffer<float4> node_params,
      vecmem::data::vector_buffer<float> node_phi,
      vecmem::data::vector_buffer<unsigned int> node_index,
      const vecmem::data::vector_buffer<float>& bin_rads,
      const vecmem::data::vector_buffer<unsigned int>& eta_bin_views_buf,
      const vecmem::data::vector_buffer<unsigned int>& pair_work_begin_buf,
      const vecmem::data::vector_buffer<uint2>& work_items_buf,
      const unsigned int nWorkMax, const unsigned int nSp,
      const vecmem::data::vector_buffer<unsigned char>& static_blob,
      vecmem::data::vector_buffer<unsigned int>& zero_buf) const;

  /// Stage 3: run the CCA, extract paths, fit and disambiguate into seeds.
  edm::seed_collection::buffer extract_seeds(
      vecmem::data::vector_buffer<unsigned int>& output_graph,
      vecmem::data::vector_buffer<unsigned char>& levels,
      vecmem::data::vector_buffer<unsigned char>& has_parent,
      vecmem::data::vector_buffer<float4>& reducedSP,
      const unsigned int nConnectedEdgesMax,
      const unsigned int* d_nConnectedEdges, const unsigned int nSp,
      const vecmem::data::vector_view<unsigned int>& counters_view,
      vecmem::vector<unsigned int>& h_counters) const;

  /// @}

  /// GBTS seed-finding configuration (binTables sorted by (bin1, bin2) and
  /// de-duplicated by the constructor).
  gbts_seedfinder_config m_config;
  /// Number of bin pairs in m_config.binTables
  unsigned int m_nBinPairs = 0;
  /// Largest number of bin pairs sharing one inner bin
  unsigned int m_maxPairsPerBin1 = 0;
  /// Counters of the previous event (pinned host memory, copied
  /// asynchronously at the end of graph making; checked at the start of
  /// the next event for capacity overflows)
  mutable vecmem::vector<unsigned int> m_last_counters;
  mutable bool m_have_last_counters = false;
  mutable unsigned int m_last_nSp = 0;
  /// @name Static tables, packed into one (pinned) host blob that is
  /// uploaded with a single copy per event
  /// @{
  /// A table inside the blob: byte offset and element count
  struct table_section {
    unsigned int offset = 0;
    unsigned int count = 0;
  };
  /// Typed view of a table section of a device copy of the blob
  template <typename T>
  static vecmem::data::vector_view<T> section_view(
      const vecmem::data::vector_buffer<unsigned char>& blob,
      const table_section& section) {
    return vecmem::data::vector_view<T>(
        section.count, reinterpret_cast<T*>(blob.ptr() + section.offset));
  }
  /// The packed static tables (pinned host memory)
  vecmem::vector<unsigned char> m_static_blob;
  table_section m_sec_volumeToLayerMap;
  table_section m_sec_surfaceToLayerMap;
  table_section m_sec_layerType;
  table_section m_sec_layer_info;
  table_section m_sec_layer_geo;
  table_section m_sec_tau_lut;
  /// m_config.binTables as (bin1, bin2)
  table_section m_sec_bin_pairs;
  /// Per bin pair: index of the first pair with the same bin1
  table_section m_sec_pair_group_begin;
  /// @}

};  // class gbts_seeding_algorithm

}  // namespace traccc::device
