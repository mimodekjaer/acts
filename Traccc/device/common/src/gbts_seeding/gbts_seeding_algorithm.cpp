/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/gbts_seeding/device/gbts_seeding_algorithm.hpp"

#include "traccc/gbts_seeding/gbts_seeding_config.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <algorithm>
#include <cstring>
#include <functional>
#include <memory_resource>
#include <utility>
#include <vector>

namespace traccc::device {

namespace {
/// Host memory resource for the algorithm's pinned static tables
vecmem::memory_resource* host_resource(const memory_resource& mr) {
  return (mr.host != nullptr) ? mr.host : std::pmr::get_default_resource();
}
}  // namespace

// Stage 1:
// Bin the spacepoints by layer in eta and phi,
// compute the node parameters (x, y, z, w)
// and the bin-wise min/max radius for the graph-building cuts.
auto gbts_seeding_algorithm::make_nodes(
    const edm::spacepoint_collection::const_view& spacepoints,
    const edm::measurement_collection::const_view& measurements,
    vecmem::data::vector_buffer<unsigned int>& zero_buf) const
    -> node_making_output {
  const gbts_seedfinder_config& cfg = m_config;
  // Every per-node buffer is sized for the capacity of the spacepoint
  // collection; the actual counts are only ever read on the device.
  const unsigned int nSp = spacepoints.capacity();
  // The named counters are the head of the zeroed buffer (see operator()).
  unsigned int* d_counters = zero_buf.ptr();

  // 0. Upload the static tables (one copy).
  vecmem::data::vector_buffer<unsigned char> static_blob(
      static_cast<unsigned int>(m_static_blob.size()), mr().main);
  copy().setup(static_blob)->ignore();
  copy()(vecmem::get_data(m_static_blob), static_blob)->ignore();
  const auto volumeToLayerMap_buf =
      section_view<const short>(static_blob, m_sec_volumeToLayerMap);
  const auto surfaceToLayerMap_buf =
      section_view<const std::pair<unsigned int, unsigned int>>(
          static_blob, m_sec_surfaceToLayerMap);
  const auto layerType_buf =
      section_view<const char>(static_blob, m_sec_layerType);
  const auto layer_info_buf =
      section_view<const std::pair<unsigned int, unsigned int>>(
          static_blob, m_sec_layer_info);
  const auto layer_geo_buf =
      section_view<const std::pair<float, float>>(static_blob, m_sec_layer_geo);
  const auto tau_lut_buf =
      section_view<const float>(static_blob, m_sec_tau_lut);
  const auto bin_pairs_buf =
      section_view<const uint2>(static_blob, m_sec_bin_pairs);

  // 1. Fused binning: assign each spacepoint a layer (or reject it), write
  //    its reduced parameters, count its eta bin and write its node sort
  //    key, all in a single pass. Rejected spacepoints and unused slots get a
  //    key that sorts last, so the sort below works on the full capacity.
  vecmem::data::vector_buffer<float4> reducedSP_buf(nSp, mr().main);
  copy().setup(reducedSP_buf)->ignore();

  vecmem::data::vector_buffer<gbts_sort_key_t> sort_keys_buf(nSp, mr().main);
  copy().setup(sort_keys_buf)->ignore();

  // Per-eta-bin (min r, max r): initialised by the binning kernel,
  // accumulated by the node gathering kernel (float bits, atomic min / max).
  vecmem::data::vector_buffer<float> bin_rads_buf(2 * cfg.n_eta_bins,
                                                  mr().main);
  copy().setup(bin_rads_buf)->ignore();
  const vecmem::data::vector_view<unsigned int> bin_rads_bits(
      2 * cfg.n_eta_bins, reinterpret_cast<unsigned int*>(bin_rads_buf.ptr()));
  vecmem::data::vector_buffer<unsigned int> eta_bin_views_buf(
      2 * cfg.n_eta_bins, mr().main);
  copy().setup(eta_bin_views_buf)->ignore();

  gbts_bin_spacepoints_kernel(
      {nSp, cfg.n_eta_bins, spacepoints, measurements, volumeToLayerMap_buf,
       surfaceToLayerMap_buf, layerType_buf, layer_info_buf, layer_geo_buf,
       reducedSP_buf, sort_keys_buf, cfg.volumeToLayerMap.size(),
       cfg.surfaceToLayerMap.size(), bin_rads_bits,
       cfg.gbts_count_spacepoints_by_layer_params});

  // Per-node outputs of the gather (node sorting) kernel.
  vecmem::data::vector_buffer<float4> node_params_buf(nSp, mr().main);
  copy().setup(node_params_buf)->ignore();
  vecmem::data::vector_buffer<float> node_phi_buf(nSp, mr().main);
  copy().setup(node_phi_buf)->ignore();
  vecmem::data::vector_buffer<unsigned int> node_index_buf(nSp, mr().main);
  copy().setup(node_index_buf)->ignore();

  // 2. Sort the node keys (rejected keys last), in place.
  const gbts_sort_nodes_payload sort_nodes_payload{
      nSp,
      cfg.n_eta_bins,
      d_counters + gbts_counter::nNodes,
      eta_bin_views_buf,
      bin_rads_bits,
      reducedSP_buf,
      sort_keys_buf,
      node_params_buf,
      node_phi_buf,
      node_index_buf,
      tau_lut_buf,
      cfg.gbts_sort_nodes_params};
  gbts_sort_node_keys_kernel(sort_nodes_payload);

  // 3. Gather the nodes into their sorted slots, pack their parameters and
  //    produce the eta-bin node ranges and the node count.
  gbts_sort_nodes_kernel(sort_nodes_payload);

  // 4. The graph-making work list, on the device (no synchronisation).

  vecmem::data::vector_buffer<unsigned int> pair_work_begin_buf(m_nBinPairs + 1,
                                                                mr().main);
  copy().setup(pair_work_begin_buf)->ignore();
  // Upper bound of the work item count: one chunk per pair plus the chunks
  // of the inner bins (each inner bin is shared by at most m_maxPairsPerBin1
  // pairs).
  const unsigned int nWorkMax =
      m_nBinPairs +
      m_maxPairsPerBin1 * (nSp / gbts_consts::node_buffer_length + 1u);
  vecmem::data::vector_buffer<uint2> work_items_buf(nWorkMax, mr().main);
  copy().setup(work_items_buf)->ignore();

  gbts_build_edge_work_list_kernel(
      {cfg.n_eta_bins, m_nBinPairs, gbts_consts::node_buffer_length,
       bin_pairs_buf, eta_bin_views_buf, pair_work_begin_buf, work_items_buf,
       d_counters + gbts_counter::nWork});

  return node_making_output{std::move(reducedSP_buf),
                            std::move(node_params_buf),
                            std::move(node_phi_buf),
                            std::move(node_index_buf),
                            std::move(bin_rads_buf),
                            std::move(eta_bin_views_buf),
                            std::move(pair_work_begin_buf),
                            std::move(work_items_buf),
                            nWorkMax,
                            nSp,
                            std::move(static_blob)};
}

// Stage 2:
// Find edges between compatible nodes
// The main output is a graph in
// the form of an edge list (array of node index pairs)
// and an accompanying array of edge parameters
// (exp(-eta), curvature, extrapolated phi at node1, extrapolated phi at node2).
auto gbts_seeding_algorithm::create_edges(
    vecmem::data::vector_buffer<float4> node_params,
    vecmem::data::vector_buffer<float> node_phi,
    vecmem::data::vector_buffer<unsigned int> node_index,
    const vecmem::data::vector_buffer<float>& bin_rads,
    const vecmem::data::vector_buffer<unsigned int>& eta_bin_views_buf,
    const vecmem::data::vector_buffer<unsigned int>& pair_work_begin_buf,
    const vecmem::data::vector_buffer<uint2>& work_items_buf,
    const unsigned int nWorkMax, const unsigned int nSp,
    const vecmem::data::vector_buffer<unsigned char>& static_blob,
    vecmem::data::vector_buffer<unsigned int>& zero_buf) const
    -> graph_making_output {
  const gbts_seedfinder_config& cfg = m_config;
  unsigned int* d_counters = zero_buf.ptr();
  const vecmem::data::vector_view<unsigned int> counters_view(
      gbts_counter::nCounters, d_counters);
  // The edge CSR ([node] = bucket begin after the scan) is the zeroed tail
  // of the shared buffer.
  const vecmem::data::vector_view<unsigned int> num_incoming_edges_buf(
      nSp + 1, d_counters + gbts_counter::nCounters);

  // 1. Count the edges per inner node, then write them in canonical (inner
  //    node bucket, outer node ascending) order. The work list lives on the
  //    device; its size is bounded by one chunk per pair plus the chunks of
  //    the inner bins (each inner bin is shared by at most m_maxPairsPerBin1
  //    pairs).
  vecmem::data::vector_buffer<unsigned int> edge_counts_buf(
      nWorkMax * gbts_consts::node_buffer_length, mr().main);
  // Scratch of accepted outer nodes recorded by the count pass so the fill
  // pass can replay them instead of walking the outer nodes again.
  const unsigned int scratch_work_items =
      std::min(nWorkMax, gbts_make_graph_edges_max_scratch_items);
  vecmem::data::vector_buffer<unsigned int> edge_scratch_buf(
      scratch_work_items * gbts_make_graph_edges_scratch_edges *
          gbts_consts::node_buffer_length,
      mr().main);
  copy().setup(edge_scratch_buf)->ignore();
  vecmem::data::vector_buffer<unsigned char> block_overflow_buf(
      scratch_work_items, mr().main);
  copy().setup(block_overflow_buf)->ignore();
  vecmem::data::vector_buffer<unsigned int> overflow_items_buf(nWorkMax,
                                                               mr().main);
  copy().setup(overflow_items_buf)->ignore();
  vecmem::data::vector_buffer<float4> item_info_buf(scratch_work_items,
                                                    mr().main);
  copy().setup(item_info_buf)->ignore();
  copy().setup(edge_counts_buf)->ignore();
  // setup edge param converter
  const float max_Kappa =
      std::max(cfg.gbts_make_graph_edges_params.max_Kappa_low_tau,
               cfg.gbts_make_graph_edges_params.max_Kappa_high_tau);
  edge_params_converter edge_param_converter(max_Kappa,
                                             cfg.gbts_sort_nodes_params.maxTau);

  const auto bin_pairs_buf =
      section_view<const uint2>(static_blob, m_sec_bin_pairs);
  const auto pair_group_begin_buf =
      section_view<const unsigned int>(static_blob, m_sec_pair_group_begin);
  gbts_make_graph_edges_payload make_graph_edges_payload{
      nWorkMax,
      d_counters + gbts_counter::nWork,
      d_counters + gbts_counter::workCursorCount,
      work_items_buf,
      pair_work_begin_buf,
      bin_pairs_buf,
      pair_group_begin_buf,
      eta_bin_views_buf,
      bin_rads,
      node_params,
      node_phi,
      cfg.gbts_dphi_window_params,
      cfg.gbts_make_graph_edges_params,
      edge_param_converter,
      edge_counts_buf,
      num_incoming_edges_buf,
      {},
      {},
      edge_scratch_buf,
      scratch_work_items,
      block_overflow_buf,
      overflow_items_buf,
      d_counters + gbts_counter::nOverflowItems,
      item_info_buf,
      {},
      0u,
      nullptr,
      nullptr};

  // Count pass + inclusive scan of the per-node counts (in the launcher).
  gbts_count_graph_edges_kernel(make_graph_edges_payload);

  // 2. Write the edges into buffers sized by the capacity; the edge count
  //    only lives on the device (deterministic truncation at the capacity,
  //    reported after the single synchronisation below).
  const unsigned int nEdgesMax = cfg.max_edges_per_spacepoint * nSp;
  // Packed per-edge parameter buffer ([exp(-eta), curv, phi_z, phi_w]).
  vecmem::data::vector_buffer<short4> edge_params_buf(nEdgesMax, mr().main);
  copy().setup(edge_params_buf)->ignore();
  vecmem::data::vector_buffer<uint2> edge_nodes_buf(nEdgesMax, mr().main);
  copy().setup(edge_nodes_buf)->ignore();
  // The per-edge "kept" flags are initialised by the fill pass (no memset).
  vecmem::data::vector_buffer<unsigned char> edge_kept_buf(nEdgesMax,
                                                           mr().main);
  copy().setup(edge_kept_buf)->ignore();
  vecmem::data::vector_buffer<int> reIndexer_buf(nEdgesMax, mr().main);
  copy().setup(reIndexer_buf)->ignore();

  make_graph_edges_payload.edge_nodes = edge_nodes_buf;
  make_graph_edges_payload.edge_params = edge_params_buf;
  make_graph_edges_payload.reindexer = edge_kept_buf;
  make_graph_edges_payload.nEdgesMax = nEdgesMax;
  make_graph_edges_payload.nEdges = d_counters + gbts_counter::nEdges;
  make_graph_edges_payload.nEdgesTotal = d_counters + gbts_counter::nEdgesTotal;
  make_graph_edges_payload.work_cursor =
      d_counters + gbts_counter::workCursorFill;
  // Fill pass: every edge lands in its canonical slot.
  gbts_make_graph_edges_kernel(make_graph_edges_payload);

  // 3. Edge matching to create edge-to-edge connections.
  vecmem::data::vector_buffer<unsigned char> num_neighbours_buf(nEdgesMax,
                                                                mr().main);
  copy().setup(num_neighbours_buf)->ignore();

  // Only the first num_neighbours[e] entries of an edge's row are ever read,
  // so the buffer does not need to be initialised.
  vecmem::data::vector_buffer<unsigned int> neighbours_buf(
      cfg.max_num_neighbours * nEdgesMax, mr().main);
  copy().setup(neighbours_buf)->ignore();

  gbts_match_graph_edges_kernel(
      {nEdgesMax, d_counters + gbts_counter::nEdges, cfg.max_num_neighbours,
       cfg.gbts_match_graph_edges_params, edge_params_buf, edge_nodes_buf,
       num_incoming_edges_buf, num_neighbours_buf, neighbours_buf,
       edge_kept_buf, edge_param_converter});

  gbts_reindex_edges_kernel({nEdgesMax, edge_kept_buf, reIndexer_buf});

  // No synchronisation: the kept-edge count (last entry of the scan) stays
  // on the device; the compacted graph is sized by its capacity and the
  // counters are read back asynchronously (checked at the next event).
  const unsigned int* d_nConnectedEdges = reinterpret_cast<const unsigned int*>(
      reIndexer_buf.ptr() + nEdgesMax - 1);
  const unsigned int nConnectedEdgesMax =
      cfg.max_connected_edges_per_spacepoint * nSp;

  const unsigned int nIntsPerEdge = 2 + 1 + cfg.max_num_neighbours;
  vecmem::data::vector_buffer<unsigned int> output_graph_buf(
      nConnectedEdgesMax * nIntsPerEdge, mr().main);
  copy().setup(output_graph_buf)->ignore();

  // CCA levels (double buffered); initialised to 1 by the compression
  // kernel so a level counts the maximum number of edge segments for a seed
  // originating at the edge.
  vecmem::data::vector_buffer<unsigned char> levels_buf(nConnectedEdgesMax,
                                                        mr().main);
  copy().setup(levels_buf)->ignore();
  // Per-edge "has a settled parent" CCA marks (zero-initialised for the
  // kept edges by the compression kernel).
  vecmem::data::vector_buffer<uint4> nei_cache_buf(nConnectedEdgesMax,
                                                   mr().main);
  copy().setup(nei_cache_buf)->ignore();
  vecmem::data::vector_buffer<unsigned char> has_parent_buf(nConnectedEdgesMax,
                                                            mr().main);
  copy().setup(has_parent_buf)->ignore();

  gbts_compress_graph_kernel(
      {nEdgesMax, d_counters + gbts_counter::nEdges, d_nConnectedEdges,
       nConnectedEdgesMax, cfg.max_num_neighbours, node_index, edge_nodes_buf,
       num_neighbours_buf, neighbours_buf, reIndexer_buf, output_graph_buf,
       has_parent_buf, levels_buf, nei_cache_buf});

  // The kept-edge count must outlive this stage's transient buffers: keep
  // a device-side copy in the (persistent) counters.
  unsigned int* d_nConnectedEdges_persistent =
      d_counters + gbts_counter::nConnectedEdges;
  copy()(
      vecmem::data::vector_view<const unsigned int>(1u, d_nConnectedEdges),
      vecmem::data::vector_view<unsigned int>(1u, d_nConnectedEdges_persistent))
      ->ignore();

  return graph_making_output{
      std::move(nei_cache_buf), std::move(output_graph_buf),
      std::move(levels_buf),    std::move(has_parent_buf),
      nConnectedEdgesMax,       d_nConnectedEdges_persistent};
}

// Stage 3:
// Find seed candidates as long chains of connected edges using a CCA
// Then fit the potential seeds (eta, phi, curvature).
// Finally, disambiguate them by repeated seed-vs-edge bidding rounds.
auto gbts_seeding_algorithm::extract_seeds(
    vecmem::data::vector_buffer<unsigned int>& output_graph,
    vecmem::data::vector_buffer<unsigned char>& levels,
    vecmem::data::vector_buffer<unsigned char>& has_parent,
    vecmem::data::vector_buffer<uint4>& nei_cache,
    vecmem::data::vector_buffer<float4>& reducedSP,
    const unsigned int nConnectedEdgesMax,
    const unsigned int* d_nConnectedEdges, const unsigned int nSp,
    const vecmem::data::vector_view<unsigned int>& counters_view,
    unsigned int* cca_scratch) const -> edm::seed_collection::buffer {
  const gbts_seedfinder_config& cfg = m_config;
  unsigned int* d_counters = counters_view.ptr();

  // 6. Find longest segments with the CCA (deterministic longest-path
  //    relaxation), then count the terminus rows.
  vecmem::data::vector_buffer<unsigned char>& levels_buf = levels;

  vecmem::data::vector_buffer<int2> outgoing_paths_buf(nConnectedEdgesMax,
                                                       mr().main);
  copy().setup(outgoing_paths_buf)->ignore();

  vecmem::data::vector_buffer<unsigned int> row_sizes_buf(nConnectedEdgesMax,
                                                          mr().main);
  copy().setup(row_sizes_buf)->ignore();

  // Edge bids, double buffered across the bidding rounds: the reset kernel
  // of round r zeroes the half used by round r + 1, so no memsets are needed
  // inside the loop. Both halves start zeroed (done by the terminus kernel).
  vecmem::data::vector_buffer<unsigned long long int> edge_bids_buf(
      2 * nConnectedEdgesMax, mr().main);
  copy().setup(edge_bids_buf)->ignore();
  vecmem::data::vector_buffer<unsigned long long int> hit_bids_buf(nSp,
                                                                   mr().main);
  copy().setup(hit_bids_buf)->ignore();

  // The path store gets a fixed capacity instead of a row-count readback.
  const unsigned int nRows = cfg.max_rows_per_spacepoint * nSp;
  vecmem::data::vector_buffer<char> seed_ambiguity_buf(nRows, mr().main);
  copy().setup(seed_ambiguity_buf)->ignore();

  // CCA sweeps + finishing pass, terminus counting (also zeroes the bids and
  // the ambiguity flags) and the row-size scan.
  gbts_run_cca_and_count_kernel(
      {nConnectedEdgesMax, d_nConnectedEdges, cfg.max_num_neighbours,
       cfg.minLevel, output_graph, levels_buf, outgoing_paths_buf, has_parent,
       0u, cca_scratch, nei_cache},
      {nConnectedEdgesMax, d_nConnectedEdges, outgoing_paths_buf, has_parent,
       row_sizes_buf, edge_bids_buf, hit_bids_buf,
       cca_scratch + traccc::device::gbts_run_cca_row_count_slot, nRows,
       seed_ambiguity_buf});

  // The row count stays on the device (written by the terminus step); the
  // path store gets a fixed capacity instead of a readback.
  const unsigned int* row_count =
      cca_scratch + traccc::device::gbts_run_cca_row_count_slot;
  // Launch-size hint only (the kernels grid-stride to the device count).
  const unsigned int nRowsGrid = nSp / 2u;

  vecmem::data::vector_buffer<int2> path_store_buf(nRows, mr().main);
  copy().setup(path_store_buf)->ignore();
  vecmem::data::vector_buffer<int2> seed_proposals_buf(nRows, mr().main);
  copy().setup(seed_proposals_buf)->ignore();

  // Lays out the path store and fits every path (segment fit fused).
  gbts_fill_path_store_kernel(
      {nRows, nRowsGrid, row_count, nConnectedEdgesMax, d_nConnectedEdges,
       cfg.max_num_neighbours, path_store_buf, output_graph, levels_buf,
       outgoing_paths_buf, row_sizes_buf, seed_proposals_buf,
       seed_ambiguity_buf, cfg.minLevel, reducedSP,
       d_counters + gbts_counter::nProps, cfg.gbts_fit_segments_params,
       cfg.gbts_make_graph_edges_params.max_z0,
       vecmem::data::vector_view<unsigned long long int>(nConnectedEdgesMax,
                                                         edge_bids_buf.ptr())});

  // 7. Disambiguate seeds through the initial bid and repeated seed-vs-edge
  //    bidding rounds. The proposal / rejection counts are not read back:
  //    every later kernel loops over the rows and the seed output is sized
  //    by the (upper bound) row count, which saves two synchronisations.
  // 8. Output buffer (at most two seeds per proposal, at most one proposal
  //    per row).
  const unsigned int nSeeds = nRows;
  edm::seed_collection::buffer output_seeds(
      2 * nSeeds, mr().main, vecmem::data::buffer_type::resizable);
  copy().setup(output_seeds)->ignore();

  // Bidding rounds (none by default), hit bidding and seed conversion.
  const unsigned int edge_size = 1u + 2u + cfg.max_num_neighbours;
  gbts_finish_seeds_kernel(
      {nRows, nRowsGrid, row_count, nConnectedEdgesMax, d_nConnectedEdges,
       cfg.edge_bidding_rounds, path_store_buf, seed_proposals_buf,
       seed_ambiguity_buf, edge_bids_buf, d_counters + gbts_counter::nRejected},
      {nRows, nRowsGrid, row_count, nSeeds, edge_size, output_graph,
       seed_proposals_buf, path_store_buf, seed_ambiguity_buf, hit_bids_buf,
       d_counters + gbts_counter::nRejected},
      {nRows, nRowsGrid, row_count, nSeeds, cfg.max_num_neighbours,
       seed_proposals_buf, seed_ambiguity_buf, path_store_buf, output_graph,
       reducedSP, output_seeds, hit_bids_buf, cfg.gbts_convert_seeds_params});

  // Deferred capacity checks: the counters of this event (including the
  // device-side connected-edge count copied into them) are read back
  // asynchronously and inspected at the start of the next event.
  copy()(counters_view, m_last_counters)->ignore();
  m_have_last_counters = true;
  m_last_nSp = nSp;

  // No synchronisation here: the caller reads the seed count.
  return output_seeds;
}

void gbts_seeding_algorithm::gbts_run_cca_and_count_kernel(
    const gbts_run_cca_iteration_payload& cca,
    const gbts_count_terminus_edges_payload& terminus) const {
  gbts_run_cca_kernel(cca);
  gbts_count_terminus_edges_kernel(terminus);
}

void gbts_seeding_algorithm::gbts_run_cca_kernel(
    const gbts_run_cca_iteration_payload& payload) const {
  gbts_run_cca_iteration_payload iteration = payload;
  // The relaxation sweeps (a sweep after convergence returns at once; at
  // most max_cca_iter, and no more than the bin-DAG depth requires) and the
  // finishing pass.
  // Levels of a chain of L edges settle after sweep L - 1, their counts one
  // sweep later: L sweeps for chains of at most L edges.
  const unsigned int nSweeps =
      std::min<unsigned int>(traccc::device::gbts_consts::max_cca_iter + 1u,
                             std::max(m_maxChainLength, 1u));
  for (unsigned int iter = 0; iter < nSweeps; ++iter) {
    iteration.iter = static_cast<unsigned char>(iter);
    gbts_run_cca_iteration_kernel(iteration);
  }
  iteration.iter = traccc::device::gbts_run_cca_finish_pass;
  gbts_run_cca_iteration_kernel(iteration);
}

void gbts_seeding_algorithm::gbts_finish_seeds_kernel(
    const gbts_seed_bidding_payload& bidding,
    const gbts_bid_seeds_for_hits_payload& hits,
    const gbts_convert_seeds_payload& convert) const {
  // The initial bid is placed by gbts_fill_path_store and the classification
  // by the hit bidding; the (optional) bidding rounds need the classification
  // first.
  if (bidding.nRounds > 0u) {
    gbts_bid_seeds_kernel(bidding);
  }
  gbts_bid_seeds_for_hits_kernel(hits);
  gbts_convert_seeds_kernel(convert);
}

void gbts_seeding_algorithm::gbts_bid_seeds_kernel(
    const gbts_seed_bidding_payload& payload) const {
  // Classify the proposals in a launch of their own (deterministic: no
  // bidding marks are written concurrently).
  gbts_rebid_seeds_for_edges_kernel(
      gbts_make_rebid_seeds_for_edges_payload(payload, 0u, true));
  for (unsigned int round = 0; round < payload.nRounds; ++round) {
    gbts_rebid_seeds_for_edges_kernel(
        gbts_make_rebid_seeds_for_edges_payload(payload, round, false));
    gbts_reset_edge_bids_kernel(
        gbts_make_reset_edge_bids_payload(payload, round));
  }
}

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const memory_resource& mr,
    const vecmem::copy& copy, std::unique_ptr<const Logger> callers_logger)
    : messaging(std::move(callers_logger)),
      algorithm_base{mr, copy},
      m_config{cfg},
      m_last_counters(gbts_counter::nCounters, host_resource(mr)),
      m_static_blob(host_resource(mr)) {
  // The edge-making kernel relies on the bin pairs being sorted by
  // (bin1, bin2) without duplicates: this is what makes the edges come out in
  // canonical order. Pairs referring to non-existent eta bins are dropped.
  std::vector<std::pair<unsigned int, unsigned int>>& binTables =
      m_config.binTables;
  const std::size_t nInput = binTables.size();
  std::erase_if(binTables,
                [this](const std::pair<unsigned int, unsigned int>& p) {
                  return (p.first >= m_config.n_eta_bins) ||
                         (p.second >= m_config.n_eta_bins);
                });
  if (binTables.size() != nInput) {
    TRACCC_ERROR("Dropped " << nInput - binTables.size()
                            << " bin pairs referring to eta bins >= "
                            << m_config.n_eta_bins);
  }
  std::ranges::sort(binTables);
  const auto duplicates = std::ranges::unique(binTables);
  if (!duplicates.empty()) {
    TRACCC_WARNING("Removed " << duplicates.size()
                              << " duplicate bin pairs from binTables");
    binTables.erase(duplicates.begin(), duplicates.end());
  }
  m_nBinPairs = static_cast<unsigned int>(binTables.size());
  // Depth of the bin DAG (pairs go from an inner to an outer bin): the
  // longest chain of edges is at most the longest path through the pairs.
  // Memoised longest path from every bin; the pairs are sorted by bin1.
  {
    std::vector<unsigned int> depth(m_config.n_eta_bins, 0u);
    std::vector<unsigned char> done(m_config.n_eta_bins, 0u);
    std::function<unsigned int(unsigned int)> longest =
        [&](unsigned int bin) -> unsigned int {
      if (done[bin] != 0u) {
        return depth[bin];
      }
      done[bin] = 1u;  // (cycles are impossible: pairs go outward)
      auto first =
          std::lower_bound(binTables.begin(), binTables.end(),
                           std::pair<unsigned int, unsigned int>{bin, 0u});
      unsigned int best = 0u;
      for (auto it = first; it != binTables.end() && it->first == bin; ++it) {
        best = std::max(best, 1u + longest(it->second));
      }
      depth[bin] = best;
      return best;
    };
    for (unsigned int bin = 0; bin < m_config.n_eta_bins; ++bin) {
      m_maxChainLength = std::max(m_maxChainLength, longest(bin));
    }
  }
  m_maxPairsPerBin1 = 0;
  for (unsigned int i = 0, run = 0; i < m_nBinPairs; i++) {
    run = (i > 0 && binTables[i - 1].first == binTables[i].first) ? run + 1 : 1;
    m_maxPairsPerBin1 = std::max(m_maxPairsPerBin1, run);
  }

  // Precompute the static per-pair tables.
  std::vector<uint2> bin_pairs(m_nBinPairs);
  std::vector<unsigned int> pair_group_begin(m_nBinPairs);
  for (unsigned int i = 0; i < m_nBinPairs; i++) {
    bin_pairs[i] = uint2{binTables[i].first, binTables[i].second};
    pair_group_begin[i] =
        (i > 0 && binTables[i - 1].first == binTables[i].first)
            ? pair_group_begin[i - 1]
            : i;
  }
  std::vector<float> tau_lut(cfg.tau_lut.begin(), cfg.tau_lut.end());
  if (tau_lut.empty()) {
    // A size-1 dummy so the sort-nodes kernel always gets a valid view.
    tau_lut.push_back(0.0f);
  }

  // Pack every static table into one pinned host blob (16-byte aligned
  // sections), uploaded with a single copy per event.
  std::vector<unsigned char> blob;
  auto pack = [&blob](const auto& table) {
    using value_t = typename std::decay_t<decltype(table)>::value_type;
    table_section section;
    section.offset = static_cast<unsigned int>((blob.size() + 15u) & ~15u);
    section.count = static_cast<unsigned int>(table.size());
    blob.resize(section.offset + section.count * sizeof(value_t));
    if (section.count > 0) {
      std::memcpy(blob.data() + section.offset, table.data(),
                  section.count * sizeof(value_t));
    }
    return section;
  };
  m_sec_volumeToLayerMap = pack(cfg.volumeToLayerMap);
  m_sec_surfaceToLayerMap = pack(cfg.surfaceToLayerMap);
  m_sec_layerType = pack(cfg.layerInfo.type);
  m_sec_layer_info = pack(cfg.layerInfo.info);
  m_sec_layer_geo = pack(cfg.layerInfo.geo);
  m_sec_tau_lut = pack(tau_lut);
  m_sec_bin_pairs = pack(bin_pairs);
  m_sec_pair_group_begin = pack(pair_group_begin);
  m_static_blob.assign(blob.begin(), blob.end());
}

auto gbts_seeding_algorithm::operator()(
    const edm::spacepoint_collection::const_view& spacepoints,
    const edm::measurement_collection::const_view& measurements) const
    -> output_type {
  // The capacity bounds the spacepoint count without a synchronisation; the
  // actual counts are only read on the device.
  const unsigned int nSp = spacepoints.capacity();
  TRACCC_DEBUG("nSp (capacity) " << nSp);
  if (nSp == 0) {
    TRACCC_WARNING("No spacepoints were found in the event");
    return {0, mr().main};
  }
  // The eta bin index has to fit into its node sort key field.
  if (m_config.n_eta_bins > gbts_sort_key_max_eta_bins) {
    TRACCC_ERROR("Too many eta bins (" << m_config.n_eta_bins << ") for the "
                                       << gbts_sort_key_eta_bits
                                       << "-bit node sort key field");
    return {0, mr().main};
  }

  // Capacity checks of the previous event (its counters were read back
  // asynchronously; the caller has synchronised since).
  if (m_have_last_counters) {
    const unsigned int edge_cap =
        m_config.max_edges_per_spacepoint * m_last_nSp;
    if (m_last_counters[gbts_counter::nEdgesTotal] > edge_cap) {
      TRACCC_WARNING("Previous event: edge buffer capacity ("
                     << edge_cap << ") exceeded, "
                     << m_last_counters[gbts_counter::nEdgesTotal] - edge_cap
                     << " edges were dropped; raise max_edges_per_spacepoint");
    }
    const unsigned int conn_cap =
        m_config.max_connected_edges_per_spacepoint * m_last_nSp;
    if (m_last_counters[gbts_counter::nConnectedEdges] > conn_cap) {
      TRACCC_WARNING("Previous event: compacted graph capacity ("
                     << conn_cap << ") exceeded, "
                     << m_last_counters[gbts_counter::nConnectedEdges] -
                            conn_cap
                     << " connected edges were dropped; raise "
                        "max_connected_edges_per_spacepoint");
    }
    m_have_last_counters = false;
  }

  // One zeroed buffer, [named counters | edge CSR (nSp + 1) | CCA scratch],
  // so a single memset per event initialises everything the kernels do not
  // initialise themselves.
  vecmem::data::vector_buffer<unsigned int> zero_buf(
      gbts_counter::nCounters + nSp + 1 +
          traccc::device::gbts_run_cca_scratch_size,
      mr().main);
  copy().setup(zero_buf)->ignore();
  copy().memset(zero_buf, 0)->ignore();
  const vecmem::data::vector_view<unsigned int> counters_view(
      gbts_counter::nCounters, zero_buf.ptr());
  unsigned int* cca_scratch = zero_buf.ptr() + gbts_counter::nCounters + nSp + 1;

  // Stage 1: bin spacepoints and create nodes with the parameters (eta, phi,
  // r, z). No synchronisation: an event without nodes produces no edges.
  node_making_output nodes = make_nodes(spacepoints, measurements, zero_buf);

  // Stage 2: graph. The per-node buffers are moved in so they are released
  // when create_edges returns, along with all the edge transients.
  graph_making_output graph = create_edges(
      std::move(nodes.node_params), std::move(nodes.node_phi),
      std::move(nodes.node_index), nodes.bin_rads, nodes.eta_bin_views_buf,
      nodes.pair_work_begin_buf, nodes.work_items_buf, nodes.nWorkMax,
      nodes.nSp, nodes.static_blob, zero_buf);
  if (graph.nConnectedEdgesMax == 0) {
    // A zero compacted-graph capacity (max_connected_edges_per_spacepoint
    // == 0) leaves nothing for seed extraction.
    return {0, mr().main};
  }

  // Stage 3: Create seeds from the graph edges.
  return extract_seeds(graph.output_graph, graph.levels, graph.has_parent,
                       graph.nei_cache, nodes.reducedSP,
                       graph.nConnectedEdgesMax, graph.d_nConnectedEdges, nSp,
                       counters_view, cca_scratch);
}

}  // namespace traccc::device
