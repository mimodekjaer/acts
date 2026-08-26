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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory_resource>
#include <unordered_map>
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
  // Layout of the zeroed buffer: [counters | eta node counters | edge CSR]
  unsigned int* d_counters = zero_buf.ptr();
  const vecmem::data::vector_view<unsigned int> eta_node_counter_buf(
      cfg.n_eta_bins, zero_buf.ptr() + gbts_counter::nCounters);

  // Check that the number of eta bins is compatible with the node sort key
  // field width.
  if (cfg.n_eta_bins > gbts_sort_key_max_eta_bins) {
    TRACCC_ERROR("Too many eta bins (" << cfg.n_eta_bins << ") for the "
                                       << gbts_sort_key_eta_bits
                                       << "-bit node sort key field");
    return node_making_output{};
  }

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
  vecmem::data::vector_buffer<unsigned int> sort_values_buf(nSp, mr().main);
  copy().setup(sort_values_buf)->ignore();

  gbts_bin_spacepoints_kernel(
      {nSp, cfg.n_eta_bins, spacepoints, measurements, volumeToLayerMap_buf,
       surfaceToLayerMap_buf, layerType_buf, layer_info_buf, layer_geo_buf,
       reducedSP_buf, eta_node_counter_buf, sort_keys_buf, sort_values_buf,
       cfg.volumeToLayerMap.size(), cfg.surfaceToLayerMap.size(),
       cfg.gbts_count_spacepoints_by_layer_params});

  // 2. Node ranges of the eta bins and the graph-making work list, on the
  //    device (no synchronisation).

  vecmem::data::vector_buffer<unsigned int> eta_bin_views_buf(
      2 * cfg.n_eta_bins, mr().main);
  copy().setup(eta_bin_views_buf)->ignore();
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
       eta_node_counter_buf, bin_pairs_buf, eta_bin_views_buf,
       pair_work_begin_buf, work_items_buf, d_counters + gbts_counter::nNodes,
       d_counters + gbts_counter::nWork});

  // 3. Sort the nodes and pack their parameters.
  vecmem::data::vector_buffer<float4> node_params_buf(nSp, mr().main);
  copy().setup(node_params_buf)->ignore();
  vecmem::data::vector_buffer<float> node_phi_buf(nSp, mr().main);
  copy().setup(node_phi_buf)->ignore();
  vecmem::data::vector_buffer<unsigned int> node_index_buf(nSp, mr().main);
  copy().setup(node_index_buf)->ignore();

  gbts_sort_nodes_kernel(
      {nSp, cfg.n_eta_bins, d_counters + gbts_counter::nNodes, reducedSP_buf,
       sort_keys_buf, sort_values_buf, node_params_buf, node_phi_buf,
       node_index_buf, tau_lut_buf, cfg.gbts_sort_nodes_params});

  vecmem::data::vector_buffer<float> bin_rads_buf(2 * cfg.n_eta_bins,
                                                  mr().main);
  copy().setup(bin_rads_buf)->ignore();

  gbts_find_minmax_radius_kernel(
      {cfg.n_eta_bins, eta_bin_views_buf, node_params_buf, bin_rads_buf});

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
    vecmem::data::vector_buffer<unsigned int>& zero_buf,
    vecmem::vector<unsigned int>& h_counters) const -> graph_making_output {
  const gbts_seedfinder_config& cfg = m_config;
  unsigned int* d_counters = zero_buf.ptr();
  const vecmem::data::vector_view<unsigned int> counters_view(
      gbts_counter::nCounters, d_counters);
  // The edge CSR ([node] = bucket begin after the scan) is the zeroed tail
  // of the shared buffer.
  const vecmem::data::vector_view<unsigned int> num_incoming_edges_buf(
      nSp + 1, d_counters + gbts_counter::nCounters + cfg.n_eta_bins);

  // 1. Count the edges per inner node, then write them in canonical (inner
  //    node bucket, outer node ascending) order. The work list lives on the
  //    device; its size is bounded by one chunk per pair plus the chunks of
  //    the inner bins (each inner bin is shared by at most m_maxPairsPerBin1
  //    pairs).
  vecmem::data::vector_buffer<unsigned int> edge_counts_buf(
      nWorkMax * gbts_consts::node_buffer_length, mr().main);
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

  // The one synchronisation of the graph making: the number of kept edges
  // (last entry of the scan) sizes the compacted graph.
  copy()(counters_view, h_counters)->ignore();
  int h_nConnectedEdges = 0;
  copy()(vecmem::data::vector_view<const int>(
             1u, reIndexer_buf.ptr() + nEdgesMax - 1),
         vecmem::data::vector_view<int>(1u, &h_nConnectedEdges))
      ->wait();
  const unsigned int nEdges = h_counters[gbts_counter::nEdges];
  const unsigned int nConnectedEdges =
      static_cast<unsigned int>(h_nConnectedEdges);
  TRACCC_DEBUG("Created " << nEdges << " edges, found " << nConnectedEdges
                          << " connected edges for seed extraction");
  if (h_counters[gbts_counter::nEdgesTotal] > nEdgesMax) {
    TRACCC_WARNING("Edge buffer capacity ("
                   << nEdgesMax << ") exceeded: "
                   << h_counters[gbts_counter::nEdgesTotal] - nEdgesMax
                   << " edges were dropped; raise max_edges_per_spacepoint");
  }
  if (nConnectedEdges == 0) {
    TRACCC_WARNING("No connected edges were found");
    return graph_making_output{};
  }

  const unsigned int nIntsPerEdge = 2 + 1 + cfg.max_num_neighbours;
  vecmem::data::vector_buffer<unsigned int> output_graph_buf(
      nConnectedEdges * nIntsPerEdge, mr().main);
  copy().setup(output_graph_buf)->ignore();

  // CCA levels (double buffered); initialised to 1 by the compression
  // kernel so a level counts the maximum number of edge segments for a seed
  // originating at the edge.
  vecmem::data::vector_buffer<unsigned char> levels_buf(2 * nConnectedEdges,
                                                        mr().main);
  copy().setup(levels_buf)->ignore();

  gbts_compress_graph_kernel(
      {nEdgesMax, d_counters + gbts_counter::nEdges, nConnectedEdges,
       cfg.max_num_neighbours, node_index, edge_nodes_buf, num_neighbours_buf,
       neighbours_buf, reIndexer_buf, output_graph_buf, levels_buf});

  return graph_making_output{std::move(output_graph_buf), std::move(levels_buf),
                             nConnectedEdges};
}

// Stage 3:
// Find seed candidates as long chains of connected edges using a CCA
// Then fit the potential seeds (eta, phi, curvature).
// Finally, disambiguate them by repeated seed-vs-edge bidding rounds.
auto gbts_seeding_algorithm::extract_seeds(
    vecmem::data::vector_buffer<unsigned int>& output_graph,
    vecmem::data::vector_buffer<unsigned char>& levels,
    vecmem::data::vector_buffer<float4>& reducedSP,
    const unsigned int nConnectedEdges, const unsigned int nSp,
    const vecmem::data::vector_view<unsigned int>& counters_view,
    vecmem::vector<unsigned int>& h_counters) const
    -> edm::seed_collection::buffer {
  const gbts_seedfinder_config& cfg = m_config;
  unsigned int* d_counters = counters_view.ptr();
  // The device counters of this stage are only used by the kernels.
  static_cast<void>(h_counters);

  // 6. Find longest segments with CCA.
  // active_edges is the per-edge "next iter index" flag: it holds `iter`
  // while the edge is active in iteration `iter`, and -1 once it settles.
  // Iteration 0 writes every entry before any later iteration reads it, so
  // no initialisation is required.
  vecmem::data::vector_buffer<char> active_edges_buf(nConnectedEdges,
                                                     mr().main);
  copy().setup(active_edges_buf)->ignore();

  vecmem::data::vector_buffer<unsigned char>& levels_buf = levels;

  vecmem::data::vector_buffer<int2> outgoing_paths_buf(nConnectedEdges,
                                                       mr().main);
  copy().setup(outgoing_paths_buf)->ignore();

  // Per-iteration active-edge counters for fused CCA implementations
  // (initialised by the kernel itself).
  vecmem::data::vector_buffer<unsigned int> cca_active_buf(
      traccc::device::gbts_consts::max_cca_iter + 1u, mr().main);
  copy().setup(cca_active_buf)->ignore();

  gbts_run_cca_kernel({nConnectedEdges, cfg.max_num_neighbours, cfg.minLevel,
                       output_graph, levels_buf, active_edges_buf,
                       outgoing_paths_buf, 0u, cca_active_buf.ptr()});

  vecmem::data::vector_buffer<unsigned int> row_sizes_buf(nConnectedEdges,
                                                          mr().main);
  copy().setup(row_sizes_buf)->ignore();

  // Edge bids, double buffered across the bidding rounds: the reset kernel
  // of round r zeroes the half used by round r + 1, so no memsets are needed
  // inside the loop. Both halves start zeroed (done by the terminus kernel).
  vecmem::data::vector_buffer<unsigned long long int> edge_bids_buf(
      2 * nConnectedEdges, mr().main);
  copy().setup(edge_bids_buf)->ignore();
  vecmem::data::vector_buffer<unsigned long long int> hit_bids_buf(nSp,
                                                                   mr().main);
  copy().setup(hit_bids_buf)->ignore();

  gbts_count_terminus_edges_kernel({nConnectedEdges, outgoing_paths_buf,
                                    row_sizes_buf, edge_bids_buf,
                                    hit_bids_buf});

  // The row count stays on the device (last entry of the scanned row
  // sizes); the path store gets a fixed capacity instead of a readback.
  const unsigned int* row_count = row_sizes_buf.ptr() + nConnectedEdges - 1;
  const unsigned int nRows = cfg.max_rows_per_connected_edge * nConnectedEdges;
  const unsigned int nRowsGrid = nConnectedEdges;

  vecmem::data::vector_buffer<int2> path_store_buf(nRows, mr().main);
  copy().setup(path_store_buf)->ignore();
  vecmem::data::vector_buffer<int2> seed_proposals_buf(nRows, mr().main);
  copy().setup(seed_proposals_buf)->ignore();
  vecmem::data::vector_buffer<char> seed_ambiguity_buf(nRows, mr().main);
  copy().setup(seed_ambiguity_buf)->ignore();

  // Lays out the path store and fits every path (segment fit fused).
  gbts_fill_path_store_kernel(
      {nRows, nRowsGrid, row_count, nConnectedEdges, cfg.max_num_neighbours,
       path_store_buf, output_graph, levels_buf, outgoing_paths_buf,
       row_sizes_buf, seed_proposals_buf, seed_ambiguity_buf, cfg.minLevel,
       reducedSP, d_counters + gbts_counter::nProps,
       cfg.gbts_fit_segments_params, cfg.gbts_make_graph_edges_params.max_z0});

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

  // Bidding rounds, hit bidding and seed conversion (fused by the CUDA
  // backend into one cooperative kernel).
  const unsigned int edge_size = 1u + 2u + cfg.max_num_neighbours;
  gbts_finish_seeds_kernel(
      {nRows, nRowsGrid, row_count, nConnectedEdges, cfg.edge_bidding_rounds,
       path_store_buf, seed_proposals_buf, seed_ambiguity_buf, edge_bids_buf,
       d_counters + gbts_counter::nRejected},
      {nRows, nRowsGrid, row_count, nSeeds, edge_size, output_graph,
       seed_proposals_buf, path_store_buf, seed_ambiguity_buf, hit_bids_buf},
      {nRows, nRowsGrid, row_count, nSeeds, cfg.max_num_neighbours,
       seed_proposals_buf, seed_ambiguity_buf, path_store_buf, output_graph,
       reducedSP, output_seeds, hit_bids_buf, cfg.gbts_convert_seeds_params});

  // No synchronisation here: the caller reads the seed count.
  return output_seeds;
}

void gbts_seeding_algorithm::gbts_run_cca_kernel(
    const gbts_run_cca_iteration_payload& payload) const {
  gbts_run_cca_iteration_payload iteration = payload;
  for (unsigned char iter = 0; iter < traccc::device::gbts_consts::max_cca_iter;
       ++iter) {
    iteration.iter = iter;
    gbts_run_cca_iteration_kernel(iteration);
  }
}

void gbts_seeding_algorithm::gbts_finish_seeds_kernel(
    const gbts_seed_bidding_payload& bidding,
    const gbts_bid_seeds_for_hits_payload& hits,
    const gbts_convert_seeds_payload& convert) const {
  gbts_bid_seeds_kernel(bidding);
  gbts_bid_seeds_for_hits_kernel(hits);
  gbts_convert_seeds_kernel(convert);
}

void gbts_seeding_algorithm::gbts_bid_seeds_kernel(
    const gbts_seed_bidding_payload& payload) const {
  gbts_bid_seeds_for_edges_kernel(
      gbts_make_bid_seeds_for_edges_payload(payload));
  for (unsigned int round = 0; round < payload.nRounds; ++round) {
    gbts_rebid_seeds_for_edges_kernel(
        gbts_make_rebid_seeds_for_edges_payload(payload, round));
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

  // One zeroed buffer for the named counters, the eta node counters and the
  // edge CSR: a single memset per event.
  vecmem::data::vector_buffer<unsigned int> zero_buf(
      gbts_counter::nCounters + m_config.n_eta_bins + nSp + 1, mr().main);
  copy().setup(zero_buf)->ignore();
  copy().memset(zero_buf, 0)->ignore();
  const vecmem::data::vector_view<unsigned int> counters_view(
      gbts_counter::nCounters, zero_buf.ptr());
  vecmem::vector<unsigned int> h_counters(gbts_counter::nCounters,
                                          mr().host ? mr().host : &(mr().main));

  // Stage 1: bin spacepoints and create nodes with the parameters (eta, phi,
  // r, z). No synchronisation: an event without nodes produces no edges.
  node_making_output nodes = make_nodes(spacepoints, measurements, zero_buf);

  // Stage 2: graph. The per-node buffers are moved in so they are released
  // when create_gbts_edges_from_nodes returns, along with all the edge/link
  // transients.
  graph_making_output graph = create_edges(
      std::move(nodes.node_params), std::move(nodes.node_phi),
      std::move(nodes.node_index), nodes.bin_rads, nodes.eta_bin_views_buf,
      nodes.pair_work_begin_buf, nodes.work_items_buf, nodes.nWorkMax,
      nodes.nSp, nodes.static_blob, zero_buf, h_counters);
  if (graph.nConnectedEdges == 0) {
    // No connected edges survived graph making -> no seeds.
    return {0, mr().main};
  }

  // Stage 3: Create seeds from the graph edges.
  return extract_seeds(graph.output_graph, graph.levels, nodes.reducedSP,
                       graph.nConnectedEdges, nSp, counters_view, h_counters);
}

}  // namespace traccc::device
