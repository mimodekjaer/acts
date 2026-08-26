/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/gbts_seeding/gbts_seeding_algorithm.hpp"

#include "../utils/barrier.hpp"
#include "../utils/get_queue.hpp"
#include "../utils/parallel_algorithms.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/gbts_seeding/device/gbts_bid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_bid_seeds_for_hits.hpp"
#include "traccc/gbts_seeding/device/gbts_bin_spacepoints.hpp"
#include "traccc/gbts_seeding/device/gbts_compress_graph.hpp"
#include "traccc/gbts_seeding/device/gbts_convert_seeds.hpp"
#include "traccc/gbts_seeding/device/gbts_count_terminus_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_fill_path_store.hpp"
#include "traccc/gbts_seeding/device/gbts_find_minmax_radius.hpp"
#include "traccc/gbts_seeding/device/gbts_fit_segments.hpp"
#include "traccc/gbts_seeding/device/gbts_link_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_make_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_match_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_rebid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reindex_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reset_edge_bids.hpp"
#include "traccc/gbts_seeding/device/gbts_run_cca_iteration.hpp"
#include "traccc/gbts_seeding/device/gbts_sort_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_sort_nodes.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>

// System include(s).
#include <algorithm>

namespace traccc::alpaka {

namespace kernels {

// ---------------------------------------------------------------------------
// Stage 1 — nodes-making kernels
// ---------------------------------------------------------------------------

/// Alpaka kernel for running @c traccc::device::gbts_bin_spacepoints
struct gbts_bin_spacepoints {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_bin_spacepoints_payload payload) const {
    device::gbts_bin_spacepoints(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_sort_nodes
struct gbts_sort_nodes {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, const device::gbts_sort_nodes_payload payload) const {
    device::gbts_sort_nodes(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_find_minmax_radius
struct gbts_find_minmax_radius {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_find_minmax_radius_payload payload) const {
    auto& shared_min = ::alpaka::declareSharedVar<
        float[device::gbts_find_minmax_radius_block_size], __COUNTER__>(acc);
    auto& shared_max = ::alpaka::declareSharedVar<
        float[device::gbts_find_minmax_radius_block_size], __COUNTER__>(acc);
    const alpaka::barrier<TAcc> barrier(&acc);

    device::gbts_find_minmax_radius(
        details::thread_id1{acc}, barrier, payload,
        {vecmem::data::vector_view<float>(
             device::gbts_find_minmax_radius_block_size, &shared_min[0]),
         vecmem::data::vector_view<float>(
             device::gbts_find_minmax_radius_block_size, &shared_max[0])});
  }
};

// ---------------------------------------------------------------------------
// Stage 2 — graph-making kernels
// ---------------------------------------------------------------------------

/// Alpaka kernel for running @c traccc::device::gbts_make_graph_edges
struct gbts_make_graph_edges {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_make_graph_edges_payload payload) const {
    auto& phi = ::alpaka::declareSharedVar<
        float[traccc::device::gbts_consts::node_buffer_length], __COUNTER__>(
        acc);
    auto& node_pack = ::alpaka::declareSharedVar<
        traccc::float4[traccc::device::gbts_consts::node_buffer_length],
        __COUNTER__>(acc);
    const alpaka::barrier<TAcc> barrier(&acc);

    device::gbts_make_graph_edges(
        details::thread_id1{acc}, barrier, payload,
        {vecmem::data::vector_view<float>(
             traccc::device::gbts_consts::node_buffer_length, &phi[0]),
         vecmem::data::vector_view<traccc::float4>(
             traccc::device::gbts_consts::node_buffer_length, &node_pack[0])});
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_link_graph_edges
struct gbts_link_graph_edges {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_link_graph_edges_payload payload) const {
    device::gbts_link_graph_edges(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_sort_graph_edges_small
struct gbts_sort_graph_edges_small {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_sort_graph_edges_payload payload) const {
    device::gbts_sort_graph_edges_small(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_sort_graph_edges_large
struct gbts_sort_graph_edges_large {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_sort_graph_edges_payload payload) const {
    auto& bucket_cache = ::alpaka::declareSharedVar<
        unsigned int[device::gbts_sort_graph_edges_cache_size], __COUNTER__>(
        acc);
    auto& large_flags = ::alpaka::declareSharedVar<
        unsigned int[device::gbts_sort_graph_edges_block_size], __COUNTER__>(
        acc);
    const alpaka::barrier<TAcc> barrier(&acc);

    device::gbts_sort_graph_edges_large(
        details::thread_id1{acc}, barrier, payload,
        {vecmem::data::vector_view<unsigned int>(
             device::gbts_sort_graph_edges_cache_size, &bucket_cache[0]),
         vecmem::data::vector_view<unsigned int>(
             device::gbts_sort_graph_edges_block_size, &large_flags[0])});
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_match_graph_edges
struct gbts_match_graph_edges {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_match_graph_edges_payload payload) const {
    device::gbts_match_graph_edges(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_compress_graph
struct gbts_compress_graph {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_compress_graph_payload payload) const {
    device::gbts_compress_graph(details::thread_id1{acc}, payload);
  }
};

// ---------------------------------------------------------------------------
// Stage 3 — graph-processing kernels
// ---------------------------------------------------------------------------

/// Alpaka kernel for running @c traccc::device::gbts_run_cca_iteration
struct gbts_run_cca_iteration {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_run_cca_iteration_payload payload) const {
    device::gbts_run_cca_iteration(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_count_terminus_edges
struct gbts_count_terminus_edges {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_count_terminus_edges_payload payload) const {
    device::gbts_count_terminus_edges(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_fill_path_store
struct gbts_fill_path_store {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_fill_path_store_payload payload) const {
    device::gbts_fill_path_store(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_fit_segments
struct gbts_fit_segments {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, const device::gbts_fit_segments_payload payload) const {
    device::gbts_fit_segments(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_bid_seeds_for_edges
struct gbts_bid_seeds_for_edges {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_bid_seeds_for_edges_payload payload) const {
    device::gbts_bid_seeds_for_edges(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_reset_edge_bids
struct gbts_reset_edge_bids {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_reset_edge_bids_payload payload) const {
    device::gbts_reset_edge_bids(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_rebid_seeds_for_edges
struct gbts_rebid_seeds_for_edges {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_rebid_seeds_for_edges_payload payload) const {
    device::gbts_rebid_seeds_for_edges(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_bid_seeds_for_hits
struct gbts_bid_seeds_for_hits {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_bid_seeds_for_hits_payload payload) const {
    device::gbts_bid_seeds_for_hits(details::thread_id1{acc}, payload);
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_convert_seeds
struct gbts_convert_seeds {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, const device::gbts_convert_seeds_payload payload) const {
    device::gbts_convert_seeds(details::thread_id1{acc}, payload);
  }
};

}  // namespace kernels

// ===========================================================================
// gbts_seeding_algorithm: kernel launchers
// ===========================================================================

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const memory_resource& mr,
    const vecmem::copy& copy, alpaka::queue& q,
    std::unique_ptr<const Logger> logger)
    : device::gbts_seeding_algorithm(cfg, mr, copy, std::move(logger)),
      alpaka::algorithm_base{q} {}

void gbts_seeding_algorithm::gbts_bin_spacepoints_kernel(
    const device::gbts_bin_spacepoints_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nSp - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_bin_spacepoints{}, payload);
}

void gbts_seeding_algorithm::gbts_sort_nodes_kernel(
    const device::gbts_sort_nodes_payload& payload) const {
  // Order the nodes by their (eta bin, phi, spacepoint index bits) keys,
  // carrying the full spacepoint index along as the value.
  details::sort_by_key(details::get_queue(queue()), mr(),
                       payload.sort_keys.ptr(),
                       payload.sort_keys.ptr() + payload.nNodes,
                       payload.sort_values.ptr());

  const unsigned int n_threads = 256;
  const unsigned int n_blocks = 1 + (payload.nNodes - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_sort_nodes{}, payload);
}

void gbts_seeding_algorithm::gbts_find_minmax_radius_kernel(
    const device::gbts_find_minmax_radius_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nEtaBins - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_find_minmax_radius{}, payload);
}

void gbts_seeding_algorithm::gbts_make_graph_edges_kernel(
    const device::gbts_make_graph_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = payload.nUsedBinPairs;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_make_graph_edges{}, payload);
  vecmem::device_vector<unsigned int> d_num_outgoing_edges(
      payload.num_outgoing_edges);
  details::inclusive_scan(
      details::get_queue(queue()), mr(), d_num_outgoing_edges.begin(),
      d_num_outgoing_edges.end(), d_num_outgoing_edges.begin());
}

void gbts_seeding_algorithm::gbts_link_graph_edges_kernel(
    const device::gbts_link_graph_edges_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_link_graph_edges{}, payload);
}

void gbts_seeding_algorithm::gbts_sort_graph_edges_kernel(
    const device::gbts_sort_graph_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  // Small buckets: one thread per edge.
  const unsigned int n_blocks_small = 1 + (payload.nEdges - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks_small, n_threads),
                      kernels::gbts_sort_graph_edges_small{}, payload);
  // Large buckets: blocks stride over the nodes a block's worth at a time.
  const unsigned int n_blocks_large =
      std::min(1 + (payload.nNodes - 1) / device::gbts_sort_graph_edges_block_size,
               4096u);
  ::alpaka::exec<Acc>(
      details::get_queue(queue()),
      makeWorkDiv<Acc>(n_blocks_large,
                       device::gbts_sort_graph_edges_block_size),
      kernels::gbts_sort_graph_edges_large{}, payload);
}

void gbts_seeding_algorithm::gbts_match_graph_edges_kernel(
    const device::gbts_match_graph_edges_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_match_graph_edges{}, payload);
}

void gbts_seeding_algorithm::gbts_reindex_edges_kernel(
    const device::gbts_reindex_edges_payload& payload) const {
  // Compact the kept edges with a prefix sum over their 0/1 flags.
  vecmem::device_vector<int> d_reIndexer(payload.reIndexer);
  details::inclusive_scan(details::get_queue(queue()), mr(),
                          d_reIndexer.begin(),
                          d_reIndexer.begin() + payload.nEdges,
                          d_reIndexer.begin());
}

void gbts_seeding_algorithm::gbts_compress_graph_kernel(
    const device::gbts_compress_graph_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_compress_graph{}, payload);
}

void gbts_seeding_algorithm::gbts_run_cca_iteration_kernel(
    const device::gbts_run_cca_iteration_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_run_cca_iteration{}, payload);
}

void gbts_seeding_algorithm::gbts_count_terminus_edges_kernel(
    const device::gbts_count_terminus_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_count_terminus_edges{}, payload);
  // Lay out the path store: each terminus edge gets a contiguous row range.
  vecmem::device_vector<unsigned int> d_row_sizes(payload.row_sizes);
  details::inclusive_scan(details::get_queue(queue()), mr(),
                          d_row_sizes.begin(),
                          d_row_sizes.begin() + payload.nConnectedEdges,
                          d_row_sizes.begin());
}

void gbts_seeding_algorithm::gbts_fill_path_store_kernel(
    const device::gbts_fill_path_store_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_fill_path_store{}, payload);
}

void gbts_seeding_algorithm::gbts_fit_segments_kernel(
    const device::gbts_fit_segments_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_fit_segments{}, payload);
}

void gbts_seeding_algorithm::gbts_bid_seeds_for_edges_kernel(
    const device::gbts_bid_seeds_for_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_bid_seeds_for_edges{}, payload);
}

void gbts_seeding_algorithm::gbts_reset_edge_bids_kernel(
    const device::gbts_reset_edge_bids_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_reset_edge_bids{}, payload);
}

void gbts_seeding_algorithm::gbts_rebid_seeds_for_edges_kernel(
    const device::gbts_rebid_seeds_for_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_rebid_seeds_for_edges{}, payload);
}

void gbts_seeding_algorithm::gbts_bid_seeds_for_hits_kernel(
    const device::gbts_bid_seeds_for_hits_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_bid_seeds_for_hits{}, payload);
}

void gbts_seeding_algorithm::gbts_convert_seeds_kernel(
    const device::gbts_convert_seeds_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_convert_seeds{}, payload);
}

}  // namespace traccc::alpaka
