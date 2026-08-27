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
#include "traccc/gbts_seeding/device/gbts_sort_nodes.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// Thrust include(s).
#include <thrust/iterator/transform_iterator.h>

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>

// System include(s).
#include <algorithm>

namespace traccc::alpaka {

namespace kernels {

/// Widens a 1-byte flag to int (for the kept-flag prefix sum)
struct byte_to_int {
  ALPAKA_FN_HOST_ACC int operator()(unsigned char k) const {
    return static_cast<int>(k);
  }
};

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

/// Alpaka kernel for running @c traccc::device::gbts_build_edge_work_list
struct gbts_build_edge_work_list {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_build_edge_work_list_payload payload) const {
    auto& scratch = ::alpaka::declareSharedVar<
        unsigned int[device::gbts_build_edge_work_list_block_size],
        __COUNTER__>(acc);
    const alpaka::barrier<TAcc> barrier(&acc);
    device::gbts_build_edge_work_list(
        details::thread_id1{acc}, barrier, payload,
        {vecmem::data::vector_view<unsigned int>(
            device::gbts_build_edge_work_list_block_size, &scratch[0])});
  }
};

/// Alpaka kernel for running @c traccc::device::gbts_sort_nodes
struct gbts_sort_nodes {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc, const device::gbts_sort_nodes_payload payload) const {
    auto& min_bits = ::alpaka::declareSharedVar<
        unsigned int[device::gbts_sort_nodes_block_size], __COUNTER__>(acc);
    auto& max_bits = ::alpaka::declareSharedVar<
        unsigned int[device::gbts_sort_nodes_block_size], __COUNTER__>(acc);
    const alpaka::barrier<TAcc> barrier(&acc);
    device::gbts_sort_nodes(
        details::thread_id1{acc}, barrier, payload,
        {vecmem::data::vector_view<unsigned int>(
             device::gbts_sort_nodes_block_size, &min_bits[0]),
         vecmem::data::vector_view<unsigned int>(
             device::gbts_sort_nodes_block_size, &max_bits[0])});
  }
};

// ---------------------------------------------------------------------------
// Stage 2 — graph-making kernels
// ---------------------------------------------------------------------------

/// Alpaka kernel for running @c traccc::device::gbts_make_graph_edges
///
/// @tparam fill false: count the edges, true: write them
template <bool fill>
struct gbts_make_graph_edges {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      const device::gbts_make_graph_edges_payload payload) const {
    auto& work_slot = ::alpaka::declareSharedVar<
        unsigned int[traccc::device::gbts_make_graph_edges_scratch_size],
        __COUNTER__>(acc);
    const alpaka::barrier<TAcc> barrier(&acc);

    device::gbts_make_graph_edges<fill>(
        details::thread_id1{acc}, barrier, payload,
        {vecmem::data::vector_view<unsigned int>(
            traccc::device::gbts_make_graph_edges_scratch_size,
            &work_slot[0])});
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
    auto& changed =
        ::alpaka::declareSharedVar<unsigned int[1], __COUNTER__>(acc);
    const alpaka::barrier<TAcc> barrier(&acc);
    device::gbts_run_cca_iteration(
        details::thread_id1{acc}, barrier, payload,
        {vecmem::data::vector_view<unsigned int>(1u, &changed[0])});
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

void gbts_seeding_algorithm::gbts_sort_node_keys_kernel(
    const device::gbts_sort_nodes_payload& payload) const {
  // Order the nodes by their (eta bin, phi, spacepoint index bits) keys,
  // carrying the full spacepoint index along as the value.
  details::sort(details::get_queue(queue()), mr(), payload.sort_keys.ptr(),
                payload.sort_keys.ptr() + payload.nKeys);
}

void gbts_seeding_algorithm::gbts_sort_nodes_kernel(
    const device::gbts_sort_nodes_payload& payload) const {
  const unsigned int n_threads = device::gbts_sort_nodes_block_size;
  const unsigned int n_blocks = 1 + (payload.nKeys - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_sort_nodes{}, payload);
}

void gbts_seeding_algorithm::gbts_build_edge_work_list_kernel(
    const device::gbts_build_edge_work_list_payload& payload) const {
  ::alpaka::exec<Acc>(
      details::get_queue(queue()),
      makeWorkDiv<Acc>(1u, device::gbts_build_edge_work_list_block_size),
      kernels::gbts_build_edge_work_list{}, payload);
}

void gbts_seeding_algorithm::gbts_count_graph_edges_kernel(
    const device::gbts_make_graph_edges_payload& payload) const {
  // One thread per inner node of a chunk: the block size must equal the
  // chunk size.
  const unsigned int n_threads =
      traccc::device::gbts_consts::node_buffer_length;
  // The blocks stride over the device-side work list.
  const unsigned int n_blocks =
      std::min(payload.nWorkMax, device::gbts_make_graph_edges_max_blocks);
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_make_graph_edges<false>{}, payload);
  // Turn the per-node edge counts into the bucket begin/end offsets.
  vecmem::device_vector<unsigned int> d_num_outgoing_edges(
      payload.num_outgoing_edges);
  details::inclusive_scan(
      details::get_queue(queue()), mr(), d_num_outgoing_edges.begin(),
      d_num_outgoing_edges.end(), d_num_outgoing_edges.begin());
}

void gbts_seeding_algorithm::gbts_make_graph_edges_kernel(
    const device::gbts_make_graph_edges_payload& payload) const {
  const unsigned int n_threads =
      traccc::device::gbts_consts::node_buffer_length;
  // The blocks stride over the device-side work list.
  const unsigned int n_blocks =
      std::min(payload.nWorkMax, device::gbts_make_graph_edges_max_fill_blocks);
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_make_graph_edges<true>{}, payload);
}

void gbts_seeding_algorithm::gbts_match_graph_edges_kernel(
    const device::gbts_match_graph_edges_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks =
      std::min(1u + (payload.nEdgesMax - 1u) / n_threads, 8192u);
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_match_graph_edges{}, payload);
}

void gbts_seeding_algorithm::gbts_reindex_edges_kernel(
    const device::gbts_reindex_edges_payload& payload) const {
  // Compact the kept edges with a prefix sum over their 0/1 flags.
  // The 1-byte flags are widened on the fly; the sum is accumulated in int.
  auto kept_int = thrust::make_transform_iterator(payload.kept.ptr(),
                                                  kernels::byte_to_int{});
  details::inclusive_scan(details::get_queue(queue()), mr(), kept_int,
                          kept_int + payload.nEdgesMax,
                          payload.reIndexer.ptr());
}

void gbts_seeding_algorithm::gbts_compress_graph_kernel(
    const device::gbts_compress_graph_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks =
      std::min(1u + (payload.nEdgesMax - 1u) / n_threads, 8192u);
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_compress_graph{}, payload);
}

void gbts_seeding_algorithm::gbts_run_cca_iteration_kernel(
    const device::gbts_run_cca_iteration_payload& payload) const {
  const unsigned int n_threads = 256;
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
  details::inclusive_scan(
      details::get_queue(queue()), mr(), d_row_sizes.begin(),
      d_row_sizes.begin() + payload.nConnectedEdges, d_row_sizes.begin());
  // Publish the total row count (last entry of the scanned row sizes).
  ::alpaka::memcpy(
      details::get_queue(queue()),
      ::alpaka::createView(details::get_device(), payload.row_count, 1u),
      ::alpaka::createView(details::get_device(),
                           d_row_sizes.data() + payload.nConnectedEdges - 1u,
                           1u));
}

void gbts_seeding_algorithm::gbts_fill_path_store_kernel(
    const device::gbts_fill_path_store_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_fill_path_store{}, payload);
}

void gbts_seeding_algorithm::gbts_reset_edge_bids_kernel(
    const device::gbts_reset_edge_bids_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_reset_edge_bids{}, payload);
}

void gbts_seeding_algorithm::gbts_rebid_seeds_for_edges_kernel(
    const device::gbts_rebid_seeds_for_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_rebid_seeds_for_edges{}, payload);
}

void gbts_seeding_algorithm::gbts_bid_seeds_for_hits_kernel(
    const device::gbts_bid_seeds_for_hits_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_bid_seeds_for_hits{}, payload);
}

void gbts_seeding_algorithm::gbts_convert_seeds_kernel(
    const device::gbts_convert_seeds_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  ::alpaka::exec<Acc>(details::get_queue(queue()),
                      makeWorkDiv<Acc>(n_blocks, n_threads),
                      kernels::gbts_convert_seeds{}, payload);
}

}  // namespace traccc::alpaka
