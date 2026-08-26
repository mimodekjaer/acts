/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"

// Project include(s).
#include "traccc/gbts_seeding/device/gbts_bid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_bid_seeds_for_hits.hpp"
#include "traccc/gbts_seeding/device/gbts_bin_spacepoints.hpp"
#include "traccc/gbts_seeding/device/gbts_build_edge_work_list.hpp"
#include "traccc/gbts_seeding/device/gbts_compress_graph.hpp"
#include "traccc/gbts_seeding/device/gbts_convert_seeds.hpp"
#include "traccc/gbts_seeding/device/gbts_count_terminus_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_fill_path_store.hpp"
#include "traccc/gbts_seeding/device/gbts_find_minmax_radius.hpp"
#include "traccc/gbts_seeding/device/gbts_fit_segments.hpp"
#include "traccc/gbts_seeding/device/gbts_make_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_match_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_rebid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reindex_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reset_edge_bids.hpp"
#include "traccc/gbts_seeding/device/gbts_run_cca_iteration.hpp"
#include "traccc/gbts_seeding/device/gbts_seed_bidding.hpp"
#include "traccc/gbts_seeding/device/gbts_sort_nodes.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <algorithm>
#include <memory_resource>

// CUDA include(s).
#include <cooperative_groups.h>

// CUB include(s).
#include <cub/device/device_radix_sort.cuh>

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace traccc::cuda {

namespace kernels {

using float4 = traccc::float4;
using uint2 = traccc::uint2;
using int2 = traccc::int2;

// ---------------------------------------------------------------------------
// Stage 1 — nodes-making kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::gbts_bin_spacepoints
__global__ void gbts_bin_spacepoints(
    const device::gbts_bin_spacepoints_payload payload) {
  device::gbts_bin_spacepoints(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_build_edge_work_list
__global__ void gbts_build_edge_work_list(
    const device::gbts_build_edge_work_list_payload payload) {
  __shared__ unsigned int scratch[device::gbts_build_edge_work_list_block_size];
  const traccc::cuda::barrier barrier;
  device::gbts_build_edge_work_list(
      details::thread_id1{}, barrier, payload,
      {vecmem::data::vector_view<unsigned int>(
          device::gbts_build_edge_work_list_block_size, scratch)});
}

/// CUDA kernel for running @c traccc::device::gbts_sort_nodes
__global__ void gbts_sort_nodes(const device::gbts_sort_nodes_payload payload) {
  device::gbts_sort_nodes(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_find_minmax_radius
__global__ void gbts_find_minmax_radius(
    const device::gbts_find_minmax_radius_payload payload) {
  __shared__ float shared_min[device::gbts_find_minmax_radius_block_size];
  __shared__ float shared_max[device::gbts_find_minmax_radius_block_size];
  const traccc::cuda::barrier barrier;

  device::gbts_find_minmax_radius(
      details::thread_id1{}, barrier, payload,
      {vecmem::data::vector_view<float>(
           device::gbts_find_minmax_radius_block_size, shared_min),
       vecmem::data::vector_view<float>(
           device::gbts_find_minmax_radius_block_size, shared_max)});
}

// ---------------------------------------------------------------------------
// Stage 2 — graph-making kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::gbts_make_graph_edges
///
/// @tparam fill false: count the edges, true: write them
template <bool fill>
__global__ void gbts_make_graph_edges(
    const device::gbts_make_graph_edges_payload payload) {
  __shared__ float phi[traccc::device::gbts_consts::node_buffer_length];
  __shared__ float4 node_pack[traccc::device::gbts_consts::node_buffer_length];
  __shared__ unsigned int work_slot[1];
  const traccc::cuda::barrier barrier;

  device::gbts_make_graph_edges<fill>(
      details::thread_id1{}, barrier, payload,
      {vecmem::data::vector_view<float>(
           traccc::device::gbts_consts::node_buffer_length, phi),
       vecmem::data::vector_view<float4>(
           traccc::device::gbts_consts::node_buffer_length, node_pack),
       vecmem::data::vector_view<unsigned int>(1u, work_slot)});
}

/// CUDA kernel for running @c traccc::device::gbts_match_graph_edges
__global__ void gbts_match_graph_edges(
    const device::gbts_match_graph_edges_payload payload) {
  device::gbts_match_graph_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_compress_graph
__global__ void gbts_compress_graph(
    const device::gbts_compress_graph_payload payload) {
  device::gbts_compress_graph(details::thread_id1{}, payload);
}

// ---------------------------------------------------------------------------
// Stage 3 — graph-processing kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::gbts_run_cca_iteration
__global__ void gbts_run_cca_iteration(
    const device::gbts_run_cca_iteration_payload payload) {
  device::gbts_run_cca_iteration(details::thread_id1{}, payload);
}

/// CUDA kernel running all CCA iterations in one cooperative launch, with
/// a grid-wide barrier between the iterations
__global__ void gbts_run_cca(
    const device::gbts_run_cca_iteration_payload payload) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  device::gbts_run_cca_iteration_payload iteration = payload;
  for (unsigned char iter = 0; iter < traccc::device::gbts_consts::max_cca_iter;
       ++iter) {
    iteration.iter = iter;
    device::gbts_run_cca_iteration(details::thread_id1{}, iteration);
    grid.sync();
  }
}

/// Fused CCA for the common case of one edge per thread (the grid covers
/// all edges): the neighbour list of the edge stays in registers across
/// the iterations and the loop stops as soon as no edge is active any more.
/// Same per-iteration semantics as @c traccc::device::gbts_run_cca_iteration.
template <unsigned int MAX_NEI>
__global__ void gbts_run_cca_cached(
    const device::gbts_run_cca_iteration_payload payload) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const vecmem::device_vector<const unsigned int> d_output_graph(
      payload.output_graph);
  vecmem::device_vector<unsigned char> d_levels(payload.levels);
  vecmem::device_vector<int2> d_outgoing_paths(payload.outgoing_paths);
  unsigned int* active_counters = payload.active_counters;

  const unsigned int n = payload.nConnectedEdges;
  const unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
  const bool has_edge = edge < n;
  constexpr unsigned char max_iter = traccc::device::gbts_consts::max_cca_iter;

  // Cache the neighbour list of the edge.
  unsigned int nNeighbours = 0u;
  unsigned int nei[MAX_NEI];
  if (has_edge) {
    const unsigned int edge_pos = (2u + 1u + payload.max_num_neighbours) * edge;
    nNeighbours = d_output_graph[edge_pos + device::gbts_consts::nNei];
    for (unsigned int k = 0u; k < nNeighbours; ++k) {
      nei[k] = d_output_graph[edge_pos + device::gbts_consts::nei_start + k];
    }
  }
  // Iteration in which the edge is (re)visited next; -1 once settled.
  int active = 0;

  if (edge == 0u) {
    active_counters[0] = 0u;
  }
  grid.sync();

  for (unsigned char iter = 0; iter < max_iter; ++iter) {
    const unsigned int toggle = iter % 2u;
    const unsigned int levelLoad = toggle * n;
    const unsigned int levelStore = (1u - toggle) * n;
    bool stays_active = false;

    if (has_edge && active == static_cast<int>(iter)) {
      unsigned char next_level = d_levels[levelLoad + edge];
      bool localChange = false;
      for (unsigned int k = 0u; k < nNeighbours; ++k) {
        const unsigned char forward_level = d_levels[levelLoad + nei[k]];
        if (next_level == forward_level) {
          next_level = forward_level + 1;
          localChange = true;
          break;
        }
      }
      if (localChange) {
        if (iter == max_iter - 1) {
          d_outgoing_paths[edge].y = -1;
          active = -1;
        } else {
          active = static_cast<int>(iter) + 1;
          stays_active = true;
        }
      } else {
        active = -1;
        int out_paths = 0;
        for (unsigned int k = 0u; k < nNeighbours; ++k) {
          if (next_level == 1 + d_levels[levelLoad + nei[k]]) {
            out_paths += 1 + d_outgoing_paths[nei[k]].x;
          }
          // flag as not terminus edge
          d_outgoing_paths[nei[k]].y = -1;
        }
        // flag as long enough segment to become a seed
        d_outgoing_paths[edge] = int2{
            out_paths, static_cast<int>(next_level >= payload.minLevel) - 1};
      }
      d_levels[levelStore + edge] = next_level;
    }

    // Count the edges that stay active (block aggregated), zero the next
    // counter, and stop when nothing is left to do.
    const int block_active = __syncthreads_count(stays_active ? 1 : 0);
    if (threadIdx.x == 0u && block_active != 0) {
      atomicAdd(active_counters + iter,
                static_cast<unsigned int>(block_active));
    }
    if (edge == 0u) {
      active_counters[iter + 1u] = 0u;
    }
    grid.sync();
    if (active_counters[iter] == 0u) {
      break;
    }
  }
}

/// Fused seed-vs-edge bidding for the common case of one path-store row
/// per thread (the grid covers all rows): the edge chain of the row's
/// proposal is walked once and kept in registers across the rounds. Same
/// per-step semantics as gbts_bid_seeds_for_edges / gbts_rebid_seeds_for_edges
/// / gbts_reset_edge_bids.
template <unsigned int MAX_LEN>
__global__ void gbts_bid_seeds_cached(
    const device::gbts_seed_bidding_payload payload) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const vecmem::device_vector<const int2> d_path_store(payload.path_store);
  vecmem::device_vector<int2> d_seed_proposals(payload.seed_proposals);
  vecmem::device_vector<char> d_seed_ambiguity(payload.seed_ambiguity);
  vecmem::device_vector<unsigned long long int> d_edge_bids(payload.edge_bids);
  const unsigned int n = payload.nConnectedEdges;

  const unsigned int prop_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int nThreads = gridDim.x * blockDim.x;
  int2 prop = int2{0, -1};
  if (prop_idx < payload.nRows) {
    prop = d_seed_proposals[prop_idx];
  }
  const bool has_prop = prop.y >= 0;

  // Walk the chain once: edges of the path in bidding order.
  unsigned int chain[MAX_LEN];
  unsigned int length = 0u;
  if (has_prop) {
    int2 path = int2{0, prop.y};
    while (path.y >= 0 && length < MAX_LEN) {
      path = d_path_store[static_cast<unsigned int>(path.y)];
      chain[length++] = static_cast<unsigned int>(path.x);
    }
  }
  const unsigned long long int seed_bid =
      (static_cast<unsigned long long int>(prop.x) << 32) |
      static_cast<unsigned long long int>(prop_idx);

  // Bid for the first @c depth edges of the chain into @c bids.
  const auto bid = [&](unsigned long long int* bids, const unsigned int depth) {
    d_seed_proposals[prop_idx] = prop;
    for (unsigned int k = 0u; k < depth; ++k) {
      const unsigned long long int competing_offer =
          atomicMax(bids + chain[k], seed_bid);
      if (competing_offer > seed_bid) {
        d_seed_ambiguity[prop_idx] = -1;
      } else if (competing_offer != 0ull) {
        d_seed_ambiguity[static_cast<unsigned int>(competing_offer &
                                                   0xFFFFFFFFull)] = -1;
      }
    }
  };

  // Initial bid: terminus edge only.
  if (has_prop) {
    bid(d_edge_bids.data(), (length > 0u) ? 1u : 0u);
  }

  for (unsigned int round = 0u; round < payload.nRounds; ++round) {
    const unsigned int half = (round + 1u) % 2u;
    unsigned long long int* bids = d_edge_bids.data() + half * n;
    unsigned long long int* bids_next = d_edge_bids.data() + (1u - half) * n;
    grid.sync();

    // --- rebid ---
    if (has_prop) {
      const char ambi = d_seed_ambiguity[prop_idx];
      bool do_bid = true;
      if (round == 0u) {
        if (ambi == 0) {
          // rebid 'best seed from edge' in later rounds
          d_seed_ambiguity[prop_idx] = 1;
        } else {
          d_seed_ambiguity[prop_idx] = -2;
          atomicAdd(payload.nRejectedPropsCounter, 1u);
          do_bid = false;
        }
      } else if ((ambi == -2) | (ambi == 0)) {
        // only rebid for maybes
        do_bid = false;
      }
      if (do_bid) {
        bid(bids, length);
      }
    }
    grid.sync();

    // --- reset: zero the next round's bids, then re-evaluate the maybes ---
    for (unsigned int idx = prop_idx; idx < n; idx += nThreads) {
      bids_next[idx] = 0ull;
    }
    if (has_prop) {
      const char ambi = d_seed_ambiguity[prop_idx];
      if (!((ambi == -2) | (ambi == 0))) {
        bool isgood = true;
        for (unsigned int k = 0u; k < length; ++k) {
          const unsigned long long int best_bid = bids[chain[k]];
          if (d_seed_ambiguity[static_cast<unsigned int>(best_bid &
                                                         0xFFFFFFFFull)] == 0) {
            isgood = false;
            break;
          }
        }
        if (isgood) {
          d_seed_ambiguity[prop_idx] = 1;
        } else {
          d_seed_ambiguity[prop_idx] = -2;
          atomicAdd(payload.nRejectedPropsCounter, 1u);
        }
      }
    }
  }
}

/// CUDA kernel running the whole seed-vs-edge bidding sequence in one
/// cooperative launch, with grid-wide barriers between the steps
__global__ void gbts_bid_seeds(
    const device::gbts_seed_bidding_payload payload) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const details::thread_id1 thread_id{};
  device::gbts_bid_seeds_for_edges(
      thread_id, device::gbts_make_bid_seeds_for_edges_payload(payload));
  for (unsigned int round = 0u; round < payload.nRounds; ++round) {
    grid.sync();
    device::gbts_rebid_seeds_for_edges(
        thread_id,
        device::gbts_make_rebid_seeds_for_edges_payload(payload, round));
    grid.sync();
    device::gbts_reset_edge_bids(
        thread_id, device::gbts_make_reset_edge_bids_payload(payload, round));
  }
}

/// CUDA kernel for running @c traccc::device::gbts_count_terminus_edges
__global__ void gbts_count_terminus_edges(
    const device::gbts_count_terminus_edges_payload payload) {
  device::gbts_count_terminus_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_fill_path_store
__global__ void gbts_fill_path_store(
    const device::gbts_fill_path_store_payload payload) {
  device::gbts_fill_path_store(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_fit_segments
__global__ void gbts_fit_segments(
    const device::gbts_fit_segments_payload payload) {
  device::gbts_fit_segments(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_bid_seeds_for_edges
__global__ void gbts_bid_seeds_for_edges(
    const device::gbts_bid_seeds_for_edges_payload payload) {
  device::gbts_bid_seeds_for_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_reset_edge_bids
__global__ void gbts_reset_edge_bids(
    const device::gbts_reset_edge_bids_payload payload) {
  device::gbts_reset_edge_bids(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_rebid_seeds_for_edges
__global__ void gbts_rebid_seeds_for_edges(
    const device::gbts_rebid_seeds_for_edges_payload payload) {
  device::gbts_rebid_seeds_for_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_bid_seeds_for_hits
__global__ void gbts_bid_seeds_for_hits(
    const device::gbts_bid_seeds_for_hits_payload payload) {
  device::gbts_bid_seeds_for_hits(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_convert_seeds
__global__ void gbts_convert_seeds(
    const device::gbts_convert_seeds_payload payload) {
  device::gbts_convert_seeds(details::thread_id1{}, payload);
}

}  // namespace kernels

// ===========================================================================
// gbts_seeding_algorithm: kernel launchers
// ===========================================================================

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const memory_resource& mr,
    const vecmem::copy& copy, const stream_wrapper& str,
    std::unique_ptr<const Logger> logger)
    : device::gbts_seeding_algorithm(cfg, mr, copy, std::move(logger)),
      cuda::algorithm_base{str} {}

void gbts_seeding_algorithm::gbts_bin_spacepoints_kernel(
    const device::gbts_bin_spacepoints_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nSp - 1) / n_threads;
  kernels::gbts_bin_spacepoints<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_sort_nodes_kernel(
    const device::gbts_sort_nodes_payload& payload) const {
  // Order the nodes by their (eta bin, phi, spacepoint index bits) keys,
  // carrying the full spacepoint index along as the value.
  // Stable radix sort of the significant key bits only (the eta bin and phi
  // fields; the keys are written in spacepoint order, so the low index bits
  // are redundant for a stable sort): fewer radix passes than a full 64-bit
  // sort.
  unsigned int eta_bits = 0u;
  while ((1u << eta_bits) <= payload.nEtaBins) {
    ++eta_bits;
  }
  const int begin_bit = static_cast<int>(device::gbts_sort_key_phi_shift);
  const int end_bit =
      static_cast<int>(device::gbts_sort_key_eta_shift + eta_bits);

  cudaStream_t cuda_stream = details::get_stream(stream());
  vecmem::data::vector_buffer<unsigned long long int> keys_alt(payload.nKeys,
                                                               mr().main);
  vecmem::data::vector_buffer<unsigned int> values_alt(payload.nKeys,
                                                       mr().main);
  cub::DoubleBuffer<unsigned long long int> d_keys(payload.sort_keys.ptr(),
                                                   keys_alt.ptr());
  cub::DoubleBuffer<unsigned int> d_values(payload.sort_values.ptr(),
                                           values_alt.ptr());
  std::size_t temp_bytes = 0u;
  TRACCC_CUDA_ERROR_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes, d_keys, d_values, static_cast<int>(payload.nKeys),
      begin_bit, end_bit, cuda_stream));
  vecmem::data::vector_buffer<char> temp(
      static_cast<unsigned int>(std::max<std::size_t>(temp_bytes, 1u)),
      mr().main);
  TRACCC_CUDA_ERROR_CHECK(cub::DeviceRadixSort::SortPairs(
      temp.ptr(), temp_bytes, d_keys, d_values, static_cast<int>(payload.nKeys),
      begin_bit, end_bit, cuda_stream));
  if (d_values.Current() != payload.sort_values.ptr()) {
    // The sorted values ended up in the alternate buffer.
    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(payload.sort_values.ptr(), d_values.Current(),
                        payload.nKeys * sizeof(unsigned int),
                        cudaMemcpyDeviceToDevice, cuda_stream));
  }

  const unsigned int n_threads = 256;
  const unsigned int n_blocks = 1 + (payload.nKeys - 1) / n_threads;
  kernels::gbts_sort_nodes<<<n_blocks, n_threads, 0,
                             details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_build_edge_work_list_kernel(
    const device::gbts_build_edge_work_list_payload& payload) const {
  kernels::gbts_build_edge_work_list<<<
      1, device::gbts_build_edge_work_list_block_size, 0,
      details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_find_minmax_radius_kernel(
    const device::gbts_find_minmax_radius_payload& payload) const {
  // One block per eta bin.
  const unsigned int n_threads = device::gbts_find_minmax_radius_block_size;
  const unsigned int n_blocks = payload.nEtaBins;
  kernels::gbts_find_minmax_radius<<<n_blocks, n_threads, 0,
                                     details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
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
  kernels::gbts_make_graph_edges<false>
      <<<n_blocks, n_threads, 0, details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
  // Turn the per-node edge counts into the bucket begin/end offsets.
  vecmem::device_vector<unsigned int> d_num_outgoing_edges(
      payload.num_outgoing_edges);
  thrust::inclusive_scan(
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(details::get_stream(stream())),
      d_num_outgoing_edges.begin(), d_num_outgoing_edges.end(),
      d_num_outgoing_edges.begin());
}

void gbts_seeding_algorithm::gbts_make_graph_edges_kernel(
    const device::gbts_make_graph_edges_payload& payload) const {
  const unsigned int n_threads =
      traccc::device::gbts_consts::node_buffer_length;
  // The blocks stride over the device-side work list.
  const unsigned int n_blocks =
      std::min(payload.nWorkMax, device::gbts_make_graph_edges_max_blocks);
  kernels::gbts_make_graph_edges<true>
      <<<n_blocks, n_threads, 0, details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_match_graph_edges_kernel(
    const device::gbts_match_graph_edges_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
  kernels::gbts_match_graph_edges<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_reindex_edges_kernel(
    const device::gbts_reindex_edges_payload& payload) const {
  vecmem::device_vector<int> d_reIndexer(payload.reIndexer);
  thrust::inclusive_scan(
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(details::get_stream(stream())),
      d_reIndexer.begin(), d_reIndexer.begin() + payload.nEdges,
      d_reIndexer.begin());
}

void gbts_seeding_algorithm::gbts_compress_graph_kernel(
    const device::gbts_compress_graph_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
  kernels::gbts_compress_graph<<<n_blocks, n_threads, 0,
                                 details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

namespace {

/// Launch a grid-stride kernel cooperatively, with as many blocks as can be
/// resident at once (at most the number needed for @c n_items). Returns
/// false when the device does not support cooperative launches.
template <typename payload_t>
bool launch_cooperative(void (*kernel)(payload_t), const unsigned int n_items,
                        const unsigned int n_threads, const payload_t& payload,
                        cudaStream_t stream,
                        const bool require_full_grid = false) {
  static const int cooperative = []() {
    int device = 0;
    TRACCC_CUDA_ERROR_CHECK(cudaGetDevice(&device));
    int value = 0;
    TRACCC_CUDA_ERROR_CHECK(
        cudaDeviceGetAttribute(&value, cudaDevAttrCooperativeLaunch, device));
    return value;
  }();
  static const int n_sms = []() {
    int device = 0;
    TRACCC_CUDA_ERROR_CHECK(cudaGetDevice(&device));
    int value = 0;
    TRACCC_CUDA_ERROR_CHECK(
        cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
    return value;
  }();
  if (cooperative == 0) {
    return false;
  }
  int blocks_per_sm = 0;
  TRACCC_CUDA_ERROR_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, static_cast<int>(n_threads), 0));
  const unsigned int max_blocks = static_cast<unsigned int>(blocks_per_sm) *
                                  static_cast<unsigned int>(n_sms);
  const unsigned int needed =
      (n_items == 0u) ? 1u : 1u + (n_items - 1u) / n_threads;
  if (require_full_grid && needed > max_blocks) {
    return false;
  }
  const unsigned int n_blocks = std::max(1u, std::min(needed, max_blocks));
  void* args[] = {const_cast<payload_t*>(&payload)};
  TRACCC_CUDA_ERROR_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(kernel), dim3(n_blocks), dim3(n_threads),
      args, 0u, stream));
  return true;
}

}  // namespace

void gbts_seeding_algorithm::gbts_run_cca_kernel(
    const device::gbts_run_cca_iteration_payload& payload) const {
  constexpr unsigned int max_cached_nei = 16u;
  // Large blocks: the cost of an iteration is dominated by the grid-wide
  // barrier, which scales with the number of blocks.
  const unsigned int n_threads = 1024u;
  if (payload.max_num_neighbours <= max_cached_nei) {
    // Fast path: one edge per thread with the neighbour lists cached in
    // registers. Only possible when the whole grid can be resident.
    if (launch_cooperative(kernels::gbts_run_cca_cached<max_cached_nei>,
                           payload.nConnectedEdges, n_threads, payload,
                           details::get_stream(stream()),
                           /*require_full_grid=*/true)) {
      return;
    }
  }
  if (!launch_cooperative(kernels::gbts_run_cca, payload.nConnectedEdges,
                          n_threads, payload, details::get_stream(stream()))) {
    device::gbts_seeding_algorithm::gbts_run_cca_kernel(payload);
  }
}

void gbts_seeding_algorithm::gbts_bid_seeds_kernel(
    const device::gbts_seed_bidding_payload& payload) const {
  // Fast path: one row per thread with the proposal's edge chain cached in
  // registers (paths have at most max_cca_iter + 1 edges).
  if (launch_cooperative(kernels::gbts_bid_seeds_cached<
                             traccc::device::gbts_consts::max_cca_iter + 1u>,
                         payload.nRows, 1024u, payload,
                         details::get_stream(stream()),
                         /*require_full_grid=*/true)) {
    return;
  }
  if (!launch_cooperative(kernels::gbts_bid_seeds, payload.nRows, 1024u,
                          payload, details::get_stream(stream()))) {
    device::gbts_seeding_algorithm::gbts_bid_seeds_kernel(payload);
  }
}

void gbts_seeding_algorithm::gbts_run_cca_iteration_kernel(
    const device::gbts_run_cca_iteration_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;

  kernels::gbts_run_cca_iteration<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_count_terminus_edges_kernel(
    const device::gbts_count_terminus_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
  kernels::gbts_count_terminus_edges<<<n_blocks, n_threads, 0,
                                       details::get_stream(stream())>>>(
      payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
  // Lay out the path store: each terminus edge gets a contiguous row range.
  vecmem::device_vector<unsigned int> d_row_sizes(payload.row_sizes);
  thrust::inclusive_scan(
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(details::get_stream(stream())),
      d_row_sizes.begin(), d_row_sizes.begin() + payload.nConnectedEdges,
      d_row_sizes.begin());
}

void gbts_seeding_algorithm::gbts_fill_path_store_kernel(
    const device::gbts_fill_path_store_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  kernels::gbts_fill_path_store<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_fit_segments_kernel(
    const device::gbts_fit_segments_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  kernels::gbts_fit_segments<<<n_blocks, n_threads, 0,
                               details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_bid_seeds_for_edges_kernel(
    const device::gbts_bid_seeds_for_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  kernels::gbts_bid_seeds_for_edges<<<n_blocks, n_threads, 0,
                                      details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_reset_edge_bids_kernel(
    const device::gbts_reset_edge_bids_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  kernels::gbts_reset_edge_bids<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_rebid_seeds_for_edges_kernel(
    const device::gbts_rebid_seeds_for_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  kernels::gbts_rebid_seeds_for_edges<<<n_blocks, n_threads, 0,
                                        details::get_stream(stream())>>>(
      payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_bid_seeds_for_hits_kernel(
    const device::gbts_bid_seeds_for_hits_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  kernels::gbts_bid_seeds_for_hits<<<n_blocks, n_threads, 0,
                                     details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_convert_seeds_kernel(
    const device::gbts_convert_seeds_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRows - 1) / n_threads;
  kernels::gbts_convert_seeds<<<n_blocks, n_threads, 0,
                                details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
