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
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace traccc::cuda {

namespace kernels {

/// Widens a 1-byte flag to int (for the kept-flag prefix sum)
struct byte_to_int {
  __host__ __device__ int operator()(unsigned char k) const {
    return static_cast<int>(k);
  }
};

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
  __shared__ unsigned int min_bits[device::gbts_sort_nodes_block_size];
  __shared__ unsigned int max_bits[device::gbts_sort_nodes_block_size];
  const traccc::cuda::barrier barrier;
  device::gbts_sort_nodes(details::thread_id1{}, barrier, payload,
                          {vecmem::data::vector_view<unsigned int>(
                               device::gbts_sort_nodes_block_size, min_bits),
                           vecmem::data::vector_view<unsigned int>(
                               device::gbts_sort_nodes_block_size, max_bits)});
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
  __shared__ unsigned int
      work_slot[traccc::device::gbts_make_graph_edges_scratch_size];
  const traccc::cuda::barrier barrier;

  device::gbts_make_graph_edges<fill>(
      details::thread_id1{}, barrier, payload,
      {vecmem::data::vector_view<float>(
           traccc::device::gbts_consts::node_buffer_length, phi),
       vecmem::data::vector_view<float4>(
           traccc::device::gbts_consts::node_buffer_length, node_pack),
       vecmem::data::vector_view<unsigned int>(
           traccc::device::gbts_make_graph_edges_scratch_size, work_slot)});
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
/// One CCA update of @c edge (identical to the body of
/// traccc::device::gbts_run_cca_iteration). Returns true when the edge is
/// still active for the next iteration.
__device__ inline bool gbts_cca_update(
    const unsigned int edge, const unsigned int nNeighbours,
    const unsigned int* nei, const unsigned char iter,
    const unsigned int levelLoad, const unsigned int levelStore,
    const unsigned char minLevel,
    vecmem::device_vector<unsigned char>& d_levels,
    vecmem::device_vector<int2>& d_outgoing_paths) {
  constexpr unsigned char max_iter = traccc::device::gbts_consts::max_cca_iter;
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
  bool stays_active = false;
  if (localChange) {
    if (iter == max_iter - 1) {
      d_outgoing_paths[edge].y = -1;
    } else {
      stays_active = true;
    }
  } else {
    int out_paths = 0;
    for (unsigned int k = 0u; k < nNeighbours; ++k) {
      if (next_level == 1 + d_levels[levelLoad + nei[k]]) {
        out_paths += 1 + d_outgoing_paths[nei[k]].x;
      }
      // flag as not terminus edge
      d_outgoing_paths[nei[k]].y = -1;
    }
    // flag as long enough segment to become a seed
    d_outgoing_paths[edge] =
        int2{out_paths, static_cast<int>(next_level >= minLevel) - 1};
  }
  d_levels[levelStore + edge] = next_level;
  return stays_active;
}

template <unsigned int MAX_NEI, bool EXTRA_EDGES>
__device__ inline void gbts_run_cca_cached_body(
    cooperative_groups::grid_group& grid,
    const device::gbts_run_cca_iteration_payload& payload) {
  const vecmem::device_vector<const unsigned int> d_output_graph(
      payload.output_graph);
  vecmem::device_vector<unsigned char> d_levels(payload.levels);
  vecmem::device_vector<int2> d_outgoing_paths(payload.outgoing_paths);
  vecmem::device_vector<char> d_active_edges(payload.active_edges);
  unsigned int* active_counters = payload.active_counters;
  __shared__ unsigned int block_active_sum;

  const unsigned int n = payload.nConnectedEdges;
  const unsigned int edge_size = 2u + 1u + payload.max_num_neighbours;
  const unsigned int nThreads = gridDim.x * blockDim.x;
  const unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
  const bool has_edge = edge < n;
  constexpr unsigned char max_iter = traccc::device::gbts_consts::max_cca_iter;

  // Cache the neighbour list of the thread's first edge; edges beyond the
  // grid (rare: the grid is sized for the expected edge count) are handled
  // uncached below with the per-edge active flags.
  unsigned int nNeighbours = 0u;
  unsigned int nei[MAX_NEI];
  if (has_edge) {
    const unsigned int edge_pos = edge_size * edge;
    nNeighbours = d_output_graph[edge_pos + device::gbts_consts::nNei];
    for (unsigned int k = 0u; k < nNeighbours; ++k) {
      nei[k] = d_output_graph[edge_pos + device::gbts_consts::nei_start + k];
    }
  }
  // Iteration in which the cached edge is (re)visited next; -1 once settled.
  int active = 0;

  if (edge == 0u) {
    active_counters[0] = 0u;
  }
  grid.sync();

  for (unsigned char iter = 0; iter < max_iter; ++iter) {
    const unsigned int toggle = iter % 2u;
    const unsigned int levelLoad = toggle * n;
    const unsigned int levelStore = (1u - toggle) * n;
    bool stays_cached = false;
    unsigned int my_extra_active = 0u;

    if (has_edge && active == static_cast<int>(iter)) {
      stays_cached =
          gbts_cca_update(edge, nNeighbours, nei, iter, levelLoad, levelStore,
                          payload.minLevel, d_levels, d_outgoing_paths);
      active = stays_cached ? static_cast<int>(iter) + 1 : -1;
    }
    // Uncached extra edges (only when the grid does not cover all edges).
    if constexpr (EXTRA_EDGES) {
      for (unsigned int e = edge + nThreads; e < n; e += nThreads) {
        if (iter != 0 && d_active_edges[e] != static_cast<char>(iter)) {
          continue;
        }
        const unsigned int edge_pos = edge_size * e;
        const unsigned int nn =
            d_output_graph[edge_pos + device::gbts_consts::nNei];
        unsigned int ne[MAX_NEI];
        for (unsigned int k = 0u; k < nn; ++k) {
          ne[k] = d_output_graph[edge_pos + device::gbts_consts::nei_start + k];
        }
        const bool stays =
            gbts_cca_update(e, nn, ne, iter, levelLoad, levelStore,
                            payload.minLevel, d_levels, d_outgoing_paths);
        d_active_edges[e] = stays ? static_cast<char>(iter + 1u) : -1;
        my_extra_active += stays ? 1u : 0u;
      }
    }

    // Count the edges that stay active (block aggregated), zero the next
    // counter, and stop when nothing is left to do.
    unsigned int block_active =
        static_cast<unsigned int>(__syncthreads_count(stays_cached ? 1 : 0));
    if constexpr (EXTRA_EDGES) {
      if (threadIdx.x == 0u) {
        block_active_sum = 0u;
      }
      __syncthreads();
      if (my_extra_active != 0u) {
        atomicAdd(&block_active_sum, my_extra_active);
      }
      __syncthreads();
      block_active += block_active_sum;
    }
    if (threadIdx.x == 0u && block_active != 0u) {
      atomicAdd(active_counters + iter, block_active);
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

/// Cached CCA as a kernel of its own
template <unsigned int MAX_NEI, bool EXTRA_EDGES>
__global__ void gbts_run_cca_cached(
    const device::gbts_run_cca_iteration_payload payload) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  gbts_run_cca_cached_body<MAX_NEI, EXTRA_EDGES>(grid, payload);
}

/// Block-wide inclusive scan (blockDim.x a multiple of 32, <= 1024).
/// @c warp_sums is a shared array of 32 entries; returns the inclusive
/// prefix of @c v and sets @c block_total.
__device__ inline unsigned int gbts_block_inclusive_scan(
    unsigned int v, unsigned int* warp_sums, unsigned int& block_total) {
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int nwarps = blockDim.x / 32u;
  for (unsigned int o = 1u; o < 32u; o *= 2u) {
    const unsigned int u = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) {
      v += u;
    }
  }
  if (lane == 31u) {
    warp_sums[warp] = v;
  }
  __syncthreads();
  if (warp == 0u) {
    unsigned int w = (lane < nwarps) ? warp_sums[lane] : 0u;
    for (unsigned int o = 1u; o < 32u; o *= 2u) {
      const unsigned int u = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) {
        w += u;
      }
    }
    warp_sums[lane] = w;
  }
  __syncthreads();
  block_total = warp_sums[nwarps - 1u];
  v += (warp > 0u) ? warp_sums[warp - 1u] : 0u;
  __syncthreads();
  return v;
}

/// Payloads of the fused CCA + terminus counting kernel
struct gbts_cca_rows_payloads {
  device::gbts_run_cca_iteration_payload cca;
  device::gbts_count_terminus_edges_payload terminus;
};

/// CUDA kernel running the cached CCA, then the terminus-edge counting, the
/// inclusive scan of the row sizes (grid-wide) and the zeroing of the edge
/// and hit bids, all in one cooperative launch
template <unsigned int MAX_NEI, bool EXTRA_EDGES>
__global__ void gbts_cca_rows_cached(const gbts_cca_rows_payloads payloads) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __shared__ unsigned int warp_sums[32];
  __shared__ unsigned int block_offset_shared;

  gbts_run_cca_cached_body<MAX_NEI, EXTRA_EDGES>(grid, payloads.cca);
  grid.sync();

  const device::gbts_count_terminus_edges_payload& t = payloads.terminus;
  const vecmem::device_vector<const int2> d_outgoing_paths(t.outgoing_paths);
  vecmem::device_vector<unsigned int> d_row_sizes(t.row_sizes);
  vecmem::device_vector<unsigned long long int> d_edge_bids(t.edge_bids);
  vecmem::device_vector<unsigned long long int> d_hit_bids(t.hit_bids);
  unsigned int* block_sums =
      payloads.cca.active_counters + device::gbts_run_cca_row_count_slot + 1u;
  const unsigned int n = t.nConnectedEdges;
  const unsigned int nThreads = gridDim.x * blockDim.x;
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Zero the bids (grid-stride).
  for (unsigned int i = tid; i < d_edge_bids.size(); i += nThreads) {
    d_edge_bids[i] = 0ull;
  }
  for (unsigned int i = tid; i < d_hit_bids.size(); i += nThreads) {
    d_hit_bids[i] = 0ull;
  }

  // Phase 1: every block scans a contiguous chunk of the row sizes.
  const unsigned int chunk =
      ((n + gridDim.x - 1u) / gridDim.x + blockDim.x - 1u) / blockDim.x *
      blockDim.x;
  const unsigned int begin = blockIdx.x * chunk;
  const unsigned int end = (begin + chunk < n) ? begin + chunk : n;
  unsigned int carry = 0u;
  for (unsigned int base = begin; base < end; base += blockDim.x) {
    const unsigned int e = base + threadIdx.x;
    unsigned int v = 0u;
    if (e < end) {
      const int2 out_paths = d_outgoing_paths[e];
      v = (out_paths.y == -1) ? 0u
                              : 1u + static_cast<unsigned int>(out_paths.x);
    }
    unsigned int block_total = 0u;
    const unsigned int p = gbts_block_inclusive_scan(v, warp_sums, block_total);
    if (e < end) {
      d_row_sizes[e] = carry + p;
    }
    carry += block_total;
  }
  if (threadIdx.x == 0u) {
    block_sums[blockIdx.x] = carry;
  }
  grid.sync();

  // Phase 2: add the prefix of the preceding blocks (redundantly per
  // block), publish the total row count.
  if (threadIdx.x < 32u) {
    unsigned int acc = 0u;
    for (unsigned int b = threadIdx.x; b < blockIdx.x; b += 32u) {
      acc += block_sums[b];
    }
    for (unsigned int o = 16u; o > 0u; o /= 2u) {
      acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    if (threadIdx.x == 0u) {
      block_offset_shared = acc;
    }
  }
  __syncthreads();
  const unsigned int block_offset = block_offset_shared;
  if (block_offset != 0u) {
    for (unsigned int e = begin + threadIdx.x; e < end; e += blockDim.x) {
      d_row_sizes[e] += block_offset;
    }
  }
  if (blockIdx.x == gridDim.x - 1u && threadIdx.x == 0u) {
    *t.row_count = block_offset + carry;
  }
}

/// Bid of one proposal for the first @c depth edges of its path (walked on
/// the fly), see traccc::device::details::gbts_create_seed_candidate.
__device__ inline void gbts_bid_uncached(
    const unsigned int prop_idx, const int2 prop, const unsigned int depth,
    unsigned long long int* bids,
    const vecmem::device_vector<const int2>& d_path_store,
    vecmem::device_vector<int2>& d_seed_proposals,
    vecmem::device_vector<char>& d_seed_ambiguity) {
  d_seed_proposals[prop_idx] = prop;
  const unsigned long long int seed_bid =
      (static_cast<unsigned long long int>(prop.x) << 32) |
      static_cast<unsigned long long int>(prop_idx);
  int2 path = int2{0, prop.y};
  unsigned int k = 0u;
  while (path.y >= 0 && k < depth) {
    path = d_path_store[static_cast<unsigned int>(path.y)];
    ++k;
    const unsigned long long int competing_offer =
        atomicMax(bids + static_cast<unsigned int>(path.x), seed_bid);
    if (competing_offer > seed_bid) {
      d_seed_ambiguity[prop_idx] = -1;
    } else if (competing_offer != 0ull) {
      d_seed_ambiguity[static_cast<unsigned int>(competing_offer &
                                                 0xFFFFFFFFull)] = -1;
    }
  }
}

/// Fused seed-vs-edge bidding: rows [0, nThreads) have their proposal's
/// edge chain walked once and kept in registers across the rounds; rows
/// beyond the grid (rare: the grid is sized for the expected row count) are
/// processed uncached by the same threads. Same per-step semantics as
/// gbts_bid_seeds_for_edges / gbts_rebid_seeds_for_edges /
/// gbts_reset_edge_bids.
template <unsigned int MAX_LEN>
__device__ inline void gbts_bid_seeds_cached_body(
    cooperative_groups::grid_group& grid,
    const device::gbts_seed_bidding_payload& payload) {
  const vecmem::device_vector<const int2> d_path_store(payload.path_store);
  vecmem::device_vector<int2> d_seed_proposals(payload.seed_proposals);
  vecmem::device_vector<char> d_seed_ambiguity(payload.seed_ambiguity);
  vecmem::device_vector<unsigned long long int> d_edge_bids(payload.edge_bids);
  const unsigned int n = payload.nConnectedEdges;
  const unsigned int nRows =
      (*payload.row_count < payload.nRows) ? *payload.row_count : payload.nRows;
  const unsigned int nThreads = gridDim.x * blockDim.x;
  const unsigned int prop_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // The cached row of this thread.
  int2 prop = int2{0, -1};
  if (prop_idx < nRows) {
    prop = d_seed_proposals[prop_idx];
  }
  const bool has_prop = prop.y >= 0;
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

  // Bid for the first @c depth edges of the cached chain into @c bids.
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
  // Proposal of an uncached extra row (rows beyond the grid).
  const auto extra_prop = [&](const unsigned int row) {
    return d_seed_proposals[row];
  };

  // Initial bid: terminus edge only.
  if (has_prop) {
    bid(d_edge_bids.data(), (length > 0u) ? 1u : 0u);
  }
  for (unsigned int row = prop_idx + nThreads; row < nRows; row += nThreads) {
    const int2 p = extra_prop(row);
    if (p.y >= 0) {
      gbts_bid_uncached(row, p, 1u, d_edge_bids.data(), d_path_store,
                        d_seed_proposals, d_seed_ambiguity);
    }
  }

  for (unsigned int round = 0u; round < payload.nRounds; ++round) {
    const unsigned int half = (round + 1u) % 2u;
    unsigned long long int* bids = d_edge_bids.data() + half * n;
    unsigned long long int* bids_next = d_edge_bids.data() + (1u - half) * n;
    grid.sync();

    // --- rebid ---
    const auto rebid_decision = [&](const unsigned int row) {
      const char ambi = d_seed_ambiguity[row];
      if (round == 0u) {
        if (ambi == 0) {
          // rebid 'best seed from edge' in later rounds
          d_seed_ambiguity[row] = 1;
          return true;
        }
        d_seed_ambiguity[row] = -2;
        atomicAdd(payload.nRejectedPropsCounter, 1u);
        return false;
      }
      // only rebid for maybes
      return !((ambi == -2) | (ambi == 0));
    };
    if (has_prop && rebid_decision(prop_idx)) {
      bid(bids, length);
    }
    for (unsigned int row = prop_idx + nThreads; row < nRows; row += nThreads) {
      const int2 p = extra_prop(row);
      if (p.y >= 0 && rebid_decision(row)) {
        gbts_bid_uncached(row, p, 0xFFFFFFFFu, bids, d_path_store,
                          d_seed_proposals, d_seed_ambiguity);
      }
    }
    grid.sync();

    // --- reset: zero the next round's bids, then re-evaluate the maybes ---
    for (unsigned int idx = prop_idx; idx < n; idx += nThreads) {
      bids_next[idx] = 0ull;
    }
    const auto reset_row = [&](const unsigned int row, const bool isgood) {
      if (isgood) {
        d_seed_ambiguity[row] = 1;
      } else {
        d_seed_ambiguity[row] = -2;
        atomicAdd(payload.nRejectedPropsCounter, 1u);
      }
    };
    const auto is_maybe = [&](const unsigned int row) {
      const char ambi = d_seed_ambiguity[row];
      return !((ambi == -2) | (ambi == 0));
    };
    if (has_prop && is_maybe(prop_idx)) {
      bool isgood = true;
      for (unsigned int k = 0u; k < length; ++k) {
        const unsigned long long int best_bid = bids[chain[k]];
        if (d_seed_ambiguity[static_cast<unsigned int>(best_bid &
                                                       0xFFFFFFFFull)] == 0) {
          isgood = false;
          break;
        }
      }
      reset_row(prop_idx, isgood);
    }
    for (unsigned int row = prop_idx + nThreads; row < nRows; row += nThreads) {
      const int2 p = extra_prop(row);
      if (p.y < 0 || !is_maybe(row)) {
        continue;
      }
      bool isgood = true;
      int2 path = int2{0, p.y};
      while (path.y >= 0) {
        path = d_path_store[static_cast<unsigned int>(path.y)];
        const unsigned long long int best_bid =
            bids[static_cast<unsigned int>(path.x)];
        if (d_seed_ambiguity[static_cast<unsigned int>(best_bid &
                                                       0xFFFFFFFFull)] == 0) {
          isgood = false;
          break;
        }
      }
      reset_row(row, isgood);
    }
  }
}

/// Payloads of the fused seed finishing kernel
struct gbts_finish_seeds_payloads {
  device::gbts_seed_bidding_payload bidding;
  device::gbts_bid_seeds_for_hits_payload hits;
  device::gbts_convert_seeds_payload convert;
};

/// CUDA kernel running the bidding sequence (register cached), the hit
/// bidding and the seed conversion in one cooperative launch
template <unsigned int MAX_LEN>
__global__ void gbts_finish_seeds_cached(
    const gbts_finish_seeds_payloads payloads) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  gbts_bid_seeds_cached_body<MAX_LEN>(grid, payloads.bidding);
  grid.sync();
  device::gbts_bid_seeds_for_hits(details::thread_id1{}, payloads.hits);
  grid.sync();
  device::gbts_convert_seeds(details::thread_id1{}, payloads.convert);
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

void gbts_seeding_algorithm::gbts_sort_node_keys_kernel(
    const device::gbts_sort_nodes_payload& payload) const {
  // Keys-only radix sort of the (eta bin, quantised phi) bits above the
  // spacepoint index: 3 one-sweep passes at most, stable in the index.
  unsigned int eta_bits = 0u;
  while ((1u << eta_bits) <= payload.nEtaBins) {
    ++eta_bits;
  }
  const int begin_bit = static_cast<int>(device::gbts_sort_key_phi_shift);
  const int end_bit =
      static_cast<int>(device::gbts_sort_key_eta_shift + eta_bits);

  cudaStream_t cuda_stream = details::get_stream(stream());
  vecmem::data::vector_buffer<device::gbts_sort_key_t> keys_alt(payload.nKeys,
                                                                mr().main);
  cub::DoubleBuffer<device::gbts_sort_key_t> d_keys(payload.sort_keys.ptr(),
                                                    keys_alt.ptr());
  std::size_t temp_bytes = 0u;
  TRACCC_CUDA_ERROR_CHECK(cub::DeviceRadixSort::SortKeys(
      nullptr, temp_bytes, d_keys, static_cast<int>(payload.nKeys), begin_bit,
      end_bit, cuda_stream));
  vecmem::data::vector_buffer<char> temp(
      static_cast<unsigned int>(std::max<std::size_t>(temp_bytes, 1u)),
      mr().main);
  TRACCC_CUDA_ERROR_CHECK(cub::DeviceRadixSort::SortKeys(
      temp.ptr(), temp_bytes, d_keys, static_cast<int>(payload.nKeys),
      begin_bit, end_bit, cuda_stream));
  if (d_keys.Current() != payload.sort_keys.ptr()) {
    // The sorted keys ended up in the alternate buffer.
    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(payload.sort_keys.ptr(), d_keys.Current(),
                        payload.nKeys * sizeof(device::gbts_sort_key_t),
                        cudaMemcpyDeviceToDevice, cuda_stream));
  }
}

void gbts_seeding_algorithm::gbts_sort_nodes_kernel(
    const device::gbts_sort_nodes_payload& payload) const {
  const unsigned int n_threads = device::gbts_sort_nodes_block_size;
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
  const unsigned int n_blocks =
      std::min(1u + (payload.nEdgesMax - 1u) / n_threads, 8192u);
  kernels::gbts_match_graph_edges<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_reindex_edges_kernel(
    const device::gbts_reindex_edges_payload& payload) const {
  // The 1-byte flags are widened on the fly; the sum is accumulated in int.
  const unsigned char* kept = payload.kept.ptr();
  auto kept_int = thrust::make_transform_iterator(kept, kernels::byte_to_int{});
  thrust::inclusive_scan(
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(details::get_stream(stream())),
      kept_int, kept_int + payload.nEdgesMax, payload.reIndexer.ptr());
}

void gbts_seeding_algorithm::gbts_compress_graph_kernel(
    const device::gbts_compress_graph_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks =
      std::min(1u + (payload.nEdgesMax - 1u) / n_threads, 8192u);
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

void gbts_seeding_algorithm::gbts_run_cca_and_count_kernel(
    const device::gbts_run_cca_iteration_payload& cca,
    const device::gbts_count_terminus_edges_payload& terminus) const {
  constexpr unsigned int max_cached_nei = 16u;
  const unsigned int n_threads = 1024u;
  const kernels::gbts_cca_rows_payloads payloads{cca, terminus};
  if (cca.max_num_neighbours <= max_cached_nei) {
    if (launch_cooperative(kernels::gbts_cca_rows_cached<max_cached_nei, false>,
                           cca.nConnectedEdges, n_threads, payloads,
                           details::get_stream(stream()),
                           /*require_full_grid=*/true)) {
      return;
    }
    if (launch_cooperative(kernels::gbts_cca_rows_cached<max_cached_nei, true>,
                           cca.nConnectedEdges, n_threads, payloads,
                           details::get_stream(stream()))) {
      return;
    }
  }
  device::gbts_seeding_algorithm::gbts_run_cca_and_count_kernel(cca, terminus);
}

void gbts_seeding_algorithm::gbts_run_cca_kernel(
    const device::gbts_run_cca_iteration_payload& payload) const {
  constexpr unsigned int max_cached_nei = 16u;
  // Large blocks: the cost of an iteration is dominated by the grid-wide
  // barrier, which scales with the number of blocks.
  const unsigned int n_threads = 1024u;
  if (payload.max_num_neighbours <= max_cached_nei) {
    // Fast path: one edge per thread with the neighbour lists cached in
    // registers. Only possible when the whole grid can be resident.
    // Grid covers every edge: one cached edge per thread. Otherwise the
    // variant that also handles the edges beyond the grid uncached.
    if (launch_cooperative(kernels::gbts_run_cca_cached<max_cached_nei, false>,
                           payload.nConnectedEdges, n_threads, payload,
                           details::get_stream(stream()),
                           /*require_full_grid=*/true)) {
      return;
    }
    if (launch_cooperative(kernels::gbts_run_cca_cached<max_cached_nei, true>,
                           payload.nConnectedEdges, n_threads, payload,
                           details::get_stream(stream()))) {
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
  if (!launch_cooperative(kernels::gbts_bid_seeds, payload.nRowsGrid, 1024u,
                          payload, details::get_stream(stream()))) {
    device::gbts_seeding_algorithm::gbts_bid_seeds_kernel(payload);
  }
}

void gbts_seeding_algorithm::gbts_finish_seeds_kernel(
    const device::gbts_seed_bidding_payload& bidding,
    const device::gbts_bid_seeds_for_hits_payload& hits,
    const device::gbts_convert_seeds_payload& convert) const {
  // One cooperative kernel: cached bidding rounds, hit bidding, conversion.
  // The grid is sized for the expected row count (extra rows are handled
  // uncached / by grid-striding).
  const kernels::gbts_finish_seeds_payloads payloads{bidding, hits, convert};
  if (launch_cooperative(kernels::gbts_finish_seeds_cached<
                             traccc::device::gbts_consts::max_cca_iter + 1u>,
                         bidding.nRowsGrid, 1024u, payloads,
                         details::get_stream(stream()))) {
    return;
  }
  device::gbts_seeding_algorithm::gbts_finish_seeds_kernel(bidding, hits,
                                                           convert);
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
  // Publish the total row count (last entry of the scanned row sizes).
  TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
      payload.row_count, d_row_sizes.data() + payload.nConnectedEdges - 1u,
      sizeof(unsigned int), cudaMemcpyDeviceToDevice,
      details::get_stream(stream())));
}

void gbts_seeding_algorithm::gbts_fill_path_store_kernel(
    const device::gbts_fill_path_store_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_fill_path_store<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_bid_seeds_for_edges_kernel(
    const device::gbts_bid_seeds_for_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_bid_seeds_for_edges<<<n_blocks, n_threads, 0,
                                      details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_reset_edge_bids_kernel(
    const device::gbts_reset_edge_bids_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_reset_edge_bids<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_rebid_seeds_for_edges_kernel(
    const device::gbts_rebid_seeds_for_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_rebid_seeds_for_edges<<<n_blocks, n_threads, 0,
                                        details::get_stream(stream())>>>(
      payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_bid_seeds_for_hits_kernel(
    const device::gbts_bid_seeds_for_hits_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_bid_seeds_for_hits<<<n_blocks, n_threads, 0,
                                     details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_convert_seeds_kernel(
    const device::gbts_convert_seeds_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_convert_seeds<<<n_blocks, n_threads, 0,
                                details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
