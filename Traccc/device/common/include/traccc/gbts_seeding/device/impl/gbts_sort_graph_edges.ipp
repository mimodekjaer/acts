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
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

namespace details {

/// Rank @c outer among the @c count bucket entries starting at @c keys
/// (a global or shared-memory view) and copy the edge to its canonical slot
/// @c begin + rank.
template <typename keys_t>
TRACCC_HOST_DEVICE inline void gbts_place_graph_edge(
    const keys_t& keys, const unsigned int count, const unsigned int begin,
    const unsigned int outer, const unsigned int edge,
    const vecmem::device_vector<const uint2>& d_edge_nodes,
    const vecmem::device_vector<const short4>& d_edge_params,
    vecmem::device_vector<uint2>& d_edge_nodes_sorted,
    vecmem::device_vector<short4>& d_edge_params_sorted) {
  // Outer node indices are unique within a bucket, so the rank is a
  // bijection onto [begin, begin + count).
  unsigned int rank = 0u;
  for (unsigned int k = 0u; k < count; k++) {
    rank += (keys[k] < outer) ? 1u : 0u;
  }
  const unsigned int sorted_idx = begin + rank;
  d_edge_nodes_sorted[sorted_idx] = d_edge_nodes[edge];
  d_edge_params_sorted[sorted_idx] = d_edge_params[edge];
}

}  // namespace details

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_sort_graph_edges_small(
    const thread_id_t& thread_id,
    const gbts_sort_graph_edges_payload& payload) {
  const vecmem::device_vector<const uint2> d_edge_nodes(payload.edge_nodes);
  const vecmem::device_vector<const short4> d_edge_params(payload.edge_params);
  const vecmem::device_vector<const unsigned int> d_num_outgoing_edges(
      payload.num_outgoing_edges);
  const vecmem::device_vector<const unsigned int> d_bucket_outer(
      payload.bucket_outer);
  vecmem::device_vector<uint2> d_edge_nodes_sorted(payload.edge_nodes_sorted);
  vecmem::device_vector<short4> d_edge_params_sorted(
      payload.edge_params_sorted);

  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int blockDimX = thread_id.getBlockDimX();
  const unsigned int gridDimX = thread_id.getGridDimX();

  for (unsigned int globalIndex = globalIdx; globalIndex < payload.nEdges;
       globalIndex += blockDimX * gridDimX) {
    const uint2 nodes = d_edge_nodes[globalIndex];
    const unsigned int begin = d_num_outgoing_edges[nodes.y];
    const unsigned int end = d_num_outgoing_edges[nodes.y + 1u];
    if (end - begin > gbts_sort_graph_edges_small_bucket) {
      continue;  // handled by gbts_sort_graph_edges_large
    }
    details::gbts_place_graph_edge(&d_bucket_outer[begin], end - begin, begin,
                                   nodes.x, globalIndex, d_edge_nodes,
                                   d_edge_params, d_edge_nodes_sorted,
                                   d_edge_params_sorted);
  }
}

template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_sort_graph_edges_large(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_sort_graph_edges_payload& payload,
    const gbts_sort_graph_edges_shared_payload& shared_payload) {
  const vecmem::device_vector<const uint2> d_edge_nodes(payload.edge_nodes);
  const vecmem::device_vector<const short4> d_edge_params(payload.edge_params);
  const vecmem::device_vector<const unsigned int> d_num_outgoing_edges(
      payload.num_outgoing_edges);
  const vecmem::device_vector<const unsigned int> d_bucket_outer(
      payload.bucket_outer);
  const vecmem::device_vector<const unsigned int> d_bucket_edge(
      payload.bucket_edge);
  vecmem::device_vector<uint2> d_edge_nodes_sorted(payload.edge_nodes_sorted);
  vecmem::device_vector<short4> d_edge_params_sorted(
      payload.edge_params_sorted);
  vecmem::device_vector<unsigned int> cache(shared_payload.bucket_cache);
  vecmem::device_vector<unsigned int> large_flags(shared_payload.large_flags);

  const unsigned int threadIndex = thread_id.getLocalThreadIdX();
  const unsigned int blockSize = thread_id.getBlockDimX();

  const unsigned int blockIndex = thread_id.getBlockIdX();
  const unsigned int nBlocks = thread_id.getGridDimX();

  // Every thread tests one node, then the block works through the flagged
  // buckets together. Large buckets cluster in node index (inner layers), so
  // the nodes of one block are interleaved with a stride of the grid size to
  // spread such runs over all blocks. Every branch below is uniform across
  // the block, so the barriers are always reached by all of its threads.
  for (unsigned int node0 = 0u; node0 < payload.nNodes;
       node0 += nBlocks * blockSize) {
    const unsigned int my_node = node0 + threadIndex * nBlocks + blockIndex;
    unsigned int is_large = 0u;
    if (my_node < payload.nNodes) {
      is_large = (d_num_outgoing_edges[my_node + 1u] -
                      d_num_outgoing_edges[my_node] >
                  gbts_sort_graph_edges_small_bucket)
                     ? 1u
                     : 0u;
    }
    large_flags[threadIndex] = is_large;
    barrier.blockBarrier();

    for (unsigned int j = 0u; j < blockSize; ++j) {
      if (large_flags[j] == 0u) {
        continue;
      }
      const unsigned int node = node0 + j * nBlocks + blockIndex;
      const unsigned int begin = d_num_outgoing_edges[node];
      const unsigned int end = d_num_outgoing_edges[node + 1u];
      const unsigned int count = end - begin;

      if (count <= gbts_sort_graph_edges_cache_size) {
        for (unsigned int k = threadIndex; k < count; k += blockSize) {
          cache[k] = d_bucket_outer[begin + k];
        }
        barrier.blockBarrier();
        for (unsigned int k = threadIndex; k < count; k += blockSize) {
          details::gbts_place_graph_edge(
              cache, count, begin, cache[k], d_bucket_edge[begin + k],
              d_edge_nodes, d_edge_params, d_edge_nodes_sorted,
              d_edge_params_sorted);
        }
        // The cache is rewritten for the next bucket.
        barrier.blockBarrier();
      } else {
        for (unsigned int k = threadIndex; k < count; k += blockSize) {
          details::gbts_place_graph_edge(
              &d_bucket_outer[begin], count, begin, d_bucket_outer[begin + k],
              d_bucket_edge[begin + k], d_edge_nodes, d_edge_params,
              d_edge_nodes_sorted, d_edge_params_sorted);
        }
      }
    }
    // The flags are rewritten for the next batch of nodes.
    barrier.blockBarrier();
  }
}

}  // namespace traccc::device
