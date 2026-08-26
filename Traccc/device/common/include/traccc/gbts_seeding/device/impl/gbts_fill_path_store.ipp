/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_fill_path_store(
    const thread_id_t& thread_id, const gbts_fill_path_store_payload& payload) {
  vecmem::device_vector<int2> d_path_store(payload.path_store);
  const vecmem::device_vector<const unsigned int> d_output_graph(
      payload.output_graph);
  const vecmem::device_vector<const unsigned char> d_levels(payload.levels);
  const vecmem::device_vector<const int2> d_outgoing_paths(
      payload.outgoing_paths);
  const vecmem::device_vector<const unsigned int> d_row_sizes(
      payload.row_sizes);

  // Row-major output graph: each edge owns a contiguous block of
  // edge_size = 2 + 1 + max_num_neighbours ints.
  const unsigned int edge_size = 2u + 1u + payload.max_num_neighbours;

  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int blockDimX = thread_id.getBlockDimX();
  const unsigned int gridDimX = thread_id.getGridDimX();

  for (unsigned int row = globalIdx; row < payload.nRows;
       row += blockDimX * gridDimX) {

    unsigned int lo = 0u;
    unsigned int hi = payload.nConnectedEdges;
    while (lo < hi) {
      const unsigned int mid = lo + (hi - lo) / 2u;
      if (d_row_sizes[mid] > row) {
        hi = mid;
      } else {
        lo = mid + 1u;
      }
    }
    const unsigned int root = lo;
    const unsigned int base =
        d_row_sizes[root] -
        (1u + static_cast<unsigned int>(d_outgoing_paths[root].x));

    // Descend the preorder layout: at each node, skip its own row, then
    // find the child whose subtree range contains the remaining offset.
    unsigned int cur_edge = root;
    unsigned int cur_row = base;
    unsigned int offset = row - base;
    int parent_row = -1;
    while (offset > 0u) {
      --offset;
      const unsigned int edge_pos = edge_size * cur_edge;
      const unsigned int nNei = d_output_graph[edge_pos + gbts_consts::nNei];
      const unsigned char level = d_levels[cur_edge];
      unsigned int acc = 0u;
      bool found = false;
      for (unsigned int k = 0u; k < nNei; ++k) {
        const unsigned int child =
            d_output_graph[edge_pos + gbts_consts::nei_start + k];
        if (level != d_levels[child] + 1u) {
          continue;
        }
        const unsigned int size =
            1u + static_cast<unsigned int>(d_outgoing_paths[child].x);
        if (offset < size) {
          parent_row = static_cast<int>(cur_row);
          cur_row = cur_row + 1u + acc;
          cur_edge = child;
          found = true;
          break;
        }
        offset -= size;
        acc += size;
      }
      if (!found) {
        // This should never happen, but here we guard against a potential out-of-bounds access.
        break;
      }
    }
    d_path_store[row] = int2{static_cast<int>(cur_edge), parent_row};
  }
}

}  // namespace traccc::device
