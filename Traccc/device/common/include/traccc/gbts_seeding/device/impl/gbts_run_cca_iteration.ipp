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
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_run_cca_iteration(
    const thread_id_t& thread_id,
    const gbts_run_cca_iteration_payload& payload) {
  const vecmem::device_vector<const unsigned int> d_output_graph(
      payload.output_graph);
  vecmem::device_vector<unsigned char> d_levels(payload.levels);
  vecmem::device_vector<char> d_active_edges(payload.active_edges);
  vecmem::device_vector<int2> d_outgoing_paths(payload.outgoing_paths);
  vecmem::device_vector<unsigned char> d_has_parent(payload.has_parent);

  const unsigned char iter = payload.iter;

  const unsigned int edge_size = 2 + 1 + payload.max_num_neighbours;

  const unsigned int nConnectedEdges =
      (*payload.d_nConnectedEdges < payload.nConnectedEdges)
          ? *payload.d_nConnectedEdges
          : payload.nConnectedEdges;

  if (iter >= traccc::device::gbts_consts::max_cca_iter) {
    // Deterministic subtree counting, one pass per level (children strictly
    // before parents): pass p counts the edges of level p + 2 -
    // max_cca_iter. The first levels half holds the final level of every
    // settled edge (a settling edge rewrites its unchanged level, so both
    // halves agree).
    const unsigned char lvl =
        static_cast<unsigned char>(iter - gbts_consts::max_cca_iter + 2u);
    const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
    const unsigned int stride =
        thread_id.getBlockDimX() * thread_id.getGridDimX();
    for (unsigned int e = globalIdx; e < nConnectedEdges; e += stride) {
      if (d_levels[e] != lvl) {
        continue;
      }
      const unsigned int edge_pos = edge_size * e;
      const unsigned int nNeighbours =
          d_output_graph[edge_pos + gbts_consts::nNei];
      int out_paths = 0;
      for (unsigned int nIdx = 0; nIdx < nNeighbours; ++nIdx) {
        const unsigned int c =
            d_output_graph[edge_pos + gbts_consts::nei_start + nIdx];
        if (lvl == d_levels[c] + 1u) {
          out_paths += 1 + d_outgoing_paths[c].x;
        }
      }
      d_outgoing_paths[e].x = out_paths;
    }
    return;
  }
  const unsigned int toggle = iter % 2;
  const unsigned int levelLoad = toggle * nConnectedEdges;
  const unsigned int levelStore = (1 - toggle) * nConnectedEdges;

  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int blockDimX = thread_id.getBlockDimX();
  const unsigned int gridDimX = thread_id.getGridDimX();

  for (unsigned int globalIndex = globalIdx; globalIndex < nConnectedEdges;
       globalIndex += blockDimX * gridDimX) {
    if (iter != 0) {
      if (d_active_edges[globalIndex] != iter) {
        continue;
      }
    }

    const unsigned int edge_pos = edge_size * globalIndex;
    const unsigned int nNeighbours =
        d_output_graph[edge_pos + gbts_consts::nNei];

    unsigned char next_level = d_levels[levelLoad + globalIndex];

    bool localChange = false;
    for (unsigned int nIdx = 0; nIdx < nNeighbours; nIdx++) {
      const unsigned int nextglobalIndex =
          d_output_graph[edge_pos + gbts_consts::nei_start + nIdx];
      const unsigned char forward_level = d_levels[levelLoad + nextglobalIndex];
      if (next_level == forward_level) {
        next_level = forward_level + 1;
        localChange = true;
        break;
      }
    }
    if (localChange) {
      if (iter == traccc::device::gbts_consts::max_cca_iter - 1) {
        // Never settled: excluded from the roots; the subtree size is
        // written here so the counting passes read no garbage.
        d_outgoing_paths[globalIndex] = int2{0, -1};
        d_active_edges[globalIndex] = -1;
      } else {
        d_active_edges[globalIndex] = static_cast<char>(iter + 1u);
      }
    } else {
      d_active_edges[globalIndex] = -1;
      for (unsigned int nIdx = 0; nIdx < nNeighbours; ++nIdx) {
        const unsigned int nextglobalIndex =
            d_output_graph[edge_pos + gbts_consts::nei_start + nIdx];
        // Mark the neighbour as having a settled parent (idempotent, so
        // it cannot race with the neighbour's own settle write).
        d_has_parent[nextglobalIndex] = 1u;
      }
      // Flag as long enough to become a seed; the subtree size is counted
      // deterministically per level after the CCA has converged (a settling
      // edge must not read a same-iteration settling child's count).
      d_outgoing_paths[globalIndex] =
          int2{0, static_cast<int>(next_level >= payload.minLevel) - 1};
    }
    // store new level
    d_levels[levelStore + globalIndex] = next_level;
  }
}

}  // namespace traccc::device
