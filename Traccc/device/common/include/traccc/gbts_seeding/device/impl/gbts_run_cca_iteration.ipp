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

template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_run_cca_iteration(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_run_cca_iteration_payload& payload,
    const gbts_run_cca_iteration_shared_payload& shared_payload) {
  const vecmem::device_vector<const unsigned int> d_output_graph(
      payload.output_graph);
  vecmem::device_vector<unsigned char> d_levels(payload.levels);
  vecmem::device_vector<int2> d_outgoing_paths(payload.outgoing_paths);
  vecmem::device_vector<unsigned char> d_has_parent(payload.has_parent);
  vecmem::device_vector<unsigned int> shared_changed(shared_payload.changed);

  constexpr unsigned char max_iter = traccc::device::gbts_consts::max_cca_iter;
  const unsigned char iter = payload.iter;
  unsigned int* counters = payload.counters;

  const bool finishing = (iter == gbts_run_cca_finish_pass);
  // Converged: nothing changed in the previous sweep (block-uniform test;
  // the finishing pass always runs).
  if (!finishing && (iter > 0u) && (counters[iter - 1u] == 0u)) {
    return;
  }

  const unsigned int edge_size = 2u + 1u + payload.max_num_neighbours;
  const unsigned int n = (*payload.d_nConnectedEdges < payload.nConnectedEdges)
                             ? *payload.d_nConnectedEdges
                             : payload.nConnectedEdges;

  const unsigned int threadIndex = thread_id.getLocalThreadIdX();
  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int stride =
      thread_id.getBlockDimX() * thread_id.getGridDimX();

  if (finishing) {
    // Finishing pass: parent marks and terminus flags.
    for (unsigned int e = globalIdx; e < n; e += stride) {
      const unsigned char lvl = d_levels[e];
      if (lvl > max_iter) {
        // Did not settle: excluded from the roots, no parent marks.
        d_outgoing_paths[e] = int2{0, -1};
        continue;
      }
      const unsigned int edge_pos = edge_size * e;
      const unsigned int nNei = d_output_graph[edge_pos + gbts_consts::nNei];
      for (unsigned int k = 0u; k < nNei; ++k) {
        // Idempotent, race-free mark.
        d_has_parent[d_output_graph[edge_pos + gbts_consts::nei_start + k]] =
            1u;
      }
      d_outgoing_paths[e].y = static_cast<int>(lvl >= payload.minLevel) - 1;
    }
    return;
  }

  if (threadIndex == 0u) {
    shared_changed[0] = 0u;
  }
  if ((globalIdx == 0u) && (iter + 1u <= max_iter)) {
    // The counter of the next sweep (nobody writes it before that sweep).
    counters[iter + 1u] = 0u;
  }
  barrier.blockBarrier();

  bool changed = false;
  // Descending index order: an edge's neighbours have higher indices.
  for (unsigned int g = globalIdx; g < n; g += stride) {
    const unsigned int e = n - 1u - g;
    const unsigned int edge_pos = edge_size * e;
    const unsigned int nNei = d_output_graph[edge_pos + gbts_consts::nNei];
    unsigned int max_level = 0u;
    for (unsigned int k = 0u; k < nNei; ++k) {
      const unsigned int c =
          d_output_graph[edge_pos + gbts_consts::nei_start + k];
      const unsigned int l = d_levels[c];
      max_level = (l > max_level) ? l : max_level;
    }
    unsigned int lvl = 1u + max_level;
    if (lvl > max_iter + 1u) {
      lvl = max_iter + 1u;
    }
    // Subtree row count from the neighbours one level below (final one
    // sweep after the levels below are final).
    int count = 0;
    if (lvl <= max_iter) {
      for (unsigned int k = 0u; k < nNei; ++k) {
        const unsigned int c =
            d_output_graph[edge_pos + gbts_consts::nei_start + k];
        if (d_levels[c] + 1u == lvl) {
          // The counts are not initialised before the first sweep.
          count += 1 + ((iter == 0u) ? 0 : d_outgoing_paths[c].x);
        }
      }
    }
    if ((iter == 0u) || (d_levels[e] != lvl) ||
        (d_outgoing_paths[e].x != count)) {
      d_levels[e] = static_cast<unsigned char>(lvl);
      d_outgoing_paths[e].x = count;
      changed = true;
    }
  }
  if (changed) {
    vecmem::device_atomic_ref<unsigned int,
                              vecmem::device_address_space::local>(
        shared_changed[0])
        .fetch_add(1u);
  }
  barrier.blockBarrier();
  if ((threadIndex == 0u) && (shared_changed[0] != 0u)) {
    vecmem::device_atomic_ref<unsigned int>(counters[iter]).fetch_add(1u);
  }
}

}  // namespace traccc::device
