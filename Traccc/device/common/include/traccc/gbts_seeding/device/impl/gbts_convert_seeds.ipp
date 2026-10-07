// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

// Project include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

// System include(s).
#include <array>

namespace traccc::device {

namespace detail {

struct Tracklet {
  unsigned int
      nodes[traccc::device::gbts_consts::max_seed_candidate_length + 1];
  int size;
};

}  // namespace detail

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_convert_seeds(
    const thread_id_t& thread_id, const gbts_convert_seeds_payload& payload) {
  edm::seed_collection::device seeds_device(payload.output_seeds);
  const vecmem::device_vector<const int2> d_seed_proposals(
      payload.seed_proposals);
  const vecmem::device_vector<const char> d_seed_ambiguity(
      payload.seed_ambiguity);
  const vecmem::device_vector<const int2> d_path_store(payload.path_store);
  const vecmem::device_vector<const uint2> d_output_edge_nodes(
      payload.output_edge_nodes);
  vecmem::device_vector<unsigned long long int> d_hit_bids(payload.hit_bids);

  const float best_hit_frac = payload.gbts_convert_seeds_params.best_hit_frac;

  const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
  const unsigned int blockDimX = thread_id.getBlockDimX();
  const unsigned int gridDimX = thread_id.getGridDimX();

  const unsigned int path_count =
      vecmem::device_vector<const unsigned int>(payload.path_count)[0];
  const unsigned int nPaths =
      (path_count < payload.nPathsMax) ? path_count : payload.nPathsMax;
  for (unsigned int prop_idx = globalIdx; prop_idx < nPaths;
       prop_idx += blockDimX * gridDimX) {
    const int2 prop = d_seed_proposals[prop_idx];
    if (prop.y < 0) {
      continue;
    }
    if (d_seed_ambiguity[prop_idx] == -2) {
      continue;
    }
    char best_for_hit = 0;
    detail::Tracklet seed;
    seed.size = 0;
    int2 path = int2{0, prop.y};
    while (path.y >= 0) {
      path = d_path_store[static_cast<unsigned int>(path.y)];
      seed.nodes[seed.size++] =
          d_output_edge_nodes[static_cast<unsigned int>(path.x)].x;
      best_for_hit +=
          (prop_idx == (d_hit_bids[seed.nodes[seed.size - 1]] & 0xFFFFFFFFLL));
    }
    seed.nodes[seed.size++] =
        d_output_edge_nodes[static_cast<unsigned int>(path.x)].y;
    best_for_hit +=
        (prop_idx == (d_hit_bids[seed.nodes[seed.size - 1]] & 0xFFFFFFFFLL));

    // Reject the seed if more than best_hit_frac of its hits went to better
    // seeds, as the CPU GBTS does.
    if (static_cast<float>(seed.size - best_for_hit) >
        best_hit_frac * static_cast<float>(seed.size)) {
      continue;
    }
    // Output the whole path as one seed, innermost spacepoint first.
    std::array<unsigned int, edm::seed_max_spacepoints> sp_indices{};
    // A local copy: the namespace-scope constant cannot be referenced from
    // device code.
    constexpr unsigned int max_sp = edm::seed_max_spacepoints;
    const unsigned int n_sp = (static_cast<unsigned int>(seed.size) < max_sp)
                                  ? static_cast<unsigned int>(seed.size)
                                  : max_sp;
    for (unsigned int i = 0; i < n_sp; ++i) {
      sp_indices[i] = seed.nodes[static_cast<unsigned int>(seed.size) - 1u - i];
    }
    seeds_device.push_back({sp_indices, n_sp, static_cast<float>(prop.x)});
  }
}

}  // namespace traccc::device
