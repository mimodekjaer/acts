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

/// Curvature (1/m) of a triplet, innermost first, from the conformal map
/// centred on its last spacepoint, as the CPU GBTS estimates it.
TRACCC_HOST_DEVICE inline float gbts_estimate_curvature(
    const traccc::float4& sp0, const traccc::float4& sp1,
    const traccc::float4& sp2) {
  const float r0 = math::sqrt(sp2.x * sp2.x + sp2.y * sp2.y);
  const float cosA = sp2.x / r0;
  const float sinA = sp2.y / r0;
  float u[2];
  float v[2];
  const traccc::float4* sps[2] = {&sp0, &sp1};
  for (unsigned int k = 0; k < 2; k++) {
    const float dx = sps[k]->x - sp2.x;
    const float dy = sps[k]->y - sp2.y;
    const float r2_inv = 1.0f / (dx * dx + dy * dy);
    const float xn = dx * cosA + dy * sinA;
    const float yn = -dx * sinA + dy * cosA;
    u[k] = xn * r2_inv;
    v[k] = yn * r2_inv;
  }
  const float du = u[0] - u[1];
  if (du == 0.0f) {
    return 0.0f;
  }
  const float A = (v[0] - v[1]) / du;
  const float B = v[1] - A * u[1];
  return 1000.0f * B / math::sqrt(1 + A * A);  // from mm^-1 to m^-1
}

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
  const vecmem::device_vector<const float4> d_sp_params(payload.reducedSP);
  vecmem::device_vector<unsigned long long int> d_hit_bids(payload.hit_bids);

  const float best_hit_frac = payload.gbts_convert_seeds_params.best_hit_frac;
  const float dcurv_cut_m = payload.gbts_convert_seeds_params.dropout_dcurv_m;
  const unsigned int split_min_size =
      payload.gbts_convert_seeds_params.split_min_size;
  const unsigned int split_max_size =
      payload.gbts_convert_seeds_params.split_max_size;
  const float split_max_eta = payload.gbts_convert_seeds_params.split_max_eta;

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
    // The spacepoints of the seed, innermost first.
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
    const float quality = static_cast<float>(prop.x);

    // Split a short central seed into two seeds dropping one spacepoint
    // each, unless its triplets agree on the curvature, as the CPU GBTS
    // does. The pseudorapidity is the one of the outermost edge.
    bool split = false;
    if ((n_sp >= split_min_size) && (n_sp <= split_max_size)) {
      const traccc::float4 sp_out = d_sp_params[sp_indices[n_sp - 1u]];
      const traccc::float4 sp_in = d_sp_params[sp_indices[n_sp - 2u]];
      const float tau = (sp_out.z - sp_in.z) /
                        (math::sqrt(sp_out.x * sp_out.x + sp_out.y * sp_out.y) -
                         math::sqrt(sp_in.x * sp_in.x + sp_in.y * sp_in.y));
      const float abs_eta =
          math::fabs(math::log(math::sqrt(1.0f + tau * tau) - tau));
      if (abs_eta < split_max_eta) {
        const unsigned int mid = n_sp / 2u;
        // the seed, the seed without its first and without its middle
        // spacepoint, each as (first, middle, last)
        const unsigned int t[3][3] = {
            {0u, mid, n_sp - 1u},
            {1u, 1u + (n_sp - 1u) / 2u, n_sp - 1u},
            {0u,
             ((n_sp - 1u) / 2u < mid) ? (n_sp - 1u) / 2u
                                      : (n_sp - 1u) / 2u + 1u,
             n_sp - 1u}};
        float curv[3];
        for (unsigned int k = 0; k < 3; ++k) {
          curv[k] =
              detail::gbts_estimate_curvature(d_sp_params[sp_indices[t[k][0]]],
                                              d_sp_params[sp_indices[t[k][1]]],
                                              d_sp_params[sp_indices[t[k][2]]]);
        }
        split = (math::fabs(curv[1] - curv[0]) >= dcurv_cut_m) ||
                (math::fabs(curv[2] - curv[0]) >= dcurv_cut_m) ||
                (math::fabs(curv[2] - curv[1]) >= dcurv_cut_m);
      }
    }

    if (!split) {
      seeds_device.push_back({sp_indices, n_sp, quality});
      continue;
    }
    // the drop-out seeds: without the first and without the middle
    // spacepoint
    for (const unsigned int skip : {0u, n_sp / 2u}) {
      std::array<unsigned int, edm::seed_max_spacepoints> drop_out{};
      unsigned int n = 0;
      for (unsigned int i = 0; i < n_sp; ++i) {
        if (i != skip) {
          drop_out[n++] = sp_indices[i];
        }
      }
      seeds_device.push_back({drop_out, n, quality});
    }
  }
}

}  // namespace traccc::device
