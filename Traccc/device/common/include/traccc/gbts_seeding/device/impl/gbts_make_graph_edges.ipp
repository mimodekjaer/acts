/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"
#include "traccc/utils/trigonometric_helpers.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

namespace detail {

/// Padding added to the phi search window. The window only pre-selects
/// candidates; the exact delta-phi cut in gbts_edge_passes_cuts decides, so
/// the padding merely guards against rounding differences between the two.
inline constexpr float gbts_phi_window_eps = 1e-5f;

/// First index in [begin, end) of @c phis whose value is >= @c value
/// (the range must be sorted ascending).
template <typename vector_t>
TRACCC_HOST_DEVICE inline unsigned int gbts_phi_lower_bound(
    const vector_t& phis, unsigned int begin, unsigned int end,
    const float value) {
  while (begin < end) {
    const unsigned int mid = begin + (end - begin) / 2u;
    if (phis[mid] < value) {
      begin = mid + 1u;
    } else {
      end = mid;
    }
  }
  return begin;
}

/// First index in [begin, end) of @c phis whose value is > @c value
/// (the range must be sorted ascending).
template <typename vector_t>
TRACCC_HOST_DEVICE inline unsigned int gbts_phi_upper_bound(
    const vector_t& phis, unsigned int begin, unsigned int end,
    const float value) {
  while (begin < end) {
    const unsigned int mid = begin + (end - begin) / 2u;
    if (phis[mid] <= value) {
      begin = mid + 1u;
    } else {
      end = mid;
    }
  }
  return begin;
}

/// A phi window [lo, hi] split at the +/- pi boundary into up to two
/// intervals, ordered by ascending phi (and therefore ascending node index).
struct gbts_phi_window {
  float lo_a, hi_a;  // first interval
  float lo_b, hi_b;  // second interval, hi_b < lo_b when unused
  bool whole;        // window spans the full circle
};

TRACCC_HOST_DEVICE inline gbts_phi_window gbts_make_phi_window(const float lo,
                                                               const float hi) {
  gbts_phi_window w{lo, hi, 1.0f, 0.0f, false};
  if (hi - lo >= traccc::device::TWO_PI_F) {
    w.whole = true;
  } else if (lo < -traccc::device::PI_F) {
    // wraps below -pi: [-pi, hi] then [lo + 2pi, pi]
    w.lo_a = -traccc::device::PI_F - 1.0f;
    w.hi_a = hi;
    w.lo_b = lo + traccc::device::TWO_PI_F;
    w.hi_b = traccc::device::PI_F + 1.0f;
  } else if (hi > traccc::device::PI_F) {
    // wraps above pi: [-pi, hi - 2pi] then [lo, pi]
    w.lo_a = -traccc::device::PI_F - 1.0f;
    w.hi_a = hi - traccc::device::TWO_PI_F;
    w.lo_b = lo;
    w.hi_b = traccc::device::PI_F + 1.0f;
  }
  return w;
}

/// The RZ-doublet + curvature cuts for a candidate edge (node1 -> node2).
/// Node params are float4 (tau_min, tau_max, r, z). Returns true when the
/// candidate passes; the derived quantities are then valid.
TRACCC_HOST_DEVICE inline bool gbts_edge_passes_cuts(
    const float4 node_params_1, const float4 node_params_2, const float phi1,
    const float phi2, const float deltaPhi,
    const gbts_make_graph_edges_params& ap, float& tau, float& dr, float& dz,
    float& curv) {
  const float tau_min1 = node_params_1.x;
  const float tau_max1 = node_params_1.y;
  const float r1 = node_params_1.z;
  const float z1 = node_params_1.w;
  const float tau_min2 = node_params_2.x;
  const float tau_max2 = node_params_2.y;
  const float r2 = node_params_2.z;
  const float z2 = node_params_2.w;
  dr = r2 - r1;

  if (dr < ap.minDeltaRadius) {
    return false;
  }
  dz = z2 - z1;
  tau = dz / dr;
  const float ftau = math::fabs(tau);

  if ((ftau < tau_min2) || (ftau > tau_max2)) {
    return false;
  }
  if ((ftau < tau_min1) || (ftau > tau_max1)) {
    return false;
  }
  // RZ doublet filter cuts
  const float z0 = z1 - r1 * tau;
  if ((z0 < ap.min_z0) || (z0 > ap.max_z0)) {
    return false;
  }
  const float zouter = z0 + ap.maxOuterRadius * tau;
  if (zouter < ap.cut_zMinU || zouter > ap.cut_zMaxU) {
    return false;
  }

  const float dphi = traccc::detail::wrap_phi(phi2 - phi1);
  if (math::fabs(dphi) > deltaPhi) {
    return false;
  }
  curv = dphi / dr;
  const float curv_max = ftau < ap.max_Kappa_change_tau ? ap.max_Kappa_low_tau
                                                        : ap.max_Kappa_high_tau;
  if (math::fabs(curv) > curv_max) {
    return false;
  }
  return true;
}

/// Walk the slab entries [begin, end) of the phi-sorted outer-node slab that
/// fall into [lo, hi] and either count or write the edges of inner node
/// @c node1. Returns the updated count / cursor.
template <bool fill>
TRACCC_HOST_DEVICE inline unsigned int gbts_walk_slab_interval(
    const vecmem::device_vector<float>& shared_phi,
    const vecmem::device_vector<float4>& shared_node_pack,
    const unsigned int slab_begin, const unsigned int slab_size, const float lo,
    const float hi, const float4 np1, const float phi1,
    const unsigned int node1, const float deltaPhi,
    const gbts_make_graph_edges_params& ap,
    const edge_params_converter& edge_params_maker,
    vecmem::device_vector<uint2>& d_edge_nodes,
    vecmem::device_vector<short4>& d_edge_params,
    vecmem::device_vector<unsigned char>& d_reindexer, unsigned int cursor,
    const unsigned int cursor_end) {
  if (hi < shared_phi[0] || lo > shared_phi[slab_size - 1u]) {
    return cursor;
  }
  for (unsigned int j = gbts_phi_lower_bound(shared_phi, 0u, slab_size, lo);
       j < slab_size; j++) {
    const float phi2 = shared_phi[j];
    if (phi2 > hi) {
      break;
    }
    const float4 np2 = shared_node_pack[j];
    float tau, dr, dz, curv;
    if (!gbts_edge_passes_cuts(np1, np2, phi1, phi2, deltaPhi, ap, tau, dr, dz,
                               curv)) {
      continue;
    }
    if constexpr (fill) {
      if (cursor >= cursor_end) {
        // Count / fill mismatch guard: never write outside the bucket.
        return cursor;
      }
      const float eta = -1 * math::log(math::sqrt(1.0f + tau * tau) - tau);
      // edge linking order is inside->out
      d_edge_nodes[cursor] = uint2{slab_begin + j, node1};
      d_reindexer[cursor] = 0u;
      const bool inflate_matching_cuts = (ap.long_edge_dz < math::fabs(dz)) ||
                                         (ap.long_edge_dr < math::fabs(dr));
      d_edge_params[cursor] = edge_params_maker.make_edge_params(
          eta, curv, phi2 + curv * np2.z, phi1 + curv * np1.z,
          inflate_matching_cuts);
      // edge params: (eta, curvature, extrapolated phi at node2,
      //               extrapolated phi at node1)
    }
    cursor++;
  }
  return cursor;
}

}  // namespace detail

template <bool fill, concepts::thread_id1 thread_id_t,
          concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_make_graph_edges(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_make_graph_edges_payload& payload,
    const gbts_make_graph_edges_shared_payload& shared) {
  const unsigned int threadIndex = thread_id.getLocalThreadIdX();
  const unsigned int blockSize = thread_id.getBlockDimX();

  const vecmem::device_vector<const uint2> d_work_items(payload.work_items);
  const vecmem::device_vector<const unsigned int> d_pair_work_begin(
      payload.pair_work_begin);
  const vecmem::device_vector<const uint2> d_bin_pairs(payload.bin_pairs);
  const vecmem::device_vector<const unsigned int> d_pair_group_begin(
      payload.pair_group_begin);
  const vecmem::device_vector<const unsigned int> d_eta_bin_views(
      payload.eta_bin_views);
  const vecmem::device_vector<const float> d_bin_rads(payload.bin_rads);
  const vecmem::device_vector<const float4> d_node_params(payload.node_params);
  const vecmem::device_vector<const float> d_node_phi(payload.node_phi);
  vecmem::device_vector<unsigned int> d_edge_counts(payload.edge_counts);
  vecmem::device_vector<unsigned int> d_num_outgoing_edges(
      payload.num_outgoing_edges);
  vecmem::device_vector<uint2> d_edge_nodes(payload.edge_nodes);
  vecmem::device_vector<short4> d_edge_params(payload.edge_params);
  vecmem::device_vector<unsigned char> d_reindexer(payload.reindexer);

  vecmem::device_vector<float> shared_phi(shared.phi);
  vecmem::device_vector<float4> shared_node_pack(shared.node_pack);
  vecmem::device_vector<unsigned int> shared_work_slot(shared.work_slot);

  const gbts_make_graph_edges_params& ap = payload.gbts_make_graph_edges_params;
  constexpr unsigned int chunk_size = gbts_consts::node_buffer_length;

  const unsigned int nWork = *payload.nWork;
  // Dynamic scheduling: every block grabs the next work item until the list
  // is exhausted. The assignment of items to blocks is arbitrary, the output
  // positions of every item are not.
  for (;;) {
    if (threadIndex == 0u) {
      shared_work_slot[0] =
          vecmem::device_atomic_ref<unsigned int>(*payload.work_cursor)
              .fetch_add(1u);
    }
    barrier.blockBarrier();
    const unsigned int work = shared_work_slot[0];
    if (work >= nWork) {
      break;
    }
    if constexpr (fill) {
      if ((work == 0u) && (threadIndex == 0u)) {
        // The scanned per-node counts end with the total edge count.
        const unsigned int total =
            d_num_outgoing_edges[d_num_outgoing_edges.size() - 1u];
        *payload.nEdgesTotal = total;
        *payload.nEdges =
            (total < payload.nEdgesMax) ? total : payload.nEdgesMax;
      }
    }
    // --- Block-uniform setup ------------------------------------------------
    const uint2 item = d_work_items[work];
    const unsigned int pair = item.x;
    const unsigned int chunk = item.y;
    const uint2 bins = d_bin_pairs[pair];

    const unsigned int begin1 = d_eta_bin_views[2u * bins.x];
    const unsigned int end1 = d_eta_bin_views[2u * bins.x + 1u];
    const unsigned int begin2 = d_eta_bin_views[2u * bins.y];
    const unsigned int end2 = d_eta_bin_views[2u * bins.y + 1u];

    const unsigned int chunk_begin = begin1 + chunk * chunk_size;
    const unsigned int chunk_end =
        (chunk_begin + chunk_size < end1) ? chunk_begin + chunk_size : end1;
    const unsigned int num_nodes1 = chunk_end - chunk_begin;

    // delta-phi window of the bin pair from the radial separation of the bins
    const float rb1 = d_bin_rads[2u * bins.x];
    const float rb2 = d_bin_rads[2u * bins.y + 1u];
    const float maxDeltaR = math::fabs(rb2 - rb1);
    const gbts_dphi_window_params& dp = payload.gbts_dphi_window_params;
    float deltaPhi = dp.min_delta_phi + dp.dphi_coeff * maxDeltaR;
    if (maxDeltaR < dp.low_dr_threshold) {
      deltaPhi = dp.min_delta_phi_low_dr + dp.dphi_coeff_low_dr * maxDeltaR;
    }
    const float window = deltaPhi + detail::gbts_phi_window_eps;

    // Outer-node index ranges that can pair with any node of the chunk: up to
    // two (wraparound), the lower-index one first. Computed redundantly by
    // every thread from block-uniform data.
    const detail::gbts_phi_window block_window = detail::gbts_make_phi_window(
        d_node_phi[chunk_begin] - window, d_node_phi[chunk_end - 1u] + window);
    unsigned int range_begin[2] = {begin2, end2};
    unsigned int range_end[2] = {end2, end2};
    if (!block_window.whole) {
      range_begin[0] = detail::gbts_phi_lower_bound(d_node_phi, begin2, end2,
                                                    block_window.lo_a);
      range_end[0] = detail::gbts_phi_upper_bound(d_node_phi, range_begin[0],
                                                  end2, block_window.hi_a);
      if (block_window.lo_b <= block_window.hi_b) {
        range_begin[1] = detail::gbts_phi_lower_bound(d_node_phi, range_end[0],
                                                      end2, block_window.lo_b);
        range_end[1] = detail::gbts_phi_upper_bound(d_node_phi, range_begin[1],
                                                    end2, block_window.hi_b);
      }
    }

    // --- Per-thread setup
    // -----------------------------------------------------
    const bool active = threadIndex < num_nodes1;
    const unsigned int node1 = chunk_begin + (active ? threadIndex : 0u);
    const float phi1 = d_node_phi[node1];
    const float4 np1 = d_node_params[node1];
    const detail::gbts_phi_window my_window =
        detail::gbts_make_phi_window(phi1 - window, phi1 + window);

    unsigned int cursor = 0u;
    unsigned int cursor_end = 0u;
    if constexpr (fill) {
      if (active) {
        cursor = d_num_outgoing_edges[node1];
        cursor_end = d_num_outgoing_edges[node1 + 1u];
        // Deterministic truncation at the edge buffer capacity.
        if (cursor_end > payload.nEdgesMax) {
          cursor_end = payload.nEdgesMax;
        }
        // Edges of the preceding pairs of the same inner bin come first in the
        // bucket; they were processed by the block of the same chunk index.
        for (unsigned int p = d_pair_group_begin[pair]; p < pair; p++) {
          const unsigned int wb = d_pair_work_begin[p];
          if (d_pair_work_begin[p + 1u] > wb) {
            cursor += d_edge_counts[(wb + chunk) * blockSize + threadIndex];
          }
        }
      }
    }

    // --- Stream the outer-node ranges through shared memory
    // -------------------
    for (unsigned int r = 0u; r < 2u; r++) {
      for (unsigned int slab_begin = range_begin[r]; slab_begin < range_end[r];
           slab_begin += chunk_size) {
        const unsigned int slab_size = (slab_begin + chunk_size < range_end[r])
                                           ? chunk_size
                                           : range_end[r] - slab_begin;
        // The previous slab is fully consumed before it gets overwritten.
        barrier.blockBarrier();
        for (unsigned int j = threadIndex; j < slab_size; j += blockSize) {
          shared_phi[j] = d_node_phi[slab_begin + j];
          shared_node_pack[j] = d_node_params[slab_begin + j];
        }
        barrier.blockBarrier();

        if (!active) {
          continue;
        }
        if (my_window.whole) {
          cursor = detail::gbts_walk_slab_interval<fill>(
              shared_phi, shared_node_pack, slab_begin, slab_size,
              -traccc::device::PI_F - 1.0f, traccc::device::PI_F + 1.0f, np1,
              phi1, node1, deltaPhi, ap, payload.edge_params_maker,
              d_edge_nodes, d_edge_params, d_reindexer, cursor, cursor_end);
        } else {
          cursor = detail::gbts_walk_slab_interval<fill>(
              shared_phi, shared_node_pack, slab_begin, slab_size,
              my_window.lo_a, my_window.hi_a, np1, phi1, node1, deltaPhi, ap,
              payload.edge_params_maker, d_edge_nodes, d_edge_params,
              d_reindexer, cursor, cursor_end);
          if (my_window.lo_b <= my_window.hi_b) {
            cursor = detail::gbts_walk_slab_interval<fill>(
                shared_phi, shared_node_pack, slab_begin, slab_size,
                my_window.lo_b, my_window.hi_b, np1, phi1, node1, deltaPhi, ap,
                payload.edge_params_maker, d_edge_nodes, d_edge_params,
                d_reindexer, cursor, cursor_end);
          }
        }
      }
    }

    if constexpr (!fill) {
      const unsigned int count = active ? cursor : 0u;
      d_edge_counts[work * blockSize + threadIndex] = count;
      if (count > 0u) {
        vecmem::device_atomic_ref<unsigned int>(
            d_num_outgoing_edges[node1 + 1u])
            .fetch_add(count);
      }
    }
  }  // work item loop
}

}  // namespace traccc::device
