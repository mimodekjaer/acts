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

/// Block-cooperative search of up to four boundaries in the phi-sorted node
/// range [begin, end): boundary k is lower_bound(values[k]) for even k and
/// upper_bound(values[k]) for odd k (only the first @c n_values are
/// searched). Round 1 probes blockSize evenly spaced nodes, round 2 lets 32
/// threads per boundary refine inside the probe interval. All threads of
/// the block must call this; the results are broadcast through
/// @c shared_bounds (>= 8 entries). @c shared_phi must hold >= blockSize
/// entries and is clobbered.
template <typename barrier_t>
TRACCC_HOST_DEVICE inline void gbts_block_find_bounds(
    const barrier_t& barrier, const unsigned int threadIndex,
    const unsigned int blockSize,
    const vecmem::device_vector<const float>& d_node_phi,
    const unsigned int begin, const unsigned int end, const float* values,
    const unsigned int n_values, vecmem::device_vector<float>& shared_phi,
    vecmem::device_vector<unsigned int>& shared_bounds, unsigned int* bounds) {
  const unsigned int n = end - begin;
  // Round 1: probe positions pos(t) = begin + t * n / blockSize.
  const auto pos = [&](const unsigned int t) {
    return begin +
           static_cast<unsigned int>(
               (static_cast<unsigned long long int>(t) * n) / blockSize);
  };
  shared_phi[threadIndex] = d_node_phi[pos(threadIndex)];
  barrier.blockBarrier();
  // The predicate P_k(i): the boundary lies after node i.
  const auto pred = [&](const unsigned int k, const float phi) {
    return (k % 2u == 0u) ? (phi < values[k]) : (phi <= values[k]);
  };
  // Find, per boundary, the last probe satisfying the predicate (or none).
  for (unsigned int k = 0u; k < n_values; ++k) {
    const bool p_here = pred(k, shared_phi[threadIndex]);
    const bool p_next = (threadIndex + 1u < blockSize)
                            ? pred(k, shared_phi[threadIndex + 1u])
                            : false;
    if (threadIndex == 0u && !p_here) {
      shared_bounds[k] = blockSize;  // sentinel: boundary == begin
    } else if (p_here && !p_next) {
      shared_bounds[k] = threadIndex;
    }
  }
  barrier.blockBarrier();
  // Round 2: 32 threads per boundary scan the probe interval
  // (pos(m), pos(m + 1)], with pos(blockSize) == end, in chunks of 32.
  const unsigned int k = threadIndex / 32u;
  const unsigned int lane = threadIndex % 32u;
  if (k < n_values) {
    const unsigned int m = shared_bounds[k];
    if (m == blockSize) {
      if (lane == 0u) {
        shared_bounds[4u + k] = begin;
      }
    } else {
      const unsigned int first = pos(m) + 1u;
      const unsigned int last = (m + 1u < blockSize) ? pos(m + 1u) : end;
      for (unsigned int i = first + lane; i <= last; i += 32u) {
        // P(i - 1) holds by construction for i == first.
        const bool p_prev = (i == first) ? true : pred(k, d_node_phi[i - 1u]);
        const bool p_here = (i < end) ? pred(k, d_node_phi[i]) : false;
        if (p_prev && !p_here) {
          shared_bounds[4u + k] = i;
        }
      }
    }
  }
  barrier.blockBarrier();
  for (unsigned int b = 0u; b < n_values; ++b) {
    bounds[b] = shared_bounds[4u + b];
  }
  // The scratch is reused by the next work item only after further
  // barriers, so no trailing barrier is needed here.
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
/// Write the accepted edge (node1 -> node2) at @c cursor.
TRACCC_HOST_DEVICE inline void gbts_emit_edge(
    const unsigned int cursor, const unsigned int node2,
    const unsigned int node1, const float4 np1, const float4 np2,
    const float phi1, const float phi2, const float tau, const float dr,
    const float dz, const float curv, const gbts_make_graph_edges_params& ap,
    const edge_params_converter& edge_params_maker,
    vecmem::device_vector<uint2>& d_edge_nodes,
    vecmem::device_vector<short4>& d_edge_params,
    vecmem::device_vector<unsigned char>& d_reindexer) {
  const float eta = -1 * math::log(math::sqrt(1.0f + tau * tau) - tau);
  // edge linking order is inside->out
  d_edge_nodes[cursor] = uint2{node2, node1};
  d_reindexer[cursor] = 0u;
  const bool inflate_matching_cuts =
      (ap.long_edge_dz < math::fabs(dz)) || (ap.long_edge_dr < math::fabs(dr));
  d_edge_params[cursor] = edge_params_maker.make_edge_params(
      eta, curv, phi2 + curv * np2.z, phi1 + curv * np1.z,
      inflate_matching_cuts);
  // edge params: (eta, curvature, extrapolated phi at node2,
  //               extrapolated phi at node1)
}

/// Walk the phi-sorted outer nodes [begin, end) (absolute indices into
/// @c phis / @c packs) that fall into [lo, hi] and either count or write the
/// edges of inner node @c node1. Returns the updated count / cursor.
template <bool fill, typename phi_vector_t, typename pack_vector_t>
TRACCC_HOST_DEVICE inline unsigned int gbts_walk_range_interval(
    const phi_vector_t& phis, const pack_vector_t& packs,
    const unsigned int begin, const unsigned int end, const float lo,
    const float hi, const float4 np1, const float phi1,
    const unsigned int node1, const float deltaPhi,
    const gbts_make_graph_edges_params& ap,
    const edge_params_converter& edge_params_maker,
    vecmem::device_vector<uint2>& d_edge_nodes,
    vecmem::device_vector<short4>& d_edge_params,
    vecmem::device_vector<unsigned char>& d_reindexer, unsigned int cursor,
    const unsigned int cursor_end, unsigned int* scratch,
    const unsigned int scratch_stride) {
  if (begin >= end || hi < phis[begin] || lo > phis[end - 1u]) {
    return cursor;
  }
  for (unsigned int j = gbts_phi_lower_bound(phis, begin, end, lo); j < end;
       j++) {
    const float phi2 = phis[j];
    if (phi2 > hi) {
      break;
    }
    const float4 np2 = packs[j];
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
      gbts_emit_edge(cursor, j, node1, np1, np2, phi1, phi2, tau, dr, dz, curv,
                     ap, edge_params_maker, d_edge_nodes, d_edge_params,
                     d_reindexer);
    } else {
      // Remember the accepted outer node so the fill pass can skip the walk.
      if (scratch != nullptr && cursor < gbts_make_graph_edges_scratch_edges) {
        scratch[cursor * scratch_stride] = j;
      }
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
  const unsigned int blockIndex = thread_id.getBlockIdX();

  if constexpr (fill) {
    // Blocks without a work item to replay that are also beyond the dynamic
    // part of the grid have nothing to do: leave before any setup.
    if ((blockIndex >= gbts_make_graph_edges_max_blocks) &&
        (blockIndex >= *payload.nWork)) {
      return;
    }
  }

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
  vecmem::device_vector<unsigned int> d_edge_scratch(payload.edge_scratch);
  vecmem::device_vector<unsigned char> d_block_overflow(payload.block_overflow);
  vecmem::device_vector<unsigned int> d_overflow_items(payload.overflow_items);

  vecmem::device_vector<float> shared_phi(shared.phi);
  vecmem::device_vector<float4> shared_node_pack(shared.node_pack);
  vecmem::device_vector<unsigned int> shared_work_slot(shared.work_slot);

  const gbts_make_graph_edges_params& ap = payload.gbts_make_graph_edges_params;
  constexpr unsigned int chunk_size = gbts_consts::node_buffer_length;

  const unsigned int nWork = *payload.nWork;
  // Count pass: every block grabs the next work item until the list is
  // exhausted (dynamic scheduling; the assignment of items to blocks is
  // arbitrary, the output positions of every item are not).
  // Fill pass: block b first replays the edges recorded for work item b by
  // the count pass (no barrier, all items in flight at once), then the
  // blocks grab the items that must be re-walked (scratch overflow or no
  // scratch) from the overflow list.
  bool static_item = fill;
  // Only the first blocks take part in the dynamic loop of the fill pass
  // (the overflow list is short; every participating block costs one
  // atomic on the cursor).
  const bool grabs = !fill || (blockIndex < gbts_make_graph_edges_max_blocks);
  for (;;) {
    unsigned int work = 0u;
    bool replay = false;
    if (static_item) {
      static_item = false;
      work = blockIndex;
      if (!((work < nWork) && (work < payload.scratch_work_items) &&
            (d_block_overflow[work] == 0u))) {
        if (!grabs) {
          break;
        }
        continue;
      }
      replay = true;
    } else {
      if (!grabs) {
        break;
      }
      if (threadIndex == 0u) {
        const unsigned int idx =
            vecmem::device_atomic_ref<unsigned int>(*payload.work_cursor)
                .fetch_add(1u);
        unsigned int grabbed = nWork;  // sentinel: nothing left
        if constexpr (fill) {
          if (idx < *payload.n_overflow) {
            grabbed = d_overflow_items[idx];
          }
        } else {
          grabbed = idx;
        }
        shared_work_slot[0] = grabbed;
      }
      barrier.blockBarrier();
      work = shared_work_slot[0];
      if (work >= nWork) {
        break;
      }
    }
    // Accepted outer nodes of the thread (count pass), slot-major in global
    // memory: [(work * K + k) * blockSize + thread], so the fill pass reads
    // every slot row coalesced.
    constexpr unsigned int scratch_edges = gbts_make_graph_edges_scratch_edges;
    const bool has_scratch = work < payload.scratch_work_items;
    unsigned int* const global_scratch =
        has_scratch ? d_edge_scratch.data() +
                          (work * scratch_edges) * blockSize + threadIndex
                    : nullptr;
    unsigned int* scratch = fill ? nullptr : global_scratch;
    const unsigned int scratch_stride = blockSize;
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

    // --- Fill pass fast path: no thread of the block overflowed the scratch
    //     of the count pass, so the accepted outer nodes are simply replayed
    //     (block-uniform decision; the range search and slab walk below are
    //     skipped).
    if constexpr (fill) {
      if (replay) {
        if (active) {
          const unsigned int count =
              d_edge_counts[work * blockSize + threadIndex];
          for (unsigned int k = 0u; k < count && cursor < cursor_end; ++k) {
            const unsigned int node2 = global_scratch[k * blockSize];
            const float phi2 = d_node_phi[node2];
            const float4 np2 = d_node_params[node2];
            float tau, dr, dz, curv;
            // Recomputes the (identical) derived quantities of the edge.
            detail::gbts_edge_passes_cuts(np1, np2, phi1, phi2, deltaPhi, ap,
                                          tau, dr, dz, curv);
            detail::gbts_emit_edge(cursor, node2, node1, np1, np2, phi1, phi2,
                                   tau, dr, dz, curv, ap,
                                   payload.edge_params_maker, d_edge_nodes,
                                   d_edge_params, d_reindexer);
            cursor++;
          }
        }
        continue;
      }
    }

    // Outer-node index ranges that can pair with any node of the chunk: up to
    // two (wraparound), the lower-index one first. Computed redundantly by
    // every thread from block-uniform data.
    const detail::gbts_phi_window block_window = detail::gbts_make_phi_window(
        d_node_phi[chunk_begin] - window, d_node_phi[chunk_end - 1u] + window);
    unsigned int range_begin[2] = {begin2, end2};
    unsigned int range_end[2] = {end2, end2};
    if (!block_window.whole) {
      const bool has_b = block_window.lo_b <= block_window.hi_b;
      const float values[4] = {block_window.lo_a, block_window.hi_a,
                               block_window.lo_b, block_window.hi_b};
      unsigned int bounds[4] = {begin2, end2, end2, end2};
      // Cooperative search (barriers inside): block-uniform call.
      detail::gbts_block_find_bounds(
          barrier, threadIndex, blockSize, d_node_phi, begin2, end2, values,
          has_b ? 4u : 2u, shared_phi, shared_work_slot, bounds);
      range_begin[0] = bounds[0];
      range_end[0] = bounds[1];
      if (has_b) {
        range_begin[1] = bounds[2];
        range_end[1] = bounds[3];
      }
    }

    // --- Walk the outer-node ranges directly (global memory, L1 resident:
    //     the threads of a block read the same lines). No barriers from here
    //     to the end of the item.
    if (active) {
      for (unsigned int r = 0u; r < 2u; r++) {
        const unsigned int rb = range_begin[r];
        const unsigned int re = range_end[r];
        if (rb >= re) {
          continue;
        }
        if (my_window.whole) {
          cursor = detail::gbts_walk_range_interval<fill>(
              d_node_phi, d_node_params, rb, re, -traccc::device::PI_F - 1.0f,
              traccc::device::PI_F + 1.0f, np1, phi1, node1, deltaPhi, ap,
              payload.edge_params_maker, d_edge_nodes, d_edge_params,
              d_reindexer, cursor, cursor_end, scratch, scratch_stride);
        } else {
          cursor = detail::gbts_walk_range_interval<fill>(
              d_node_phi, d_node_params, rb, re, my_window.lo_a, my_window.hi_a,
              np1, phi1, node1, deltaPhi, ap, payload.edge_params_maker,
              d_edge_nodes, d_edge_params, d_reindexer, cursor, cursor_end,
              scratch, scratch_stride);
          if (my_window.lo_b <= my_window.hi_b) {
            cursor = detail::gbts_walk_range_interval<fill>(
                d_node_phi, d_node_params, rb, re, my_window.lo_b,
                my_window.hi_b, np1, phi1, node1, deltaPhi, ap,
                payload.edge_params_maker, d_edge_nodes, d_edge_params,
                d_reindexer, cursor, cursor_end, scratch, scratch_stride);
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
      // Items with a thread beyond the scratch capacity (block-wide maximum
      // count) or without scratch are listed for the fill pass to re-walk.
      bool needs_walk = true;
      if (has_scratch) {
        if (threadIndex == 0u) {
          shared_work_slot[9] = 0u;
        }
        barrier.blockBarrier();
        vecmem::device_atomic_ref<unsigned int,
                                  vecmem::device_address_space::local>(
            shared_work_slot[9])
            .fetch_max(count);
        barrier.blockBarrier();
        needs_walk = shared_work_slot[9] > scratch_edges;
        if (threadIndex == 0u) {
          d_block_overflow[work] = needs_walk ? 1u : 0u;
        }
        // The shared maximum is only rewritten after the next item's
        // barriers.
      }
      if (needs_walk && (threadIndex == 0u)) {
        const unsigned int slot =
            vecmem::device_atomic_ref<unsigned int>(*payload.n_overflow)
                .fetch_add(1u);
        d_overflow_items[slot] = work;
      }
    }
  }  // work item loop

  if constexpr (fill) {
    // Zero the kept flags beyond the edge count so that a prefix sum over the
    // whole capacity ends with the kept-edge count. The first
    // gbts_make_graph_edges_max_blocks blocks take part (they are the ones
    // that never leave early above).
    if (blockIndex < gbts_make_graph_edges_max_blocks) {
      const unsigned int total =
          d_num_outgoing_edges[d_num_outgoing_edges.size() - 1u];
      const unsigned int nEdges =
          (total < payload.nEdgesMax) ? total : payload.nEdgesMax;
      const unsigned int nBlocks = thread_id.getGridDimX();
      const unsigned int zeroing_blocks =
          (nBlocks < gbts_make_graph_edges_max_blocks)
              ? nBlocks
              : gbts_make_graph_edges_max_blocks;
      const unsigned int stride = blockSize * zeroing_blocks;
      for (unsigned int i = nEdges + blockIndex * blockSize + threadIndex;
           i < payload.nEdgesMax; i += stride) {
        d_reindexer[i] = 0u;
      }
    }
  }
}

}  // namespace traccc::device
