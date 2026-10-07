// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_runtime_api.h>

/// CUDA kernels for the module map graph construction in ModuleMapCuda.
namespace ActsPlugins::detail {

// Precision strategy
// ------------------
// The cut values come from the reference double-precision implementation, but
// consumer GPUs run FP64 at a small fraction of the FP32 rate. The kernels
// below therefore evaluate everything in float, in a way that reproduces the
// double-precision decisions exactly:
//  - Quantities whose float evaluation is exact, or equals the correctly
//    rounded double result (single float divisions of float inputs), are
//    compared directly. This requires IEEE division (no --use_fast_math).
//  - Other quantities are float approximations with a proven error bound.
//    A cut is decided from the approximation unless it lies within that bound
//    of a cut edge; only those rare cases are evaluated in double precision.

// Relative error margin for the float approximations (12 float ulps). The
// error bounds it has to cover are given next to each use.
constexpr float kFilterErrScale = 6.f * std::numeric_limits<float>::epsilon();

// Reference double-precision geometry of one doublet edge, used when a float
// approximation is too close to a cut edge. Must stay arithmetically identical
// to the reference implementation.
template <class T>
struct EdgeGeoReference {
  T z0;
  T phi_slope;
  double dydx;
  double dzdr;
};

template <class T>
__device__ EdgeGeoReference<T> edge_geo_reference(
    int SP1, int SP2, const T *__restrict__ R, const T *__restrict__ z,
    const T *__restrict__ x, const T *__restrict__ y, const T *__restrict__ phi,
    double pi, T epsilon) {
  double R1 = R[SP1];
  T R2 = R[SP2];
  T z1 = z[SP1];
  T z2 = z[SP2];

  double dr = R2 - R1;
  double dz = z2 - z1;
  double dphi_loc = phi[SP2] - phi[SP1];
  T pipi = 2 * pi;

  if (dphi_loc > pi) {
    dphi_loc -= pipi;
  } else if (dphi_loc < -pi) {
    dphi_loc += pipi;
  }

  EdgeGeoReference<T> ref{};
  if (abs(dr) > epsilon) {
    ref.phi_slope = dphi_loc / dr;
    ref.z0 = z1 - R1 * dz / dr;
  } else {
    ref.z0 = 0;
    ref.phi_slope = 0;
  }

  double dx = x[SP2] - x[SP1];
  const double dy = y[SP2] - y[SP1];
  double dr_slope = R[SP2] - R[SP1];
  const double dz_slope = z[SP2] - z[SP1];
  if (abs(dx) < epsilon) {
    dx = 0;
  }
  if (abs(dr_slope) < epsilon) {
    dr_slope = 0;
  }
  ref.dydx = dx ? dy / dx : 0;
  ref.dzdr = dr_slope ? dz_slope / dr_slope : 0;
  return ref;
}

// Float geometry of each doublet edge, used by the triplet cuts:
//  - geo:   (z0, phi_slope, deta, dphi); deta and dphi are exact, z0 and
//           phi_slope are float approximations of the reference.
//  - slope: (dydx, dzdr, z0 error margin, unused); dydx and dzdr are the
//           correctly rounded float values of the reference slopes.
template <class T>
__global__ __launch_bounds__(512, 3) void hits_geometric_cuts_packed(
    float4 *geo, float4 *slope, const int *__restrict__ SPi,
    const int *__restrict__ SPo, const T *__restrict__ R,
    const T *__restrict__ z, const T *__restrict__ x, const T *__restrict__ y,
    const T *__restrict__ eta, const T *__restrict__ phi, T pi, T epsilon,
    int nb_doublets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets) {
    return;
  }

  const int SP1 = SPi[i];
  const int SP2 = SPo[i];
  const T R1 = R[SP1];
  const T R2 = R[SP2];
  const T z1 = z[SP1];
  const T z2 = z[SP2];

  const T deta = eta[SP2] - eta[SP1];
  T dphi = phi[SP2] - phi[SP1];
  if (dphi > pi) {
    dphi -= 2 * pi;
  } else if (dphi < -pi) {
    dphi += 2 * pi;
  }

  const T dr = R2 - R1;
  const T dz = z2 - z1;

  T z0 = 0;
  T phi_slope = 0;
  T z0_err = 0;
  // The reference tests the double difference R2 - R1 against epsilon
  if (abs(static_cast<double>(R2) - static_cast<double>(R1)) > epsilon) {
    // With u = 2^-24: phi_slope is within ~3u|phi_slope| of the reference,
    // z0 within ~5u|q| + 2u|z1|; both are covered by kFilterErrScale
    phi_slope = dphi / dr;
    const T q = R1 * dz / dr;
    z0 = z1 - q;
    z0_err = kFilterErrScale * (fabs(z1) + fabs(q)) + T{1e-30};
  }
  geo[i] = make_float4(z0, phi_slope, deta, dphi);

  // dx, dy, dr and dz are float differences in the reference as well
  const T dx = x[SP2] - x[SP1];
  const T dy = y[SP2] - y[SP1];
  const T dydx = (fabs(dx) < epsilon) ? T{0} : dy / dx;
  const T dzdr = (fabs(dr) < epsilon) ? T{0} : dz / dr;
  slope[i] = make_float4(dydx, dzdr, z0_err, 0.f);
}

static __global__ void count_source_hits_per_doublet(
    int *nb_src_hits_per_doublet, const int *modules1, const int *indices,
    int nb_doublets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets) {
    return;
  }

  int module1 = modules1[i];
  nb_src_hits_per_doublet[i] = indices[module1 + 1] - indices[module1];
}

static __global__ void compact_active_doublets(int *active_doublets,
                                               const int *active_flags,
                                               const int *active_offsets,
                                               int nb_doublets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets || active_flags[i] == 0) {
    return;
  }

  active_doublets[active_offsets[i]] = i;
}

// Fill work_to_item[offsets[i]..offsets[i+1]) with i. A group of
// kThreadsPerItem threads cooperates on each item, so the writes are coalesced
// and long ranges do not serialize on a single thread.
template <int kThreadsPerItem>
__global__ void build_work_to_item(int *work_to_item,
                                   const int *__restrict__ offsets,
                                   int nb_items) {
  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t item = tid / kThreadsPerItem;
  const int lane = tid % kThreadsPerItem;
  if (item >= static_cast<std::size_t>(nb_items)) {
    return;
  }
  const int begin = offsets[item];
  const int end = offsets[item + 1];
  for (int i = begin + lane; i < end; i += kThreadsPerItem) {
    work_to_item[i] = static_cast<int>(item);
  }
}

// Doublet pair selection shared by count_doublet_edges and
// build_doublet_edges_active. The phi wrap is exact in float (Sterbenz lemma),
// dr is a float difference in the reference too, and phi_slope is a single
// float division; only z0 needs the filtered evaluation.
template <class T>
__device__ __forceinline__ bool doublet_pair_passes(
    T R1, T z1, T eta1, T phi1, const T *__restrict__ R,
    const T *__restrict__ z, const T *__restrict__ eta,
    const T *__restrict__ phi, int l, T z0_min, T z0_max, T deta_min,
    T deta_max, T phi_slope_min, T phi_slope_max, T dphi_min, T dphi_max, T pi,
    T epsilon) {
  const T deta = eta[l] - eta1;
  if (!((deta_min <= deta) && (deta <= deta_max))) {
    return false;
  }

  T dphi = phi[l] - phi1;
  if (dphi > pi) {
    dphi -= 2 * pi;
  } else if (dphi < -pi) {
    dphi += 2 * pi;
  }
  if (!((dphi_min <= dphi) && (dphi <= dphi_max))) {
    return false;
  }

  // R and z are only loaded for pairs that pass the eta/phi windows
  const T dr = R[l] - R1;
  if (!(fabs(dr) > epsilon)) {
    // Degenerate dr: z0 and phi_slope are defined as 0
    return (z0_min <= T{0}) && (T{0} <= z0_max) && (phi_slope_min <= T{0}) &&
           (T{0} <= phi_slope_max);
  }

  const T phi_slope = dphi / dr;
  if (!((phi_slope_min <= phi_slope) && (phi_slope <= phi_slope_max))) {
    return false;
  }

  // With u = 2^-24, the float z0 differs from the rounded double reference by
  // less than 6u(|z1| + |q|); the margin is 8u(|z1| + |q|). Non-finite values
  // fail the margin test and take the double-precision path.
  const T dz = z[l] - z1;
  const T q = R1 * dz / dr;
  const T z0_fast = z1 - q;
  const T margin =
      T{4} * std::numeric_limits<T>::epsilon() * (fabs(z1) + fabs(q)) +
      T{1e-30};
  if (fabs(z0_fast - z0_min) > margin && fabs(z0_fast - z0_max) > margin) {
    return (z0_min <= z0_fast) && (z0_fast <= z0_max);
  }

  const T z0 = z1 - static_cast<double>(R1) * dz / static_cast<double>(dr);
  return (z0_min <= z0) && (z0 <= z0_max);
}

// The count pass records the accepted hit pairs as a bit mask per source hit,
// so the build pass can write the edges without evaluating the cuts again.
// Target modules with more hits than fit in the mask are re-evaluated instead.
constexpr int kMaxMaskedTargetHits = 64;

template <class T>
__global__ __launch_bounds__(512, 3) void count_doublet_edges(
    int *nb_edges_per_src_hit, std::uint64_t *pair_masks,
    const int *__restrict__ src_work_to_doublet,
    const int *__restrict__ doublet_offsets, const int *__restrict__ modules1,
    const int *__restrict__ modules2, const T *__restrict__ R,
    const T *__restrict__ z, const T *__restrict__ eta,
    const T *__restrict__ phi, const T *__restrict__ z0_min,
    const T *__restrict__ deta_min, const T *__restrict__ phi_slope_min,
    const T *__restrict__ dphi_min, const T *__restrict__ z0_max,
    const T *__restrict__ deta_max, const T *__restrict__ phi_slope_max,
    const T *__restrict__ dphi_max, const int *__restrict__ indices, T pi,
    T epsilon, int sum_nb_src_hits_per_doublet) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sum_nb_src_hits_per_doublet) {
    return;
  }

  const int doublet_idx = src_work_to_doublet[i];
  const int module1 = modules1[doublet_idx];
  const int module2 = modules2[doublet_idx];
  const int src_begin = doublet_offsets[doublet_idx];
  const int k = indices[module1] + (i - src_begin);
  const T phi_SP1 = phi[k];
  const T eta_SP1 = eta[k];
  const T R_SP1 = R[k];
  const T z_SP1 = z[k];
  const T z0_min_v = z0_min[doublet_idx];
  const T deta_min_v = deta_min[doublet_idx];
  const T phi_slope_min_v = phi_slope_min[doublet_idx];
  const T dphi_min_v = dphi_min[doublet_idx];
  const T z0_max_v = z0_max[doublet_idx];
  const T deta_max_v = deta_max[doublet_idx];
  const T phi_slope_max_v = phi_slope_max[doublet_idx];
  const T dphi_max_v = dphi_max[doublet_idx];

  const int begin2 = indices[module2];
  const int end2 = indices[module2 + 1];
  const bool use_mask = (end2 - begin2) <= kMaxMaskedTargetHits;

  int edges = 0;
  std::uint64_t mask = 0;
  for (int l = begin2; l < end2; l++) {
    const bool pass = doublet_pair_passes<T>(
        R_SP1, z_SP1, eta_SP1, phi_SP1, R, z, eta, phi, l, z0_min_v, z0_max_v,
        deta_min_v, deta_max_v, phi_slope_min_v, phi_slope_max_v, dphi_min_v,
        dphi_max_v, pi, epsilon);
    edges += pass;
    if (use_mask && pass) {
      mask |= std::uint64_t{1} << (l - begin2);
    }
  }

  nb_edges_per_src_hit[i] = edges;
  pair_masks[i] = mask;
}

static __global__ void mark_active_src_work(int *flags,
                                            const int *__restrict__ edge_sum,
                                            int sum_nb_src_hits_per_doublet) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sum_nb_src_hits_per_doublet) {
    return;
  }
  flags[i] = (edge_sum[i + 1] > edge_sum[i]) ? 1 : 0;
}

template <class T>
__global__ __launch_bounds__(512, 3) void build_doublet_edges_active(
    int *reduced_M1_hits, int *reduced_M2_hits, int nb_active_src,
    const std::uint64_t *__restrict__ pair_masks,
    const int *__restrict__ active_src_work,
    const int *__restrict__ src_work_to_doublet,
    const int *__restrict__ doublet_offsets, const int *__restrict__ indices,
    const int *__restrict__ edge_sum, const int *__restrict__ modules1,
    const int *__restrict__ modules2, const T *__restrict__ R,
    const T *__restrict__ z, const T *__restrict__ eta,
    const T *__restrict__ phi, const T *__restrict__ z0_min,
    const T *__restrict__ deta_min, const T *__restrict__ phi_slope_min,
    const T *__restrict__ dphi_min, const T *__restrict__ z0_max,
    const T *__restrict__ deta_max, const T *__restrict__ phi_slope_max,
    const T *__restrict__ dphi_max, T pi, T epsilon) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= nb_active_src) {
    return;
  }

  const int src_work_i = active_src_work[a];
  const int doublet_idx = src_work_to_doublet[src_work_i];
  const int module1 = modules1[doublet_idx];
  const int module2 = modules2[doublet_idx];
  const int k = indices[module1] + (src_work_i - doublet_offsets[doublet_idx]);
  const int begin2 = indices[module2];
  const int end2 = indices[module2 + 1];
  int out = edge_sum[src_work_i];

  if ((end2 - begin2) <= kMaxMaskedTargetHits) {
    // Emit the pairs recorded by the count pass, lowest bit first, which is
    // the same order as the loop below
    std::uint64_t mask = pair_masks[src_work_i];
    while (mask != 0) {
      const int bit = __ffsll(static_cast<long long>(mask)) - 1;
      reduced_M1_hits[out] = k;
      reduced_M2_hits[out] = begin2 + bit;
      out++;
      mask &= mask - 1;
    }
    return;
  }

  const T phi_SP1 = phi[k];
  const T eta_SP1 = eta[k];
  const T R_SP1 = R[k];
  const T z_SP1 = z[k];
  const T z0_min_v = z0_min[doublet_idx];
  const T deta_min_v = deta_min[doublet_idx];
  const T phi_slope_min_v = phi_slope_min[doublet_idx];
  const T dphi_min_v = dphi_min[doublet_idx];
  const T z0_max_v = z0_max[doublet_idx];
  const T deta_max_v = deta_max[doublet_idx];
  const T phi_slope_max_v = phi_slope_max[doublet_idx];
  const T dphi_max_v = dphi_max[doublet_idx];

  for (int l = begin2; l < end2; l++) {
    if (!doublet_pair_passes<T>(R_SP1, z_SP1, eta_SP1, phi_SP1, R, z, eta, phi,
                                l, z0_min_v, z0_max_v, deta_min_v, deta_max_v,
                                phi_slope_min_v, phi_slope_max_v, dphi_min_v,
                                dphi_max_v, pi, epsilon)) {
      continue;
    }
    reduced_M1_hits[out] = k;
    reduced_M2_hits[out] = l;
    out++;
  }
}

static __global__ void doublet_edge_sum(int *edge_sum,
                                        const int *doublet_offsets,
                                        const int *nb_edges_per_src_hit,
                                        int nb_doublets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets + 1) {
    return;
  }
  edge_sum[i] = nb_edges_per_src_hit[doublet_offsets[i]];
}

static __global__ void count_triplet_hits(int *src_hits_per_triplet,
                                          const int *modules12_map,
                                          const int *modules23_map,
                                          const int *edge_indices,
                                          int nb_triplets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_triplets) {
    return;
  }

  // One work item per M1->M2 edge, if the M2->M3 doublet has edges at all
  const int module12 = modules12_map[i];
  const int module23 = modules23_map[i];
  const int nb_edges_M12 = edge_indices[module12 + 1] - edge_indices[module12];
  const int nb_edges_M23 = edge_indices[module23 + 1] - edge_indices[module23];
  src_hits_per_triplet[i] = (nb_edges_M23 == 0) ? 0 : nb_edges_M12;
}

// Decide lo <= v <= hi for a float approximation v whose reference value is
// within err of it. Returns 1 (pass) or 0 (fail) when v is farther than err
// from both edges, so the reference value gives the same decision, and -1
// when the decision needs the reference value. NaN gives -1.
template <class T>
__device__ __forceinline__ int in_range_filtered(T v, T err, T lo, T hi) {
  if (fabs(v - lo) > err && fabs(v - hi) > err) {
    return (lo <= v) && (v <= hi);
  }
  return -1;
}

template <class T>
__device__ __forceinline__ bool in_range(T v, T lo, T hi) {
  return (lo <= v) && (v <= hi);
}

// A pair of doublet edges (k: M1->M2, l: M2->M3) of a module triplet whose
// cuts could not be decided from the float geometry
struct TripletFallbackPair {
  int triplet_index;
  int k;
  int l;
};

// Capacity of the fallback queue (typically ~100 pairs per event are queued).
// On overflow the host reruns the triplet cuts with inline fallback.
constexpr int kTripletFallbackCapacity = 1 << 20;

// Triplet cuts on pairs of doublet edges (k: M1->M2, l: M2->M3), evaluated
// on the float geometry from hits_geometric_cuts_packed. Pairs that cannot be
// decided from it are resolved with edge_geo_reference:
//  - kDeferFallback = true: queued for triplet_pair_cuts_fallback, which keeps
//    double-precision code (and its register pressure) out of this kernel.
//  - kDeferFallback = false: resolved inline; used if the queue overflows.
// The slope differences are within ~3u(|slope_k| + |slope_l|) of the
// reference, u = 2^-24.
template <typename T, bool kDeferFallback>
__global__ void triplet_pair_cuts_fused_m23(
    bool *edge_tag, int nb_work_items, const int *__restrict__ triplet_offsets,
    const int *__restrict__ work_to_triplet,
    const int *__restrict__ modules12_map,
    const int *__restrict__ modules23_map, const float4 *__restrict__ geo,
    const float4 *__restrict__ edge_slope, const T *__restrict__ MD12_z0_min,
    const T *__restrict__ MD12_phi_slope_min,
    const T *__restrict__ MD12_deta_min, const T *__restrict__ MD12_dphi_min,
    const T *__restrict__ MD12_z0_max, const T *__restrict__ MD12_phi_slope_max,
    const T *__restrict__ MD12_deta_max, const T *__restrict__ MD12_dphi_max,
    const T *__restrict__ MD23_z0_min, const T *__restrict__ MD23_phi_slope_min,
    const T *__restrict__ MD23_deta_min, const T *__restrict__ MD23_dphi_min,
    const T *__restrict__ MD23_z0_max, const T *__restrict__ MD23_phi_slope_max,
    const T *__restrict__ MD23_deta_max, const T *__restrict__ MD23_dphi_max,
    const T *__restrict__ diff_dydx_min, const T *__restrict__ diff_dydx_max,
    const T *__restrict__ diff_dzdr_min, const T *__restrict__ diff_dzdr_max,
    const int *__restrict__ M1_SP, const int *__restrict__ M2_SP,
    const int *__restrict__ edge_indices,
    const int *__restrict__ doublet_src_offsets,
    const int *__restrict__ doublet_module1,
    const int *__restrict__ hit_indices,
    const int *__restrict__ edge_sum_per_src_hit, const T *__restrict__ R,
    const T *__restrict__ z, const T *__restrict__ x, const T *__restrict__ y,
    const T *__restrict__ phi, T pi, T epsilon,
    TripletFallbackPair *fallback_pairs, int *fallback_count,
    int fallback_capacity) {
  int work_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (work_i >= nb_work_items) {
    return;
  }

  // Work items only exist for triplets where both doublets have edges (see
  // count_triplet_hits), so no emptiness check is needed here
  const int triplet_index = work_to_triplet[work_i];
  const int module12 = modules12_map[triplet_index];
  const int module23 = modules23_map[triplet_index];
  const int k =
      edge_indices[module12] + (work_i - triplet_offsets[triplet_index]);

  // ---- cuts on edge k (module doublet M1-M2) ----
  const float4 gk = geo[k];
  if (!(in_range<T>(gk.z, MD12_deta_min[triplet_index],
                    MD12_deta_max[triplet_index]) &&
        in_range<T>(gk.w, MD12_dphi_min[triplet_index],
                    MD12_dphi_max[triplet_index]))) {
    return;
  }

  const float4 sk = edge_slope[k];
  const T z0_min12 = MD12_z0_min[triplet_index];
  const T z0_max12 = MD12_z0_max[triplet_index];
  const T ps_min12 = MD12_phi_slope_min[triplet_index];
  const T ps_max12 = MD12_phi_slope_max[triplet_index];
  const int z0_ok_k = in_range_filtered<T>(gk.x, sk.z, z0_min12, z0_max12);
  const int ps_ok_k = in_range_filtered<T>(gk.y, kFilterErrScale * fabs(gk.y),
                                           ps_min12, ps_max12);
  if (z0_ok_k == 0 || ps_ok_k == 0) {
    return;
  }
  bool k_undecided = (z0_ok_k < 0 || ps_ok_k < 0);

  // Inline mode only: the reference geometry of edge k, computed at most once
  EdgeGeoReference<T> ref_k{};
  bool have_ref_k = false;
  if constexpr (!kDeferFallback) {
    if (k_undecided) {
      ref_k = edge_geo_reference<T>(M1_SP[k], M2_SP[k], R, z, x, y, phi, pi,
                                    epsilon);
      have_ref_k = true;
      if (!(in_range<T>(ref_k.z0, z0_min12, z0_max12) &&
            in_range<T>(ref_k.phi_slope, ps_min12, ps_max12))) {
        return;
      }
      k_undecided = false;
    }
  }

  const int SP2 = M2_SP[k];
  const int m23_src_module = doublet_module1[module23];
  const int src_work_i =
      doublet_src_offsets[module23] + (SP2 - hit_indices[m23_src_module]);
  const int begin = edge_sum_per_src_hit[src_work_i];
  const int end = edge_sum_per_src_hit[src_work_i + 1];
  if (begin == end) {
    return;
  }

  // Load the per-triplet cuts once instead of on every loop iteration
  const T z0_min = MD23_z0_min[triplet_index];
  const T z0_max = MD23_z0_max[triplet_index];
  const T ps_min = MD23_phi_slope_min[triplet_index];
  const T ps_max = MD23_phi_slope_max[triplet_index];
  const T deta_min = MD23_deta_min[triplet_index];
  const T deta_max = MD23_deta_max[triplet_index];
  const T dphi_min = MD23_dphi_min[triplet_index];
  const T dphi_max = MD23_dphi_max[triplet_index];
  const T dydx_min = diff_dydx_min[triplet_index];
  const T dydx_max = diff_dydx_max[triplet_index];
  const T dzdr_min = diff_dzdr_min[triplet_index];
  const T dzdr_max = diff_dzdr_max[triplet_index];

  bool any_accepted = false;

  for (int l = begin; l < end; ++l) {
    // ---- cuts on edge l (module doublet M2-M3) ----
    const float4 gl = geo[l];
    if (!(in_range<T>(gl.z, deta_min, deta_max) &&
          in_range<T>(gl.w, dphi_min, dphi_max))) {
      continue;
    }

    const float4 sl = edge_slope[l];
    const int z0_ok = in_range_filtered<T>(gl.x, sl.z, z0_min, z0_max);
    if (z0_ok == 0) {
      continue;
    }
    const int ps_ok = in_range_filtered<T>(gl.y, kFilterErrScale * fabs(gl.y),
                                           ps_min, ps_max);
    if (ps_ok == 0) {
      continue;
    }

    // ---- cuts on the slope differences between k and l ----
    const int dydx_ok = in_range_filtered<T>(
        sk.x - sl.x, kFilterErrScale * (fabs(sk.x) + fabs(sl.x)), dydx_min,
        dydx_max);
    if (dydx_ok == 0) {
      continue;
    }
    const int dzdr_ok = in_range_filtered<T>(
        sk.y - sl.y, kFilterErrScale * (fabs(sk.y) + fabs(sl.y)), dzdr_min,
        dzdr_max);
    if (dzdr_ok == 0) {
      continue;
    }

    // ---- undecided cuts are resolved with the reference geometry ----
    if (k_undecided || z0_ok < 0 || ps_ok < 0 || dydx_ok < 0 || dzdr_ok < 0) {
      if constexpr (kDeferFallback) {
        const int slot = atomicAdd(fallback_count, 1);
        if (slot < fallback_capacity) {
          fallback_pairs[slot] = TripletFallbackPair{triplet_index, k, l};
        }
        continue;
      } else {
        const EdgeGeoReference<T> ref_l = edge_geo_reference<T>(
            M1_SP[l], M2_SP[l], R, z, x, y, phi, pi, epsilon);
        if (z0_ok < 0 && !in_range<T>(ref_l.z0, z0_min, z0_max)) {
          continue;
        }
        if (ps_ok < 0 && !in_range<T>(ref_l.phi_slope, ps_min, ps_max)) {
          continue;
        }
        if (dydx_ok < 0 || dzdr_ok < 0) {
          if (!have_ref_k) {
            ref_k = edge_geo_reference<T>(M1_SP[k], M2_SP[k], R, z, x, y, phi,
                                          pi, epsilon);
            have_ref_k = true;
          }
          if (dydx_ok < 0 &&
              !in_range<T>(static_cast<T>(ref_k.dydx - ref_l.dydx), dydx_min,
                           dydx_max)) {
            continue;
          }
          if (dzdr_ok < 0 &&
              !in_range<T>(static_cast<T>(ref_k.dzdr - ref_l.dzdr), dzdr_min,
                           dzdr_max)) {
            continue;
          }
        }
      }
    }

    edge_tag[l] = true;
    any_accepted = true;
  }

  if (any_accepted) {
    edge_tag[k] = true;
  }
}

// Resolves the pairs queued by triplet_pair_cuts_fused_m23<T, true> with the
// reference geometry. Grid-stride loop, the queue length is only known on the
// device.
template <typename T>
__global__ void triplet_pair_cuts_fallback(
    bool *edge_tag, const TripletFallbackPair *__restrict__ fallback_pairs,
    const int *__restrict__ fallback_count, int fallback_capacity,
    const float4 *__restrict__ geo, const T *__restrict__ MD12_z0_min,
    const T *__restrict__ MD12_phi_slope_min,
    const T *__restrict__ MD12_deta_min, const T *__restrict__ MD12_dphi_min,
    const T *__restrict__ MD12_z0_max, const T *__restrict__ MD12_phi_slope_max,
    const T *__restrict__ MD12_deta_max, const T *__restrict__ MD12_dphi_max,
    const T *__restrict__ MD23_z0_min, const T *__restrict__ MD23_phi_slope_min,
    const T *__restrict__ MD23_deta_min, const T *__restrict__ MD23_dphi_min,
    const T *__restrict__ MD23_z0_max, const T *__restrict__ MD23_phi_slope_max,
    const T *__restrict__ MD23_deta_max, const T *__restrict__ MD23_dphi_max,
    const T *__restrict__ diff_dydx_min, const T *__restrict__ diff_dydx_max,
    const T *__restrict__ diff_dzdr_min, const T *__restrict__ diff_dzdr_max,
    const int *__restrict__ M1_SP, const int *__restrict__ M2_SP,
    const T *__restrict__ R, const T *__restrict__ z, const T *__restrict__ x,
    const T *__restrict__ y, const T *__restrict__ phi, T pi, T epsilon) {
  const int n = min(*fallback_count, fallback_capacity);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const TripletFallbackPair p = fallback_pairs[i];
    const int t = p.triplet_index;
    const float4 gk = geo[p.k];
    const float4 gl = geo[p.l];
    const EdgeGeoReference<T> ref_k = edge_geo_reference<T>(
        M1_SP[p.k], M2_SP[p.k], R, z, x, y, phi, pi, epsilon);
    const EdgeGeoReference<T> ref_l = edge_geo_reference<T>(
        M1_SP[p.l], M2_SP[p.l], R, z, x, y, phi, pi, epsilon);

    const bool pass = in_range<T>(gk.z, MD12_deta_min[t], MD12_deta_max[t]) &&
                      in_range<T>(gk.w, MD12_dphi_min[t], MD12_dphi_max[t]) &&
                      in_range<T>(ref_k.z0, MD12_z0_min[t], MD12_z0_max[t]) &&
                      in_range<T>(ref_k.phi_slope, MD12_phi_slope_min[t],
                                  MD12_phi_slope_max[t]) &&
                      in_range<T>(gl.z, MD23_deta_min[t], MD23_deta_max[t]) &&
                      in_range<T>(gl.w, MD23_dphi_min[t], MD23_dphi_max[t]) &&
                      in_range<T>(ref_l.z0, MD23_z0_min[t], MD23_z0_max[t]) &&
                      in_range<T>(ref_l.phi_slope, MD23_phi_slope_min[t],
                                  MD23_phi_slope_max[t]) &&
                      in_range<T>(static_cast<T>(ref_k.dydx - ref_l.dydx),
                                  diff_dydx_min[t], diff_dydx_max[t]) &&
                      in_range<T>(static_cast<T>(ref_k.dzdr - ref_l.dzdr),
                                  diff_dzdr_min[t], diff_dzdr_max[t]);
    if (pass) {
      edge_tag[p.k] = true;
      edge_tag[p.l] = true;
    }
  }
}

}  // namespace ActsPlugins::detail
