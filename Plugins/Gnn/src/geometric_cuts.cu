#ifndef MMG_DOUBLET_MAX_THREADS_PER_BLOCK
#define MMG_DOUBLET_MAX_THREADS_PER_BLOCK 512
#endif

#ifndef MMG_DOUBLET_MIN_BLOCKS_PER_MP
#define MMG_DOUBLET_MIN_BLOCKS_PER_MP 3
#endif

#ifndef MMG_DOUBLET_LAUNCH_BOUNDS
#define MMG_DOUBLET_LAUNCH_BOUNDS \
  __launch_bounds__(MMG_DOUBLET_MAX_THREADS_PER_BLOCK, \
                    MMG_DOUBLET_MIN_BLOCKS_PER_MP)
#endif

// The kernels live in ActsPlugins so that they do not clash with the
// identically named ModuleMapGraph kernels pulled in by EdgeLayerConnector.cu
namespace ActsPlugins {

template <class T>
__device__ bool apply_geometric_cuts(
    int i, const T &z0, const T &phi_slope, const T &deta, const T &dphi,
    const T *z0_min, const T *phi_slope_min, const T *deta_min,
    const T *dphi_min, const T *z0_max, const T *phi_slope_max,
    const T *deta_max, const T *dphi_max) {
  bool accept = (z0_min[i] <= z0) * (dphi_min[i] <= dphi) *
                (phi_slope_min[i] <= phi_slope) * (deta_min[i] <= deta);
  accept *= (z0 <= z0_max[i]) * (dphi <= dphi_max[i]) *
            (phi_slope <= phi_slope_max[i]) * (deta <= deta_max[i]);
  return accept;
}

template <class T>
__global__ __launch_bounds__(512, 3) void hits_geometric_cuts_packed(
    float4 *geo, double2 *edge_slope, int *SPi, int *SPo, T *R, T *z, T *x,
    T *y, T *eta, T *phi, data_type2 pi, T epsilon, int nb_doublets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets) {
    return;
  }

  int SP1 = SPi[i];
  int SP2 = SPo[i];
  data_type2 R1 = R[SP1];
  T R2 = R[SP2];
  T z1 = z[SP1];
  T z2 = z[SP2];

  data_type2 dr = R2 - R1;
  data_type2 dz = z2 - z1;
  T deta_v = eta[SP2] - eta[SP1];
  data_type2 dphi_loc = phi[SP2] - phi[SP1];
  T pipi = 2 * pi;

  if (dphi_loc > pi) {
    dphi_loc -= pipi;
  } else if (dphi_loc < -pi) {
    dphi_loc += pipi;
  }
  T dphi_v = dphi_loc;

  T z0_v;
  T phi_slope_v;
  if (abs(dr) > epsilon) {
    phi_slope_v = dphi_loc / dr;
    z0_v = z1 - R1 * dz / dr;
  } else {
    z0_v = 0;
    phi_slope_v = 0;
  }
  geo[i] = make_float4(z0_v, phi_slope_v, deta_v, dphi_v);

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
  edge_slope[i] =
      make_double2(dx ? dy / dx : 0, dr_slope ? dz_slope / dr_slope : 0);
}

__global__ void count_source_hits_per_doublet(
    int *nb_src_hits_per_doublet, const int *modules1, const int *indices,
    int nb_doublets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets) {
    return;
  }

  int module1 = modules1[i];
  nb_src_hits_per_doublet[i] = indices[module1 + 1] - indices[module1];
}

__global__ void compact_active_doublets(int *active_doublets,
                                        const int *active_flags,
                                        const int *active_offsets,
                                        int nb_doublets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets || active_flags[i] == 0) {
    return;
  }

  active_doublets[active_offsets[i]] = i;
}

__global__ void build_src_work_to_doublet(
    int *src_work_to_doublet, const int *__restrict__ doublet_offsets,
    int nb_doublets) {
  const int doublet_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (doublet_idx >= nb_doublets) {
    return;
  }
  const int src_begin = doublet_offsets[doublet_idx];
  const int src_end = doublet_offsets[doublet_idx + 1];
  for (int i = src_begin; i < src_end; ++i) {
    src_work_to_doublet[i] = doublet_idx;
  }
}

__global__ void build_work_to_triplet(
    int *work_to_triplet, const int *__restrict__ triplet_offsets,
    int nb_triplets) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nb_triplets) {
    return;
  }
  const int begin = triplet_offsets[t];
  const int end = triplet_offsets[t + 1];
  for (int i = begin; i < end; ++i) {
    work_to_triplet[i] = t;
  }
}

template <class T>
__global__ MMG_DOUBLET_LAUNCH_BOUNDS void count_doublet_edges(
    int *nb_edges_per_src_hit, const int *__restrict__ src_work_to_doublet,
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

  const data_type2 pi_d = pi;
  int edges = 0;
  for (int l = indices[module2]; l < indices[module2 + 1]; l++) {
    T deta = eta[l] - eta_SP1;
    T dphi = phi[l] - phi_SP1;
    if (dphi > pi_d) {
      dphi -= 2 * pi_d;
    } else if (dphi < -pi_d) {
      dphi += 2 * pi_d;
    }

    if (!((deta_min_v <= deta) * (deta <= deta_max_v) *
          (dphi_min_v <= dphi) * (dphi <= dphi_max_v))) {
      continue;
    }

    data_type2 dr = R[l] - R_SP1;
    T z0;
    T phi_slope;
    if (abs(dr) > epsilon) {
      phi_slope = dphi / dr;
      z0 = z_SP1 - (data_type2)R_SP1 * (z[l] - z_SP1) / dr;
    } else {
      z0 = 0;
      phi_slope = 0;
    }

    if ((z0_min_v <= z0) * (z0 <= z0_max_v) *
        (phi_slope_min_v <= phi_slope) *
        (phi_slope <= phi_slope_max_v)) {
      edges++;
    }
  }

  nb_edges_per_src_hit[i] = edges;
}

__global__ void mark_active_src_work(
    int *flags, const int *__restrict__ edge_sum,
    int sum_nb_src_hits_per_doublet) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sum_nb_src_hits_per_doublet) {
    return;
  }
  flags[i] = (edge_sum[i + 1] > edge_sum[i]) ? 1 : 0;
}

template <class T>
__global__ MMG_DOUBLET_LAUNCH_BOUNDS void build_doublet_edges_active(
    int *reduced_M1_hits, int *reduced_M2_hits, int nb_active_src,
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

  const data_type2 pi_d = pi;
  int out = edge_sum[src_work_i];
  for (int l = indices[module2]; l < indices[module2 + 1]; l++) {
    T deta = eta[l] - eta_SP1;
    T dphi = phi[l] - phi_SP1;
    if (dphi > pi_d) {
      dphi -= 2 * pi_d;
    } else if (dphi < -pi_d) {
      dphi += 2 * pi_d;
    }

    if (!((deta_min_v <= deta) * (deta <= deta_max_v) *
          (dphi_min_v <= dphi) * (dphi <= dphi_max_v))) {
      continue;
    }

    data_type2 dr = R[l] - R_SP1;
    T z0;
    T phi_slope;
    if (abs(dr) > epsilon) {
      phi_slope = dphi / dr;
      z0 = z_SP1 - (data_type2)R_SP1 * (z[l] - z_SP1) / dr;
    } else {
      z0 = 0;
      phi_slope = 0;
    }

    if (!((z0_min_v <= z0) * (z0 <= z0_max_v) *
          (phi_slope_min_v <= phi_slope) *
          (phi_slope <= phi_slope_max_v))) {
      continue;
    }
    reduced_M1_hits[out] = k;
    reduced_M2_hits[out] = l;
    out++;
  }
}

__global__ void doublet_edge_sum(int *edge_sum, const int *doublet_offsets,
                                 const int *nb_edges_per_src_hit,
                                 int nb_doublets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets + 1) {
    return;
  }
  edge_sum[i] = nb_edges_per_src_hit[doublet_offsets[i]];
}

__global__ void count_triplet_hits(int *src_hits_per_triplet,
                                   const int *modules12_map,
                                   const int *modules23_map,
                                   const int *edge_indices, int nb_triplets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_triplets) {
    return;
  }

  int module12 = modules12_map[i];
  int module23 = modules23_map[i];

  int nb_hits_M12 = edge_indices[module12 + 1] - edge_indices[module12];
  int nb_hits_M23 = edge_indices[module23 + 1] - edge_indices[module23];
  if (nb_hits_M12 == 0 || nb_hits_M23 == 0) {
    src_hits_per_triplet[i] = 0;
    return;
  }

  int shift12 = edge_indices[module12];
  int last12 = shift12 + nb_hits_M12 - 1;
  src_hits_per_triplet[i] = last12 - shift12 + 1;
}

template <typename T>
__global__ void triplet_pair_cuts_fused_m23(
    bool *edge_tag, int nb_work_items, int nb_triplets,
    const int *triplet_offsets, const int *__restrict__ work_to_triplet,
    const int *modules12_map, const int *modules23_map,
    const float4 *__restrict__ geo, const double2 *__restrict__ edge_slope,
    T *MD12_z0_min, T *MD12_phi_slope_min, T *MD12_deta_min,
    T *MD12_dphi_min, T *MD12_z0_max, T *MD12_phi_slope_max,
    T *MD12_deta_max, T *MD12_dphi_max, T *MD23_z0_min,
    T *MD23_phi_slope_min, T *MD23_deta_min, T *MD23_dphi_min,
    T *MD23_z0_max, T *MD23_phi_slope_max, T *MD23_deta_max,
    T *MD23_dphi_max, T *diff_dydx_min, T *diff_dydx_max,
    T *diff_dzdr_min, T *diff_dzdr_max, int *M1_SP, int *M2_SP,
    int *edge_indices, const int *__restrict__ doublet_src_offsets,
    const int *__restrict__ doublet_module1, const int *__restrict__ hit_indices,
    const int *__restrict__ edge_sum_per_src_hit) {
  int work_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (work_i >= nb_work_items) {
    return;
  }

  int triplet_index = work_to_triplet[work_i];

  int module12 = modules12_map[triplet_index];
  int module23 = modules23_map[triplet_index];
  int nb_hits_M12 = edge_indices[module12 + 1] - edge_indices[module12];
  int nb_hits_M23 = edge_indices[module23 + 1] - edge_indices[module23];
  if (nb_hits_M12 == 0 || nb_hits_M23 == 0) {
    return;
  }

  int shift12 = edge_indices[module12];
  int k = shift12 + (work_i - triplet_offsets[triplet_index]);

  const float4 gk = geo[k];
  if (!apply_geometric_cuts(
          triplet_index, gk.x, gk.y, gk.z, gk.w, MD12_z0_min,
          MD12_phi_slope_min, MD12_deta_min, MD12_dphi_min, MD12_z0_max,
          MD12_phi_slope_max, MD12_deta_max, MD12_dphi_max)) {
    return;
  }

  int SP2 = M2_SP[k];
  const int m23_src_module = doublet_module1[module23];
  const int src_work_i =
      doublet_src_offsets[module23] + (SP2 - hit_indices[m23_src_module]);
  const int begin = edge_sum_per_src_hit[src_work_i];
  const int end = edge_sum_per_src_hit[src_work_i + 1];
  const double2 sk = edge_slope[k];

  for (int l = begin; l < end; ++l) {
    const float4 gl = geo[l];
    const T z0l = gl.x;
    const T psl = gl.y;
    const T del = gl.z;
    const T dpl = gl.w;
    bool accept = (MD23_z0_min[triplet_index] <= z0l) *
                  (z0l <= MD23_z0_max[triplet_index]) *
                  (MD23_phi_slope_min[triplet_index] <= psl) *
                  (psl <= MD23_phi_slope_max[triplet_index]) *
                  (MD23_deta_min[triplet_index] <= del) *
                  (del <= MD23_deta_max[triplet_index]) *
                  (MD23_dphi_min[triplet_index] <= dpl) *
                  (dpl <= MD23_dphi_max[triplet_index]);
    if (!accept) {
      continue;
    }

    const double2 sl = edge_slope[l];
    T diff_dydx = static_cast<T>(sk.x - sl.x);
    if (!((diff_dydx >= diff_dydx_min[triplet_index]) *
          (diff_dydx <= diff_dydx_max[triplet_index]))) {
      continue;
    }

    T diff_dzdr = static_cast<T>(sk.y - sl.y);
    if (!((diff_dzdr >= diff_dzdr_min[triplet_index]) *
          (diff_dzdr <= diff_dzdr_max[triplet_index]))) {
      continue;
    }

    edge_tag[l] = true;
    edge_tag[k] = true;
  }
}

}  // namespace ActsPlugins
