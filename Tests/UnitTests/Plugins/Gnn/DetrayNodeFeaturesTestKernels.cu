// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"
#include "ActsPlugins/Gnn/detail/DetrayNodeFeatures.cuh"

#include "DetrayNodeFeaturesTestKernels.hpp"

namespace ActsTests::GnnDetray {

namespace {

using point3_t = detray::dpoint3D<algebra_t>;

__global__ void nodeFeaturesKernel(std::size_t n, const float *xyz, float *f,
                                   ActsPlugins::detail::NodeFeatureScales s) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const point3_t p{xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
  ActsPlugins::detail::writeNodeFeatures(p, f + 4 * i, s);
}

template <typename view_t>
__global__ void globalPositionKernel(
    view_t detView, std::size_t n, const detray::geometry::identifier *surfaces,
    const point2_t *locals, float *xyz) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const detray::detector_device_t<toy_detector_t> det(detView);
  const auto gp = ActsPlugins::detail::measurementGlobalPosition(
      det, surfaces[i], locals[i]);
  for (unsigned int k = 0; k < 3; ++k) {
    xyz[3 * i + k] = gp[k];
  }
}

template <typename T>
T *toDevice(const std::vector<T> &v) {
  T *d{};
  ACTS_CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  ACTS_CUDA_CHECK(
      cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <typename T>
std::vector<T> toHost(T *d, std::size_t n) {
  std::vector<T> v(n);
  ACTS_CUDA_CHECK(
      cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  ACTS_CUDA_CHECK(cudaFree(d));
  return v;
}

}  // namespace

std::vector<float> deviceNodeFeatures(const std::vector<float> &xyz,
                                      const Scales &scales) {
  const std::size_t n = xyz.size() / 3;
  float *dXyz = toDevice(xyz);
  float *dF{};
  ACTS_CUDA_CHECK(cudaMalloc(&dF, 4 * n * sizeof(float)));
  nodeFeaturesKernel<<<(n + 255) / 256, 256>>>(
      n, dXyz, dF, {scales.r, scales.phi, scales.z, scales.eta});
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_CUDA_CHECK(cudaFree(dXyz));
  return toHost(dF, 4 * n);
}

std::vector<float> deviceGlobalPositions(
    detray::detector_buffer_t<toy_detector_t> &detector,
    const std::vector<detray::geometry::identifier> &surfaces,
    const std::vector<point2_t> &locals) {
  const std::size_t n = surfaces.size();
  auto *dSurfaces = toDevice(surfaces);
  auto *dLocals = toDevice(locals);
  float *dXyz{};
  ACTS_CUDA_CHECK(cudaMalloc(&dXyz, 3 * n * sizeof(float)));
  globalPositionKernel<<<(n + 255) / 256, 256>>>(detray::get_data(detector), n,
                                                 dSurfaces, dLocals, dXyz);
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_CUDA_CHECK(cudaFree(dSurfaces));
  ACTS_CUDA_CHECK(cudaFree(dLocals));
  return toHost(dXyz, 3 * n);
}

}  // namespace ActsTests::GnnDetray
