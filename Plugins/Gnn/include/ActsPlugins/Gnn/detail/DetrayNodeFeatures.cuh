// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

// The detray algebra calls constexpr host functions from device code. Without
// this flag nvcc only warns and the device results are wrong.
#if defined(__CUDACC__) && !defined(__CUDACC_RELAXED_CONSTEXPR__)
#error "DetrayNodeFeatures.cuh must be compiled with --expt-relaxed-constexpr"
#endif

#include <cstddef>

#include <detray/definitions/algebra.hpp>
#include <detray/geometry/identifier.hpp>
#include <detray/geometry/tracking_surface.hpp>

namespace ActsPlugins::detail {

/// Divisors of the (r, phi, z, eta) node features, the same as the feature
/// scales of the host feature creation, e.g. {1000, pi, 1000, 1}
struct NodeFeatureScales {
  float r = 1.f;
  float phi = 1.f;
  float z = 1.f;
  float eta = 1.f;
};

/// Writes the (r, phi, z, eta) node features of a global position to @p f.
///
/// The features are computed in double precision with the detray vector
/// getters and are then rounded to float and divided by the scale, which is
/// the same order of operations as the host feature creation with
/// Acts::VectorHelpers. This way, both give the same features for the same
/// positions.
template <typename point3_t>
__host__ __device__ inline void writeNodeFeatures(const point3_t &global,
                                                  float *f,
                                                  const NodeFeatureScales &s) {
  using algebra_t = detray::array<double>;
  const detray::dpoint3D<algebra_t> p{static_cast<double>(global[0]),
                                      static_cast<double>(global[1]),
                                      static_cast<double>(global[2])};
  f[0] = static_cast<float>(detray::vector::perp(p)) / s.r;
  f[1] = static_cast<float>(detray::vector::phi(p)) / s.phi;
  f[2] = static_cast<float>(global[2]) / s.z;
  f[3] = static_cast<float>(detray::vector::eta(p)) / s.eta;
}

/// Global position of a measurement with local position @p local on the
/// detray surface @p surface of the detector @p det
template <typename detector_t>
__host__ __device__ inline auto measurementGlobalPosition(
    const detector_t &det, detray::geometry::identifier surface,
    const detray::dpoint2D<typename detector_t::algebra_type> &local) {
  const detray::tracking_surface sf{det, surface};
  return sf.local_to_global(typename detector_t::geometry_context{}, local, {});
}

/// Writes the GNN node features of one space point: the (r, phi, z, eta) of
/// the space point, followed by the (r, phi, z, eta) of its first and second
/// measurement if @p nFeatures is 12. Single measurement (pixel) space points
/// repeat the space point features for both measurements, as the measurement
/// is at the space point. The global positions of the two measurements of
/// strip space points come from their surfaces in the detray detector.
template <typename detector_t, typename point3_t>
__host__ __device__ inline void writeSpacePointFeatures(
    const detector_t &det, const point3_t &spacePoint,
    unsigned int nMeasurements, const detray::geometry::identifier *surfaces,
    const detray::dpoint2D<typename detector_t::algebra_type> *locals,
    std::size_t nFeatures, float *f, const NodeFeatureScales &s) {
  writeNodeFeatures(spacePoint, f, s);
  for (std::size_t j = 4, m = 0; j + 4 <= nFeatures; j += 4, ++m) {
    if (nMeasurements < 2) {
      for (std::size_t k = 0; k < 4; ++k) {
        f[j + k] = f[k];
      }
    } else {
      writeNodeFeatures(measurementGlobalPosition(det, surfaces[m], locals[m]),
                        f + j, s);
    }
  }
}

}  // namespace ActsPlugins::detail
