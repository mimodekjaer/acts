// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

#include <detray/core/detector.hpp>
#include <detray/definitions/algebra.hpp>
#include <detray/detectors/toy_metadata.hpp>
#include <detray/geometry/identifier.hpp>

namespace ActsTests::GnnDetray {

using algebra_t = detray::array<float>;
using point2_t = detray::dpoint2D<algebra_t>;
using toy_detector_t = detray::host::detector<detray::toy_metadata<algebra_t>>;

/// Divisors of the (r, phi, z, eta) features
struct Scales {
  float r, phi, z, eta;
};

/// (r, phi, z, eta) node features of the points [x0, y0, z0, x1, ...],
/// computed on the device
std::vector<float> deviceNodeFeatures(const std::vector<float> &xyz,
                                      const Scales &scales);

/// Global positions of measurements on surfaces of the toy detector,
/// computed on the device
std::vector<float> deviceGlobalPositions(
    detray::detector_buffer_t<toy_detector_t> &detector,
    const std::vector<detray::geometry::identifier> &surfaces,
    const std::vector<point2_t> &locals);

}  // namespace ActsTests::GnnDetray
