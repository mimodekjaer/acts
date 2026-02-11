// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/ModuleMapCpu.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

using namespace Acts;
using namespace ActsPlugins;

namespace ActsTests {

BOOST_AUTO_TEST_SUITE(GnnSuite)

BOOST_AUTO_TEST_CASE(ModuleMapCpu_invalid_path_throws) {
  ModuleMapCpu::Config cfg;
  cfg.moduleMapPath = "/nonexistent/path/module_map.root";

  auto logger = getDefaultLogger("TestLogger", Logging::ERROR);

  BOOST_CHECK_THROW(ModuleMapCpu(cfg, std::move(logger)), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(ModuleMapCpu_edge_features_basic) {
  constexpr std::size_t nFeatures = 4;
  //                  R     Phi   Z     Eta
  float src[4] = {100.f, 0.1f, 200.f, 1.0f};
  float tgt[4] = {200.f, 0.2f, 400.f, 1.5f};

  auto ef = detail::computeEdgeFeatures(src, tgt, nFeatures);

  // dr = 200 - 100 = 100
  BOOST_CHECK_CLOSE(ef[0], 100.f, 1e-4f);

  // dphi = resetAngle(pi * (0.2 - 0.1)) / pi = (pi * 0.1) / pi = 0.1
  BOOST_CHECK_CLOSE(ef[1], 0.1f, 1e-4f);

  // dz = 400 - 200 = 200
  BOOST_CHECK_CLOSE(ef[2], 200.f, 1e-4f);

  // deta = 1.5 - 1.0 = 0.5
  BOOST_CHECK_CLOSE(ef[3], 0.5f, 1e-4f);

  // phislope = 0.1 / 100 = 0.001
  BOOST_CHECK_CLOSE(ef[4], 0.001f, 1e-4f);

  // rphislope = 150 * 0.001 = 0.15
  BOOST_CHECK_CLOSE(ef[5], 0.15f, 1e-4f);
}

BOOST_AUTO_TEST_CASE(ModuleMapCpu_edge_features_zero_dr) {
  constexpr std::size_t nFeatures = 4;
  float src[4] = {100.f, 0.1f, 200.f, 1.0f};
  float tgt[4] = {100.f, 0.3f, 300.f, 1.2f};

  auto ef = detail::computeEdgeFeatures(src, tgt, nFeatures);

  // dr = 0
  BOOST_CHECK_SMALL(ef[0], 1e-6f);
  // phislope and rphislope should be 0 when dr = 0
  BOOST_CHECK_SMALL(ef[4], 1e-6f);
  BOOST_CHECK_SMALL(ef[5], 1e-6f);
}

BOOST_AUTO_TEST_CASE(ModuleMapCpu_edge_features_phi_wrapping) {
  constexpr std::size_t nFeatures = 4;
  // Phi values that when multiplied by pi and differenced, exceed pi
  // phi_src = 0.9, phi_tgt = -0.9
  // rawDphi = pi * (-0.9 - 0.9) = pi * (-1.8) = -5.654...
  // After wrapping: -5.654 + 2*pi = 0.628...
  // dphi = 0.628 / pi = 0.2
  float src[4] = {100.f, 0.9f, 200.f, 1.0f};
  float tgt[4] = {200.f, -0.9f, 400.f, 1.5f};

  auto ef = detail::computeEdgeFeatures(src, tgt, nFeatures);

  // rawDphi = pi * (-0.9 - 0.9) = pi * -1.8 = -5.6548...
  // -5.6548 < -pi, so add 2*pi: -5.6548 + 6.2831 = 0.6283...
  // dphi = 0.6283 / pi = 0.2
  BOOST_CHECK_CLOSE(ef[1], 0.2f, 1e-3f);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
