// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"
#include "ActsPlugins/Gnn/DetrayModuleIdTable.cuh"

#include <cstdint>
#include <numbers>
#include <optional>
#include <random>
#include <vector>

#include <detray/geometry/tracking_surface.hpp>
#include <detray/test/common/build_toy_detector.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

#include "DetrayNodeFeaturesTestKernels.hpp"

using namespace ActsPlugins;
using namespace ActsTests::GnnDetray;

BOOST_AUTO_TEST_SUITE(GnnSuite)

// The device features must be identical to the host feature creation of the
// GNN examples, which uses Acts::VectorHelpers
BOOST_AUTO_TEST_CASE(detray_node_features_match_vector_helpers) {
  const Scales scales{1000.f, std::numbers::pi_v<float>, 1000.f, 1.f};
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> xy(-1000.f, 1000.f);
  std::uniform_real_distribution<float> z(-3000.f, 3000.f);
  const std::size_t n = 100000;
  std::vector<float> xyz(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    xyz[3 * i] = xy(rng);
    xyz[3 * i + 1] = xy(rng);
    xyz[3 * i + 2] = z(rng);
  }

  const auto f = deviceNodeFeatures(xyz, scales);

  std::size_t nDiff = 0;
  for (std::size_t i = 0; i < n; ++i) {
    // Same as createFeatures in the GNN examples
    const Acts::Vector3 p{xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
    float ref[4] = {static_cast<float>(Acts::VectorHelpers::perp(p)),
                    static_cast<float>(Acts::VectorHelpers::phi(p)),
                    static_cast<float>(p.z()),
                    static_cast<float>(Acts::VectorHelpers::eta(p))};
    ref[0] /= scales.r;
    ref[1] /= scales.phi;
    ref[2] /= scales.z;
    ref[3] /= scales.eta;
    for (std::size_t k = 0; k < 4; ++k) {
      nDiff += (f[4 * i + k] != ref[k]);
    }
  }
  // The double precision results of the host and device math libraries can
  // differ in the last bit, which very rarely changes the rounding to float
  BOOST_CHECK_LE(nDiff, 4u);
}

BOOST_AUTO_TEST_CASE(detray_global_positions_and_module_ids) {
  vecmem::host_memory_resource hostMr;
  vecmem::cuda::device_memory_resource deviceMr;
  vecmem::cuda::copy copy;

  auto [det, names] = detray::build_toy_detector<algebra_t>(hostMr);

  // Module ids for every second sensitive surface
  auto moduleId =
      [](detray::geometry::identifier id) -> std::optional<std::uint64_t> {
    if (id.index() % 2 == 0) {
      return 1000u + id.index();
    }
    return std::nullopt;
  };
  const DetrayModuleIdTable table(det, moduleId, deviceMr, copy);
  BOOST_CHECK_EQUAL(table.size(), det.surfaces().size());
  BOOST_CHECK_GT(table.nSensitive(), 0u);
  BOOST_CHECK_GT(table.nUnmapped(), 0u);

  std::vector<std::uint64_t> hostTable(table.size());
  copy(vecmem::data::vector_view<const std::uint64_t>(
           static_cast<unsigned int>(table.size()), table.data()),
       vecmem::data::vector_view<std::uint64_t>(
           static_cast<unsigned int>(hostTable.size()), hostTable.data()),
       vecmem::copy::type::device_to_host)
      ->wait();

  std::vector<detray::geometry::identifier> surfaces;
  std::vector<point2_t> locals;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> loc(-5.f, 5.f);
  for (const auto &sf : det.surfaces()) {
    if (sf.is_sensitive() && sf.index() % 2 == 0) {
      BOOST_CHECK_EQUAL(hostTable.at(sf.index()), 1000u + sf.index());
    } else {
      BOOST_CHECK_EQUAL(hostTable.at(sf.index()),
                        DetrayModuleIdTable::kInvalid);
    }
    if (sf.is_sensitive()) {
      surfaces.push_back(sf.identifier());
      locals.push_back({loc(rng), loc(rng)});
    }
  }
  BOOST_REQUIRE(!surfaces.empty());

  // Global positions on the device agree with the host
  auto detBuffer = detray::get_buffer(det, deviceMr, copy);
  const auto xyz = deviceGlobalPositions(detBuffer, surfaces, locals);
  for (std::size_t i = 0; i < surfaces.size(); ++i) {
    const detray::tracking_surface sf{det, surfaces[i]};
    const auto ref = sf.local_to_global({}, locals[i], {});
    for (unsigned int k = 0; k < 3; ++k) {
      BOOST_CHECK_CLOSE(xyz[3 * i + k], ref[k], 1e-3);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
