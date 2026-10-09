// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/CudaTrackBuilding.hpp"
#include "ActsPlugins/Gnn/GnnPipeline.hpp"
#include "ActsPlugins/Gnn/ModuleMapCuda.hpp"
#include "ActsPlugins/Gnn/Stages.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include <cuda_runtime_api.h>
#include <traccc/definitions/common.hpp>
#include <traccc/edm/measurement_collection.hpp>
#include <traccc/edm/spacepoint_collection.hpp>
#include <traccc/edm/track_container.hpp>
#include <traccc/finding/finding_config.hpp>
#include <traccc/fitting/fitting_config.hpp>
#include <traccc/geometry/detector_buffer.hpp>
#include <traccc/seeding/detail/track_params_estimation_config.hpp>
#include <traccc/utils/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

namespace traccc {
class magnetic_field;
}

namespace ActsPlugins {

/// @addtogroup gnn_plugin
/// @{

/// Default Kalman fitter configuration for GNN track candidates.
///
/// The initial parameters of the candidates come from three of their space
/// points and are less precise than those of a track finder. With the traccc
/// defaults, a third of the ITk ttbar candidates fail in the forward
/// propagation, mostly because the momentum estimate drops below the 600 MeV
/// cut of the fitter, or because a measurement is missed by the navigation.
/// A lower momentum cut and the scattering noise estimate in the mask
/// tolerance of the navigation bring the fraction of fitted candidates from
/// 54% to 93%.
/// @return fitting configuration
inline traccc::fitting_config gnnFittingConfig() {
  traccc::fitting_config cfg;
  cfg.min_pT = 100.f * traccc::unit<float>::MeV;
  cfg.min_p = 100.f * traccc::unit<float>::MeV;
  cfg.propagation.navigation.estimate_scattering_noise = true;
  return cfg;
}

/// GNN track reconstruction on the GPU for traccc event data.
///
/// The space points and measurements stay on the device for the whole chain:
/// the node features and module ids are made from the traccc space points,
/// their measurements and the detray geometry, the graph is built with the
/// module map and classified, and the track candidates of the CUDA track
/// building are written as traccc tracks. Their initial parameters are
/// estimated from the first, middle and last space point along the
/// trajectory. Then, depending on the mode, the candidates are fitted with the
/// traccc Kalman fitter, or they seed the traccc combinatorial Kalman filter.
class TracccGnnChain {
 public:
  /// What is done with the GNN track candidates
  enum class Mode {
    /// Fit the measurements of the candidates
    Fit,
    /// Use the candidates as seeds of the combinatorial Kalman filter, which
    /// needs the measurements sorted by surface
    CombinatorialKalmanFilter,
  };

  /// Configuration of the chain
  struct Config {
    /// Graph construction, needs to accept device inputs
    std::shared_ptr<ModuleMapCuda> graphConstructor;
    /// Edge classifiers, run in this order
    std::vector<std::shared_ptr<EdgeClassificationBase>> edgeClassifiers;
    /// Configuration of the CUDA track building (junction removal)
    CudaTrackBuilding::Config trackBuilding;

    /// Number of node features: 4 for (r, phi, z, eta) of the space point, or
    /// 12 for the space point followed by its two measurements
    std::size_t nNodeFeatures = 12;
    /// Divisors of the (r, phi, z, eta) node features
    std::array<float, 4> featureScales = {1000.f, std::numbers::pi_v<float>,
                                          1000.f, 1.f};

    /// Minimum number of space points of a track candidate
    std::size_t minSpacePoints = 3;
    /// Maximum number of space points of a track candidate, larger ones are
    /// dropped
    std::size_t maxSpacePoints = 64;

    /// Parameter estimation from the first, middle and last space point
    traccc::track_params_estimation_config paramEstimation;
    /// What is done with the track candidates
    Mode mode = Mode::Fit;
    /// Kalman fitter configuration (Mode::Fit)
    traccc::fitting_config fitting = gnnFittingConfig();
    /// Track finding configuration (Mode::CombinatorialKalmanFilter)
    traccc::finding_config finding;
  };

  /// Track candidates and fitted tracks of an event, in device memory
  struct Result {
    /// Track candidates with estimated initial parameters
    traccc::edm::track_container<traccc::default_algebra>::buffer candidates;
    /// Fitted tracks (Mode::Fit) or tracks of the combinatorial Kalman
    /// filter (Mode::CombinatorialKalmanFilter)
    traccc::edm::track_container<traccc::default_algebra>::buffer tracks;
    /// Number of track candidates
    std::size_t nCandidates = 0;
  };

  /// @param cfg chain configuration
  /// @param moduleIds device lookup from detray surface index to module id
  ///        (DetrayModuleIdTable::data()), indexed by the surface index
  /// @param mr traccc memory resources (device and pinned host)
  /// @param copy vecmem copy for the stream
  /// @param stream CUDA stream of the chain
  /// @param logger logger
  TracccGnnChain(const Config &cfg, const std::uint64_t *moduleIds,
                 const traccc::memory_resource &mr, vecmem::copy &copy,
                 cudaStream_t stream,
                 std::unique_ptr<const Acts::Logger> logger);
  ~TracccGnnChain();

  TracccGnnChain(const TracccGnnChain &) = delete;
  TracccGnnChain &operator=(const TracccGnnChain &) = delete;

  /// Run the chain on the measurements and space points of an event
  /// @param detector traccc detector buffer
  /// @param field magnetic field for the parameter estimation and the fit
  /// @param measurements measurements in device memory
  /// @param spacePoints space points in device memory
  /// @param timing optional timing of the GNN stages
  /// @return candidates and fitted tracks
  Result operator()(
      const traccc::detector_buffer &detector,
      const traccc::magnetic_field &field,
      const traccc::edm::measurement_collection::const_view &measurements,
      const traccc::edm::spacepoint_collection::const_view &spacePoints,
      GnnTiming *timing = nullptr) const;

  /// Access the configuration
  /// @return configuration
  const Config &config() const { return m_cfg; }

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
  Config m_cfg;
  std::unique_ptr<const Acts::Logger> m_logger;

  const Acts::Logger &logger() const { return *m_logger; }
};

/// @}

}  // namespace ActsPlugins
