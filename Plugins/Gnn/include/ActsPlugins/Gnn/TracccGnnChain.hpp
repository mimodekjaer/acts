// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/DetrayModuleIdTable.hpp"
#include "ActsPlugins/Gnn/GnnPipeline.hpp"
#include "ActsPlugins/Gnn/ModuleMapCuda.hpp"
#include "ActsPlugins/Gnn/Stages.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <vector>

#include <traccc/bfield/magnetic_field.hpp>
#include <traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp>
#include <traccc/cuda/fitting/kalman_fitting_algorithm.hpp>
#include <traccc/cuda/seeding/seed_parameter_estimation_algorithm.hpp>
#include <traccc/cuda/utils/algorithm_base.hpp>
#include <traccc/cuda/utils/stream_wrapper.hpp>
#include <traccc/definitions/common.hpp>
#include <traccc/device/algorithm_base.hpp>
#include <traccc/edm/measurement_collection.hpp>
#include <traccc/edm/seed_collection.hpp>
#include <traccc/edm/spacepoint_collection.hpp>
#include <traccc/edm/track_container.hpp>
#include <traccc/finding/finding_config.hpp>
#include <traccc/fitting/fitting_config.hpp>
#include <traccc/geometry/detector_buffer.hpp>
#include <traccc/seeding/detail/track_params_estimation_config.hpp>
#include <traccc/utils/algorithm.hpp>
#include <traccc/utils/memory_resource.hpp>
#include <traccc/utils/messaging.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/copy.hpp>

namespace ActsPlugins {

namespace detail {
class DeviceMemory;
}

/// @addtogroup gnn_plugin
/// @{

/// Track candidates and tracks of the GNN chain for one event, in device
/// memory
struct TracccGnnResult {
  /// Track candidates with estimated initial parameters
  traccc::edm::track_container<traccc::default_algebra>::buffer candidates;
  /// Fitted tracks (TracccGnnChain::Mode::Fit) or tracks of the combinatorial
  /// Kalman filter (TracccGnnChain::Mode::CombinatorialKalmanFilter); empty
  /// for TracccGnnChain::Mode::Candidates
  traccc::edm::track_container<traccc::default_algebra>::buffer tracks;
  /// Number of track candidates
  std::size_t nCandidates = 0;
  /// Device copy of the measurements, when the chain was given host
  /// measurements (the candidates and tracks refer to it)
  traccc::edm::measurement_collection::buffer measurements;
};

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
/// A traccc algorithm: it is constructed with the traccc memory resources,
/// copy object and CUDA stream, and runs its work on that stream like the
/// traccc algorithms it uses. The space points and measurements stay on the
/// device for the whole chain: the node features and module ids are made from
/// the traccc space points, their measurements and the detray geometry, the
/// graph is built with the module map and classified, and the track candidates
/// of the CUDA track building are written as traccc tracks. Their initial
/// parameters are estimated from the first, middle and last space point along
/// the trajectory. Then, depending on the mode, the candidates are fitted with
/// the traccc Kalman fitter, or they seed the traccc combinatorial Kalman
/// filter.
class TracccGnnChain
    : public traccc::algorithm<TracccGnnResult(
          const traccc::detector_buffer &, const traccc::magnetic_field &,
          const traccc::edm::measurement_collection::const_view &,
          const traccc::edm::spacepoint_collection::const_view &)>,
      public traccc::messaging,
      public traccc::device::algorithm_base,
      public traccc::cuda::algorithm_base {
 public:
  /// What is done with the GNN track candidates
  enum class Mode {
    /// Stop after the track candidates, without initial parameters
    Candidates,
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
    /// Keep only the highest-scoring edge at nodes with several incoming or
    /// several outgoing edges before the connected components
    bool doJunctionRemoval = true;

    /// Number of node features: 4 for (r, phi, z, eta) of the space point, or
    /// 12 for the space point followed by its two measurements
    std::size_t nNodeFeatures = 12;
    /// Divisors of the (r, phi, z, eta) node features
    std::array<float, 4> featureScales = {1000.f, std::numbers::pi_v<float>,
                                          1000.f, 1.f};

    /// Minimum number of space points of a track candidate
    std::size_t minSpacePoints = 3;
    /// Maximum number of space points of a track candidate (at most 64),
    /// larger ones are dropped
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

  /// @param cfg chain configuration
  /// @param moduleIds lookup from detray surface index to module id, must
  ///        outlive the chain
  /// @param mr traccc memory resources (device and pinned host)
  /// @param copy vecmem copy object for @p stream
  /// @param stream CUDA stream of the chain
  /// @param logger logger
  TracccGnnChain(const Config &cfg, const DetrayModuleIdTable &moduleIds,
                 const traccc::memory_resource &mr, const vecmem::copy &copy,
                 const traccc::cuda::stream_wrapper &stream,
                 std::unique_ptr<const Acts::Logger> logger);

  /// Run the chain on the measurements and space points of an event
  /// @param detector traccc detector buffer
  /// @param field magnetic field for the parameter estimation and the fit
  /// @param measurements measurements in device memory
  /// @param spacePoints space points in device memory
  /// @return candidates and fitted tracks
  output_type operator()(
      const traccc::detector_buffer &detector,
      const traccc::magnetic_field &field,
      const traccc::edm::measurement_collection::const_view &measurements,
      const traccc::edm::spacepoint_collection::const_view &spacePoints)
      const override;

  /// Run the chain and time its GNN stages
  /// @param detector traccc detector buffer
  /// @param field magnetic field for the parameter estimation and the fit
  /// @param measurements measurements in device memory
  /// @param spacePoints space points in device memory
  /// @param timing timing of the GNN stages
  /// @return candidates and fitted tracks
  output_type operator()(
      const traccc::detector_buffer &detector,
      const traccc::magnetic_field &field,
      const traccc::edm::measurement_collection::const_view &measurements,
      const traccc::edm::spacepoint_collection::const_view &spacePoints,
      GnnTiming &timing) const;

  /// Run the chain on measurements and space points in host memory, e.g.
  /// from the CPU reconstruction. They are copied to the device first.
  /// @param detector traccc detector buffer
  /// @param field magnetic field for the parameter estimation and the fit
  /// @param measurements measurements in host memory
  /// @param spacePoints space points in host memory
  /// @param timing optional timing of the GNN stages
  /// @return candidates and fitted tracks, with the device copy of the
  ///         measurements they refer to
  output_type operator()(
      const traccc::detector_buffer &detector,
      const traccc::magnetic_field &field,
      const traccc::edm::measurement_collection::host &measurements,
      const traccc::edm::spacepoint_collection::host &spacePoints,
      GnnTiming *timing = nullptr) const;

  /// Access the configuration
  /// @return configuration
  const Config &config() const { return m_cfg; }

 private:
  using Clock = std::chrono::high_resolution_clock;

  /// The nodes of the graph, sorted by module id
  struct Nodes {
    /// [nNodes, nNodeFeatures] node features
    Tensor<float> features;
    /// Module id of each node
    vecmem::unique_alloc_ptr<std::uint64_t[]> moduleIds;
    /// Space point index of each node
    vecmem::unique_alloc_ptr<int[]> spacePoints;
  };

  output_type run(
      const traccc::detector_buffer &detector,
      const traccc::magnetic_field &field,
      const traccc::edm::measurement_collection::const_view &measurements,
      const traccc::edm::spacepoint_collection::const_view &spacePoints,
      GnnTiming *timing) const;

  /// Node features and module ids of the space points, sorted by module id
  Nodes makeNodes(
      const traccc::detector_buffer &detector,
      const traccc::edm::measurement_collection::const_view &measurements,
      const traccc::edm::spacepoint_collection::const_view &spacePoints,
      std::size_t nSpacePoints, detail::DeviceMemory &mem,
      const ExecutionContext &ctx) const;

  /// Graph construction and edge classification. Takes the node features of
  /// @p nodes. Returns nothing if no edges are left.
  std::optional<PipelineTensors> buildGraph(Nodes &nodes,
                                            const ExecutionContext &ctx,
                                            Clock::time_point start,
                                            GnnTiming *timing) const;

  /// Track candidates from the connected components of the classified graph.
  /// Writes the candidates (with their measurements in trajectory order) and
  /// their number to @p result and returns their seeds for the parameter
  /// estimation.
  traccc::edm::seed_collection::buffer buildCandidates(
      const PipelineTensors &graph, const int *nodeSpacePoints,
      const traccc::edm::spacepoint_collection::const_view &spacePoints,
      detail::DeviceMemory &mem, output_type &result) const;

  Config m_cfg;
  /// Device lookup from detray surface index to module id
  const std::uint64_t *m_moduleIds;
  traccc::cuda::seed_parameter_estimation_algorithm m_paramEstimation;
  traccc::cuda::kalman_fitting_algorithm m_fitting;
  traccc::cuda::combinatorial_kalman_filter_algorithm m_finding;
};

/// @}

}  // namespace ActsPlugins
