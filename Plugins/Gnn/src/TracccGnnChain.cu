// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/DetrayModuleIdTable.cuh"
#include "ActsPlugins/Gnn/TracccGnnChain.hpp"
#include "ActsPlugins/Gnn/detail/ConnectedComponents.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"
#include "ActsPlugins/Gnn/detail/DetrayNodeFeatures.cuh"
#include "ActsPlugins/Gnn/detail/DeviceMemory.cuh"
#include "ActsPlugins/Gnn/detail/JunctionRemoval.hpp"
#include "ActsPlugins/Gnn/detail/NvtxUtils.hpp"

#include <chrono>
#include <limits>
#include <optional>

#include <cub/device/device_radix_sort.cuh>
#include <thrust/scan.h>
#include <traccc/bfield/magnetic_field.hpp>
#include <traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp>
#include <traccc/cuda/fitting/kalman_fitting_algorithm.hpp>
#include <traccc/cuda/seeding/seed_parameter_estimation_algorithm.hpp>
#include <traccc/cuda/utils/stream_wrapper.hpp>
#include <traccc/edm/seed_collection.hpp>
#include <traccc/edm/track_collection.hpp>
#include <traccc/geometry/detector.hpp>
#include <traccc/utils/detector_buffer_bfield_visitor.hpp>
#include <vecmem/containers/device_vector.hpp>

using namespace Acts;

namespace ActsPlugins {

namespace {

using algebra_t = traccc::default_algebra;
using spacepoint_device_t = traccc::edm::spacepoint_collection::const_device;
using measurement_device_t = traccc::edm::measurement_collection::const_device;

constexpr unsigned int kBlock = 256;

unsigned int nBlocks(std::size_t n) {
  return static_cast<unsigned int>((n + kBlock - 1) / kBlock);
}

/// Node features and module ids of the space points
template <typename detector_t>
__global__ void tracccNodeFeatures(
    detray::detector_view_t<detector_t> detView,
    traccc::edm::measurement_collection::const_view measurementsView,
    traccc::edm::spacepoint_collection::const_view spacePointsView,
    const std::uint64_t *moduleIdTable, detail::NodeFeatureScales scales,
    std::size_t nFeatures, float *features, std::uint64_t *moduleIds) {
  const spacepoint_device_t spacePoints(spacePointsView);
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= spacePoints.size()) {
    return;
  }
  const measurement_device_t measurements(measurementsView);
  const detray::detector_device_t<detector_t> det(detView);

  const auto sp = spacePoints.at(static_cast<unsigned int>(i));
  const bool twoMeasurements =
      sp.measurement_index_2() !=
      traccc::edm::spacepoint_collection::device::INVALID_MEASUREMENT_INDEX;
  const auto m1 = measurements.at(sp.measurement_index_1());
  detray::geometry::identifier surfaces[2] = {m1.surface_link(),
                                              m1.surface_link()};
  traccc::point2 locals[2] = {m1.local_position(), m1.local_position()};
  if (twoMeasurements) {
    const auto m2 = measurements.at(sp.measurement_index_2());
    surfaces[1] = m2.surface_link();
    locals[1] = m2.local_position();
  }

  detail::writeSpacePointFeatures(det, sp.global(), twoMeasurements ? 2u : 1u,
                                  surfaces, locals, nFeatures,
                                  features + i * nFeatures, scales);
  moduleIds[i] = moduleIdTable[surfaces[0].index()];
}

__global__ void gatherRows(std::size_t n, std::size_t nCols, const int *rows,
                           const float *in, float *out) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * nCols) {
    return;
  }
  out[i] = in[static_cast<std::size_t>(rows[i / nCols]) * nCols + i % nCols];
}

/// Marks the labels that are kept as track candidates
__global__ void markCandidates(const int *bounds, const int *numLabels,
                               unsigned int minSize, unsigned int maxSize,
                               unsigned int *keep) {
  const std::size_t l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= static_cast<std::size_t>(*numLabels)) {
    return;
  }
  const auto size = static_cast<unsigned int>(bounds[l + 1] - bounds[l]);
  keep[l] = (size >= minSize && size <= maxSize) ? 1u : 0u;
}

/// Squared distance from the origin, which increases along the trajectory of
/// tracks from the beam line both in the barrel and in the endcaps
__device__ float distance2(const spacepoint_device_t &sps, unsigned int i) {
  const auto g = sps.at(i).global();
  return g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
}

/// Writes the seed (first, middle and last space point along the trajectory)
/// and the measurements of each track candidate, in trajectory order
template <unsigned int kMaxSize>
__global__ void fillCandidates(
    const int *bounds, const int *sortedSpacePoints, const int *numLabels,
    const unsigned int *keep, const unsigned int *candidateIndex,
    traccc::edm::spacepoint_collection::const_view spacePointsView,
    traccc::edm::seed_collection::view seedsView,
    traccc::edm::track_collection<algebra_t>::view tracksView) {
  const std::size_t l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= static_cast<std::size_t>(*numLabels) || keep[l] == 0u) {
    return;
  }
  const spacepoint_device_t spacePoints(spacePointsView);
  traccc::edm::seed_collection::device seeds(seedsView);
  traccc::edm::track_collection<algebra_t>::device tracks(tracksView);

  // Space points of the candidate, in trajectory order
  unsigned int sp[kMaxSize];
  float r2[kMaxSize];
  const int begin = bounds[l];
  const unsigned int size = static_cast<unsigned int>(bounds[l + 1] - begin);
  for (unsigned int k = 0; k < size; ++k) {
    const auto idx = static_cast<unsigned int>(sortedSpacePoints[begin + k]);
    const float r = distance2(spacePoints, idx);
    unsigned int j = k;
    for (; j > 0 && r2[j - 1] > r; --j) {
      sp[j] = sp[j - 1];
      r2[j] = r2[j - 1];
    }
    sp[j] = idx;
    r2[j] = r;
  }

  const unsigned int c = candidateIndex[l];
  auto seed = seeds.at(c);
  seed.bottom_index() = sp[0];
  seed.middle_index() = sp[size / 2];
  seed.top_index() = sp[size - 1];

  seed.quality() = 0.f;

  auto track = tracks.at(c);
  track.fit_outcome() = traccc::track_fit_outcome::UNKNOWN;
  track.ndf() = 0.f;
  track.chi2() = 0.f;
  track.pval() = 0.f;
  track.nholes() = 0u;
  using link_t = traccc::edm::track_constituent_link;
  for (unsigned int k = 0; k < size; ++k) {
    const auto s = spacePoints.at(sp[k]);
    track.constituent_links().push_back(
        {link_t::measurement, s.measurement_index_1()});
    if (s.measurement_index_2() !=
        traccc::edm::spacepoint_collection::device::INVALID_MEASUREMENT_INDEX) {
      track.constituent_links().push_back(
          {link_t::measurement, s.measurement_index_2()});
    }
  }
}

/// Copies the estimated parameters to the track candidates
__global__ void setParameters(
    unsigned int n,
    traccc::bound_track_parameters_collection_types::const_view paramsView,
    traccc::edm::track_collection<algebra_t>::view tracksView) {
  const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) {
    return;
  }
  const traccc::bound_track_parameters_collection_types::const_device params(
      paramsView);
  traccc::edm::track_collection<algebra_t>::device tracks(tracksView);
  tracks.at(c).params() = params.at(c);
}

}  // namespace

struct TracccGnnChain::Impl {
  const std::uint64_t *moduleIds;
  traccc::memory_resource mr;
  vecmem::copy &copy;
  cudaStream_t stream;
  traccc::cuda::stream_wrapper tracccStream;
  traccc::cuda::seed_parameter_estimation_algorithm paramEstimation;
  traccc::cuda::kalman_fitting_algorithm fitting;
  traccc::cuda::combinatorial_kalman_filter_algorithm finding;

  Impl(const Config &cfg, const std::uint64_t *moduleIds_,
       const traccc::memory_resource &mr_, vecmem::copy &copy_,
       cudaStream_t stream_, const Logger &logger)
      : moduleIds(moduleIds_),
        mr(mr_),
        copy(copy_),
        stream(stream_),
        tracccStream(stream_),
        paramEstimation(
            cfg.paramEstimation, mr_, copy_, tracccStream,
            traccc::getDefaultLogger("GnnParamEstimation", logger.level())),
        fitting(cfg.fitting, mr_, copy_, tracccStream,
                traccc::getDefaultLogger("GnnFitting", logger.level())),
        finding(cfg.finding, mr_, copy_, tracccStream,
                traccc::getDefaultLogger("GnnFinding", logger.level())) {}
};

TracccGnnChain::TracccGnnChain(const Config &cfg,
                               const std::uint64_t *moduleIds,
                               const traccc::memory_resource &mr,
                               vecmem::copy &copy, cudaStream_t stream,
                               std::unique_ptr<const Logger> logger)
    : m_cfg(cfg), m_logger(std::move(logger)) {
  if (m_cfg.graphConstructor == nullptr) {
    throw std::invalid_argument("TracccGnnChain: no graph constructor");
  }
  if (m_cfg.nNodeFeatures != 4 && m_cfg.nNodeFeatures != 12) {
    throw std::invalid_argument(
        "TracccGnnChain: nNodeFeatures must be 4 or 12");
  }
  if (m_cfg.minSpacePoints < 1 || m_cfg.maxSpacePoints > 64) {
    throw std::invalid_argument(
        "TracccGnnChain: need 1 <= minSpacePoints, maxSpacePoints <= 64");
  }
  m_impl = std::make_unique<Impl>(m_cfg, moduleIds, mr, copy, stream,
                                  this->logger());
}

TracccGnnChain::~TracccGnnChain() = default;

TracccGnnChain::Result TracccGnnChain::operator()(
    const traccc::detector_buffer &detector,
    const traccc::magnetic_field &field,
    const traccc::edm::measurement_collection::const_view &measurements,
    const traccc::edm::spacepoint_collection::const_view &spacePoints,
    GnnTiming *timing) const {
  using Clock = std::chrono::high_resolution_clock;
  const cudaStream_t stream = m_impl->stream;
  detail::DeviceMemory mem(stream, &m_impl->mr.main);
  const ExecutionContext ctx{Device::Cuda(), stream, &m_impl->mr.main};

  Result result{{{}, {}, measurements}, {{}, {}, measurements}, 0};
  const std::size_t n = m_impl->copy.get_size(spacePoints);
  ACTS_DEBUG("Received " << n << " space points");
  if (n == 0) {
    return result;
  }
  const std::size_t nF = m_cfg.nNodeFeatures;

  // ---------------------------------------------------------------
  // Node features and module ids, nodes sorted by module id (stable)
  // ---------------------------------------------------------------
  auto t0 = Clock::now();
  ACTS_NVTX_START(gnn_chain_node_features);
  auto unsortedFeatures = mem.make<float>(n * nF);
  auto unsortedModuleIds = mem.make<std::uint64_t>(n);
  const detail::NodeFeatureScales scales{
      m_cfg.featureScales[0], m_cfg.featureScales[1], m_cfg.featureScales[2],
      m_cfg.featureScales[3]};
  traccc::detector_buffer_visitor<traccc::detector_type_list>(
      detector, [&]<detray::concepts::detector detector_t>(
                    const detray::detector_view_t<detector_t> &det) {
        tracccNodeFeatures<detector_t><<<nBlocks(n), kBlock, 0, stream>>>(
            det, measurements, spacePoints, m_impl->moduleIds, scales, nF,
            unsortedFeatures.get(), unsortedModuleIds.get());
      });
  ACTS_CUDA_CHECK(cudaGetLastError());

  auto order = mem.make<int>(n);
  auto nodeSpacePoints = mem.make<int>(n);
  auto moduleIds = mem.make<std::uint64_t>(n);
  detail::iota<<<nBlocks(n), kBlock, 0, stream>>>(n, order.get());
  ACTS_CUDA_CHECK(cudaGetLastError());
  {
    std::size_t bytes = 0;
    ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, unsortedModuleIds.get(), moduleIds.get(), order.get(),
        nodeSpacePoints.get(), n, 0, 64, stream));
    auto temp = mem.make<std::byte>(bytes);
    ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.get(), bytes, unsortedModuleIds.get(), moduleIds.get(),
        order.get(), nodeSpacePoints.get(), n, 0, 64, stream));
  }
  auto nodeFeatures = Tensor<float>::Create({n, nF}, ctx);
  gatherRows<<<nBlocks(n * nF), kBlock, 0, stream>>>(
      n, nF, nodeSpacePoints.get(), unsortedFeatures.get(),
      nodeFeatures.data());
  ACTS_CUDA_CHECK(cudaGetLastError());
  unsortedFeatures.reset();
  unsortedModuleIds.reset();
  ACTS_NVTX_STOP(gnn_chain_node_features);

  // ------------------------------------
  // Graph construction and classification
  // ------------------------------------
  std::optional<PipelineTensors> graph;
  try {
    ACTS_NVTX_START(gnn_chain_graph);
    graph.emplace((*m_cfg.graphConstructor)(std::move(nodeFeatures),
                                            moduleIds.get(), ctx));
    ACTS_NVTX_STOP(gnn_chain_graph);
    if (timing != nullptr) {
      timing->graphBuildingTime = Clock::now() - t0;
      timing->classifierTimes.clear();
    }
    for (const auto &classifier : m_cfg.edgeClassifiers) {
      auto tc = Clock::now();
      ACTS_NVTX_START(gnn_chain_classifier);
      graph.emplace((*classifier)(std::move(*graph), ctx));
      ACTS_NVTX_STOP(gnn_chain_classifier);
      if (timing != nullptr) {
        timing->classifierTimes.push_back(Clock::now() - tc);
      }
    }
  } catch (const NoEdgesError &) {
    ACTS_DEBUG("No edges left in the graph");
    return result;
  }
  auto &tensors = *graph;

  // ---------------------------------------------------
  // Track building: junction removal, connected components
  // ---------------------------------------------------
  auto t2 = Clock::now();
  ACTS_NVTX_START(gnn_chain_track_building);
  const auto numNodes = tensors.nodeFeatures.shape().at(0);
  auto numEdges = static_cast<std::size_t>(tensors.edgeIndex.shape().at(1));
  const std::int64_t *src = tensors.edgeIndex.data();
  const std::int64_t *tgt = tensors.edgeIndex.data() + numEdges;
  auto counters = mem.make<int>(2);
  const int *cudaNumEdges = nullptr;
  vecmem::unique_alloc_ptr<std::int64_t[]> jrEdges;
  if (m_cfg.trackBuilding.doJunctionRemoval && numEdges > 0) {
    jrEdges = mem.make<std::int64_t>(2 * numEdges);
    detail::junctionRemovalCudaAsync(
        numEdges, numNodes, tensors.edgeScores->data(), src, tgt, jrEdges.get(),
        jrEdges.get() + numEdges, counters.get(), stream, &m_impl->mr.main);
    src = jrEdges.get();
    tgt = jrEdges.get() + numEdges;
    cudaNumEdges = counters.get();
  }
  auto labels = mem.make<int>(numNodes);
  auto bounds = mem.make<int>(numNodes + 1);
  auto sortedSpacePoints = mem.make<int>(numNodes);
  int *numLabels = counters.get() + 1;
  detail::connectedComponentsCudaAsync(numEdges, cudaNumEdges, src, tgt,
                                       numNodes, labels.get(), numLabels, mem);
  // The nodes are the sorted space points, nodeSpacePoints maps them back
  detail::findTrackCandidateBoundsAsync(labels.get(), nodeSpacePoints.get(),
                                        sortedSpacePoints.get(), bounds.get(),
                                        numNodes, numLabels, mem);

  // ---------------------------------------------------------------
  // Track candidates: seeds and measurements, sorted by radius
  // ---------------------------------------------------------------
  auto keep = mem.make<unsigned int>(numNodes + 1);
  auto candidateIndex = mem.make<unsigned int>(numNodes + 1);
  mem.memset(keep.get(), (numNodes + 1), 0);
  markCandidates<<<nBlocks(numNodes), kBlock, 0, stream>>>(
      bounds.get(), numLabels, static_cast<unsigned int>(m_cfg.minSpacePoints),
      static_cast<unsigned int>(m_cfg.maxSpacePoints), keep.get());
  ACTS_CUDA_CHECK(cudaGetLastError());
  thrust::exclusive_scan(mem.policy(), keep.get(), keep.get() + numNodes + 1,
                         candidateIndex.get());

  // The candidate count is needed on the host to size the buffers
  int hostCounters[2] = {};
  unsigned int nCandidates = 0;
  mem.toHost(hostCounters, counters.get(), 2);
  mem.toHost(&nCandidates, candidateIndex.get() + numNodes, 1);
  mem.synchronize();
  ACTS_DEBUG("Labels: " << hostCounters[1]
                        << ", track candidates: " << nCandidates);
  result.nCandidates = nCandidates;
  if (nCandidates == 0) {
    return result;
  }

  traccc::edm::seed_collection::buffer seeds(nCandidates, m_impl->mr.main);
  m_impl->copy.setup(seeds)->ignore();
  // At most two measurements per space point
  result.candidates.tracks = {
      std::vector<std::size_t>(nCandidates, 2 * m_cfg.maxSpacePoints),
      m_impl->mr.main, m_impl->mr.host, vecmem::data::buffer_type::resizable};
  m_impl->copy.setup(result.candidates.tracks)->ignore();
  fillCandidates<64><<<nBlocks(numNodes), kBlock, 0, stream>>>(
      bounds.get(), sortedSpacePoints.get(), numLabels, keep.get(),
      candidateIndex.get(), spacePoints, seeds, result.candidates.tracks);
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_NVTX_STOP(gnn_chain_track_building);
  auto t3 = Clock::now();
  if (timing != nullptr) {
    timing->trackBuildingTime = t3 - t2;
  }

  // -------------------------------------
  // Parameter estimation and track fitting
  // -------------------------------------
  ACTS_NVTX_START(gnn_chain_param_estimation);
  const auto params =
      m_impl->paramEstimation(field, measurements, spacePoints, seeds);
  setParameters<<<nBlocks(nCandidates), kBlock, 0, stream>>>(
      nCandidates, params, result.candidates.tracks);
  ACTS_CUDA_CHECK(cudaGetLastError());

  ACTS_NVTX_STOP(gnn_chain_param_estimation);
  if (m_cfg.mode == Mode::Fit) {
    ACTS_NVTX_START(gnn_chain_fit);
    result.tracks = m_impl->fitting(
        detector, field,
        traccc::edm::track_container<algebra_t>::const_view{result.candidates});
    mem.synchronize();
    ACTS_NVTX_STOP(gnn_chain_fit);
    ACTS_DEBUG("Fitted " << nCandidates << " track candidates");
  } else {
    ACTS_NVTX_START(gnn_chain_ckf);
    result.tracks = m_impl->finding(detector, field, measurements, params);
    mem.synchronize();
    ACTS_NVTX_STOP(gnn_chain_ckf);
    ACTS_DEBUG("Ran the combinatorial Kalman filter on " << nCandidates
                                                         << " seeds");
  }
  return result;
}

TracccGnnChain::Result TracccGnnChain::operator()(
    const traccc::detector_buffer &detector,
    const traccc::magnetic_field &field,
    const traccc::edm::measurement_collection::host &measurements,
    const traccc::edm::spacepoint_collection::host &spacePoints,
    GnnTiming *timing) const {
  // Copy the event to the device and run on it there. The buffers live until
  // the chain is done with them, which the synchronization at its end ensures.
  traccc::edm::measurement_collection::buffer measurementBuffer(
      static_cast<unsigned int>(measurements.size()), m_impl->mr.main);
  m_impl->copy.setup(measurementBuffer)->ignore();
  m_impl->copy(vecmem::get_data(measurements), measurementBuffer)->ignore();
  traccc::edm::spacepoint_collection::buffer spacePointBuffer(
      static_cast<unsigned int>(spacePoints.size()), m_impl->mr.main);
  m_impl->copy.setup(spacePointBuffer)->ignore();
  m_impl->copy(vecmem::get_data(spacePoints), spacePointBuffer)->ignore();
  Result result =
      (*this)(detector, field, measurementBuffer, spacePointBuffer, timing);
  // The results refer to the measurements: keep the device copy alive
  result.measurements = std::move(measurementBuffer);
  return result;
}

}  // namespace ActsPlugins
