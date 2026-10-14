// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/TracccGnnChain.hpp"
#include "ActsPlugins/Gnn/detail/ConnectedComponents.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"
#include "ActsPlugins/Gnn/detail/DetrayNodeFeatures.cuh"
#include "ActsPlugins/Gnn/detail/DeviceMemory.cuh"
#include "ActsPlugins/Gnn/detail/JunctionRemoval.hpp"
#include "ActsPlugins/Gnn/detail/NvtxUtils.hpp"

#include <stdexcept>
#include <string>

#include <cub/device/device_radix_sort.cuh>
#include <thrust/scan.h>
#include <traccc/edm/track_collection.hpp>
#include <traccc/geometry/detector.hpp>
#include <traccc/utils/detector_buffer_bfield_visitor.hpp>

using namespace Acts;

namespace ActsPlugins {

namespace {

using algebra_t = traccc::default_algebra;
using spacepoint_device_t = traccc::edm::spacepoint_collection::const_device;
using measurement_device_t = traccc::edm::measurement_collection::const_device;

/// Largest track candidate that fillCandidates can sort in registers
constexpr unsigned int kMaxCandidateSize = 64;

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

/// out[i] = in[rows[i]] for rows of nCols elements
__global__ void gatherRows(std::size_t n, std::size_t nCols, const int *rows,
                           const float *in, float *out) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * nCols) {
    return;
  }
  out[i] = in[static_cast<std::size_t>(rows[i / nCols]) * nCols + i % nCols];
}

/// Marks the labels that are kept as track candidates
__global__ void markCandidates(
    const int *bounds, const int *sortedSpacePoints, const int *numLabels,
    traccc::edm::spacepoint_collection::const_view spacePointsView,
    unsigned int minSize, unsigned int maxSize, unsigned int minMeasurements,
    unsigned int *keep) {
  const std::size_t l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= static_cast<std::size_t>(*numLabels)) {
    return;
  }
  const auto size = static_cast<unsigned int>(bounds[l + 1] - bounds[l]);
  bool pass = size >= minSize && size <= maxSize;
  if (pass && minMeasurements > size) {
    const spacepoint_device_t spacePoints(spacePointsView);
    unsigned int nMeasurements = 0;
    for (int k = bounds[l]; k < bounds[l + 1]; ++k) {
      nMeasurements +=
          spacePoints.at(static_cast<unsigned int>(sortedSpacePoints[k]))
                      .measurement_index_2() ==
                  traccc::edm::spacepoint_collection::device::
                      INVALID_MEASUREMENT_INDEX
              ? 1u
              : 2u;
    }
    pass = nMeasurements >= minMeasurements;
  }
  keep[l] = pass ? 1u : 0u;
}

/// Copies a space point collection into another one from @p offset on
__global__ void appendSpacePoints(
    traccc::edm::spacepoint_collection::const_view fromView,
    traccc::edm::spacepoint_collection::view toView, unsigned int offset) {
  const spacepoint_device_t from(fromView);
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= from.size()) {
    return;
  }
  traccc::edm::spacepoint_collection::device to(toView);
  const auto s = from.at(i);
  auto t = to.at(offset + i);
  t.measurement_index_1() = s.measurement_index_1();
  t.measurement_index_2() = s.measurement_index_2();
  t.global() = s.global();
  t.z_variance() = s.z_variance();
  t.radius_variance() = s.radius_variance();
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

  // Space points of the candidate, in trajectory order (insertion sort)
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

TracccGnnChain::TracccGnnChain(const Config &cfg,
                               const DetrayModuleIdTable &moduleIds,
                               const traccc::memory_resource &mr,
                               const vecmem::copy &copy,
                               const traccc::cuda::stream_wrapper &stream,
                               std::unique_ptr<const Logger> logger,
                               vecmem::memory_resource *temporaryMr)
    : traccc::messaging(std::move(logger)),
      traccc::device::algorithm_base(mr, copy),
      traccc::cuda::algorithm_base(stream),
      m_cfg(cfg),
      m_moduleIds(moduleIds.data()),
      m_temporaryMr(temporaryMr != nullptr ? temporaryMr : &mr.main),
      m_paramEstimation(cfg.paramEstimation, mr, copy, stream,
                        this->logger().cloneWithSuffix("ParamEstimation")),
      m_fitting(cfg.fitting, mr, copy, stream,
                this->logger().cloneWithSuffix("Fitting")),
      m_finding(cfg.finding, mr, copy, stream,
                this->logger().cloneWithSuffix("Finding")) {
  if (m_cfg.graphConstructor == nullptr) {
    throw std::invalid_argument("TracccGnnChain: no graph constructor");
  }
  if (m_cfg.nNodeFeatures != 4 && m_cfg.nNodeFeatures != 12) {
    throw std::invalid_argument(
        "TracccGnnChain: nNodeFeatures must be 4 or 12");
  }
  if (m_cfg.minSpacePoints < 1 || m_cfg.maxSpacePoints > kMaxCandidateSize) {
    throw std::invalid_argument(
        "TracccGnnChain: need 1 <= minSpacePoints, maxSpacePoints <= " +
        std::to_string(kMaxCandidateSize));
  }
}

TracccGnnChain::output_type TracccGnnChain::operator()(
    const traccc::detector_buffer &detector,
    const traccc::magnetic_field &field,
    const traccc::edm::measurement_collection::const_view &measurements,
    const traccc::edm::spacepoint_collection::const_view &spacePoints) const {
  return run(detector, field, measurements, spacePoints, nullptr);
}

TracccGnnChain::output_type TracccGnnChain::operator()(
    const traccc::detector_buffer &detector,
    const traccc::magnetic_field &field,
    const traccc::edm::measurement_collection::const_view &measurements,
    const traccc::edm::spacepoint_collection::const_view &spacePoints,
    GnnTiming &timing) const {
  return run(detector, field, measurements, spacePoints, &timing);
}

TracccGnnChain::output_type TracccGnnChain::operator()(
    const traccc::detector_buffer &detector,
    const traccc::magnetic_field &field,
    const traccc::edm::measurement_collection::const_view &measurements,
    const std::vector<traccc::edm::spacepoint_collection::const_view>
        &spacePointCollections,
    GnnTiming *timing) const {
  if (spacePointCollections.size() == 1) {
    return run(detector, field, measurements, spacePointCollections.front(),
               timing);
  }
  const auto cudaStream = static_cast<cudaStream_t>(stream().cudaStream());
  std::vector<unsigned int> sizes;
  unsigned int total = 0;
  for (const auto &view : spacePointCollections) {
    sizes.push_back(copy().get_size(view));
    total += sizes.back();
  }
  traccc::edm::spacepoint_collection::buffer spacePoints(total, *m_temporaryMr);
  copy().setup(spacePoints)->ignore();
  unsigned int offset = 0;
  for (std::size_t c = 0; c < spacePointCollections.size(); ++c) {
    if (sizes[c] > 0) {
      appendSpacePoints<<<nBlocks(sizes[c]), kBlock, 0, cudaStream>>>(
          spacePointCollections[c], spacePoints, offset);
      ACTS_CUDA_CHECK(cudaGetLastError());
    }
    offset += sizes[c];
  }
  output_type result = run(detector, field, measurements, spacePoints, timing);
  // The concatenated space points are freed here
  stream().synchronize();
  return result;
}

TracccGnnChain::output_type TracccGnnChain::operator()(
    const traccc::detector_buffer &detector,
    const traccc::magnetic_field &field,
    const traccc::edm::measurement_collection::host &measurements,
    const traccc::edm::spacepoint_collection::host &spacePoints,
    GnnTiming *timing) const {
  // Copy the event to the device and run on it there
  traccc::edm::measurement_collection::buffer measurementBuffer(
      static_cast<unsigned int>(measurements.size()), mr().main);
  copy().setup(measurementBuffer)->ignore();
  copy()(vecmem::get_data(measurements), measurementBuffer)->ignore();
  traccc::edm::spacepoint_collection::buffer spacePointBuffer(
      static_cast<unsigned int>(spacePoints.size()), mr().main);
  copy().setup(spacePointBuffer)->ignore();
  copy()(vecmem::get_data(spacePoints), spacePointBuffer)->ignore();

  output_type result =
      run(detector, field, measurementBuffer, spacePointBuffer, timing);
  // The space point buffer is freed here, so the chain must be done with it
  stream().synchronize();
  // The results refer to the measurements: keep the device copy alive
  result.measurements = std::move(measurementBuffer);
  return result;
}

TracccGnnChain::output_type TracccGnnChain::run(
    const traccc::detector_buffer &detector,
    const traccc::magnetic_field &field,
    const traccc::edm::measurement_collection::const_view &measurements,
    const traccc::edm::spacepoint_collection::const_view &spacePoints,
    GnnTiming *timing) const {
  const auto cudaStream = static_cast<cudaStream_t>(stream().cudaStream());
  detail::DeviceMemory mem(cudaStream, m_temporaryMr);
  const ExecutionContext ctx{Device::Cuda(), cudaStream, m_temporaryMr};

  output_type result;
  result.candidates.measurements = measurements;
  result.tracks.measurements = measurements;

  const std::size_t nSpacePoints = copy().get_size(spacePoints);
  ACTS_DEBUG("Received " << nSpacePoints << " space points");
  if (nSpacePoints == 0) {
    return result;
  }

  // Nodes, graph construction and classification
  const auto t0 = Clock::now();
  ACTS_NVTX_START(gnn_chain_node_features);
  Nodes nodes =
      makeNodes(detector, measurements, spacePoints, nSpacePoints, mem, ctx);
  ACTS_NVTX_STOP(gnn_chain_node_features);
  const std::optional<PipelineTensors> graph =
      buildGraph(nodes, ctx, t0, timing);
  if (!graph.has_value()) {
    ACTS_DEBUG("No edges left in the graph");
    return result;
  }

  // Track candidates with their seeds
  const auto t1 = Clock::now();
  ACTS_NVTX_START(gnn_chain_track_building);
  const traccc::edm::seed_collection::buffer seeds = buildCandidates(
      *graph, nodes.spacePoints.get(), spacePoints, mem, result);
  ACTS_NVTX_STOP(gnn_chain_track_building);
  if (timing != nullptr) {
    timing->trackBuildingTime = Clock::now() - t1;
  }
  const auto nCandidates = static_cast<unsigned int>(result.nCandidates);
  if (nCandidates == 0 || m_cfg.mode == Mode::Candidates) {
    mem.synchronize();
    return result;
  }

  // Initial parameters of the candidates
  ACTS_NVTX_START(gnn_chain_param_estimation);
  const auto params =
      m_paramEstimation(field, measurements, spacePoints, seeds);
  setParameters<<<nBlocks(nCandidates), kBlock, 0, cudaStream>>>(
      nCandidates, params, result.candidates.tracks);
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_NVTX_STOP(gnn_chain_param_estimation);

  // Fit the candidates, or use them as seeds of the track finding
  if (m_cfg.mode == Mode::Fit) {
    ACTS_NVTX_START(gnn_chain_fit);
    result.tracks = m_fitting(
        detector, field,
        traccc::edm::track_container<algebra_t>::const_view{result.candidates});
    mem.synchronize();
    ACTS_NVTX_STOP(gnn_chain_fit);
    ACTS_DEBUG("Fitted " << nCandidates << " track candidates");
  } else {
    ACTS_NVTX_START(gnn_chain_ckf);
    result.tracks = m_finding(detector, field, measurements, params);
    mem.synchronize();
    ACTS_NVTX_STOP(gnn_chain_ckf);
    ACTS_DEBUG("Ran the combinatorial Kalman filter on " << nCandidates
                                                         << " seeds");
  }
  return result;
}

TracccGnnChain::Nodes TracccGnnChain::makeNodes(
    const traccc::detector_buffer &detector,
    const traccc::edm::measurement_collection::const_view &measurements,
    const traccc::edm::spacepoint_collection::const_view &spacePoints,
    std::size_t nSpacePoints, detail::DeviceMemory &mem,
    const ExecutionContext &ctx) const {
  const cudaStream_t cudaStream = mem.stream();
  const std::size_t n = nSpacePoints;
  const std::size_t nFeatures = m_cfg.nNodeFeatures;

  // Features and module ids in space point order
  auto unsortedFeatures = mem.make<float>(n * nFeatures);
  auto unsortedModuleIds = mem.make<std::uint64_t>(n);
  const detail::NodeFeatureScales scales{
      m_cfg.featureScales[0], m_cfg.featureScales[1], m_cfg.featureScales[2],
      m_cfg.featureScales[3]};
  traccc::detector_buffer_visitor<traccc::detector_type_list>(
      detector, [&]<detray::concepts::detector detector_t>(
                    const detray::detector_view_t<detector_t> &det) {
        tracccNodeFeatures<detector_t><<<nBlocks(n), kBlock, 0, cudaStream>>>(
            det, measurements, spacePoints, m_moduleIds, scales, nFeatures,
            unsortedFeatures.get(), unsortedModuleIds.get());
      });
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Stable sort of the space point indices by module id, which the module
  // map needs; the features follow
  auto moduleIds = mem.make<std::uint64_t>(n);
  auto nodeSpacePoints = mem.make<int>(n);
  auto spacePointIndices = mem.make<int>(n);
  detail::iota<<<nBlocks(n), kBlock, 0, cudaStream>>>(n,
                                                      spacePointIndices.get());
  ACTS_CUDA_CHECK(cudaGetLastError());
  {
    std::size_t bytes = 0;
    ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, unsortedModuleIds.get(), moduleIds.get(),
        spacePointIndices.get(), nodeSpacePoints.get(), n, 0, 64, cudaStream));
    auto temp = mem.make<std::byte>(bytes);
    ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.get(), bytes, unsortedModuleIds.get(), moduleIds.get(),
        spacePointIndices.get(), nodeSpacePoints.get(), n, 0, 64, cudaStream));
  }
  auto features = Tensor<float>::Create({n, nFeatures}, ctx);
  gatherRows<<<nBlocks(n * nFeatures), kBlock, 0, cudaStream>>>(
      n, nFeatures, nodeSpacePoints.get(), unsortedFeatures.get(),
      features.data());
  ACTS_CUDA_CHECK(cudaGetLastError());
  return Nodes{std::move(features), std::move(moduleIds),
               std::move(nodeSpacePoints)};
}

std::optional<PipelineTensors> TracccGnnChain::buildGraph(
    Nodes &nodes, const ExecutionContext &ctx, Clock::time_point start,
    GnnTiming *timing) const {
  std::optional<PipelineTensors> graph;
  try {
    ACTS_NVTX_START(gnn_chain_graph);
    graph.emplace((*m_cfg.graphConstructor)(std::move(nodes.features),
                                            nodes.moduleIds.get(), ctx));
    ACTS_NVTX_STOP(gnn_chain_graph);
    if (timing != nullptr) {
      timing->graphBuildingTime = Clock::now() - start;
      timing->classifierTimes.clear();
    }
    for (const auto &classifier : m_cfg.edgeClassifiers) {
      const auto t = Clock::now();
      ACTS_NVTX_START(gnn_chain_classifier);
      graph.emplace((*classifier)(std::move(*graph), ctx));
      ACTS_NVTX_STOP(gnn_chain_classifier);
      if (timing != nullptr) {
        timing->classifierTimes.push_back(Clock::now() - t);
      }
    }
  } catch (const NoEdgesError &) {
    return std::nullopt;
  }
  return graph;
}

traccc::edm::seed_collection::buffer TracccGnnChain::buildCandidates(
    const PipelineTensors &graph, const int *nodeSpacePoints,
    const traccc::edm::spacepoint_collection::const_view &spacePoints,
    detail::DeviceMemory &mem, output_type &result) const {
  const cudaStream_t cudaStream = mem.stream();
  const std::size_t numNodes = graph.nodeFeatures.shape().at(0);
  const std::size_t numEdges = graph.edgeIndex.shape().at(1);
  const std::int64_t *src = graph.edgeIndex.data();
  const std::int64_t *tgt = graph.edgeIndex.data() + numEdges;

  // Everything up to the candidate count runs without host synchronization.
  // counters[0]: number of edges after junction removal, [1]: number of
  // labels
  auto counters = mem.make<int>(2);
  const int *cudaNumEdges = nullptr;
  int *numLabels = counters.get() + 1;
  vecmem::unique_alloc_ptr<std::int64_t[]> jrEdges;
  if (m_cfg.doJunctionRemoval && numEdges > 0) {
    jrEdges = mem.make<std::int64_t>(2 * numEdges);
    detail::junctionRemovalCudaAsync(
        numEdges, numNodes, graph.edgeScores->data(), src, tgt, jrEdges.get(),
        jrEdges.get() + numEdges, counters.get(), mem);
    src = jrEdges.get();
    tgt = jrEdges.get() + numEdges;
    cudaNumEdges = counters.get();
  }

  // Connected components, and the space points of each component
  auto labels = mem.make<int>(numNodes);
  auto bounds = mem.make<int>(numNodes + 1);
  auto sortedSpacePoints = mem.make<int>(numNodes);
  detail::connectedComponentsCudaAsync(numEdges, cudaNumEdges, src, tgt,
                                       numNodes, labels.get(), numLabels, mem);
  detail::findTrackCandidateBoundsAsync(labels.get(), nodeSpacePoints,
                                        sortedSpacePoints.get(), bounds.get(),
                                        numNodes, numLabels, mem);

  // Components within the size limits become candidates, candidateIndex
  // numbers them
  auto keep = mem.make<unsigned int>(numNodes + 1);
  auto candidateIndex = mem.make<unsigned int>(numNodes + 1);
  mem.memset(keep.get(), numNodes + 1, 0);
  markCandidates<<<nBlocks(numNodes), kBlock, 0, cudaStream>>>(
      bounds.get(), sortedSpacePoints.get(), numLabels, spacePoints,
      static_cast<unsigned int>(m_cfg.minSpacePoints),
      static_cast<unsigned int>(m_cfg.maxSpacePoints),
      static_cast<unsigned int>(m_cfg.minMeasurements), keep.get());
  ACTS_CUDA_CHECK(cudaGetLastError());
  thrust::exclusive_scan(mem.policy(), keep.get(), keep.get() + numNodes + 1,
                         candidateIndex.get());

  // The candidate count is needed on the host to size the output buffers
  int hostCounters[2] = {};
  unsigned int nCandidates = 0;
  mem.toHost(hostCounters, counters.get(), 2);
  mem.toHost(&nCandidates, candidateIndex.get() + numNodes, 1);
  mem.synchronize();
  if (m_cfg.doJunctionRemoval) {
    ACTS_DEBUG("Edges after junction removal: " << hostCounters[0]);
  }
  ACTS_DEBUG("Labels: " << hostCounters[1]
                        << ", track candidates: " << nCandidates);
  result.nCandidates = nCandidates;
  if (nCandidates == 0) {
    return {};
  }

  traccc::edm::seed_collection::buffer seeds(nCandidates, *m_temporaryMr);
  copy().setup(seeds)->ignore();
  // At most two measurements per space point
  result.candidates.tracks = {
      std::vector<std::size_t>(nCandidates, 2 * m_cfg.maxSpacePoints),
      mr().main, mr().host, vecmem::data::buffer_type::resizable};
  copy().setup(result.candidates.tracks)->ignore();
  fillCandidates<kMaxCandidateSize>
      <<<nBlocks(numNodes), kBlock, 0, cudaStream>>>(
          bounds.get(), sortedSpacePoints.get(), numLabels, keep.get(),
          candidateIndex.get(), spacePoints, seeds, result.candidates.tracks);
  ACTS_CUDA_CHECK(cudaGetLastError());
  return seeds;
}

}  // namespace ActsPlugins
