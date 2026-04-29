// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/EdgeLayerConnector.hpp"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"

#include <MMG/CUDA_edge_layer_connector>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

using namespace Acts;

namespace {

struct MmgResult {
  int maxHitsPerTrack{};
  std::vector<int> nbHits;
  std::vector<int> flatHits;
};

MmgResult buildTracksWithMmg(int numSpacepoints, int* spacepointIDs,
                             int numEdges, int* edgeSrc, int* edgeTgt,
                             float* edgeScores,
                             const ActsPlugins::EdgeLayerConnector::Config& cfg,
                             cudaStream_t stream) {
  CUDA_edges<float> edges(numEdges, edgeSrc, edgeTgt, edgeScores, true);
  ACTS_CUDA_CHECK(cudaDeviceSynchronize());
  ACTS_CUDA_CHECK(cudaGetLastError());

  CUDA_graph<float> graph(numSpacepoints, spacepointIDs, &edges);
  ACTS_CUDA_CHECK(cudaDeviceSynchronize());
  ACTS_CUDA_CHECK(cudaGetLastError());

  CUDA_edge_layer_connector<float> connector(
      &graph, cfg.weightsCut, static_cast<int>(cfg.minHits),
      static_cast<int>(cfg.nBlocks), static_cast<int>(cfg.maxHitsPerTrack));
  ACTS_CUDA_CHECK(cudaDeviceSynchronize());
  ACTS_CUDA_CHECK(cudaGetLastError());

  connector.build_tracks();
  ACTS_CUDA_CHECK(cudaDeviceSynchronize());
  ACTS_CUDA_CHECK(cudaGetLastError());

  const int maxHitsPerTrack = connector.cuda_tracks()->max_hits_per_track();
  const int tracksSize = connector.cuda_tracks()->size();
  const int nbTracks = connector.cuda_tracks()->nb_tracks();
  ACTS_CUDA_CHECK(cudaDeviceSynchronize());
  ACTS_CUDA_CHECK(cudaGetLastError());

  std::vector<int> nbHits(nbTracks);
  ACTS_CUDA_CHECK(
      cudaMemcpyAsync(nbHits.data(), connector.cuda_tracks()->nb_hits(),
                      nbTracks * sizeof(int), cudaMemcpyDeviceToHost, stream));

  std::vector<int> flatHits(tracksSize);
  ACTS_CUDA_CHECK(cudaMemcpyAsync(
      flatHits.data(), connector.cuda_tracks()->hits(),
      tracksSize * sizeof(int), cudaMemcpyDeviceToHost, stream));

  ACTS_CUDA_CHECK(cudaDeviceSynchronize());
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  ACTS_CUDA_CHECK(cudaGetLastError());

  return {maxHitsPerTrack, std::move(nbHits), std::move(flatHits)};
}

}  // namespace

namespace ActsPlugins {

std::vector<std::vector<int>> EdgeLayerConnector::operator()(
    PipelineTensors tensors, std::vector<int>& spacepointIDs,
    const ExecutionContext& execContext) {
  ACTS_DEBUG("Get event data");

  const auto numEdges = static_cast<int>(tensors.edgeIndex.shape().at(1));
  const auto numSpacepoints = static_cast<int>(spacepointIDs.size());

  auto stream = execContext.stream.value();

  // Convert std::int64_t edge indices to int using Tensor for memory management
  auto srcInt64Ptr = tensors.edgeIndex.data();
  auto tgtInt64Ptr = tensors.edgeIndex.data() + numEdges;

  auto edgeSrc =
      Tensor<int>::Create({1, static_cast<std::size_t>(numEdges)}, execContext);
  auto edgeTgt =
      Tensor<int>::Create({1, static_cast<std::size_t>(numEdges)}, execContext);
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

  thrust::copy(thrust::cuda::par.on(stream), srcInt64Ptr,
               srcInt64Ptr + numEdges, edgeSrc.data());
  thrust::copy(thrust::cuda::par.on(stream), tgtInt64Ptr,
               tgtInt64Ptr + numEdges, edgeTgt.data());
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy spacepoint IDs to GPU
  auto spacepointIDsTensor =
      Tensor<int>::Create({1, spacepointIDs.size()}, execContext);
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_CUDA_CHECK(cudaMemcpyAsync(
      spacepointIDsTensor.data(), spacepointIDs.data(),
      spacepointIDs.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

  ACTS_DEBUG("Build tracks with MMG...");
  auto [maxHitsPerTrack, nbHits, flatHits] = buildTracksWithMmg(
      numSpacepoints, spacepointIDsTensor.data(), numEdges, edgeSrc.data(),
      edgeTgt.data(), tensors.edgeScores->data(), m_cfg, stream);

  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

  ACTS_DEBUG("maxHitsPerTrack: " << maxHitsPerTrack
                                 << ", tracksSize: " << flatHits.size()
                                 << ", nbTracks: " << nbHits.size());

  std::vector<std::vector<int>> trackCandidates;
  trackCandidates.reserve(nbHits.size());

  std::size_t minTrackSize = std::numeric_limits<std::size_t>::max();
  std::size_t maxTrackSize = 0;
  std::size_t avgTrackSize = 0;

  for (std::size_t trackIdx = 0; trackIdx < nbHits.size(); ++trackIdx) {
    const int* trackBegin = flatHits.data() + trackIdx * maxHitsPerTrack;
    const int* trackEnd = trackBegin + nbHits[trackIdx];
    trackCandidates.emplace_back(trackBegin, trackEnd);

    // Debug print the first 10 tracks
    if (trackIdx < 10) {
      ACTS_DEBUG("Track " << trackIdx << ": " << [&]() {
        std::ostringstream oss;
        for (const auto& hit : trackCandidates.back()) {
          oss << hit << " ";
        }
        return oss.str();
      }());
    }

    const std::size_t trackSize = trackCandidates.back().size();
    minTrackSize = std::min(minTrackSize, trackSize);
    maxTrackSize = std::max(maxTrackSize, trackSize);
    avgTrackSize += trackSize;
  }

  avgTrackSize /= nbHits.size();
  ACTS_DEBUG("Min/Avg/Max track size: " << minTrackSize << "/" << avgTrackSize
                                        << "/" << maxTrackSize);

  return trackCandidates;
}

}  // namespace ActsPlugins
