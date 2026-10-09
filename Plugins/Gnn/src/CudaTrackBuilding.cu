// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Utilities/Zip.hpp"
#include "ActsPlugins/Gnn/CudaTrackBuilding.hpp"
#include "ActsPlugins/Gnn/detail/ConnectedComponents.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"
#include "ActsPlugins/Gnn/detail/DeviceMemory.cuh"
#include "ActsPlugins/Gnn/detail/JunctionRemoval.hpp"

using namespace Acts;

namespace ActsPlugins {

std::vector<std::vector<int>> CudaTrackBuilding::operator()(
    PipelineTensors tensors, std::vector<int>& spacePointIds,
    const ExecutionContext& execContext) {
  ACTS_VERBOSE("Start CUDA track building");
  if (!(tensors.edgeIndex.device().isCuda() &&
        tensors.edgeScores.value().device().isCuda())) {
    throw std::runtime_error(
        "CudaTrackBuilding expects tensors to be on CUDA!");
  }

  const auto numSpacePoints = spacePointIds.size();
  auto numEdges = static_cast<std::size_t>(tensors.edgeIndex.shape().at(1));

  if (numEdges == 0) {
    ACTS_DEBUG("No edges remained after edge classification");
    return {};
  }

  detail::DeviceMemory mem(execContext);
  auto stream = mem.stream();

  auto cudaSrcPtr = tensors.edgeIndex.data();
  auto cudaTgtPtr = tensors.edgeIndex.data() + numEdges;

  // Everything below is enqueued without host synchronization; the edge and
  // label counts stay in device memory until the results are copied back.
  // counters[0]: number of edges after junction removal, [1]: number of labels
  auto cudaCountersAlloc = mem.make<int>(2);
  int* cudaCounters = cudaCountersAlloc.get();
  const int* cudaNumEdges = nullptr;

  vecmem::unique_alloc_ptr<std::int64_t[]> cudaJrEdges;
  if (m_cfg.doJunctionRemoval) {
    assert(tensors.edgeScores->shape().at(0) ==
           tensors.edgeIndex.shape().at(1));
    ACTS_DEBUG("Do junction removal...");
    cudaJrEdges = mem.make<std::int64_t>(2 * numEdges);
    detail::junctionRemovalCudaAsync(
        numEdges, numSpacePoints, tensors.edgeScores->data(), cudaSrcPtr,
        cudaTgtPtr, cudaJrEdges.get(), cudaJrEdges.get() + numEdges,
        cudaCounters, stream, execContext.memoryResource);
    cudaSrcPtr = cudaJrEdges.get();
    cudaTgtPtr = cudaJrEdges.get() + numEdges;
    cudaNumEdges = cudaCounters;
  }

  auto cudaLabelsAlloc = mem.make<int>(numSpacePoints);
  auto cudaSpacePointIdsAlloc = mem.make<int>(2 * numSpacePoints);
  auto cudaBoundsAlloc = mem.make<int>(numSpacePoints + 1);
  int* cudaLabels = cudaLabelsAlloc.get();
  int* cudaSpacePointIds = cudaSpacePointIdsAlloc.get();
  int* cudaBounds = cudaBoundsAlloc.get();
  int* cudaSortedSpacePointIds = cudaSpacePointIds + numSpacePoints;

  detail::connectedComponentsCudaAsync(numEdges, cudaNumEdges, cudaSrcPtr,
                                       cudaTgtPtr, numSpacePoints, cudaLabels,
                                       cudaCounters + 1, mem);

  // Sort the space point IDs by label and compute the bounds of each label
  mem.toDevice(cudaSpacePointIds, spacePointIds.data(), numSpacePoints);
  detail::findTrackCandidateBoundsAsync(cudaLabels, cudaSpacePointIds,
                                        cudaSortedSpacePointIds, cudaBounds,
                                        numSpacePoints, cudaCounters + 1, mem);

  int counters[2]{};
  mem.toHost(counters, cudaCounters, 2);
  mem.toHost(spacePointIds.data(), cudaSortedSpacePointIds, numSpacePoints);
  mem.synchronize();

  const auto numberLabels = static_cast<std::size_t>(counters[1]);
  if (m_cfg.doJunctionRemoval) {
    ACTS_DEBUG("Removed " << numEdges - counters[0]
                          << " edges in junction removal");
    if (counters[0] == 0) {
      ACTS_WARNING(
          "No edges remained after junction removal, this should not happen!");
    }
  }
  ACTS_VERBOSE("Found " << numberLabels << " track candidates");

  std::vector<int> bounds(numberLabels + 1);
  mem.toHost(bounds.data(), cudaBounds, numberLabels + 1);
  mem.synchronize();
  ACTS_CUDA_CHECK(cudaGetLastError());

  if (m_cfg.doJunctionRemoval && counters[0] == 0) {
    return {};
  }

  ACTS_DEBUG("Bounds size: " << bounds.size());
  if (numberLabels >= 6) {
    ACTS_DEBUG("Bounds: " << bounds.at(0) << ", " << bounds.at(1) << ", "
                          << bounds.at(2) << ", ..., "
                          << bounds.at(numberLabels - 2) << ", "
                          << bounds.at(numberLabels - 1) << ", "
                          << bounds.at(numberLabels));
  } else {
    ACTS_DEBUG("Bounds: " << [&]() {
      std::stringstream ss;
      ss << bounds.at(0);
      for (std::size_t i = 1; i <= numberLabels; ++i) {
        ss << ", " << bounds.at(i);
      }
      return ss.str();
    }());
  }

  std::vector<std::vector<int>> trackCandidates;
  trackCandidates.reserve(numberLabels);
  for (std::size_t label = 0ul; label < numberLabels; ++label) {
    int start = bounds.at(label);
    int end = bounds.at(label + 1);

    assert(start >= 0);
    assert(end <= static_cast<int>(numSpacePoints));
    assert(start <= end);

    if (end - start < m_cfg.minCandidateSize) {
      continue;
    }

    trackCandidates.emplace_back(spacePointIds.begin() + start,
                                 spacePointIds.begin() + end);
  }

  return trackCandidates;
}

}  // namespace ActsPlugins
