// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/detail/CudaUtils.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"
#include "ActsPlugins/Gnn/detail/DeviceMemory.cuh"
#include "ActsPlugins/Gnn/detail/JunctionRemoval.hpp"

#include <cub/device/device_select.cuh>

namespace ActsPlugins::detail {

namespace {

using Key = unsigned long long;

// Ordering key of an edge at a junction: the highest score wins, ties go to
// the smallest edge index. Scores are positive (sigmoid outputs), so their
// bit patterns order like the values.
__device__ Key edgeKey(const float *scores, std::size_t i) {
  return (static_cast<Key>(__float_as_uint(scores[i])) << 32) |
         (0xffffffffu - static_cast<unsigned>(i));
}

__global__ void countAndMaxJunctionEdges(std::size_t nEdges,
                                         const float *scores,
                                         const std::int64_t *srcNodes,
                                         const std::int64_t *dstNodes,
                                         int *numInEdges, int *numOutEdges,
                                         Key *maxInKey, Key *maxOutKey) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nEdges) {
    return;
  }

  const auto srcNode = srcNodes[i];
  const auto dstNode = dstNodes[i];
  const Key key = edgeKey(scores, i);

  atomicAdd(&numInEdges[dstNode], 1);
  atomicAdd(&numOutEdges[srcNode], 1);
  atomicMax(&maxInKey[dstNode], key);
  atomicMax(&maxOutKey[srcNode], key);
}

// An edge is removed if it is an incoming edge of a node with several incoming
// edges, or an outgoing edge of a node with several outgoing edges, and it is
// not the best edge there.
__global__ void fillKeepMask(std::size_t nEdges, const float *scores,
                             const std::int64_t *srcNodes,
                             const std::int64_t *dstNodes,
                             const int *numInEdges, const int *numOutEdges,
                             const Key *maxInKey, const Key *maxOutKey,
                             char *keep) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nEdges) {
    return;
  }

  const auto srcNode = srcNodes[i];
  const auto dstNode = dstNodes[i];
  const Key key = edgeKey(scores, i);

  const bool removeIn = numInEdges[dstNode] >= 2 && maxInKey[dstNode] != key;
  const bool removeOut = numOutEdges[srcNode] >= 2 && maxOutKey[srcNode] != key;
  keep[i] = !(removeIn || removeOut);
}

}  // namespace

void junctionRemovalCudaAsync(std::size_t nEdges, std::size_t nNodes,
                              const float *scores, const std::int64_t *srcNodes,
                              const std::int64_t *dstNodes,
                              std::int64_t *srcNodesOut,
                              std::int64_t *dstNodesOut, int *numEdgesOut,
                              cudaStream_t stream,
                              std::pmr::memory_resource *mr) {
  DeviceMemory mem(stream, mr);
  if (nEdges == 0) {
    mem.memset(numEdgesOut, 1, 0);
    return;
  }

  // Per node: number of incoming and outgoing edges and the best edge of each
  auto counts = mem.make<int>(2 * nNodes);
  auto keys = mem.make<Key>(2 * nNodes);
  auto keepAlloc = mem.make<char>(nEdges);
  int *numInEdges = counts.get();
  int *numOutEdges = counts.get() + nNodes;
  Key *maxInKey = keys.get();
  Key *maxOutKey = keys.get() + nNodes;
  char *keep = keepAlloc.get();
  mem.memset(counts.get(), 2 * nNodes, 0);
  mem.memset(keys.get(), 2 * nNodes, 0);

  const dim3 blockSize = 256;
  const dim3 gridSizeEdges = (nEdges + blockSize.x - 1) / blockSize.x;
  countAndMaxJunctionEdges<<<gridSizeEdges, blockSize, 0, stream>>>(
      nEdges, scores, srcNodes, dstNodes, numInEdges, numOutEdges, maxInKey,
      maxOutKey);
  ACTS_CUDA_CHECK(cudaGetLastError());
  fillKeepMask<<<gridSizeEdges, blockSize, 0, stream>>>(
      nEdges, scores, srcNodes, dstNodes, numInEdges, numOutEdges, maxInKey,
      maxOutKey, keep);
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Stable compaction of the kept edges. CUB writes the number of kept edges
  // to device memory, thrust::copy_if would need a synchronization.
  std::size_t tempBytes = 0;
  ACTS_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, tempBytes, srcNodes, keep,
                                             srcNodesOut, numEdgesOut, nEdges,
                                             stream));
  auto tempAlloc = mem.make<std::byte>(tempBytes);
  void *temp = tempAlloc.get();
  ACTS_CUDA_CHECK(cub::DeviceSelect::Flagged(temp, tempBytes, srcNodes, keep,
                                             srcNodesOut, numEdgesOut, nEdges,
                                             stream));
  ACTS_CUDA_CHECK(cub::DeviceSelect::Flagged(temp, tempBytes, dstNodes, keep,
                                             dstNodesOut, numEdgesOut, nEdges,
                                             stream));
}

}  // namespace ActsPlugins::detail
