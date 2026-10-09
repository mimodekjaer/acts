// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsPlugins/Gnn/detail/CudaUtils.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"
#include "ActsPlugins/Gnn/detail/DeviceMemory.cuh"

#include <cstdint>

#include <cub/device/device_radix_sort.cuh>
#include <thrust/scan.h>

namespace ActsPlugins::detail {

/// Connected components with a lock-free union-find, following ECL-CC
/// (https://doi.org/10.1145/3208040.3208041).
///
/// Every node starts as the root of its own tree (parent[v] == v). Each edge
/// hooks the root with the larger index to the root with the smaller index,
/// so parent[v] <= v holds at all times, the trees cannot form cycles, and
/// every component ends up labelled by its smallest node index. The whole
/// algorithm is three kernels without any host-side iteration.

/// Root of the tree that @p node belongs to, with path halving: every second
/// node on the way up is redirected to its grandparent. Since a parent is
/// always an ancestor with a smaller index, concurrent hooks and halvings
/// can only shorten paths, never break them.
template <typename TLabel>
__device__ TLabel findRoot(TLabel *parent, TLabel node) {
  volatile TLabel *vparent = parent;
  TLabel curr = vparent[node];
  if (curr != node) {
    TLabel prev = node;
    TLabel next;
    while (curr > (next = vparent[curr])) {
      vparent[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

/// Union the endpoints of each edge. If numEdges is not null the number of
/// edges is read from device memory, maxEdges is then an upper bound.
template <typename TEdge, typename TLabel>
__global__ void unionFindHook(std::size_t maxEdges, const int *numEdges,
                              const TEdge *sourceEdges,
                              const TEdge *targetEdges, TLabel *parent) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t n = numEdges != nullptr ? *numEdges : maxEdges;
  if (i >= n) {
    return;
  }

  TLabel u = findRoot(parent, static_cast<TLabel>(sourceEdges[i]));
  TLabel v = findRoot(parent, static_cast<TLabel>(targetEdges[i]));
  while (u != v) {
    if (u < v) {
      TLabel tmp = u;
      u = v;
      v = tmp;
    }
    // Hook the larger root u to the smaller root v, if u is still a root.
    // Otherwise another thread hooked u in the meantime: retry from the
    // current roots.
    TLabel old = atomicCAS(&parent[u], u, v);
    if (old == u) {
      break;
    }
    u = findRoot(parent, old);
    v = findRoot(parent, v);
  }
}

/// Point every node directly to its root. The forest is static here, so a
/// read-only walk suffices and each thread only writes its own entry. (Path
/// halving by another thread could overwrite an already flattened entry with
/// a non-root ancestor, so findRoot must not be used.)
template <typename TLabel>
__global__ void unionFindFlatten(std::size_t numNodes, TLabel *parent) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numNodes) {
    return;
  }
  volatile TLabel *vparent = parent;
  const TLabel start = vparent[i];
  TLabel root = start;
  TLabel next;
  while (root > (next = vparent[root])) {
    root = next;
  }
  if (root != start) {
    vparent[i] = root;
  }
}

template <typename T>
__global__ void makeLabelMask(std::size_t nLabels, const T *labels,
                              T *labelMask) {
  std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i >= nLabels) {
    return;
  }

  labelMask[labels[i]] = 1;
}

template <typename T>
__global__ void mapEdgeLabels(std::size_t nLabels, T *labels,
                              const T *mapping) {
  std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i >= nLabels) {
    return;
  }

  labels[i] = mapping[labels[i]];
}

/// Connected components without host synchronization. Writes labels in
/// [0, numLabels) ordered by the smallest node index of each component, and
/// the number of labels to numLabels (device memory). If numEdges is not null
/// the number of edges is read from device memory and nEdges is an upper
/// bound.
template <typename TEdges, typename TLabel>
void connectedComponentsCudaAsync(std::size_t nEdges, const int *numEdges,
                                  const TEdges *sourceEdges,
                                  const TEdges *targetEdges, std::size_t nNodes,
                                  TLabel *labels, TLabel *numLabels,
                                  DeviceMemory &mem) {
  const cudaStream_t stream = mem.stream();
  const dim3 blockDim = 256;
  const dim3 gridDimNodes = (nNodes + blockDim.x - 1) / blockDim.x;

  if (nNodes == 0) {
    mem.memset(numLabels, 1, 0);
    return;
  }

  detail::iota<<<gridDimNodes, blockDim, 0, stream>>>(nNodes, labels);
  ACTS_CUDA_CHECK(cudaGetLastError());
  if (nEdges > 0) {
    const dim3 gridDimEdges = (nEdges + blockDim.x - 1) / blockDim.x;
    unionFindHook<<<gridDimEdges, blockDim, 0, stream>>>(
        nEdges, numEdges, sourceEdges, targetEdges, labels);
    ACTS_CUDA_CHECK(cudaGetLastError());
  }
  unionFindFlatten<<<gridDimNodes, blockDim, 0, stream>>>(nNodes, labels);
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Relabel to consecutive labels, e.g. for components 0 3 5 3 0 0:
  // mask 1 0 0 1 0 1 (0), exclusive sum 0 1 1 1 2 2 (3), labels 0 1 2 1 0 0.
  // The extra last element of the sum is the number of labels.
  auto maskBuffer = mem.make<TLabel>(2 * (nNodes + 1));
  TLabel *mask = maskBuffer.get();
  TLabel *prefixSum = mask + nNodes + 1;
  mem.memset(mask, nNodes + 1, 0);
  makeLabelMask<<<gridDimNodes, blockDim, 0, stream>>>(nNodes, labels, mask);
  ACTS_CUDA_CHECK(cudaGetLastError());

  thrust::exclusive_scan(mem.policy(), mask, mask + nNodes + 1, prefixSum);

  mapEdgeLabels<<<gridDimNodes, blockDim, 0, stream>>>(nNodes, labels,
                                                       prefixSum);
  ACTS_CUDA_CHECK(cudaGetLastError());
  mem.copyDevice(numLabels, prefixSum + nNodes, 1);
}

/// Connected components, returns the number of labels (synchronizes).
template <typename TEdges, typename TLabel>
TLabel connectedComponentsCuda(std::size_t nEdges, const TEdges *sourceEdges,
                               const TEdges *targetEdges, std::size_t nNodes,
                               TLabel *labels, DeviceMemory &mem) {
  auto cudaNumLabels = mem.make<TLabel>(1);
  connectedComponentsCudaAsync(nEdges, static_cast<const int *>(nullptr),
                               sourceEdges, targetEdges, nNodes, labels,
                               cudaNumLabels.get(), mem);
  TLabel nLabels{};
  mem.toHost(&nLabels, cudaNumLabels.get(), 1);
  mem.synchronize();
  return nLabels;
}

/// bounds[l] = first position of label l in the sorted labels, for
/// l in [0, numLabels]
template <typename TLabel>
__global__ void labelBounds(const TLabel *sortedLabels, std::size_t size,
                            const TLabel *numLabels, TLabel *bounds) {
  const std::size_t l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l > static_cast<std::size_t>(*numLabels)) {
    return;
  }
  std::size_t lo = 0;
  std::size_t hi = size;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (static_cast<std::size_t>(sortedLabels[mid]) < l) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  bounds[l] = lo;
}

/// Groups the space points by label without host synchronization: stable
/// sort of the space point IDs by label into sortedSpacePointIds, and the
/// bounds of each label (bounds[l] .. bounds[l + 1], size numLabels + 1, at
/// most numSpacePoints + 1). The number of labels is read from device memory.
template <typename TLabel, typename TSpacePointId>
void findTrackCandidateBoundsAsync(const TLabel *labels,
                                   const TSpacePointId *spacePointIds,
                                   TSpacePointId *sortedSpacePointIds,
                                   TLabel *bounds, std::size_t numSpacePoints,
                                   const TLabel *numLabels, DeviceMemory &mem) {
  if (numSpacePoints == 0) {
    return;
  }
  const cudaStream_t stream = mem.stream();
  // There are at most numSpacePoints labels, so only the lower bits need to
  // be sorted. thrust always sorts all bits, so CUB is used directly here.
  int endBit = 1;
  while (endBit < 31 && (std::size_t{1} << endBit) < numSpacePoints) {
    ++endBit;
  }

  auto sortedLabels = mem.make<TLabel>(numSpacePoints);
  std::size_t tempBytes = 0;
  ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, tempBytes, labels, sortedLabels.get(), spacePointIds,
      sortedSpacePointIds, numSpacePoints, 0, endBit, stream));
  auto temp = mem.make<std::byte>(tempBytes);
  ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp.get(), tempBytes, labels, sortedLabels.get(), spacePointIds,
      sortedSpacePointIds, numSpacePoints, 0, endBit, stream));

  const dim3 blockDim = 256;
  const dim3 gridDim = (numSpacePoints + 1 + blockDim.x - 1) / blockDim.x;
  labelBounds<<<gridDim, blockDim, 0, stream>>>(
      sortedLabels.get(), numSpacePoints, numLabels, bounds);
  ACTS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ActsPlugins::detail
