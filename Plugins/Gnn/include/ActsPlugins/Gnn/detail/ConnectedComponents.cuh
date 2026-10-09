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

#include <cstdint>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace ActsPlugins::detail {

/// Implementation of the FastSV algorithm as shown in
/// https://arxiv.org/abs/1910.05971

/// Hooking step of the FastSV algorithm
template <typename TEdge, typename TLabel>
__device__ void hookEdgesImpl(std::size_t i, const TEdge *sourceEdges,
                              const TEdge *targetEdges, const TLabel *labels,
                              TLabel *labelsNext, bool &changed) {
  auto u = sourceEdges[i];
  auto v = targetEdges[i];

  if (labels[u] == labels[labels[u]] && labels[v] < labels[u]) {
    atomicMin(labelsNext + labels[u], labels[v]);
    changed = true;
    //printf("Edge (%i,%i): set labelsNext[%i] = labels[%i] = %i\n", u, v, labels[u], v, labels[v]);
  } else if (labels[v] == labels[labels[v]] && labels[u] < labels[v]) {
    atomicMin(labelsNext + labels[v], labels[u]);
    changed = true;
    //printf("Edge (%i,%i): set labelsNext[%i] = labels[%i] = %i\n", u, v, labels[v], u, labels[u]);
  } else {
    //printf("Edge (%i,%i): no action\n", u, v);
  }
}

/// Shortcutting step of the FastSV algorithm
template <typename TEdge, typename TLabel>
__device__ void shortcutImpl(std::size_t i, const TEdge *sourceEdges,
                             const TEdge *targetEdges, const TLabel *labels,
                             TLabel *labelsNext, bool &changed) {
  if (labels[i] != labels[labels[i]]) {
    labelsNext[i] = labels[labels[i]];
    //printf("Vertex %i: labelsNext[%i] = labels[%i] = %i\n", i, i, labels[i], labels[labels[i]]);
    changed = true;
  }
}

/// Implementation of the FastSV algorithm in a single kernel
/// NOTE: This can only run in one block due to synchronization
template <typename TEdge, typename TLabel>
__global__ void labelConnectedComponents(std::size_t numEdges,
                                         const TEdge *sourceEdges,
                                         const TEdge *targetEdges,
                                         std::size_t numNodes, TLabel *labels,
                                         TLabel *labelsNext) {
  // Currently this kernel works only with 1 block
  assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);

  for (std::size_t i = threadIdx.x; i < numNodes; i += blockDim.x) {
    labels[i] = i;
    labelsNext[i] = i;
  }

  __syncthreads();
  bool changed = false;

  do {
    changed = false;

    //printf("Iteration %i\n", n);

    // Tree hooking for each edge;
    for (std::size_t i = threadIdx.x; i < numEdges; i += blockDim.x) {
      hookEdgesImpl(i, sourceEdges, targetEdges, labels, labelsNext, changed);
    }
    __syncthreads();

    for (std::size_t i = threadIdx.x; i < numNodes; i += blockDim.x) {
      labels[i] = labelsNext[i];
    }

    /*if(threadIdx.x == 0 ) {
      for(int i=0; i<numNodes; ++i) {
        printf("Vertex %i - label %i\n", i, labels[i]);
      }
    }*/

    // Shortcutting
    for (std::size_t i = threadIdx.x; i < numNodes; i += blockDim.x) {
      shortcutImpl(i, sourceEdges, targetEdges, labels, labelsNext, changed);
    }

    for (std::size_t i = threadIdx.x; i < numNodes; i += blockDim.x) {
      labels[i] = labelsNext[i];
    }

    /*if(threadIdx.x == 0 ) {
      for(int i=0; i<numNodes; ++i) {
        printf("Vertex after Shortcutting %i - label %i\n", i, labels[i]);
      }
    }*/

  } while (__syncthreads_or(changed));
}

/// Hooking-kernel for implementing the FastSV loop on the host
template <typename TEdge, typename TLabel>
__global__ void hookEdges(std::size_t numEdges, const TEdge *sourceEdges,
                          const TEdge *targetEdges, const TLabel *labels,
                          TLabel *labelsNext, int *globalChanged) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  bool changed = false;
  if (i < numEdges) {
    hookEdgesImpl(i, sourceEdges, targetEdges, labels, labelsNext, changed);
  }

  if (__syncthreads_or(changed) && threadIdx.x == 0) {
    *globalChanged = true;
  }
}

/// Shortcutting-kernel for implementing the FastSV loop on the host
template <typename TEdge, typename TLabel>
__global__ void shortcut(std::size_t numNodes, const TEdge *sourceEdges,
                         const TEdge *targetEdges, const TLabel *labels,
                         TLabel *labelsNext, int *globalChanged) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  bool changed = false;
  if (i < numNodes) {
    shortcutImpl(i, sourceEdges, targetEdges, labels, labelsNext, changed);
  }

  if (__syncthreads_or(changed) && threadIdx.x == 0) {
    *globalChanged = true;
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

/// Union-find connected components, following ECL-CC
/// (https://doi.org/10.1145/3208040.3208041). A root is only ever hooked to a
/// smaller root, so parent[v] <= v holds at all times and every component ends
/// up labelled by its smallest node index, the labelling FastSV converges to.
/// Unlike the single-block FastSV kernel this runs on the whole device without
/// any host-side iteration.
template <typename TLabel>
__device__ TLabel findRoot(TLabel *parent, TLabel node) {
  volatile TLabel *vparent = parent;
  TLabel curr = vparent[node];
  if (curr != node) {
    // Path halving: only ever redirects to an ancestor
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
    // Hook the larger root u to the smaller root v, if u is still a root
    TLabel old = atomicCAS(&parent[u], u, v);
    if (old == u) {
      break;
    }
    u = findRoot(parent, old);
    v = findRoot(parent, v);
  }
}

/// Point every node directly to its root. This must not use the path halving
/// of findRoot: a halving write by another thread could overwrite an already
/// flattened entry with a non-root ancestor. The forest is static here, so a
/// read-only walk suffices and each thread only writes its own entry.
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
                                  cudaStream_t stream) {
  const dim3 blockDim = 256;
  const dim3 gridDimNodes = (nNodes + blockDim.x - 1) / blockDim.x;

  if (nNodes == 0) {
    ACTS_CUDA_CHECK(cudaMemsetAsync(numLabels, 0, sizeof(TLabel), stream));
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
  TLabel *mask{}, *prefixSum{};
  ACTS_CUDA_CHECK(
      cudaMallocAsync(&mask, 2 * (nNodes + 1) * sizeof(TLabel), stream));
  prefixSum = mask + nNodes + 1;
  ACTS_CUDA_CHECK(
      cudaMemsetAsync(mask, 0, (nNodes + 1) * sizeof(TLabel), stream));
  makeLabelMask<<<gridDimNodes, blockDim, 0, stream>>>(nNodes, labels, mask);
  ACTS_CUDA_CHECK(cudaGetLastError());

  std::size_t tempBytes = 0;
  ACTS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, mask,
                                                prefixSum, nNodes + 1, stream));
  void *temp{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&temp, tempBytes, stream));
  ACTS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, tempBytes, mask,
                                                prefixSum, nNodes + 1, stream));

  mapEdgeLabels<<<gridDimNodes, blockDim, 0, stream>>>(nNodes, labels,
                                                       prefixSum);
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_CUDA_CHECK(cudaMemcpyAsync(numLabels, prefixSum + nNodes, sizeof(TLabel),
                                  cudaMemcpyDeviceToDevice, stream));

  ACTS_CUDA_CHECK(cudaFreeAsync(temp, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(mask, stream));
}

/// Connected components, returns the number of labels (synchronizes).
/// useOneCudaBlock is kept for interface compatibility and has no effect, the
/// union-find implementation always uses the whole device.
template <typename TEdges, typename TLabel>
TLabel connectedComponentsCuda(std::size_t nEdges, const TEdges *sourceEdges,
                               const TEdges *targetEdges, std::size_t nNodes,
                               TLabel *labels, cudaStream_t stream,
                               bool useOneCudaBlock = true) {
  static_cast<void>(useOneCudaBlock);
  TLabel *cudaNumLabels{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&cudaNumLabels, sizeof(TLabel), stream));
  connectedComponentsCudaAsync(nEdges, static_cast<const int *>(nullptr),
                               sourceEdges, targetEdges, nNodes, labels,
                               cudaNumLabels, stream);
  TLabel nLabels{};
  ACTS_CUDA_CHECK(cudaMemcpyAsync(&nLabels, cudaNumLabels, sizeof(TLabel),
                                  cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(cudaNumLabels, stream));
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  return nLabels;
}

/// Kernel to compute the bounds for each label in the labels array.
/// assume we have labels
/// 0, 0, 0, 1, 1, 2, 2, 2, 2
/// subtract the previous label
/// 0, 0, 0, 1, 0, 1, 0, 0, 0
/// set the bounds
/// 0, 3, 5
template <typename TLabel>
__global__ void setBounds(const TLabel *labels, TLabel *bounds,
                          std::size_t size, std::size_t numLabels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  } else if (idx == 0) {
    bounds[0] = 0;
    return;
  }

  if (idx == size - 1) {
    bounds[numLabels] = size;
  }

  auto diff = labels[idx] - labels[idx - 1];
  if (diff != 0) {
    //printf("idx %d, diff %d, labels[idx] %d\n", idx, diff, labels[idx]);
    bounds[labels[idx]] = idx;
  }
}

/// Function to find the bounds for each label in the labels array.
/// @param labels The array of labels (size: numSpacePoints)
/// @param spacePointIds The array of space point IDs (size: numSpacePoints)
/// @param bounds The array to store the bounds for each label (size: numLabels)
/// @param numSpacePoints The number of space points
/// @param numLabels The number of unique labels
/// @param stream The CUDA stream to use for the operation
template <typename TLabel, typename TSpacePointId>
void findTrackCandidateBounds(TLabel *labels, TSpacePointId *spacePointIds,
                              TLabel *bounds, std::size_t numSpacePoints,
                              std::size_t numLabels, cudaStream_t stream) {
  // Sort the labels and space point IDs by labels
  thrust::sort_by_key(thrust::device.on(stream), labels,
                      labels + numSpacePoints, spacePointIds);

  // Set the bounds for each label
  dim3 blockSize = 1024;
  dim3 gridSize = (numSpacePoints + blockSize.x - 1) / blockSize.x;
  setBounds<<<gridSize, blockSize, 0, stream>>>(labels, bounds, numSpacePoints,
                                                numLabels);
  ACTS_CUDA_CHECK(cudaGetLastError());
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

/// Asynchronous version of findTrackCandidateBounds: stable-sorts the space
/// point IDs by label into sortedSpacePointIds and writes the bounds of each
/// label (size numLabels + 1, at most numSpacePoints + 1). The number of
/// labels is read from device memory.
template <typename TLabel, typename TSpacePointId>
void findTrackCandidateBoundsAsync(const TLabel *labels,
                                   const TSpacePointId *spacePointIds,
                                   TSpacePointId *sortedSpacePointIds,
                                   TLabel *bounds, std::size_t numSpacePoints,
                                   const TLabel *numLabels,
                                   cudaStream_t stream) {
  if (numSpacePoints == 0) {
    return;
  }
  int endBit = 1;
  while (endBit < 31 && (std::size_t{1} << endBit) < numSpacePoints) {
    ++endBit;
  }

  TLabel *sortedLabels{};
  ACTS_CUDA_CHECK(
      cudaMallocAsync(&sortedLabels, numSpacePoints * sizeof(TLabel), stream));
  std::size_t tempBytes = 0;
  ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, tempBytes, labels, sortedLabels, spacePointIds,
      sortedSpacePointIds, numSpacePoints, 0, endBit, stream));
  void *temp{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&temp, tempBytes, stream));
  ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp, tempBytes, labels, sortedLabels, spacePointIds, sortedSpacePointIds,
      numSpacePoints, 0, endBit, stream));

  const dim3 blockDim = 256;
  const dim3 gridDim = (numSpacePoints + 1 + blockDim.x - 1) / blockDim.x;
  labelBounds<<<gridDim, blockDim, 0, stream>>>(sortedLabels, numSpacePoints,
                                                numLabels, bounds);
  ACTS_CUDA_CHECK(cudaGetLastError());

  ACTS_CUDA_CHECK(cudaFreeAsync(temp, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(sortedLabels, stream));
}

}  // namespace ActsPlugins::detail
