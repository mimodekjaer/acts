// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cstdint>
#include <utility>

#include <cuda_runtime_api.h>

namespace ActsPlugins::detail {

/// Function to perform the basic junction reomval algorithm on a graph with
/// scored edges: In case a node has more than one in/out edge, only the
/// edge with the largest score is kept.
/// NOTE: The function expects pointers on the device
/// NOTE: The function returns a pointer to device memory. The caller is
/// responsible for freeing the memory with cudaFreeAsync(ptr, stream).
/// TODO: Use some type of RAII type in the future
/// Remove all but the highest-scoring edge at each junction (several
/// incoming or several outgoing edges at a node; ties go to the smallest edge
/// index), without host synchronization. The kept edges are written in their
/// original order to srcNodesOut / dstNodesOut (capacity nEdges each), their
/// number to numEdgesOut (device memory).
void junctionRemovalCudaAsync(std::size_t nEdges, std::size_t nNodes,
                              const float *scores,
                              const std::int64_t *srcNodes,
                              const std::int64_t *dstNodes,
                              std::int64_t *srcNodesOut,
                              std::int64_t *dstNodesOut, int *numEdgesOut,
                              cudaStream_t stream);

/// Synchronous version, returns a new allocation holding
/// [src(nEdgesOut) | dst(nEdgesOut)] and nEdgesOut.
std::pair<std::int64_t *, std::size_t> junctionRemovalCuda(
    std::size_t nEdges, std::size_t nNodes, const float *scores,
    const std::int64_t *srcNodes, const std::int64_t *dstNodes,
    cudaStream_t stream);

}  // namespace ActsPlugins::detail
