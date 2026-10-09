// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cstdint>
#include <memory_resource>

#include <cuda_runtime_api.h>

namespace ActsPlugins::detail {

/// Remove all but the highest-scoring edge at each junction (several
/// incoming or several outgoing edges at a node; ties go to the smallest edge
/// index), without host synchronization. The kept edges are written in their
/// original order to srcNodesOut / dstNodesOut (capacity nEdges each), their
/// number to numEdgesOut (device memory). Temporary device memory is taken
/// from @p mr, or is stream-ordered if it is null.
void junctionRemovalCudaAsync(std::size_t nEdges, std::size_t nNodes,
                              const float *scores, const std::int64_t *srcNodes,
                              const std::int64_t *dstNodes,
                              std::int64_t *srcNodesOut,
                              std::int64_t *dstNodesOut, int *numEdgesOut,
                              cudaStream_t stream,
                              std::pmr::memory_resource *mr = nullptr);

}  // namespace ActsPlugins::detail
