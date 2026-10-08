// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Custom TensorRT plugins replacing the torch_scatter ScatterElements(add)
// aggregation of the InteractionGNN edge classifier. The exported ONNX graph
// transposes the [E,C] messages, tiles the index C times and scatters element
// by element with fp16 atomics, which is slow under contention and not
// deterministic. Here the edges are grouped by node once per event
// (ActsGnnBuildCsr) and every aggregation is a warp-per-node sum in a fixed
// order with fp32 accumulation (ActsGnnSegmentSum).
//
// Compiled into ActsPluginGnn (registered via registerTensorRTGnnPlugins) and,
// with ACTS_GNN_TRT_PLUGIN_LIBRARY, as a standalone library for
// trtexec --dynamicPlugins.

#include "ActsPlugins/Gnn/detail/TensorRTGnnPlugins.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include <NvInfer.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ActsPlugins::detail {

namespace {

using namespace nvinfer1;

constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kBuildCsrName[] = "ActsGnnBuildCsr";
constexpr char kSegmentSumName[] = "ActsGnnSegmentSum";

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr std::size_t kAlignment = 256;

constexpr std::size_t alignUp(std::size_t n) {
  return (n + kAlignment - 1) / kAlignment * kAlignment;
}

int numBlocks(std::int64_t nThreads) {
  return static_cast<int>((nThreads + kBlockSize - 1) / kBlockSize);
}

// Number of key bits needed to represent node indices in [0, nNodes)
int keyBits(std::int64_t nNodes) {
  int bits = 1;
  while (bits < 32 && (std::int64_t{1} << bits) < nNodes) {
    ++bits;
  }
  return bits;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void iotaKernel(int *values, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    values[i] = i;
  }
}

// offsets[n] = first position in sortedKeys with key >= n, for n in [0, N]
__global__ void segmentOffsetsKernel(const unsigned *__restrict__ sortedKeys,
                                     int nEdges, int nNodes,
                                     int *__restrict__ offsets) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n > nNodes) {
    return;
  }
  int lo = 0;
  int hi = nEdges;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (sortedKeys[mid] < static_cast<unsigned>(n)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  offsets[n] = lo;
}

template <typename T>
__device__ float toFloat(T v);
template <>
__device__ float toFloat<float>(float v) {
  return v;
}
template <>
__device__ float toFloat<__half>(__half v) {
  return __half2float(v);
}

template <typename T>
__device__ T fromFloat(float v);
template <>
__device__ float fromFloat<float>(float v) {
  return v;
}
template <>
__device__ __half fromFloat<__half>(float v) {
  return __float2half_rn(v);
}

// One warp per (node, direction): warps [0, N) sum the messages of edges
// arriving at the node, warps [N, 2N) the messages of edges leaving it.
// Edges are visited in increasing edge index, so the result is deterministic.
template <typename T>
__global__ void segmentSumKernel(const T *__restrict__ messages, int nChannels,
                                 const int *__restrict__ permDst,
                                 const int *__restrict__ offDst,
                                 const int *__restrict__ permSrc,
                                 const int *__restrict__ offSrc, int nNodes,
                                 T *__restrict__ aggDst,
                                 T *__restrict__ aggSrc) {
  const std::int64_t warp =
      (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (warp >= 2 * static_cast<std::int64_t>(nNodes)) {
    return;
  }

  const bool isDst = warp < nNodes;
  const int node = static_cast<int>(isDst ? warp : warp - nNodes);
  const int *perm = isDst ? permDst : permSrc;
  const int *off = isDst ? offDst : offSrc;
  T *out = isDst ? aggDst : aggSrc;

  const int begin = off[node];
  const int end = off[node + 1];
  for (int c = lane; c < nChannels; c += kWarpSize) {
    float acc = 0.f;
    for (int k = begin; k < end; ++k) {
      acc += toFloat(
          messages[static_cast<std::int64_t>(perm[k]) * nChannels + c]);
    }
    out[static_cast<std::int64_t>(node) * nChannels + c] = fromFloat<T>(acc);
  }
}

// ---------------------------------------------------------------------------
// Common plugin boilerplate
// ---------------------------------------------------------------------------

PluginFieldCollection emptyFieldCollection{0, nullptr};

class PluginBase : public IPluginV3,
                   public IPluginV3OneCore,
                   public IPluginV3OneBuild,
                   public IPluginV3OneRuntime {
 public:
  IPluginCapability *getCapabilityInterface(
      PluginCapabilityType type) noexcept override {
    switch (type) {
      case PluginCapabilityType::kBUILD:
        return static_cast<IPluginV3OneBuild *>(this);
      case PluginCapabilityType::kRUNTIME:
        return static_cast<IPluginV3OneRuntime *>(this);
      case PluginCapabilityType::kCORE:
        return static_cast<IPluginV3OneCore *>(this);
    }
    return nullptr;
  }

  const char *getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char *getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  int32_t configurePlugin(const DynamicPluginTensorDesc * /*in*/,
                          int32_t /*nbInputs*/,
                          const DynamicPluginTensorDesc * /*out*/,
                          int32_t /*nbOutputs*/) noexcept override {
    return 0;
  }

  int32_t onShapeChange(const PluginTensorDesc * /*in*/, int32_t /*nbInputs*/,
                        const PluginTensorDesc * /*out*/,
                        int32_t /*nbOutputs*/) noexcept override {
    return 0;
  }

  IPluginV3 *attachToContext(
      IPluginResourceContext * /*context*/) noexcept override {
    return clone();
  }

  const PluginFieldCollection *getFieldsToSerialize() noexcept override {
    return &emptyFieldCollection;
  }
};

bool isLinear(const DynamicPluginTensorDesc &d) {
  return d.desc.format == TensorFormat::kLINEAR;
}

bool isFloatOrHalf(const DynamicPluginTensorDesc &d) {
  return d.desc.type == DataType::kFLOAT || d.desc.type == DataType::kHALF;
}

// ---------------------------------------------------------------------------
// ActsGnnBuildCsr
// ---------------------------------------------------------------------------

class BuildCsrPlugin final : public PluginBase {
 public:
  const char *getPluginName() const noexcept override { return kBuildCsrName; }

  IPluginV3 *clone() noexcept override { return new BuildCsrPlugin(*this); }

  int32_t getNbOutputs() const noexcept override { return 4; }

  int32_t getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs,
                             const DataType * /*inputTypes*/,
                             int32_t /*nbInputs*/) const noexcept override {
    for (int32_t i = 0; i < nbOutputs; ++i) {
      outputTypes[i] = DataType::kINT32;
    }
    return 0;
  }

  int32_t getOutputShapes(const DimsExprs *inputs, int32_t nbInputs,
                          const DimsExprs * /*shapeInputs*/,
                          int32_t /*nbShapeInputs*/, DimsExprs *outputs,
                          int32_t nbOutputs,
                          IExprBuilder &exprBuilder) noexcept override {
    if (nbInputs != 2 || nbOutputs != 4) {
      return -1;
    }
    const IDimensionExpr *nEdges = inputs[0].d[1];
    const IDimensionExpr *nOffsets = exprBuilder.operation(
        DimensionOperation::kSUM, *inputs[1].d[0], *exprBuilder.constant(1));
    for (int32_t i = 0; i < 4; ++i) {
      outputs[i].nbDims = 1;
      outputs[i].d[0] = (i % 2 == 0) ? nEdges : nOffsets;
    }
    return 0;
  }

  bool supportsFormatCombination(int32_t pos,
                                 const DynamicPluginTensorDesc *inOut,
                                 int32_t /*nbInputs*/,
                                 int32_t /*nbOutputs*/) noexcept override {
    const auto &d = inOut[pos];
    if (!isLinear(d)) {
      return false;
    }
    if (pos == 1) {
      // node_features: only its shape is used
      return isFloatOrHalf(d);
    }
    return d.desc.type == DataType::kINT32;
  }

  std::size_t getWorkspaceSize(const DynamicPluginTensorDesc *inputs,
                               int32_t /*nbInputs*/,
                               const DynamicPluginTensorDesc * /*outputs*/,
                               int32_t /*nbOutputs*/) const noexcept override {
    return workspaceSize(inputs[0].max.d[1], 32);
  }

  int32_t enqueue(const PluginTensorDesc *inputDesc,
                  const PluginTensorDesc * /*outputDesc*/,
                  const void *const *inputs, void *const *outputs,
                  void *workspace, cudaStream_t stream) noexcept override {
    const auto nEdges = static_cast<int>(inputDesc[0].dims.d[1]);
    const auto nNodes = static_cast<int>(inputDesc[1].dims.d[0]);
    const auto *edgeList = static_cast<const unsigned *>(inputs[0]);

    std::size_t cubBytes = cubTempBytes(nEdges, keyBits(nNodes));
    auto *ws = static_cast<char *>(workspace);
    auto *sortedKeys = reinterpret_cast<unsigned *>(ws);
    auto *edgeIds = reinterpret_cast<int *>(ws + alignUp(nEdges * 4ul));
    void *cubTemp = ws + 2 * alignUp(nEdges * 4ul);

    if (nEdges > 0) {
      iotaKernel<<<numBlocks(nEdges), kBlockSize, 0, stream>>>(edgeIds,
                                                               nEdges);
    }

    // Output 0/1: grouped by target node (edge_list row 1)
    // Output 2/3: grouped by source node (edge_list row 0)
    for (int dir = 0; dir < 2; ++dir) {
      const unsigned *keys = edgeList + (dir == 0 ? 1 : 0) * nEdges;
      auto *perm = static_cast<int *>(outputs[2 * dir]);
      auto *offsets = static_cast<int *>(outputs[2 * dir + 1]);
      if (nEdges > 0) {
        // Stable radix sort: edges of a node keep increasing edge index
        if (cub::DeviceRadixSort::SortPairs(cubTemp, cubBytes, keys,
                                            sortedKeys, edgeIds, perm, nEdges,
                                            0, keyBits(nNodes),
                                            stream) != cudaSuccess) {
          return -1;
        }
      }
      segmentOffsetsKernel<<<numBlocks(nNodes + 1), kBlockSize, 0, stream>>>(
          sortedKeys, nEdges, nNodes, offsets);
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }

 private:
  static std::size_t cubTempBytes(std::int64_t nEdges, int bits) {
    std::size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, static_cast<const unsigned *>(nullptr),
        static_cast<unsigned *>(nullptr), static_cast<const int *>(nullptr),
        static_cast<int *>(nullptr), static_cast<int>(nEdges), 0, bits);
    return bytes;
  }

  static std::size_t workspaceSize(std::int64_t nEdges, int bits) {
    return 2 * alignUp(nEdges * 4ul) + alignUp(cubTempBytes(nEdges, bits));
  }
};

// ---------------------------------------------------------------------------
// ActsGnnSegmentSum
// ---------------------------------------------------------------------------

class SegmentSumPlugin final : public PluginBase {
 public:
  const char *getPluginName() const noexcept override {
    return kSegmentSumName;
  }

  IPluginV3 *clone() noexcept override { return new SegmentSumPlugin(*this); }

  int32_t getNbOutputs() const noexcept override { return 2; }

  int32_t getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs,
                             const DataType *inputTypes,
                             int32_t /*nbInputs*/) const noexcept override {
    for (int32_t i = 0; i < nbOutputs; ++i) {
      outputTypes[i] = inputTypes[0];
    }
    return 0;
  }

  int32_t getOutputShapes(const DimsExprs *inputs, int32_t nbInputs,
                          const DimsExprs * /*shapeInputs*/,
                          int32_t /*nbShapeInputs*/, DimsExprs *outputs,
                          int32_t nbOutputs,
                          IExprBuilder &exprBuilder) noexcept override {
    if (nbInputs != 5 || nbOutputs != 2) {
      return -1;
    }
    const IDimensionExpr *nNodes = exprBuilder.operation(
        DimensionOperation::kSUB, *inputs[2].d[0], *exprBuilder.constant(1));
    for (int32_t i = 0; i < 2; ++i) {
      outputs[i].nbDims = 2;
      outputs[i].d[0] = nNodes;
      outputs[i].d[1] = inputs[0].d[1];
    }
    return 0;
  }

  bool supportsFormatCombination(int32_t pos,
                                 const DynamicPluginTensorDesc *inOut,
                                 int32_t /*nbInputs*/,
                                 int32_t /*nbOutputs*/) noexcept override {
    const auto &d = inOut[pos];
    if (!isLinear(d)) {
      return false;
    }
    if (pos == 0) {
      return isFloatOrHalf(d);
    }
    if (pos <= 4) {
      return d.desc.type == DataType::kINT32;
    }
    return d.desc.type == inOut[0].desc.type;
  }

  int32_t enqueue(const PluginTensorDesc *inputDesc,
                  const PluginTensorDesc * /*outputDesc*/,
                  const void *const *inputs, void *const *outputs,
                  void * /*workspace*/, cudaStream_t stream) noexcept override {
    const auto nChannels = static_cast<int>(inputDesc[0].dims.d[1]);
    const auto nNodes = static_cast<int>(inputDesc[2].dims.d[0] - 1);
    if (nNodes <= 0) {
      return 0;
    }
    const auto *permDst = static_cast<const int *>(inputs[1]);
    const auto *offDst = static_cast<const int *>(inputs[2]);
    const auto *permSrc = static_cast<const int *>(inputs[3]);
    const auto *offSrc = static_cast<const int *>(inputs[4]);
    const int blocks =
        numBlocks(2 * static_cast<std::int64_t>(nNodes) * kWarpSize);

    if (inputDesc[0].type == DataType::kHALF) {
      segmentSumKernel<__half><<<blocks, kBlockSize, 0, stream>>>(
          static_cast<const __half *>(inputs[0]), nChannels, permDst, offDst,
          permSrc, offSrc, nNodes, static_cast<__half *>(outputs[0]),
          static_cast<__half *>(outputs[1]));
    } else {
      segmentSumKernel<float><<<blocks, kBlockSize, 0, stream>>>(
          static_cast<const float *>(inputs[0]), nChannels, permDst, offDst,
          permSrc, offSrc, nNodes, static_cast<float *>(outputs[0]),
          static_cast<float *>(outputs[1]));
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }
};

// ---------------------------------------------------------------------------
// Creators
// ---------------------------------------------------------------------------

template <typename Plugin, const char *Name>
class Creator final : public IPluginCreatorV3One {
 public:
  IPluginV3 *createPlugin(const char * /*name*/,
                          const PluginFieldCollection * /*fc*/,
                          TensorRTPhase /*phase*/) noexcept override {
    return new Plugin();
  }
  const PluginFieldCollection *getFieldNames() noexcept override {
    return &emptyFieldCollection;
  }
  const char *getPluginName() const noexcept override { return Name; }
  const char *getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char *getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }
};

Creator<BuildCsrPlugin, kBuildCsrName> buildCsrCreator;
Creator<SegmentSumPlugin, kSegmentSumName> segmentSumCreator;

IPluginCreatorInterface *const allCreators[] = {&buildCsrCreator,
                                                &segmentSumCreator};

}  // namespace

void registerTensorRTGnnPlugins() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    for (auto *creator : allCreators) {
      getPluginRegistry()->registerCreator(*creator, kPluginNamespace);
    }
  });
}

}  // namespace ActsPlugins::detail

#ifdef ACTS_GNN_TRT_PLUGIN_LIBRARY
// Entry points used by IPluginRegistry::loadLibrary (trtexec --dynamicPlugins).
// The plugins do not log, so the logger finder is not needed.
extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder * /*finder*/) {}

extern "C" nvinfer1::IPluginCreatorInterface *const *getCreators(
    std::int32_t &nbCreators) {
  nbCreators = 2;
  return ActsPlugins::detail::allCreators;
}
#endif
