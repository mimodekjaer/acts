// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/ModuleMapCuda.hpp"
#include "ActsPlugins/Gnn/detail/CudaUtils.cuh"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"
#include "ActsPlugins/Gnn/detail/DeviceMemory.cuh"
#include "ActsPlugins/Gnn/detail/ModuleMapKernels.cuh"
#include "ActsPlugins/Gnn/detail/ModuleMapUtils.cuh"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <MMG/CUDA_module_map_doublet>
#include <MMG/CUDA_module_map_triplet>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// The double-precision reference arithmetic in ModuleMapKernels.cuh matches
// the module map's data_type2
static_assert(std::is_same_v<data_type2, double>);

namespace {

struct CastBoolToInt {
  int __host__ __device__ operator()(bool b) const {
    return static_cast<int>(b);
  }
};

/// Exclusive prefix sum on the stream of @p mem, without synchronization and
/// with the temporary memory taken from its resource. In-place operation
/// (in == out) is supported.
template <typename InputIt, typename OutputIt>
void exclusiveSum(InputIt in, OutputIt out, std::size_t n,
                  ActsPlugins::detail::DeviceMemory &mem) {
  thrust::exclusive_scan(mem.policy(), in, in + n, out);
}

template <typename T>
std::string debugPrintEdges(std::size_t nbEdges, const T *cudaSrc,
                            const T *cudaDst,
                            ActsPlugins::detail::DeviceMemory &mem) {
  std::stringstream ss;
  if (nbEdges == 0) {
    return "zero edges remained";
  }
  nbEdges = std::min(10ul, nbEdges);
  std::vector<T> src(nbEdges), dst(nbEdges);
  mem.toHost(src.data(), cudaSrc, nbEdges);
  mem.toHost(dst.data(), cudaDst, nbEdges);
  mem.synchronize();
  for (std::size_t i = 0; i < nbEdges; ++i) {
    ss << src.at(i) << " ";
  }
  ss << "\n";
  for (std::size_t i = 0; i < nbEdges; ++i) {
    ss << dst.at(i) << " ";
  }
  return ss.str();
}

}  // namespace

using namespace Acts;

namespace ActsPlugins {

class ModuleMapCuda::Impl {
 public:
  std::unique_ptr<CUDA_module_map_doublet<float>> cudaModuleMapDoublet;
  std::unique_ptr<CUDA_module_map_triplet<float>> cudaModuleMapTriplet;

  /// Device memory of the module map lookup table, lives as long as the
  /// module map
  vecmem::cuda::device_memory_resource moduleMapMemory;
  vecmem::unique_alloc_ptr<std::uint64_t[]> cudaModuleMapKeys;
  vecmem::unique_alloc_ptr<int[]> cudaModuleMapVals;
  std::size_t cudaModuleMapSize{};

  /// Builds the graph for node features and module ids in device memory
  PipelineTensors buildGraph(Tensor<float> nodeFeatures,
                             const std::uint64_t *cudaModuleIds,
                             detail::DeviceMemory &mem,
                             const ExecutionContext &execContext,
                             const ModuleMapCuda::Config &cfg,
                             const Logger &logger) const;

  /// Returns the [2, nEdges] edge index and the [nEdges, 6] edge features
  std::pair<Tensor<std::int64_t>, Tensor<float>> makeEdges(
      const detail::HitArrays<float> &hits, const int *cudaHitOffsets,
      const float *cudaNodeFeatures, std::size_t nNodeFeatures,
      detail::DeviceMemory &mem, const ExecutionContext &execContext,
      const ModuleMapCuda::Config &cfg, const Logger &logger) const;
};

ModuleMapCuda::ModuleMapCuda(const Config &cfg,
                             std::unique_ptr<const Logger> logger_)
    : m_impl(std::make_unique<Impl>()),
      m_cfg(cfg),
      m_logger(std::move(logger_)) {
  module_map_triplet<float> moduleMapCpu;
  moduleMapCpu.read_TTree(cfg.moduleMapPath.c_str());
  if (!moduleMapCpu) {
    throw std::runtime_error("Cannot retrieve ModuleMap from " +
                             cfg.moduleMapPath);
  }

  ACTS_DEBUG("ModuleMap GPU block dim: " << m_cfg.gpuBlocks);

  if (m_cfg.retainMemPool) {
    // By default the stream-ordered memory pool returns all free memory to the
    // system at every synchronization, so each cudaMallocAsync after a sync
    // becomes a real allocation. Keep the memory in the pool instead.
    cudaMemPool_t memPool{};
    ACTS_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, m_cfg.gpuDevice));
    std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
    ACTS_CUDA_CHECK(cudaMemPoolSetAttribute(
        memPool, cudaMemPoolAttrReleaseThreshold, &threshold));
  }

  m_impl->cudaModuleMapDoublet =
      std::make_unique<CUDA_module_map_doublet<float>>(moduleMapCpu);
  m_impl->cudaModuleMapDoublet->HostToDevice();
  m_impl->cudaModuleMapTriplet =
      std::make_unique<CUDA_module_map_triplet<float>>(moduleMapCpu);
  m_impl->cudaModuleMapTriplet->HostToDevice();

  ACTS_DEBUG("# of modules = " << moduleMapCpu.module_map().size());

  // check if we actually have a module map
  std::map<std::uint64_t, int> test;

  std::vector<std::uint64_t> keys;
  keys.reserve(m_impl->cudaModuleMapDoublet->module_map().size());
  std::vector<int> vals;
  vals.reserve(m_impl->cudaModuleMapDoublet->module_map().size());

  for (auto [key, value] : m_impl->cudaModuleMapDoublet->module_map()) {
    auto [it, success] = test.insert({key, value});
    if (!success) {
      throw std::runtime_error("Duplicate key in module map");
    }
    keys.push_back(key);
    vals.push_back(value);
  }

  // copy module map to device
  m_impl->cudaModuleMapSize = m_impl->cudaModuleMapDoublet->module_map().size();
  m_impl->cudaModuleMapKeys = vecmem::make_unique_alloc<std::uint64_t[]>(
      m_impl->moduleMapMemory, m_impl->cudaModuleMapSize);
  m_impl->cudaModuleMapVals = vecmem::make_unique_alloc<int[]>(
      m_impl->moduleMapMemory, m_impl->cudaModuleMapSize);

  using KeyView = vecmem::data::vector_view<std::uint64_t>;
  using ValView = vecmem::data::vector_view<int>;
  const auto size = static_cast<KeyView::size_type>(m_impl->cudaModuleMapSize);
  vecmem::cuda::copy copy;
  copy(vecmem::data::vector_view<const std::uint64_t>(size, keys.data()),
       KeyView(size, m_impl->cudaModuleMapKeys.get()),
       vecmem::copy::type::host_to_device)
      ->wait();
  copy(vecmem::data::vector_view<const int>(size, vals.data()),
       ValView(size, m_impl->cudaModuleMapVals.get()),
       vecmem::copy::type::host_to_device)
      ->wait();
}

ModuleMapCuda::~ModuleMapCuda() = default;

PipelineTensors ModuleMapCuda::operator()(
    std::vector<float> &inputValues, std::size_t numNodes,
    const std::vector<std::uint64_t> &moduleIds,
    const ExecutionContext &execContext) {
  assert(execContext.device.isCuda());

  if (moduleIds.empty()) {
    throw NoEdgesError{};
  }

  const auto nHits = moduleIds.size();
  assert(inputValues.size() % moduleIds.size() == 0);
  const auto nFeatures = inputValues.size() / moduleIds.size();

  // Use the default stream if none is given
  const cudaStream_t stream = execContext.stream.value_or(cudaStreamLegacy);
  detail::DeviceMemory mem(stream, execContext.memoryResource);

  // Node features and module ids to the device
  auto nodeFeatures = Tensor<float>::Create({nHits, nFeatures}, execContext);
  mem.toDevice(nodeFeatures.data(), inputValues.data(), inputValues.size());
  auto cudaModuleIds = mem.make<std::uint64_t>(nHits);
  mem.toDevice(cudaModuleIds.get(), moduleIds.data(), nHits);

  return m_impl->buildGraph(std::move(nodeFeatures), cudaModuleIds.get(), mem,
                            execContext, m_cfg, logger());
}

PipelineTensors ModuleMapCuda::operator()(Tensor<float> nodeFeatures,
                                          const std::uint64_t *moduleIds,
                                          const ExecutionContext &execContext) {
  assert(execContext.device.isCuda());
  if (!nodeFeatures.device().isCuda()) {
    throw std::invalid_argument("ModuleMapCuda: node features must be on CUDA");
  }
  if (nodeFeatures.shape().at(0) == 0) {
    throw NoEdgesError{};
  }

  const cudaStream_t stream = execContext.stream.value_or(cudaStreamLegacy);
  detail::DeviceMemory mem(stream, execContext.memoryResource);
  return m_impl->buildGraph(std::move(nodeFeatures), moduleIds, mem,
                            execContext, m_cfg, logger());
}

PipelineTensors ModuleMapCuda::Impl::buildGraph(
    Tensor<float> nodeFeatures, const std::uint64_t *cudaModuleIds,
    detail::DeviceMemory &mem, const ExecutionContext &execContext,
    const ModuleMapCuda::Config &cfg, const Logger &logger) const {
  auto t0 = std::chrono::high_resolution_clock::now();
  const cudaStream_t stream = mem.stream();
  const auto nHits = nodeFeatures.shape().at(0);
  const auto nFeatures = nodeFeatures.shape().at(1);
  float *cudaNodeFeaturePtr = nodeFeatures.data();

  const dim3 blockDim = cfg.gpuBlocks;
  const dim3 gridDimHits = (nHits + blockDim.x - 1) / blockDim.x;
  ACTS_VERBOSE("gridDimHits: " << gridDimHits.x
                               << ", blockDim: " << blockDim.x);

  // The module map kernels read the unscaled hit coordinates as one array
  // per coordinate
  auto cudaHitCoordinates = mem.make<float>(6 * nHits);
  float *cudaR = cudaHitCoordinates.get();
  float *cudaPhi = cudaR + nHits;
  float *cudaZ = cudaPhi + nHits;
  float *cudaX = cudaZ + nHits;
  float *cudaY = cudaX + nHits;
  float *cudaEta = cudaY + nHits;
  const detail::HitArrays<float> hits{cudaR, cudaZ,   cudaX,
                                      cudaY, cudaEta, cudaPhi};

  detail::preprocessHitFeatures<<<gridDimHits, blockDim, 0, stream>>>(
      nHits, nFeatures, cudaNodeFeaturePtr, cudaR, cudaPhi, cudaZ, cudaEta,
      cudaX, cudaY, cfg.rScale, cfg.phiScale, cfg.zScale);
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Offsets of the hits of each module in the (module-sorted) hits: the hit
  // counts per module, followed by an exclusive scan
  auto cudaHitOffsets = mem.make<int>(cudaModuleMapSize + 1);
  mem.memset(cudaHitOffsets.get(), cudaModuleMapSize + 1, 0);
  detail::mapModuleIdsToNbHits<<<gridDimHits, blockDim, 0, stream>>>(
      cudaHitOffsets.get(), nHits, cudaModuleIds, cudaModuleMapSize,
      cudaModuleMapKeys.get(), cudaModuleMapVals.get());
  ACTS_CUDA_CHECK(cudaGetLastError());
  exclusiveSum(cudaHitOffsets.get(), cudaHitOffsets.get(),
               cudaModuleMapSize + 1, mem);

  ///////////////////////////////////
  // Perform module map inference
  ////////////////////////////////////

  if (cfg.debugSynchronize) {
    mem.synchronize();
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  // Builds the edges, and in its final pass writes the edge index and edge
  // features directly
  auto [edgeIndex, edgeFeatures] =
      makeEdges(hits, cudaHitOffsets.get(), cudaNodeFeaturePtr, nFeatures, mem,
                execContext, cfg, logger);
  ACTS_CUDA_CHECK(cudaGetLastError());
  if (cfg.debugSynchronize) {
    mem.synchronize();
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  ACTS_DEBUG("Preparation: " << ms(t0, t1));
  ACTS_DEBUG("Inference + postprocessing: " << ms(t1, t2));

  return {std::move(nodeFeatures),
          std::move(edgeIndex),
          std::move(edgeFeatures),
          {}};
}

std::pair<Tensor<std::int64_t>, Tensor<float>> ModuleMapCuda::Impl::makeEdges(
    const detail::HitArrays<float> &hits, const int *cudaHitOffsets,
    const float *cudaNodeFeatures, std::size_t nNodeFeatures,
    detail::DeviceMemory &mem, const ExecutionContext &execContext,
    const ModuleMapCuda::Config &cfg, const Logger &logger) const {
  // The doublet stage has one work item per (module doublet, source hit)
  // pair, the triplet stage one per (module triplet, M1-M2 edge) pair. The
  // *Offsets arrays are exclusive scans of the item counts, so their last
  // element is the total.
  cudaStream_t stream = mem.stream();
  const dim3 blockDim = cfg.gpuBlocks;
  auto gridFor = [&](std::size_t nThreads) {
    return dim3(
        static_cast<unsigned int>((nThreads + blockDim.x - 1) / blockDim.x));
  };

  // The cut windows of the module map, as device pointers
  // (the accessors of the module map are not const)
  auto cutsOf = [](auto &md) {
    return detail::DoubletCuts<float>{
        md.cuda_z0_min(),   md.cuda_z0_max(),        md.cuda_deta_min(),
        md.cuda_deta_max(), md.cuda_phi_slope_min(), md.cuda_phi_slope_max(),
        md.cuda_dphi_min(), md.cuda_dphi_max()};
  };
  const detail::DoubletCuts<float> doubletCuts = cutsOf(*cudaModuleMapDoublet);
  const detail::TripletCuts<float> tripletCuts{
      cutsOf(cudaModuleMapTriplet->module12()),
      cutsOf(cudaModuleMapTriplet->module23()),
      cudaModuleMapTriplet->cuda_diff_dydx_min(),
      cudaModuleMapTriplet->cuda_diff_dydx_max(),
      cudaModuleMapTriplet->cuda_diff_dzdr_min(),
      cudaModuleMapTriplet->cuda_diff_dzdr_max()};

  // ---------------------------------------------
  // count source hits per module doublet (sync 1)
  // ---------------------------------------------
  const int nDoublets = cudaModuleMapDoublet->size();
  ACTS_DEBUG("nDoublets: " << nDoublets);

  auto cudaDoubletWorkOffsets = mem.make<int>(nDoublets + 1);

  detail::count_source_hits_per_doublet<<<gridFor(nDoublets), blockDim, 0,
                                          stream>>>(
      cudaDoubletWorkOffsets.get(), cudaModuleMapDoublet->cuda_module1(),
      cudaHitOffsets, nDoublets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  exclusiveSum(cudaDoubletWorkOffsets.get(), cudaDoubletWorkOffsets.get(),
               nDoublets + 1, mem);

  int nDoubletWork{};
  mem.toHost(&nDoubletWork, &cudaDoubletWorkOffsets.get()[nDoublets], 1);
  mem.synchronize();
  ACTS_DEBUG("nDoubletWork: " << nDoubletWork);

  if (nDoubletWork == 0) {
    throw NoEdgesError{};
  }

  // ------------------------------------------------------------
  // count doublet edges and triplet work items together (sync 2)
  // ------------------------------------------------------------
  auto cudaWorkToDoublet = mem.make<int>(nDoubletWork);
  constexpr int kThreadsPerDoublet = 32;
  detail::build_work_to_item<kThreadsPerDoublet>
      <<<gridFor(static_cast<std::size_t>(nDoublets) * kThreadsPerDoublet),
         blockDim, 0, stream>>>(cudaWorkToDoublet.get(),
                                cudaDoubletWorkOffsets.get(), nDoublets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  auto cudaWorkEdgeOffsets = mem.make<int>(nDoubletWork + 1);
  // Accepted hit pairs per source hit, filled by the count pass and reused by
  // the build pass
  auto cudaPairMasks = mem.make<std::uint64_t>(nDoubletWork);
  detail::count_doublet_edges<float>
      <<<gridFor(nDoubletWork), blockDim, 0, stream>>>(
          cudaWorkEdgeOffsets.get(), cudaPairMasks.get(),
          cudaWorkToDoublet.get(), cudaDoubletWorkOffsets.get(),
          cudaModuleMapDoublet->cuda_module1(),
          cudaModuleMapDoublet->cuda_module2(), hits, doubletCuts,
          cudaHitOffsets, detail::g_pi, cfg.epsilon, nDoubletWork);
  ACTS_CUDA_CHECK(cudaGetLastError());

  exclusiveSum(cudaWorkEdgeOffsets.get(), cudaWorkEdgeOffsets.get(),
               nDoubletWork + 1, mem);

  auto cudaActiveWorkFlags = mem.make<int>(nDoubletWork + 1);
  auto cudaActiveWorkOffsets = mem.make<int>(nDoubletWork + 1);
  detail::mark_active_src_work<<<gridFor(nDoubletWork), blockDim, 0, stream>>>(
      cudaActiveWorkFlags.get(), cudaWorkEdgeOffsets.get(), nDoubletWork);
  ACTS_CUDA_CHECK(cudaGetLastError());
  exclusiveSum(cudaActiveWorkFlags.get(), cudaActiveWorkOffsets.get(),
               nDoubletWork + 1, mem);

  // Edge offsets per module doublet. These only depend on the edge counts, so
  // the triplet work can be counted before the doublet edges are built.
  auto cudaDoubletEdgeOffsets = mem.make<int>(nDoublets + 1);
  detail::doublet_edge_sum<<<gridFor(nDoublets + 1), blockDim, 0, stream>>>(
      cudaDoubletEdgeOffsets.get(), cudaDoubletWorkOffsets.get(),
      cudaWorkEdgeOffsets.get(), nDoublets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  const int nTriplets = cudaModuleMapTriplet->size();
  auto cudaTripletWorkOffsets = mem.make<int>(nTriplets + 1);
  detail::count_triplet_hits<<<gridFor(nTriplets), blockDim, 0, stream>>>(
      cudaTripletWorkOffsets.get(), cudaModuleMapTriplet->cuda_module12_map(),
      cudaModuleMapTriplet->cuda_module23_map(), cudaDoubletEdgeOffsets.get(),
      nTriplets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  exclusiveSum(cudaTripletWorkOffsets.get(), cudaTripletWorkOffsets.get(),
               nTriplets + 1, mem);

  int nDoubletEdges{};
  int nActiveWork{};
  int nTripletWork{};
  mem.toHost(&nDoubletEdges, &cudaWorkEdgeOffsets.get()[nDoubletWork], 1);
  mem.toHost(&nActiveWork, &cudaActiveWorkOffsets.get()[nDoubletWork], 1);
  mem.toHost(&nTripletWork, &cudaTripletWorkOffsets.get()[nTriplets], 1);
  mem.synchronize();
  ACTS_DEBUG("nDoubletEdges: " << nDoubletEdges);
  ACTS_DEBUG("nActiveWork: " << nActiveWork);
  ACTS_DEBUG("nTripletWork: " << nTripletWork);

  if (nDoubletEdges == 0 || nTripletWork == 0) {
    throw NoEdgesError{};
  }

  // ------------------
  // build doublet edges
  // ------------------
  ACTS_DEBUG("Allocate " << (2ul * nDoubletEdges * sizeof(int)) * 1.0e-6
                         << " MB for edges");
  auto cudaEdgeSrc = mem.make<int>(nDoubletEdges);
  auto cudaEdgeTgt = mem.make<int>(nDoubletEdges);

  // nDoubletEdges > 0 implies nActiveWork > 0
  auto cudaActiveWork = mem.make<int>(nActiveWork);
  detail::
      compact_active_doublets<<<gridFor(nDoubletWork), blockDim, 0, stream>>>(
          cudaActiveWork.get(), cudaActiveWorkFlags.get(),
          cudaActiveWorkOffsets.get(), nDoubletWork);
  ACTS_CUDA_CHECK(cudaGetLastError());

  detail::build_doublet_edges_active<float>
      <<<gridFor(nActiveWork), blockDim, 0, stream>>>(
          cudaEdgeSrc.get(), cudaEdgeTgt.get(), nActiveWork,
          cudaPairMasks.get(), cudaActiveWork.get(), cudaWorkToDoublet.get(),
          cudaDoubletWorkOffsets.get(), cudaHitOffsets,
          cudaWorkEdgeOffsets.get(), cudaModuleMapDoublet->cuda_module1(),
          cudaModuleMapDoublet->cuda_module2(), hits, doubletCuts, detail::g_pi,
          cfg.epsilon);
  ACTS_CUDA_CHECK(cudaGetLastError());

  ACTS_VERBOSE("First 10 doublet edges:\n"
               << debugPrintEdges(nDoubletEdges, cudaEdgeSrc.get(),
                                  cudaEdgeTgt.get(), mem));

  // -----------------------------
  // build doublets geometric cuts
  // -----------------------------
  auto cudaEdgeGeometry = mem.make<float4>(nDoubletEdges);
  auto cudaEdgeSlopes = mem.make<float4>(nDoubletEdges);

  detail::hits_geometric_cuts_packed<<<gridFor(nDoubletEdges), blockDim, 0,
                                       stream>>>(
      cudaEdgeGeometry.get(), cudaEdgeSlopes.get(), cudaEdgeSrc.get(),
      cudaEdgeTgt.get(), hits, detail::g_pi, cfg.epsilon, nDoubletEdges);
  ACTS_CUDA_CHECK(cudaGetLastError());
  if (cfg.debugSynchronize) {
    mem.synchronize();
  }

  auto cudaEdgeMask = mem.make<bool>(nDoubletEdges + 1);
  mem.memset(cudaEdgeMask.get(), (nDoubletEdges + 1), 0);

  // -------------------------
  // loop over module triplets
  // -------------------------
  auto cudaWorkToTriplet = mem.make<int>(nTripletWork);
  constexpr int kThreadsPerTriplet = 8;
  detail::build_work_to_item<kThreadsPerTriplet>
      <<<gridFor(static_cast<std::size_t>(nTriplets) * kThreadsPerTriplet),
         blockDim, 0, stream>>>(cudaWorkToTriplet.get(),
                                cudaTripletWorkOffsets.get(), nTriplets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Pairs whose cuts cannot be decided from the float geometry are queued by
  // the triplet kernel and resolved by triplet_pair_cuts_fallback
  auto cudaFallbackPairs =
      mem.make<detail::TripletFallbackPair>(detail::kTripletFallbackCapacity);
  auto cudaFallbackCount = mem.make<int>(1);
  mem.memset(cudaFallbackCount.get(), 1, 0);

  auto launchTripletCuts = [&](auto deferFallback) {
    constexpr bool kDefer = decltype(deferFallback)::value;
    detail::triplet_pair_cuts_fused_m23<float, kDefer>
        <<<gridFor(nTripletWork), blockDim, 0, stream>>>(
            cudaEdgeMask.get(), nTripletWork, cudaTripletWorkOffsets.get(),
            cudaWorkToTriplet.get(), cudaModuleMapTriplet->cuda_module12_map(),
            cudaModuleMapTriplet->cuda_module23_map(), cudaEdgeGeometry.get(),
            cudaEdgeSlopes.get(), tripletCuts, cudaEdgeSrc.get(),
            cudaEdgeTgt.get(), cudaDoubletEdgeOffsets.get(),
            cudaDoubletWorkOffsets.get(), cudaModuleMapDoublet->cuda_module1(),
            cudaHitOffsets, cudaWorkEdgeOffsets.get(), hits, detail::g_pi,
            cfg.epsilon, cudaFallbackPairs.get(), cudaFallbackCount.get(),
            detail::kTripletFallbackCapacity);
    ACTS_CUDA_CHECK(cudaGetLastError());
  };

  launchTripletCuts(std::true_type{});

  constexpr int kFallbackBlocks = 64;
  constexpr int kFallbackThreads = 256;
  detail::triplet_pair_cuts_fallback<float>
      <<<kFallbackBlocks, kFallbackThreads, 0, stream>>>(
          cudaEdgeMask.get(), cudaFallbackPairs.get(), cudaFallbackCount.get(),
          detail::kTripletFallbackCapacity, cudaEdgeGeometry.get(), tripletCuts,
          cudaEdgeSrc.get(), cudaEdgeTgt.get(), hits, detail::g_pi,
          cfg.epsilon);
  ACTS_CUDA_CHECK(cudaGetLastError());

  //------------------------
  // edges reduction (sync 3)
  //------------------------
  auto cudaEdgeMaskScan = mem.make<int>(nDoubletEdges + 1);
  auto scanMask = [&]() {
    exclusiveSum(
        thrust::make_transform_iterator(cudaEdgeMask.get(), CastBoolToInt{}),
        cudaEdgeMaskScan.get(), nDoubletEdges + 1, mem);
  };
  scanMask();

  int nGraphEdges{};
  int nFallbackPairs{};
  mem.toHost(&nGraphEdges, &cudaEdgeMaskScan.get()[nDoubletEdges], 1);
  mem.toHost(&nFallbackPairs, cudaFallbackCount.get(), 1);
  mem.synchronize();
  ACTS_DEBUG("nFallbackPairs: " << nFallbackPairs);

  if (nFallbackPairs > detail::kTripletFallbackCapacity) {
    // Some queued pairs were dropped. All tags set so far are correct (they
    // are only set for pairs that pass), so rerunning the triplet cuts with
    // inline fallback completes the mask.
    ACTS_WARNING("Triplet fallback queue overflow ("
                 << nFallbackPairs << " > " << detail::kTripletFallbackCapacity
                 << "), rerunning triplet cuts with inline fallback");
    launchTripletCuts(std::false_type{});
    scanMask();
    mem.toHost(&nGraphEdges, &cudaEdgeMaskScan.get()[nDoubletEdges], 1);
    mem.synchronize();
  }

  ACTS_DEBUG("nGraphEdges: " << nGraphEdges);
  if (nGraphEdges == 0) {
    throw NoEdgesError{};
  }

  // Compact the surviving edges straight into the output tensors and compute
  // the edge features in the same pass
  const auto nEdges = static_cast<std::size_t>(nGraphEdges);
  auto edgeIndex = Tensor<std::int64_t>::Create({2, nEdges}, execContext);
  auto edgeFeatures = Tensor<float>::Create(
      {nEdges, static_cast<std::size_t>(detail::g_nEdgeFeatures)}, execContext);

  detail::compactEdgesAndMakeFeatures<<<gridFor(nDoubletEdges), blockDim, 0,
                                        stream>>>(
      nDoubletEdges, cudaEdgeMask.get(), cudaEdgeMaskScan.get(),
      cudaEdgeSrc.get(), cudaEdgeTgt.get(), nEdges, nNodeFeatures,
      cudaNodeFeatures, edgeIndex.data(), edgeFeatures.data());
  ACTS_CUDA_CHECK(cudaGetLastError());

  ACTS_VERBOSE("First 10 graph edges:\n"
               << debugPrintEdges(nEdges, edgeIndex.data(),
                                  edgeIndex.data() + nEdges, mem));

  return {std::move(edgeIndex), std::move(edgeFeatures)};
}

}  // namespace ActsPlugins

// clang-format off
// EdgeLayerConnector implementation included here due to ODR violations in ModuleMapGraph
#include "EdgeLayerConnector.cu"
// clang-format on
