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
      const detail::HitArrays<float> &hits, int *cuda_hit_indice,
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
  auto &features = inputValues;

  // Get stream if available, otherwise use default stream
  cudaStream_t stream = cudaStreamLegacy;
  if (execContext.stream) {
    ACTS_DEBUG("Got stream " << *execContext.stream);
    stream = execContext.stream.value();
  }
  detail::DeviceMemory mem(stream, execContext.memoryResource);
  vecmem::cuda::async_copy copy{vecmem::cuda::stream_wrapper{stream}};

  /////////////////////////
  // Prepare input data
  ////////////////////////

  // Full node features to device

  auto nodeFeatures = Tensor<float>::Create({nHits, nFeatures}, execContext);
  using FloatView = vecmem::data::vector_view<float>;
  copy(vecmem::data::vector_view<const float>(
           static_cast<FloatView::size_type>(features.size()), features.data()),
       FloatView(static_cast<FloatView::size_type>(features.size()),
                 nodeFeatures.data()),
       vecmem::copy::type::host_to_device)
      ->ignore();

  // Module IDs to device
  auto cudaModuleIds = mem.make<std::uint64_t>(nHits);
  using IdView = vecmem::data::vector_view<std::uint64_t>;
  copy(vecmem::data::vector_view<const std::uint64_t>(
           static_cast<IdView::size_type>(nHits), moduleIds.data()),
       IdView(static_cast<IdView::size_type>(nHits), cudaModuleIds.get()),
       vecmem::copy::type::host_to_device)
      ->ignore();

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

  cudaStream_t stream = execContext.stream.value_or(cudaStreamLegacy);
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

  // Allocate memory for transposed node features that are needed for the
  // module map kernels in one block
  auto cudaNodeFeaturesTransposed = mem.make<float>(6 * nHits);

  float *transposed = cudaNodeFeaturesTransposed.get();
  const detail::HitArrays<float> hits{
      /*R=*/transposed + 0 * nHits,   /*z=*/transposed + 2 * nHits,
      /*x=*/transposed + 3 * nHits,   /*y=*/transposed + 4 * nHits,
      /*eta=*/transposed + 5 * nHits, /*phi=*/transposed + 1 * nHits};

  // Allocate helper nb hits memory
  auto cudaNbHits = mem.make<int>(cudaModuleMapSize + 1);
  mem.memset(cudaNbHits.get(), (cudaModuleMapSize + 1), 0);

  detail::preprocessHitFeatures<<<gridDimHits, blockDim, 0, stream>>>(
      nHits, nFeatures, cudaNodeFeaturePtr, transposed + 0 * nHits,
      transposed + 1 * nHits, transposed + 2 * nHits, transposed + 5 * nHits,
      transposed + 3 * nHits, transposed + 4 * nHits, cfg.rScale, cfg.phiScale,
      cfg.zScale);
  ACTS_CUDA_CHECK(cudaGetLastError());

  detail::mapModuleIdsToNbHits<<<gridDimHits, blockDim, 0, stream>>>(
      cudaNbHits.get(), nHits, cudaModuleIds, cudaModuleMapSize,
      cudaModuleMapKeys.get(), cudaModuleMapVals.get());
  ACTS_CUDA_CHECK(cudaGetLastError());

  exclusiveSum(cudaNbHits.get(), cudaNbHits.get(), cudaModuleMapSize + 1, mem);
  int *cudaHitIndice = cudaNbHits.get();

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
      makeEdges(hits, cudaHitIndice, cudaNodeFeaturePtr, nFeatures, mem,
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
    const detail::HitArrays<float> &hits, int *cuda_hit_indice,
    const float *cudaNodeFeatures, std::size_t nNodeFeatures,
    detail::DeviceMemory &mem, const ExecutionContext &execContext,
    const ModuleMapCuda::Config &cfg, const Logger &logger) const {
  cudaStream_t stream = mem.stream();
  const dim3 block_dim = cfg.gpuBlocks;
  auto gridFor = [&](std::size_t nThreads) {
    return dim3(
        static_cast<unsigned int>((nThreads + block_dim.x - 1) / block_dim.x));
  };

  // The cut windows of the module map, as device pointers
  // (the accessors of the module map are not const)
  auto doubletCuts = [](auto &md) {
    return detail::DoubletCuts<float>{
        md.cuda_z0_min(),   md.cuda_z0_max(),        md.cuda_deta_min(),
        md.cuda_deta_max(), md.cuda_phi_slope_min(), md.cuda_phi_slope_max(),
        md.cuda_dphi_min(), md.cuda_dphi_max()};
  };
  const detail::DoubletCuts<float> doublet_cuts =
      doubletCuts(*cudaModuleMapDoublet);
  const detail::TripletCuts<float> triplet_cuts{
      doubletCuts(cudaModuleMapTriplet->module12()),
      doubletCuts(cudaModuleMapTriplet->module23()),
      cudaModuleMapTriplet->cuda_diff_dydx_min(),
      cudaModuleMapTriplet->cuda_diff_dydx_max(),
      cudaModuleMapTriplet->cuda_diff_dzdr_min(),
      cudaModuleMapTriplet->cuda_diff_dzdr_max()};

  // ---------------------------------------------
  // count source hits per module doublet (sync 1)
  // ---------------------------------------------
  const int nb_doublets = cudaModuleMapDoublet->size();
  ACTS_DEBUG("nb doublets " << nb_doublets);

  auto cuda_nb_src_hits_per_doublet = mem.make<int>(nb_doublets + 1);

  detail::count_source_hits_per_doublet<<<gridFor(nb_doublets), block_dim, 0,
                                          stream>>>(
      cuda_nb_src_hits_per_doublet.get(), cudaModuleMapDoublet->cuda_module1(),
      cuda_hit_indice, nb_doublets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  exclusiveSum(cuda_nb_src_hits_per_doublet.get(),
               cuda_nb_src_hits_per_doublet.get(), nb_doublets + 1, mem);

  int sum_nb_src_hits_per_doublet{};
  mem.toHost(&sum_nb_src_hits_per_doublet,
             &cuda_nb_src_hits_per_doublet.get()[nb_doublets], 1);
  mem.synchronize();
  ACTS_DEBUG("sum_nb_hits_per_doublet: " << sum_nb_src_hits_per_doublet);

  if (sum_nb_src_hits_per_doublet == 0) {
    throw NoEdgesError{};
  }

  // ------------------------------------------------------------
  // count doublet edges and triplet work items together (sync 2)
  // ------------------------------------------------------------
  auto cuda_src_work_to_doublet = mem.make<int>(sum_nb_src_hits_per_doublet);
  constexpr int kThreadsPerDoublet = 32;
  detail::build_work_to_item<kThreadsPerDoublet>
      <<<gridFor(static_cast<std::size_t>(nb_doublets) * kThreadsPerDoublet),
         block_dim, 0, stream>>>(cuda_src_work_to_doublet.get(),
                                 cuda_nb_src_hits_per_doublet.get(),
                                 nb_doublets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  auto cuda_edge_sum_per_src_hit =
      mem.make<int>(sum_nb_src_hits_per_doublet + 1);
  // Accepted hit pairs per source hit, filled by the count pass and reused by
  // the build pass
  auto cuda_pair_masks = mem.make<std::uint64_t>(sum_nb_src_hits_per_doublet);
  detail::count_doublet_edges<float>
      <<<gridFor(sum_nb_src_hits_per_doublet), block_dim, 0, stream>>>(
          cuda_edge_sum_per_src_hit.get(), cuda_pair_masks.get(),
          cuda_src_work_to_doublet.get(), cuda_nb_src_hits_per_doublet.get(),
          cudaModuleMapDoublet->cuda_module1(),
          cudaModuleMapDoublet->cuda_module2(), hits, doublet_cuts,
          cuda_hit_indice, detail::g_pi, cfg.epsilon,
          sum_nb_src_hits_per_doublet);
  ACTS_CUDA_CHECK(cudaGetLastError());

  exclusiveSum(cuda_edge_sum_per_src_hit.get(), cuda_edge_sum_per_src_hit.get(),
               sum_nb_src_hits_per_doublet + 1, mem);

  auto cuda_active_src_flags = mem.make<int>(sum_nb_src_hits_per_doublet + 1);
  auto cuda_active_src_offsets = mem.make<int>(sum_nb_src_hits_per_doublet + 1);
  detail::mark_active_src_work<<<gridFor(sum_nb_src_hits_per_doublet),
                                 block_dim, 0, stream>>>(
      cuda_active_src_flags.get(), cuda_edge_sum_per_src_hit.get(),
      sum_nb_src_hits_per_doublet);
  ACTS_CUDA_CHECK(cudaGetLastError());
  exclusiveSum(cuda_active_src_flags.get(), cuda_active_src_offsets.get(),
               sum_nb_src_hits_per_doublet + 1, mem);

  // Edge offsets per module doublet. These only depend on the edge counts, so
  // the triplet work can be counted before the doublet edges are built.
  auto cuda_edge_sum = mem.make<int>(nb_doublets + 1);
  detail::doublet_edge_sum<<<gridFor(nb_doublets + 1), block_dim, 0, stream>>>(
      cuda_edge_sum.get(), cuda_nb_src_hits_per_doublet.get(),
      cuda_edge_sum_per_src_hit.get(), nb_doublets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  const int nb_triplets = cudaModuleMapTriplet->size();
  auto cuda_src_hits_per_triplet = mem.make<int>(nb_triplets + 1);
  detail::count_triplet_hits<<<gridFor(nb_triplets), block_dim, 0, stream>>>(
      cuda_src_hits_per_triplet.get(),
      cudaModuleMapTriplet->cuda_module12_map(),
      cudaModuleMapTriplet->cuda_module23_map(), cuda_edge_sum.get(),
      nb_triplets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  exclusiveSum(cuda_src_hits_per_triplet.get(), cuda_src_hits_per_triplet.get(),
               nb_triplets + 1, mem);

  int nb_doublet_edges{};
  int nb_active_src{};
  int nb_src_hits_per_triplet_sum{};
  mem.toHost(&nb_doublet_edges,
             &cuda_edge_sum_per_src_hit.get()[sum_nb_src_hits_per_doublet], 1);
  mem.toHost(&nb_active_src,
             &cuda_active_src_offsets.get()[sum_nb_src_hits_per_doublet], 1);
  mem.toHost(&nb_src_hits_per_triplet_sum,
             &cuda_src_hits_per_triplet.get()[nb_triplets], 1);
  mem.synchronize();
  ACTS_DEBUG("nb_doublet_edges: " << nb_doublet_edges);
  ACTS_DEBUG("nb_active_src: " << nb_active_src);
  ACTS_DEBUG("nb_src_hits_per_triplet_sum: " << nb_src_hits_per_triplet_sum);

  if (nb_doublet_edges == 0 || nb_src_hits_per_triplet_sum == 0) {
    throw NoEdgesError{};
  }

  // ------------------
  // build doublet edges
  // ------------------
  ACTS_DEBUG("Allocate " << (2ul * nb_doublet_edges * sizeof(int)) * 1.0e-6
                         << " MB for edges");
  auto cuda_reduced_M1_hits = mem.make<int>(nb_doublet_edges);
  auto cuda_reduced_M2_hits = mem.make<int>(nb_doublet_edges);

  // nb_doublet_edges > 0 implies nb_active_src > 0
  auto cuda_active_src_work = mem.make<int>(nb_active_src);
  detail::compact_active_doublets<<<gridFor(sum_nb_src_hits_per_doublet),
                                    block_dim, 0, stream>>>(
      cuda_active_src_work.get(), cuda_active_src_flags.get(),
      cuda_active_src_offsets.get(), sum_nb_src_hits_per_doublet);
  ACTS_CUDA_CHECK(cudaGetLastError());

  detail::build_doublet_edges_active<float>
      <<<gridFor(nb_active_src), block_dim, 0, stream>>>(
          cuda_reduced_M1_hits.get(), cuda_reduced_M2_hits.get(), nb_active_src,
          cuda_pair_masks.get(), cuda_active_src_work.get(),
          cuda_src_work_to_doublet.get(), cuda_nb_src_hits_per_doublet.get(),
          cuda_hit_indice, cuda_edge_sum_per_src_hit.get(),
          cudaModuleMapDoublet->cuda_module1(),
          cudaModuleMapDoublet->cuda_module2(), hits, doublet_cuts,
          detail::g_pi, cfg.epsilon);
  ACTS_CUDA_CHECK(cudaGetLastError());

  ACTS_VERBOSE("First 10 doublet edges:\n"
               << debugPrintEdges(nb_doublet_edges, cuda_reduced_M1_hits.get(),
                                  cuda_reduced_M2_hits.get(), mem));

  // -----------------------------
  // build doublets geometric cuts
  // -----------------------------
  auto cuda_geo = mem.make<float4>(nb_doublet_edges);
  auto cuda_edge_slope = mem.make<float4>(nb_doublet_edges);

  detail::hits_geometric_cuts_packed<<<gridFor(nb_doublet_edges), block_dim, 0,
                                       stream>>>(
      cuda_geo.get(), cuda_edge_slope.get(), cuda_reduced_M1_hits.get(),
      cuda_reduced_M2_hits.get(), hits, detail::g_pi, cfg.epsilon,
      nb_doublet_edges);
  ACTS_CUDA_CHECK(cudaGetLastError());
  if (cfg.debugSynchronize) {
    mem.synchronize();
  }

  auto cuda_mask = mem.make<bool>(nb_doublet_edges + 1);
  mem.memset(cuda_mask.get(), (nb_doublet_edges + 1), 0);

  // -------------------------
  // loop over module triplets
  // -------------------------
  auto cuda_work_to_triplet = mem.make<int>(nb_src_hits_per_triplet_sum);
  constexpr int kThreadsPerTriplet = 8;
  detail::build_work_to_item<kThreadsPerTriplet>
      <<<gridFor(static_cast<std::size_t>(nb_triplets) * kThreadsPerTriplet),
         block_dim, 0, stream>>>(cuda_work_to_triplet.get(),
                                 cuda_src_hits_per_triplet.get(), nb_triplets);
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Pairs whose cuts cannot be decided from the float geometry are queued by
  // the triplet kernel and resolved by triplet_pair_cuts_fallback
  auto cuda_fallback_pairs =
      mem.make<detail::TripletFallbackPair>(detail::kTripletFallbackCapacity);
  auto cuda_fallback_count = mem.make<int>(1);
  mem.memset(cuda_fallback_count.get(), 1, 0);

  auto launch_triplet_cuts = [&](auto defer_fallback) {
    constexpr bool kDefer = decltype(defer_fallback)::value;
    detail::triplet_pair_cuts_fused_m23<float, kDefer>
        <<<gridFor(nb_src_hits_per_triplet_sum), block_dim, 0, stream>>>(
            cuda_mask.get(), nb_src_hits_per_triplet_sum,
            cuda_src_hits_per_triplet.get(), cuda_work_to_triplet.get(),
            cudaModuleMapTriplet->cuda_module12_map(),
            cudaModuleMapTriplet->cuda_module23_map(), cuda_geo.get(),
            cuda_edge_slope.get(), triplet_cuts, cuda_reduced_M1_hits.get(),
            cuda_reduced_M2_hits.get(), cuda_edge_sum.get(),
            cuda_nb_src_hits_per_doublet.get(),
            cudaModuleMapDoublet->cuda_module1(), cuda_hit_indice,
            cuda_edge_sum_per_src_hit.get(), hits, detail::g_pi, cfg.epsilon,
            cuda_fallback_pairs.get(), cuda_fallback_count.get(),
            detail::kTripletFallbackCapacity);
    ACTS_CUDA_CHECK(cudaGetLastError());
  };

  launch_triplet_cuts(std::true_type{});

  constexpr int kFallbackBlocks = 64;
  constexpr int kFallbackThreads = 256;
  detail::triplet_pair_cuts_fallback<float>
      <<<kFallbackBlocks, kFallbackThreads, 0, stream>>>(
          cuda_mask.get(), cuda_fallback_pairs.get(), cuda_fallback_count.get(),
          detail::kTripletFallbackCapacity, cuda_geo.get(), triplet_cuts,
          cuda_reduced_M1_hits.get(), cuda_reduced_M2_hits.get(), hits,
          detail::g_pi, cfg.epsilon);
  ACTS_CUDA_CHECK(cudaGetLastError());

  //------------------------
  // edges reduction (sync 3)
  //------------------------
  auto cuda_mask_sum = mem.make<int>(nb_doublet_edges + 1);
  auto scan_mask = [&]() {
    exclusiveSum(
        thrust::make_transform_iterator(cuda_mask.get(), CastBoolToInt{}),
        cuda_mask_sum.get(), nb_doublet_edges + 1, mem);
  };
  scan_mask();

  int nb_graph_edges{};
  int nb_fallback_pairs{};
  mem.toHost(&nb_graph_edges, &cuda_mask_sum.get()[nb_doublet_edges], 1);
  mem.toHost(&nb_fallback_pairs, cuda_fallback_count.get(), 1);
  mem.synchronize();
  ACTS_DEBUG("nb_fallback_pairs: " << nb_fallback_pairs);

  if (nb_fallback_pairs > detail::kTripletFallbackCapacity) {
    // Some queued pairs were dropped. All tags set so far are correct (they
    // are only set for pairs that pass), so rerunning the triplet cuts with
    // inline fallback completes the mask.
    ACTS_WARNING("Triplet fallback queue overflow ("
                 << nb_fallback_pairs << " > "
                 << detail::kTripletFallbackCapacity
                 << "), rerunning triplet cuts with inline fallback");
    launch_triplet_cuts(std::false_type{});
    scan_mask();
    mem.toHost(&nb_graph_edges, &cuda_mask_sum.get()[nb_doublet_edges], 1);
    mem.synchronize();
  }

  ACTS_DEBUG("nb_graph_edges: " << nb_graph_edges);
  if (nb_graph_edges == 0) {
    throw NoEdgesError{};
  }

  // Compact the surviving edges straight into the output tensors and compute
  // the edge features in the same pass
  const auto nEdges = static_cast<std::size_t>(nb_graph_edges);
  auto edgeIndex = Tensor<std::int64_t>::Create({2, nEdges}, execContext);
  auto edgeFeatures = Tensor<float>::Create(
      {nEdges, static_cast<std::size_t>(detail::g_nEdgeFeatures)}, execContext);

  detail::compactEdgesAndMakeFeatures<<<gridFor(nb_doublet_edges), block_dim, 0,
                                        stream>>>(
      nb_doublet_edges, cuda_mask.get(), cuda_mask_sum.get(),
      cuda_reduced_M1_hits.get(), cuda_reduced_M2_hits.get(), nEdges,
      nNodeFeatures, cudaNodeFeatures, edgeIndex.data(), edgeFeatures.data());
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
