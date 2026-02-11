// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/ModuleMapCpu.hpp"

#include <TTree_hits>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <geometric_cuts>
#include <hit>
#include <module_map_triplet>
#include <numbers>
#include <set>
#include <vector>

using namespace Acts;
using Clock = std::chrono::high_resolution_clock;

namespace ActsPlugins {

namespace detail {

std::array<float, 6> computeEdgeFeatures(const float *srcFeatures,
                                         const float *tgtFeatures,
                                         std::size_t /*nFeatures*/) {
  constexpr float pi = std::numbers::pi_v<float>;
  enum NodeFeatures { r = 0, phi, z, eta };

  float dr = tgtFeatures[r] - srcFeatures[r];

  float rawDphi = pi * (tgtFeatures[phi] - srcFeatures[phi]);
  if (rawDphi > pi) {
    rawDphi -= 2.f * pi;
  } else if (rawDphi < -pi) {
    rawDphi += 2.f * pi;
  }
  float dphi = rawDphi / pi;

  float dz = tgtFeatures[z] - srcFeatures[z];
  float deta = tgtFeatures[eta] - srcFeatures[eta];

  float phislope = 0.f;
  float rphislope = 0.f;
  if (dr != 0.f) {
    phislope = std::clamp(dphi / dr, -100.f, 100.f);
    float avgR = 0.5f * (tgtFeatures[r] + srcFeatures[r]);
    rphislope = avgR * phislope;
  }

  return {dr, dphi, dz, deta, phislope, rphislope};
}

}  // namespace detail

class ModuleMapCpu::Impl {
 public:
  module_map_triplet<float> moduleMap;
};

ModuleMapCpu::ModuleMapCpu(const Config &cfg,
                           std::unique_ptr<const Logger> logger_)
    : m_impl(std::make_unique<Impl>()),
      m_cfg(cfg),
      m_logger(std::move(logger_)) {
  try {
    m_impl->moduleMap.read_TTree(cfg.moduleMapPath.c_str());
  } catch (const std::exception &e) {
    throw std::runtime_error("Cannot retrieve ModuleMap from " +
                             cfg.moduleMapPath + ": " + e.what());
  }
  if (!m_impl->moduleMap) {
    throw std::runtime_error("Cannot retrieve ModuleMap from " +
                             cfg.moduleMapPath);
  }

  ACTS_DEBUG("Loaded module map from " << m_cfg.moduleMapPath);
  ACTS_DEBUG("# of doublets = " << m_impl->moduleMap.map_doublet().size());
  ACTS_DEBUG("# of triplets = " << m_impl->moduleMap.map_triplet().size());
}

ModuleMapCpu::~ModuleMapCpu() = default;

PipelineTensors ModuleMapCpu::operator()(
    std::vector<float> &inputValues, std::size_t numNodes,
    const std::vector<std::uint64_t> &moduleIds,
    const ExecutionContext & /*execContext*/) {
  auto t0 = Clock::now();

  if (moduleIds.empty()) {
    throw NoEdgesError{};
  }

  const auto nHits = moduleIds.size();
  assert(inputValues.size() % nHits == 0);
  const auto nFeatures = inputValues.size() / nHits;

  // Node feature layout: [R(0), Phi(1), Z(2), Eta(3), ...]
  constexpr std::size_t rIdx = 0;
  constexpr std::size_t phiIdx = 1;
  constexpr std::size_t zIdx = 2;

  // -------------------------------------------------------
  // Step 1: Build TTree_hits from input features
  // -------------------------------------------------------
  // TTree_hits(true_graph=false, extra_features=false)
  TTree_hits<float> ttHits(false, false);

  for (std::size_t i = 0; i < nHits; ++i) {
    const float *f = inputValues.data() + i * nFeatures;
    float R = f[rIdx] * m_cfg.rScale;
    float phi = f[phiIdx] * m_cfg.phiScale;
    float z = f[zIdx] * m_cfg.zScale;
    float x = R * std::cos(phi);
    float y = R * std::sin(phi);

    hit<float> h(/*hit_id=*/i, x, y, z,
                 /*particle_id=*/0, /*module_ID=*/moduleIds[i],
                 /*hardware=*/"", /*barrel_endcap=*/0,
                 /*particle_ID1=*/0, /*particle_ID2=*/0);
    ttHits.push_back(h);
  }

  ACTS_DEBUG("Populated TTree_hits with " << ttHits.size() << " hits");

  auto &moduleMap = m_impl->moduleMap;

  // -------------------------------------------------------
  // Step 2: Doublet loop
  // Replicate graph_creator.cpp lines 183-225
  // -------------------------------------------------------
  const std::size_t nDoublets = moduleMap.map_doublet().size();

  // Per-doublet arrays of surviving hit pairs
  std::vector<std::vector<int>> M1_SP(nDoublets);
  std::vector<std::vector<int>> M2_SP(nDoublets);

  std::size_t doubletIdx = 0;
  std::size_t nbDoubletPass = 0;

  for (auto it = moduleMap.map_doublet().begin();
       it != moduleMap.map_doublet().end(); ++it, ++doubletIdx) {
    std::uint64_t module1 = it->first[0];
    std::uint64_t module2 = it->first[1];

    auto hits1 = ttHits.moduleID_equal_range(module1);
    auto hits2 = ttHits.moduleID_equal_range(module2);

    for (auto sp1It = hits1.first; sp1It != hits1.second; ++sp1It) {
      int SP1 = sp1It->second;

      for (auto sp2It = hits2.first; sp2It != hits2.second; ++sp2It) {
        int SP2 = sp2It->second;

        geometric_cuts<float> cuts(ttHits, SP1, SP2);
        if (it->second.cuts(cuts)) {
          continue;
        }

        M1_SP[doubletIdx].push_back(SP1);
        M2_SP[doubletIdx].push_back(SP2);
        ++nbDoubletPass;
      }
    }
  }

  auto t1 = Clock::now();
  ACTS_DEBUG("Doublet loop: " << nbDoubletPass << " passing pairs");

  if (nbDoubletPass == 0) {
    throw NoEdgesError{};
  }

  // -------------------------------------------------------
  // Step 3: Triplet loop
  // Replicate graph_creator.cpp lines 243-411
  // -------------------------------------------------------
  // Collect unique edges as (src, tgt) pairs
  std::set<std::pair<int, int>> edgeSet;

  for (auto it = moduleMap.map_triplet().begin();
       it != moduleMap.map_triplet().end(); ++it) {
    std::vector<std::uint64_t> modules12 = {it->first[0], it->first[1]};
    std::vector<std::uint64_t> modules23 = {it->first[1], it->first[2]};

    auto pair12It = moduleMap.map_pairs().find(modules12);
    auto pair23It = moduleMap.map_pairs().find(modules23);
    if (pair12It == moduleMap.map_pairs().end() ||
        pair23It == moduleMap.map_pairs().end()) {
      continue;
    }

    int ind12 = pair12It->second;
    int ind23 = pair23It->second;

    const auto &hitsM1 = M1_SP[ind12];
    const auto &hitsM2_from12 = M2_SP[ind12];
    const auto &hitsM2_from23 = M1_SP[ind23];
    const auto &hitsM3 = M2_SP[ind23];

    if (hitsM2_from12.empty() || hitsM2_from23.empty()) {
      continue;
    }

    auto &triplet = it->second;

    for (std::size_t i = 0; i < hitsM2_from12.size(); ++i) {
      int SP1 = hitsM1[i];
      int SP2 = hitsM2_from12[i];

      // Re-check doublet 1->2 cuts against triplet-level module cuts
      geometric_cuts<float> cuts12(ttHits, SP1, SP2);
      if (triplet.modules12().cuts(cuts12)) {
        continue;
      }

      // Find SP2 in the 2->3 doublet list
      auto findIt = std::find(hitsM2_from23.begin(), hitsM2_from23.end(), SP2);
      if (findIt == hitsM2_from23.end()) {
        continue;
      }

      // Iterate over all SP3 candidates sharing the same SP2
      for (std::size_t j = findIt - hitsM2_from23.begin();
           j < hitsM3.size() && hitsM2_from23[j] == SP2; ++j) {
        int SP3 = hitsM3[j];

        // Re-check doublet 2->3 cuts against triplet-level module cuts
        geometric_cuts<float> cuts23(ttHits, SP2, SP3);
        if (triplet.modules23().cuts(cuts23)) {
          continue;
        }

        // Triplet-level dy/dx cut
        float diffDydx = Diff_dydx(ttHits, SP1, SP2, SP3);
        if (diffDydx < triplet.diff_dydx_min() ||
            diffDydx > triplet.diff_dydx_max()) {
          continue;
        }

        // Triplet-level dz/dr cut
        float diffDzdr = Diff_dzdr(ttHits, SP1, SP2, SP3);
        if (diffDzdr < triplet.diff_dzdr_min() ||
            diffDzdr > triplet.diff_dzdr_max()) {
          continue;
        }

        // Triplet passed: add edges SP1->SP2 and SP2->SP3
        edgeSet.emplace(SP1, SP2);
        edgeSet.emplace(SP2, SP3);
      }
    }
  }

  auto t2 = Clock::now();
  ACTS_DEBUG("Triplet loop: " << edgeSet.size() << " unique edges");

  if (edgeSet.empty()) {
    throw NoEdgesError{};
  }

  // -------------------------------------------------------
  // Step 4: Build output tensors
  // -------------------------------------------------------
  const std::size_t nEdges = edgeSet.size();
  ExecutionContext cpuCtx{Device::Cpu(), {}};

  // Node features: copy of original inputValues
  auto nodeFeatures = Tensor<float>::Create({nHits, nFeatures}, cpuCtx);
  std::copy(inputValues.begin(), inputValues.end(), nodeFeatures.data());

  // Edge index: shape [2, nEdges]
  auto edgeIndex = Tensor<std::int64_t>::Create({2, nEdges}, cpuCtx);
  std::int64_t *srcPtr = edgeIndex.data();
  std::int64_t *tgtPtr = edgeIndex.data() + nEdges;

  // Edge features: shape [nEdges, 6]
  constexpr std::size_t nEdgeFeatures = 6;
  auto edgeFeatures = Tensor<float>::Create({nEdges, nEdgeFeatures}, cpuCtx);

  std::size_t edgeIdx = 0;
  for (const auto &[src, tgt] : edgeSet) {
    srcPtr[edgeIdx] = src;
    tgtPtr[edgeIdx] = tgt;

    // Edge features from original (non-rescaled) node features
    const float *srcFeatures = inputValues.data() + src * nFeatures;
    const float *tgtFeatures = inputValues.data() + tgt * nFeatures;
    auto ef = detail::computeEdgeFeatures(srcFeatures, tgtFeatures, nFeatures);
    float *efPtr = edgeFeatures.data() + edgeIdx * nEdgeFeatures;
    std::copy(ef.begin(), ef.end(), efPtr);

    ++edgeIdx;
  }

  auto t3 = Clock::now();

  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  ACTS_DEBUG("Preparation: " << ms(t0, t1));
  ACTS_DEBUG("Triplet inference: " << ms(t1, t2));
  ACTS_DEBUG("Postprocessing: " << ms(t2, t3));
  ACTS_DEBUG("Total: " << ms(t0, t3));
  ACTS_DEBUG("Edges: " << nEdges);

  return {std::move(nodeFeatures),
          std::move(edgeIndex),
          std::move(edgeFeatures),
          {}};
}

}  // namespace ActsPlugins
