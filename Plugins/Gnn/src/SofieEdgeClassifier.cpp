// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/SofieEdgeClassifier.hpp"

#include <variant>

#include "gnn_large_dynamic.hxx"
#include "gnn_small_dynamic.hxx"

using namespace Acts;

namespace ActsPlugins {

struct SofieEdgeClassifier::Impl {
  using Variant = std::variant<TMVA_SOFIE_gnn::Session, TMVA_SOFIE_gnn_large::Session>;
  std::unique_ptr<Variant> session;

  Impl(const std::string &modelPath, std::size_t maxEdges, std::size_t maxNodes,
       const Logger &logger) {
    ACTS_DEBUG("Attempting to load model from: " << modelPath);

    // Try loading as small model first
    try {
      session = std::make_unique<Variant>(TMVA_SOFIE_gnn::Session(modelPath, maxEdges, maxNodes));
      ACTS_INFO("Successfully loaded small model from " << modelPath);
      return;
    } catch (const std::exception &e) {
      ACTS_DEBUG("Failed to load as small model: " << e.what());
    }

    // Try loading as large model
    try {
      session = std::make_unique<Variant>(TMVA_SOFIE_gnn_large::Session(modelPath, maxEdges, maxNodes));
      ACTS_INFO("Successfully loaded large model from " << modelPath);
      return;
    } catch (const std::exception &e) {
      ACTS_DEBUG("Failed to load as large model: " << e.what());
    }

    throw std::runtime_error(
        "Failed to load model from " + modelPath +
        " as either small or large variant. Check debug logs for details.");
  }

  std::vector<float> infer(std::size_t numNodes, const float *nodeFeatures,
                           std::size_t numEdges, const std::int64_t *edgeIndex,
                           const float *edgeFeatures) {
    return std::visit(
        [&](auto &sess) {
          return sess.infer(numNodes, nodeFeatures, numEdges, edgeIndex,
                            edgeFeatures);
        },
        *session);
  }
};

SofieEdgeClassifier::SofieEdgeClassifier(const Config &cfg,
                                         std::unique_ptr<const Logger> _logger)
    : m_logger(std::move(_logger)),
      m_cfg(cfg),
      m_impl(std::make_unique<Impl>(m_cfg.modelPath, m_cfg.maxEdges,
                                    m_cfg.maxNodes, logger())) {
  ACTS_INFO("Max edges: " << m_cfg.maxEdges
                          << ", max nodes: " << m_cfg.maxNodes);
}

SofieEdgeClassifier::~SofieEdgeClassifier() = default;

PipelineTensors SofieEdgeClassifier::operator()(
    PipelineTensors tensors, const ExecutionContext &execContext) {
  auto numNodes = tensors.nodeFeatures.shape()[0];
  auto numEdges = tensors.edgeIndex.shape()[1];

  ACTS_DEBUG("SOFIE edge classification: " << numNodes << " nodes, " << numEdges
                                           << " edges");

  // Check if number of nodes exceeds maximum
  if (numNodes > m_cfg.maxNodes) {
    ACTS_WARNING("Number of nodes (" << numNodes
                                     << ") exceeds configured maximum ("
                                     << m_cfg.maxNodes << "), return 0 edges");
    throw NoEdgesError{};
  }

  // Check if number of edges exceeds maximum and apply limit if needed
  if (numEdges > m_cfg.maxEdges) {
    ACTS_WARNING("Number of edges ("
                 << numEdges << ") exceeds maximum (" << m_cfg.maxEdges
                 << "), shrink edge tensor to " << m_cfg.maxEdges);

    auto [newEdgeIndex, newEdgeFeatures] =
        applyEdgeLimit(tensors.edgeIndex, tensors.edgeFeatures, m_cfg.maxEdges,
                       execContext.stream);
    tensors.edgeIndex = std::move(newEdgeIndex);
    tensors.edgeFeatures = std::move(newEdgeFeatures);

    numEdges = tensors.edgeIndex.shape()[1];
  }

  // SOFIE is CPU-only, clone tensors to CPU if needed
  ExecutionContext cpuContext{Device::Cpu(), {}};
  bool onCuda = tensors.nodeFeatures.device().isCuda();

  Tensor<float> cpuNodeFeatures = onCuda
                                      ? tensors.nodeFeatures.clone(cpuContext)
                                      : std::move(tensors.nodeFeatures);
  Tensor<std::int64_t> cpuEdgeIndex = onCuda
                                          ? tensors.edgeIndex.clone(cpuContext)
                                          : std::move(tensors.edgeIndex);

  if (!tensors.edgeFeatures.has_value()) {
    throw std::runtime_error(
        "SofieEdgeClassifier requires edge features (edge_attr)");
  }

  Tensor<float> cpuEdgeFeatures = onCuda
                                      ? tensors.edgeFeatures->clone(cpuContext)
                                      : std::move(*tensors.edgeFeatures);

  // Run inference (protected by mutex since Session is not thread-safe)
  std::vector<float> result;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    result = m_impl->infer(numNodes, cpuNodeFeatures.data(), numEdges,
                           cpuEdgeIndex.data(), cpuEdgeFeatures.data());
  }

  // Copy results into a Tensor
  auto scores = Tensor<float>::Create({numEdges, 1ul}, cpuContext);
  std::memcpy(scores.data(), result.data(), numEdges * sizeof(float));

  sigmoid(scores);
  auto [newScores, newEdgeIndex] =
      applyScoreCut(scores, cpuEdgeIndex, m_cfg.cut);

  ACTS_DEBUG("Finished SOFIE edge classification, after cut: "
             << newEdgeIndex.shape()[1] << " edges.");

  if (newEdgeIndex.shape()[1] == 0) {
    throw NoEdgesError{};
  }

  // Clone back to CUDA if original tensors were on CUDA
  if (onCuda) {
    return {cpuNodeFeatures.clone(execContext),
            newEdgeIndex.clone(execContext),
            {},
            newScores.clone(execContext)};
  }

  return {std::move(cpuNodeFeatures),
          std::move(newEdgeIndex),
          {},
          std::move(newScores)};
}

}  // namespace ActsPlugins
