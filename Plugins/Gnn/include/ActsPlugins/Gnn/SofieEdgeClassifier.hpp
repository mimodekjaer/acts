// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/Stages.hpp"

#include <memory>
#include <mutex>

namespace ActsPlugins {
/// @addtogroup gnn_plugin
/// @{

class SofieEdgeClassifier final : public EdgeClassificationBase {
 public:
  struct Config {
    std::string modelPath;
    float cut = 0.5;
    std::size_t maxEdges = 300000;
    std::size_t maxNodes = 100000;
  };

  SofieEdgeClassifier(const Config &cfg,
                      std::unique_ptr<const Acts::Logger> logger);
  ~SofieEdgeClassifier();

  PipelineTensors operator()(PipelineTensors tensors,
                             const ExecutionContext &execContext = {}) override;

  Config config() const { return m_cfg; }

 private:
  std::unique_ptr<const Acts::Logger> m_logger;
  const auto &logger() const { return *m_logger; }

  Config m_cfg;

  struct Impl;
  std::unique_ptr<Impl> m_impl;
  mutable std::mutex m_mutex;
};

/// @}
}  // namespace ActsPlugins
