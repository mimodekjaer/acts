// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

namespace ActsPlugins::detail {

/// Register the custom TensorRT plugins used by GNN edge classifier engines
/// with the global TensorRT plugin registry. Must be called before an engine
/// that contains these plugins is deserialized. Thread safe and idempotent.
///
/// Plugins (namespace "", version "1"):
/// - ActsGnnBuildCsr: inputs edge_list [2,E] int32, node_features [N,F];
///   outputs perm_dst [E], off_dst [N+1], perm_src [E], off_src [N+1] (int32).
///   Edges grouped by target (row 1) and source (row 0) node, stable in edge
///   index.
/// - ActsGnnSegmentSum: inputs messages [E,C], perm_dst, off_dst, perm_src,
///   off_src; outputs agg_dst [N,C], agg_src [N,C]. Deterministic per-node sum
///   with fp32 accumulation, replacing torch_scatter's ScatterElements(add).
void registerTensorRTGnnPlugins();

}  // namespace ActsPlugins::detail
