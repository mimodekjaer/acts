/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cstddef>
#include <limits>

namespace traccc::device::details {
/// These indices in clusterization will only range from 0 to
/// max_cells_per_partition, so we only need a short. Keep a fallback
/// unsigned int index for enormous partitions.
using index_t = unsigned short;
using fallback_index_t = unsigned int;

/// The limit on the stack size in terms of cells per thread.
static constexpr std::size_t CELLS_PER_THREAD_STACK_LIMIT = 32;

/// Sentinel value terminating the cluster linked lists
static constexpr unsigned int INVALID_CELL =
    std::numeric_limits<unsigned int>::max();
}  // namespace traccc::device::details
