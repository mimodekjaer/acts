/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/clusterization/details/measurement_creation.hpp"
#include "traccc/utils/detray_conversion.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void aggregate_clusters(
    const global_index_t globalIndex, const clustering_config& cfg,
    const edm::silicon_cell_collection::const_view& cells_view,
    const detector_design_description::const_view& det_descr_view,
    const detector_conditions_description::const_view& det_cond_view,
    const vecmem::data::vector_view<const unsigned int>& cluster_prefix_view,
    const vecmem::data::vector_view<const unsigned int>& next_cell_view,
    edm::measurement_collection::view measurements_view,
    vecmem::data::vector_view<unsigned int> disjoint_set_view,
    vecmem::data::vector_view<unsigned int> cluster_size_view) {
  const edm::silicon_cell_collection::const_device cells(cells_view);
  if (globalIndex >= cells.size()) {
    return;
  }

  // Only the threads of the "root" cells do any work. A cell is a root if
  // the inclusive prefix sum of the root flags changes at its position.
  const vecmem::device_vector<const unsigned int> cluster_prefix(
      cluster_prefix_view);
  const unsigned int prefix = cluster_prefix.at(globalIndex);
  const unsigned int prefix_before =
      (globalIndex > 0u) ? cluster_prefix.at(globalIndex - 1u) : 0u;
  if (prefix == prefix_before) {
    return;
  }
  const unsigned int link = prefix - 1u;

  const vecmem::device_vector<const unsigned int> next_cell(next_cell_view);
  const detector_design_description::const_device det_desc(det_descr_view);
  const detector_conditions_description::const_device det_cond(det_cond_view);
  edm::measurement_collection::device measurements(measurements_view);
  vecmem::device_vector<unsigned int> disjoint_set(disjoint_set_view);
  vecmem::device_vector<unsigned int> cluster_size(cluster_size_view);

  assert(link < measurements.size());
  edm::measurement_collection::device::proxy_type out = measurements.at(link);

  /*
   * Now, we iterate over all cells of the cluster, in cell order.
   *
   * Implemented here is a weighted version of Welford's algorithm. To read
   * more about this algorithm, see the following sources:
   *
   * [1] https://doi.org/10.1080/00401706.1962.10490022
   * [2] The Art of Computer Programming, Donald E. Knuth, second edition,
   * chapter 4.2.2.
   *
   * The core idea of Welford's algorithm is to use the recurrence relation
   *
   * $$\sigma^2_n = (1 - \frac{w_n}{W_n}) \sigma^2_{n-1} + \frac{w_n}{W_n}
   * (x_n - \mu_n) (x_n - \mu_{n-1})$$
   *
   * Which makes the algorithm less prone to catastrophic cancellation and
   * other unwanted effects. In addition, we offset the entire computation
   * by the first cell in the cluster, which brings the entire computation
   * closer to zero where floating point precision is higher. This relies on
   * the following:
   *
   * $$\mu(x_1, \ldots, x_n) = \mu(x_1 - C, \ldots, x_n - C) + C$$
   *
   * and
   *
   * $$\sigma^2(x_1, \ldots, x_n) = \sigma^2(x_1 - C, \ldots, x_n - C)$$
   */
  scalar totalWeight = 0.f;
  point2 mean{0.f, 0.f}, var{0.f, 0.f}, offset{0.f, 0.f}, width{0.f, 0.f},
      pitch{0.f, 0.f};

  unsigned int min_channel0 = std::numeric_limits<unsigned int>::max();
  unsigned int max_channel0 = std::numeric_limits<unsigned int>::lowest();
  unsigned int min_channel1 = std::numeric_limits<unsigned int>::max();
  unsigned int max_channel1 = std::numeric_limits<unsigned int>::lowest();

  const unsigned int module_idx = cells.module_index().at(globalIndex);
  const auto module_cd = det_cond.at(module_idx);
  const unsigned int design_idx = module_cd.module_to_design_id();
  const auto module_dd = det_desc.at(design_idx);

  unsigned int tmp_cluster_size = 0;

  unsigned int pos = globalIndex;
  while (pos != details::INVALID_CELL) {
    const edm::silicon_cell cell = cells.at(pos);
    const channel_id c0 = cell.channel0();
    const channel_id c1 = cell.channel1();
    const scalar activation = cell.activation();

    totalWeight += activation;
    scalar weight_factor = activation / totalWeight;

    point2 cell_position = traccc::details::position_from_cell(cell, module_dd);

    min_channel0 = std::min(min_channel0, c0);
    min_channel1 = std::min(min_channel1, c1);
    max_channel0 = std::max(max_channel0, c0);
    max_channel1 = std::max(max_channel1, c1);

    if (pos == globalIndex) {
      offset = cell_position;
    }

    cell_position = cell_position - offset;

    const point2 diff_old = cell_position - mean;
    mean = mean + diff_old * weight_factor;
    const point2 diff_new = cell_position - mean;

    var[0] = (1.f - weight_factor) * var[0] +
             weight_factor * (diff_old[0] * diff_new[0]);
    var[1] = (1.f - weight_factor) * var[1] +
             weight_factor * (diff_old[1] * diff_new[1]);

    tmp_cluster_size++;

    if (disjoint_set.capacity()) {
      disjoint_set.at(pos) = link;
    }

    pos = next_cell.at(pos);
  }

  if (cluster_size.capacity()) {
    cluster_size.at(link) = tmp_cluster_size;
  }

  unsigned int delta0 = (max_channel0 - min_channel0) + 1;
  unsigned int delta1 = (max_channel1 - min_channel1) + 1;

  vector2 cluster_lower_position = {(module_dd.bin_edges_x()).at(min_channel0),
                                    (module_dd.bin_edges_y()).at(min_channel1)};

  vector2 cluster_upper_position = {
      (module_dd.bin_edges_x()).at(max_channel0 + 1),
      (module_dd.bin_edges_y()).at(max_channel1 + 1)};

  width[0] = cluster_upper_position[0] - cluster_lower_position[0];
  width[1] = cluster_upper_position[1] - cluster_lower_position[1];

  pitch[0] = width[0] / static_cast<float>(delta0);
  pitch[1] = width[1] / static_cast<float>(delta1);

  var = var + point2{pitch[0] * pitch[0] / static_cast<scalar>(12.),
                     pitch[1] * pitch[1] / static_cast<scalar>(12.)};

  /*
   * Fill output vector with calculated cluster properties
   */
  const auto position = mean + offset + module_cd.measurement_translation();
  out.local_position() = utils::to_float_array<default_algebra>(position);
  out.local_variance() = utils::to_float_array<default_algebra>(var);
  out.surface_link() = module_cd.geometry_id();
  // Set a unique identifier for the measurement: the index of the first
  // (root) cell of the cluster. This is unique and deterministic, and lets
  // the measurement sorting define a deterministic total order.
  out.identifier() = globalIndex;
  // Set the dimensionality of the measurement.
  out.dimensions() = module_dd.dimensions();
  // Set the measurement's subspace.
  out.set_subspace(module_dd.subspace());
  // Set the index of the cluster that would be created for this measurement
  out.cluster_index() = link;

  if (cfg.diameter_strategy == clustering_diameter_strategy::CHANNEL0) {
    out.diameter() = static_cast<float>(width[0]);
  } else if (cfg.diameter_strategy == clustering_diameter_strategy::CHANNEL1) {
    out.diameter() = static_cast<float>(width[1]);
  } else if (cfg.diameter_strategy == clustering_diameter_strategy::MAXIMUM) {
    out.diameter() = static_cast<float>(std::max(width[0], width[1]));
  } else if (cfg.diameter_strategy == clustering_diameter_strategy::DIAGONAL) {
    out.diameter() = static_cast<float>(
        math::sqrt(width[0] * width[0] + width[1] * width[1]));
  }
}

}  // namespace traccc::device
