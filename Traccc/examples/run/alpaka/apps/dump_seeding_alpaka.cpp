/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Validation helper: runs the Alpaka "seeding stage" chain (clusterization,
// measurement sorting, spacepoint formation, (GBTS) seeding and track
// parameter estimation) on a number of events, and dumps all intermediate
// results with full precision so that two builds can be compared bit-by-bit.

// Project include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/alpaka/gbts_seeding/gbts_seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/seed_parameter_estimation_algorithm.hpp"
#include "traccc/alpaka/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/alpaka/seeding/triplet_seeding_algorithm.hpp"
#include "traccc/alpaka/utils/make_magnetic_field.hpp"
#include "traccc/alpaka/utils/queue.hpp"
#include "traccc/alpaka/utils/vecmem_objects.hpp"
#include "traccc/examples/make_magnetic_field.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/detector_buffer.hpp"
#include "traccc/geometry/host_detector.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/logging.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_gbts_seeding.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/seeding/detail/track_params_estimation_config.hpp"

// VecMem include(s).
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char* argv[]) {
  std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
      "DumpSeedingAlpaka", traccc::Logging::Level::INFO);

  traccc::opts::detector detector_opts;
  traccc::opts::magnetic_field bfield_opts;
  traccc::opts::input_data input_opts;
  traccc::opts::clusterization clusterization_opts;
  traccc::opts::track_seeding seeding_opts;
  traccc::opts::track_gbts_seeding seeding_gbts_opts;
  traccc::opts::logging logging_opts;
  traccc::opts::program_options program_opts{
      "Alpaka seeding-stage result dump",
      {detector_opts, bfield_opts, input_opts, clusterization_opts,
       seeding_opts, seeding_gbts_opts, logging_opts},
      argc,
      argv,
      logger->cloneWithSuffix("Options")};

  const char* out_env = std::getenv("TRACCC_DUMP_OUT");
  const std::string out_name = out_env ? out_env : "dump.txt";
  const char* rep_env = std::getenv("TRACCC_DUMP_REPEAT");
  const int repeat = rep_env ? std::atoi(rep_env) : 1;

  // Memory resources.
  vecmem::host_memory_resource host_mr;
  traccc::alpaka::queue queue;
  traccc::alpaka::vecmem_objects vo(queue);
  vecmem::binary_page_memory_resource cached_pinned_host_mr(vo.host_mr());
  vecmem::memory_resource& device_mr = vo.device_mr();
  vecmem::binary_page_memory_resource cached_device_mr(device_mr);
  traccc::memory_resource mr{cached_device_mr, &cached_pinned_host_mr};
  vecmem::copy& copy = vo.async_copy();

  // Geometry.
  traccc::detector_design_description::host det_descr{host_mr};
  traccc::detector_conditions_description::host det_cond{host_mr};
  traccc::io::read_detector_description(
      det_descr, det_cond, detector_opts.detector_file,
      detector_opts.digitization_file, detector_opts.conditions_file,
      traccc::data_format::json);
  traccc::host_detector detector;
  traccc::io::read_detector(detector, host_mr, detector_opts.detector_file,
                            detector_opts.material_file,
                            detector_opts.grid_file);

  // Device geometry copies (mirrors full_chain_algorithm).
  std::vector<unsigned int> sizes(det_descr.size());
  for (std::size_t i = 0; i < det_descr.size(); ++i) {
    auto d = det_descr.at(i);
    sizes[i] = std::max(static_cast<unsigned int>(d.bin_edges_x().size()),
                        static_cast<unsigned int>(d.bin_edges_y().size()));
  }
  traccc::detector_design_description::buffer device_det_descr(
      sizes, device_mr, &host_mr, vecmem::data::buffer_type::resizable);
  traccc::detector_conditions_description::buffer device_det_cond(
      static_cast<traccc::detector_conditions_description::buffer::size_type>(
          det_cond.size()),
      device_mr);
  copy.setup(device_det_descr)->wait();
  copy(vecmem::get_data(det_descr), device_det_descr)->wait();
  copy(vecmem::get_data(det_cond), device_det_cond)->wait();
  const traccc::detector_buffer device_detector =
      traccc::buffer_from_host_detector(detector, device_mr, copy);

  const auto host_field = traccc::details::make_magnetic_field(bfield_opts);
  const auto device_field =
      traccc::alpaka::make_magnetic_field(host_field, queue);

  // Algorithms.
  const traccc::clustering_config clustering_cfg(clusterization_opts);
  const traccc::seedfinder_config seedfinder_config(seeding_opts);
  const traccc::seedfilter_config seedfilter_config(seeding_opts);
  const traccc::spacepoint_grid_config spacepoint_grid_config(seeding_opts);
  const traccc::gbts_seedfinder_config gbts_config(seeding_gbts_opts);
  const traccc::track_params_estimation_config tp_config;

  traccc::alpaka::clusterization_algorithm ca(mr, copy, queue, clustering_cfg,
                                              logger->clone("CA"));
  traccc::alpaka::measurement_sorting_algorithm ms(mr, copy, queue,
                                                   logger->clone("MS"));
  traccc::alpaka::silicon_pixel_spacepoint_formation_algorithm sf(
      mr, copy, queue, logger->clone("SF"));
  traccc::alpaka::triplet_seeding_algorithm sa(
      seedfinder_config, spacepoint_grid_config, seedfilter_config, mr, copy,
      queue, logger->clone("SA"));
  traccc::alpaka::gbts_seeding_algorithm gbts(gbts_config, mr, copy, queue,
                                              logger->clone("GBTS"));
  traccc::alpaka::seed_parameter_estimation_algorithm tp(
      tp_config, mr, copy, queue, logger->clone("TP"));

  std::FILE* out = std::fopen(out_name.c_str(), "w");
  if (out == nullptr) {
    std::fprintf(stderr, "Could not open %s\n", out_name.c_str());
    return 1;
  }

  for (std::size_t event = input_opts.skip;
       event < input_opts.skip + input_opts.events; ++event) {
    traccc::edm::silicon_cell_collection::host cells{host_mr};
    traccc::io::read_cells(cells, event, input_opts.directory, logger->clone(),
                           &det_cond, input_opts.format, true,
                           input_opts.use_acts_geom_source);

    for (int rep = 0; rep < repeat; ++rep) {
      traccc::edm::silicon_cell_collection::buffer cells_buffer(
          static_cast<unsigned int>(cells.size()), cached_device_mr);
      copy(vecmem::get_data(cells), cells_buffer)->ignore();

      auto unsorted = ca(cells_buffer, device_det_descr, device_det_cond);
      auto measurements = ms(unsorted);
      auto spacepoints = sf(device_detector, measurements);
      traccc::edm::seed_collection::buffer seeds;
      if (seeding_gbts_opts.useGBTS) {
        seeds = gbts(spacepoints, measurements);
      } else {
        seeds = sa(spacepoints);
      }
      auto params = tp(device_field, measurements, spacepoints, seeds);
      queue.synchronize();

      traccc::edm::measurement_collection::host h_meas{host_mr};
      traccc::edm::spacepoint_collection::host h_sp{host_mr};
      traccc::edm::seed_collection::host h_seeds{host_mr};
      traccc::bound_track_parameters_collection_types::host h_params;
      copy(measurements, h_meas)->wait();
      copy(spacepoints, h_sp)->wait();
      copy(seeds, h_seeds)->wait();
      copy(params, h_params)->wait();
      queue.synchronize();

      std::fprintf(out, "E %zu rep %d cells %zu M %zu S %zu D %zu P %zu\n",
                   event, rep, cells.size(), h_meas.size(), h_sp.size(),
                   h_seeds.size(), h_params.size());
      for (std::size_t i = 0; i < h_meas.size(); ++i) {
        const auto m = h_meas.at(i);
        std::fprintf(out, "M %zu %lu %a %a %a %a %u %a %u %u %u %u\n", i,
                     static_cast<unsigned long>(m.surface_link().value()),
                     m.local_position()[0], m.local_position()[1],
                     m.local_variance()[0], m.local_variance()[1],
                     m.dimensions(), m.diameter(), m.identifier(),
                     m.cluster_index(), m.subspace()[0], m.subspace()[1]);
      }
      for (std::size_t i = 0; i < h_sp.size(); ++i) {
        const auto s = h_sp.at(i);
        std::fprintf(out, "S %zu %u %u %a %a %a %a %a\n", i,
                     s.measurement_index_1(), s.measurement_index_2(), s.x(),
                     s.y(), s.z(), s.z_variance(), s.radius_variance());
      }
      for (std::size_t i = 0; i < h_seeds.size(); ++i) {
        const auto s = h_seeds.at(i);
        std::fprintf(out, "D %zu %u %u %u %a\n", i, s.bottom_index(),
                     s.middle_index(), s.top_index(), s.quality());
      }
      for (std::size_t i = 0; i < h_params.size(); ++i) {
        const auto& p = h_params.at(i);
        std::fprintf(out, "P %zu %lu", i,
                     static_cast<unsigned long>(p.surface_link().value()));
        for (unsigned int j = 0; j < traccc::e_bound_size; ++j) {
          std::fprintf(out, " %a", traccc::getter::element(p.vector(), j, 0));
        }
        for (unsigned int j = 0; j < traccc::e_bound_size; ++j) {
          for (unsigned int k = 0; k < traccc::e_bound_size; ++k) {
            std::fprintf(out, " %a",
                         traccc::getter::element(p.covariance(), j, k));
          }
        }
        std::fprintf(out, "\n");
      }
    }
  }
  std::fclose(out);
  return 0;
}
