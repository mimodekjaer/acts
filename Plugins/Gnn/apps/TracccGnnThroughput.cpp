// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Throughput of the GNN track finding on traccc event data: traccc
// clusterization, measurement sorting and space point formation, then the
// GNN chain (module map graph, edge classification, track building), one
// instance per concurrent slot on its own CUDA stream, like
// traccc_throughput_mt_cuda.

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/DetrayModuleIdTable.hpp"
#include "ActsPlugins/Gnn/ModuleMapCuda.hpp"
#include "ActsPlugins/Gnn/TracccGnnChain.hpp"
#ifdef ACTS_GNN_WITH_TENSORRT
#include "ActsPlugins/Gnn/TensorRTEdgeClassifier.hpp"
#endif
#ifdef ACTS_GNN_WITH_ONNX
#include "ActsPlugins/Gnn/OnnxEdgeClassifier.hpp"
#endif
#ifdef ACTS_GNN_TORCH_BACKEND
#include "ActsPlugins/Gnn/TorchEdgeClassifier.hpp"
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>
#include <traccc/bfield/construct_const_bfield.hpp>
#include <traccc/clusterization/clustering_config.hpp>
#include <traccc/cuda/clusterization/clusterization_algorithm.hpp>
#include <traccc/cuda/clusterization/measurement_sorting_algorithm.hpp>
#include <traccc/cuda/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp>
#include <traccc/cuda/utils/make_magnetic_field.hpp>
#include <traccc/cuda/utils/stream_wrapper.hpp>
#include <traccc/definitions/common.hpp>
#include <traccc/edm/silicon_cell_collection.hpp>
#include <traccc/edm/track_collection.hpp>
#include <traccc/geometry/detector.hpp>
#include <traccc/geometry/detector_buffer.hpp>
#include <traccc/geometry/detector_design_description.hpp>
#include <traccc/geometry/host_detector.hpp>
#include <traccc/io/read_cells.hpp>
#include <traccc/io/read_detector.hpp>
#include <traccc/io/read_detector_description.hpp>
#include <traccc/io/read_magnetic_field.hpp>
#include <traccc/utils/memory_resource.hpp>
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#include <vecmem/utils/cuda/stream_wrapper.hpp>

using namespace ActsPlugins;

namespace {

struct Options {
  std::string detectorFile;
  std::string materialFile;
  std::string gridFile;
  std::string digitizationFile;
  std::string conditionsFile;
  std::string inputDirectory;
  std::size_t inputEvents = 10;
  std::size_t skip = 0;
  bool useActsGeomSource = true;
  std::string bfieldFile;
  float bfieldZ = 2.f;

  std::string moduleMapPath;
  std::string moduleIdMap;
  std::string model;
  float cut = 0.5f;
  std::size_t nodeFeatures = 12;
  int gpuBlocks = 512;
  std::string inputNames = "x,edge_index,edge_attr,output";
  std::size_t minSpacePoints = 3;
  std::size_t maxSpacePoints = 64;
  bool junctionRemoval = true;

  std::size_t slots = 1;
  std::size_t coldRunEvents = 10;
  std::size_t processedEvents = 100;
  bool stageTiming = false;
  std::string dumpCandidates;
  std::string logFile;
  std::string logLevel = "INFO";
};

void printHelp() {
  std::cout << R"(Usage: ActsGnnTracccThroughput [options]

Detector and input (traccc conventions):
  --detector-file FILE        detray geometry json
  --material-file FILE        detray material json (optional)
  --grid-file FILE            detray surface grids json (optional)
  --digitization-file FILE    digitization configuration json
  --conditions-file FILE      detector conditions json (default: digitization file)
  --input-directory DIR       directory with eventNNNNNNNNN-cells.csv
  --input-events N            number of events to read (default 10)
  --skip N                    first event (default 0)
  --use-acts-geom-source B    cells carry ACTS geometry ids (default 1)
  --bfield-file FILE          covfie field file; without it a constant field
  --bfield-z T                constant field in Tesla (default 2)

GNN:
  --module-map PREFIX         module map prefix (.doublets.root/.triplets.root)
  --module-id-map FILE        csv "geometry_id,module_id": ACTS geometry id of a
                              module to the module id of the module map
                              (default: module id = ACTS geometry id)
  --model FILE                edge classifier: .engine (TensorRT), .onnx, .pt
  --cut X                     edge score cut (default 0.5)
  --node-features N           4 or 12 (default 12)
  --model-io a,b,c,d          input/output tensor names of a TensorRT engine
                              (default x,edge_index,edge_attr,output)
  --gpu-blocks N              threads per block of the module map (default 512)
  --min-spacepoints N         candidate size limits (default 3 and 64)
  --max-spacepoints N
  --junction-removal B        (default 1)

Run:
  --slots N                   concurrent event slots, each on its own stream
                              and thread (default 1)
  --cold-run-events N         warm-up events (default 10)
  --processed-events N        timed events (default 100)
  --stage-timing B            synchronize after every stage and report the
                              mean time per stage (default 0)
  --dump-candidates FILE      write the candidates of the first event
  --log-file FILE             append a csv line with the throughput
  --log-level LEVEL           VERBOSE, DEBUG, INFO (default), WARNING
)";
}

Options parseOptions(int argc, char **argv) {
  Options o;
  std::map<std::string, std::string> kv;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      printHelp();
      std::exit(0);
    }
    if (a.rfind("--", 0) != 0 || i + 1 >= argc) {
      throw std::invalid_argument("Unexpected argument " + a);
    }
    kv[a.substr(2)] = argv[++i];
  }
  auto str = [&](const char *k, std::string &v) {
    if (auto it = kv.find(k); it != kv.end()) {
      v = it->second;
      kv.erase(it);
    }
  };
  auto num = [&](const char *k, auto &v) {
    if (auto it = kv.find(k); it != kv.end()) {
      std::istringstream(it->second) >> v;
      kv.erase(it);
    }
  };
  auto flag = [&](const char *k, bool &v) {
    if (auto it = kv.find(k); it != kv.end()) {
      v = (it->second == "1" || it->second == "true" || it->second == "on");
      kv.erase(it);
    }
  };
  str("detector-file", o.detectorFile);
  str("material-file", o.materialFile);
  str("grid-file", o.gridFile);
  str("digitization-file", o.digitizationFile);
  str("conditions-file", o.conditionsFile);
  str("input-directory", o.inputDirectory);
  num("input-events", o.inputEvents);
  num("skip", o.skip);
  flag("use-acts-geom-source", o.useActsGeomSource);
  str("bfield-file", o.bfieldFile);
  num("bfield-z", o.bfieldZ);
  str("module-map", o.moduleMapPath);
  str("module-id-map", o.moduleIdMap);
  str("model", o.model);
  num("cut", o.cut);
  num("node-features", o.nodeFeatures);
  str("model-io", o.inputNames);
  num("gpu-blocks", o.gpuBlocks);
  num("min-spacepoints", o.minSpacePoints);
  num("max-spacepoints", o.maxSpacePoints);
  flag("junction-removal", o.junctionRemoval);
  num("slots", o.slots);
  num("cold-run-events", o.coldRunEvents);
  num("processed-events", o.processedEvents);
  flag("stage-timing", o.stageTiming);
  str("dump-candidates", o.dumpCandidates);
  str("log-file", o.logFile);
  str("log-level", o.logLevel);
  if (!kv.empty()) {
    throw std::invalid_argument("Unknown option --" + kv.begin()->first);
  }
  for (const auto &[name, value] :
       {std::pair{"detector-file", o.detectorFile},
        std::pair{"digitization-file", o.digitizationFile},
        std::pair{"input-directory", o.inputDirectory},
        std::pair{"module-map", o.moduleMapPath},
        std::pair{"model", o.model}}) {
    if (value.empty()) {
      throw std::invalid_argument(std::string("Missing --") + name);
    }
  }
  if (o.conditionsFile.empty()) {
    o.conditionsFile = o.digitizationFile;
  }
  return o;
}

Acts::Logging::Level logLevel(const std::string &s) {
  if (s == "VERBOSE") {
    return Acts::Logging::VERBOSE;
  }
  if (s == "DEBUG") {
    return Acts::Logging::DEBUG;
  }
  if (s == "WARNING") {
    return Acts::Logging::WARNING;
  }
  return Acts::Logging::INFO;
}

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) {
    out.push_back(item);
  }
  return out;
}

/// Optional csv map from the ACTS geometry id of a module to its module id
std::unordered_map<std::uint64_t, std::uint64_t> readModuleIdMap(
    const std::string &file) {
  std::unordered_map<std::uint64_t, std::uint64_t> map;
  if (file.empty()) {
    return map;
  }
  std::ifstream in(file);
  if (!in) {
    throw std::runtime_error("Cannot open module id map " + file);
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || !std::isdigit(static_cast<unsigned char>(line[0]))) {
      continue;
    }
    const auto cols = split(line, ',');
    if (cols.size() < 2) {
      continue;
    }
    map[std::stoull(cols[0])] = std::stoull(cols[1]);
  }
  return map;
}

/// The edge classifier for a model file, by extension
std::shared_ptr<EdgeClassificationBase> makeClassifier(
    const Options &o, std::unique_ptr<const Acts::Logger> logger) {
  const auto ext = std::filesystem::path(o.model).extension().string();
  if (ext == ".engine") {
#ifdef ACTS_GNN_WITH_TENSORRT
    TensorRTEdgeClassifier::Config cfg;
    cfg.modelPath = o.model;
    cfg.cut = o.cut;
    cfg.numExecutionContexts = o.slots;
    const auto names = split(o.inputNames, ',');
    if (names.size() != 4) {
      throw std::invalid_argument("--model-io needs four names");
    }
    cfg.nodeFeaturesName = names[0];
    cfg.edgeIndexName = names[1];
    cfg.edgeFeaturesName = names[2];
    cfg.outputName = names[3];
    return std::make_shared<TensorRTEdgeClassifier>(cfg, std::move(logger));
#else
    throw std::invalid_argument("Built without TensorRT");
#endif
  }
  if (ext == ".onnx") {
#ifdef ACTS_GNN_WITH_ONNX
    OnnxEdgeClassifier::Config cfg;
    cfg.modelPath = o.model;
    cfg.cut = o.cut;
    cfg.device = Device::Cuda();
    return std::make_shared<OnnxEdgeClassifier>(cfg, std::move(logger));
#else
    throw std::invalid_argument("Built without ONNX runtime");
#endif
  }
  if (ext == ".pt") {
#ifdef ACTS_GNN_TORCH_BACKEND
    TorchEdgeClassifier::Config cfg;
    cfg.modelPath = o.model;
    cfg.cut = o.cut;
    cfg.useEdgeFeatures = true;
    return std::make_shared<TorchEdgeClassifier>(cfg, std::move(logger));
#else
    throw std::invalid_argument("Built without torch");
#endif
  }
  throw std::invalid_argument("Unknown model type " + o.model);
}

/// Mean durations of the stages of one slot, in milliseconds
struct StageTimes {
  double clusterization = 0;
  double spacePoints = 0;
  double graph = 0;
  double classifier = 0;
  double trackBuilding = 0;
  std::size_t events = 0;
  std::size_t candidates = 0;

  StageTimes &operator+=(const StageTimes &o) {
    clusterization += o.clusterization;
    spacePoints += o.spacePoints;
    graph += o.graph;
    classifier += o.classifier;
    trackBuilding += o.trackBuilding;
    events += o.events;
    candidates += o.candidates;
    return *this;
  }
};

/// One concurrent event slot: a CUDA stream with its memory resources and
/// the algorithms of the chain, as in traccc's full_chain_algorithm
class Slot {
 public:
  Slot(const Options &o, const DetrayModuleIdTable &moduleIds,
       const traccc::detector_buffer &detector,
       const traccc::detector_design_description::buffer &detDescr,
       const traccc::detector_conditions_description::buffer &detCond,
       const traccc::magnetic_field &field,
       std::shared_ptr<ModuleMapCuda> graphConstructor,
       std::shared_ptr<EdgeClassificationBase> classifier,
       const Acts::Logger &logger)
      : m_cachedDeviceMr(m_deviceMr),
        m_cachedPinnedMr(m_pinnedMr),
        m_stream(),
        m_tracccStream(m_stream.stream()),
        m_copy(m_stream),
        m_mr{m_cachedDeviceMr, &m_cachedPinnedMr},
        m_detector(detector),
        m_detDescr(detDescr),
        m_detCond(detCond),
        m_field(field),
        m_clusterization(m_mr, m_copy, m_tracccStream,
                         traccc::clustering_config{},
                         logger.cloneWithSuffix("Clusterization")),
        m_sorting(m_mr, m_copy, m_tracccStream,
                  logger.cloneWithSuffix("Sorting")),
        m_spacePoints(m_mr, m_copy, m_tracccStream,
                      logger.cloneWithSuffix("SpacePoints")),
        m_chain(
            chainConfig(o, std::move(graphConstructor), std::move(classifier)),
            moduleIds, m_mr, m_copy, m_tracccStream,
            logger.cloneWithSuffix("Gnn")),
        m_stageTiming(o.stageTiming) {}

  /// Runs the chain on the cells of one event, returns the candidates
  TracccGnnResult process(
      const traccc::edm::silicon_cell_collection::host &cells,
      StageTimes &times) {
    using Clock = std::chrono::high_resolution_clock;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t0 = Clock::now();

    traccc::edm::silicon_cell_collection::buffer cellsBuffer(
        static_cast<unsigned int>(cells.size()), m_cachedDeviceMr);
    m_copy(vecmem::get_data(cells), cellsBuffer)->ignore();
    const auto unsorted = m_clusterization(cellsBuffer, m_detDescr, m_detCond);
    const auto measurements = m_sorting(unsorted);
    if (m_stageTiming) {
      m_stream.synchronize();
    }
    const auto t1 = Clock::now();

    const auto spacePoints = m_spacePoints(m_detector, measurements);
    if (m_stageTiming) {
      m_stream.synchronize();
    }
    const auto t2 = Clock::now();

    GnnTiming timing;
    auto result =
        m_chain(m_detector, m_field, measurements, spacePoints, timing);
    m_stream.synchronize();

    times.events += 1;
    times.candidates += result.nCandidates;
    if (m_stageTiming) {
      times.clusterization += ms(t0, t1);
      times.spacePoints += ms(t1, t2);
      times.graph += timing.graphBuildingTime.count();
      for (const auto &t : timing.classifierTimes) {
        times.classifier += t.count();
      }
      times.trackBuilding += timing.trackBuildingTime.count();
    }
    return result;
  }

  const vecmem::copy &copy() const { return m_copy; }
  vecmem::memory_resource &hostMr() { return m_cachedPinnedMr; }

 private:
  static TracccGnnChain::Config chainConfig(
      const Options &o, std::shared_ptr<ModuleMapCuda> graphConstructor,
      std::shared_ptr<EdgeClassificationBase> classifier) {
    TracccGnnChain::Config cfg;
    cfg.graphConstructor = std::move(graphConstructor);
    cfg.edgeClassifiers = {std::move(classifier)};
    cfg.doJunctionRemoval = o.junctionRemoval;
    cfg.nNodeFeatures = o.nodeFeatures;
    cfg.minSpacePoints = o.minSpacePoints;
    cfg.maxSpacePoints = o.maxSpacePoints;
    cfg.mode = TracccGnnChain::Mode::Candidates;
    return cfg;
  }

  vecmem::cuda::device_memory_resource m_deviceMr;
  vecmem::binary_page_memory_resource m_cachedDeviceMr;
  vecmem::cuda::host_memory_resource m_pinnedMr;
  vecmem::binary_page_memory_resource m_cachedPinnedMr;
  vecmem::cuda::stream_wrapper m_stream;
  traccc::cuda::stream_wrapper m_tracccStream;
  vecmem::cuda::async_copy m_copy;
  traccc::memory_resource m_mr;

  const traccc::detector_buffer &m_detector;
  const traccc::detector_design_description::buffer &m_detDescr;
  const traccc::detector_conditions_description::buffer &m_detCond;
  const traccc::magnetic_field &m_field;

  traccc::cuda::clusterization_algorithm m_clusterization;
  traccc::cuda::measurement_sorting_algorithm m_sorting;
  traccc::cuda::silicon_pixel_spacepoint_formation_algorithm m_spacePoints;
  TracccGnnChain m_chain;
  bool m_stageTiming;
};

/// Writes the measurement indices of the candidates of an event
void dumpCandidates(const TracccGnnResult &result, Slot &slot,
                    const std::string &file) {
  traccc::edm::track_collection<traccc::default_algebra>::host tracks{
      slot.hostMr()};
  slot.copy()(result.candidates.tracks, tracks,
              vecmem::copy::type::device_to_host)
      ->wait();
  std::ofstream out(file);
  out << "candidate,n_measurements,measurement_indices\n";
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    const auto track = tracks.at(i);
    out << i << "," << track.constituent_links().size();
    for (const auto &link : track.constituent_links()) {
      out << "," << link.index;
    }
    out << "\n";
  }
}

}  // namespace

int main(int argc, char **argv) {
  Options o;
  try {
    o = parseOptions(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n\n";
    printHelp();
    return 1;
  }
  const auto loggerPtr =
      Acts::getDefaultLogger("GnnThroughput", logLevel(o.logLevel));
  const Acts::Logger &log = *loggerPtr;
  ACTS_LOCAL_LOGGER(loggerPtr->clone());
  using Clock = std::chrono::high_resolution_clock;
  auto seconds = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };

  // ----- Detector, field and events on the host, read once -----
  vecmem::host_memory_resource hostMr;
  vecmem::cuda::host_memory_resource pinnedHostMr;

  traccc::detector_design_description::host detDescr{hostMr};
  traccc::detector_conditions_description::host detCond{hostMr};
  traccc::io::read_detector_description(detDescr, detCond, o.detectorFile,
                                        o.digitizationFile, o.conditionsFile);
  traccc::host_detector detector;
  traccc::io::read_detector(detector, hostMr, o.detectorFile, o.materialFile,
                            o.gridFile);

  traccc::magnetic_field hostField;
  if (!o.bfieldFile.empty()) {
    traccc::io::read_magnetic_field(hostField, o.bfieldFile);
  } else {
    hostField = traccc::construct_const_bfield(
        0.f, 0.f, o.bfieldZ * traccc::unit<float>::T);
  }
  const traccc::magnetic_field field =
      traccc::cuda::make_magnetic_field(hostField);

  std::vector<traccc::edm::silicon_cell_collection::host> events;
  const auto tRead0 = Clock::now();
  for (std::size_t i = 0; i < o.inputEvents; ++i) {
    events.emplace_back(pinnedHostMr);
    traccc::io::read_cells(events.back(), o.skip + i, o.inputDirectory,
                           log.cloneWithSuffix("Cells"), &detCond,
                           traccc::data_format::csv, true, o.useActsGeomSource);
  }
  ACTS_INFO("Read " << events.size() << " events in "
                    << seconds(tRead0, Clock::now()) << " s");

  // ----- Device copies shared by all slots -----
  vecmem::cuda::device_memory_resource sharedDeviceMr;
  vecmem::cuda::copy syncCopy;
  std::vector<unsigned int> descrSizes(detDescr.size());
  for (std::size_t i = 0; i < detDescr.size(); ++i) {
    const auto design = detDescr.at(i);
    descrSizes[i] =
        std::max(static_cast<unsigned int>(design.bin_edges_x().size()),
                 static_cast<unsigned int>(design.bin_edges_y().size()));
  }
  traccc::detector_design_description::buffer deviceDetDescr(
      descrSizes, sharedDeviceMr, &hostMr,
      vecmem::data::buffer_type::resizable);
  syncCopy.setup(deviceDetDescr)->wait();
  syncCopy(vecmem::get_data(detDescr), deviceDetDescr)->wait();
  traccc::detector_conditions_description::buffer deviceDetCond(
      static_cast<unsigned int>(detCond.size()), sharedDeviceMr);
  syncCopy(vecmem::get_data(detCond), deviceDetCond)->wait();
  const traccc::detector_buffer deviceDetector =
      traccc::buffer_from_host_detector(detector, sharedDeviceMr, syncCopy);

  // ----- Module ids: detray surface -> ACTS geometry id -> module id -----
  std::unordered_map<std::uint64_t, std::uint64_t> barcodeToActs;
  for (std::size_t i = 0; i < detCond.size(); ++i) {
    barcodeToActs[detCond.geometry_id().at(i).value()] =
        detCond.acts_geometry_id().at(i);
  }
  const auto actsToModule = readModuleIdMap(o.moduleIdMap);
  auto moduleId =
      [&](detray::geometry::identifier id) -> std::optional<std::uint64_t> {
    const auto acts = barcodeToActs.find(id.value());
    if (acts == barcodeToActs.end()) {
      return std::nullopt;
    }
    if (actsToModule.empty()) {
      return acts->second;
    }
    const auto module = actsToModule.find(acts->second);
    if (module == actsToModule.end()) {
      return std::nullopt;
    }
    return module->second;
  };
  std::optional<DetrayModuleIdTable> moduleIds;
  traccc::host_detector_visitor<traccc::detector_type_list>(
      detector,
      [&]<detray::concepts::detector detector_t>(const detector_t &det) {
        moduleIds.emplace(det, moduleId, sharedDeviceMr, syncCopy);
      });
  ACTS_INFO("Module ids: " << moduleIds->nSensitive() << " sensitive surfaces, "
                           << moduleIds->nUnmapped() << " without module id");

  // ----- GNN stages shared by the slots -----
  ModuleMapCuda::Config mmCfg;
  mmCfg.moduleMapPath = o.moduleMapPath;
  mmCfg.rScale = 1000.f;
  mmCfg.phiScale = std::numbers::pi_v<float>;
  mmCfg.zScale = 1000.f;
  mmCfg.etaScale = 1.f;
  mmCfg.gpuBlocks = o.gpuBlocks;
  auto graphConstructor =
      std::make_shared<ModuleMapCuda>(mmCfg, log.cloneWithSuffix("ModuleMap"));
  auto classifier = makeClassifier(o, log.cloneWithSuffix("Classifier"));

  std::vector<std::unique_ptr<Slot>> slots;
  for (std::size_t i = 0; i < o.slots; ++i) {
    slots.push_back(std::make_unique<Slot>(o, *moduleIds, deviceDetector,
                                           deviceDetDescr, deviceDetCond, field,
                                           graphConstructor, classifier, log));
  }

  // ----- Processing: each slot runs on its own thread -----
  auto run = [&](std::size_t nEvents, const char *name) {
    std::atomic<std::size_t> next{0};
    std::vector<StageTimes> times(o.slots);
    std::vector<std::thread> threads;
    const auto t0 = Clock::now();
    for (std::size_t s = 0; s < o.slots; ++s) {
      threads.emplace_back([&, s]() {
        for (std::size_t i = next++; i < nEvents; i = next++) {
          slots[s]->process(events[i % events.size()], times[s]);
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }
    const double dt = seconds(t0, Clock::now());
    StageTimes total;
    for (const auto &t : times) {
      total += t;
    }
    ACTS_INFO(name << ": " << nEvents << " events in " << std::fixed
                   << std::setprecision(3) << dt << " s, "
                   << std::setprecision(1) << nEvents / dt << " events/s, "
                   << std::setprecision(1)
                   << static_cast<double>(total.candidates) / nEvents
                   << " candidates/event");
    if (o.stageTiming) {
      const double n = static_cast<double>(total.events);
      ACTS_INFO(name << " mean per event [ms]: clusterization "
                     << std::setprecision(2) << total.clusterization / n
                     << ", space points " << total.spacePoints / n << ", graph "
                     << total.graph / n << ", classifier "
                     << total.classifier / n << ", track building "
                     << total.trackBuilding / n);
    }
    return dt;
  };

  run(o.coldRunEvents, "Warm-up");
  const double dt = run(o.processedEvents, "Processing");

  if (!o.dumpCandidates.empty()) {
    StageTimes times;
    const auto result = slots[0]->process(events[0], times);
    dumpCandidates(result, *slots[0], o.dumpCandidates);
    ACTS_INFO("Wrote " << result.nCandidates << " candidates of event "
                       << o.skip << " to " << o.dumpCandidates);
  }
  if (!o.logFile.empty()) {
    std::ofstream out(o.logFile, std::ios::app);
    out << "\"" << o.inputDirectory << "\"," << o.slots << "," << o.inputEvents
        << "," << o.coldRunEvents << "," << o.processedEvents << "," << dt
        << "\n";
  }
  slots.clear();
  return 0;
}
