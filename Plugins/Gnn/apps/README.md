# GNN track finding throughput on traccc event data

`ActsGnnTracccThroughput` runs the GPU track finding chain on traccc input
without the ACTS examples framework: traccc clusterization, measurement
sorting and space point formation, then the GNN (module map graph, edge
classification, junction removal and connected components) and stops at the
track candidates. It mirrors `traccc_throughput_mt_cuda`: the events are read
once, every concurrent slot owns a CUDA stream, its memory resources and one
instance of each algorithm, and the throughput is measured over a warm-up and
a timed phase.

## Build

Dependencies beyond a CUDA ACTS build: a module map (ModuleMapGraph) and one
inference backend. TensorRT is the fast one; ONNX runtime and torch work too.

```sh
cmake -S acts -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
  -DACTS_BUILD_PLUGIN_GNN=ON -DACTS_ENABLE_CUDA=ON \
  -DACTS_GNN_ENABLE_MODULEMAP=ON -DACTS_GNN_ENABLE_TRACCC=ON \
  -DACTS_GNN_ENABLE_TENSORRT=ON -DTensorRT_ROOT=<tensorrt install> \
  -DACTS_GNN_ENABLE_TORCH=OFF -DACTS_GNN_ENABLE_ONNX=OFF \
  -DACTS_BUILD_PLUGIN_JSON=ON -DCMAKE_CUDA_ARCHITECTURES=<sm>
cmake --build build --target ActsGnnTracccThroughput -j
```

`ACTS_GNN_ENABLE_TRACCC` builds vecmem, detray and traccc as part of ACTS.
ModuleMapGraph needs ROOT and Boost (program_options, unit_test_framework,
graph, regex) through CMake.

## Inputs

- traccc detector files: detray geometry, material and surface grid json,
  and the digitization configuration (ACTS geometry hierarchy map json).
- Events as traccc csv cells, `event<N>-cells.csv`.
- A module map, given as the prefix of `<prefix>.doublets.root` and
  `<prefix>.triplets.root`.
- The edge classifier model. For TensorRT, build an engine from the ONNX
  export with dynamic shapes; the limits are the largest graph the engine
  accepts and set the device memory of one execution context:

```sh
trtexec --onnx=gnn.onnx --saveEngine=gnn.engine --fp16 \
  --minShapes=x:1x12,edge_index:2x1,edge_attr:1x6 \
  --optShapes=x:100000x12,edge_index:2x1000000,edge_attr:1000000x6 \
  --maxShapes=x:400000x12,edge_index:2x6000000,edge_attr:6000000x6
```

  The tensor names are those of the ONNX export; `--model-io` passes them
  if they differ from `x,edge_index,edge_attr,output`.
- Module ids. The graph is built from the module id of each space point,
  taken from the module map keyed by the ACTS geometry id of the surface. If
  the module map uses other ids (the GNN4ITk module map uses Athena
  identifiers), `--module-id-map` takes a csv with one line
  `acts_geometry_id,module_id` per module, where the first column is the
  ACTS geometry id value of the detray surface (`source` in the geometry
  json). Surfaces without a line get no module id and no graph edges.

## Running on ODD

The public ODD module map comes with a matching GNN model
(`https://acts.web.cern.ch/ci/gnn/odd_module_map_v02.tar`), and the traccc
data package (`traccc-data-v10`) has the ODD detector files and single and
ten muon events. ODD cells carry ACTS geometry ids, and the ODD module map is
keyed by them, so no module id map is needed:

```sh
ActsGnnTracccThroughput \
  --detector-file geometries/odd/odd-detray_geometry_detray.json \
  --material-file geometries/odd/odd-detray_material_detray.json \
  --grid-file geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file geometries/odd/odd-digi-geometric-config.json \
  --input-directory odd/geant4_10muon_5GeV/ --input-events 10 \
  --module-map module_map_odd_2k_events.1e-03.float.v1_3_PATCH \
  --model gnn_odd_module_map.engine \
  --cold-run-events 10 --processed-events 200 --slots 4
```

On an H100 with the ten muon events this gives about 330 events/s with one
slot and 515 events/s with four; with `--stage-timing 1` (one slot) the
classifier takes 2.4 ms of the 3 ms per event, the module map 0.2 ms and the
track building 0.1 ms. The events are small, so the numbers are a
functionality check, not a performance figure.

## Running on ITk

traccc forms pixel space points only, so the GNN sees no strip space points.
Use a pixel-only model (`--node-features 4`) with a matching module map; a
pixel+strip model gets incomplete graphs.

The ITk cells of the traccc test data carry detray barcodes, not ACTS
geometry ids, so pass `--use-acts-geom-source 0`. The GNN4ITk module map and
model are not public; with them and the module id map:

```sh
ActsGnnTracccThroughput \
  --detector-file detray_detector_geometry_fr_fixed.json \
  --material-file detray_detector_material_maps.json \
  --grid-file detray_detector_surface_grids_fr_fixed.json \
  --digitization-file ITk_digitization_config.json \
  --input-directory ttbar_mu200/ --input-events 10 --use-acts-geom-source 0 \
  --module-map <itk module map prefix> --model <itk engine> \
  --module-id-map <acts geometry id to module id csv> \
  --cold-run-events 10 --processed-events 100 --slots 2 --stage-timing 1
```

Size the TensorRT engine for ttbar graphs (several 100k nodes and millions
of edges) and keep in mind that every slot holds one execution context with
the device memory of the largest profile.

## Output

Events per second for the warm-up and the timed phase, the mean number of
candidates per event, and with `--stage-timing 1` the mean time of each
stage per event (the stream is synchronized after every stage, so use one
slot for that). `--dump-candidates` writes the measurement indices of the
candidates of the first event, to compare with another pipeline, and
`--log-file` appends one csv line per run.

## CUB versus thrust in the track building

The junction removal compacts the kept edges with `cub::DeviceSelect`, and
the candidate bounds and the module id sort use `cub::DeviceRadixSort`,
which keep the counts on the device. The alternative, `thrust::copy_if` and
`thrust::stable_sort_by_key` with the non-synchronizing policy, was measured
with the same app on the ODD ten muon events (one slot, 500 events, three
runs each, identical candidates):

| variant | track building per event |
|---|---|
| CUB | 0.11 ms |
| thrust | 0.15 to 0.18 ms |

CUB stays. The gap comes from the host synchronization that `copy_if` needs
to learn the number of kept edges; on ttbar graphs the radix sort of only the
needed label bits adds to it.
