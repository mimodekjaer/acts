#!/bin/bash

# Setup ACTS environment
source ~/actsdev/setup_lcg.sh

# Setup Python environment for ACTS
source ./build/python/setup.sh

# Model and data paths
GNN=/eos/user/b/bhuth/model_storage/module_map_flavour/gnn/gnn_v9_meanrms/gnn_v9_meanrms.onnx
MODULE_MAP=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/module_maps/v9/ttbar_single_part_MMTag_1p3p0/merged_ttbar_plus_singles_mmg1.3.0_cleaned_mean_rms_replaced_thr5_tol1e-10_float

INPUT_ROOT=/eos/user/b/bhuth/data/dump/ttbar_ef_tracking_v9/user.avallier.45819556.EXT0._000034.DumpGNNITk_v9.root
#INPUT_ROOT=~/eos/data/dump/user.avallier.mc21_14TeV.900495.PG_single_muonpm_Pt10_etaFlatnp0_43.DumpGNNITk_single_muonpm_pt10_pu0_v9.1.e8481_s4149_r14697/user.avallier.43824508.EXT0._000001.DumpGNNITk_single_muonpm_pt10_pu0_v9.1.root

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-./output_gnn4itk}
mkdir -p "$OUTPUT_DIR"

# Number of events
EVENTS=${EVENTS:-1}

echo "GNN=$GNN"
echo "MODULE_MAP=$MODULE_MAP"
echo "INPUT_ROOT=$INPUT_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "EVENTS=$EVENTS"

# Run GNN4ITk script with Sofie backend
export ACTS_SEQUENCER_DISABLE_FPEMON=1
$PREFIX python3 Examples/Scripts/Python/gnn4itk_example.py "$@" \
    --inputRootDump "$INPUT_ROOT" \
    --moduleMapPath "$MODULE_MAP" \
    --gnnModel "$GNN" \
    --cpuOnly \
    --outputDir "$OUTPUT_DIR" \
    --events "$EVENTS"
    #--bufferEvents 20 \

echo "GNN4ITk processing complete. Results in: $OUTPUT_DIR"
