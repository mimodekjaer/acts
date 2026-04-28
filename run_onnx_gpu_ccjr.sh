#!/bin/bash

source /shared/setup_lcg.sh
source ./build/this_acts_withdeps.sh

# Model and data paths
GNN=/eos/user/b/bhuth/model_storage/module_map_flavour/gnn/gnn_v9_meanrms/gnn_v9_meanrms.onnx
MODULE_MAP=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/module_maps/v9/ttbar_single_part_MMTag_1p3p0/merged_ttbar_plus_singles_mmg1.3.0_cleaned_mean_rms_replaced_thr5_tol1e-10_float
INPUT_ROOT=/eos/user/b/bhuth/data/dump/ttbar_ef_tracking_v9/user.avallier.45819556.EXT0._000034.DumpGNNITk_v9.root


# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-./output_gnn4itk_gpu_ccjr}
mkdir -p "$OUTPUT_DIR"

echo "GNN=$GNN"
echo "MODULE_MAP=$MODULE_MAP"
echo "INPUT_ROOT=$INPUT_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"

# Run GNN4ITk script with Sofie backend
export ACTS_SEQUENCER_DISABLE_FPEMON=1
python3 Examples/Scripts/Python/gnn4itk_example.py \
    --inputRootDump "$INPUT_ROOT" \
    --moduleMapPath "$MODULE_MAP" \
    --gnnModel "$GNN" \
    --outputDir "$OUTPUT_DIR" \
 #   --bufferEvents 10 "$@" \

