#!/bin/bash

# Simple example for running Morgoth on MacBook (CPU only, no sudo required)
# This script runs a NORMAL detection task on the Sandor EDF test data

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate the morgoth environment
conda activate morgoth

# Configuration
dataset_dir="test_data/Sandor/EDF"
data_format="edf"
sampling_rate=0  # 0 means the sampling rate is in the file
result_dir="test_data/Sandor/results"
already_format_channel_order='no'
already_average_montage='no'
allow_missing_channels='no'
leave_one_hemisphere_out='no'
polarity=-1  # Sandor EEG EDF file has a polarity flip
rewrite_results='yes'

echo "==================================================================="
echo "Running NORMAL detection on Sandor EDF files (CPU mode)"
echo "==================================================================="
echo ""

# Step 1: Continuous 1-second step event-level prediction for NORMAL
echo "Step 1/2: Running event-level prediction (this may take a few minutes)..."
python finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/NORMAL.pth \
            --abs_pos_emb \
            --dataset NORMAL \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --leave_one_hemisphere_out ${leave_one_hemisphere_out} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_NORMAL_1sStep \
            --prediction_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results ${rewrite_results}

echo ""
echo "==================================================================="
echo "Step 2/2: Running EEG-level prediction..."
python EEG_level_head.py \
        --mode predict \
        --dataset NORMAL \
        --task_model checkpoints/NORMAL_EEGlevel.pth \
        --test_csv_dir ${result_dir}/pred_NORMAL_1sStep \
        --result_dir ${result_dir} \
        --device cpu

echo ""
echo "==================================================================="
echo "Done! Results saved to: ${result_dir}"
echo "==================================================================="

