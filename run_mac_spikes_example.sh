#!/bin/bash

# SPIKES detection example for MacBook (CPU only, no sudo required)
# This script runs spike detection on EDF test data

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate the morgoth environment
conda activate morgoth

# Configuration
dataset_dir="test_data/SPIKES/HEP/edf"
result_dir="test_data/SPIKES/HEP/results"

echo "==================================================================="
echo "Running SPIKES detection on EDF files (CPU mode)"
echo "==================================================================="
echo ""

# Step 1: Event-level prediction for spikes
echo "Step 1/2: Running event-level spike prediction (this may take several minutes)..."
python finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/SPIKES.pth \
            --abs_pos_emb \
            --dataset SPIKES \
            --data_format edf \
            --sampling_rate 0 \
            --already_format_channel_order no \
            --already_average_montage no \
            --allow_missing_channels no \
            --max_length_hour no \
            --leave_one_hemisphere_out no \
            --polarity 1 \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_1pStep \
            --prediction_slipping_step 1 \
            --smooth_result ema \
            --need_spikes_10s_result yes \
            --spikes_10s_result_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results yes

echo ""
echo "==================================================================="
echo "Step 2/2: Running EEG-level spike prediction..."
python EEG_level_head.py \
          --mode predict \
          --dataset SPIKES \
          --task_model checkpoints/SPIKES_EEGlevel.pth \
          --test_csv_dir ${result_dir}/pred_1pStep  \
          --result_dir ${result_dir} \
          --device cpu \
          --align_spike_detection_and_location

echo ""
echo "==================================================================="
echo "Done! Results saved to: ${result_dir}"
echo "==================================================================="

