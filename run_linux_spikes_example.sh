#!/bin/bash

# SPIKES detection on Linux with GPU (NVIDIA A40, CUDA 12.4)
# Mirrors run_mac_spikes_example.sh but uses GPU instead of CPU.
# Run with the 'morgoth' conda env active.

set -euo pipefail

dataset_dir="test_data/SPIKES/HEP/edf"
result_dir="test_data/SPIKES/HEP/results"

echo "==================================================================="
echo "Running SPIKES detection on EDF files (GPU mode)"
echo "==================================================================="
echo ""

echo "Step 1/2: Event-level spike prediction..."
OMP_NUM_THREADS=1 python -m torch.distributed.run \
            --nnodes=1 --nproc_per_node=1 --master_port=29500 \
            finetune_classification.py \
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
            --rewrite_results yes

echo ""
echo "==================================================================="
echo "Step 2/2: EEG-level spike prediction..."
python EEG_level_head.py \
          --mode predict \
          --dataset SPIKES \
          --task_model checkpoints/SPIKES_EEGlevel.pth \
          --test_csv_dir ${result_dir}/pred_1pStep \
          --result_dir ${result_dir} \
          --align_spike_detection_and_location

echo ""
echo "==================================================================="
echo "Done! Results saved to: ${result_dir}"
echo "==================================================================="
