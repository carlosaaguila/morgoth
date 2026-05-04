#!/bin/bash

# NORMAL detection on Linux with GPU (NVIDIA A40, CUDA 12.4)
# Mirrors run_mac_example.sh but uses GPU instead of CPU.
# Run with the 'morgoth' conda env active.

set -euo pipefail

dataset_dir="test_data/Sandor/EDF"
data_format="edf"
sampling_rate=0
result_dir="test_data/Sandor/results"
already_format_channel_order='no'
already_average_montage='no'
allow_missing_channels='no'
leave_one_hemisphere_out='no'
polarity=-1  # Sandor EDF files have a polarity flip
rewrite_results='yes'

echo "==================================================================="
echo "Running NORMAL detection on Sandor EDF files (GPU mode)"
echo "==================================================================="
echo ""

echo "Step 1/2: Event-level prediction..."
OMP_NUM_THREADS=1 python -m torch.distributed.run \
            --nnodes=1 --nproc_per_node=1 --master_port=29500 \
            finetune_classification.py \
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
            --rewrite_results ${rewrite_results}

echo ""
echo "==================================================================="
echo "Step 2/2: EEG-level prediction..."
python EEG_level_head.py \
        --mode predict \
        --dataset NORMAL \
        --task_model checkpoints/NORMAL_EEGlevel.pth \
        --test_csv_dir ${result_dir}/pred_NORMAL_1sStep \
        --result_dir ${result_dir}

echo ""
echo "==================================================================="
echo "Done! Results saved to: ${result_dir}"
echo "==================================================================="
