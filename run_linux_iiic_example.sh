#!/bin/bash

# IIIC (Seizure + patterns) detection on Linux with GPU (NVIDIA A40, CUDA 12.4)
# Mirrors run_mac_iiic_example.sh but uses GPU instead of CPU.
# Run with the 'morgoth' conda env active.

set -euo pipefail

dataset_dir="test_data/IIIC/segments_raw"
result_dir="test_data/IIIC/results"

echo "==================================================================="
echo "Running IIIC detection on MAT files (GPU mode)"
echo "==================================================================="
echo ""

echo "Step 1/2: Event-level IIIC prediction..."
OMP_NUM_THREADS=1 python -m torch.distributed.run \
            --nnodes=1 --nproc_per_node=1 --master_port=29500 \
            finetune_classification.py \
            --predict \
            --model base_patch200_200 \
            --task_model checkpoints/IIIC.pth \
            --abs_pos_emb \
            --dataset IIIC \
            --data_format mat \
            --sampling_rate 200 \
            --already_format_channel_order no \
            --already_average_montage no \
            --allow_missing_channels no \
            --max_length_hour no \
            --leave_one_hemisphere_out no \
            --polarity 1 \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_1sStep \
            --prediction_slipping_step_second 1 \
            --rewrite_results yes

echo ""
echo "==================================================================="
echo "Step 2/2: EEG-level predictions for each IIIC subtype..."

IIIC_datasets=("SEIZURE" "LPD" "GPD" "LRDA" "GRDA")
for IIIC_dataset in "${IIIC_datasets[@]}"; do
    echo "Processing ${IIIC_dataset}..."
    python EEG_level_head.py \
            --mode predict \
            --dataset ${IIIC_dataset} \
            --task_model checkpoints/${IIIC_dataset}_EEGlevel.pth \
            --test_csv_dir ${result_dir}/pred_1sStep \
            --result_dir ${result_dir}
done

echo ""
echo "==================================================================="
echo "Done! Results saved to: ${result_dir}"
echo "==================================================================="
