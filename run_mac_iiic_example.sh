#!/bin/bash

# IIIC (Seizure + patterns) detection example for MacBook (CPU only, no sudo required)
# This script runs IIIC pattern detection on MAT test data

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate the morgoth environment
conda activate morgoth

# Configuration
dataset_dir="test_data/IIIC/segments_raw"
result_dir="test_data/IIIC/results"

echo "==================================================================="
echo "Running IIIC detection on MAT files (CPU mode)"
echo "==================================================================="
echo ""

# Step 1: Event-level prediction
echo "Step 1/2: Running event-level IIIC prediction (this may take several minutes)..."
python finetune_classification.py \
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
            --device cpu \
            --distributed False \
            --rewrite_results yes

echo ""
echo "==================================================================="
echo "Step 2/2: Running EEG-level predictions for each IIIC subtype..."

# IIIC includes multiple subtypes: SEIZURE, LPD, GPD, LRDA, GRDA
IIIC_datasets=("SEIZURE" "LPD" "GPD" "LRDA" "GRDA")
for IIIC_dataset in "${IIIC_datasets[@]}"; do
    echo "Processing ${IIIC_dataset}..."
    python EEG_level_head.py \
            --mode predict \
            --dataset ${IIIC_dataset} \
            --task_model checkpoints/${IIIC_dataset}_EEGlevel.pth \
            --test_csv_dir ${result_dir}/pred_1sStep \
            --result_dir ${result_dir} \
            --device cpu
done

echo ""
echo "==================================================================="
echo "Done! Results saved to: ${result_dir}"
echo "==================================================================="

