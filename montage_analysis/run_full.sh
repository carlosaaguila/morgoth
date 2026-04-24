#!/bin/bash
# Run full montage analysis on the EMU dataset (~18-22 hrs on Apple MPS)
# From the morgoth project root directory

eval "$(conda shell.bash hook)"
conda activate morgoth

python montage_analysis/run_seizure_analysis.py \
    --device mps \
    "$@"
