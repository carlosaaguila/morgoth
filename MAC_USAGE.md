# Morgoth MacBook (Apple Silicon) Usage Guide

## Overview
This guide explains how to run Morgoth EEG analysis on your MacBook with Apple Silicon (M1/M2/M3) using CPU only.

## Setup Summary
Your environment has been configured with:
- **Conda environment**: `morgoth` with ARM64-compatible packages
- **Python**: 3.12
- **PyTorch**: CPU version (2.9.1)
- **All dependencies**: Installed and compatible with Apple Silicon

## How to Run Examples

### 1. NORMAL Detection (Already tested - works!)
Detects normal vs abnormal EEG patterns.

```bash
./run_mac_example.sh
```

**Input**: EDF files in `test_data/Sandor/EDF/`
**Output**: Results in `test_data/Sandor/results/`
- Event-level predictions: `pred_NORMAL_1sStep/`
- EEG-level summary: `pred_EEG_level_NORMAL.csv`

### 2. SPIKES Detection
Detects epileptiform spikes in EEG.

```bash
./run_mac_spikes_example.sh
```

**Input**: EDF files in `test_data/SPIKES/HEP/edf/`
**Output**: Results in `test_data/SPIKES/HEP/results/`

### 3. IIIC Detection (Seizures + Patterns)
Detects seizures and ictal-interictal-injury continuum patterns (LPD, GPD, LRDA, GRDA).

```bash
./run_mac_iiic_example.sh
```

**Input**: MAT files in `test_data/IIIC/segments_raw/`
**Output**: Results in `test_data/IIIC/results/`

## Understanding the Output

### Event-Level Results
Each EEG file gets a CSV file with predictions for each time window (typically 1-second windows).

Example: `test_data/Sandor/results/pred_NORMAL_1sStep/ID-004.csv`

### EEG-Level Results
A summary CSV file with one row per EEG recording showing overall predictions.

Example: `test_data/Sandor/results/pred_EEG_level_NORMAL.csv`

Columns include:
- `file_name`: Name of the EEG file
- `probability`: Prediction probability
- `pred_class`: Predicted class (0 or 1)
- `confidence`: Confidence score

## Running on Your Own Data

To analyze your own EEG files:

1. Prepare your data:
   - EDF format: Place files in a directory
   - MAT format: Ensure proper preprocessing (see main README.md)

2. Modify one of the example scripts:
   - Change `dataset_dir` to your data directory
   - Change `result_dir` to your desired output directory
   - Adjust `polarity` if needed (1 or -1)

3. Run the script:
   ```bash
   ./run_mac_example.sh  # or your modified script
   ```

## Performance Note
CPU processing is slower than GPU. Expected times:
- Small EEG files (5-10 mins): ~20-30 seconds per file
- Large EEG files (hours): Several minutes per file

## Troubleshooting

### If you get "command not found: python"
Make sure the conda environment is activated:
```bash
conda activate morgoth
```

### If you get architecture errors
The environment needs to be rebuilt for ARM64. Contact for help if this happens.

### If you need to reinstall
```bash
conda remove -n morgoth --all -y
conda create -n morgoth python=3.12 -y
conda activate morgoth
pip install torch torchvision torchaudio
pip install -r requirements_windows.txt
pip install pyhealth
```

## Available Detection Models
All pretrained models are in `checkpoints/`:
- **NORMAL.pth**: Normal vs abnormal detection
- **SPIKES.pth**: Spike detection
- **IIIC.pth**: Seizure + IIIC patterns
- **SLOWING.pth**: Focal/generalized slowing
- **BS.pth**: Burst suppression
- **FOCGENSPIKES.pth**: Focal vs generalized spikes
- **SLEEP.pth** / **SLEEPPSG.pth**: Sleep staging

Each has a corresponding `*_EEGlevel.pth` file for EEG-level aggregation.

## Citation
If you use this tool, please cite the original Morgoth paper (see main README.md).

## Support
For issues specific to MacBook/ARM64, check the modifications made to:
- `EEG_level_head.py`: Lines 1180-1181, 1543, 1567 (CPU device handling)

For general Morgoth questions, see: https://github.com/bdsp-core/morgoth

