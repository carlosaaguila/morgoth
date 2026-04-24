# Morgoth Montage Analysis

Tests 19 different EEG electrode configurations (simulating wearable devices)
for seizure detection using the morgoth IIIC + SEIZURE_EEGlevel models on the
EMU dataset.

## Quick Start

```bash
# From the morgoth project root:

# Full run — all 3210 files, all 19 montages (~18-22 hrs on Apple MPS)
conda run -n morgoth python montage_analysis/run_seizure_analysis.py --device mps

# Quick test — 5 files per class, all montages (~3 min on MPS)
conda run -n morgoth python montage_analysis/run_seizure_analysis.py \
    --max_files 5 --device mps

# Single montage
conda run -n morgoth python montage_analysis/run_seizure_analysis.py \
    --montage full --device mps

# Resume an interrupted run (cached JSONs are reused automatically)
conda run -n morgoth python montage_analysis/run_seizure_analysis.py --device mps
```

## Outputs

| File | Description |
|---|---|
| `summary_metrics.csv` | Aggregate F1/sensitivity/precision for all montages |
| `ANALYSIS_LOG.md` | Auto-generated run log with per-montage tables |
| `results/{montage}_per_file.csv` | Per-file predictions + windowed F1 |
| `results/{montage}_{file}.json` | Per-file cache (enables resuming) |
| `results/{montage}_errors.csv` | Skipped files with error reason |
| `run.log` | Full timestamped console log |

## Analysis Design

### Montage Simulation
Each montage specifies which monopolar electrodes to keep active. All others
are zeroed. Common-average is computed over active channels only, then placed
into the correct positions of the 19-channel array — accurately simulating a
device with only those electrodes.

### Two Evaluation Levels
1. **Windowed (1-second):** Per-second F1/sensitivity/precision against EDF
   annotation ground truth
2. **File-level:** Binary seizure-present/absent per file (seizure folder = 1,
   interictal folder = 0)

### Signal Processing
- Resample to 200 Hz → bandpass 0.5–70 Hz → 60 Hz notch
- 10-second sliding windows with 1-second step
- Normalization: ±600 µV clip + per-channel min-max scale

### Model Pipeline
```
EDF → monopolar channels (19) → channel mask → common average
    → backbone (IIIC.pth, base_patch200_200) → 6-class probs/second
    → SEIZURE class probs → windowed predictions
    → SEIZURE_EEGlevel.pth → file-level prediction
```

## Montages (19 total)

| Montage | Electrodes | Description |
|---|---|---|
| `full` | 16 bipolar pairs | Full bilateral chain |
| `uneeg_left_front` | F7–T3 | UnEEG single left anterior |
| `uneeg_left_back` | T3–T5 | UnEEG single left posterior |
| `uneeg_right_front` | F8–T4 | UnEEG single right anterior |
| `uneeg_right_back` | T4–T6 | UnEEG single right posterior |
| `uneeg_bilateral_back2` | T3–T5, T4–T6 | Bilateral posterior 2-ch |
| `uneeg_bilateral_front2` | F7–T3, F8–T4 | Bilateral anterior 2-ch |
| `uneeg_vert_left` | C3–T3 | Vertical left custom |
| `uneeg_vert_right` | C4–T4 | Vertical right custom |
| `uneeg_diag_left_front` | F3–T3 | Diagonal left front |
| `uneeg_diag_left_back` | P3–T3 | Diagonal left back |
| `uneeg_diag_right_front` | F4–T4 | Diagonal right front |
| `uneeg_diag_right_back` | P4–T4 | Diagonal right back |
| `uneeg_diag_bilateral_front` | F3–T3, F4–T4 | Bilateral diagonal front |
| `uneeg_diag_bilateral_back` | P3–T3, P4–T4 | Bilateral diagonal back |
| `uneeg_vert_bilateral` | C3–T3, C4–T4 | Vertical bilateral |
| `epiminder_2` | C3–P3, C4–P4 | Epiminder 2-channel |
| `ceribell` | 8-ch lateral chain | Ceribell configuration |
