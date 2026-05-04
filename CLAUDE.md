# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch context

This is the `linux-GPU` branch. It targets Linux servers with NVIDIA GPUs exclusively. Do not add CPU fallbacks, Mac compatibility shims, or Windows paths here — those belong on `main` or platform-specific branches.

## Environment

Always use the `morgoth` conda environment unless the user says otherwise:

```bash
conda activate morgoth
# Python 3.12, PyTorch 2.4, CUDA 12.4
```

## Running inference

All GPU inference goes through `torch.distributed.run`. Never use `--device cpu` or `--distributed False` on this branch.

**Step 1 — event-level (continuous):**
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --master_port=29500 \
    finetune_classification.py --predict ...
```

**Step 2 — EEG-level aggregation:**
```bash
python EEG_level_head.py --mode predict --dataset <NAME> \
    --task_model checkpoints/<NAME>_EEGlevel.pth \
    --test_csv_dir <step1_output_dir> --result_dir <result_dir>
```

Use the example scripts as templates:
- `run_linux_example.sh` — NORMAL detection
- `run_linux_iiic_example.sh` — seizure/IIIC patterns
- `run_linux_spikes_example.sh` — spike detection

For EEGs too large for GPU memory, use `continuous_event_level_longeeg.sh` which segments automatically.

## Training

```bash
bash pretrain.sh              # tokenizer + masked pretraining
bash train_classification.sh  # event-level fine-tuning
bash train_EEG_level_head.sh  # EEG-level head fine-tuning
```

## Architecture

The pipeline has two stages:

**Event-level** (`finetune_classification.py` + `task_model.py` + `backbone.py`): A NeuralTransformer (`base_patch200_200`) reads 10-second EEG windows (19 channels, standard 10-20 montage, resampled to 200 Hz) and outputs per-second class probabilities. The backbone uses a VQNSP tokenizer pretrained in `tokenizer.py`/`train_tokenizer.py`. All preprocessing (bandpass, resample, montage, clip, normalize, epoch) happens automatically in `data_provider.py`.

**EEG-level** (`EEG_level_head.py`): A `CNNTransformerClassifier` reads the CSV of per-second probabilities from step 1 and outputs a single EEG-level prediction. Each task has its own `*_EEGlevel.pth` checkpoint.

**Supporting files:**
- `utils.py` — loss functions, distributed training utilities, `NativeScalerWithGradNormCount`
- `calibration.py` — Platt scaling and isotonic regression (CPU-only, runs on probability outputs)
- `segment_long_eeg.py` — splits long EEG files and merges segment-level results

## Key parameters

| Parameter | Notes |
|---|---|
| `--dataset` | `IIIC`, `SPIKES`, `FOC_GEN_SPIKES`, `BS`, `SLOWING`, `NORMAL`, `SLEEPPSG`, `MGBSLEEP3stages` |
| `--data_format` | `edf` or `mat` |
| `--sampling_rate` | Set to `0` if the file contains this info; otherwise specify explicitly |
| `--polarity` | `-1` for datasets with inverted EEG signal (e.g. Sandor) |
| `--rewrite_results` | Default `no`; set `yes` to overwrite existing output |
| `--prediction_slipping_step_second` | Sliding window step in seconds; use `1` for 1-second resolution |

## Guiding principles

1. **Don't assume. Don't hide confusion. Surface tradeoffs.** If a parameter or behavior is ambiguous, ask — don't guess and silently proceed.
2. **Minimum code that solves the problem. Nothing speculative.** Don't add handling for cases that don't exist yet.
3. **Touch only what you must. Clean up only your own mess.** Scope changes tightly; don't refactor surrounding code.
4. **Define success criteria. Loop until verified.** Before calling a task done, confirm the output files exist and look correct.

## Known issues

- NumPy/Pandas binary incompatibility: `conda install numpy=1.26.4`
- GPU driver ≤ 550: switch to `pytorch-cuda=12.1`
- `torch._C` errors from mixed conda/pip torch installs: install torch via conda first, then `pip install -r requirements.txt` (with torch removed from requirements)
