"""
test_single_patient.py
======================
Minimal smoke-test that runs the Morgoth base model (backbone only, IIIC.pth)
in continuous-windowed prediction mode on a single patient's EDF files
(seizure + interictal) and writes one CSV per file, in the exact column
layout of the SVM example:

    ,sz_prob,pred,label
    <window-end-seconds>,<prob>,<0|1>,<0|1>

How it works
------------
1. For each EDF matching the patient prefix under `data/seizure/` and
   `data/interictal/`, load with MNE, clean channel names, and extract
   the 19-channel monopolar array used by Morgoth.
2. Bandpass (0.5–70 Hz) + 60 Hz notch, resample to 200 Hz, common-average
   reference over available channels, clip to ±600 µV, per-channel min-max
   to [-1, 1].
3. Slide a 10-s window with a 1-s step; batch windows through the base
   backbone (IIIC.pth, 6-class head) and take softmax class-1 as sz_prob.
4. Per-window ground-truth = any seizure sample inside that window in
   the EDF annotations. pred = (sz_prob > threshold).
5. Save one CSV per file to `--out_dir` (default: results_single/<patient>/).

Usage
-----
    conda run -n morgoth python montage_analysis/test_single_patient.py
    conda run -n morgoth python montage_analysis/test_single_patient.py \
        --patient EMU0878 --threshold 0.5 --device cpu
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from typing import Any

import mne
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# allow `import backbone`, `import utils` from morgoth root
MORGOTH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MORGOTH_DIR)

# reuse all preprocessing + model loading from the main montage script
from montage_analysis.run_seizure_analysis import (  # noqa: E402
    FS_TARGET,
    MONO_CHANNELS,
    STEP_SEC,
    WIN_SEC,
    apply_montage_and_infer,
    load_and_preprocess_edf,
    load_backbone,
    load_eeg_level,
    per_second_gt,
    save_continuous_csv,
)
import utils as morgoth_utils  # noqa: E402


def find_patient_files(patient: str, sz_dir: str, iic_dir: str) -> list[tuple[str, bool]]:
    """Return [(edf_path, is_seizure), …] for every file whose basename
    starts with `patient` (e.g. 'EMU0878')."""
    sz = sorted(glob.glob(os.path.join(sz_dir, f"{patient}*.edf")))
    iic = sorted(glob.glob(os.path.join(iic_dir, f"{patient}*.edf")))
    return [(p, True) for p in sz] + [(p, False) for p in iic]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, default="EMU0878")
    ap.add_argument(
        "--sz_dir",
        type=str,
        default=os.path.join(MORGOTH_DIR, "data", "seizure"),
    )
    ap.add_argument(
        "--iic_dir",
        type=str,
        default=os.path.join(MORGOTH_DIR, "data", "interictal"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results_single"),
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = os.path.join(args.out_dir, args.patient)
    os.makedirs(out_dir, exist_ok=True)

    files = find_patient_files(args.patient, args.sz_dir, args.iic_dir)
    if not files:
        raise SystemExit(
            f"No EDF files found for patient '{args.patient}' "
            f"in {args.sz_dir} or {args.iic_dir}"
        )
    print(f"[{args.patient}] {len(files)} file(s):")
    for p, is_sz in files:
        print(f"  {'SEIZURE' if is_sz else 'IIC    '}  {os.path.basename(p)}")

    print("Loading backbone (IIIC.pth)…")
    bb = load_backbone(device)
    # EEG-level head is unused here, but keeps apply_montage_and_infer happy.
    eeg = load_eeg_level(device)
    input_chans = morgoth_utils.get_input_chans(MONO_CHANNELS)

    # "full" montage = keep all 19 monopolar channels
    full_keep = set(MONO_CHANNELS)

    for edf_path, is_seizure in files:
        fname = os.path.basename(edf_path)
        print(f"\n→ {fname}  (is_seizure={is_seizure})")

        full_sig, label_df, orig_fs, available, err = load_and_preprocess_edf(
            edf_path)
        if err:
            print(f"  SKIP: {err}")
            continue

        res: dict[str, Any] | None
        res, merr = apply_montage_and_infer(
            full_sig, full_keep, available, bb, eeg, device,
            input_chans, args.batch_size,
        )
        if merr or res is None:
            print(f"  SKIP: {merr}")
            continue

        n_win = res["n_windows"]
        sz_probs = np.asarray(res["sz_probs"], dtype=float)
        preds = (sz_probs > args.threshold).astype(int)
        gt = per_second_gt(label_df, orig_fs, n_win)

        out_csv = os.path.join(out_dir, fname.replace(".edf", ".csv"))
        save_continuous_csv(out_csv, sz_probs, preds, gt)

        pos = int(gt.sum())
        pp = int(preds.sum())
        print(f"  {n_win} windows  |  gt_pos={pos}  pred_pos={pp}"
              f"  |  saved → {out_csv}")

    print(f"\nDone. CSVs in: {out_dir}")


if __name__ == "__main__":
    main()
