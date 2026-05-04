"""
Calculate per-montage performance metrics for Morgoth seizure detection.

Mirrors the patient-grouped aggregation in example_model/get_metrics.py:
  1. Group seizure clips by patient (EMU####)
  2. Per patient: compute per-clip segment metrics, then concatenate all patient
     windows and call compute_metrics again for AUROC/AUPRC/specificity
  3. Override recall_event, fp, precision_event, f1_event from per-clip averages
  4. Average patient-level metrics across patients → montage summary

Interictal clips are included in the patient-level AUC window concatenation
for patients that have seizure clips (matching get_metrics.py). They also
appear in the per-file CSV.

Outputs:
  montage_metrics.csv          — one row per montage
  montage_metrics_per_patient.csv — one row per patient per montage
  montage_metrics_per_file.csv — one row per clip (seizure + interictal)
"""

import os
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'example_model', 'funcs'))
from calc_metrics_update import compute_metrics

STRIDE = 1  # seconds (STEP_SEC from morgoth)

MONTAGE_LIST = [
    'full',
    'uneeg_left_front',
    'uneeg_left_back',
    'uneeg_right_front',
    'uneeg_right_back',
    'uneeg_bilateral_back2',
    'uneeg_bilateral_front2',
    'uneeg_vert_left',
    'uneeg_vert_right',
    'uneeg_diag_left_front',
    'uneeg_diag_left_back',
    'uneeg_diag_right_front',
    'uneeg_diag_right_back',
    'uneeg_diag_bilateral_front',
    'uneeg_diag_bilateral_back',
    'uneeg_vert_bilateral',
    'epiminder_2',
    'ceribell',
]


def load_csv(path):
    df = pd.read_csv(path, index_col=0)
    label = df['label'].values.astype(int)
    pred  = df['pred'].values.astype(int)
    prob  = df['sz_prob'].values.astype(float)
    return label, pred, prob


def patient_id_from_fname(fname):
    """EMU0878_seizure_0 -> EMU0878"""
    return fname.split('_')[0]


def compute_montage_metrics(montage, results_dir, results_iic_dir, verbose=False):
    sz_csvs  = sorted(glob.glob(os.path.join(results_dir,     montage, '*.csv')))
    iic_csvs = sorted(glob.glob(os.path.join(results_iic_dir, montage, '*.csv')))

    if not sz_csvs and not iic_csvs:
        print(f"  [{montage}] no data, skipping")
        return None, [], []

    # ── group seizure clips by patient ────────────────────────────────────────
    patient_clips = defaultdict(list)  # patient_id -> list of (fname, label, pred, prob)
    patient_iic_clips = defaultdict(list)  # patient_id -> IIC clips (AUC windows only)
    seg_rows = []  # per-file rows for the per-file CSV

    for f in sz_csvs:
        fname = os.path.basename(f)[:-4]
        try:
            label, pred, prob = load_csv(f)
        except Exception as e:
            print(f"  [{montage}] skip {fname}: {e}")
            continue

        pid = patient_id_from_fname(fname)
        patient_clips[pid].append((fname, label, pred, prob))

        # per-file segment metrics (only for clips with seizures)
        if np.any(label == 1):
            try:
                m = compute_metrics(label, pred, prob, stride=STRIDE)
                m['file'] = fname
                m['patient_id'] = pid
                m['montage'] = montage
                m['clip_type'] = 'seizure'
                seg_rows.append(m)
            except Exception as e:
                print(f"  [{montage}] metrics error {fname}: {e}")

    for f in iic_csvs:
        fname = os.path.basename(f)[:-4]
        try:
            label, pred, prob = load_csv(f)
        except Exception as e:
            print(f"  [{montage}] skip iic {fname}: {e}")
            continue
        pid = patient_id_from_fname(fname)
        patient_iic_clips[pid].append((fname, label, pred, prob))

    # ── per-patient metrics (mirrors get_metrics.py patient_metrics) ──────────
    patient_rows = []

    for pid, clips in patient_clips.items():
        # skip patients with no seizure labels (all clips are iic-like)
        has_seizure = any(np.any(label == 1) for _, label, _, _ in clips)
        if not has_seizure:
            continue

        segment_metrics = []
        auc_label, auc_pred, auc_prob = [], [], []

        for fname, label, pred, prob in clips:
            try:
                m = compute_metrics(label, pred, prob, stride=STRIDE)
                m['file'] = fname
                segment_metrics.append(m)
            except Exception as e:
                print(f"  [{montage}] patient metrics error {fname}: {e}")
                continue
            auc_label.extend(label)
            auc_pred.extend(pred)
            auc_prob.extend(prob)

        # include IIC clips in segment_metrics and AUC windows (matches get_metrics.py)
        for fname, label, pred, prob in patient_iic_clips.get(pid, []):
            try:
                m = compute_metrics(label, pred, prob, stride=STRIDE)
                m['file'] = fname
                segment_metrics.append(m)
            except Exception as e:
                print(f"  [{montage}] patient metrics error iic {fname}: {e}")
                continue
            auc_label.extend(label)
            auc_pred.extend(pred)
            auc_prob.extend(prob)

        if not segment_metrics:
            continue

        seg_df = pd.DataFrame(segment_metrics)

        # patient-level: concatenate all windows and call compute_metrics
        # (gives AUROC, AUPRC, specificity, total_dura from combined windows)
        try:
            pat_m = compute_metrics(
                np.array(auc_label), np.array(auc_pred),
                np.array(auc_prob), stride=STRIDE)
        except Exception as e:
            print(f"  [{montage}] patient-level compute_metrics failed {pid}: {e}")
            continue

        # override event-level metrics from per-clip aggregation (matches get_metrics.py)
        pat_m['avg_sz_dura']      = (np.nansum(seg_df['total_sz_dura'].values) /
                                     np.nansum(seg_df['num_sz'].values))
        pat_m['num_sz']           = int(np.nansum(seg_df['num_sz'].values))
        pat_m['num_pred']         = int(np.nansum(seg_df['num_pred'].values))
        pat_m['recall_event']     = np.nanmean(seg_df['recall_event'].values)
        pat_m['fp']               = np.nanmean(seg_df['fp'].values)

        num_pred_arr = seg_df['num_pred'].values
        prec_arr     = seg_df['precision_event'].values
        denom = np.nansum(num_pred_arr)
        pat_m['precision_event']  = (np.nansum(prec_arr * num_pred_arr) / denom
                                     if denom > 0 else np.nan)

        re, pr = pat_m['recall_event'], pat_m['precision_event']
        pat_m['f1_event'] = (2 * re * pr / (re + pr)
                             if (re + pr) > 0 else 0.0)

        pat_m['patient_id'] = pid
        pat_m['montage']    = montage
        patient_rows.append(pat_m)

    # ── interictal per-file rows (for per-file CSV only) ─────────────────────
    for pid, iic_clips in patient_iic_clips.items():
        for fname, label, pred, prob in iic_clips:
            try:
                m = compute_metrics(label, pred, prob, stride=STRIDE)
                m['file'] = fname
                m['patient_id'] = pid
                m['montage'] = montage
                m['clip_type'] = 'interictal'
                seg_rows.append(m)
            except Exception as e:
                print(f"  [{montage}] metrics error iic {fname}: {e}")

    # ── aggregate patient rows → montage summary ──────────────────────────────
    if not patient_rows:
        return None, [], seg_rows

    pat_df = pd.DataFrame(patient_rows)
    pat_df[['precision_event', 'f1_event']] = (
        pat_df[['precision_event', 'f1_event']].fillna(0.0))

    num_pred_total = pat_df['num_pred'].sum(skipna=True)

    recall_event    = pat_df['recall_event'].mean(skipna=True)
    fp_per_hour     = pat_df['fp'].mean(skipna=True)
    precision_event = (
        (pat_df['precision_event'] * pat_df['num_pred']).sum(skipna=True)
        / num_pred_total if num_pred_total > 0 else np.nan)
    f1_event = (2 * recall_event * precision_event / (recall_event + precision_event)
                if (recall_event + precision_event) > 0 else 0.0)

    result = {
        'montage':            montage,
        'n_sz_files':         len(sz_csvs),
        'n_iic_files':        len(iic_csvs),
        'n_patients':         len(patient_rows),
        'num_sz':             int(pat_df['num_sz'].sum(skipna=True)),
        'total_sz_dura_min':  pat_df['total_sz_dura'].sum(skipna=True),
        'avg_sz_dura_min':    (pat_df['total_sz_dura'].sum(skipna=True) /
                               pat_df['num_sz'].sum(skipna=True)),
        'recall_event':       recall_event,
        'fp_per_hour':        fp_per_hour,
        'precision_event':    precision_event,
        'f1_event':           f1_event,
        'specificity':        pat_df['tn'].mean(skipna=True),
        'auroc_sample':       pat_df['auroc_sample'].mean(skipna=True),
        'auprc_sample':       pat_df['auprc_sample'].mean(skipna=True),
    }

    if verbose:
        print(f"  [{montage}]  recall={result['recall_event']:.3f}"
              f"  prec={result['precision_event']:.3f}"
              f"  f1={result['f1_event']:.3f}"
              f"  fp/h={result['fp_per_hour']:.3f}"
              f"  spec={result['specificity']:.3f}"
              f"  auroc={result['auroc_sample']:.3f}"
              f"  n_patients={result['n_patients']}")

    return result, patient_rows, seg_rows


def main():
    ap = argparse.ArgumentParser(description='Compute per-montage Morgoth metrics')
    ap.add_argument('--results',      default=os.path.join(HERE, 'results'))
    ap.add_argument('--results_iic',  default=os.path.join(HERE, 'results_interictal'))
    ap.add_argument('--out',          default=os.path.join(HERE, 'montage_metrics.csv'))
    ap.add_argument('--out_per_patient', default=os.path.join(HERE, 'montage_metrics_per_patient.csv'))
    ap.add_argument('--out_per_file', default=os.path.join(HERE, 'montage_metrics_per_file.csv'))
    ap.add_argument('--montage',      default=None)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    montages = ([args.montage] if args.montage
                else [m for m in MONTAGE_LIST
                      if os.path.isdir(os.path.join(args.results, m))
                      or os.path.isdir(os.path.join(args.results_iic, m))])

    print(f"Running {len(montages)} montages")

    summary_rows, all_patient_rows, all_seg_rows = [], [], []

    for m in montages:
        print(f"Processing {m}...")
        result, patient_rows, seg_rows = compute_montage_metrics(
            m, args.results, args.results_iic, verbose=args.verbose)
        if result is not None:
            summary_rows.append(result)
        all_patient_rows.extend(patient_rows)
        all_seg_rows.extend(seg_rows)

    # ── write outputs ─────────────────────────────────────────────────────────
    col_order = [
        'montage', 'n_sz_files', 'n_iic_files', 'n_patients', 'num_sz',
        'total_sz_dura_min', 'avg_sz_dura_min',
        'recall_event', 'fp_per_hour', 'precision_event', 'f1_event',
        'specificity', 'auroc_sample', 'auprc_sample',
    ]
    df_sum = pd.DataFrame(summary_rows)
    col_order = [c for c in col_order if c in df_sum.columns]
    df_sum = df_sum[col_order]
    df_sum.to_csv(args.out, index=False)
    print(f"\nSummary -> {args.out}")
    print(df_sum.to_string(index=False))

    if all_patient_rows:
        pd.DataFrame(all_patient_rows).to_csv(args.out_per_patient, index=False)
        print(f"Per-patient -> {args.out_per_patient}")

    if all_seg_rows:
        pd.DataFrame(all_seg_rows).to_csv(args.out_per_file, index=False)
        print(f"Per-file -> {args.out_per_file}")


if __name__ == '__main__':
    main()
