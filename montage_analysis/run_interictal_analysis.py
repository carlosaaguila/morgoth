"""
Morgoth Interictal Analysis — Matched to Seizure Run
====================================================
Mirrors run_seizure_analysis.py but processes ONLY the interictal clips
in data/interictal that match seizure clips already processed in
montage_analysis/results/<montage>/*.json.

Matching: seizure file `EMU0878_seizure_3.edf` -> interictal `EMU0878_iic_3.edf`.
For each montage subdir under results/, we read the existing seizure JSON
filenames, swap `_seizure_` -> `_iic_`, check the interictal EDF exists,
and run inference on it.

Outputs (separate from seizure run, no clobber):
  results_interictal/<montage>/<basename>.csv   — per-window preds (ts, sz_prob, pred, label)
  results_interictal/<montage>_<file>.json      — per-file cache (resumable)
  results_interictal/<montage>_per_file.csv     — per-file summary
  summary_metrics_interictal.csv                — aggregate metrics
  ANALYSIS_LOG_INTERICTAL.md                    — human-readable log

caveman: same brain, diff data. label always 0 -> windowed_f1 NaN, but we
still report file-level FP rate (predictions on truly negative clips).

Usage:
  conda run -n morgoth python montage_analysis/run_interictal_analysis.py
  conda run -n morgoth python montage_analysis/run_interictal_analysis.py --montage ceribell
  conda run -n morgoth python montage_analysis/run_interictal_analysis.py --workers 4
"""

import os
import sys
import argparse
import logging
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# reuse everything from the seizure script (same preprocessing, models, montages)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_seizure_analysis import (  # noqa: E402
    MORGOTH_DIR, MONTAGE_MONO, MONO_CHANNELS,
    load_and_preprocess_edf, apply_montage_and_infer, per_second_gt,
    save_continuous_csv, metrics, _init_worker, _WORKER,
)
import utils as morgoth_utils  # noqa: E402

OUTPUT_DIR        = HERE
SZ_RESULTS_DIR    = os.path.join(OUTPUT_DIR, 'results')              # source of seizure JSONs
IIC_RESULTS_DIR   = os.path.join(OUTPUT_DIR, 'results_interictal')   # our outputs
EMU_IIC_DIR       = os.path.join(MORGOTH_DIR, 'data', 'interictal')


# ── discover matched interictal files ────────────────────────────────────────
import re as _re

_PATIENT_RE = _re.compile(r'^(EMU\d+)_seizure_\d+\.edf$')


def _seizure_edfs_from_results(montages: dict, sz_results_dir: str, log) -> set[str]:
    """Scan results/ for `<montage>_<file>.edf.json`, return set of seizure EDF
    basenames that have at least one cached montage result.
    """
    found: set[str] = set()
    try:
        entries = os.listdir(sz_results_dir)
    except FileNotFoundError:
        log.warning(f"no seizure results dir: {sz_results_dir}")
        return found
    for m in montages:
        prefix = f"{m}_"
        for fn in entries:
            if fn.startswith(prefix) and fn.endswith('.edf.json'):
                sz_edf = fn[len(prefix):-len('.json')]
                if '_seizure_' in sz_edf:
                    found.add(sz_edf)
    return found


def discover_matched_files(montages: dict, sz_results_dir: str, iic_dir: str,
                            mode: str, log) -> list[str]:
    """Build list of interictal EDF paths to process.

    mode='clip'    : paired by patient + clip index (EMU0878_seizure_3 -> EMU0878_iic_3)
    mode='patient' : all interictal EDFs for any patient seen in seizure cache
                     (default; better for population-level FP rates)
    """
    sz_set = _seizure_edfs_from_results(montages, sz_results_dir, log)
    if not sz_set:
        return []

    if mode == 'clip':
        wanted: set[str] = set()
        missing: set[str] = set()
        for sz_edf in sz_set:
            iic_edf = sz_edf.replace('_seizure_', '_iic_')
            iic_path = os.path.join(iic_dir, iic_edf)
            if os.path.exists(iic_path):
                wanted.add(iic_path)
            else:
                missing.add(iic_edf)
        if missing:
            log.warning(f"{len(missing)} paired iic files missing on disk "
                        f"(e.g. {sorted(missing)[:3]})")
        return sorted(wanted)

    # mode == 'patient'
    patients: set[str] = set()
    for sz_edf in sz_set:
        m = _PATIENT_RE.match(sz_edf)
        if m:
            patients.add(m.group(1))
    log.info(f"Patients with cached seizure results: {len(patients)}")

    try:
        all_iic = os.listdir(iic_dir)
    except FileNotFoundError:
        log.warning(f"no interictal dir: {iic_dir}")
        return []

    wanted = set()
    for fn in all_iic:
        if not fn.endswith('.edf') or '_iic_' not in fn:
            continue
        pid = fn.split('_iic_')[0]
        if pid in patients:
            wanted.add(os.path.join(iic_dir, fn))
    return sorted(wanted)


# ── per-file worker (interictal: ground truth all zeros) ─────────────────────
def process_one_iic_file(task: dict) -> dict:
    edf_path    = task['edf_path']
    montages    = task['montages']
    results_dir = task['results_dir']

    device      = _WORKER['device']
    batch_size  = _WORKER['batch_size']
    bb          = _WORKER['bb']
    eeg         = _WORKER['eeg']
    input_chans = _WORKER['input_chans']

    fname = os.path.basename(edf_path)
    out: dict[str, dict] = {}

    # fast path: all montage caches present
    all_cached = all(
        os.path.exists(os.path.join(results_dir, f'{m}_{fname}.json'))
        for m in montages
    )
    if all_cached:
        for m in montages:
            try:
                with open(os.path.join(results_dir, f'{m}_{fname}.json')) as fh:
                    c = json.load(fh)
                out[m] = {
                    'row': {
                        'file': fname,
                        'file_true': c['file_true'],
                        'file_pred': c['file_pred'],
                        'eeg_prob':  c['eeg_prob'],
                        'n_windows': c['n_windows'],
                    },
                    'win_pred':  c['win_pred'],
                    'file_true': c['file_true'],
                    'file_pred': c['file_pred'],
                    'from_cache': True,
                }
            except Exception as e:
                out[m] = {'error': f'cache read failed: {e}'}
        return {'file': fname, 'montages': out}

    full_sig, label_df, orig_fs, available, err = load_and_preprocess_edf(edf_path)
    if err:
        for m in montages:
            out[m] = {'error': err}
        return {'file': fname, 'montages': out, 'preprocess_error': err}

    for m, keep in montages.items():
        cache_path = os.path.join(results_dir, f'{m}_{fname}.json')
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as fh:
                    c = json.load(fh)
                out[m] = {
                    'row': {
                        'file': fname,
                        'file_true': c['file_true'],
                        'file_pred': c['file_pred'],
                        'eeg_prob':  c['eeg_prob'],
                        'n_windows': c['n_windows'],
                    },
                    'win_pred':  c['win_pred'],
                    'file_true': c['file_true'],
                    'file_pred': c['file_pred'],
                    'from_cache': True,
                }
                continue
            except Exception:
                pass

        res, merr = apply_montage_and_infer(
            full_sig, set(keep), available, bb, eeg, device,
            input_chans, batch_size,
        )
        if merr:
            out[m] = {'error': merr}
            continue

        n_win  = res['n_windows']
        # interictal: ground truth is all zeros (no seizure annotated)
        gt_sec = np.zeros(n_win, dtype=int)
        wp     = res['windowed_preds'].tolist()
        gt_l   = gt_sec.tolist()
        file_true = 0   # interictal = negative class

        cont_csv = os.path.join(results_dir, m, fname.replace('.edf', '.csv'))
        save_continuous_csv(cont_csv, res['sz_probs'], res['windowed_preds'], gt_sec)

        cache_payload = {
            'win_gt': gt_l, 'win_pred': wp,
            'file_true': file_true, 'file_pred': res['file_pred'],
            'eeg_prob':  res['eeg_prob'], 'n_windows': n_win,
        }
        with open(cache_path, 'w') as fh:
            json.dump(cache_payload, fh)

        out[m] = {
            'row': {
                'file': fname,
                'file_true': file_true,
                'file_pred': res['file_pred'],
                'eeg_prob':  res['eeg_prob'],
                'n_windows': n_win,
            },
            'win_pred':  wp,
            'file_true': file_true,
            'file_pred': res['file_pred'],
            'from_cache': False,
        }

    return {'file': fname, 'montages': out}


def init_log(p: str, n_files: int, montages: dict) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    Path(p).write_text(
        f"# Morgoth Interictal Analysis — Run Log\n\n"
        f"**Started:** {ts}\n\n"
        f"**Interictal dir:** {EMU_IIC_DIR}\n\n"
        f"**Matched files:** {n_files}\n\n"
        f"**Montages ({len(montages)}):** {list(montages.keys())}\n\n"
        "---\n\n"
    )


def alog(p: str, txt: str) -> None:
    with open(p, 'a') as f:
        f.write(txt + '\n')


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_files',  type=int, default=None)
    ap.add_argument('--montage',    type=str, default=None,
                    help='Run only this single montage (else all matched).')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--device',     type=str, default='cpu')
    ap.add_argument('--iic_dir',    type=str, default=EMU_IIC_DIR)
    ap.add_argument('--sz_results_dir', type=str, default=SZ_RESULTS_DIR,
                    help='Where seizure JSONs live (used to derive matched set).')
    ap.add_argument('--match', type=str, default='patient', choices=['patient', 'clip'],
                    help="'patient' (default): all iic clips for any patient with "
                         "a cached seizure result. 'clip': strict 1:1 pair by index "
                         "(EMU####_seizure_N -> EMU####_iic_N).")
    ap.add_argument('--out_dir',    type=str, default=IIC_RESULTS_DIR)
    ap.add_argument('--workers',    type=int, default=1)
    ap.add_argument('--torch_threads', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, 'ANALYSIS_LOG_INTERICTAL.md')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, 'run_interictal.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger(__name__)

    montages = {k: v for k, v in MONTAGE_MONO.items()
                if args.montage is None or k == args.montage}
    log.info(f"Montages requested: {list(montages.keys())}")

    # ensure per-montage subdirs
    for m in montages:
        os.makedirs(os.path.join(args.out_dir, m), exist_ok=True)

    # discover the matched interictal files
    log.info(f"Match mode: {args.match}")
    files = discover_matched_files(
        montages, args.sz_results_dir, args.iic_dir, args.match, log)
    if args.max_files:
        files = files[:args.max_files]
    log.info(f"Matched interictal files to process: {len(files)}")
    init_log(log_path, len(files), montages)

    if not files:
        log.warning("Nothing to do — no matched interictal files found.")
        return

    # parallelism
    n_workers = max(1, int(args.workers))
    per_worker_threads = (args.torch_threads
                          if args.torch_threads and args.torch_threads > 0
                          else max(1, (os.cpu_count() or 4) // n_workers))
    log.info(f"Workers: {n_workers}  |  torch threads/worker: {per_worker_threads}")

    accum = {m: {'win_pred': [], 'file_true': [], 'file_pred': [],
                 'rows': [], 'errors': []}
             for m in montages}

    def _merge(file_result: dict) -> None:
        fname = file_result['file']
        if 'preprocess_error' in file_result:
            log.warning(f"SKIP {fname}: {file_result['preprocess_error']}")
        for m, payload in file_result['montages'].items():
            if 'error' in payload:
                log.warning(f"  SKIP {fname}/{m}: {payload['error']}")
                accum[m]['errors'].append({'file': fname, 'error': payload['error']})
                continue
            accum[m]['win_pred'].extend(payload['win_pred'])
            accum[m]['file_true'].append(payload['file_true'])
            accum[m]['file_pred'].append(payload['file_pred'])
            accum[m]['rows'].append(payload['row'])

    tasks = [{
        'edf_path':    p,
        'montages':    montages,
        'results_dir': args.out_dir,
    } for p in files]

    if n_workers == 1:
        log.info("Loading models (serial)…")
        _init_worker(args.device, args.batch_size, per_worker_threads)
        log.info("Loaded. Beginning serial run.")
        for t in tqdm(tasks, desc='iic_files', unit='file'):
            _merge(process_one_iic_file(t))
    else:
        log.info(f"Launching ProcessPoolExecutor with {n_workers} workers…")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(args.device, args.batch_size, per_worker_threads),
        ) as pool:
            futs = [pool.submit(process_one_iic_file, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc='iic_files', unit='file'):
                try:
                    _merge(fut.result())
                except Exception as e:
                    log.exception(f"worker failed: {e}")

    # ── summary (interictal: all-negative GT, so report FP rate) ─────────────
    summary_rows = []
    for m in montages:
        a    = accum[m]
        n    = len(a['file_true'])
        n_fp = int(np.sum(a['file_pred'])) if n else 0
        n_win = len(a['win_pred'])
        n_win_fp = int(np.sum(a['win_pred'])) if n_win else 0
        nerr  = len(a['errors'])

        file_fp_rate = (n_fp / n) if n else float('nan')
        win_fp_rate  = (n_win_fp / n_win) if n_win else float('nan')

        log.info(f"\n[{m}]")
        log.info(f"  file-level: false-positive rate = {file_fp_rate:.3f}  "
                 f"({n_fp}/{n} clips flagged seizure)")
        log.info(f"  windowed:   false-positive rate = {win_fp_rate:.3f}  "
                 f"({n_win_fp}/{n_win} windows flagged)")

        pd.DataFrame(a['rows']).to_csv(
            os.path.join(args.out_dir, f'{m}_per_file.csv'), index=False)
        if a['errors']:
            pd.DataFrame(a['errors']).to_csv(
                os.path.join(args.out_dir, f'{m}_errors.csv'), index=False)

        alog(log_path,
             f"\n---\n## Montage: `{m}`\n"
             f"**Channels:** {montages[m]}\n\n"
             f"| Level | FP rate | N flagged | N total |\n"
             f"|---|---|---|---|\n"
             f"| Windowed (1s) | {win_fp_rate:.3f} | {n_win_fp} | {n_win} |\n"
             f"| File-level    | {file_fp_rate:.3f} | {n_fp} | {n} |\n\n"
             f"Errors/skipped: {nerr}\n")

        summary_rows.append({
            'montage':         m,
            'n_files':         n,
            'file_fp_count':   n_fp,
            'file_fp_rate':    file_fp_rate,
            'n_windows':       n_win,
            'window_fp_count': n_win_fp,
            'window_fp_rate':  win_fp_rate,
            'n_errors':        nerr,
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics_interictal.csv'), index=False)
    log.info(f"\n{'='*60}\nSummary:\n{df_sum.to_string(index=False)}")

    cols = list(df_sum.columns)
    hdr  = '| ' + ' | '.join(cols) + ' |'
    sep  = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    rows = []
    for _, r in df_sum.iterrows():
        cells = [f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c]) for c in cols]
        rows.append('| ' + ' | '.join(cells) + ' |')
    alog(log_path, f"\n---\n## Summary\n\n" + '\n'.join([hdr, sep] + rows)
         + f"\n\n**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
