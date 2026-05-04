"""
Morgoth Interictal Analysis (Linux/GPU)
========================================
GPU-optimised version of run_interictal_analysis.py.
Imports shared infrastructure from run_seizure_analysis_linux.py.

Usage:
  conda run -n morgoth python montage_analysis/run_interictal_analysis_linux.py \\
        --patient EMU0878 --device cuda
  conda run -n morgoth python montage_analysis/run_interictal_analysis_linux.py \\
        --device cuda --match patient
"""

import os
import sys
import re as _re
import time
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Import all shared infrastructure from the linux seizure script
from run_seizure_analysis_linux import (
    MORGOTH_DIR, MONTAGE_MONO, MONO_CHANNELS,
    load_and_preprocess_edf, apply_all_montages_batched,
    per_second_gt, save_continuous_csv, metrics,
    _init_worker, _WORKER,
)
import utils as morgoth_utils

OUTPUT_DIR      = HERE
SZ_RESULTS_DIR  = os.path.join(OUTPUT_DIR, 'results_gpu')
IIC_RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results_interictal_gpu')
EMU_IIC_DIR     = os.path.join(MORGOTH_DIR, 'data', 'interictal')

_PATIENT_RE = _re.compile(r'^(EMU\d+)_seizure_\d+\.edf$')


# ── discover matched interictal files ────────────────────────────────────────
def _seizure_edfs_from_results(montages: dict, sz_results_dir: str, log) -> set:
    found: set = set()
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
                            mode: str, patient: str | None, log) -> list:
    sz_set = _seizure_edfs_from_results(montages, sz_results_dir, log)
    if not sz_set:
        return []

    if patient:
        sz_set = {f for f in sz_set if f.startswith(patient)}
        log.info(f"--patient {patient}: {len(sz_set)} cached seizure EDFs match")

    if mode == 'clip':
        wanted: set = set()
        missing: set = set()
        for sz_edf in sz_set:
            iic_edf  = sz_edf.replace('_seizure_', '_iic_')
            iic_path = os.path.join(iic_dir, iic_edf)
            if os.path.exists(iic_path):
                wanted.add(iic_path)
            else:
                missing.add(iic_edf)
        if missing:
            log.warning(f"{len(missing)} paired iic files missing "
                        f"(e.g. {sorted(missing)[:3]})")
        return sorted(wanted)

    # mode == 'patient'
    patients: set = set()
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


# ── per-file worker (all montages batched; GT = all zeros) ───────────────────
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
    out: dict = {}

    # Fast path: all montage caches present
    if all(os.path.exists(os.path.join(results_dir, f'{m}_{fname}.json'))
           for m in montages):
        for m in montages:
            try:
                with open(os.path.join(results_dir, f'{m}_{fname}.json')) as fh:
                    c = json.load(fh)
                out[m] = {
                    'row':       {'file': fname, 'file_true': c['file_true'],
                                  'file_pred': c['file_pred'], 'eeg_prob': c['eeg_prob'],
                                  'n_windows': c['n_windows']},
                    'win_pred':  c['win_pred'],
                    'file_true': c['file_true'], 'file_pred': c['file_pred'],
                    'from_cache': True,
                }
            except Exception as e:
                out[m] = {'error': f'cache read failed: {e}'}
        return {'file': fname, 'montages': out}

    full_sig, label_df, orig_fs, available, err = load_and_preprocess_edf(edf_path)
    if err:
        return {'file': fname, 'montages': {m: {'error': err} for m in montages},
                'preprocess_error': err}

    pending, cached_m = {}, {}
    for m, keep in montages.items():
        cp = os.path.join(results_dir, f'{m}_{fname}.json')
        if os.path.exists(cp):
            try:
                with open(cp) as fh:
                    cached_m[m] = json.load(fh)
                continue
            except Exception:
                pass
        pending[m] = keep

    if pending:
        batch_results = apply_all_montages_batched(
            full_sig, pending, available, bb, eeg, device, input_chans, batch_size,
        )
    else:
        batch_results = {}

    for m, keep in montages.items():
        if m in cached_m:
            c = cached_m[m]
            out[m] = {
                'row':       {'file': fname, 'file_true': c['file_true'],
                              'file_pred': c['file_pred'], 'eeg_prob': c['eeg_prob'],
                              'n_windows': c['n_windows']},
                'win_pred':  c['win_pred'],
                'file_true': c['file_true'], 'file_pred': c['file_pred'],
                'from_cache': True,
            }
            continue

        res, merr = batch_results.get(m, (None, "missing from batch"))
        if merr:
            out[m] = {'error': merr}
            continue

        n_win  = res['n_windows']
        gt_sec = np.zeros(n_win, dtype=int)   # interictal: all negative
        wp     = res['windowed_preds'].tolist()
        file_true = 0

        cont_csv = os.path.join(results_dir, m, fname.replace('.edf', '.csv'))
        save_continuous_csv(cont_csv, res['sz_probs'], res['windowed_preds'], gt_sec)

        cache_payload = {
            'win_gt': gt_sec.tolist(), 'win_pred': wp,
            'file_true': file_true, 'file_pred': res['file_pred'],
            'eeg_prob': res['eeg_prob'], 'n_windows': n_win,
        }
        with open(os.path.join(results_dir, f'{m}_{fname}.json'), 'w') as fh:
            json.dump(cache_payload, fh)

        out[m] = {
            'row':       {'file': fname, 'file_true': file_true,
                          'file_pred': res['file_pred'], 'eeg_prob': res['eeg_prob'],
                          'n_windows': n_win},
            'win_pred':  wp,
            'file_true': file_true, 'file_pred': res['file_pred'],
            'from_cache': False,
        }

    return {'file': fname, 'montages': out}


def init_log(p: str, n_files: int, montages: dict) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    Path(p).write_text(
        f"# Morgoth Interictal Analysis (GPU) — Run Log\n\n"
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
    ap.add_argument('--max_files',   type=int, default=None)
    ap.add_argument('--patient',     type=str, default=None,
                    help='Filter to a single patient, e.g. EMU0878.')
    ap.add_argument('--montage',     type=str, default=None)
    ap.add_argument('--batch_size',  type=int, default=256)
    ap.add_argument('--device',      type=str, default='cuda')
    ap.add_argument('--iic_dir',     type=str, default=EMU_IIC_DIR)
    ap.add_argument('--sz_results_dir', type=str, default=SZ_RESULTS_DIR)
    ap.add_argument('--match',       type=str, default='patient',
                    choices=['patient', 'clip'])
    ap.add_argument('--out_dir',     type=str, default=None)
    ap.add_argument('--workers',     type=int, default=1)
    ap.add_argument('--torch_threads', type=int, default=0)
    args = ap.parse_args()

    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    out_dir  = args.out_dir or IIC_RESULTS_DIR
    log_path = os.path.join(OUTPUT_DIR, 'ANALYSIS_LOG_INTERICTAL_gpu.md')
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, 'run_interictal_gpu.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger(__name__)

    montages = {k: v for k, v in MONTAGE_MONO.items()
                if args.montage is None or k == args.montage}
    log.info(f"Montages: {list(montages.keys())}")

    for m in montages:
        os.makedirs(os.path.join(out_dir, m), exist_ok=True)

    log.info(f"Match mode: {args.match}")
    files = discover_matched_files(
        montages, args.sz_results_dir, args.iic_dir,
        args.match, args.patient, log,
    )
    if args.max_files:
        files = files[:args.max_files]
    log.info(f"Matched interictal files to process: {len(files)}")
    init_log(log_path, len(files), montages)

    if not files:
        log.warning("Nothing to do — no matched interictal files found. "
                    "Run the seizure script first to populate results_gpu/.")
        return

    n_workers = max(1, int(args.workers))
    per_worker_threads = (args.torch_threads if args.torch_threads > 0
                          else max(1, (os.cpu_count() or 4) // n_workers))
    log.info(f"Workers: {n_workers}  |  torch_threads/worker: {per_worker_threads}  "
             f"|  batch_size: {args.batch_size}")

    accum = {m: {'win_pred': [], 'file_true': [], 'file_pred': [],
                 'rows': [], 'errors': []}
             for m in montages}
    file_times: list[tuple[str, float]] = []

    def _merge(file_result: dict, elapsed: float) -> None:
        fname = file_result['file']
        file_times.append((fname, elapsed))
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

    tasks = [{'edf_path': p, 'montages': montages, 'results_dir': out_dir}
             for p in files]

    total_t0 = time.perf_counter()

    if n_workers == 1:
        log.info("Loading models (serial)…")
        _init_worker(args.device, args.batch_size, per_worker_threads)
        log.info("Loaded. Beginning run.")
        for t in tqdm(tasks, desc='iic_files', unit='file'):
            t0 = time.perf_counter()
            r  = process_one_iic_file(t)
            elapsed = time.perf_counter() - t0
            cached = all(v.get('from_cache') for v in r['montages'].values()
                         if 'error' not in v)
            log.info(f"  {r['file']}  {elapsed:.1f}s"
                     + ("  [cached]" if cached else ""))
            _merge(r, elapsed)
    else:
        log.info(f"Launching {n_workers} worker processes…")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(args.device, args.batch_size, per_worker_threads),
        ) as pool:
            futs = [pool.submit(process_one_iic_file, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc='iic_files', unit='file'):
                try:
                    r = fut.result()
                    _merge(r, 0.0)
                except Exception as e:
                    log.exception(f"worker failed: {e}")

    total_elapsed = time.perf_counter() - total_t0
    log.info(f"\nTotal wall time: {total_elapsed:.1f}s  "
             f"({total_elapsed/max(len(files),1):.1f}s/file avg)")
    if file_times:
        log.info("Per-file times:")
        for fn, t in file_times:
            log.info(f"  {fn}: {t:.1f}s")

    alog(log_path,
         f"\n## Timing\n- Total: {total_elapsed:.1f}s\n"
         + "\n".join(f"- {fn}: {t:.1f}s" for fn, t in file_times) + "\n")

    # ── FP rate summary (all-negative GT) ─────────────────────────────────────
    summary_rows = []
    for m in montages:
        a     = accum[m]
        n     = len(a['file_true'])
        n_fp  = int(np.sum(a['file_pred'])) if n else 0
        n_win = len(a['win_pred'])
        n_wfp = int(np.sum(a['win_pred'])) if n_win else 0
        nerr  = len(a['errors'])

        file_fp_rate = n_fp / n       if n     else float('nan')
        win_fp_rate  = n_wfp / n_win  if n_win else float('nan')

        log.info(f"\n[{m}]  file FP rate={file_fp_rate:.3f} ({n_fp}/{n})  "
                 f"windowed FP rate={win_fp_rate:.3f} ({n_wfp}/{n_win})")

        pd.DataFrame(a['rows']).to_csv(
            os.path.join(out_dir, f'{m}_per_file.csv'), index=False)
        if a['errors']:
            pd.DataFrame(a['errors']).to_csv(
                os.path.join(out_dir, f'{m}_errors.csv'), index=False)

        alog(log_path,
             f"\n---\n## Montage: `{m}`\n"
             f"**Channels:** {montages[m]}\n\n"
             f"| Level | FP rate | N flagged | N total |\n"
             f"|---|---|---|---|\n"
             f"| Windowed (1s) | {win_fp_rate:.3f} | {n_wfp} | {n_win} |\n"
             f"| File-level    | {file_fp_rate:.3f} | {n_fp} | {n} |\n\n"
             f"Errors/skipped: {nerr}\n")

        summary_rows.append({
            'montage': m, 'n_files': n,
            'file_fp_count': n_fp, 'file_fp_rate': file_fp_rate,
            'n_windows': n_win, 'window_fp_count': n_wfp,
            'window_fp_rate': win_fp_rate, 'n_errors': nerr,
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics_interictal_gpu.csv'),
                  index=False)
    log.info(f"\n{'='*60}\nSummary:\n{df_sum.to_string(index=False)}")

    cols    = list(df_sum.columns)
    hdr     = '| ' + ' | '.join(cols) + ' |'
    sep     = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    rows_md = []
    for _, r in df_sum.iterrows():
        cells = [f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c])
                 for c in cols]
        rows_md.append('| ' + ' | '.join(cells) + ' |')
    alog(log_path, f"\n---\n## Summary\n\n" + '\n'.join([hdr, sep] + rows_md)
         + f"\n\n**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
