"""
Morgoth Seizure Detection — Montage Analysis
============================================
Tests different EEG montage configurations (simulating wearable devices)
on the EMU dataset using the morgoth IIIC + SEIZURE_EEGlevel models.

Outputs:
  results/<montage>_per_file.csv         — per-file predictions and windowed F1
  results/<montage>_<file>.json          — per-file cache (resumable)
  results/<montage>/<basename>.csv       — continuous per-window predictions
                                           (columns: timestamp, sz_prob, pred, label)
  summary_metrics.csv                    — aggregate F1/sens/prec across montages
  ANALYSIS_LOG.md                        — human-readable run log
  run.log                                — full run log

Usage:
  conda run -n morgoth python montage_analysis/run_seizure_analysis.py
  conda run -n morgoth python montage_analysis/run_seizure_analysis.py --max_files 10
  conda run -n morgoth python montage_analysis/run_seizure_analysis.py --montage full
"""

import os
import sys
import re
import argparse
import logging
import json
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mne
import torch
import torch.nn.functional as F
from scipy.signal import resample as scipy_resample, butter, filtfilt, iirnotch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from einops import rearrange
from timm.models import create_model

warnings.filterwarnings("ignore")
mne.set_log_level('WARNING')

# ── path setup ────────────────────────────────────────────────────────────────
MORGOTH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MORGOTH_DIR)

import backbone   # registers timm custom models (base_patch200_200 etc.)
import utils as morgoth_utils
from EEG_level_head import CNNTransformerClassifier, load_model_parameters

OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
# Local data lives in <morgoth>/data/{seizure,interictal}; override with
# --sz_dir / --iic_dir to point somewhere else.
EMU_SZ_DIR  = os.path.join(MORGOTH_DIR, 'data', 'seizure')
EMU_IIC_DIR = os.path.join(MORGOTH_DIR, 'data', 'interictal')
IIIC_CKPT   = os.path.join(MORGOTH_DIR, 'checkpoints', 'IIIC.pth')
SZ_LVL_CKPT = os.path.join(MORGOTH_DIR, 'checkpoints', 'SEIZURE_EEGlevel.pth')

# ── EEG constants ─────────────────────────────────────────────────────────────
FS_TARGET = 200
WIN_SEC   = 10
STEP_SEC  = 1
WIN_SAMP  = WIN_SEC  * FS_TARGET   # 2000 samples
STEP_SAMP = STEP_SEC * FS_TARGET   # 200 samples

# Standard 19-channel monopolar order used by morgoth
MONO_CHANNELS = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                 'T4', 'T6', 'O2']
MONO_IDX = {ch: i for i, ch in enumerate(MONO_CHANNELS)}

# ── montage definitions ───────────────────────────────────────────────────────
def _mono_from_pairs(pairs):
    needed = set()
    for p in pairs:
        a, b = p.upper().split('-')
        needed.add(a)
        needed.add(b)
    return sorted(needed)

MONTAGE_MONO = {
    'full':
        _mono_from_pairs(['FP1-F7','F7-T3','T3-T5','T5-O1',
                          'FP2-F8','F8-T4','T4-T6','T6-O2',
                          'FP1-F3','F3-C3','C3-P3','P3-O1',
                          'FP2-F4','F4-C4','C4-P4','P4-O2']),
    'uneeg_left_front':          _mono_from_pairs(['F7-T3']),
    'uneeg_left_back':           _mono_from_pairs(['T3-T5']),
    'uneeg_right_front':         _mono_from_pairs(['F8-T4']),
    'uneeg_right_back':          _mono_from_pairs(['T4-T6']),
    'uneeg_bilateral_back2':     _mono_from_pairs(['T3-T5','T4-T6']),
    'uneeg_bilateral_front2':    _mono_from_pairs(['F7-T3','F8-T4']),
    'uneeg_vert_left':           _mono_from_pairs(['C3-T3']),
    'uneeg_vert_right':          _mono_from_pairs(['C4-T4']),
    'uneeg_diag_left_front':     _mono_from_pairs(['F3-T3']),
    'uneeg_diag_left_back':      _mono_from_pairs(['P3-T3']),
    'uneeg_diag_right_front':    _mono_from_pairs(['F4-T4']),
    'uneeg_diag_right_back':     _mono_from_pairs(['P4-T4']),
    'uneeg_diag_bilateral_front':_mono_from_pairs(['F3-T3','F4-T4']),
    'uneeg_diag_bilateral_back': _mono_from_pairs(['P3-T3','P4-T4']),
    'uneeg_vert_bilateral':      _mono_from_pairs(['C3-T3','C4-T4']),
    'epiminder_2':               _mono_from_pairs(['C3-P3','C4-P4']),
    'ceribell':                  _mono_from_pairs(['FP1-F7','F7-T3','T3-T5','T5-O1',
                                                   'FP2-F8','F8-T4','T4-T6','T6-O2']),
}

# ── signal preprocessing ──────────────────────────────────────────────────────
_BP_B, _BP_A = butter(4, [0.5/100, 70.0/100], btype='band')
_NO_B, _NO_A = iirnotch(60.0/100, 30.0)

def bandpass_notch(sig):
    sig = filtfilt(_BP_B, _BP_A, sig, axis=1).astype(np.float32)
    sig = filtfilt(_NO_B, _NO_A, sig, axis=1).astype(np.float32)
    return sig

def resample_to_target(sig, orig_fs):
    if int(orig_fs) == FS_TARGET:
        return sig
    n = int(sig.shape[1] * FS_TARGET / orig_fs)
    return scipy_resample(sig, n, axis=1).astype(np.float32)

def clip_and_scale(sig):
    """Clip at ±500 µV, per-channel min-max scale to [-100, 100].

    Matches Morgoth's official `utils.Clipping()` + `utils.Scaling()` transforms
    (see `finetune_classification.py`). The model forward pass then divides by
    100 internally, yielding input in [-1, 1].
    """
    sig = np.clip(sig, -500.0, 500.0)
    for ch in range(sig.shape[0]):
        mn, mx = sig[ch].min(), sig[ch].max()
        if mx - mn > 1e-6:
            sig[ch] = 200.0 * (sig[ch] - mn) / (mx - mn) - 100.0
    return sig.astype(np.float32)

# ── EDF loading ───────────────────────────────────────────────────────────────
def load_edf_file(file_name):
    raw = mne.io.read_raw_edf(file_name, preload=True, verbose=0)
    fs  = raw.info['sfreq']
    df  = raw.to_data_frame().set_index('time')
    times = raw.times
    annotations = raw.annotations
    label = np.zeros(len(times), dtype=int)
    if annotations:
        for anno in annotations:
            sz_onset = anno['onset']
            sz_dura  = anno['duration']
            sz_end   = sz_onset + sz_dura
            label[(times >= sz_onset) & (times <= sz_end)] = 1
    label_df = pd.DataFrame({'time': times, 'labels': label})
    return raw, df, label_df, fs

def clean_channel_names(raw):
    """Normalise channel names: strip prefixes, uppercase, deduplicate."""
    raw.rename_channels(
        {n: n.replace('EEG','').replace('eeg','').replace('POL','').replace('pol','').strip()
         for n in raw.ch_names})
    new_names = {ch: re.sub(r"\(.*?\)", "", ch).split('-')[0].strip().upper()
                 for ch in raw.ch_names}
    cnt = Counter(new_names.values())
    final = {old: (new if cnt[new] == 1 else old) for old, new in new_names.items()}
    raw.rename_channels(final)
    return raw

def load_and_preprocess_edf(edf_path):
    """
    Load EDF → clean channels → build full 19-channel monopolar array
    resampled to 200 Hz and bandpass/notch filtered.

    Returns:
        full_sig  : (19, n_samp) float32 — zeros for unavailable channels
        label_df  : per-sample seizure labels
        orig_fs   : original sampling rate
        available : list of MONO_CHANNELS found in file
    """
    try:
        raw, _, label_df, orig_fs = load_edf_file(edf_path)
    except Exception as e:
        return None, None, None, None, f"load error: {e}"

    raw = clean_channel_names(raw)

    available = [ch for ch in MONO_CHANNELS if ch in raw.ch_names]
    if not available:
        return None, None, None, None, "no standard channels"

    raw_sel = raw.copy().pick(available)
    sig_raw = raw_sel.get_data(units='uV').astype(np.float32)

    # resample + filter
    sig_rs  = resample_to_target(sig_raw, orig_fs)
    sig_rs  = bandpass_notch(sig_rs)

    # place into full 19-channel array
    n_samp   = sig_rs.shape[1]
    full_sig = np.zeros((19, n_samp), dtype=np.float32)
    for i, ch in enumerate(available):
        full_sig[MONO_IDX[ch]] = sig_rs[i]

    return full_sig, label_df, float(orig_fs), available, None

# ── per-montage inference ─────────────────────────────────────────────────────
def apply_montage_and_infer(full_sig, keep_channels, available,
                             backbone, eeg_level_model, device,
                             input_chans, batch_size):
    """
    Given the full 19-channel preprocessed signal, apply montage masking,
    run backbone and EEG-level head.

    Returns (sz_probs, windowed_preds, n_windows) or raises.
    """
    active = [ch for ch in keep_channels if ch in available]
    if not active:
        return None, "no active channels for montage"

    # zero out channels not in keep_channels
    masked = np.zeros_like(full_sig)
    for ch in active:
        masked[MONO_IDX[ch]] = full_sig[MONO_IDX[ch]]

    # common-average montage over active channels only
    active_idx = [MONO_IDX[ch] for ch in active]
    avg = masked[active_idx].mean(axis=0, keepdims=True)
    montaged = np.zeros_like(masked)
    for idx in active_idx:
        montaged[idx] = masked[idx] - avg[0]

    montaged = clip_and_scale(montaged)

    n_samp = montaged.shape[1]
    if n_samp < WIN_SAMP:
        return None, "file too short"

    n_windows = (n_samp - WIN_SAMP) // STEP_SAMP + 1
    windows   = np.lib.stride_tricks.sliding_window_view(
        montaged, WIN_SAMP, axis=1)[:, ::STEP_SAMP, :]  # (19, n_win, 2000)
    windows   = windows.transpose(1, 0, 2)[:n_windows]  # (n_win, 19, 2000)

    win_tensor = torch.from_numpy(windows)

    # backbone → (n_win, 6) logits
    backbone.eval()
    all_logits = []
    with torch.no_grad():
        for s in range(0, n_windows, batch_size):
            batch = win_tensor[s:s+batch_size].to(device)
            batch = batch / 100.0
            batch = rearrange(batch, 'B N (A T) -> B N A T', T=200)
            ac_type = 'cpu' if str(device) == 'cpu' else 'cpu'  # MPS autocast not stable; skip
            with torch.autocast(device_type=ac_type, enabled=(ac_type == 'cpu')):
                out = backbone(batch, input_chans=input_chans)
            all_logits.append(out.cpu().to(torch.float32))
    logits = torch.cat(all_logits, dim=0)
    probs  = F.softmax(logits, dim=-1).numpy()   # (n_win, 6)

    sz_probs      = probs[:, 1]                  # SEIZURE class
    windowed_preds = (sz_probs > 0.5).astype(int)

    # EEG-level head (always CPU — MPS missing some transformer ops)
    eeg_in  = torch.tensor(probs, dtype=torch.float32).unsqueeze(0)  # (1, n_win, 6)
    lengths = torch.tensor([n_windows], dtype=torch.long)
    eeg_level_model.eval()
    with torch.no_grad():
        eeg_out  = eeg_level_model(eeg_in, lengths=lengths)   # CPU tensors
        eeg_prob = torch.sigmoid(eeg_out).item()

    return {
        'sz_probs':       sz_probs,
        'windowed_preds': windowed_preds,
        'eeg_prob':       eeg_prob,
        'file_pred':      int(eeg_prob > 0.5),
        'n_windows':      n_windows,
    }, None

# ── ground truth helper ───────────────────────────────────────────────────────
def per_second_gt(label_df, orig_fs, n_windows):
    labels  = label_df['labels'].values
    fs_int  = int(round(orig_fs))
    win_samp = int(round(WIN_SEC * orig_fs))  # full 10-second window at orig_fs
    gt      = np.zeros(n_windows, dtype=int)
    for i in range(n_windows):
        s = i * fs_int
        e = min(s + win_samp, len(labels))    # check full window, not just 1 stride second
        if e > s and labels[s:e].max() > 0:
            gt[i] = 1
    return gt

# ── continuous per-window CSV writer ──────────────────────────────────────────
def window_end_timestamps(n_windows: int) -> np.ndarray:
    """Seconds-since-clip-start for the END of each sliding window.

    Mirrors the SVM example's convention (window end minus 1 sample), so
    for WIN_SEC=10, STEP_SEC=1, FS_TARGET=200 the first ts = 9.995.
    """
    return WIN_SEC - (1.0 / FS_TARGET) + np.arange(n_windows) * STEP_SEC


def save_continuous_csv(out_path: str, sz_probs: np.ndarray,
                         windowed_preds: np.ndarray, gt: np.ndarray) -> None:
    """Write per-window predictions in the same shape as the SVM example:
    index (timestamp s), sz_prob, pred, label.
    """
    ts = window_end_timestamps(len(sz_probs))
    df = pd.DataFrame({
        'sz_prob': sz_probs.astype(float),
        'pred':    windowed_preds.astype(float),
        'label':   gt.astype(float),
    }, index=ts)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)


# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {'f1': float('nan'), 'sensitivity': float('nan'),
                'precision': float('nan'),
                'n_pos': int(y_true.sum()), 'n_total': len(y_true)}
    return {
        'f1':          f1_score(y_true, y_pred, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'precision':   precision_score(y_true, y_pred, zero_division=0),
        'n_pos':       int(y_true.sum()),
        'n_total':     len(y_true),
    }

# ── model loaders ─────────────────────────────────────────────────────────────
def load_backbone(device):
    class _A:
        model='base_patch200_200'; nb_classes=6; drop=0.0; drop_path=0.1
        attn_drop_rate=0.0; use_mean_pooling=True; init_scale=0.001
        rel_pos_bias=True; abs_pos_emb=True; layer_scale_init_value=0.1
        qkv_bias=True; task_model=IIIC_CKPT
    m = create_model(
        _A.model, pretrained=False, num_classes=_A.nb_classes,
        drop_rate=_A.drop, drop_path_rate=_A.drop_path,
        attn_drop_rate=_A.attn_drop_rate, drop_block_rate=None,
        use_mean_pooling=_A.use_mean_pooling, init_scale=_A.init_scale,
        use_rel_pos_bias=_A.rel_pos_bias, use_abs_pos_emb=_A.abs_pos_emb,
        init_values=_A.layer_scale_init_value, qkv_bias=_A.qkv_bias,
    ).to(torch.float32)
    morgoth_utils.load_from_task_model(args=_A, model_without_ddp=m)
    m.to(device).eval()
    return m

def load_eeg_level(device=None):
    # EEG-level head stays on CPU (MPS missing transformer ops)
    m = CNNTransformerClassifier(input_dim=6, output_dim=1, pe_max_length=15000)
    m = load_model_parameters(m, SZ_LVL_CKPT, device='cpu')
    return m.eval()

# ── log helpers ───────────────────────────────────────────────────────────────
def init_log(p):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    Path(p).write_text(
        f"# Morgoth Montage Analysis — Run Log\n\n"
        f"**Started:** {ts}\n\n"
        f"**Dataset:** {EMU_SZ_DIR}  |  {EMU_IIC_DIR}\n\n"
        f"**Models:** {IIIC_CKPT}  |  {SZ_LVL_CKPT}\n\n"
        "---\n\n"
    )

def alog(p, txt):
    with open(p, 'a') as f:
        f.write(txt + '\n')

# ── worker-process state for parallel execution ───────────────────────────────
# Each process (or the main process, in serial mode) loads the model once into
# these module-level globals and reuses them across files. Pickling models
# through a task queue per file would be wasteful.
_WORKER: dict[str, Any] = {}


def _init_worker(device_str: str, batch_size: int, torch_threads: int) -> None:
    """ProcessPoolExecutor initializer. Sets thread count, loads models."""
    # Cap each worker's BLAS thread pool so N workers don't oversubscribe
    # the physical cores. Call before loading the model so the first forward
    # pass uses the right thread count.
    os.environ.setdefault('OMP_NUM_THREADS',   str(torch_threads))
    os.environ.setdefault('MKL_NUM_THREADS',   str(torch_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(torch_threads))
    try:
        torch.set_num_threads(torch_threads)
    except Exception:
        pass

    device = torch.device(device_str)
    _WORKER['device']      = device
    _WORKER['batch_size']  = batch_size
    _WORKER['bb']          = load_backbone(device)
    _WORKER['eeg']         = load_eeg_level(device)
    _WORKER['input_chans'] = morgoth_utils.get_input_chans(MONO_CHANNELS)


def process_one_file(task: dict) -> dict:
    """Process a single EDF file across all requested montages.

    Uses the globals populated by `_init_worker` (or by `main()` in serial mode)
    for the backbone / EEG-level head / input_chans. All outputs (continuous
    CSV, per-file JSON cache) are written directly from here because each
    (montage, file) pair has a unique path, so there is no write contention
    between workers.

    Returns a dict keyed by montage name with either a populated result payload
    or an `error` entry. The caller merges these into the aggregate accumulator.
    """
    edf_path   = task['edf_path']
    is_seizure = task['is_seizure']
    montages   = task['montages']        # dict[str, list[str]]
    results_dir = task['results_dir']

    device      = _WORKER['device']
    batch_size  = _WORKER['batch_size']
    bb          = _WORKER['bb']
    eeg         = _WORKER['eeg']
    input_chans = _WORKER['input_chans']

    fname = os.path.basename(edf_path)
    out: dict[str, dict] = {}

    # Fast path: if every montage is already cached, just load caches.
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
                        'win_f1':    c.get('win_f1', float('nan')),
                    },
                    'win_gt':    c['win_gt'],
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
                        'win_f1':    c.get('win_f1', float('nan')),
                    },
                    'win_gt':    c['win_gt'],
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
        gt_sec = per_second_gt(label_df, orig_fs, n_win)
        wp     = res['windowed_preds'].tolist()
        gt_l   = gt_sec.tolist()

        win_f1 = (f1_score(gt_l, wp, zero_division=0)
                  if len(np.unique(gt_l)) > 1 else float('nan'))
        file_true = int(is_seizure)

        # continuous per-window CSV (SVM-example-shaped)
        cont_csv = os.path.join(results_dir, m, fname.replace('.edf', '.csv'))
        save_continuous_csv(
            cont_csv, res['sz_probs'], res['windowed_preds'], gt_sec,
        )

        cache_payload = {
            'win_gt': gt_l, 'win_pred': wp,
            'file_true': file_true, 'file_pred': res['file_pred'],
            'eeg_prob':  res['eeg_prob'], 'n_windows': n_win,
            'win_f1':    win_f1,
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
                'win_f1':    win_f1,
            },
            'win_gt':    gt_l,
            'win_pred':  wp,
            'file_true': file_true,
            'file_pred': res['file_pred'],
            'from_cache': False,
        }

    return {'file': fname, 'montages': out}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_files', type=int, default=None)
    ap.add_argument('--montage',   type=str, default=None)
    ap.add_argument('--batch_size',type=int, default=64)
    ap.add_argument('--device',    type=str, default='cpu')
    ap.add_argument('--sz_dir',    type=str, default=EMU_SZ_DIR)
    ap.add_argument('--iic_dir',   type=str, default=EMU_IIC_DIR)
    ap.add_argument('--workers',   type=int, default=1,
                    help='Number of parallel worker processes (file-level). '
                         '1 = serial (default). On an M2 Max (12 cores, 32 GB), '
                         '--workers 4 --torch_threads 3 is a reasonable starting point.')
    ap.add_argument('--torch_threads', type=int, default=0,
                    help='BLAS/torch threads per worker. 0 = auto (cores // workers).')
    ap.add_argument('--complete_patients', action='store_true',
                    help='Only process seizure clips for patients (EMU####) that '
                         'already have at least one cached montage JSON in results/. '
                         'Skips iic side entirely. Use to finish off partially-done '
                         'patients without expanding the cohort.')
    ap.add_argument('--skip_summary', action='store_true',
                    help='Skip per-montage rollup CSVs and summary_metrics.csv. '
                         'Use when resuming partial runs — only fills in missing '
                         'per-file JSON caches + windowed CSVs in results/<montage>/.')
    args = ap.parse_args()

    device = torch.device(args.device)
    sz_dir, iic_dir = args.sz_dir, args.iic_dir

    results_dir = os.path.join(OUTPUT_DIR, 'results')
    log_path    = os.path.join(OUTPUT_DIR, 'ANALYSIS_LOG.md')
    os.makedirs(results_dir, exist_ok=True)
    init_log(log_path)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, 'run.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger(__name__)

    # ── file lists ────────────────────────────────────────────────────────────
    sz_files  = sorted([os.path.join(sz_dir,  f)
                        for f in os.listdir(sz_dir)  if f.endswith('.edf')])
    iic_files = sorted([os.path.join(iic_dir, f)
                        for f in os.listdir(iic_dir) if f.endswith('.edf')])

    # caveman: complete_patients = only finish patients we already touched
    if args.complete_patients:
        # scan results/ for any cached <montage>_<EMU####_seizure_N>.edf.json
        pat_re = re.compile(r'(EMU\d+)_seizure_\d+\.edf\.json$')
        known: set[str] = set()
        try:
            for fn in os.listdir(results_dir):
                m = pat_re.search(fn)
                if m:
                    known.add(m.group(1))
        except FileNotFoundError:
            pass
        log.info(f"--complete_patients: {len(known)} patients have prior results")
        sz_files = [p for p in sz_files
                    if os.path.basename(p).split('_seizure_')[0] in known]
        iic_files = []   # focus only on seizure side for completion
        log.info(f"--complete_patients: {len(sz_files)} seizure clips queued "
                 f"(iic side disabled)")

    if args.max_files:
        sz_files  = sz_files[:args.max_files]
        iic_files = iic_files[:args.max_files]

    all_files = [(p, True) for p in sz_files] + [(p, False) for p in iic_files]
    log.info(f"Seizure: {len(sz_files)}  |  Interictal: {len(iic_files)}")
    alog(log_path, f"## Files\n- Seizure: {len(sz_files)}\n- Interictal: {len(iic_files)}\n")

    # ── montage selection ─────────────────────────────────────────────────────
    montages = {k: v for k, v in MONTAGE_MONO.items()
                if args.montage is None or k == args.montage}
    log.info(f"Running {len(montages)} montage(s): {list(montages.keys())}")
    alog(log_path, f"## Montages ({len(montages)})\n```json\n"
         + json.dumps({k: v for k,v in montages.items()}, indent=2) + "\n```\n")

    # ensure per-montage output subdirectories exist up front (avoids races
    # when many workers write their first CSV concurrently)
    for m in montages:
        os.makedirs(os.path.join(results_dir, m), exist_ok=True)

    # ── parallelism setup ─────────────────────────────────────────────────────
    n_workers = max(1, int(args.workers))
    if args.torch_threads and args.torch_threads > 0:
        per_worker_threads = args.torch_threads
    else:
        per_worker_threads = max(1, (os.cpu_count() or 4) // n_workers)

    log.info(f"Workers: {n_workers}  |  torch threads/worker: {per_worker_threads}")
    alog(log_path,
         f"## Parallelism\n- workers: {n_workers}\n"
         f"- torch_threads/worker: {per_worker_threads}\n")

    # per-montage accumulators
    accum = {m: {'win_true':[], 'win_pred':[], 'file_true':[], 'file_pred':[],
                 'rows':[], 'errors':[]}
             for m in montages}

    def _merge(file_result: dict) -> None:
        fname = file_result['file']
        if 'preprocess_error' in file_result:
            log.warning(f"SKIP {fname}: {file_result['preprocess_error']}")
        for m, payload in file_result['montages'].items():
            if 'error' in payload:
                log.warning(f"  SKIP {fname}/{m}: {payload['error']}")
                accum[m]['errors'].append(
                    {'file': fname, 'error': payload['error']})
                continue
            accum[m]['win_true'].extend(payload['win_gt'])
            accum[m]['win_pred'].extend(payload['win_pred'])
            accum[m]['file_true'].append(payload['file_true'])
            accum[m]['file_pred'].append(payload['file_pred'])
            accum[m]['rows'].append(payload['row'])

    # build task list
    tasks = [{
        'edf_path':   p,
        'is_seizure': is_sz,
        'montages':   montages,
        'results_dir': results_dir,
    } for p, is_sz in all_files]

    # ── dispatch ──────────────────────────────────────────────────────────────
    if n_workers == 1:
        # serial: load models in-process into the same globals the worker uses
        log.info("Loading backbone (IIIC.pth)…")
        _init_worker(args.device, args.batch_size, per_worker_threads)
        log.info("Loaded. Beginning serial run.")
        for t in tqdm(tasks, desc='files', unit='file'):
            _merge(process_one_file(t))
    else:
        log.info(f"Launching ProcessPoolExecutor with {n_workers} workers "
                 f"(each loads its own backbone)…")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(args.device, args.batch_size, per_worker_threads),
        ) as pool:
            futures = [pool.submit(process_one_file, t) for t in tasks]
            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc='files', unit='file'):
                try:
                    _merge(fut.result())
                except Exception as e:
                    log.exception(f"worker failed: {e}")

    # ── per-montage results ───────────────────────────────────────────────────
    if args.skip_summary:
        log.info("--skip_summary: skipping per-montage rollup CSVs and summary.")
        log.info(f"\n{'='*60}\nDone (no summary written). "
                 f"Windowed CSVs + JSON caches are in {results_dir}.")
        return

    summary_rows = []
    for m in montages:
        a      = accum[m]
        wm     = metrics(a['win_true'],  a['win_pred'])
        fm     = metrics(a['file_true'], a['file_pred'])
        nerr   = len(a['errors'])

        log.info(f"\n[{m}]")
        log.info(f"  windowed  F1={wm['f1']:.3f}  sens={wm['sensitivity']:.3f}  prec={wm['precision']:.3f}  (n={wm['n_total']})")
        log.info(f"  file-level F1={fm['f1']:.3f}  sens={fm['sensitivity']:.3f}  prec={fm['precision']:.3f}  (n={fm['n_total']})")

        pd.DataFrame(a['rows']).to_csv(
            os.path.join(results_dir, f'{m}_per_file.csv'), index=False)
        if a['errors']:
            pd.DataFrame(a['errors']).to_csv(
                os.path.join(results_dir, f'{m}_errors.csv'), index=False)

        alog(log_path,
             f"\n---\n## Montage: `{m}`\n"
             f"**Channels:** {montages[m]}\n\n"
             f"| Level | F1 | Sensitivity | Precision | N_pos | N_total |\n"
             f"|---|---|---|---|---|---|\n"
             f"| Windowed (1s) | {wm['f1']:.3f} | {wm['sensitivity']:.3f} | {wm['precision']:.3f} | {wm['n_pos']} | {wm['n_total']} |\n"
             f"| File-level    | {fm['f1']:.3f} | {fm['sensitivity']:.3f} | {fm['precision']:.3f} | {fm['n_pos']} | {fm['n_total']} |\n\n"
             f"Errors/skipped: {nerr}\n")

        summary_rows.append({
            'montage':              m,
            'windowed_f1':          wm['f1'],
            'windowed_sensitivity': wm['sensitivity'],
            'windowed_precision':   wm['precision'],
            'windowed_n_pos':       wm['n_pos'],
            'windowed_n_total':     wm['n_total'],
            'file_f1':              fm['f1'],
            'file_sensitivity':     fm['sensitivity'],
            'file_precision':       fm['precision'],
            'file_n_pos':           fm['n_pos'],
            'file_n_total':         fm['n_total'],
            'n_errors':             nerr,
        })

    # ── summary ───────────────────────────────────────────────────────────────
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics.csv'), index=False)
    log.info(f"\n{'='*60}\nSummary:\n{df_sum.to_string(index=False)}")

    cols = list(df_sum.columns)
    hdr  = '| ' + ' | '.join(cols) + ' |'
    sep  = '| ' + ' | '.join(['---']*len(cols)) + ' |'
    rows = []
    for _, r in df_sum.iterrows():
        cells = [f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c]) for c in cols]
        rows.append('| ' + ' | '.join(cells) + ' |')

    alog(log_path, f"\n---\n## Summary\n\n" + '\n'.join([hdr, sep] + rows)
         + f"\n\n**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
