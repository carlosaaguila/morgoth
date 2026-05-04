"""
Morgoth Seizure Detection — Montage Analysis (Linux/GPU)
=========================================================
GPU-optimised drop-in replacement for run_seizure_analysis.py.

Key changes vs the Mac/CPU version:
  • Default device: cuda
  • All montages for a file are batched into a SINGLE backbone forward pass
    (17× fewer kernel launches, much better GPU utilisation).
  • torch.autocast('cuda') for fp16 mixed-precision (≈2× throughput on A40).
  • EEG-level head runs on the same GPU device (not pinned to CPU).
  • --patient flag to filter to a single EMU patient.
  • --batch_size defaults to 256 (was 64).
  • Timing logged per file and in total.
  • Output written to results_gpu/ (separate from CPU results_/).

Usage:
  conda run -n morgoth python montage_analysis/run_seizure_analysis_linux.py \\
        --patient EMU0878 --device cuda
  conda run -n morgoth python montage_analysis/run_seizure_analysis_linux.py \\
        --device cuda --batch_size 512
"""

import os
import sys
import re
import time
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

import backbone   # registers timm custom models
import utils as morgoth_utils
from EEG_level_head import CNNTransformerClassifier, load_model_parameters

OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
EMU_SZ_DIR  = os.path.join(MORGOTH_DIR, 'data', 'seizure')
EMU_IIC_DIR = os.path.join(MORGOTH_DIR, 'data', 'interictal')
IIIC_CKPT   = os.path.join(MORGOTH_DIR, 'checkpoints', 'IIIC.pth')
SZ_LVL_CKPT = os.path.join(MORGOTH_DIR, 'checkpoints', 'SEIZURE_EEGlevel.pth')

# ── EEG constants ─────────────────────────────────────────────────────────────
FS_TARGET = 200
WIN_SEC   = 10
STEP_SEC  = 1
WIN_SAMP  = WIN_SEC  * FS_TARGET   # 2000
STEP_SAMP = STEP_SEC * FS_TARGET   # 200

MONO_CHANNELS = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                 'T4', 'T6', 'O2']
MONO_IDX = {ch: i for i, ch in enumerate(MONO_CHANNELS)}

# ── montage definitions ───────────────────────────────────────────────────────
def _mono_from_pairs(pairs):
    needed = set()
    for p in pairs:
        a, b = p.upper().split('-')
        needed.add(a); needed.add(b)
    return sorted(needed)

MONTAGE_MONO = {
    'full':
        _mono_from_pairs(['FP1-F7','F7-T3','T3-T5','T5-O1',
                          'FP2-F8','F8-T4','T4-T6','T6-O2',
                          'FP1-F3','F3-C3','C3-P3','P3-O1',
                          'FP2-F4','F4-C4','C4-P4','P4-O2']),
    'uneeg_left_front':           _mono_from_pairs(['F7-T3']),
    'uneeg_left_back':            _mono_from_pairs(['T3-T5']),
    'uneeg_right_front':          _mono_from_pairs(['F8-T4']),
    'uneeg_right_back':           _mono_from_pairs(['T4-T6']),
    'uneeg_bilateral_back2':      _mono_from_pairs(['T3-T5','T4-T6']),
    'uneeg_bilateral_front2':     _mono_from_pairs(['F7-T3','F8-T4']),
    'uneeg_vert_left':            _mono_from_pairs(['C3-T3']),
    'uneeg_vert_right':           _mono_from_pairs(['C4-T4']),
    'uneeg_diag_left_front':      _mono_from_pairs(['F3-T3']),
    'uneeg_diag_left_back':       _mono_from_pairs(['P3-T3']),
    'uneeg_diag_right_front':     _mono_from_pairs(['F4-T4']),
    'uneeg_diag_right_back':      _mono_from_pairs(['P4-T4']),
    'uneeg_diag_bilateral_front': _mono_from_pairs(['F3-T3','F4-T4']),
    'uneeg_diag_bilateral_back':  _mono_from_pairs(['P3-T3','P4-T4']),
    'uneeg_vert_bilateral':       _mono_from_pairs(['C3-T3','C4-T4']),
    'epiminder_2':                _mono_from_pairs(['C3-P3','C4-P4']),
    'ceribell':                   _mono_from_pairs(['FP1-F7','F7-T3','T3-T5','T5-O1',
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
    return raw, label_df, fs

def clean_channel_names(raw):
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
    try:
        raw, label_df, orig_fs = load_edf_file(edf_path)
    except Exception as e:
        return None, None, None, None, f"load error: {e}"

    raw = clean_channel_names(raw)
    available = [ch for ch in MONO_CHANNELS if ch in raw.ch_names]
    if not available:
        return None, None, None, None, "no standard channels"

    raw_sel = raw.copy().pick(available)
    sig_raw = raw_sel.get_data(units='uV').astype(np.float32)
    sig_rs  = resample_to_target(sig_raw, orig_fs)
    sig_rs  = bandpass_notch(sig_rs)

    n_samp   = sig_rs.shape[1]
    full_sig = np.zeros((19, n_samp), dtype=np.float32)
    for i, ch in enumerate(available):
        full_sig[MONO_IDX[ch]] = sig_rs[i]

    return full_sig, label_df, float(orig_fs), available, None

# ── ground truth helper ───────────────────────────────────────────────────────
def per_second_gt(label_df, orig_fs, n_windows):
    labels   = label_df['labels'].values
    fs_int   = int(round(orig_fs))
    win_samp = int(round(WIN_SEC * orig_fs))
    gt       = np.zeros(n_windows, dtype=int)
    for i in range(n_windows):
        s = i * fs_int
        e = min(s + win_samp, len(labels))
        if e > s and labels[s:e].max() > 0:
            gt[i] = 1
    return gt

# ── continuous per-window CSV ─────────────────────────────────────────────────
def window_end_timestamps(n_windows):
    return WIN_SEC - (1.0 / FS_TARGET) + np.arange(n_windows) * STEP_SEC

def save_continuous_csv(out_path, sz_probs, windowed_preds, gt):
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

def load_eeg_level(device):
    m = CNNTransformerClassifier(input_dim=6, output_dim=1, pe_max_length=15000)
    m = load_model_parameters(m, SZ_LVL_CKPT, device=str(device))
    return m.to(device).eval()

# ── log helpers ───────────────────────────────────────────────────────────────
def init_log(p):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    Path(p).write_text(
        f"# Morgoth Montage Analysis (GPU) — Run Log\n\n"
        f"**Started:** {ts}\n\n"
        f"**Dataset:** {EMU_SZ_DIR}  |  {EMU_IIC_DIR}\n\n"
        f"**Models:** {IIIC_CKPT}  |  {SZ_LVL_CKPT}\n\n"
        "---\n\n"
    )

def alog(p, txt):
    with open(p, 'a') as f:
        f.write(txt + '\n')

# ── GPU-optimised: all montages in ONE forward pass ───────────────────────────
def apply_all_montages_batched(full_sig, montages_dict, available,
                                bb, eeg_level_model, device,
                                input_chans, batch_size):
    """
    Apply all montages for a single file in one batched GPU forward pass.

    Instead of M separate forward passes (one per montage), stack all M×N_win
    windows together, run them in a single stream of batched kernel launches,
    then split results back by montage. Gives 10–30× better GPU utilisation
    on an A40 compared to sequential per-montage inference.

    Returns: dict[montage_name -> (result_dict | None, error_str | None)]
    """
    n_samp = full_sig.shape[1]
    if n_samp < WIN_SAMP:
        return {m: (None, "file too short") for m in montages_dict}

    n_windows = (n_samp - WIN_SAMP) // STEP_SAMP + 1

    # Build per-montage masked windows (CPU-side, cheap)
    all_windows_parts = []   # list of (n_win, 19, 2000) arrays
    montage_slices    = {}   # m -> (start, end) in stacked tensor
    results           = {}

    offset = 0
    for m, keep in montages_dict.items():
        active = [ch for ch in keep if ch in available]
        if not active:
            results[m] = (None, "no active channels for montage")
            continue

        masked = np.zeros_like(full_sig)
        for ch in active:
            masked[MONO_IDX[ch]] = full_sig[MONO_IDX[ch]]

        active_idx = [MONO_IDX[ch] for ch in active]
        avg = masked[active_idx].mean(axis=0, keepdims=True)
        montaged = np.zeros_like(masked)
        for idx in active_idx:
            montaged[idx] = masked[idx] - avg[0]

        montaged = clip_and_scale(montaged)

        # sliding window view: (19, n_win, 2000) → (n_win, 19, 2000)
        wins = np.lib.stride_tricks.sliding_window_view(
            montaged, WIN_SAMP, axis=1)[:, ::STEP_SAMP, :]
        wins = wins.transpose(1, 0, 2)[:n_windows]  # (n_win, 19, 2000)

        all_windows_parts.append(wins)
        montage_slices[m] = (offset, offset + n_windows)
        offset += n_windows

    if not all_windows_parts:
        return results

    # Stack all montage windows into one big tensor
    all_wins = np.concatenate(all_windows_parts, axis=0)   # (M*n_win, 19, 2000)
    total    = all_wins.shape[0]
    win_t    = torch.from_numpy(all_wins)

    # Single batched backbone forward pass with CUDA autocast
    use_autocast = str(device).startswith('cuda')
    bb.eval()
    all_logits = []
    with torch.no_grad():
        for s in range(0, total, batch_size):
            batch = win_t[s:s + batch_size].to(device)
            batch = batch / 100.0
            batch = rearrange(batch, 'B N (A T) -> B N A T', T=200)
            with torch.autocast(device_type='cuda' if use_autocast else 'cpu',
                                 enabled=use_autocast):
                out = bb(batch, input_chans=input_chans)
            all_logits.append(out.cpu().to(torch.float32))

    all_logits = torch.cat(all_logits, dim=0)          # (M*n_win, 6)
    all_probs  = F.softmax(all_logits, dim=-1).numpy() # (M*n_win, 6)

    # EEG-level head per montage (small — runs fast on GPU)
    eeg_level_model.eval()
    for m, (start, end) in montage_slices.items():
        probs          = all_probs[start:end]           # (n_win, 6)
        sz_probs       = probs[:, 1]
        windowed_preds = (sz_probs > 0.5).astype(int)
        n_win          = end - start

        eeg_in  = torch.tensor(probs, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([n_win], dtype=torch.long).to(device)
        with torch.no_grad():
            eeg_out  = eeg_level_model(eeg_in, lengths=lengths)
            eeg_prob = torch.sigmoid(eeg_out).item()

        results[m] = ({
            'sz_probs':       sz_probs,
            'windowed_preds': windowed_preds,
            'eeg_prob':       eeg_prob,
            'file_pred':      int(eeg_prob > 0.5),
            'n_windows':      n_win,
        }, None)

    return results

# ── single-montage fallback (for compatibility; used by interictal script) ────
def apply_montage_and_infer(full_sig, keep_channels, available,
                             bb, eeg_level_model, device,
                             input_chans, batch_size):
    result = apply_all_montages_batched(
        full_sig, {'_single': list(keep_channels)}, available,
        bb, eeg_level_model, device, input_chans, batch_size,
    )
    return result['_single']

# ── worker-process state ──────────────────────────────────────────────────────
_WORKER: dict[str, Any] = {}

def _init_worker(device_str: str, batch_size: int, torch_threads: int) -> None:
    os.environ.setdefault('OMP_NUM_THREADS',      str(torch_threads))
    os.environ.setdefault('MKL_NUM_THREADS',      str(torch_threads))
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

# ── per-file worker ───────────────────────────────────────────────────────────
def process_one_file(task: dict) -> dict:
    edf_path    = task['edf_path']
    is_seizure  = task['is_seizure']
    montages    = task['montages']
    results_dir = task['results_dir']

    device      = _WORKER['device']
    batch_size  = _WORKER['batch_size']
    bb          = _WORKER['bb']
    eeg         = _WORKER['eeg']
    input_chans = _WORKER['input_chans']

    fname = os.path.basename(edf_path)
    out: dict[str, dict] = {}

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
                                  'n_windows': c['n_windows'], 'win_f1': c.get('win_f1', float('nan'))},
                    'win_gt':    c['win_gt'],   'win_pred':  c['win_pred'],
                    'file_true': c['file_true'],'file_pred': c['file_pred'],
                    'from_cache': True,
                }
            except Exception as e:
                out[m] = {'error': f'cache read failed: {e}'}
        return {'file': fname, 'montages': out}

    full_sig, label_df, orig_fs, available, err = load_and_preprocess_edf(edf_path)
    if err:
        return {'file': fname, 'montages': {m: {'error': err} for m in montages},
                'preprocess_error': err}

    # Which montages still need computation?
    pending, cached_m = {}, {}
    for m, keep in montages.items():
        cp = os.path.join(results_dir, f'{m}_{fname}.json')
        if os.path.exists(cp):
            try:
                with open(cp) as fh:
                    c = json.load(fh)
                cached_m[m] = c
                continue
            except Exception:
                pass
        pending[m] = keep

    # Batch-infer all pending montages in ONE GPU forward pass
    if pending:
        batch_results = apply_all_montages_batched(
            full_sig, pending, available, bb, eeg, device, input_chans, batch_size,
        )
    else:
        batch_results = {}

    n_win_base = None
    gt_cache: dict[int, np.ndarray] = {}   # n_windows -> gt array (same for all montages)

    for m, keep in montages.items():
        # Load from cache if we skipped it above
        if m in cached_m:
            c = cached_m[m]
            out[m] = {
                'row':       {'file': fname, 'file_true': c['file_true'],
                              'file_pred': c['file_pred'], 'eeg_prob': c['eeg_prob'],
                              'n_windows': c['n_windows'], 'win_f1': c.get('win_f1', float('nan'))},
                'win_gt':    c['win_gt'],   'win_pred':  c['win_pred'],
                'file_true': c['file_true'],'file_pred': c['file_pred'],
                'from_cache': True,
            }
            continue

        res, merr = batch_results.get(m, (None, "missing from batch"))
        if merr:
            out[m] = {'error': merr}
            continue

        n_win = res['n_windows']
        if n_win not in gt_cache:
            gt_cache[n_win] = per_second_gt(label_df, orig_fs, n_win)
        gt_sec = gt_cache[n_win]

        wp       = res['windowed_preds'].tolist()
        gt_l     = gt_sec.tolist()
        win_f1   = (f1_score(gt_l, wp, zero_division=0)
                    if len(np.unique(gt_l)) > 1 else float('nan'))
        file_true = int(is_seizure)

        cont_csv = os.path.join(results_dir, m, fname.replace('.edf', '.csv'))
        save_continuous_csv(cont_csv, res['sz_probs'], res['windowed_preds'], gt_sec)

        cache_payload = {
            'win_gt': gt_l, 'win_pred': wp,
            'file_true': file_true, 'file_pred': res['file_pred'],
            'eeg_prob':  res['eeg_prob'], 'n_windows': n_win, 'win_f1': win_f1,
        }
        with open(os.path.join(results_dir, f'{m}_{fname}.json'), 'w') as fh:
            json.dump(cache_payload, fh)

        out[m] = {
            'row':       {'file': fname, 'file_true': file_true,
                          'file_pred': res['file_pred'], 'eeg_prob': res['eeg_prob'],
                          'n_windows': n_win, 'win_f1': win_f1},
            'win_gt':    gt_l,  'win_pred':  wp,
            'file_true': file_true, 'file_pred': res['file_pred'],
            'from_cache': False,
        }

    return {'file': fname, 'montages': out}

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_files',   type=int, default=None)
    ap.add_argument('--patient',     type=str, default=None,
                    help='Filter to a single patient prefix, e.g. EMU0878.')
    ap.add_argument('--montage',     type=str, default=None)
    ap.add_argument('--batch_size',  type=int, default=256)
    ap.add_argument('--device',      type=str, default='cuda')
    ap.add_argument('--sz_dir',      type=str, default=EMU_SZ_DIR)
    ap.add_argument('--iic_dir',     type=str, default=EMU_IIC_DIR)
    ap.add_argument('--workers',     type=int, default=1)
    ap.add_argument('--torch_threads', type=int, default=0)
    ap.add_argument('--out_dir',     type=str, default=None,
                    help='Override results directory (default: results_gpu/).')
    ap.add_argument('--skip_summary', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device)
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
        args.device = 'cpu'

    results_dir = args.out_dir or os.path.join(OUTPUT_DIR, 'results_gpu')
    log_path    = os.path.join(OUTPUT_DIR, 'ANALYSIS_LOG_gpu.md')
    os.makedirs(results_dir, exist_ok=True)
    init_log(log_path)

    log_file = os.path.join(OUTPUT_DIR, 'run_gpu.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger(__name__)
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"VRAM: {torch.cuda.get_device_properties(0).total_memory//2**30} GB")

    # ── file lists ────────────────────────────────────────────────────────────
    sz_files  = sorted([os.path.join(args.sz_dir,  f)
                        for f in os.listdir(args.sz_dir)  if f.endswith('.edf')])
    iic_files = sorted([os.path.join(args.iic_dir, f)
                        for f in os.listdir(args.iic_dir) if f.endswith('.edf')])

    if args.patient:
        sz_files  = [p for p in sz_files  if os.path.basename(p).startswith(args.patient)]
        iic_files = [p for p in iic_files if os.path.basename(p).startswith(args.patient)]
        log.info(f"--patient {args.patient}: {len(sz_files)} seizure, "
                 f"{len(iic_files)} interictal files")

    if args.max_files:
        sz_files  = sz_files[:args.max_files]
        iic_files = iic_files[:args.max_files]

    all_files = [(p, True) for p in sz_files] + [(p, False) for p in iic_files]
    log.info(f"Seizure: {len(sz_files)}  |  Interictal: {len(iic_files)}")
    alog(log_path, f"## Files\n- Seizure: {len(sz_files)}\n- Interictal: {len(iic_files)}\n"
                   + (f"- Patient filter: {args.patient}\n" if args.patient else ""))

    montages = {k: v for k, v in MONTAGE_MONO.items()
                if args.montage is None or k == args.montage}
    log.info(f"Running {len(montages)} montage(s): {list(montages.keys())}")

    for m in montages:
        os.makedirs(os.path.join(results_dir, m), exist_ok=True)

    n_workers = max(1, int(args.workers))
    per_worker_threads = (args.torch_threads if args.torch_threads > 0
                          else max(1, (os.cpu_count() or 4) // n_workers))
    log.info(f"Workers: {n_workers}  |  torch_threads/worker: {per_worker_threads}  "
             f"|  batch_size: {args.batch_size}")

    accum = {m: {'win_true': [], 'win_pred': [], 'file_true': [], 'file_pred': [],
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
            accum[m]['win_true'].extend(payload['win_gt'])
            accum[m]['win_pred'].extend(payload['win_pred'])
            accum[m]['file_true'].append(payload['file_true'])
            accum[m]['file_pred'].append(payload['file_pred'])
            accum[m]['rows'].append(payload['row'])

    tasks = [{'edf_path': p, 'is_seizure': is_sz, 'montages': montages,
               'results_dir': results_dir}
             for p, is_sz in all_files]

    total_t0 = time.perf_counter()

    if n_workers == 1:
        log.info("Loading backbone (IIIC.pth)…")
        _init_worker(args.device, args.batch_size, per_worker_threads)
        log.info("Model loaded. Beginning run.")
        for t in tqdm(tasks, desc='files', unit='file'):
            t0 = time.perf_counter()
            r  = process_one_file(t)
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
            fut_map = {pool.submit(process_one_file, t): t for t in tasks}
            for fut in tqdm(as_completed(fut_map), total=len(tasks),
                            desc='files', unit='file'):
                task = fut_map[fut]
                t0   = task.get('_t0', time.perf_counter())  # approx
                try:
                    r = fut.result()
                    _merge(r, time.perf_counter() - t0)
                except Exception as e:
                    log.exception(f"worker failed on {task['edf_path']}: {e}")

    total_elapsed = time.perf_counter() - total_t0
    log.info(f"\nTotal wall time: {total_elapsed:.1f}s  "
             f"({total_elapsed/max(len(all_files),1):.1f}s/file avg)")
    if file_times:
        log.info("Per-file times:")
        for fn, t in file_times:
            log.info(f"  {fn}: {t:.1f}s")

    alog(log_path,
         f"\n## Timing\n- Total: {total_elapsed:.1f}s\n"
         + "\n".join(f"- {fn}: {t:.1f}s" for fn, t in file_times) + "\n")

    if args.skip_summary:
        log.info("--skip_summary: done.")
        return

    summary_rows = []
    for m in montages:
        a    = accum[m]
        wm   = metrics(a['win_true'], a['win_pred'])
        fm   = metrics(a['file_true'], a['file_pred'])
        nerr = len(a['errors'])

        log.info(f"\n[{m}]")
        log.info(f"  windowed   F1={wm['f1']:.3f}  sens={wm['sensitivity']:.3f}  "
                 f"prec={wm['precision']:.3f}  (n={wm['n_total']})")
        log.info(f"  file-level F1={fm['f1']:.3f}  sens={fm['sensitivity']:.3f}  "
                 f"prec={fm['precision']:.3f}  (n={fm['n_total']})")

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
             f"| Windowed (1s) | {wm['f1']:.3f} | {wm['sensitivity']:.3f} | "
             f"{wm['precision']:.3f} | {wm['n_pos']} | {wm['n_total']} |\n"
             f"| File-level    | {fm['f1']:.3f} | {fm['sensitivity']:.3f} | "
             f"{fm['precision']:.3f} | {fm['n_pos']} | {fm['n_total']} |\n\n"
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

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics_gpu.csv'), index=False)
    log.info(f"\n{'='*60}\nSummary:\n{df_sum.to_string(index=False)}")

    cols = list(df_sum.columns)
    hdr  = '| ' + ' | '.join(cols) + ' |'
    sep  = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    rows_md = []
    for _, r in df_sum.iterrows():
        cells = [f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c]) for c in cols]
        rows_md.append('| ' + ' | '.join(cells) + ' |')
    alog(log_path, f"\n---\n## Summary\n\n" + '\n'.join([hdr, sep] + rows_md)
         + f"\n\n**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
