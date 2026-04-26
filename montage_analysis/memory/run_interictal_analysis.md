# run_interictal_analysis.py — memory

## why
Sister script to `run_seizure_analysis.py`. Processes ONLY interictal clips
that match seizure clips already cached in `montage_analysis/results/`.

## matching rule
Seizure JSON: `<montage>_EMU0878_seizure_3.edf.json`
→ interictal EDF: `data/interictal/EMU0878_iic_3.edf`
(swap `_seizure_` ↔ `_iic_`). Union across all montage subdirs = full target set.

## reuses (imported from run_seizure_analysis)
- `MONTAGE_MONO`, `MONO_CHANNELS`
- `load_and_preprocess_edf`, `apply_montage_and_infer`
- `per_second_gt`, `save_continuous_csv`, `metrics`
- `_init_worker`, `_WORKER` (shared model-loading globals)

So: any preprocessing/model change in the seizure script auto-applies here.

## outputs (separate, no clobber of seizure run)
- `results_interictal/<montage>/<basename>.csv` — per-window (ts, sz_prob, pred, label)
- `results_interictal/<montage>_<file>.json`    — per-file cache (resumable)
- `results_interictal/<montage>_per_file.csv`   — per-file rows
- `summary_metrics_interictal.csv`              — aggregate FP rates
- `ANALYSIS_LOG_INTERICTAL.md`, `run_interictal.log`

## metrics
Interictal = all-negative ground truth. F1/sens/prec are degenerate, so we
report **false-positive rate** instead:
- file-level FP rate = fraction of clips the model flagged as seizure
- window-level FP rate = fraction of 1s windows flagged

## resumable
Same `<montage>_<file>.json` cache pattern. Re-running skips done work.
Fast path: if every montage cache exists for a file, skip EDF load entirely.

## usage
```bash
conda run -n morgoth python montage_analysis/run_interictal_analysis.py
conda run -n morgoth python montage_analysis/run_interictal_analysis.py --montage ceribell
conda run -n morgoth python montage_analysis/run_interictal_analysis.py --workers 4 --torch_threads 3
conda run -n morgoth python montage_analysis/run_interictal_analysis.py --max_files 10
```

## flags
- `--montage X`: run one montage (else all 18)
- `--workers N`: parallel processes (each loads its own backbone)
- `--torch_threads K`: BLAS threads/worker (0 = auto cores//workers)
- `--device cpu|mps|cuda` (default cpu; MPS not stable for some ops)
- `--sz_results_dir`: where to look for seizure JSONs (default `results/`)
- `--out_dir`: where to write (default `results_interictal/`)
- `--max_files N`: smoke test
