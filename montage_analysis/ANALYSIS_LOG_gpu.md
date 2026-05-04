# Morgoth Montage Analysis (GPU) — Run Log

**Started:** 2026-04-27 16:21:02

**Dataset:** /mnt/sauce/littlab/users/aguilac/morgoth/data/seizure  |  /mnt/sauce/littlab/users/aguilac/morgoth/data/interictal

**Models:** /mnt/sauce/littlab/users/aguilac/morgoth/checkpoints/IIIC.pth  |  /mnt/sauce/littlab/users/aguilac/morgoth/checkpoints/SEIZURE_EEGlevel.pth

---

## Files
- Seizure: 3
- Interictal: 3
- Patient filter: EMU0878


## Timing
- Total: 37.4s
- EMU0878_seizure_0.edf: 5.7s
- EMU0878_seizure_1.edf: 7.4s
- EMU0878_seizure_2.edf: 4.5s
- EMU0878_iic_0.edf: 6.4s
- EMU0878_iic_1.edf: 6.4s
- EMU0878_iic_2.edf: 6.5s


---
## Montage: `full`
**Channels:** ['C3', 'C4', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.446 | 0.373 | 0.553 | 308 | 3280 |
| File-level    | 0.800 | 0.667 | 1.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_left_front`
**Channels:** ['F7', 'T3']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.000 | 0.000 | 0.000 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_left_back`
**Channels:** ['T3', 'T5']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.044 | 0.023 | 0.538 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_right_front`
**Channels:** ['F8', 'T4']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.000 | 0.000 | 0.000 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_right_back`
**Channels:** ['T4', 'T6']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.000 | 0.000 | 0.000 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_bilateral_back2`
**Channels:** ['T3', 'T4', 'T5', 'T6']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.204 | 0.117 | 0.800 | 308 | 3280 |
| File-level    | 0.500 | 0.333 | 1.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_bilateral_front2`
**Channels:** ['F7', 'F8', 'T3', 'T4']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.050 | 0.026 | 0.667 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_vert_left`
**Channels:** ['C3', 'T3']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.030 | 0.019 | 0.070 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_vert_right`
**Channels:** ['C4', 'T4']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.019 | 0.010 | 0.500 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_diag_left_front`
**Channels:** ['F3', 'T3']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.264 | 0.205 | 0.371 | 308 | 3280 |
| File-level    | 0.500 | 0.333 | 1.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_diag_left_back`
**Channels:** ['P3', 'T3']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.000 | 0.000 | 0.000 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_diag_right_front`
**Channels:** ['F4', 'T4']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.140 | 0.104 | 0.216 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_diag_right_back`
**Channels:** ['P4', 'T4']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.108 | 0.058 | 0.750 | 308 | 3280 |
| File-level    | 0.500 | 0.333 | 1.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_diag_bilateral_front`
**Channels:** ['F3', 'F4', 'T3', 'T4']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.333 | 0.403 | 0.284 | 308 | 3280 |
| File-level    | 1.000 | 1.000 | 1.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_diag_bilateral_back`
**Channels:** ['P3', 'P4', 'T3', 'T4']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.265 | 0.201 | 0.388 | 308 | 3280 |
| File-level    | 1.000 | 1.000 | 1.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `uneeg_vert_bilateral`
**Channels:** ['C3', 'C4', 'T3', 'T4']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.042 | 0.023 | 0.304 | 308 | 3280 |
| File-level    | 0.500 | 0.333 | 1.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `epiminder_2`
**Channels:** ['C3', 'C4', 'P3', 'P4']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.063 | 0.032 | 1.000 | 308 | 3280 |
| File-level    | 0.000 | 0.000 | 0.000 | 3 | 6 |

Errors/skipped: 0


---
## Montage: `ceribell`
**Channels:** ['F7', 'F8', 'FP1', 'FP2', 'O1', 'O2', 'T3', 'T4', 'T5', 'T6']

| Level | F1 | Sensitivity | Precision | N_pos | N_total |
|---|---|---|---|---|---|
| Windowed (1s) | 0.371 | 0.519 | 0.288 | 308 | 3280 |
| File-level    | 0.667 | 0.667 | 0.667 | 3 | 6 |

Errors/skipped: 0


---
## Summary

| montage | windowed_f1 | windowed_sensitivity | windowed_precision | windowed_n_pos | windowed_n_total | file_f1 | file_sensitivity | file_precision | file_n_pos | file_n_total | n_errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 0.446 | 0.373 | 0.553 | 308 | 3280 | 0.800 | 0.667 | 1.000 | 3 | 6 | 0 |
| uneeg_left_front | 0.000 | 0.000 | 0.000 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| uneeg_left_back | 0.044 | 0.023 | 0.538 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| uneeg_right_front | 0.000 | 0.000 | 0.000 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| uneeg_right_back | 0.000 | 0.000 | 0.000 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| uneeg_bilateral_back2 | 0.204 | 0.117 | 0.800 | 308 | 3280 | 0.500 | 0.333 | 1.000 | 3 | 6 | 0 |
| uneeg_bilateral_front2 | 0.050 | 0.026 | 0.667 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| uneeg_vert_left | 0.030 | 0.019 | 0.070 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| uneeg_vert_right | 0.019 | 0.010 | 0.500 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| uneeg_diag_left_front | 0.264 | 0.205 | 0.371 | 308 | 3280 | 0.500 | 0.333 | 1.000 | 3 | 6 | 0 |
| uneeg_diag_left_back | 0.000 | 0.000 | 0.000 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| uneeg_diag_right_front | 0.140 | 0.104 | 0.216 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| uneeg_diag_right_back | 0.108 | 0.058 | 0.750 | 308 | 3280 | 0.500 | 0.333 | 1.000 | 3 | 6 | 0 |
| uneeg_diag_bilateral_front | 0.333 | 0.403 | 0.284 | 308 | 3280 | 1.000 | 1.000 | 1.000 | 3 | 6 | 0 |
| uneeg_diag_bilateral_back | 0.265 | 0.201 | 0.388 | 308 | 3280 | 1.000 | 1.000 | 1.000 | 3 | 6 | 0 |
| uneeg_vert_bilateral | 0.042 | 0.023 | 0.304 | 308 | 3280 | 0.500 | 0.333 | 1.000 | 3 | 6 | 0 |
| epiminder_2 | 0.063 | 0.032 | 1.000 | 308 | 3280 | 0.000 | 0.000 | 0.000 | 3 | 6 | 0 |
| ceribell | 0.371 | 0.519 | 0.288 | 308 | 3280 | 0.667 | 0.667 | 0.667 | 3 | 6 | 0 |

**Completed:** 2026-04-27 16:21:40

