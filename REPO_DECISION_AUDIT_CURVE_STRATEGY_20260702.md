# Repository Decision Audit — Curve Strategy
**Date:** 2026-07-02  
**Purpose:** Decision support before committing to next modelling step  
**Do not implement model changes based on this document alone.**

---

## 1. Executive Summary

The validated attenuation-curve framework (`y = c_hat - n·log(r/r0) + ε`) is working. The physics is correct. The bottleneck is **c_hat initialization instability** in the query-conditioned model. Three concrete paths exist for tomorrow:

- **Option A** (recommended for tomorrow): Fix 2-line init in `CurvePriorCNN2D_Query`, rerun 7-run grid. Low risk, high confidence.
- **Option B**: Use existing Q3/Q2 stability predictions for attenuation-curve plots — already available, no rerun needed.
- **Option C**: XGBoost v8 two-stage physics model is already complete, achieves RMSE(PGV)=1.83 on the full sensor set and RMSE(log)=0.502 — arguably the best physics-informed result to date on the full dataset.

---

## 2. Validated — Do Not Modify

| Script / Module | Status | Reason |
|---|---|---|
| `train_cnn_curveprior_linec_v1.py` | ✅ Validated | Produced P1/P3_corrected_n; n-fitting correct |
| `src/ml/cnn/cnn_curveprior_linec_utils.py` | ✅ Validated | CurvePriorCNN2D, n-fitting, losses |
| `src/ml/cnn/config_cnn_curveprior_linec_v1.py` | ✅ Validated | Correct column names, split fractions |
| `train_cnn_v2.py` | ✅ Validated | Row-wise 2D CNN baseline 0.595/2.348 |
| `src/ml/xgboost/xgb_utils.py` | ✅ Validated | `make_event_level_test_split` |
| `src/ml/mlp/mlp_utils.py` | ✅ Validated | All MLP pipelines |
| `src/utils/geometry_utils.py` | ✅ Validated | Track-specific distance logic |
| `train_xgb_v8.py` + `config_xgb_v8.py` | ✅ Ran to completion | Best physics XGB (1.83 mm/s) |

---

## 3. Experimental — Safe to Modify

| Script / Module | Status |
|---|---|
| `train_cnn_curvequery_linec_v1.py` | Active — needs init fix |
| `src/ml/cnn/cnn_curvequery_linec_utils.py` | Active — `CurvePriorCNN2D_Query.__init__` needs final-layer init |
| `src/ml/cnn/config_cnn_curvequery_linec_v1.py` | Active |
| `train_cnn_twohead_linec_v1.py` | Superseded by curve-prior |
| `slurm/run_curvequery_stability_linec_v1.slurm` | Will be replaced by init-fix SLURM |

---

## 4. Completed Experiment Table

| # | Model | Target | Input | Split | RMSE(log) | RMSE(PGV) | R²(log) | Notes |
|---|-------|--------|-------|-------|-----------|-----------|---------|-------|
| 1 | XGBoost v1/v2 | log1p(PGV_z) | Par v1, all sensors | event 15%/15% | — | ~6.7 | 0.22 | MP14-19 corrupted |
| 2 | XGBoost v3 | log1p(PGV_z) | Par v1, MP1-13 | event 15%/15% | — | 1.84 | 0.44 | First clean run |
| 3 | **XGBoost v4** | log1p(PGV_z) | Par v2, MP1-13 | event 15%/15% | — | **1.79** | — | ← tabular best |
| 4 | XGBoost v5 | log1p(PGV_z) | Par v3, MP1-13 | event 15%/15% | — | 1.97 | — | per-line, degraded |
| 5 | XGBoost v6 | c_i (event intensity) | Par v4, all sensors | event 15%/15% | 0.555 | 1.98 | 0.537 | ✅ complete, n=0.93 |
| 6 | XGBoost v7 | (c_i, n_i) jointly | Par v4, all sensors | event 15%/15% | 0.551 | 1.95 | 0.543 | ✅ complete, per-event n |
| 7 | **XGBoost v8 varA** | log(PGV_z) via physics | Par v4, all sensors | event 15%/15% | **0.502** | **1.83** | 0.391 | ✅ complete, 2-stage |
| 8 | **XGBoost v8 varB** | ε residual | Par v4, all sensors | event 15%/15% | **0.511** | **1.84** | 0.381 | ✅ complete, residual |
| 9 | MLP v2 | log1p(PGV_z) | Par v2 | event 15%/15% | — | 1.84 | — | |
| 10 | MLP v6 | log1p(PGV_z) | Par v3 | event 15%/15% | — | 2.19 | — | per-line, degraded |
| 11 | **CNN v2 (line-C)** | log(PGV_z) row-wise | Wf v3 + Par v2 | event 15%/15% | **0.595** | **2.348** | **0.654** | ← waveform baseline |
| 12 | Two-head C/D | 5-vector log(PGV) | Wf v3 + Par v2 | event 65/15/20 | 0.613 | 2.620 | 0.645 | bugs present |
| 13 | Oracle (line-C) | c_i power-law | Par v2 | event 65/15/20 | 0.334 | 1.811 | 0.895 | physics ceiling |
| 14 | **P1_corrected_n** | Curve-only c_hat | Wf v3 + Par v2 | event 65/15/20 | **0.6146** | 2.483 | 0.643 | ✅ validated |
| 15 | **P3_corrected_n** | Curve+ε+MP4 weight | Wf v3 + Par v2 | event 65/15/20 | **0.5993** | **2.401** | **0.660** | ✅ validated ← curve best |
| 16 | Q1_all (first-pass) | Curve-only query | Wf v3 + Par v2 | event 65/15/20 | 1.485 | 6.570 | −1.08 | ❌ collapsed |
| 17 | Q3_all (first-pass) | Curve+query ε | Wf v3 + Par v2 | event 65/15/20 | 0.619 | 2.366 | 0.638 | chaotic, rescued |
| 18 | Q1_all_lr1e4 | Curve-only query | Wf v3 + Par v2 | event 65/15/20 | 0.662 | 2.342 | — | still unstable |
| 19 | **Q2_all_lr1e4** | Curve+query ε | Wf v3 + Par v2 | event 65/15/20 | **0.613** | 2.416 | 0.646 | ✅ best ckpt saved |
| 20 | **Q3_all_lr1e4** | Curve+query ε+MP4 | Wf v3 + Par v2 | event 65/15/20 | **0.620** | **2.375** | — | ✅ best ckpt saved |
| 21 | Q3_holdout_MP4 (stability) | Curve+query ε | Wf v3 + Par v2 | event 65/15/20 | 0.458 (non-held) | — | — | held-out MP4: 0.971 |
| 22 | **Q3_holdout_MP2 (stability)** | Curve+query ε | Wf v3 + Par v2 | event 65/15/20 | 0.698 (non-held) | — | — | **held-out MP2: 0.491, 0.25 mm/s** |

---

## 5. XGBoost v6 / v7 / v8 Status

### Completion status: ✅ ALL THREE COMPLETED

All were run on **2026-05-09** and have full output directories with models, OOF predictions, and summary metrics.

| Version | Output dir | Target | n used | Sensor set | RMSE(log) test | RMSE(PGV) test |
|---|---|---|---|---|---|---|
| v6 | `xgb_v006_20260509_220122` | c_i per event | n_global = 0.93 | All (1697 events × 9 sensors) | 0.555 | 1.98 |
| v7 | `xgb_v007_20260509_220559` | (c_i, n_i) per event | per-event n_i | Good-fit events only | 0.551 | 1.95 |
| v8 varA | `xgb_v008_20260509_221501` | log(PGV) via c_hat | n_global = 0.93 | All (1443 train-val events) | **0.502** | **1.83** |
| v8 varB | same | ε residual on physics | n_global = 0.93 | same | **0.511** | **1.84** |

### Key limitations vs curveprior/query

1. **n value**: v6/v7/v8 use `n_global = 0.93` fitted on the full sensor cross-section (all lines, all sensors). The curveprior uses track-specific values fitted on line-C only: `n_track1 = 1.0655`, `n_track2 = 1.3246`. The global 0.93 averages over track geometry and is physically less accurate.

2. **Sensor set**: v6-v8 evaluate on all 9+ sensors across all lines. The line-C curveprior/query evaluate on 5 sensors only. **Direct RMSE comparison is not valid.**

3. **Parquet v4 required**: v6-v8 depend on `parquet_v004_20260509_190849` (event-level, one row per train event with fitted c_i/n_i). This is a different format from parquet v2.

### Adapting v6-v8 to line-C curveprior split

**Estimated effort: medium, ~1–2 days.** Steps:
- Filter parquet v4 event-level data to line-C events only
- Replace `n_global = 0.93` with track-specific `n_track1 = 1.0655`, `n_track2 = 1.3246`
- Use the curveprior split (seed=42, 65/15/20) instead of the XGBoost 85/15 split
- Then retrain v6 (or reuse v8 stage-1 model with updated physics)

**Is this worth doing?** The v8 two-stage architecture is conceptually identical to the query model but uses tabular FO features (octave bands, time-domain stats) instead of raw waveforms. Running v8 adapted to line-C would give a clean tabular-vs-waveform comparison. However, the CNN curvequery model is already implemented and will produce better results once the init fix is applied.

**Recommendation:** Not a priority for tomorrow unless CNN experiments fail.

---

## 6. Query Init-Fix Feasibility

### Exact location of the bug

**File:** `src/ml/cnn/cnn_curvequery_linec_utils.py`  
**Class:** `CurvePriorCNN2D_Query`  
**Lines:** ~93–100 (intensity_head construction), ~112–120 (residual_head construction)

```python
# Current (no explicit init):
ih.append(nn.Linear(in_sz, 1))
self.intensity_head = nn.Sequential(*ih)

# Current residual head (no explicit init):
rh.append(nn.Linear(in_sz, 1))
self.residual_head = nn.Sequential(*rh)
```

The `nn.Linear(in_sz, 1)` uses default Kaiming uniform init, which for `in_sz = 32` gives `std ≈ 1/√32 ≈ 0.177`. With 80-dim embeddings flowing through the 80→64→32 intensity head, by the time they reach the final layer the activations can be order-of-magnitude large, producing c_hat >> 5 at initialization.

### Q&A

1. **Where is `CurvePriorCNN2D_Query` defined?**  
   `src/ml/cnn/cnn_curvequery_linec_utils.py`, class definition around line 35.

2. **Where are `intensity_head` and `residual_head` created?**  
   `__init__` method, lines ~93–100 (intensity) and ~112–120 (residual).

3. **Is the final layer of `intensity_head` currently initialized explicitly?**  
   **No.** Default Kaiming uniform.

4. **Is the final layer of `residual_head` currently initialized explicitly?**  
   **No.** Default Kaiming uniform.

5. **Is `c_train_mean` computed anywhere?**  
   **No.** It is not currently computed anywhere in any script.

6. **Can `c_train_mean` be passed cleanly into the model constructor?**  
   **Yes.** Add `c_hat_init: float = 0.0` parameter to `__init__`. One new parameter, no config change needed (or add to config with default 0.0).

7. **Is there any reason NOT to set final weights to zero and bias to `c_train_mean`?**  
   No. This is standard practice for physics-informed output heads. Setting the final weight to zero means `c_hat ≈ c_hat_init` at epoch 0 regardless of encoder output. The gradient from L_c then pulls c_hat toward the correct event intensity from the first step.

8. **Does this touch validated curveprior code?**  
   **No.** Only `cnn_curvequery_linec_utils.py` and `train_cnn_curvequery_linec_v1.py`. The curveprior files (`cnn_curveprior_linec_utils.py`, `config_cnn_curveprior_linec_v1.py`, `train_cnn_curveprior_linec_v1.py`) are untouched.

9. **Will Q1/Q2/Q3 all use the same initialized c_hat head?**  
   Yes. The `c_hat_init` parameter controls all variants uniformly.

10. **For Q1, can residual be safely disabled without touching residual init?**  
    Yes. When `enable_residual_head=False`, `self.residual_head = None` and no residual init is needed. The zero-init is only applied to the existing `nn.Linear` at the head of the residual when it is enabled.

### Precise patch plan (DO NOT IMPLEMENT YET)

**Change 1: `src/ml/cnn/cnn_curvequery_linec_utils.py`**

In `CurvePriorCNN2D_Query.__init__`, add parameter `c_hat_init: float = 0.0` to the signature.

After the line `self.intensity_head = nn.Sequential(*ih)`, add:
```python
# Zero-init final linear layer; bias = c_hat_init keeps c_hat ≈ 0 at start
nn.init.zeros_(self.intensity_head[-1].weight)
nn.init.constant_(self.intensity_head[-1].bias, c_hat_init)
```

After the line `self.residual_head = nn.Sequential(*rh)` (inside the `if enable_residual_head:` block), add:
```python
nn.init.zeros_(self.residual_head[-1].weight)
nn.init.zeros_(self.residual_head[-1].bias)
```

**Change 2: `train_cnn_curvequery_linec_v1.py`**

After fitting attenuation exponents and before model construction, add:
```python
# Compute c_hat_init = mean c_target across training events
log_dist = np.log(dist_arr[train_idx] / cfg.features.r0)
n_vec_np = np.where(track_arr[train_idx] == 1, n_track1, n_track2)
c_target_all = tgt_log[train_idx] + n_vec_np[:, None] * log_dist
c_hat_init = float(c_target_all.mean())
print(f"c_hat_init (train mean c_target): {c_hat_init:.4f}")
```

Pass `c_hat_init=c_hat_init` to `CurvePriorCNN2D_Query(...)`.

**Total code change: ~10 lines across 2 experimental files.**

---

## 7. Event-Batched Architecture Feasibility

### Which existing script is closest to event-batched CurveNet?

`train_cnn_curveprior_linec_v1.py` **already is event-batched.** Each batch element is one event; the model processes one waveform and predicts a 5-sensor profile in one forward pass. This is exactly the event-batched CurveNet architecture.

### Does fixed-output curveprior already output a 5-sensor profile?

**Yes.** `CurvePriorCNN2D` outputs:
- `c_hat`: scalar event intensity  
- `epsilon_hat`: (B, 5) residual vector for [MP4, MP8, MP10, MP1, MP2]

The full prediction profile is `y_pred = c_hat - n_track * log(r/r0) + epsilon_hat` computed over the fixed distances [2.5, 4.0, 8.0, 16.0, 23.0] m (track 1) or [6.5, 8.0, 12.0, 20.0, 27.0] m (track 2).

### Can P3 predictions be used for attenuation-curve plots?

**Yes, but P3 has no saved predictions.** The `train_cnn_curveprior_linec_v1.py` script does NOT save `predictions.parquet` or `model.pth`. The P3_corrected_n output directory contains only:
- `config_snapshot.json` (385 bytes)
- `metrics.json` (1978 bytes)

To generate P3 attenuation-curve plots, we would need to **rerun training** or modify the script to save predictions. Rerunning takes ~5 minutes on the cluster.

### What is missing to make curveprior arbitrary-distance?

The curveprior uses fixed output distances per track. Making it arbitrary-distance requires:
1. Converting the 5-output epsilon_hat to a function of distance (the curvequery approach), OR
2. Smooth curve fitting: given (c_hat, n, distances, epsilon_hat at 5 points), fit a smooth curve and interpolate — this is possible analytically.

### Option ranking: A vs B vs C

| Option | Description | Risk | Time | Result quality |
|---|---|---|---|---|
| **A** | Fix curvequery init (`c_hat_init`) | **LOW** — 2 lines in experimental files | 15 min code + cluster run | **HIGH** — expected to fully fix instability, match P3 |
| **B** | Add `predictions.parquet` save to curveprior, rerun P3 | **VERY LOW** — 5 lines in training script | 15 min code + 5 min cluster | **MEDIUM** — P3 curves exist but fixed-output only |
| **C** | New event-batched query implementation | **HIGH** — substantial new code | 3–4 hours | **MEDIUM** — uncertain result |

**Recommendation: A first, then B as guaranteed fallback.**

---

## 8. Prediction and Plot Availability

### Available predictions by model

| Model | predictions.parquet | Columns | Notes |
|---|---|---|---|
| **P3_corrected_n** | ❌ None | — | model.pth also missing; must rerun |
| **P1_corrected_n** | ❌ None | — | same |
| **Q1_all_lr1e4** | ✅ `cnn_curvequery_stability_linec_v001_vQ1_all_20260701_233311/` | all 11 cols | 340 test events × 5 sensors |
| **Q2_all_lr1e4** | ✅ `cnn_curvequery_stability_linec_v001_vQ2_all_20260701_233615/` | all 11 cols | 340 × 5 |
| **Q3_all_lr1e4** | ✅ `cnn_curvequery_stability_linec_v001_vQ3_all_20260701_234441/` | all 11 cols | 340 × 5 |
| **Q3_holdout_MP4** | ✅ `cnn_curvequery_stability_linec_v001_vQ3_holdout_MP4_20260701_235551/` | all 11 cols + held_predictions | 300 + 340 MP4-only |
| **Q3_holdout_MP2** | ✅ `cnn_curvequery_stability_linec_v001_vQ3_holdout_MP2_20260702_000009/` | all 11 cols + held_predictions | 272 + 340 MP2-only |
| **XGBoost v8** | ✅ `xgb_v008_20260509_221501/oof_predictions.parquet` | event_id, distance, target, pred_A, pred_B | OOF only, full sensor set |

### Predictions.parquet column inventory (curvequery)

```
event_id     — string, e.g. "20240904_175139.mat"
sensor       — "MP4" | "MP8" | "MP10" | "MP1" | "MP2"
track        — int, 1 or 2
distance     — float, active-track distance in metres (2.5–27.0)
pred_log     — float, log(PGV_z) predicted (clamped [-10, 5])
target_log   — float, log(PGV_z) measured
pred_pgv     — float, exp(pred_log) in mm/s
target_pgv   — float, exp(target_log) in mm/s
epsilon      — float, residual head output (0 for Q1)
c_hat        — float, predicted event intensity
c_target     — float, inverted intensity from measured data
```

**n used for reconstruction** is NOT in the parquet file — it is in `config_snapshot.json` per run.

### Are predictions saved for all splits or test only?

**Test only** for the current curvequery implementation. The training loop saves predictions only for the test split (+ held-out sensor for holdout runs). Validation split predictions are not saved.

### Can attenuation-curve plots be made immediately?

**Yes, from Q3_all_lr1e4.** The predictions.parquet has 340 events × 5 sensors = 1700 rows. For any event with all 5 sensors in the test set:
- Plot `target_pgv` vs `distance` (measured profile)
- Plot `pred_pgv` vs `distance` (predicted profile)
- Overlay `exp(c_hat - n * log(r/r0))` as the pure physics curve

Selection criteria for interesting events:
```python
# High-PGV events (MP4 > 4 mm/s):
df[df['sensor']=='MP4'].nlargest(10, 'target_pgv')

# MP4 prediction failures (large residual at MP4):
mp4 = df[df['sensor']=='MP4'].copy()
mp4['error'] = abs(mp4['pred_pgv'] - mp4['target_pgv'])
mp4.nlargest(10, 'error')

# Good far-field (MP2 with low error):
mp2 = df[df['sensor']=='MP2'].copy()
mp2['error'] = abs(mp2['pred_pgv'] - mp2['target_pgv'])
mp2.nsmallest(10, 'error')
```

---

## 9. Split Balance Quick Audit

### Split methodology

| Model family | Split method | Split fractions | Seed(s) | Event count |
|---|---|---|---|---|
| XGBoost v1–v8, MLP v1–v6, CNN v1–v2 | `make_event_level_test_split` + `make_event_level_val_split` | 15% test (seed=42), 15% of remaining val (seed=43) | 42, 43 | 1443/1480 train-val; ~260 test |
| Curveprior, Curvequery | `make_event_splits` (single shuffle) | 65/15/20 | 42 | 1103 train / 254 val / 340 test |

**These are NOT the same splits.** The test event sets do not overlap identically. Cross-family RMSE comparisons are approximate.

**No row-level leakage risk** in either family — both operate strictly by event_id, with sensor rows following their event assignment.

**Event IDs are NOT explicitly saved** by any script for the curveprior/query family. They can be reconstructed exactly from:
```python
np.random.seed(42)
idx = np.arange(1697)
np.random.shuffle(idx)
train_idx, val_idx, test_idx = idx[:1103], idx[1103:1357], idx[1357:]
```
Applied to `df['event_id'].unique()` from parquet v002 line-C filtered subset (8485 rows, 1697 events, after cleaning).

### Split distribution summary

| Split | Events | Track 1 | Track 2 | PGV>4 mm/s | PGV>8 mm/s | Mean max_PGV | c_target mean±std |
|---|---|---|---|---|---|---|---|
| **Train** | 1103 | 545 (49.4%) | 558 (50.6%) | 559 (50.7%) | 555 (50.3%) | 6.58 mm/s | 0.316 ± 0.532 |
| **Val** | 254 | 121 (47.6%) | 133 (52.4%) | 128 (50.4%) | 127 (50.0%) | 6.63 mm/s | 0.332 ± 0.528 |
| **Test** | 340 | 180 (52.9%) | 160 (47.1%) | 175 (51.5%) | 171 (50.3%) | 6.55 mm/s | 0.289 ± 0.529 |

**Split balance verdict: Excellent.** Track proportions are within ±5% across splits. High-PGV fractions are nearly identical (~50% in all splits). c_target distributions are consistent (mean ≈ 0.3, std ≈ 0.53). No stratification is needed or recommended.

**Minor observation:** The test set has slightly more Track 1 events (52.9%) vs train (49.4%). Difference is 3.5 percentage points — within normal random variation for a 65/15/20 shuffle.

---

## 10. Final Recommended Strategy for Tomorrow

### Decision table

| Option | Description | Impl. risk | Cluster time | Expected RMSE(log) | P(presentation-ready) | Helps story |
|---|---|---|---|---|---|---|
| **A. Query init-fix** | Zero-init `intensity_head`+`residual_head`, pass `c_hat_init` | **Very low** (10 lines, 2 experimental files) | ~3 hrs (7 runs) | Q1: ~0.61, Q3: ~0.59 | **90%** | ✅ Validates arbitrary-distance curve |
| **B. P3 prediction save** | Add `predictions.parquet` + `model.pth` save to curveprior, rerun | **Very low** (5 lines, 1 validated file) | ~10 min | Same as before: 0.5993 | **99%** | ✅ Best attenuation-curve plots |
| **C. New event-batched** | Implement new smooth-residual model | High (200+ lines) | ~3 hrs | Unknown | **30%** | Uncertain |
| **D. XGBoost v6–v8 on line-C** | Adapt physics XGB to line-C split+sensors | Medium (new parquet build + refit) | ~2 hrs | ~0.55? | **50%** | Partial (different sensor set) |
| **E. Split stratification** | Redesign split | Low but unnecessary | N/A | Marginal gain | N/A | No |

### Recommended plan

**Do in this order:**

#### Step 1 (immediate, 15 min): Capture P3 predictions
Modify `train_cnn_curveprior_linec_v1.py` to save `predictions.parquet` and `model.pth` before the metrics JSON. Add to the curveprior SLURM or run P3 standalone. This is **non-destructive** and gives the best-validated model's predictions for plotting.

Expected outputs: `predictions.parquet` with 340×5 = 1700 rows, full profile per event.

#### Step 2 (15 min code + cluster submit): Query init fix
Apply the 10-line init fix to `cnn_curvequery_linec_utils.py` and `train_cnn_curvequery_linec_v1.py`. Create `slurm/run_curvequery_initfix_linec_v1.slurm`. Run the same 7-run grid as stability (Q1/Q2/Q3 all-sensor + Q3_holdout_MP4 + Q3_holdout_MP2).

Expected outcome based on diagnostic evidence:
- c_hat at epoch 1 will be ≈ c_train_mean ≈ 0.3 (vs 80–164 currently)
- frac_hi at epoch 1 will be 0.000 (vs 1.000 currently)
- Q1_all should reach RMSE(log) ≈ 0.61–0.63 (matching P1_corrected_n)
- Q3_all should reach RMSE(log) ≈ 0.59–0.60 (matching/beating P3_corrected_n)

#### Step 3 (parallel with cluster run): Plotting from existing predictions
Use `Q3_all_lr1e4/predictions.parquet` and `Q3_holdout_MP2/held_predictions.parquet` to generate attenuation-curve plots immediately. These are already available with all required columns.

Key plots to produce:
1. Measured vs predicted profile (5-point attenuation curve) for 6 representative events
2. Measured vs predicted scatter by sensor
3. Residuals vs distance
4. Held-out MP2 predictions vs measured (zero-shot far-field generalization)

---

### If the init-fix rerun produces clean Q1/Q3 results

Decision criteria:
- If Q1_all RMSE(log) ≈ 0.61–0.63 → training stable, proceed to full 12-run grid with holdouts
- If Q3_all RMSE(log) < 0.60 → query model beats P3_corrected_n → main story validated
- If Q3_holdout_MP4 held-out RMSE(log) < 1.0 → arbitrary-distance near-field works
- If Q3_holdout_MP2 held-out RMSE(log) < 0.55 → arbitrary-distance far-field excellent

**Bottom line:** The init fix is the highest-leverage single change in the codebase. The rest is plotting.
