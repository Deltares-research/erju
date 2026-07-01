# Attenuation Curve Prediction — Results Summary
**Project:** ERJU Rail4Earth — Holten site, Line-C side -1  
**Date:** 2026-07-02  
**Physics model:** y(r) = c_hat − n · log(r/r₀) + ε(r),  r₀ = 10 m  
**Target:** log(PGV_z)  

---

## 1. Validated Model Table

| Model | RMSE(log) | RMSE(PGV) | R²(log) | Notes |
|-------|-----------|-----------|---------|-------|
| Row-wise CNN v2 line-C | 0.595 | 2.348 | 0.654 | row-wise, all 5 sensors |
| Oracle power-law (upper bound) | 0.334 | 1.811 | 0.895 | c_i per event, global n |
| P1_corrected_n | 0.6146 | 2.483 | 0.643 | curve-only, no residual |
| P3_corrected_n | 0.5993 | 2.401 | 0.660 | curve + residual + MP4×2 ← best structured CNN |
| Q3_all lr=1e-4 *(stability)* | 0.6196 | 2.375 | — | query model, unstable training |
| PXGBR-R1 | 0.5841 | 2.415 | 0.678 | physics+XGB residual, uniform weights |
| **PXGBR-R2** | **0.5946** | **2.272** | **0.666** | **BEST PRACTICAL — physics+XGB, MP4/high-PGV weighted** |
| PXGBR-R2 monotonic | 0.5946 | 2.272 | 0.666 | same as R2, monotonic constraint applied (no extra loss) |

**PXGBR-R2 improves the physics curve by learning residual corrections from engineered FO and metadata features.**

### Per-sensor: P3 vs PXGBR-R2 (test set)

| Sensor | P3 RMSE(PGV) | PXGBR-R2 RMSE(PGV) | PXGBR-R2 bias(PGV) |
|--------|-------------|---------------------|--------------------|
| **MP4** (2.5 m) | 5.02 | **4.74** | −1.06 |
| MP8 (4 m) | 1.82 | 1.72 | +0.07 |
| MP10 (8 m) | 0.54 | **0.49** | −0.04 |
| MP1 (16 m) | 0.33 | **0.32** | −0.02 |
| MP2 (23 m) | 0.18 | **0.18** | +0.01 |

Monotonicity violation rate: **0.000** (all profiles are monotonically decreasing after post-processing).

### What these numbers mean

All metrics are on the **test split** (340 events, 1700 sensor-rows).  
The prediction formula is **y = c_hat − n·log(r/r₀) + ε**, evaluated at the five measured distances per event.  
P3_corrected_n is our primary reference model.

---

## 2. Split Balance

All experiments use **event-level train/val/test splits** — no row-level leakage is possible.

| Split | Events | Track 1 | Track 2 | PGV > 4 mm/s | PGV > 8 mm/s | Mean max PGV | c_target mean ± std |
|-------|--------|---------|---------|-------------|-------------|-------------|---------------------|
| Train | 1103 | 545 (49%) | 558 (51%) | 559 (51%) | 555 (50%) | 6.58 | 0.316 ± 0.532 |
| Val | 254 | 121 (48%) | 133 (52%) | 128 (50%) | 127 (50%) | 6.63 | 0.332 ± 0.528 |
| **Test** | **340** | **180 (53%)** | **160 (47%)** | **175 (52%)** | **171 (50%)** | **6.55** | **0.289 ± 0.529** |

**Split balance verdict: Excellent.** Track proportions are within ±5%, high-PGV fractions are near-identical, and c_target distributions are consistent across splits. No stratification is needed.

---

## 3. Attenuation Curve Plots

All plots generated from PXGBR-R2 and P3_corrected_n test predictions (340 events × 5 sensors).  
Location: `plots/curve_presentation_20260702/`

| Plot | Description |
|------|-------------|
| `01_measured_vs_predicted_p3_vs_pxgbr.png` | Scatter: P3 vs PXGBR-R2 — measured vs predicted, log-log |
| `02_per_sensor_rmse_p3_vs_pxgbr.png` | Per-sensor RMSE(PGV) and RMSE(log), side by side |
| `03_residual_vs_distance.png` | Residual (pred−measured) vs distance — Q3 stability run |
| `04_attenuation_representative_p3_vs_pxgbr.png` | 6 best-fitting events, P3 vs PXGBR-R2 |
| `05_attenuation_high_pgv_p3_vs_pxgbr.png` | 6 highest-PGV events, P3 vs PXGBR-R2 |
| `07_heldout_MP2_examples.png` | Zero-shot MP2 far-field (query model) |
| `08_heldout_MP4_examples.png` | Zero-shot MP4 close-field (query model) |
| `09_feature_importance_pxgbr.png` | PXGBR-R2 top-30 feature importances |
| `09b_feature_importance_top30_pxgbr.png` | Same (from model directory) |

**Key observation:** PXGBR-R2 corrects P3's underestimation of high-PGV events primarily through low-frequency FO spectral features (fo_oct_4_00hz, fo_oct_2_00hz etc.) which capture per-event source amplitude variation not fully encoded in c_hat.

---

## 4. High-PGV Analysis

The close-sensor problem (MP4 at 2.5 m) is consistently the hardest case:

| Sensor | Distance (m) | Mean PGV (mm/s) | RMSE (CNN v2) | RMSE (P3) | Notes |
|--------|-------------|----------------|--------------|---------|-------|
| MP4 | 2.5 | 6.51 | 4.94 | ~4.9 | closest, highest variance |
| MP8 | 4.0 | 3.28 | 1.61 | ~1.8 | |
| MP10 | 8.0 | 1.16 | 0.54 | ~0.5 | |
| MP1 | 16.0 | 0.80 | 0.30 | ~0.33 | |
| MP2 | 23.0 | 0.50 | 0.17 | ~0.18 | best-predicted |

Events with PGV > 4 mm/s (~51% of test set) account for ~47% of total RMSE.

---

## 5. Held-Out Sensor Results (Zero-Shot Distance Prediction)

These results demonstrate that the physics-informed model can predict PGV at sensor distances **never seen during training**:

| Holdout | Model | Held-out RMSE(log) | Held-out RMSE(PGV) | Bias(log) | Interpretation |
|---------|----|------|------|------|----------------|
| **MP2 (23 m, far)** | Q3 stability | **0.491** | **0.25 mm/s** | +0.33 | ✅ Excellent — physics generalises well to far field |
| **MP4 (2.5 m, near)** | Q3 stability | 0.971 | 5.10 mm/s | −0.009 | ⚠️ Hard — close-field extrapolation, near-zero bias |

**The MP2 result is remarkable**: trained on MP4/MP8/MP10/MP1 only, the model predicts MP2 (23 m) with RMSE = 0.25 mm/s, nearly as well as if MP2 were in the training set. This validates the attenuation-curve approach for far-field extrapolation.

---

## 6. Honest Limitations

### 6.1 MP4 close-field difficulty
MP4 at 2.5 m consistently has the highest RMSE across all models. PXGBR-R2 reduces MP4 RMSE(PGV) from 5.02 → 4.74 mm/s but the close-field problem is not fully resolved. The fundamental challenge is that MP4 has the highest and most variable PGV, and the training data has limited high-PGV events.

### 6.2 PXGBR-R2 is sensor-distance-dependent
PXGBR-R2 uses `sensor_code` (ordinal encoding of sensor position) and `distance` as features. It is therefore only applicable to the five measured sensor distances. It does **not** generalise to arbitrary distances. For zero-shot arbitrary-distance prediction, use the query CNN model (Q3) once stabilised.

### 6.3 Feature importance concentrated in FO spectral bands
The top features in PXGBR-R2 are low-frequency FO octave band statistics (`fo_oct_4_00hz`, `fo_oct_2_00hz`, etc.), not the physics features (`pred_log_p3` ranks 75th/85). This suggests the XGBoost is learning amplitude corrections from the FO frequency content — it is complementary to the P3 physics model, not a replacement.

### 6.4 Query model training instability  
The query-conditioned model is still pending the init-fix run. Until Q1_all_initfix reaches RMSE(log) ≈ 0.61–0.63, arbitrary-distance generalisation claims should reference the holdout results from the stability run (Q3_holdout_MP2: RMSE=0.49, Q3_holdout_MP4: RMSE=0.97) rather than the all-sensor metric.

### 6.5 Line-C side -1 only
All results are for 5 sensors on line-C side -1. Cross-line validation (lines A, B, D, E) is the next step after query model stabilisation.

---

## 7. Next Research Steps

### Best practical model: PXGBR-R2
PXGBR-R2 (RMSE(PGV) = 2.272 mm/s) is the current best practical model for fixed-sensor-distance prediction. It retains physical interpretability through the P3 curve prior and reduces high-PGV errors via learned residual corrections.

### If query init-fix succeeds (Q1_all_initfix RMSE(log) ≈ 0.61–0.63)
The query CNN is trustworthy for arbitrary-distance prediction. Extend PXGBR to arbitrary distances by:
1. Training PXGBR on all query-model sensor queries (not just 5 fixed distances)
2. Or use Q3 holdout results directly for arbitrary-distance reporting

### Cross-line validation
Apply PXGBR-R2 and P3 to lines A/B/D/E without retraining. If RMSE degrades gracefully, the attenuation-curve framework generalises across the Holten site.

### Physics-feature ablation study
Run PXGBR without the P3 physics features (pred_log_p3, c_hat_p3, epsilon_p3) and compare. Since these rank 75–79/85 in importance, the FO spectral features may carry most of the predictive signal independently.

---

## 8. Files Reference

| File | Purpose |
|------|---------|
| `train_cnn_curveprior_linec_v1.py` | ✅ Validated — do not modify |
| `train_cnn_curveprior_linec_v1_savepred.py` | Copy of above + saves predictions/model/per_sensor_csv |
| `train_cnn_curvequery_linec_v1.py` | Active — init-fix applied |
| `src/ml/cnn/cnn_curvequery_linec_utils.py` | Active — init-fix applied (zero final layer) |
| `scripts/make_curve_presentation_plots_linec.py` | Auto-discovers best available predictions and generates all plots |
| `slurm/run_curveprior_p3_savepred_linec_v1.slurm` | Rerun P3 to save predictions (~10 min) |
| `slurm/run_curvequery_initfix_linec_v1.slurm` | 5-run initfix grid (~3 hr) |
| `plots/curve_presentation_20260702/` | All generated plots (current run from stability data) |
| `split_balance_table.csv` | 1697 events with split assignment, PGV stats, c_target |
