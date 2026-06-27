# Curve-Prior + Residual Model Experiment
**ERJU Event-Level PGV Prediction — Line-C, Track-Conditioned Physics Prior**

**Date:** 2026-06-27  
**Status:** Implementation complete, ready for cluster submission  
**Smoke tests:** All passing locally

---

## Executive Summary

### Problem
The generic two-head multi-output event-level model achieves RMSE(PGV) = 2.620 mm/s, which is 11.6% above the row-wise CNN v2 baseline (2.348 mm/s). Event-level aggregation is not providing sufficient benefit to justify the complexity.

### Solution
Implement a **physics-informed curve-prior + residual model** that:
1. **Leverages track-specific attenuation** directly (not learned from scratch)
2. **Predicts event intensity** (single scalar) using FO waveform + metadata
3. **Optionally learns small residuals** for sensor-specific corrections
4. **Applies MP4 weighting** to address close-sensor bias

### Expected Benefit
- Constrain predictions to physically plausible profiles
- Improve interpretability (curve is explicit, not black-box)
- Reduce overfitting by using prior knowledge
- Better handle close sensors (MP4)

---

## Physics Model

### Mathematical Formulation

$$\hat{y}_{ij} = \hat{c}_i - n_{\text{track}} \log(r_{j,\text{track}} / r_0) + \hat{\epsilon}_{ij}$$

where:
- $\hat{y}_{ij}$: predicted log-PGV at event $i$, sensor $j$
- $\hat{c}_i$: predicted event intensity (scalar, learned)
- $n_{\text{track}}$: track-specific attenuation exponent (fitted on train split, fixed)
- $r_{j,\text{track}}$: active-source distance to sensor $j$ (fixed per track)
- $r_0$: reference distance (10 m)
- $\hat{\epsilon}_{ij}$: learned residual correction (optional, regularized)

### Track-Specific Distances (Active-Source)

Derived from oracle distance audit, NOT distance-to-FO.

**Track 1:**
```
r_track1 = [2.5, 4.0, 8.0, 16.0, 23.0] m
```

**Track 2:**
```
r_track2 = [6.5, 8.0, 12.0, 20.0, 27.0] m
```

**Sensor order (consistent with all models):**
```
[MP4, MP8, MP10, MP1, MP2]
```

### Attenuation Exponents

From oracle audit (initial values, then fitted on train split):

```
Track 1: n_global ≈ 1.0777
Track 2: n_global ≈ 1.3300
```

**Important:** Fit $n_{\text{track1}}$ and $n_{\text{track2}}$ **on train split only** to prevent leakage. Use fitted values for val/test reconstruction.

---

## Model Architecture

### Shared 2D CNN Encoder
Reuse from two-head model:
- 4 convolutional layers: ch [1→16→32→32→64]
- Kernels: spatial (kc=3), temporal (kt=[15,9,7,5])
- AdaptiveAvgPool → flatten to embedding

### Intensity Head
- Input: CNN embedding + metadata embedding
- Output: $c_{\text{hat}}$ (single scalar)
- 2 hidden layers: [64, 32]

### Residual Head (Optional)
- Input: CNN embedding + metadata embedding
- Output: $\hat{\epsilon}$ (5 values, one per sensor)
- 2 hidden layers: [64, 32]
- Active only in P2 and P3 variants

### Metadata Encoding
- Input: train_speed_kmh, train_type_code, track_number (normalized)
- 2 hidden layers: [32, 16]
- Concatenated with CNN embedding

---

## Variants

### P1 — Curve-Only (Baseline Physics)
```
epsilon_hat = 0
y_pred = c_hat - n_track * log(r/r0)
```

**Loss:**
```
L = L_profile + 0.2 * L_c
```

**Purpose:** Test if FO encoder + metadata can predict event intensity accurately.

---

### P2 — Curve + Residual (Regularized Corrections)
```
epsilon_hat ≠ 0 (learned, regularized)
y_pred = c_hat - n_track * log(r/r0) + epsilon_hat
```

**Loss:**
```
L = L_profile + 0.2 * L_c + 0.05 * mean(epsilon_hat^2)
```

**Purpose:** Allow small sensor-specific corrections while staying close to physics curve.

---

### P3 — Curve + Residual + MP4 Weighting (Close-Sensor Bias)
```
Same as P2, but with per-output weighting.
MP4 weight = 2.0, others = 1.0
```

**Loss (same as P2, but weighted):**
```
L_profile = mean(weights * (y_pred - y_true)^2)
L = weighted_L_profile + 0.2 * L_c + 0.05 * L_eps
```

**Purpose:** Address the close-sensor underprediction (MP4 RMSE=4.937 in CNN v2).

---

### P4 — (Optional) Monotonicity-Constrained
Not implemented yet. Would add:
```
L_mono = penalty for violations of monotonicity in y_pred
```
Can add if P2/P3 show non-monotonic profiles.

---

## Loss Functions

### Profile Loss (Main)
```python
L_profile = Huber(y_pred - y_true, delta=0.5)
           with optional per-output weights (MP4 = 2.0)
```

**Why Huber:** More robust to outliers than MSE, good for log-PGV.

### Intensity Auxiliary Loss
```python
L_c = mean((c_hat - c_target)^2)
where c_target = mean_j(y_true_j + n_track * log(r_j / r0))
```

**Purpose:** Directly supervise event intensity prediction; encourage reasonable c_hat.

**Weight:** 0.2 (not dominant, physics curve does most work).

### Residual Regularization
```python
L_eps = mean(epsilon_hat^2)
```

**Weight:** lambda_eps = 0.05 (P2/P3 only).

**Purpose:** Keep residuals small; prevent over-fitting to sensor-specific noise.

### Monotonicity Penalty (P4 if run)
```python
L_mono = mean(relu(y_pred[:,j+1] - y_pred[:,j]))
```

Would constrain predictions to decrease with distance.

---

## Training Configuration

### Shared
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 3e-4 |
| LR decay | 0.95 per 1000 steps |
| Epochs | 100 |
| Batch size | 16 |
| Patience (early stopping) | 15 |
| Gradient clip | 1.0 |
| Loss | Huber, delta=0.5 |

### Best Checkpoint
- Metric: validation RMSE(log)
- Save/restore: best model state before final evaluation
- Early stopping: on patience without improvement

### Prediction Clamping (Numerical Stability)
```python
pred_log = np.clip(pred_log, -10, 5)
before exp()
```

---

## Data & Splits

### Subset
- **Line:** C (side -1 only)
- **Sensors:** MP4, MP8, MP10, MP1, MP2 (5 sensors)
- **Events:** ~1697
- **Rows:** ~8485 (5 per event)

### Event-Level Splits
| Split | Fraction | Count |
|-------|----------|-------|
| Train | 65% | 1103 |
| Val | 15% | 254 |
| Test | 20% | 340 |

**Important:** Fit $n_{\text{track}}$ on **train split only**.

### Waveforms
- **Source:** ch51 build (channels 1165-1215, all 51 channels)
- **Slice:** [19:40] → channels 1184-1204 (line-C window, center 1194)
- **Sample rate:** 7500 samples @ 1 kHz → 7.5 seconds
- **Dtype:** float32 (scaled to microstrain)

---

## Expected Baselines

### Oracle Physics (Line-C, Track-Specific)
```
Track 1: RMSE(log) = 0.3956, R² = 0.913
Track 2: RMSE(log) = 0.4017, R² = 0.874
```
(Ceiling if we fit event intensity with target n values.)

### Row-Wise CNN v2 (Line-C Subset)
```
RMSE(PGV) = 2.348 mm/s
RMSE(log) = 0.595
R²(log) = 0.654
MP4 RMSE = 4.937, bias = -2.033
```
(Target to match or beat.)

### Two-Head D_fixed (Event-Level, No Physics)
```
RMSE(PGV) = 2.620 mm/s
RMSE(log) = 0.613
R²(log) = 0.645
```
(To be surpassed.)

---

## Success Criteria

### Minimum (Viability)
- [ ] Any variant RMSE(log) ≤ 0.62 (within row-wise range)
- [ ] No NaN/inf, stable training
- [ ] Best checkpoint properly restored

### Target (Improvement)
- [ ] At least one variant RMSE(log) < 0.595 (beats row-wise)
- [ ] P3 RMSE(log) < P2 (weighting helps)
- [ ] MP4 bias closer to 0 than -2.033

### Favorable (Interpretability + Performance)
- [ ] c_hat correlates well with c_target (R² > 0.5)
- [ ] Epsilon magnitudes stay small (mean < 0.1)
- [ ] Profiles remain mostly monotonic
- [ ] Per-track metrics show track-specific patterns

---

## Files & Submission

### New Files Created
```
train_cnn_curveprior_linec_v1.py
  Main training script

src/ml/cnn/config_cnn_curveprior_linec_v1.py
  Config with P1/P2/P3 presets

src/ml/cnn/cnn_curveprior_linec_utils.py
  Model, losses, utilities

slurm/run_curveprior_linec_v1.slurm
  SLURM job script for cluster

smoke_test_curveprior.py
  Local verification (PASSED)
```

### Cluster Submission
```bash
sbatch slurm/run_curveprior_linec_v1.slurm
```

Runs P1 → P2 → P3 sequentially.  
Expected runtime: ~30 min for all variants.

### Output Structure
```
/p/11210978-erju-ai/holten_models/
  cnn_curveprior_linec_v001_vP1_<timestamp>/
    config_snapshot.json       (includes fitted n_track1, n_track2)
    metrics.json               (best_epoch, combined, per-track)
    predictions.parquet        (full test predictions)

  cnn_curveprior_linec_v001_vP2_<timestamp>/
    [same as P1]

  cnn_curveprior_linec_v001_vP3_<timestamp>/
    [same as P1]
```

---

## Comparison & Analysis

### Metrics to Report
**Combined:**
- RMSE(PGV), MAE(PGV), RMSE(log), MAE(log), R²(log), R²(PGV)

**Per-track (T1, T2):**
- RMSE(PGV), RMSE(log), R²(log)

**Per-sensor (MP4, MP8, MP10, MP1, MP2):**
- RMSE, bias, MAE

**High-PGV (> 4 mm/s):**
- RMSE, bias

**Curve diagnostics:**
- c_hat vs c_target correlation
- epsilon magnitude distribution
- monotonicity violation rate

---

## Decision Framework

### Proceed to Full Event-Level Curve-Prior Extension IF:
- ✓ At least one variant RMSE(log) < 0.60
- ✓ Weighting shows benefit (P3 > P2)
- ✓ c_hat well-correlated (R² > 0.4)

→ Then: Implement full arbitrary-distance curve-prior (handle variable # sensors)

### Pivot to Row-Wise + Physics IF:
- ✗ All variants RMSE(log) > 0.62
- ✗ Weighting ineffective
- ✗ c_hat poorly correlated

→ Then: Use row-wise with track-conditioned weighting + oracle n_track

### Continue Diagnostic IF:
- ⚠️ Results mixed (some good, some bad)
- ⚠️ Need per-sensor analysis

→ Then: Extract detailed predictions, analyze per-sensor biases

---

## Local Verification

All smoke tests pass:
```
✓ Config loading
✓ Data loading (8485 rows, 1697 events)
✓ Dataset construction (1697 samples)
✓ Distance vectors correct
✓ Model forward pass
✓ Loss computation (weighted)
✓ Attenuation fitting (n_track1 ≈ 1.08, n_track2 ≈ 1.33)
```

**Ready for cluster submission.**

---

## Next Steps

1. **Submit:** `sbatch slurm/run_curveprior_linec_v1.slurm`
2. **Monitor:** Check `/u/camposmo/erju/erju_curveprior_v1_<jobid>.log`
3. **Download:** Metrics and predictions from all three variants
4. **Analyze:** Compare P1/P2/P3 to baselines
5. **Decide:** Proceed to full curve-prior or pivot to row-wise+physics

---

**Prepared by:** Debugging session 2026-06-27  
**Status:** Ready for production cluster run  
**Confidence:** High (physics model well-specified, smoke tests pass)
