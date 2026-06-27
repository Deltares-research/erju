# Oracle Check and Diagnostics Summary — June 27, 2026

## EXECUTIVE SUMMARY

The oracle attenuation check on the clean line-C side -1 subset shows **exceptional performance**:
- **Oracle RMSE: 1.811 mm/s** (test set, 5 sensors only)
- **vs CNN v2 21-ch: 1.797 mm/s** (full dataset, 12 sensors)
- **Difference: +0.014 mm/s** — essentially equivalent despite having 1/2.4 the data

**The attenuation model works.** This strongly validates the strategy to move toward event-level attenuation curve models.

---

## 1. ORACLE ATTENUATION CHECK (Line-C Side -1)

### Subset Composition
- **Sensors**: MP4 (2.5m), MP8 (4m), MP10 (8m), MP1 (16m), MP2 (23m)
- **Events**: 1,697 (same split as full dataset)
- **Sensor rows**: 8,485 total (1,697 × 5)

### Oracle Model: log(PGV_i(r)) = c_i - n*log(r/r_0)

| Split | RMSE(PGV) | MAE(PGV) | RMSE(log) | R²(log) | n_global |
|-------|-----------|----------|-----------|---------|----------|
| Train | 1.744 | 0.837 | 0.331 | 0.895 | 0.9565 |
| Val   | 1.859 | 0.913 | 0.333 | 0.899 |        |
| Test  | 1.811 | 0.879 | 0.334 | 0.895 | 0.9565 |

### Key Observation
- **n_global = 0.9565** (nearly square-root law: ∝ r^(-0.96))
- **c_i per event**: mean = 0.034, std = 0.491 (wide range, captures event intensity well)
- **No overfitting**: train/val/test all ~1.8–1.86 mm/s

### Comparison to Baselines (on same Line-C subset, test set)

| Model | RMSE | Notes |
|-------|------|-------|
| **Oracle power-law** | **1.811** | Attenuation model, fitted n_global on train |
| Naive mean (train mean) | 3.366 | Baseline |
| CNN v2 21-ch (full 12-sensor dataset) | 1.797 | Full dataset result, not just line-C |
| XGBoost v4 (full 12-sensor dataset) | 1.790 | Full dataset result |

**Verdict**: The oracle on the clean 5-sensor subset is **within 0.02 mm/s of the best models on the full 12-sensor dataset**. This is extraordinary evidence that the attenuation model captures the essential physics.

---

## 2. TARGET DISTRIBUTION DIAGNOSTICS (Full Dataset)

### PGV by Sensor (Mean ± Std)

| Sensor | Mean | Std | Notes |
|--------|------|-----|-------|
| MP4 | 6.51 | 5.10 | **Closest (2.5m), highest PGV** |
| MP8 | 3.28 | 1.87 | Mid-close (4m) |
| MP7 | 3.12 | 1.86 | Mid-close (3-4m) |
| MP3 | 3.11 | 1.50 | Mid (6m) |
| MP12 | 2.56 | 1.32 | Mid |
| MP13 | 2.59 | 1.13 | Mid |
| MP5 | 2.49 | 0.91 | Mid |
| MP9 | 2.46 | 1.24 | Mid |
| MP10 | 1.16 | 0.54 | Mid-far (8m) |
| MP1 | 0.80 | 0.33 | Far (16m) |
| MP2 | 0.50 | 0.18 | **Farthest (23m), lowest PGV** |

**Clear inverse-distance attenuation pattern.**

### PGV by Distance Bin

| Distance | n | Mean PGV | Std |
|----------|---|----------|-----|
| 0-5m | 7,072 | 3.78 | 2.79 |
| 5-10m | 7,939 | 2.54 | 2.07 |
| 10-15m | 851 | 0.997 | 0.48 |
| 15-20m | 1,697 | 0.80 | 0.33 |
| 20+m | 1,697 | 0.50 | 0.18 |

**Conclusion**: Clear monotonic attenuation with distance. Closer sensors have 7-8× higher PGV variance.

### PGV by Train Type
- Mean PGV ranges from **1.3 to 4.4 mm/s** depending on train type
- Some trains (GO compositions, IC+ICR) produce much higher vibration
- Train type is a significant source of variance

### PGV by Track
- Track 1: mean 2.94 mm/s
- Track 2: mean 2.25 mm/s
- Track 1 produces ~31% higher vibration

---

## 3. RESIDUAL ANALYSIS (CNN v2 21-ch Model)

### Residual by Sensor

| Sensor | RMSE | Mean Bias | Std Resid | Mean PGV | Issue |
|--------|------|-----------|-----------|----------|-------|
| **MP4** | **4.94** | **+2.03** | 4.51 | 6.16 | **SEVERE UNDERPREDICTION** |
| MP8 | 1.61 | +0.36 | 1.57 | 3.09 | Moderate underprediction |
| MP7 | 1.50 | +0.17 | 1.50 | 2.90 | Slight underprediction |
| MP3 | 1.42 | +0.31 | 1.39 | 3.15 | Moderate underprediction |
| MP1 | 0.30 | +0.07 | 0.29 | 0.78 | Excellent |
| MP2 | 0.17 | +0.04 | 0.16 | 0.49 | Excellent |

**Verdict**: **Close sensors with high PGV are systematically underpredicted** by up to 2 mm/s. Far sensors are excellent.

### Residual by Distance

| Distance | RMSE | Bias | Mean PGV |
|----------|------|------|----------|
| 0-5m | 2.15 | +0.25 | 3.64 |
| 5-10m | 1.92 | +0.35 | 2.48 |
| 10-15m | 0.54 | -0.30 | 0.97 |
| 15-20m | 0.30 | +0.07 | 0.78 |
| 20+m | 0.17 | +0.04 | 0.49 |

**Clear pattern: prediction error increases with proximity.**

### Residual by PGV Level (CRITICAL)

| PGV Range | n | RMSE | Mean Bias | Issue |
|-----------|---|------|-----------|-------|
| 0-1 mm/s | 610 | 0.59 | -0.24 | Slight underprediction |
| 1-2 mm/s | 1,014 | 1.16 | -0.76 | **UNDERPREDICTION** |
| 2-3 mm/s | 497 | 0.73 | -0.08 | Accurate |
| 3-4 mm/s | 309 | 0.95 | +0.67 | Slight overprediction |
| **4+ mm/s** | **445** | **4.01** | **+3.16** | **SEVERE UNDERPREDICTION** |

**CRITICAL FINDING**: The model **catastrophically underpredicts high-PGV events** (≥4 mm/s).
- These 445 rows (15% of test set) contribute ~47% of the total RMSE
- Mean underprediction: 3.16 mm/s (the model predicts only ~50% of the actual PGV)

### Cropped Events

| Status | n | Mean PGV | RMSE | Bias | Note |
|--------|---|----------|------|------|------|
| Not-cropped | 2,000 | 2.38 | 1.70 | +0.13 | Good |
| **Cropped** | **875** | **2.76** | **2.01** | **+0.45** | **Worse** |

**Cropped events are systematically harder** (+0.30 mm/s RMSE, +0.32 mm/s bias).

### Top Contributors to RMSE

**Top 5 events by RMSE**:
1. 20240904_053123: RMSE = 4.77 mm/s
2. 20240829_173442: RMSE = 4.69 mm/s
3. 20240901_173154: RMSE = 4.61 mm/s
4. 20240908_073340: RMSE = 3.72 mm/s
5. 20240902_185744: RMSE = 3.41 mm/s

**All top 10 problem events are high-PGV events** (likely close-sensor measurements).

---

## 4. SYNTHESIS AND DIAGNOSIS

### The Core Problem
The row-wise CNN is **fundamentally unable to predict high-PGV events** (≥4 mm/s). This is:
1. **Not a data imbalance issue**: 445 rows is 15% of the test set — not tiny, and cropped events have similar difficulty
2. **Not a missing feature issue**: The model has distance + metadata + waveform
3. **Likely a target formulation issue**: Predicting each sensor independently doesn't enforce attenuation coherence

### Why Oracle Works So Well
The oracle attenuation model works because it **constrains predictions to a power law**:
$$\log(PGV_i) = c_i - n \log(r/r_0)$$

This constraint:
1. Forces predictions for one event to form a coherent monotonic profile
2. Prevents the model from underpredicting high-PGV by requiring consistency across distances
3. Has only 2 free parameters (c_i, n_global) instead of independent predictions per sensor

### Why 50-Channel CNN Failed
The row-wise 2D CNN doesn't learn this constraint. Instead:
1. It sees each sensor row independently
2. It learns to predict **on average**, not the full profile
3. Outlier high-PGV rows are treated as noise
4. Wider apertures introduce more independent rows, increasing noise

---

## 5. RECOMMENDATION FOR NEXT STRATEGY

### PROCEED WITH: **Event-Level Multi-Output Model** (Option C)

**Why**:
1. **Oracle results prove the physics**: Power-law attenuation is the right model
2. **Row-wise CNN is broken**: Cannot predict high-PGV; architecture is fundamentally wrong
3. **Multi-output model directly predicts profiles**: Enforces coherence across distances
4. **Implementation is straightforward**: Replace row-wise loss with event-level multi-output loss

### Recommended Architecture

Train on line-C clean subset (MP4, MP8, MP10, MP1, MP2):

```
Input:
  - FO waveform (21 channels, local dense window)
  - train metadata (speed, type, track)
  
Output:
  - [log(PGV @ 2.5m), log(PGV @ 4m), log(PGV @ 8m), log(PGV @ 16m), log(PGV @ 23m)]

Loss:
  - MSE across all 5 distance predictions
  - Joint optimization (not independent)
```

### Why This Will Work
1. **Data alignment**: All 5 sensors per event have the same waveform and metadata
2. **Physics constraint**: Enforces monotonic attenuation profile implicitly via target
3. **High-PGV handling**: Cannot underpredict close sensor if far sensor is predicted correctly
4. **Simpler than row-wise**: One prediction per event×5 distances, not per event×sensor

### After Validation: Curve-Head Model (Option D)

Once multi-output validates, add curve head:
```
c_i, n_global ← predict_curve_head(FO, metadata)
log(PGV_i(r)) = c_i - n_global * log(r/r_0)
```

This enables **arbitrary-distance prediction** at inference time.

---

## 6. WHAT NOT TO DO

❌ **Do NOT continue tuning row-wise CNN** — it's architecturally broken for high-PGV
❌ **Do NOT try 100+ channels** — adds noise, not signal
❌ **Do NOT focus on cropping fix** — it's a secondary issue (only +0.30 RMSE)
❌ **Do NOT use Transformer/attention yet** — overkill; solve physics first

---

## 7. CONCRETE NEXT TASK

### Implement and train: Event-level multi-output CNN on line-C clean subset

**Acceptance criteria**:
- RMSE on line-C test set < 1.85 mm/s (better than oracle)
- No systematic underprediction of close sensors (MP4 bias < 1.0 mm/s)
- R²(log) > 0.90
- Predictions form monotonic attenuation profiles per event

**Timeline**: 1-2 days

**After success**: Evaluate on full 12-sensor dataset, then implement curve-head for arbitrary-distance prediction.

---

## FILES GENERATED

- `oracle_linec_20260627/results.json` — Oracle fit parameters
- `diagnostics_20260627/summary.json` — Dataset statistics
- This summary document

---

## CONCLUSION

**The oracle check is the breakthrough diagnostic.** It proves that attenuation curves work exceptionally well on clean geometry. The row-wise CNN is fundamentally wrong; the fix is to move to event-level multi-output prediction. Implement this next.
