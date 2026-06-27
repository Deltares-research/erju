# Query-Conditioned Curve-Prior Residual Model

## Objective

Extend the validated fixed-output curve-prior model (P3_corrected_n, RMSE log 0.5993) to **predict log-PGV for arbitrary receiver distances**.

The key innovation: **residual head takes distance query as input**, enabling out-of-sample generalization.

## Model Architecture

```
Physics:
  y_pred(r) = c_hat - n_track*log(r/r0) + epsilon_hat(r)
  
where:
  c_hat = event intensity from FO waveform + metadata
  n_track = train-fitted attenuation exponent (corrected method)
  r = active-track distance of queried receiver (meters)
  epsilon_hat(r) = learned residual as function of event embedding + distance query
  r0 = 10 m reference distance
```

### Encoder (Shared)
- 4-layer 2D CNN on 21 FO channels
- Channel progression: 1 → 16 → 32 → 32 → 64
- Temporal kernels: [15, 9, 7, 5], strides: [2, 2, 2, 2]
- Output: event embedding h_i

### Intensity Head
- Dense: [event_emb] → [64, 32] → c_hat

### Residual Head (Optional)
- Query features → embedding (3 → 16 dims)
- Dense: [event_emb, query_emb] → [64, 32] → epsilon_hat

### Query Features (Minimal Set)
For arbitrary-distance generalization:
- log_r_ratio = log(r_active_track / r0)
- r_active_track_m = distance in meters
- track_number = {0, 1}

## Variants

| Variant | Residual | λ_epsilon | MP4 Weight | Purpose |
|---------|----------|-----------|-----------|---------|
| **Q1** | No | 0.0 | 1.0 | Curve-only: test pure physics |
| **Q2** | Yes | 0.05 | 1.0 | Light residual regularization |
| **Q3** | Yes | 0.05 | 2.0 | Residual + MP4 weighting |
| **Q4** | Yes | 0.20 | 2.0 | Strong residual regularization |

## Held-Out Distance Tests

**Critical validation:** Can the model predict for sensor distances not in training?

### Test Configuration

Train on 4 sensors, test on held-out sensor:

```
holdout_MP4:   Train on [MP8, MP10, MP1, MP2]     Test on MP4 (2.5m, closest)
holdout_MP8:   Train on [MP4, MP10, MP1, MP2]     Test on MP8 (4.0m, near)
holdout_MP2:   Train on [MP4, MP8, MP10, MP1]     Test on MP2 (23m or 27m, far)
```

**Key metric:** RMSE(log) on held-out sensor test set.

### Expected Results

- **Q1 (curve-only):** RMSE(log) should degrade slightly on holdout, but stay reasonable if physics is accurate
- **Q3 (with residuals):** Should improve on holdout if learned patterns are transferable
- **Comparison:** If Q3_holdout is close to Q3_all, then distance-dependent residuals generalize well

## Dataset & Splits

### Data
- Line-C side -1, 1697 events, 5 sensors
- FO: ch51 slice [19:40] (channels 1184-1204)
- Waveform: 21 channels × 7500 samples @ 1kHz
- Target: PGV_z (vertical) only

### Splits (Event-Level)
- Train: 1103 events (65%)
- Val: 254 events (15%)
- Test: 340 events (20%)

### Held-Out Mode
- Training loss: use only non-held sensors
- Validation for early stopping: use only non-held sensors
- Test reporting: report both non-held and held-out sensor performance

## Training Settings

```
Optimizer: AdamW
Learning rate: 3e-4
Gradient clipping: max_norm=1.0
Batch size: 16
Epochs: 100 max
Early stopping: patience=15 on validation RMSE(log)
Best checkpoint: save/restore
```

## Attenuation Exponent Fitting

Always use **event-intercept corrected method**:
```
Fit on train split only
n_track1 = 1.0655 (±0.02)
n_track2 = 1.3246 (±0.02)
Fixed for val/test
```

## SLURM First-Pass Experiment

### Grid (10 Runs)

**All-Sensor (4 runs):**
- Q1_all
- Q2_all
- Q3_all
- Q4_all

**Held-Out (6 runs):**
- Q1_holdout_MP4
- Q3_holdout_MP4
- Q1_holdout_MP8
- Q3_holdout_MP8
- Q1_holdout_MP2
- Q3_holdout_MP2

### Comparison Baselines

| Model | RMSE(log) | R² | Notes |
|-------|-----------|-----|-------|
| Row-wise CNN v2 | 0.595 | 0.654 | Event-agnostic baseline |
| P3_corrected_n | 0.5993 | 0.6604 | Fixed-output curve-prior |
| P1_corrected_n | 0.6146 | 0.6429 | Curve-only (fixed-output) |
| Q1_all | ? | ? | Query curve-only (should match P1 approx) |
| Q3_all | ? | ? | Query with residuals (should match P3 approx) |

## Decision Rules

**If Q3_all ≈ P3_corrected_n:**
→ Query model is at least as good as fixed-output model
→ Proceed to cross-validation and generalization analysis

**If Q1_holdout shows reasonable performance:**
→ Physics curve generalizes to unseen distances
→ Distance-dependent residuals can add real value

**If Q3_holdout >> Q1_holdout:**
→ Learned residuals capture systematic distance-dependent patterns
→ Arbitrary-distance prediction is promising

**If Q3_holdout < 0.63 (comparable to P3_all 0.5993):**
→ Held-out distance generalization is working
→ Proceed to next phase: cross-line validation

**If Q3_holdout >> 0.65 (worse than row-wise baseline 0.595):**
→ Query residuals are not transferable across distances
→ Need more data or architecture changes
→ Hold current P3_corrected_n as production baseline

## Files

### Config & Utils
- `src/ml/cnn/config_cnn_curvequery_linec_v1.py` — Variant configs
- `src/ml/cnn/cnn_curvequery_linec_utils.py` — Model, dataset, losses

### Training
- `train_cnn_curvequery_linec_v1.py` — Main training script
- `smoke_test_curvequery_linec.py` — Pre-flight checks

### Submission
- `slurm/run_curvequery_linec_v1.slurm` — 10-run SLURM job

### Diagnostics
- `extract_curveprior_diagnostics.py` — Extract P3 baseline metrics
- `CURVEPRIOR_P3_DIAGNOSTICS_20260627.md` — P3 diagnostic report (to be generated)

## Local Validation

Before cluster submission:

```bash
# 1. Extract P3 diagnostics
python extract_curveprior_diagnostics.py \
    --model_dir /p/11210978-erju-ai/holten_models/cnn_curveprior_linec_v001_vP3_fit_corrected_20260627_211829

# 2. Run smoke checks
python smoke_test_curvequery_linec.py

# 3. Review outputs
cat CURVEPRIOR_P3_DIAGNOSTICS_20260627.md
```

If all checks pass:

```bash
# Submit to cluster
sbatch slurm/run_curvequery_linec_v1.slurm

# Monitor
squeue -u camposmo
tail -f outputs/curvequery_linec_v1/Q1_all.log
```

## Expected Runtime

- Per variant: ~3-5 min (depends on early stopping)
- 10 variants × 4-5 min = ~40-50 min total
- SLURM allocation: 12 hours (safe margin)

## Outputs Per Variant

```
/p/11210978-erju-ai/holten_models/cnn_curvequery_linec_v001_vQ*_*_*/
  ├── model.pth              # Trained weights
  ├── config_snapshot.json   # Variant config + fitted n
  └── metrics.json           # test_rmse_log, best_epoch, n values
```

## Analysis After Run

### Step 1: Verify All-Sensor Performance
```python
# Load Q1_all, Q2_all, Q3_all, Q4_all metrics
# Compare RMSE(log) against P3_corrected_n baseline (0.5993)
# Expected: Q3_all ≈ 0.60 ± 0.01
```

### Step 2: Analyze Held-Out Sensor Generalization
```python
# Per held-out sensor:
#   1. Load test predictions for held-out sensor
#   2. Compute RMSE(log) on held-out only
#   3. Compare Q1_holdout vs Q3_holdout
#   4. Benchmark against physics curve baseline
```

### Step 3: Decision
Based on results, decide:
- Continue with arbitrary-distance generalization?
- Keep P3_corrected_n as production baseline?
- Pivot to row-wise + physics features?

## Note: Do Not Implement Yet

Out of scope for this phase:
- Transformers, attention mechanisms
- PINNs (physics-informed neural networks)
- 51-channel full waveform models
- Cross-line models
- Strain-to-velocity conversion
- 60-second waveforms

These can be explored **after** query-conditioned validation succeeds.

---

**Status:** Ready for implementation and testing  
**Created:** 2026-06-27  
**Phase:** Query-conditioned curve-prior validation
