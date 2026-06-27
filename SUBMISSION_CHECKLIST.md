# IMPLEMENTATION COMPLETE: Curve-Prior + Residual Model Experiment
**Status:** Ready for cluster submission  
**Date:** 2026-06-27

---

## What Was Implemented

### Core Model (Physics-Informed)
$$\hat{y}_{ij} = \hat{c}_i - n_{\text{track}} \log(r_j/r_0) + \hat{\epsilon}_{ij}$$

- **c_hat:** Learned event intensity (from CNN + metadata)
- **n_track:** Track-specific attenuation (fitted on train, fixed for val/test)
- **epsilon_hat:** Optional learned residuals (regularized)
- **Distances:** Physical track distances (not distance-to-FO)

### Three Variants
| Variant | Setup | Lambda_eps | MP4 Weight | Purpose |
|---------|-------|-----------|-----------|---------|
| **P1** | Curve only | 0.0 | 1.0 | Physics baseline |
| **P2** | Curve + residual | 0.05 | 1.0 | Regularized corrections |
| **P3** | Curve + residual + weighting | 0.05 | 2.0 | Close-sensor bias fix |

---

## Files Created

### Training Script
```
train_cnn_curveprior_linec_v1.py
```
- Data loading (line-C subset, 1697 events)
- Dataset construction with distance vectors
- Attenuation exponent fitting on train split
- Training loop with best checkpoint save/restore
- Full metrics computation
- All variants support

### Configuration
```
src/ml/cnn/config_cnn_curveprior_linec_v1.py
```
- Preset configs for P1/P2/P3
- Physical constants (distances, r0)
- Initial attenuation exponents
- All hyperparameters configurable

### Model & Utilities
```
src/ml/cnn/cnn_curveprior_linec_utils.py
```
- CurvePriorCNN2D (2D CNN encoder + intensity head + residual head)
- CurveDataset (event-level, includes distances)
- Loss functions (profile, intensity, residual regularization, monotonicity)
- Utility functions (weights, curve targets)

### SLURM Job Script
```
slurm/run_curveprior_linec_v1.slurm
```
- Runs P1 → P2 → P3 sequentially
- GPU request: 1 H100 PCIe
- Memory: 32 GB
- Time: 12 hours
- Outputs to: `/p/11210978-erju-ai/holten_models/cnn_curveprior_linec_v001_v*_<timestamp>/`

### Local Smoke Tests
```
smoke_test_curveprior.py
```
All tests PASS:
- ✓ Config loading
- ✓ Data loading
- ✓ Distance vectors
- ✓ Model forward pass
- ✓ Loss computation
- ✓ Attenuation fitting

### Documentation
```
CURVEPRIOR_EXPERIMENT_README.md     (comprehensive description)
SUBMISSION_CHECKLIST.md              (this file)
```

---

## How to Submit

### Local Test (Optional)
```bash
cd d:\codes\erju
python smoke_test_curveprior.py
```
Expected output: All smoke tests PASS

### Cluster Submission
```bash
sbatch slurm/run_curveprior_linec_v1.slurm
```

Or copy SLURM script to cluster:
```bash
scp slurm/run_curveprior_linec_v1.slurm p-login:/u/camposmo/erju/
ssh p-login "cd /u/camposmo/erju && sbatch run_curveprior_linec_v1.slurm"
```

### Monitor Progress
```bash
# Check job status
squeue -u camposmo

# Stream logs
tail -f /u/camposmo/erju/erju_curveprior_v1_<jobid>.log

# Check directory after completion
ls -la /p/11210978-erju-ai/holten_models/ | grep curveprior
```

---

## Variants & Runtime Estimates

| Variant | Description | Est. Runtime |
|---------|-------------|--------------|
| P1 | Curve-only (physics baseline) | ~3 min |
| P2 | Curve + residual (0.05) | ~3 min |
| P3 | Curve + residual + MP4 weight (2.0) | ~3 min |
| **Total** | Sequential | ~10 min |

Total cluster run: ~10-15 min (plus queue wait).

---

## Expected Output Structure

### Per Variant Directory
```
/p/11210978-erju-ai/holten_models/cnn_curveprior_linec_v001_vP1_20260627_HHMMSS/
  config_snapshot.json
    - variant, n_track1, n_track2 (fitted on train)
    - all model/train hyperparameters
  
  metrics.json
    - best_epoch, best_val_rmse_log
    - train, val, test metrics:
      * RMSE(PGV), MAE(PGV)
      * RMSE(log), MAE(log), R²(log), R²(PGV)
      * Per-track metrics (T1, T2)
  
  predictions.parquet
    - split, track, pred_log_*, true_log_*
    - (ready for downstream analysis)
```

### Log Files
```
/u/camposmo/erju/outputs/curveprior_linec_v1/
  variant_P1.log     (stdout + stderr)
  variant_P2.log
  variant_P3.log
```

---

## Baseline Metrics (Target)

### Row-Wise CNN v2 (Line-C Subset)
```
RMSE(PGV):  2.348 mm/s     ← Target to match/beat
RMSE(log):  0.595
R²(log):    0.654
MP4 RMSE:   4.937 mm/s
MP4 bias:   -2.033 mm/s    ← Want to improve
```

### Two-Head D_fixed (Event-Level)
```
RMSE(PGV):  2.620 mm/s     ← Previous best, 11% above baseline
RMSE(log):  0.613
R²(log):    0.645
```

---

## Success Criteria

### Minimum (Viability)
- [ ] At least one variant RMSE(log) ≤ 0.62
- [ ] No crashes, NaN/inf, or CUDA OOM
- [ ] Best checkpoint properly saved and restored

### Target (Improvement Over Two-Head)
- [ ] At least one variant RMSE(log) < 0.61
- [ ] P3 shows improvement over P2 (weighting helps)

### Excellent (Beats Row-Wise)
- [ ] At least one variant RMSE(log) < 0.595
- [ ] MP4 bias closer to 0 than -2.033

### Bonus (Interpretability)
- [ ] c_hat correlates with c_target (R² > 0.4)
- [ ] Epsilon magnitudes small (mean < 0.1)
- [ ] Predictions mostly monotonic

---

## Configuration Summary

### Data & Splits
- **Subset:** Line-C, side -1
- **Sensors:** MP4, MP8, MP10, MP1, MP2
- **Events:** 1697 (train: 1103, val: 254, test: 340)
- **Waveform:** 21-channel window [19:40] (1184-1204), scaled to microstrain

### Physical Constants
- **Track 1 distances:** [2.5, 4.0, 8.0, 16.0, 23.0] m
- **Track 2 distances:** [6.5, 8.0, 12.0, 20.0, 27.0] m
- **Reference distance (r0):** 10 m
- **Initial n_track1:** 1.0777
- **Initial n_track2:** 1.3300

### Training
- **Optimizer:** AdamW
- **Learning rate:** 3e-4
- **Loss:** Huber (delta=0.5)
- **Batch size:** 16
- **Epochs:** 100
- **Early stopping:** Patience 15 (on validation RMSE log)
- **Best checkpoint:** Saved & restored

### Loss Weights
- **Main profile loss:** 1.0
- **Intensity auxiliary (L_c):** 0.2
- **Residual reg (P2/P3):** 0.05
- **MP4 weight (P3 only):** 2.0 for output 0, 1.0 for others

---

## Troubleshooting

### If Job Fails

**1. Check logs:**
```bash
cat /u/camposmo/erju/erju_curveprior_v1_<jobid>.err
tail -100 /u/camposmo/erju/erju_curveprior_v1_<jobid>.log
```

**2. Common issues:**
- CUDA OOM: Reduce batch_size (current: 16)
- Data not found: Check parquet_v002_* and holten_waveform_v003_ch51_* exist
- Module import error: Check Python path in SLURM script

**3. Resubmit after fix:**
```bash
sbatch slurm/run_curveprior_linec_v1.slurm
```

### If Results Disappointing

**Check:**
1. Is n_track fitting reasonable? (logged in output)
2. Are losses decreasing? (check learning curve)
3. Is best_epoch being restored? (check log message)
4. Compare per-track metrics (track 1 vs 2 different?)

**Options:**
- Increase patience (15 → 20) to allow more epochs
- Increase lambda_residual (0.05 → 0.1) for stronger regularization
- Decrease learning rate (3e-4 → 1e-4) for finer convergence
- Adjust MP4 weight (2.0 → 3.0) if close-sensor bias still strong

---

## Files Checklist

### Core Implementation ✅
- [x] train_cnn_curveprior_linec_v1.py
- [x] src/ml/cnn/config_cnn_curveprior_linec_v1.py
- [x] src/ml/cnn/cnn_curveprior_linec_utils.py
- [x] slurm/run_curveprior_linec_v1.slurm
- [x] smoke_test_curveprior.py

### Documentation ✅
- [x] CURVEPRIOR_EXPERIMENT_README.md
- [x] SUBMISSION_CHECKLIST.md (this file)

### NOT Modified (Unchanged) ✅
- [x] train_cnn_twohead_linec_v1.py (no changes)
- [x] Row-wise CNN scripts (no changes)
- [x] Previous data/config files (no changes)

---

## Summary

✅ **Status: READY FOR SUBMISSION**

**What's been done:**
1. ✓ Physics model fully specified and implemented
2. ✓ Three variants (P1/P2/P3) configured
3. ✓ SLURM script prepared for cluster
4. ✓ All smoke tests pass locally
5. ✓ Comprehensive documentation prepared

**Next action:**
```bash
sbatch slurm/run_curveprior_linec_v1.slurm
```

**Expected completion:** ~10-15 min on cluster H100  
**Results location:** `/p/11210978-erju-ai/holten_models/cnn_curveprior_linec_v001_v*_*/`

---

**Prepared by:** System implementation 2026-06-27  
**Decision date:** After two-head D_fixed results (RMSE: 2.620 mm/s, 11% above baseline)  
**Next decision:** After curve-prior P1/P2/P3 results available
