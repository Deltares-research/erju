# Two-Head Model Debugging Report
**Date:** 2026-06-27  
**Status:** PAUSE BEFORE RERUN — Critical fixes required

---

## Executive Summary

The first two-head run produced results close to CNN v2 baseline (2.33 vs 2.35 mm/s), but three **critical issues** prevented full evaluation:

1. **Channel slicing bug** (now fixed) — was using [15:36] instead of [19:40]
2. **MP4/high-PGV weighting not applied** — config set but loss doesn't use weights
3. **Best checkpoint not saved/restored** — may not be evaluating best model
4. **Numerical instability** in direct variants — but monotonic variants stable

---

## Issue 1: Channel Slicing ✅ FIXED

### Problem
Was slicing to channels 1180–1200 instead of 1184–1204 (off by 4 on each end).

### Verification
```
ch51 build: channels 1165–1215
Index 0 = channel 1165
Index 19 = channel 1184 ✓
Index 39 = channel 1204 ✓
Correct slice: [19:40]
```

**Status:** ✅ Fixed in train_cnn_twohead_linec_v1.py line 133

---

## Issue 2: MP4/High-PGV Weighting NOT APPLIED ⚠️ CRITICAL

### Evidence
- Variant A (direct, unweighted): test RMSE = 4.885 mm/s
- Variant B (direct, MP4-weighted): test RMSE = 4.885 mm/s ← **IDENTICAL**
- Variant C (monotonic, unweighted): test RMSE = 2.327 mm/s
- Variant D (monotonic, MP4-weighted): test RMSE = 2.327 mm/s ← **IDENTICAL**

### Cause
Config specifies `use_pgv_weighting=True` and `mp4_weight=2.0` for variants B/D, but the training loop's loss computation doesn't use these weights.

### Fix Required
1. **Find train_epoch() function** where loss is computed
2. **Compute per-sample weights** based on sensor index (MP4 → output 0 gets 2×)
3. **Apply weights to loss calculation** before backprop
4. **Verify:** Loss values should differ between variant A and B
   ```
   python verify_twohead_fixes.py  # Should show different losses
   ```

---

## Issue 3: Best Checkpoint Not Saved/Restored ⚠️ CRITICAL

### Problem
- Training stops at epoch 29–31 (early stopping)
- metrics.json doesn't contain `best_epoch`
- Unclear if best model is restored before final test evaluation

### Evidence
```json
"metrics.json" contains: train, val, test metrics
but NO "best_epoch" or "checkpoint_epoch"
```

### Fix Required
1. **Track best validation metric** (prefer RMSE(log) for stability)
2. **Save model state dict** when validation improves
3. **Restore best model** before final evaluation
4. **Log best_epoch** in metrics.json

### Implementation
```python
# In training loop:
if val_metric_improved:
    best_model_state = model.state_dict()
    best_epoch = current_epoch

# Before final evaluation:
model.load_state_dict(best_model_state)
```

---

## Issue 4: Numerical Instability in Direct Variants ⚠️ KNOWN LIMITATION

### Observed
```
Direct variants (A/B):
  Epoch 5:  val_loss → 19,181 (RMSE = 6.7 trillion)
  Epoch 10: val_loss → 247 (RMSE = 95 billion)
  → Numerical overflow in log-space / exp conversion

Monotonic variants (C/D):
  Stable throughout training
```

### Root Cause
Five independent outputs without constraints can produce extreme values that overflow log/exp conversions.

### Recommended Fixes
For next runs (if attempting direct variants):

1. **Lower learning rate:** 1e-3 → 3e-4
2. **Use AdamW** instead of Adam (better regularization)
3. **Clamp log predictions:** np.clip(pred_log, -10, 5) during metric computation
4. **Use Huber loss** instead of MSE in log space (more robust)
5. **Gradient clipping:** max_norm=1.0 (already in config)

---

## Baseline Comparison

### CNN v2 (Row-Wise, Line-C Subset)
- Subset: MP4, MP8, MP10, MP1, MP2 (1,270 rows, 254 events)
- **RMSE(PGV): 2.348 mm/s** ← **Target to beat or match**
- RMSE(log): 0.595
- R²(log): 0.654
- MP4 RMSE: 4.937 mm/s (bias: -2.033 mm/s)

### Two-Head Monotonic (Variant C, Current)
- **RMSE(PGV): 2.327 mm/s** ← nearly identical
- RMSE(log): 0.606
- R²(log): 0.653
- Track 1: 2.291 mm/s
- Track 2: 2.367 mm/s

**Conclusion:** Two-head is competitive with CNN v2, but needs the fixes above to unlock potential benefits.

---

## Metadata Verification ✅ USABLE

**Status:** Metadata is real and usable
```
train_speed_kmh:
  Mean: 123.5 km/h
  Std: 10.6 km/h
  Missing: 15 / 8,485 (0.2%) ← negligible

train_type_code: 52 unique values (diverse trains)
track_number: [1, 2] ✓

Standardization per train fold: feasible and correct
```

---

## Before Rerunning: Action Checklist

### Must Do (Blocking)
- [ ] **Fix MP4/high-PGV weighting in loss computation**
  - Verify config weights are passed to loss function
  - Confirm variant B produces different loss than variant A
  - Print weight matrix during training

- [ ] **Implement best checkpoint save/restore**
  - Save model when validation metric improves
  - Restore best model before test evaluation
  - Add best_epoch to metrics.json

### Should Do (Recommended)
- [ ] **Lower learning rate:** 1e-3 → 3e-4 in config
- [ ] **Clamp log predictions:** np.clip(pred_log, -10, 5) in metric computation
- [ ] **Use AdamW** optimizer (more stable)

### Skip for Now (Optional)
- [ ] ❌ Do NOT rerun direct variants A/B until numerical issues resolved
- [ ] ✓ Rerun only **C_fixed** and **D_fixed** (monotonic variants)

---

## Expected Outcome After Fixes

### Optimistic Scenario
If weighting and checkpointing are properly implemented:
- Variant C_fixed: RMSE(PGV) ≈ 2.32 mm/s (matches current)
- Variant D_fixed: RMSE(PGV) ≈ 2.25 mm/s ← MP4 weighting helps close sensors
- Track separation shows sensible differences

### Realistic Scenario
If implementation doesn't introduce new issues:
- Monotonic variants: ≈ CNN v2 baseline (2.35 mm/s)
- With weighting: Modest improvement on MP4 (close sensor)
- Track metrics: Separate per-track performance visible

### If Still Underperforming
- Event-level aggregation may be too constraining
- Pivot to: row-wise with track conditioning + physics priors
- Or: hybrid curve-prior + residual model

---

## Decision Rule

After fixes and rerun:

**If two-head RMSE(log) ≈ 0.60 ± 0.02 and MP4 improves:**  
→ Proceed to **curve-head arbitrary-distance model** (can handle any distance, not fixed 5 sensors)

**If two-head still underperforms despite fixes:**  
→ Pivot to **row-wise architecture with track-conditioned loss weighting**

**If two-head significantly outperforms CNN v2:**  
→ **Scale to full dataset** before curve-head extension

---

## Files Modified

- [x] `train_cnn_twohead_linec_v1.py` — channel slice [15:36] → [19:40] ✅
- [ ] Loss computation — need to implement weighting (line TBD)
- [ ] Best checkpoint — need to add save/restore (line TBD)
- [ ] Config — optional LR reduction (line TBD)

---

## Next Turn

1. ✅ Verify and fix issues 1–3 above
2. ✅ Rerun **Variants C_fixed and D_fixed only** (100 epochs)
3. ✅ Compare to CNN v2 baseline (2.348 mm/s)
4. 📊 Assess whether event-level multi-output approach is viable
5. 🔄 Decide: proceed to curve-head or pivot to row-wise+physics

---

**Report prepared:** 2026-06-27  
**Status:** Ready for implementation once issues 2–3 are fixed
