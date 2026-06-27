# Curve-Prior Attenuation Exponent Fitting Audit
**Date:** 2026-06-27  
**Status:** Critical Issue Identified & Fixed  
**Impact:** First training run used wrong fitting method, producing 77-78% underestimated exponents

---

## Executive Summary

The initial curve-prior model (P1/P2/P3 run) used an **incorrect attenuation fitting method** that produced dramatically wrong n values:

| Method | Track 1 n | Track 2 n | Error vs Oracle |
|--------|-----------|-----------|-----------------|
| **Oracle Reference** | 1.0777 | 1.3300 | — |
| **Corrected (Event-Intercept)** | 1.0655 | 1.3246 | **-1.1%, -0.4%** ✓ |
| **Current/Old (Consecutive Differences)** | 0.2402 | 0.5147 | **-77.7%, -61.3%** ✗ |

This explains why the initial P1 (curve-only) performed poorly (RMSE log: 0.847) despite correct channel slicing and dataset construction. The physics model itself was invalid.

---

## Problem: Consecutive Difference Method

### What Was Done (Wrong)
The old fitting method computed per-sensor slope estimates between consecutive sensors:

```
For each event i and sensor pair (j, j+1):
  dy = log_pgv[i,j+1] - log_pgv[i,j]
  dr = log(r[j+1] / r[j])
  n_estimate = dy / dr
  
Then: n_track = median(all n_estimates)
```

### Why This Is Wrong

The physics model is:
$$y_{ij} = c_i - n_{\text{track}} \log(r_j / r_0)$$

The consecutive-difference method implicitly assumes independent point estimates between adjacent sensors, which:
1. **Ignores event intercepts** — treats each pair in isolation
2. **Amplifies noise** — differences at short baseline (MP4 to MP8: 1.5 m) are more noisy
3. **Doesn't use full information** — discards information about absolute distances
4. **Produces biased estimates** — consecutive slopes are not equivalent to global power-law fit

**Result:** For line-C data with realistic noise, this method produces n ≈ 0.24-0.51 instead of the correct ≈ 1.08-1.33.

---

## Solution: Event-Intercept Corrected Fitting

### Method
For each track, fit global n using event-demeaned least squares:

```
1. Center design matrix within each event:
   x = log(r / r0)                    # (n_events, 5)
   x_c = x - x.mean(axis=1)          # centered by event

2. Center targets within each event:
   y = log(PGV)                       # (n_events, 5)
   y_c = y - y.mean(axis=1)          # centered by event

3. Global least squares:
   y_c = -n * x_c
   n = -sum(x_c * y_c) / sum(x_c^2)
```

### Why This Works

- **Preserves physical model:** y = c - n*x, where c_i varies per event
- **Uses all information:** Leverages global power-law structure across all events
- **Robust to noise:** Centering removes within-event systematic offsets
- **Equivalent to standard regression:** When you fit y ~ c + x with fixed n per track

### Results

```
Event counts (train split):
  Track 1: 545 complete events, 2725 sensor readings
  Track 2: 558 complete events, 2790 sensor readings

Fitted n (corrected method):
  Track 1: 1.0655  (vs oracle 1.0777, error: -1.1%)
  Track 2: 1.3246  (vs oracle 1.3300, error: -0.4%)
```

**Accuracy:** Within 1.1% of oracle — acceptable for a refit on different subset.

---

## Example Event Profiles

### Track 1, Event 2 (Strong Signal)
```
Sensor  | r(m)  | log(r/r0) | log-PGV | PGV(mm/s)
--------|-------|-----------|---------|----------
MP4     |   2.5 |   -1.3863 |  2.2709 |   9.6884  (close sensor, high PGV)
MP8     |   4.0 |   -0.9163 |  1.4103 |   4.0974
MP10    |   8.0 |   -0.2231 |  0.3644 |   1.4397
MP1     |  16.0 |    0.4700 |  0.1076 |   1.1136
MP2     |  23.0 |    0.8329 | -0.5201 |   0.5944  (far sensor, low PGV)
--------|-------|-----------|---------|----------
Per-event n_i:      1.1719
Intensity c_i:      0.4400

Predicted curve with n=1.0777:
y_pred = 0.4400 - 1.0777 * log(r/10)
```

This example shows:
- MP4 (close): strong signal, dominates loss if unweighted
- MP2 (far): weak signal, but constrains global n estimate
- Per-event n_i (1.1719) is reasonable, close to global n

---

## Decision: Which Method to Use

### For Production Training
**Use: `n_mode="fit_corrected"`**

- Fit n_track1, n_track2 on training split using corrected method
- Ensures physics model is valid
- Allows comparison: if P1 improves significantly with correct n, confirms curve-prior is viable

### For Diagnostic Comparison
**Use: `n_mode="fixed_oracle"`**

- Use oracle reference values (1.0777, 1.3300)
- Isolate effect of n value from fitting noise
- Answer: "How well does the curve-prior model work with perfectly fitted n?"

### NOT for Training
**Skip: `n_mode="fit_current_old"`**

- Consecutive-difference method is invalid
- Produces impossible n values
- Kept only for documentation/reproducibility

---

## Expected Impact on Results

### P1 (Curve-Only)
**Before (wrong n):** RMSE(log) = 0.847, R² = 0.322
- Fitted n ≈ 0.24, 0.51 makes physics curve meaningless
- Model defaults to learning residuals, essentially ignoring physics

**After (correct n):**
- If curve-prior is valid: RMSE(log) should **improve significantly** (possibly < 0.62)
- If FO cannot predict event intensity: RMSE(log) may still be > 0.62
- Either way, will be scientifically interpretable

### P2/P3 (With Residuals)
**Before:** RMSE(log) ≈ 0.636-0.646
- Residuals compensated for wrong n; model learned effective attenuation
- Performance accidentally decent due to overfitting

**After (correct n):**
- If residuals still help: small improvement or unchanged
- If residuals were just fixing wrong n: may degrade slightly
- Should see clearer separation between P1 (curve only) and P2/P3 (with residuals)

### Decision Point
**If P1_corrected_n improves by >0.1 RMSE(log):**
- Confirms curve-prior with correct physics is viable
- Proceed to full curve-prior development

**If P1_corrected_n still poor but P2/P3 help:**
- FO waveform not sufficient to predict intensity alone
- Residual model addresses this limitation
- Continue curve-prior development with residuals

**If P3_corrected_n matches/beats row-wise CNN:**
- Curve-prior is competitive
- Move to next phase: arbitrary-distance generalization

---

## Methodology: Key Learning

### Principle
> **Global attenuation exponents must be fitted with event-level intercepts included, never from adjacent-sensor differences.**

When fitting power-law models (y = c - n*log(r/r0)) on multi-event data:
1. Each event has its own intercept c_i (event intensity)
2. Differences between consecutive sensors conflate n with variations in c_i
3. Event-demeaning removes the nuisance intercepts and reveals true n
4. This is standard methodology in seismic/engineering (distance-dependent attenuation fitting)

### Verification
On line-C data with these distance vectors, the corrected method recovers oracle values within 1%, confirming the methodology is correct.

---

## Files & Settings for Next Run

### Configuration
```
n_mode = "fit_corrected"           # Use corrected fitting method
n_mode = "fixed_oracle"            # Use oracle reference n

Variants to run:
  P1_corrected_n:   curve-only
  P2_corrected_n:   curve + residual (lambda=0.05)
  P3_corrected_n:   curve + residual (lambda=0.05) + MP4 weight (2.0)
  
  P1_fixed_oracle_n:   curve-only + fixed oracle n
  P2_fixed_oracle_n:   curve + residual + fixed oracle n
  P3_fixed_oracle_n:   curve + residual + MP4 weight + fixed oracle n
  
  P3_corrected_n_lambda02:   P3 with stronger regularization (lambda=0.20)
  P3_fixed_oracle_n_lambda02:   P3 + fixed oracle + stronger regularization
```

### Training Script
```bash
python train_cnn_curveprior_linec_v1.py \
  --variant P1 \
  --n_mode fit_corrected \
  --lambda_residual 0.0 \
  --epochs 100
```

---

## Baselines to Compare Against

| Model | RMSE(log) | R²(log) | RMSE(PGV) |
|-------|-----------|---------|-----------|
| Row-wise CNN v2 (baseline) | 0.595 | 0.654 | 2.348 |
| Two-head D_fixed | 0.613 | 0.645 | 2.620 |
| P1/P2/P3 (broken n) | 0.636-0.847 | 0.322-0.617 | 2.752-3.266 |
| **P1/P2/P3 (corrected n) — expected** | ? | ? | ? |

---

## References

Seismic attenuation curve fitting methodology:
- Chapman et al. (2014) on distance-dependent ground motion prediction
- Standard practice: fit log-amplitude vs log-distance with event intercepts
- Event-demeaning ensures global power-law is identifiable separate from event magnitude variation

---

## Sign-Off

**Issue:** Consecutive-difference method produces wrong exponents (-77% error)  
**Root Cause:** Method conflates adjacent-sensor differences with global power-law; ignores event intercepts  
**Fix:** Event-intercept corrected method recovers oracle values within 1%  
**Validation:** Audit script confirms distance vectors, event counts, fitting formula  
**Status:** Ready for corrected/fixed-n training run  

**Next:** Run P1/P2/P3 with `n_mode=fit_corrected` and `n_mode=fixed_oracle` on cluster.

---

*Audit performed:* 2026-06-27  
*Audit method:* Comparative fitting analysis with diagnostics  
*Result:* Critical issue identified, fix validated, ready to proceed
