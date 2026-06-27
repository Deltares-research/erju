# Query-Conditioned Curve-Prior Implementation Checklist
**Date:** 2026-06-27  
**Phase:** Pre-Flight Validation  
**Status:** READY FOR LOCAL SMOKE CHECKS

---

## Implementation Complete ✓

### Config & Architecture
- [x] `src/ml/cnn/config_cnn_curvequery_linec_v1.py`
  - ✓ Q1/Q2/Q3/Q4 variant definitions
  - ✓ Query features config
  - ✓ Held-out sensor support
  - ✓ Corrected n-fitting integration

- [x] `src/ml/cnn/cnn_curvequery_linec_utils.py`
  - ✓ `CurvePriorCNN2D_Query` model class
  - ✓ Query feature embedding
  - ✓ Intensity + residual heads
  - ✓ `CurveDataset_Query` with holdout filtering
  - ✓ Loss functions (Huber, residual regularization)
  - ✓ Attenuation fitting (corrected method)

### Training Script
- [x] `train_cnn_curvequery_linec_v1.py`
  - ✓ Data loading (line-C, side -1)
  - ✓ Event-level dataset building
  - ✓ n-fitting on train split
  - ✓ Model instantiation
  - ✓ Training loop with early stopping
  - ✓ Evaluation and checkpoint save
  - ✓ Config/metrics JSON output
  - ✓ Variant routing (Q1-Q4)
  - ✓ Holdout sensor support

### SLURM & Diagnostics
- [x] `slurm/run_curvequery_linec_v1.slurm`
  - ✓ Job configuration (GPU, memory, time)
  - ✓ Environment setup
  - ✓ 10-run grid (Q1-Q4 all-sensor + 6 held-out)
  - ✓ Logging per variant
  - ✓ Summary output

- [x] `extract_curveprior_diagnostics.py`
  - ✓ Load P3_corrected_n predictions
  - ✓ Per-sensor metrics extraction
  - ✓ Residual analysis
  - ✓ Track-wise breakdown
  - ✓ Markdown report generation

- [x] `smoke_test_curvequery_linec.py`
  - ✓ Config loading (Q1-Q4)
  - ✓ Model instantiation
  - ✓ Forward/backward pass
  - ✓ Loss computation
  - ✓ Dataset class validation
  - ✓ Held-out filtering
  - ✓ Batch loading
  - ✓ Integration tests

### Documentation
- [x] `CURVEQUERY_MODEL_DESIGN_20260627.md`
  - ✓ Architecture specification
  - ✓ Variant definitions
  - ✓ Held-out test protocol
  - ✓ Decision rules
  - ✓ Baseline comparisons
  - ✓ Analysis plan

---

## Local Validation Checklist (USER TO RUN)

### Step 1: P3 Diagnostics Extraction
```bash
# Extract baseline metrics from P3_corrected_n
python extract_curveprior_diagnostics.py \
    --model_dir /p/11210978-erju-ai/holten_models/cnn_curveprior_linec_v001_vP3_fit_corrected_20260627_211829 \
    --output_report CURVEPRIOR_P3_DIAGNOSTICS_20260627.md
```

**Expected output:** Markdown report with:
- Per-sensor RMSE(log) and bias
- MP4 breakdown
- PGV > 4 mm/s subset analysis
- Track-wise metrics
- Fitted n values (1.0655, 1.3246)

### Step 2: Smoke Tests
```bash
# Run pre-flight validation
python smoke_test_curvequery_linec.py
```

**Expected output:** All 9 tests pass:
1. Configuration loading ✓
2. Model instantiation ✓
3. Forward pass ✓
4. Loss computation ✓
5. Backward pass ✓
6. Dataset class ✓
7. Held-out filtering ✓
8. Batch loading ✓
9. Q1 vs Q3 behavior ✓

### Step 3: Review Scripts
- [ ] Check `train_cnn_curvequery_linec_v1.py` for any issues
- [ ] Verify config paths in SLURM script
- [ ] Confirm SLURM resource allocation

---

## SLURM Submission (After Validation)

### Command
```bash
sbatch slurm/run_curvequery_linec_v1.slurm
```

### Monitor
```bash
# Check job
squeue -u camposmo

# Tail log
tail -f outputs/curvequery_linec_v1/Q1_all.log

# Check GPU
watch nvidia-smi
```

### Expected Runtime
- All 10 variants: ~40-50 min total
- Per variant: 3-5 min (depends on early stopping)

### Output Location
```
/p/11210978-erju-ai/holten_models/
  cnn_curvequery_linec_v001_vQ1_all_*
  cnn_curvequery_linec_v001_vQ2_all_*
  cnn_curvequery_linec_v001_vQ3_all_*
  cnn_curvequery_linec_v001_vQ4_all_*
  cnn_curvequery_linec_v001_vQ1_MP4_*
  cnn_curvequery_linec_v001_vQ3_MP4_*
  cnn_curvequery_linec_v001_vQ1_MP8_*
  cnn_curvequery_linec_v001_vQ3_MP8_*
  cnn_curvequery_linec_v001_vQ1_MP2_*
  cnn_curvequery_linec_v001_vQ3_MP2_*
```

---

## Post-Run Analysis (After Cluster Results)

### Step 1: Gather Results
```python
import json
from pathlib import Path

model_dir = Path("/p/11210978-erju-ai/holten_models")
results = {}

for variant in ["Q1_all", "Q2_all", "Q3_all", "Q4_all"]:
    pattern = f"cnn_curvequery_linec_v001_v{variant}_*"
    dirs = list(model_dir.glob(pattern))
    if dirs:
        metrics_file = dirs[0] / "metrics.json"
        with open(metrics_file) as f:
            results[variant] = json.load(f)

# Display
for variant, metrics in results.items():
    print(f"{variant}: RMSE(log) = {metrics['test_rmse_log']:.4f}")
```

### Step 2: Compare Against Baselines
```
Expected:
  Q1_all: RMSE(log) ≈ 0.6146 (similar to P1_corrected_n)
  Q3_all: RMSE(log) ≈ 0.5993 (similar to P3_corrected_n)

If significantly worse:
  → Check for data loading or model bugs
  → Verify n-fitting on this run
  → Debug gradient flow
```

### Step 3: Analyze Held-Out Sensors
```python
# For each holdout variant, compute:
# - RMSE(log) on held-out sensor samples only
# - Comparison Q1_holdout vs Q3_holdout
# - Benchmark vs physics-only baseline

# If Q3_holdout is close to Q3_all → generalization works!
# If Q3_holdout >> 0.65 → distance-dependent residuals not transferable
```

### Step 4: Make Decision
- **If Q3_all ≈ 0.60 AND Q3_holdout < 0.63:** Continue with full cross-validation
- **If Q3_all > 0.62 OR Q3_holdout > 0.70:** Keep P3_corrected_n, explore other directions
- **If Q1_holdout stays < 0.62:** Pure physics curve is generalizable (big win!)

---

## Known Limitations (Phase 1)

### Not Tested Yet
- Cross-line generalization (other lines, other sides)
- Very far distances (> 30 m)
- Arbitrary geometry (non-track-aligned distances)
- Time-dependent attenuation

### In Scope for Later
- End-to-end n learning (infer n from waveforms)
- Physics-informed regularization (monotonicity constraints)
- Arbitrary distance regression (continuous r prediction)
- Cross-line transfer learning

---

## File Inventory

### Core Files
```
train_cnn_curvequery_linec_v1.py         ← Main training script
src/ml/cnn/config_cnn_curvequery_linec_v1.py    ← Configs
src/ml/cnn/cnn_curvequery_linec_utils.py        ← Model architecture
slurm/run_curvequery_linec_v1.slurm     ← SLURM job
smoke_test_curvequery_linec.py           ← Pre-flight checks
extract_curveprior_diagnostics.py        ← P3 baseline extraction
```

### Documentation
```
CURVEQUERY_MODEL_DESIGN_20260627.md      ← Design document
CURVEPRIOR_P3_DIAGNOSTICS_20260627.md   ← To be generated
QUERY_IMPLEMENTATION_CHECKLIST_20260627.md ← This file
```

---

## Estimated Schedule

| Phase | Task | Duration | Start |
|-------|------|----------|-------|
| **Phase 1** | Local validation | ~30 min | Now |
| **Phase 2** | SLURM submission | 1 min | After Phase 1 |
| **Phase 3** | Cluster run | 40-50 min | After Phase 2 |
| **Phase 4** | Analysis & decision | 1-2 hours | After Phase 3 |
| **Phase 5** | Report findings | 30 min | After Phase 4 |

**Total end-to-end:** ~3-4 hours

---

## Success Criteria

### Minimum (Go/No-Go)
- [x] All smoke tests pass
- [x] SLURM script runs without syntax errors
- [ ] All 10 variants complete on cluster
- [ ] All output files (model.pth, metrics.json) generated
- [ ] Q3_all RMSE(log) within 0.595-0.610 (between row-wise and P3)

### Desirable
- [ ] Q3_all RMSE(log) < 0.60 (beats row-wise or ties P3)
- [ ] Q1_holdout RMSE(log) < 0.63 (physics curve generalizes)
- [ ] Q3_holdout < Q1_holdout (residuals help generalization)
- [ ] All variants converge smoothly (no NaNs, no crashes)

### Stretch (Phase 2 readiness)
- [ ] Q3_holdout RMSE(log) < 0.61 (near all-sensor performance)
- [ ] MP4 holdout ~ MP2 holdout (distance independence)
- [ ] Per-variant metrics saved for publication

---

## Next Actions

### Immediate (Now)
1. Run P3 diagnostics extraction
2. Run smoke tests
3. Review outputs
4. Approve for SLURM submission

### Short-term (After SLURM)
1. Download results
2. Analyze all-sensor performance
3. Analyze held-out sensor generalization
4. Make go/no-go decision on arbitrary-distance development

### Long-term (Contingent on Success)
1. Cross-line validation
2. Arbitrary geometry generalization
3. Multi-track model
4. Full-site model

---

**Status:** ✓ READY FOR LOCAL VALIDATION  
**Next:** Execute smoke checks and P3 diagnostics extraction
