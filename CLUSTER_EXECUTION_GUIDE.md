# Track-Conditioned Two-Head Multi-Output CNN — Cluster Execution Guide

## Quick Start

### 1. Copy Files to Cluster

```bash
# From your local machine:
scp train_cnn_twohead_linec_v1.py camposmo@cluster:/u/camposmo/erju/
scp src/ml/cnn/config_cnn_twohead_linec_v1.py camposmo@cluster:/u/camposmo/erju/src/ml/cnn/
scp src/ml/cnn/cnn_twohead_linec_utils.py camposmo@cluster:/u/camposmo/erju/src/ml/cnn/
scp cluster_run_twohead_v1_all_variants.sh camposmo@cluster:/u/camposmo/erju/
```

### 2. Make Script Executable

```bash
# On cluster:
chmod +x /u/camposmo/erju/cluster_run_twohead_v1_all_variants.sh
```

### 3. Submit Job

```bash
cd /u/camposmo/erju
sbatch cluster_run_twohead_v1_all_variants.sh
```

### 4. Monitor Progress

```bash
# Check job status
squeue -u camposmo

# Check output in real time
tail -f /u/camposmo/erju/erju_twohead_<JOBID>.log

# Or check variant-specific logs
tail -f /u/camposmo/erju/outputs/twohead_variants/variant_A.log
```

---

## What This Script Does

### Resource Allocation
- **GPU:** 1 × NVIDIA H100 (or available GPU)
- **CPU:** 4 cores
- **Memory:** 32 GB
- **Time:** 12 hours (should be sufficient for 100 epochs × 4 variants)
- **Partition:** GPU

### Variants Executed (Sequential)

| Variant | Configuration | Purpose |
|---------|---|---|
| **A** | Direct 2-head, unweighted MSE | Baseline direct prediction |
| **B** | Direct 2-head, MP4-weighted (2×) | Focus on close sensors / high-PGV |
| **C** | Monotonic 2-head (parameterization), unweighted | Enforce physical attenuation monotonicity |
| **D** | Monotonic 2-head (parameterization), MP4-weighted | Combine monotonicity + close-sensor weighting |

Each variant uses 100 epochs of training with early stopping and learning rate scheduling.

### Outputs per Variant

For each variant, the script creates:

```
holten_models/cnn_twohead_linec_v001_<timestamp>/
├── config_snapshot.json              # Full config (variant settings)
├── metrics.json                       # Combined metrics (RMSE, MAE, R², etc.)
├── per_track_metrics.csv              # Track 1 / Track 2 breakdown
├── per_sensor_metrics.csv             # MP4 / MP8 / MP10 / MP1 / MP2 breakdown
├── high_pgv_diagnostics.csv           # PGV > 4 mm/s analysis
├── monotonicity_diagnostics.csv       # Profile monotonicity stats
├── predictions.parquet                # Full predictions (event_id, track, y_true, y_pred)
├── learning_curve.png
├── measured_vs_predicted_loglog.png
├── residuals_vs_pgv.png
├── residuals_vs_distance.png
├── per_sensor_residuals.png
├── track1_profiles_examples.png
├── track2_profiles_examples.png
├── per_track_scatter.png
└── monotonicity_diagnostics.png
```

---

## Examining Results

### Immediately After Job Completes

```bash
# Check for errors
cat /u/camposmo/erju/erju_twohead_<JOBID>.err

# View summary
tail -100 /u/camposmo/erju/erju_twohead_<JOBID>.log

# List variant output directories
ls -lh /u/camposmo/erju/outputs/twohead_variants/
ls -lh /p/11210978-erju-ai/holten_models/ | grep cnn_twohead
```

### Key Metrics to Compare

Look at `metrics.json` for each variant:

```json
{
  "variant": "A",
  "subset": "line_c_side_minus1",
  "combined": {
    "rmse_pgv": 2.1,
    "mae_pgv": 1.2,
    "rmse_log": 0.35,
    "mae_log": 0.28,
    "r2_log": 0.88
  },
  "track_1": { "rmse_pgv": ..., "mae_pgv": ..., ... },
  "track_2": { "rmse_pgv": ..., "mae_pgv": ..., ... },
  "per_sensor": {
    "MP4": { "rmse": ..., "bias": ..., "n_samples": ... },
    ...
  }
}
```

### Generate Comparison Table

After all 4 variants complete, run this locally:

```python
import json
import pandas as pd
from pathlib import Path

output_root = Path("/p/11210978-erju-ai/holten_models")
results = []

for variant in ["A", "B", "C", "D"]:
    # Find the latest run for this variant
    dirs = sorted(output_root.glob(f"cnn_twohead_linec_v001_*"), key=lambda p: p.name)
    # (Assumes you've tagged or labelled them by variant)
    # For now, manually check each:
    metrics_path = output_root / f"variant_{variant}_metrics.json"  # adjust path
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        results.append({
            "Variant": variant,
            "Test RMSE (mm/s)": metrics["combined"]["rmse_pgv"],
            "Test MAE (mm/s)": metrics["combined"]["mae_pgv"],
            "Test R² (log)": metrics["combined"]["r2_log"],
            "MP4 bias (mm/s)": metrics["per_sensor"]["MP4"]["bias"],
            "MP4 RMSE (mm/s)": metrics["per_sensor"]["MP4"]["rmse"],
            "Track 1 RMSE": metrics["track_1"]["rmse_pgv"],
            "Track 2 RMSE": metrics["track_2"]["rmse_pgv"],
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

---

## Acceptance Criteria (From Brief)

The model is promising if:

1. ✅ **Beats scalar-only** on line-C subset
2. ✅ **Beats row-wise 21-channel CNN** on line-C subset
3. ✅ **Reduces MP4 & high-PGV underprediction** (compare to CNN v2 diagnostics)
4. ✅ **Improves/stabilizes track 1 close-sensor** predictions
5. ✅ **Mostly monotonic profiles** (Variants C & D)
6. ✅ **Sensible per-track performance** (Track 1 ≠ Track 2)

Check these against:
- Previous CNN v2 21-ch baseline: RMSE ≈ 1.797 mm/s (full dataset), MP4 RMSE ≈ 4.94
- Track-separated oracle results (from distance audit):
  - Track 1 oracle: RMSE = 1.588 mm/s
  - Track 2 oracle: RMSE = 1.254 mm/s

---

## Troubleshooting

### Job Fails Immediately

**Check 1:** Python environment
```bash
conda activate erju-torch
python -c "import torch; print(torch.__version__)"
python -c "from src.ml.cnn.config_cnn_twohead_linec_v1 import Config"
```

**Check 2:** File paths (Linux vs Windows)
- Config should auto-detect `/p/11210978-erju-ai/` on Linux
- Waveform builds: `/p/11210978-erju-ai/holten_waveform/`
- Parquet data: `/p/11210978-erju-ai/holten_parquet/`

**Check 3:** GPU memory
```bash
nvidia-smi
# H100 has 81 GB, should be plenty for batch_size=16
```

### Job Times Out

- Increase `--time=12:00:00` in SLURM header if needed
- Or reduce `--epochs 100` to `--epochs 50` in the script

### CUDA Out of Memory

- Reduce `batch_size` in config (currently 16)
- Set to 8: `cfg.train.batch_size = 8` in training script

### Waveform Build Not Found

Check available builds:
```bash
ls -lh /p/11210978-erju-ai/holten_waveform/
```

If only ch51 exists, the script will auto-slice to 21 channels (line C local window).

---

## Expected Timing

**Per variant (100 epochs):**
- Data loading: ~30s
- Training: ~15–20 min (H100 GPU)
- Evaluation & plots: ~2–3 min
- Total per variant: ~20–25 min

**All 4 variants:** ~80–100 minutes (~1.5–2 hours)

With GPU overhead and I/O, total job should complete within 4 hours easily (SLURM allocated 12 hours).

---

## Post-Job Steps

### 1. Copy Outputs Back

```bash
# Download results from cluster
scp -r camposmo@cluster:/p/11210978-erju-ai/holten_models/cnn_twohead_linec_v001_* \
    P:/11210978-erju-ai/holten_models/

# Or sync specific variant logs
scp -r camposmo@cluster:/u/camposmo/erju/outputs/twohead_variants/ \
    ~/erju_logs/
```

### 2. Comparison Analysis

Run the variant comparison notebook to assess:
- Which variant performs best overall?
- Which reduces MP4 underprediction?
- Which produces most monotonic profiles?
- Are track 1 and track 2 metrics sensible and separate?

### 3. Decision Point

Based on acceptance criteria, decide next step:
- **If promising:** Proceed to track-conditioned **arbitrary-distance curve-head model**
- **If not promising:** Debug (e.g., loss function, weighting, monotonicity constraint)
- **If marginal:** Run extended ablations (e.g., different LR, head architectures)

---

## Files Provided

| File | Purpose |
|------|---------|
| `train_cnn_twohead_linec_v1.py` | Main training script (event-level, track-conditioned) |
| `src/ml/cnn/config_cnn_twohead_linec_v1.py` | Config with 4 variant presets (A, B, C, D) |
| `src/ml/cnn/cnn_twohead_linec_utils.py` | Model architecture & loss functions |
| `cluster_run_twohead_v1_all_variants.sh` | SLURM job script (runs all 4 variants sequentially) |
| `CLUSTER_EXECUTION_GUIDE.md` | This file |

---

## Contact & Questions

If job fails or metrics are unexpected, check:
1. `/u/camposmo/erju/erju_twohead_<JOBID>.err` (error log)
2. `/u/camposmo/erju/erju_twohead_<JOBID>.log` (output log)
3. Variant-specific logs: `/u/camposmo/erju/outputs/twohead_variants/variant_*.log`

Email: fabian.campos@deltares.nl (job completion notification)

---

**Good luck with the cluster run! 🚀**
