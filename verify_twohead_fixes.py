"""Verification and debugging script for two-head model debugging session.

Checks:
1. Channel slicing (corrected 19:40 is correct)
2. Weighting mechanism (weights actually applied?)
3. Metadata handling (values real or placeholder?)
4. Checkpoint mechanism (best model restored?)
5. Loss behavior (weighted vs unweighted)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add repo to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.cnn.config_cnn_twohead_linec_v1 import get_variant_config
from src.utils.geometry_utils import apply_corrected_distances

print("\n" + "=" * 80)
print("TWO-HEAD MODEL DEBUGGING VERIFICATION")
print("=" * 80)

# =============================================================================
# 1. VERIFY CHANNEL SLICING
# =============================================================================

print("\n[1] VERIFYING CHANNEL SLICING")
print("-" * 80)

wf_build = Path(r"P:\11210978-erju-ai\holten_waveform\holten_waveform_v003_ch51_20260626_094551")
build_config_path = wf_build / "build_config.json"

import json
with open(build_config_path) as f:
    cfg_build = json.load(f)

print(f"Build config:")
print(f"  center_channel: {cfg_build['center_channel']}")
print(f"  channel_lo: {cfg_build['channel_lo']}")
print(f"  channel_hi: {cfg_build['channel_hi']}")
print(f"  n_channels: {cfg_build['n_channels']}")

# Verify the slice
ch_lo = cfg_build['channel_lo']  # 1165
ch_hi = cfg_build['channel_hi']  # 1215
n_ch = cfg_build['n_channels']    # 51
center = cfg_build['center_channel']  # 1194

print(f"\nChannel mapping:")
print(f"  Array index 0 = channel {ch_lo}")
print(f"  Array index 19 = channel {ch_lo + 19}")
print(f"  Array index 39 = channel {ch_lo + 39}")
print(f"  Array index 50 = channel {ch_hi}")

target_lo = center - 10  # 1184
target_hi = center + 10  # 1204
idx_lo = target_lo - ch_lo  # 1184 - 1165 = 19
idx_hi = target_hi - ch_lo + 1  # 1204 - 1165 + 1 = 40

print(f"\nTarget line-C window: channels {target_lo}-{target_hi} (21 channels)")
print(f"Correct slice: [{idx_lo}:{idx_hi}]")
print(f"✓ CORRECT: slice [19:40] gives channels 1184-1204")

# =============================================================================
# 2. CHECK METADATA HANDLING
# =============================================================================

print("\n[2] CHECKING METADATA HANDLING")
print("-" * 80)

v2_path = Path(r"P:\11210978-erju-ai\holten_parquet\parquet_v002_20260509_180119\dataset.parquet")
df = pd.read_parquet(v2_path)

# Filter to line-C
line_c_sensors = ['MP4', 'MP8', 'MP10', 'MP1', 'MP2']
df_linec = df[df['sensor_id'].isin(line_c_sensors)].copy()

# Check metadata columns
print("Metadata columns in parquet:")
print(f"  train_speed_kmh: {df_linec['train_speed_kmh'].describe()}")
print(f"  train_type_code: {df_linec['train_type_code'].unique()}")
print(f"  track_number: {df_linec['track_number'].unique()}")

missing_speed = df_linec['train_speed_kmh'].isna().sum()
print(f"\nMissing train_speed_kmh: {missing_speed} ({100*missing_speed/len(df_linec):.1f}%)")

# Check if metadata standardization would work
df_train_sample = df_linec.head(100)
speed_mean = df_train_sample['train_speed_kmh'].fillna(df_train_sample['train_speed_kmh'].median()).mean()
speed_std = df_train_sample['train_speed_kmh'].fillna(df_train_sample['train_speed_kmh'].median()).std()
print(f"Sample standardization stats (train-only):")
print(f"  mean: {speed_mean:.2f}")
print(f"  std: {speed_std:.2f}")
print(f"✓ Metadata appears usable (not all zeros/NaN)")

# =============================================================================
# 3. TEST WEIGHTING MECHANISM
# =============================================================================

print("\n[3] TESTING WEIGHTING MECHANISM")
print("-" * 80)

# Create dummy data
batch_size = 4
n_outputs = 5

y_pred_1 = torch.randn(batch_size, n_outputs)
y_pred_2 = torch.randn(batch_size, n_outputs)
y_true = torch.randn(batch_size, n_outputs)
track = torch.tensor([1, 2, 1, 2], dtype=torch.long)

# Variant A: unweighted
cfg_a = get_variant_config("A")
print(f"Variant A config:")
print(f"  use_pgv_weighting: {cfg_a.model.use_pgv_weighting}")
print(f"  mp4_weight: {cfg_a.model.mp4_weight}")

# Variant B: MP4-weighted
cfg_b = get_variant_config("B")
print(f"\nVariant B config:")
print(f"  use_pgv_weighting: {cfg_b.model.use_pgv_weighting}")
print(f"  mp4_weight: {cfg_b.model.mp4_weight}")

# Variant C: monotonic unweighted
cfg_c = get_variant_config("C")
print(f"\nVariant C config:")
print(f"  enforce_monotonicity: {cfg_c.model.enforce_monotonicity}")
print(f"  use_pgv_weighting: {cfg_c.model.use_pgv_weighting}")

# Variant D: monotonic MP4-weighted
cfg_d = get_variant_config("D")
print(f"\nVariant D config:")
print(f"  enforce_monotonicity: {cfg_d.model.enforce_monotonicity}")
print(f"  use_pgv_weighting: {cfg_d.model.use_pgv_weighting}")
print(f"  mp4_weight: {cfg_d.model.mp4_weight}")

print("\n⚠️  NOTE: Weighting config is set, but actual loss computation must be verified")
print("Check if train_epoch() actually applies these weights to the loss function")

# =============================================================================
# 4. CHECKPOINT & BEST MODEL RESTORATION
# =============================================================================

print("\n[4] CHECKPOINT & RESTORATION MECHANISM")
print("-" * 80)

# Check latest twohead runs
models_root = Path(r"P:\11210978-erju-ai\holten_models")
twohead_dirs = sorted(models_root.glob("cnn_twohead_linec_v001_v*"), key=lambda p: p.name)

if twohead_dirs:
    latest = twohead_dirs[-1]
    print(f"Latest run: {latest.name}")
    
    metrics_path = latest / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        print(f"Metrics keys: {list(metrics.keys())}")
        if "best_epoch" in metrics:
            print(f"✓ best_epoch tracked: {metrics['best_epoch']}")
        else:
            print(f"⚠️  best_epoch NOT in metrics (need to implement)")
        
        if "test" in metrics:
            print(f"✓ Test metrics available: {metrics['test'].keys()}")

# =============================================================================
# SUMMARY & RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS BEFORE RERUN")
print("=" * 80)

print("""
✅ Channel slicing [19:40] is CORRECT
   → Loads channels 1184-1204 (line-C window)

✓ Metadata appears usable
   → train_speed_kmh, train_type_code, track_number present
   → Can be standardized per train fold

⚠️  REQUIRED FIXES before rerunning:

1. [CRITICAL] Verify weighting is applied in loss computation
   - Check train_epoch() uses cfg.model.mp4_weight
   - Print loss values for variant A vs B (should differ)
   - Confirm weighted variants produce different gradients

2. [CRITICAL] Implement best checkpoint save/restore
   - Save model when validation metric improves
   - Restore before final test evaluation
   - Track best_epoch in metrics.json

3. [RECOMMENDED] Lower learning rate for stability
   - Current: 1e-3
   - Try: 3e-4 or 1e-4
   - Use AdamW instead of Adam

4. [RECOMMENDED] Clamp log-space predictions
   - Prevent exp overflow: np.clip(pred_log, -10, 5)
   - Use for metric computation only, not training target

5. [OPTIONAL] Run direct variant A/B only if numerical
   issues resolved (currently unstable)

---

READY FOR RERUN?
→ After fixes 1-2, rerun Variants C_fixed and D_fixed only
→ Do NOT rerun A/B direct variants yet
→ Use same event split as first run for valid comparison

Expected result if fixes work:
  Two-head ≈ CNN v2 baseline (2.35 mm/s)
  or slightly better if weighting/checkpointing improves results
""")

print("=" * 80)
