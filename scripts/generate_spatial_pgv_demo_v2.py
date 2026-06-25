"""Generate demo spatial PGV CSV v2 for QGIS visualisation.

Calibrated to actual training-data distribution from Holten:
  - Bulk of values: 0.5–3 mm/s at r=4m
  - Rare peaks up to ~15 mm/s
  - Attenuation 4m→12m: (4/12)^0.9336 ≈ 0.358

Infrastructure features (absolute channel numbers from fo_channels.csv):
  Road crossings : 1092,1711,2494,2766,7485,8670,9399,9928,11922,13690
  Cuvet          : 3340
  Lakes nearby   : 3729,4479,4968,8069
  Overpasses     : 4238,5162,5421,5923,6882,12349,14222,15462,15703,16964
  Bridges        : 7262,16096

Physics assumptions (documented for report):
  Road crossing  → abrupt track-stiffness transition + poor maintenance zone
                   → local vibration spike, σ~25m, +1.5–2.5 mm/s
  Cuvet          → reduced lateral soil support under track
                   → very local spike, σ~12m, +1 mm/s
  Lake           → soft/saturated soil, low shear-wave velocity
                   → site amplification (extended zone σ~90m), +0.8–1.5 mm/s
  Overpass       → modified foundation soil at structure edges;
                   right underneath: slight decrease (pile foundations stiffen soil)
                   → σ~60m compound shape, net moderate increase ~+1 mm/s
  Bridge         → mid-span: vibration stays in structure, less into ground (−0.4 mm/s)
                   transition ends: impact loading spikes (+1.5 mm/s, σ~12m each)

Output: docs/pgv_spatial_demo_v2.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Load channel positions
# ---------------------------------------------------------------------------
src = Path(__file__).parent.parent / "docs" / "fo_channels.csv"
df = pd.read_csv(src)
N = len(df)

# Use absolute channel number as position (1 channel = 1 m along track)
ch = df["channel"].values.astype(float)

# ---------------------------------------------------------------------------
# Calibrated attenuation model: PGV(r) = PGV_ref * (r0 / r)^n
# ---------------------------------------------------------------------------
N_GLOBAL = 0.9336
R0 = 10.0


def attenuate(pgv_r0: np.ndarray, r: float) -> np.ndarray:
    return pgv_r0 * (R0 / r) ** N_GLOBAL


# ---------------------------------------------------------------------------
# Spatial PGV field (reference distance r0 = 10 m)
# Target distribution at r=4m (calibrated against Holten training data):
#   median ~1.8 mm/s,  P90 ~4 mm/s,  P99 ~8 mm/s,  max ~14 mm/s (rare hotspots)
#
# At r=10m the same train gives ~0.80 mm/s baseline (log ≈ −0.22).
# (10/4)^0.9336 = 2.25, so 0.80 × 2.25 = 1.8 mm/s at 4m — correct baseline.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(2024)

# 1. Slow baseline variation (track quality / geology over km-scale)
#    log_base mean = −0.30  →  e^(−0.30) = 0.74 mm/s at r=10m  →  1.67 mm/s at r=4m
#    Large amplitudes give the broad undulation seen in v1.
log_base = (
    -0.30  # calibrated baseline
    + 0.55 * np.sin(2 * np.pi * ch / 5000 + 1.1)
    + 0.40 * np.sin(2 * np.pi * ch / 2200 + 3.4)
    + 0.25 * np.sin(2 * np.pi * ch / 900 + 0.7)
    + 0.15 * np.sin(2 * np.pi * ch / 400 + 2.3)
)

# 2. Medium-scale correlated noise (track irregularities, σ_spatial ~ 60m)
raw = rng.standard_normal(N)
sigma_s = 60
half = int(4 * sigma_s)
kx = np.arange(-half, half + 1, dtype=float)
kernel = np.exp(-0.5 * (kx / sigma_s) ** 2)
kernel /= kernel.sum()
med_noise = np.convolve(raw, kernel, mode="same") * 0.40

# 3. Shorter-scale correlated noise (σ_spatial ~ 15m) — section-level roughness
raw2 = rng.standard_normal(N)
sigma_s2 = 15
half2 = int(4 * sigma_s2)
kx2 = np.arange(-half2, half2 + 1, dtype=float)
kernel2 = np.exp(-0.5 * (kx2 / sigma_s2) ** 2)
kernel2 /= kernel2.sum()
short_noise = np.convolve(raw2, kernel2, mode="same") * 0.20

# 4. Fine per-channel variability
fine = rng.standard_normal(N) * 0.08

log_pgv_ref = log_base + med_noise + short_noise + fine


# ---------------------------------------------------------------------------
# Helper: add a Gaussian bump in log-space
# ---------------------------------------------------------------------------
def log_bump(ch, center, sigma, log_amp):
    return log_amp * np.exp(-0.5 * ((ch - center) / sigma) ** 2)


def log_bump_pair(
    ch, center, half_span, sigma_edge, sigma_mid, log_amp_edge, log_amp_mid
):
    """Two edge spikes + mid dip — for bridges/overpasses."""
    left = log_bump(ch, center - half_span, sigma_edge, log_amp_edge)
    right = log_bump(ch, center + half_span, sigma_edge, log_amp_edge)
    mid = log_bump(ch, center, sigma_mid, log_amp_mid)
    return left + right + mid


# ---------------------------------------------------------------------------
# Road crossings — abrupt stiffness transition, often degraded ballast
# σ ~ 25m.  Most crossings: log_amp ~1.0 (×e^1.0=×2.7 → peak ~4.5 mm/s).
# Two worst crossings get log_amp ~1.8 (×e^1.8=×6.0 → peak ~10 mm/s).
# ---------------------------------------------------------------------------
crossing_amps = rng.uniform(0.9, 1.2, size=10)
crossing_amps[3] = 1.80  # ch 2766 — badly maintained
crossing_amps[8] = 1.95  # ch 11922 — worst crossing, rural road
for c, amp in zip(
    [1092, 1711, 2494, 2766, 7485, 8670, 9399, 9928, 11922, 13690], crossing_amps
):
    log_pgv_ref += log_bump(ch, c, sigma=25, log_amp=amp)

# ---------------------------------------------------------------------------
# Cuvet — reduced lateral support, very local
# σ ~ 12m, log_amp +0.55 (×1.7 → peak ~2.8 mm/s)
# ---------------------------------------------------------------------------
log_pgv_ref += log_bump(ch, 3340, sigma=12, log_amp=0.55)

# ---------------------------------------------------------------------------
# Lakes — soft/saturated soil → site amplification (low Vs, resonance)
# Extended zone σ ~ 90m, log_amp +0.45–0.60 (×1.6–1.8 → peak ~2.8–3.2 mm/s)
# ---------------------------------------------------------------------------
for c, amp in zip([3729, 4479, 4968, 8069], [0.50, 0.58, 0.45, 0.52]):
    log_pgv_ref += log_bump(ch, c, sigma=90, log_amp=amp)

# ---------------------------------------------------------------------------
# Overpasses — approach zones: stiffness transition at abutment
# Mid-span: slightly damped (structure absorbs energy)
# edge log_amp +0.65, mid log_amp −0.15
# ---------------------------------------------------------------------------
for c in [4238, 5162, 5421, 5923, 6882, 12349, 14222, 15462, 15703, 16964]:
    log_pgv_ref += log_bump_pair(
        ch,
        c,
        half_span=30,
        sigma_edge=20,
        sigma_mid=30,
        log_amp_edge=0.65,
        log_amp_mid=-0.15,
    )

# ---------------------------------------------------------------------------
# Bridges — stronger abutment impact spikes + clear mid-span dip
# edge log_amp +1.30 (×e^1.3=×3.7 → peak ~6 mm/s), mid −0.35
# One bridge (16096) is longer: half_span=60m
# ---------------------------------------------------------------------------
log_pgv_ref += log_bump_pair(
    ch,
    7262,
    half_span=40,
    sigma_edge=12,
    sigma_mid=45,
    log_amp_edge=1.30,
    log_amp_mid=-0.35,
)
log_pgv_ref += log_bump_pair(
    ch,
    16096,
    half_span=60,
    sigma_edge=15,
    sigma_mid=55,
    log_amp_edge=1.45,
    log_amp_mid=-0.35,
)

# ---------------------------------------------------------------------------
# Convert log-field to linear PGV at r=10m, then scale to 4m and 12m
# ---------------------------------------------------------------------------
pgv_ref = np.exp(log_pgv_ref)

df["pgv_4m_mms"] = np.round(attenuate(pgv_ref, r=4.0), 3)
df["pgv_12m_mms"] = np.round(attenuate(pgv_ref, r=12.0), 3)

# ---------------------------------------------------------------------------
# Sanity check against real-data distribution
# ---------------------------------------------------------------------------
p = df["pgv_4m_mms"]
print("=== pgv_4m_mms distribution ===")
print(f"  min    : {p.min():.2f} mm/s")
print(f"  P10    : {p.quantile(0.10):.2f} mm/s")
print(f"  median : {p.median():.2f} mm/s")
print(f"  P90    : {p.quantile(0.90):.2f} mm/s")
print(f"  P99    : {p.quantile(0.99):.2f} mm/s")
print(f"  max    : {p.max():.2f} mm/s")
print(f"  > 6 mm/s : {(p > 6).sum()} channels ({100*(p>6).mean():.1f}%)")
print(f"  > 10 mm/s: {(p > 10).sum()} channels ({100*(p>10).mean():.1f}%)")

p2 = df["pgv_12m_mms"]
print("\n=== pgv_12m_mms distribution ===")
print(
    f"  median : {p2.median():.2f} mm/s  (ratio 4m/12m = {p.median()/p2.median():.2f})"
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = Path(__file__).parent.parent / "docs" / "pgv_spatial_demo_v2.csv"
df[["channel", "longitude", "latitude", "pgv_4m_mms", "pgv_12m_mms"]].to_csv(
    out, index=False
)
print(f"\nWritten {len(df):,} rows → {out}")
