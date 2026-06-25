"""Generate a demo spatial PGV CSV for QGIS visualisation.

Produces realistic-looking synthetic predictions at r=4m and r=12m
for every FO channel, based on calibrated attenuation parameters from
the Holten measurement campaign (n_global=0.9336, r0=10m).

Hotspots are placed at physically plausible locations (switches, curves,
bridge approaches) relative to the Holten sensor crossing (ch 1194).

Output: docs/pgv_spatial_demo.csv
  channel, longitude, latitude, pgv_4m_mms, pgv_12m_mms
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

# Channel index → along-track distance in metres (1 channel = 1 m)
d = (df["channel"] - df["channel"].min()).values.astype(float)

# Holten crossing reference position
CROSSING_CHANNEL = 1194
crossing_d = float(
    df.loc[df["channel"] == CROSSING_CHANNEL, "channel"].iloc[0] - df["channel"].min()
)

# ---------------------------------------------------------------------------
# Calibrated attenuation model  log(PGV) = c - n * log(r / r0)
# ---------------------------------------------------------------------------
N_GLOBAL = 0.9336
R0 = 10.0


def attenuate(pgv_r0: np.ndarray, r: float) -> np.ndarray:
    """Scale PGV from r0=10m to target distance r using power law."""
    return pgv_r0 * (R0 / r) ** N_GLOBAL


# PGV at the source (r=10m reference) – this is what we simulate spatially.
# Typical range from training data: ~3–12 mm/s at r=10m for mixed traffic.

# ---------------------------------------------------------------------------
# Spatial PGV field generation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

# 1. Low-frequency baseline: slowly varying track quality (wavelength ~1–4 km)
baseline = (
    5.5
    + 1.2 * np.sin(2 * np.pi * d / 4200 + 0.8)
    + 0.9 * np.sin(2 * np.pi * d / 2100 + 2.1)
    + 0.6 * np.sin(2 * np.pi * d / 900 + 4.3)
)

# 2. Medium-scale correlated noise (track irregularities, ~50 m correlation)
#    Manual convolution with Gaussian kernel (no scipy needed)
raw = rng.normal(0, 1.0, N)
sigma_samples = 50
kernel_half = int(4 * sigma_samples)
kx = np.arange(-kernel_half, kernel_half + 1)
kernel = np.exp(-0.5 * (kx / sigma_samples) ** 2)
kernel /= kernel.sum()
medium_noise = np.convolve(raw, kernel, mode="same") * 1.2

# 3. Fine per-channel noise
fine_noise = rng.normal(0, 0.25, N)

pgv_ref = baseline + medium_noise + fine_noise


# ---------------------------------------------------------------------------
# Hotspots (Gaussian bumps representing track defects / features)
# ---------------------------------------------------------------------------
def hotspot(d: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((d - center) / width) ** 2)


# Switch after the crossing
pgv_ref += hotspot(d, crossing_d + 230, 35, 4.5)
# Bad rail joint ahead of crossing
pgv_ref += hotspot(d, crossing_d - 420, 20, 3.8)
# Bridge approach (long, gradual)
pgv_ref += hotspot(d, crossing_d + 1600, 90, 3.0)
# Curve with cant deficiency
pgv_ref += hotspot(d, crossing_d - 1900, 70, 2.5)
# Another switch further out
pgv_ref += hotspot(d, crossing_d + 3800, 30, 5.2)
# Soft soil section
pgv_ref += hotspot(d, crossing_d - 5500, 200, 2.0)

# Physical floor: no negative PGV
pgv_ref = np.clip(pgv_ref, 0.8, None)

# ---------------------------------------------------------------------------
# Scale to the two target distances
# ---------------------------------------------------------------------------
df["pgv_4m_mms"] = np.round(attenuate(pgv_ref, r=4.0), 3)
df["pgv_12m_mms"] = np.round(attenuate(pgv_ref, r=12.0), 3)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = Path(__file__).parent.parent / "docs" / "pgv_spatial_demo.csv"
df[["channel", "longitude", "latitude", "pgv_4m_mms", "pgv_12m_mms"]].to_csv(
    out, index=False
)

print(f"Written {len(df):,} rows → {out}")
print(
    f"\nPGV at 4m  — min: {df.pgv_4m_mms.min():.2f}  median: {df.pgv_4m_mms.median():.2f}  max: {df.pgv_4m_mms.max():.2f}  mm/s"
)
print(
    f"PGV at 12m — min: {df.pgv_12m_mms.min():.2f}  median: {df.pgv_12m_mms.median():.2f}  max: {df.pgv_12m_mms.max():.2f}  mm/s"
)
print(f"\nHotspot channels (pgv_4m > 15 mm/s):")
print(
    df[df.pgv_4m_mms > 15][
        ["channel", "longitude", "latitude", "pgv_4m_mms"]
    ].to_string(index=False)
)
