"""Moveout diagnostic for the multi-channel FO waveform dataset (v2).

Measures the train-passage moveout (the inclined wavefront) across the local
21-channel window and verifies it fits inside the 30 s crop.

Geometry
--------
FO channel pitch = 1 m (consecutive channel IDs are 1 m apart).
Gauge length = 10 m is the DAS spatial-averaging window, NOT the pitch.
The 21-channel window 1184-1204 therefore spans a 20 m aperture (+/-10 m
around channel 1194).

Method
------
For each event block (21, T):
  1. Smooth |x| per channel to an RMS envelope.
  2. Take the envelope-peak arrival time per channel.
  3. Linear fit  arrival_time = slope * position + c   (position in metres,
     centred on channel 1194).
  4. apparent_speed = 1 / |slope|  [m/s];  moveout_lag = aperture * |slope| [s].
  5. R^2 of the fit flags clean single-train passages.

Outputs (saved next to the waveform build, in plots/moveout/):
  - moveout_apparent_speed_hist.png
  - moveout_lag_hist.png            (with the 30 s window for reference)
  - moveout_speed_vs_recorded.png   (apparent vs metadata train_speed_kmh)
  - moveout_examples.png            (channel x time + fitted moveout line)
  - moveout_summary.json

Run:
    python scripts/moveout_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.db.waveform.config_waveform_v2 import CONFIG as WV2
from src.db.waveform.waveform_v1_utils import _moving_rms

WAVEFORM_ROOT = Path(WV2.output.output_root_folder)
FS_HZ = WV2.signal.target_fs_hz
CHANNEL_SPACING_M = WV2.signal.channel_spacing_m       # 1.0 m
N_CHANNELS = WV2.signal.n_channels                     # 21
APERTURE_M = WV2.signal.aperture_m                     # 20.0 m
WINDOW_S = WV2.window.fixed_length_s                   # 30 s
ENVELOPE_SMOOTH_S = WV2.window.envelope_smooth_s       # 0.5 s
MAX_LAG_S = 3.0                                        # search bound for moveout

# Channel positions in metres, centred on channel 1194 (index 10).
_center_pos = (WV2.signal.center_channel - WV2.signal.channel_lo)
POSITIONS_M = (np.arange(N_CHANNELS) - _center_pos) * CHANNEL_SPACING_M


def _find_latest_build() -> Path:
    builds = sorted(WAVEFORM_ROOT.glob("holten_waveform_v002_*"), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No holten_waveform_v002_* builds in {WAVEFORM_ROOT}")
    return builds[-1]


def _channel_lags_xcorr(
    block: np.ndarray, center_pos: int, fs: float, smooth_win: int, max_lag_samples: int
) -> np.ndarray:
    """Per-channel arrival lag (s) relative to the centre channel.

    Uses envelope cross-correlation, which resolves sub-burst moveout far
    better than argmax peak-picking when channels are spatially smoothed by the
    10 m gauge length.  Lag is positive when the channel's energy arrives later.
    """
    ref = _moving_rms(block[center_pos], smooth_win)
    ref = ref - ref.mean()
    n = ref.size
    lag_axis = np.arange(-(n - 1), n)
    keep = np.abs(lag_axis) <= max_lag_samples
    lag_axis_k = lag_axis[keep]

    lags = np.zeros(block.shape[0], dtype=np.float64)
    ref_norm = np.linalg.norm(ref) + 1e-12
    for c in range(block.shape[0]):
        sig = _moving_rms(block[c], smooth_win)
        sig = sig - sig.mean()
        if np.linalg.norm(sig) < 1e-9:
            lags[c] = np.nan
            continue
        corr = correlate(sig, ref, mode="full")[keep]
        lags[c] = lag_axis_k[int(np.argmax(corr))] / fs
    return lags


def _fit_moveout(positions_m: np.ndarray, lags_s: np.ndarray) -> Dict[str, float]:
    """Linear fit of arrival lag vs position. Returns slope, speed, lag, r2."""
    finite = np.isfinite(lags_s)
    if finite.sum() < 5:
        return {"slope_s_per_m": np.nan, "apparent_speed_kmh": np.nan,
                "moveout_lag_s": np.nan, "direction": np.nan, "r2": np.nan,
                "n_channels_used": int(finite.sum())}
    pos = positions_m[finite]
    lag = lags_s[finite]
    slope, intercept = np.polyfit(pos, lag, 1)         # s/m, s
    pred = slope * pos + intercept
    ss_res = float(np.sum((lag - pred) ** 2))
    ss_tot = float(np.sum((lag - lag.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    abs_slope = abs(slope)
    speed_mps = (1.0 / abs_slope) if abs_slope > 1e-9 else np.inf
    lag_s = APERTURE_M * abs_slope                     # max lag across aperture
    return {
        "slope_s_per_m": float(slope),
        "apparent_speed_kmh": float(speed_mps * 3.6),
        "moveout_lag_s": float(lag_s),
        "direction": float(np.sign(slope)),
        "r2": float(r2),
        "n_channels_used": int(finite.sum()),
    }


def main() -> None:
    build_dir = _find_latest_build()
    print(f"Waveform v2 build : {build_dir}")
    print(f"Aperture          : {APERTURE_M:.0f} m  ({N_CHANNELS} ch x "
          f"{CHANNEL_SPACING_M:.0f} m pitch)")
    print(f"Crop window       : {WINDOW_S:.0f} s")

    waveforms = np.load(build_dir / "waveforms.npy")     # (N, 21, T)
    index_df = pd.read_parquet(build_dir / "event_index.parquet")
    ok = index_df[index_df["build_status"] == "ok"].copy()
    ok["waveform_row_idx"] = ok["waveform_row_idx"].astype(int)
    print(f"Events            : {len(ok):,}  |  block shape {waveforms.shape[1:]}")

    smooth_win = max(1, int(round(ENVELOPE_SMOOTH_S * FS_HZ)))
    max_lag_samples = int(round(MAX_LAG_S * FS_HZ))

    records: List[Dict] = []
    for _, row in ok.iterrows():
        block = waveforms[row["waveform_row_idx"]].astype(np.float64)
        lags = _channel_lags_xcorr(block, _center_pos, FS_HZ, smooth_win, max_lag_samples)
        fit = _fit_moveout(POSITIONS_M, lags)
        fit["event_id"] = row["event_id"]
        fit["recorded_speed_kmh"] = row.get("train_speed_kmh", np.nan)
        fit["was_cropped"] = bool(row.get("was_cropped", False))
        records.append(fit)

    res = pd.DataFrame(records)

    # Clean single-passage events (good linear fit, plausible speed).
    good = res[(res["r2"] > 0.6) & (res["apparent_speed_kmh"].between(10, 400))].copy()

    summary = {
        "build_dir": str(build_dir),
        "aperture_m": APERTURE_M,
        "channel_spacing_m": CHANNEL_SPACING_M,
        "window_s": WINDOW_S,
        "n_events": int(len(res)),
        "n_good_fits": int(len(good)),
        "good_fit_fraction": float(len(good) / max(len(res), 1)),
        "moveout_lag_s": {
            "median_all": float(res["moveout_lag_s"].median()),
            "p95_all": float(res["moveout_lag_s"].quantile(0.95)),
            "max_all": float(res["moveout_lag_s"].max()),
            "median_good": float(good["moveout_lag_s"].median()) if len(good) else None,
            "p95_good": float(good["moveout_lag_s"].quantile(0.95)) if len(good) else None,
            "max_good": float(good["moveout_lag_s"].max()) if len(good) else None,
        },
        "apparent_speed_kmh_good": {
            "median": float(good["apparent_speed_kmh"].median()) if len(good) else None,
            "p5": float(good["apparent_speed_kmh"].quantile(0.05)) if len(good) else None,
            "p95": float(good["apparent_speed_kmh"].quantile(0.95)) if len(good) else None,
        },
        "fraction_lag_within_window": float((res["moveout_lag_s"] < WINDOW_S).mean()),
    }

    print("\n--- Moveout summary ---")
    print(f"  Good single-passage fits : {len(good):,} / {len(res):,} "
          f"({summary['good_fit_fraction'] * 100:.1f}%)")
    print(f"  Moveout lag (good)  median={summary['moveout_lag_s']['median_good']}  "
          f"p95={summary['moveout_lag_s']['p95_good']}  max={summary['moveout_lag_s']['max_good']} s")
    print(f"  Apparent speed (good) median={summary['apparent_speed_kmh_good']['median']} km/h")
    print(f"  Fraction with lag < {WINDOW_S:.0f} s window : "
          f"{summary['fraction_lag_within_window'] * 100:.2f}%")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    out_dir = build_dir / "plots" / "moveout"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Apparent speed histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(good["apparent_speed_kmh"], bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("apparent moveout speed [km/h]")
    ax.set_ylabel("events (good fits)")
    ax.set_title(f"Apparent moveout speed  (aperture {APERTURE_M:.0f} m, n={len(good)})")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_dir / "moveout_apparent_speed_hist.png", dpi=140)
    plt.close(fig)

    # 2. Moveout lag histogram with 30 s reference
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(res["moveout_lag_s"].clip(upper=WINDOW_S * 1.2), bins=40,
            color="seagreen", edgecolor="white")
    ax.axvline(WINDOW_S, color="red", lw=1.5, ls="--", label=f"{WINDOW_S:.0f} s crop window")
    ax.set_xlabel("max moveout lag across 20 m aperture [s]")
    ax.set_ylabel("events")
    ax.set_title("Moveout lag vs crop window")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_dir / "moveout_lag_hist.png", dpi=140)
    plt.close(fig)

    # 3. Apparent vs recorded speed
    g2 = good.dropna(subset=["recorded_speed_kmh"])
    if len(g2):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(g2["recorded_speed_kmh"], g2["apparent_speed_kmh"],
                   s=10, alpha=0.4, color="darkorange")
        lim = [0, max(g2["recorded_speed_kmh"].max(), g2["apparent_speed_kmh"].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="1:1")
        ax.set_xlabel("recorded train speed [km/h]")
        ax.set_ylabel("apparent moveout speed [km/h]")
        ax.set_title("Moveout vs recorded speed (good fits)")
        ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        fig.tight_layout(); fig.savefig(out_dir / "moveout_speed_vs_recorded.png", dpi=140)
        plt.close(fig)

    # 4. Example channel x time images with fitted moveout line
    examples = good.sort_values("r2", ascending=False).head(6)
    if len(examples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (_, row) in zip(axes.flatten(), examples.iterrows()):
            ridx = int(ok.loc[ok["event_id"] == row["event_id"], "waveform_row_idx"].iloc[0])
            block = waveforms[ridx]
            vmax = np.percentile(np.abs(block), 99) + 1e-12
            T = block.shape[1] / FS_HZ
            ax.imshow(block, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax,
                      extent=[0, T, WV2.signal.channel_hi, WV2.signal.channel_lo])
            # Overlay fitted line: time vs channel position -> map position to channel ID
            ch_ids = WV2.signal.channel_lo + np.arange(N_CHANNELS)
            line_t = row["slope_s_per_m"] * POSITIONS_M
            line_t = line_t - line_t.mean() + T / 2.0   # centre for display
            ax.plot(line_t, ch_ids, "lime", lw=1.5)
            ax.set_title(f"{row['event_id']}\nv_app={row['apparent_speed_kmh']:.0f} km/h "
                         f"lag={row['moveout_lag_s']:.2f}s R2={row['r2']:.2f}", fontsize=8)
            ax.set_xlabel("time [s]", fontsize=8)
            ax.set_ylabel("channel", fontsize=8)
        fig.suptitle("Moveout examples (best linear fits)", fontsize=12)
        fig.tight_layout(); fig.savefig(out_dir / "moveout_examples.png", dpi=130)
        plt.close(fig)

    with open(out_dir / "moveout_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    res.to_parquet(out_dir / "moveout_per_event.parquet", index=False)

    print(f"\nPlots + summary saved: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
