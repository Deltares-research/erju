#!/usr/bin/env python3
"""Create a 2×5 grid of random individual test spectra from saved predictions.

Works with the existing S2/S3/S5 ``predictions_test.parquet`` files, so this
plot can be generated without retraining.

Example
-------
python plot_random_test_events.py \
  --predictions /p/.../S5_seed42_.../metrics/predictions_test.parquet \
  --aligned-events /p/.../spectral_fo_alignment_v001/aligned_events.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BANDS = [1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0,
         12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0]
V_REF_MMS = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--aligned-events", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--n-events", type=int, default=10)
    return p.parse_args()


def total_rms(db: np.ndarray) -> float:
    v = V_REF_MMS * np.power(10.0, db / 20.0)
    return float(np.sqrt(np.sum(v * v)))


def main() -> None:
    args = parse_args()
    if args.predictions.suffix.lower() == ".csv":
        df = pd.read_csv(args.predictions)
    else:
        df = pd.read_parquet(args.predictions)
    required = {"event_id", "band_nominal_hz", "measured_level_db", "predicted_level_db"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions file: {sorted(missing)}")
    df = df[df["band_nominal_hz"].isin(BANDS)].copy()
    counts = df.groupby("event_id")["band_nominal_hz"].nunique()
    valid_ids = counts[counts == len(BANDS)].index.astype(str)
    df["event_id"] = df["event_id"].astype(str)
    df = df[df["event_id"].isin(valid_ids)]

    meta_cols = [c for c in ["event_id", "train_family", "track_number", "train_speed_kmh"]
                 if c in df.columns]
    meta = df[meta_cols].drop_duplicates("event_id")
    if args.aligned_events is not None:
        aligned = (pd.read_csv(args.aligned_events)
                   if args.aligned_events.suffix.lower() == ".csv"
                   else pd.read_parquet(args.aligned_events))
        aligned["event_id"] = aligned["event_id"].astype(str)
        keep = [c for c in ["event_id", "train_speed_kmh", "track_number", "train_type"]
                if c in aligned.columns]
        meta = meta.merge(aligned[keep].drop_duplicates("event_id"), on="event_id",
                          how="left", suffixes=("", "_aligned"))
        for c in ["train_speed_kmh", "track_number"]:
            ca = f"{c}_aligned"
            if ca in meta:
                meta[c] = meta.get(c, np.nan)
                meta[c] = meta[c].fillna(meta[ca])

    event_ids = np.array(sorted(df["event_id"].unique()))
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(event_ids, size=min(args.n_events, len(event_ids)), replace=False)

    out_dir = args.output_dir or args.predictions.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(BANDS))
    labels = [f"{b:g}" for b in BANDS]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    rows = []
    for j, ax in enumerate(axes):
        if j >= len(chosen):
            ax.axis("off")
            continue
        event_id = str(chosen[j])
        g = df[df["event_id"] == event_id].set_index("band_nominal_hz").reindex(BANDS)
        yt = g["measured_level_db"].to_numpy(float)
        yp = g["predicted_level_db"].to_numpy(float)
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        true_total, pred_total = total_rms(yt), total_rms(yp)
        m = meta[meta["event_id"] == event_id]
        m = m.iloc[0] if len(m) else pd.Series(dtype=object)
        family = str(m.get("train_family", m.get("train_type", "")))
        track = m.get("track_number", "?")
        speed = m.get("train_speed_kmh", np.nan)
        short_id = event_id if len(event_id) <= 18 else f"{event_id[:8]}…{event_id[-7:]}"

        ax.plot(x, yt, "o-", color="black", lw=1.4, ms=3.5, label="Measured")
        ax.plot(x, yp, "s--", color="steelblue", lw=1.3, ms=3.2, label="Predicted")
        speed_text = f"{float(speed):.0f} km/h" if pd.notna(speed) else "speed n/a"
        ax.set_title(
            f"{short_id}\n{family}, T{track}, {speed_text}\n"
            f"RMS {true_total:.3f}→{pred_total:.3f} mm/s | RMSE {rmse:.2f} dB",
            fontsize=8,
        )
        ax.set_xticks(x[::3]); ax.set_xticklabels(labels[::3], rotation=45, ha="right")
        ax.grid(True, alpha=0.25)
        rows.append({
            "plot_order": j + 1, "event_id": event_id, "train_family": family,
            "track_number": track, "train_speed_kmh": speed,
            "measured_total_rms_mms": true_total,
            "predicted_total_rms_mms": pred_total,
            "event_spectral_rmse_db": rmse, "random_seed": args.seed,
        })

    axes[0].legend(fontsize=8)
    fig.supxlabel("One-third-octave band nominal frequency (Hz)")
    fig.supylabel("Velocity band level (dB re 1 nm/s)")
    fig.suptitle("Ten random test events: measured versus predicted spectra", fontsize=14)
    plt.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    fig.savefig(out_dir / "15_random_10_test_events.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "random_10_test_events.csv", index=False)
    print(f"Saved {out_dir / '15_random_10_test_events.png'}")


if __name__ == "__main__":
    main()
