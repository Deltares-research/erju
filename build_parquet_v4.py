"""Build Parquet v4 — event-level dataset with attenuation-curve targets.

Input : Parquet v2  (sensor-level, one row per event × accelerometer)
Output: Parquet v4  (event-level, one row per train event)

Steps
-----
1. Load the v2 sensor-level dataset.
2. Optionally exclude sensors (MP14-MP19, same as XGBoost v4).
3. Aggregate FO + train-metadata features to one row per event.
4. Add 5 spectral texture features from the 21 octave-band mean powers.
5. Scenario 1 — global n:
     Fit log(PGV) = c_i - n_global * log(r / r0) with fixed-effects OLS.
     Each event gets its own c_i; one n_global is shared by all events.
6. Scenario 2 — per-event n:
     Fit log(PGV) = c_i - n_i * log(r / r0) independently per event.
     Quality flag set 0 for poor fits (too few sensors, no distance spread,
     |n_i| > max_abs_n).
7. Merge scenario parameters into the event-level feature DataFrame.
8. Save:
     dataset.parquet             — event-level ML dataset
     attenuation_global.parquet  — Scenario 1 c_i table + diagnostics
     attenuation_per_event.parquet — Scenario 2 per-event params
     global_fit_summary.json     — scalar summary of the global fit
     build_log.txt
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from src.db.parquet.config_parquet_v4 import CONFIG
from src.db.parquet.parquet_v4_utils import (
    add_texture_features_to_event_df,
    aggregate_event_level_features,
    build_event_level_dataset_for_event_curves,
    build_event_level_dataset_for_global_curve,
    fit_event_specific_attenuation_curves,
    fit_global_attenuation_model,
)
from src.utils.geometry_utils import apply_corrected_distances

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_build_folder(output_root: Path, version_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = output_root / f"{version_name}_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_json(data: dict, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)


def _log(lines: List[str], msg: str) -> None:
    print(msg)
    lines.append(msg)


def main() -> None:
    cfg = CONFIG

    parquet_path = cfg.input_parquet_path()
    if not parquet_path.exists():
        # Auto-discover the latest v2 build
        parquet_root = Path(r"P:\11210978-erju-ai\holten_parquet")
        v2_builds = sorted(parquet_root.glob("parquet_v002_*"), key=lambda p: p.name)
        if not v2_builds:
            raise FileNotFoundError(
                f"Input Parquet not found and no parquet_v002_* builds exist: {parquet_path}"
            )
        parquet_path = v2_builds[-1] / "dataset.parquet"
        print(f"(auto-discovered latest v2: {parquet_path})")

    build_dir = _create_build_folder(
        output_root=cfg.output_root_path(),
        version_name=cfg.output.version_name,
    )
    log_lines: List[str] = []

    _log(log_lines, "=" * 70)
    _log(log_lines, "Parquet v4 — Event-Level Attenuation-Curve Dataset")
    _log(log_lines, "=" * 70)
    _log(log_lines, f"Input : {parquet_path}")
    _log(log_lines, f"Output: {build_dir}")

    # ------------------------------------------------------------------
    # 1. Load v2 sensor-level Parquet
    # ------------------------------------------------------------------
    _log(log_lines, "\n[1/6] Loading sensor-level Parquet v2 ...")
    df_sensor = pd.read_parquet(parquet_path)
    _log(log_lines, f"      Rows   : {len(df_sensor):,}")
    _log(log_lines, f"      Events : {df_sensor[cfg.event_col].nunique():,}")
    _log(log_lines, f"      Columns: {len(df_sensor.columns)}")

    # Apply geometry correction in case an older v2 build is used as input.
    # (New v2 builds already have corrected columns; this is idempotent.)
    if "effective_distance_to_active_track_m" not in df_sensor.columns:
        df_sensor = apply_corrected_distances(df_sensor)
        _log(log_lines, "      Distance correction applied from holten.json.")

    # ------------------------------------------------------------------
    # 2. Exclude unwanted sensors
    # ------------------------------------------------------------------
    if cfg.exclude_sensor_ids:
        before = len(df_sensor)
        df_sensor = df_sensor[~df_sensor["sensor_id"].isin(cfg.exclude_sensor_ids)]
        dropped = before - len(df_sensor)
        _log(log_lines, f"\n[2/6] Excluding {cfg.exclude_sensor_ids}: -{dropped} rows")
    _log(
        log_lines,
        f"      Remaining: {len(df_sensor):,} rows, "
        f"{df_sensor[cfg.event_col].nunique():,} events",
    )

    # ------------------------------------------------------------------
    # 3. Aggregate to event level
    # ------------------------------------------------------------------
    _log(log_lines, "\n[3/6] Aggregating to event level ...")
    df_event = aggregate_event_level_features(
        df=df_sensor,
        event_col=cfg.event_col,
        drop_sensor_cols=cfg.sensor_specific_cols,
    )
    _log(log_lines, f"      Events in event-level dataset: {len(df_event):,}")
    _log(log_lines, f"      Columns: {len(df_event.columns)}")

    # Identify octave-band columns for reporting
    fo_oct_cols = [
        c for c in df_event.columns if c.startswith("fo_oct_") and c.endswith("_mean")
    ]
    _log(log_lines, f"      Octave-band mean columns found: {len(fo_oct_cols)}")

    # ------------------------------------------------------------------
    # 4. Add spectral texture features
    # ------------------------------------------------------------------
    _log(log_lines, "\n[4/6] Adding spectral texture features ...")
    df_event = add_texture_features_to_event_df(
        df=df_event,
        octave_band_col_prefix="fo_oct_",
        reduction="mean",
    )
    texture_cols = [c for c in df_event.columns if "fo_texture_" in c]
    _log(log_lines, f"      Added {len(texture_cols)} texture features: {texture_cols}")

    # ------------------------------------------------------------------
    # 5. Scenario 1 — global attenuation exponent
    # ------------------------------------------------------------------
    _log(log_lines, "\n[5/6] Fitting Scenario 1 — global attenuation exponent ...")
    r0 = cfg.attenuation.r0_m

    global_fit = fit_global_attenuation_model(
        df=df_sensor,
        event_col=cfg.event_col,
        distance_col=cfg.distance_col,
        pgv_col=cfg.pgv_col,
        r0=r0,
    )
    n_global = global_fit["n_global"]
    _log(log_lines, f"      n_global         = {n_global:.4f}")
    _log(log_lines, f"      r0               = {r0:.1f} m")
    _log(log_lines, f"      Events fitted    = {global_fit['n_events']:,}")
    _log(log_lines, f"      Valid rows used  = {global_fit['n_valid_rows']:,}")
    _log(log_lines, f"      Overall RMSE(log)= {global_fit['overall_rmse_log']:.4f}")
    _log(log_lines, f"      Design R²        = {global_fit['design_r2']:.4f}")

    # ------------------------------------------------------------------
    # 6. Scenario 2 — per-event attenuation curves
    # ------------------------------------------------------------------
    _log(log_lines, "\n[6/6] Fitting Scenario 2 — per-event attenuation curves ...")
    df_per_event = fit_event_specific_attenuation_curves(
        df=df_sensor,
        event_col=cfg.event_col,
        distance_col=cfg.distance_col,
        pgv_col=cfg.pgv_col,
        r0=r0,
        min_points=cfg.attenuation.min_sensors_event,
        max_abs_n=cfg.attenuation.max_abs_n,
    )
    good_fits = int((df_per_event["quality_flag"] == 1).sum())
    _log(log_lines, f"      Events processed : {len(df_per_event):,}")
    _log(log_lines, f"      Good fits (flag=1): {good_fits:,}")
    _log(log_lines, f"      Poor fits (flag=0): {len(df_per_event) - good_fits:,}")

    fit_status_counts = df_per_event["fit_status"].value_counts().to_dict()
    for status, count in sorted(fit_status_counts.items()):
        _log(log_lines, f"        {status}: {count}")

    if good_fits > 0:
        ni_good = df_per_event.loc[df_per_event["quality_flag"] == 1, "n_i"]
        _log(
            log_lines,
            f"      n_i (good fits) — mean={ni_good.mean():.3f}  "
            f"std={ni_good.std():.3f}  "
            f"p5={ni_good.quantile(0.05):.3f}  "
            f"p95={ni_good.quantile(0.95):.3f}",
        )
        ci_good = df_per_event.loc[df_per_event["quality_flag"] == 1, "c_i"]
        _log(
            log_lines,
            f"      c_i (good fits) — mean={ci_good.mean():.3f}  "
            f"std={ci_good.std():.3f}",
        )

    # ------------------------------------------------------------------
    # 7. Build event-level ML datasets
    # ------------------------------------------------------------------
    df_s1 = build_event_level_dataset_for_global_curve(
        df_event_features=df_event,
        df_curve_params=global_fit["event_params"],
        event_col=cfg.event_col,
    )
    _log(
        log_lines,
        f"\n      Scenario 1 dataset: {len(df_s1):,} events, "
        f"{len(df_s1.columns)} columns",
    )

    df_s2 = build_event_level_dataset_for_event_curves(
        df_event_features=df_event,
        df_curve_params=df_per_event,
        event_col=cfg.event_col,
        require_good_fit=True,
    )
    _log(
        log_lines,
        f"      Scenario 2 dataset: {len(df_s2):,} events (good fits only), "
        f"{len(df_s2.columns)} columns",
    )

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    _log(log_lines, "\n      Saving outputs ...")

    dataset_path = build_dir / cfg.output.dataset_filename
    df_s1.to_parquet(dataset_path, index=False)
    _log(log_lines, f"      → {dataset_path.name}  (Scenario 1, event-level)")

    # Also save the Scenario 2 dataset under a separate name
    s2_path = build_dir / "dataset_s2.parquet"
    df_s2.to_parquet(s2_path, index=False)
    _log(log_lines, f"      → {s2_path.name}  (Scenario 2, good fits only)")

    # Scenario 1 params table
    s1_params_path = build_dir / cfg.output.scenario1_params_filename
    global_fit["event_params"].to_parquet(s1_params_path, index=False)
    _log(log_lines, f"      → {s1_params_path.name}")

    # Scenario 2 params table
    s2_params_path = build_dir / cfg.output.scenario2_params_filename
    df_per_event.to_parquet(s2_params_path, index=False)
    _log(log_lines, f"      → {s2_params_path.name}")

    # JSON summary of the global fit
    summary = {
        "n_global": float(n_global),
        "r0_m": float(r0),
        "n_events_fitted": global_fit["n_events"],
        "n_valid_rows": global_fit["n_valid_rows"],
        "overall_rmse_log": float(global_fit["overall_rmse_log"]),
        "design_r2": float(global_fit["design_r2"]),
        "scenario2_events_total": len(df_per_event),
        "scenario2_events_good": int(good_fits),
        "build_folder": str(build_dir),
        "input_parquet": str(parquet_path),
    }
    _save_json(summary, build_dir / cfg.output.global_fit_summary_filename)

    # Build log
    with open(build_dir / cfg.output.build_log_filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"Build folder    : {build_dir}")
    print(f"Scenario 1 rows : {len(df_s1):,} events  |  {len(df_s1.columns)} columns")
    print(f"Scenario 2 rows : {len(df_s2):,} events  |  {len(df_s2.columns)} columns")
    print(f"n_global        : {n_global:.4f}  (r0 = {r0:.1f} m)")


if __name__ == "__main__":
    main()
