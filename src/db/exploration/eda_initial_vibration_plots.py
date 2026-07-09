#!/usr/bin/env python
"""
eda_sensor_level_v2.py

Initial exploratory plots for Holten/ERJU sensor-level vibration data.

Purpose
-------
Use the *sensor-level* Parquet v2 database, not the event-level Parquet v4 database.

Expected sensor-level columns include:
    event_id
    sensor_id
    target_pgv_z_mms
    train_type
    train_speed_kmh
    track_number
    acc_distance_to_track_m
    effective_distance_to_active_track_m   optional; computed if possible
    train_type_code                         optional

Outputs
-------
A timestamped folder with:
    - target PGV / log-target histograms
    - PGV distributions by sensor, track, train type
    - PGV distributions by train type separated by sensor
    - PGV/log-PGV vs train speed
    - PGV/log-PGV vs distance
    - per-event attenuation/profile examples
    - summary CSVs and correlations

Run
---
python eda_sensor_level_v2.py

Optional:
python eda_sensor_level_v2.py --data "P:\\11210978-erju-ai\\holten_parquet\\parquet_v002_...\\dataset.parquet"
python eda_sensor_level_v2.py --subset linec_side_minus1
python eda_sensor_level_v2.py --subset all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Path helpers
# =============================================================================

def default_root() -> Path:
    """Project data root, compatible with Windows and Linux cluster paths."""
    if os.name == "nt":
        return Path(r"P:\11210978-erju-ai")
    return Path("/p/11210978-erju-ai")


def discover_repo_root(start: Optional[Path] = None) -> Path:
    """
    Try to find the repository root from the current script location.
    This makes the script robust when placed in src/db/exploration.
    """
    start = (start or Path(__file__).resolve()).resolve()
    candidates = [start.parent, *start.parents]
    for p in candidates:
        if (p / "src").exists() and (p / "sites").exists():
            return p
        if (p / "src").exists() and (p / "pyproject.toml").exists():
            return p
        if (p / "train_xgb_v6.py").exists():
            return p
    return Path.cwd()


def add_repo_to_path(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def latest_build(root: Path, pattern: str) -> Path:
    builds = sorted(root.glob(pattern), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No builds matching {pattern!r} under {root}")
    return builds[-1]


def find_latest_sensor_parquet(parquet_root: Path) -> Path:
    """
    Sensor-level target rows are in parquet_v002_*.

    Important:
    parquet_v004_* is event-level attenuation/features and generally does not contain
    sensor_id or target_pgv_z_mms. It is not suitable for sensor distribution plots.
    """
    build = latest_build(parquet_root, "parquet_v002_*")
    path = build / "dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset.parquet not found: {path}")
    return path


def make_output_dir(output_root: Path, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_root / f"{tag}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# =============================================================================
# Data preparation
# =============================================================================

def first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def natural_sensor_key(s: str):
    m = re.search(r"(\d+)$", str(s))
    if m:
        return (str(s)[: m.start()], int(m.group(1)))
    return (str(s), 10**9)


# Geometric sensor display order for the line-C subset used in the modelling.
# This is intentionally not the same as natural/alphanumeric ordering.
SENSOR_PLOT_ORDER = ["MP4", "MP8", "MP10", "MP1", "MP2"]
SENSOR_PLOT_ORDER_INDEX = {name: idx for idx, name in enumerate(SENSOR_PLOT_ORDER)}


def sensor_plot_key(s: str):
    s = str(s)
    if s in SENSOR_PLOT_ORDER_INDEX:
        return (0, SENSOR_PLOT_ORDER_INDEX[s])
    prefix, number = natural_sensor_key(s)
    return (1, str(prefix), int(number))


TRAIN_TYPE_GROUP_ORDER = [
    "ICM",
    "SPR_A",
    "SNG",
    "GO",
    "DDZ_DDX",
    "ICR",
    "VIRM",
    "OTHER",
]


def classify_train_type_group(value: object) -> str:
    """
    Priority-based train-type grouping for exploratory plots.

    The priority order is intentional. For example, a mixed label such as
    "SNG + SPR(A)" is classified as SPR_A because SPR_A is checked before SNG.
    """
    if pd.isna(value):
        return "OTHER"

    raw = str(value).strip()
    if raw == "" or raw.lower() in {"nan", "none", "unknown"}:
        return "OTHER"

    # Normalize common separators while keeping parentheses visible for SPR(A).
    text = raw.upper()
    text_norm = re.sub(r"[\s\-]+", "_", text)

    # 1. ICM: any train type containing ICM.
    if "ICM" in text_norm:
        return "ICM"

    # 2. SPR_A: SPR_A or SPR(A), including mixed labels like SNG + SPR(A).
    if "SPR_A" in text_norm or "SPR(A)" in text or "SPRA" in text_norm:
        return "SPR_A"

    # 3. SNG: any train type containing SNG.
    if "SNG" in text_norm:
        return "SNG"

    # 4. GO: any train type code containing the two letters GO.
    if "GO" in text_norm:
        return "GO"

    # 5. DDZ_DDX: DDZ or DDX families.
    if "DDZ" in text_norm or "DDX" in text_norm:
        return "DDZ_DDX"

    # 6. ICR: explicitly ICR. Use a token boundary so unrelated substrings do not match.
    if re.search(r"(^|[^A-Z0-9])ICR([^A-Z0-9]|$)", text):
        return "ICR"

    # 7. VIRM.
    if "VIRM" in text_norm:
        return "VIRM"

    return "OTHER"


def try_apply_corrected_distances(df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """
    Use the repository geometry correction when available.
    If the import fails, keep the dataframe unchanged and use existing distance columns.
    """
    if "effective_distance_to_active_track_m" in df.columns:
        return df

    add_repo_to_path(repo_root)
    try:
        from src.utils.geometry_utils import apply_corrected_distances  # type: ignore
    except Exception as exc:
        print(f"[WARN] Could not import apply_corrected_distances: {exc}")
        return df

    try:
        df2 = apply_corrected_distances(df.copy())
        if "effective_distance_to_active_track_m" in df2.columns:
            print("[INFO] Added effective_distance_to_active_track_m via apply_corrected_distances().")
        return df2
    except Exception as exc:
        print(f"[WARN] apply_corrected_distances failed: {exc}")
        return df


def load_line_c_sensors(repo_root: Path) -> list[str]:
    """
    Prefer sites/holten.json if available. Fallback is the line-C set used in the
    recent curve-query / geometry experiments.
    """
    fallback = ["MP4", "MP8", "MP10", "MP1", "MP2"]
    site_path = repo_root / "sites" / "holten.json"
    if not site_path.exists():
        return fallback

    try:
        with open(site_path, "r", encoding="utf-8") as f:
            site = json.load(f)

        # Variant 1: explicit line-C sensor list in some configs/files.
        for key in ("line_c_sensors", "sensor_line_c", "line_C_sensors"):
            val = site.get(key)
            if isinstance(val, list) and val:
                return [str(x) for x in val]

        # Variant 2: line_geometry/sensor_line_map as used by report_event_grid.py.
        line_geom = site.get("accelerometer", {}).get("line_geometry", {})
        sensor_line_map = line_geom.get("sensor_line_map", {})
        if sensor_line_map:
            out = []
            for sensor, line in sensor_line_map.items():
                if str(line).lower().endswith("c") or str(line).lower().endswith("line_c"):
                    out.append(str(sensor))
            if out:
                return sorted(out, key=sensor_plot_key)

    except Exception as exc:
        print(f"[WARN] Could not read line-C sensors from {site_path}: {exc}")

    return fallback


def prepare_dataframe(df_raw: pd.DataFrame, repo_root: Path, out_dir: Path) -> pd.DataFrame:
    df = df_raw.copy()

    # Store available columns early for debugging.
    pd.DataFrame({"column": list(df.columns)}).to_csv(out_dir / "00_available_columns_raw.csv", index=False)

    # The target-by-sensor plots require the sensor-level table.
    target_col = first_existing(
        df,
        [
            "target_pgv_z_mms",
            "target_pgv",
            "target_pgv_mms",
            "pgv_z_mms",
            "pgv_mms",
        ],
    )
    sensor_col = first_existing(df, ["sensor_id", "sensor", "acc_sensor", "measurement_point"])

    if target_col is None or sensor_col is None:
        msg = [
            "This dataframe does not look like the sensor-level Parquet v2 table.",
            f"Rows/cols: {df.shape[0]:,} x {df.shape[1]:,}",
            f"Found target column: {target_col}",
            f"Found sensor column: {sensor_col}",
            "",
            "For the requested plots, use a parquet_v002_* dataset, for example:",
            r"  P:\11210978-erju-ai\holten_parquet\parquet_v002_...\dataset.parquet",
            "",
            "The dataframe you loaded may be parquet_v004 event-level data, which has event features",
            "but no sensor-level target_pgv_z_mms/sensor_id rows.",
            "",
            f"Available columns: {list(df.columns)[:120]}",
        ]
        raise KeyError("\n".join(msg))

    df = try_apply_corrected_distances(df, repo_root)

    distance_col = first_existing(
        df,
        [
            "effective_distance_to_active_track_m",
            "acc_distance_to_track_m",
            "distance",
            "acc_distance_to_track_2_m",
        ],
    )
    train_type_col = first_existing(df, ["train_type", "train_type_code"])
    speed_col = first_existing(df, ["train_speed_kmh", "speed_kmh"])
    track_col = first_existing(df, ["track_number", "track"])

    # Canonical aliases used by the plotting functions.
    df["target_pgv"] = pd.to_numeric(df[target_col], errors="coerce")
    df["target_log"] = np.log(np.clip(df["target_pgv"].astype(float), 1e-9, None))
    df["sensor"] = df[sensor_col].astype(str)

    if distance_col:
        df["distance_m"] = pd.to_numeric(df[distance_col], errors="coerce")
    else:
        df["distance_m"] = np.nan

    if train_type_col:
        df["train_type_plot"] = df[train_type_col].astype(str)
    else:
        df["train_type_plot"] = "unknown"

    df["train_type_group"] = df["train_type_plot"].map(classify_train_type_group)

    if speed_col:
        df["train_speed_kmh_plot"] = pd.to_numeric(df[speed_col], errors="coerce")
    else:
        df["train_speed_kmh_plot"] = np.nan

    if track_col:
        df["track_number_plot"] = df[track_col].astype(str)
    else:
        df["track_number_plot"] = "unknown"

    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(str)

    alias_rows = [
        {"canonical": "target_pgv", "source": target_col},
        {"canonical": "target_log", "source": f"log({target_col})"},
        {"canonical": "sensor", "source": sensor_col},
        {"canonical": "distance_m", "source": distance_col or ""},
        {"canonical": "train_type_plot", "source": train_type_col or ""},
        {"canonical": "train_type_group", "source": f"grouped({train_type_col})" if train_type_col else "grouped(unknown)"},
        {"canonical": "train_speed_kmh_plot", "source": speed_col or ""},
        {"canonical": "track_number_plot", "source": track_col or ""},
    ]
    pd.DataFrame(alias_rows).to_csv(out_dir / "00_alias_mapping.csv", index=False)

    # Clean only for core plots.
    before = len(df)
    df = df[df["target_pgv"].notna()].copy()
    df = df[df["target_pgv"] > 0].copy()
    after = len(df)
    print(f"[INFO] Valid positive target rows: {after:,} / {before:,}")

    return df


def subset_dataframe(df: pd.DataFrame, subset: str, repo_root: Path) -> pd.DataFrame:
    """
    Select the data actually used for the EDA.

    all:
        all sensor-level rows.
    linec:
        only line-C sensors.
    linec_side_minus1:
        line-C sensors on acc_side_of_track == -1 if available.
        This matches the recent line-C side -1 modelling setup.
    """
    subset = subset.lower()
    if subset == "all":
        return df.copy()

    linec = load_line_c_sensors(repo_root)
    out = df[df["sensor"].isin(linec)].copy()

    if subset == "linec":
        return out

    if subset in {"linec_side_minus1", "linec_side_-1", "side_minus1"}:
        if "acc_side_of_track" in out.columns:
            out = out[pd.to_numeric(out["acc_side_of_track"], errors="coerce") == -1].copy()
        else:
            print("[WARN] acc_side_of_track not found; using line-C sensors without side filtering.")
        return out

    raise ValueError(f"Unknown subset: {subset}")


# =============================================================================
# Summaries
# =============================================================================

def save_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    overview = {
        "rows": int(len(df)),
        "events": int(df["event_id"].nunique()) if "event_id" in df.columns else np.nan,
        "sensors": int(df["sensor"].nunique()),
        "tracks": int(df["track_number_plot"].nunique()),
        "train_types": int(df["train_type_plot"].nunique()),
        "train_type_groups": int(df["train_type_group"].nunique()) if "train_type_group" in df.columns else np.nan,
        "pgv_min": float(df["target_pgv"].min()),
        "pgv_p50": float(df["target_pgv"].quantile(0.50)),
        "pgv_p90": float(df["target_pgv"].quantile(0.90)),
        "pgv_p95": float(df["target_pgv"].quantile(0.95)),
        "pgv_p99": float(df["target_pgv"].quantile(0.99)),
        "pgv_max": float(df["target_pgv"].max()),
        "log_pgv_mean": float(df["target_log"].mean()),
        "log_pgv_std": float(df["target_log"].std()),
    }
    pd.DataFrame([overview]).to_csv(out_dir / "00_overview.csv", index=False)

    df[["target_pgv", "target_log", "distance_m", "train_speed_kmh_plot"]].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_csv(out_dir / "00_basic_describe.csv")

    by_sensor = (
        df.groupby("sensor")
        .agg(
            n=("target_pgv", "size"),
            events=("event_id", "nunique") if "event_id" in df.columns else ("target_pgv", "size"),
            pgv_mean=("target_pgv", "mean"),
            pgv_median=("target_pgv", "median"),
            pgv_p90=("target_pgv", lambda x: x.quantile(0.90)),
            pgv_p95=("target_pgv", lambda x: x.quantile(0.95)),
            pgv_max=("target_pgv", "max"),
            log_mean=("target_log", "mean"),
            log_std=("target_log", "std"),
            distance_m_median=("distance_m", "median"),
        )
        .reset_index()
    )
    by_sensor["_sort"] = by_sensor["sensor"].map(sensor_plot_key)
    by_sensor = by_sensor.sort_values("_sort").drop(columns="_sort")
    by_sensor.to_csv(out_dir / "00_summary_by_sensor.csv", index=False)

    (
        df.groupby("track_number_plot")
        .agg(
            n=("target_pgv", "size"),
            events=("event_id", "nunique") if "event_id" in df.columns else ("target_pgv", "size"),
            pgv_mean=("target_pgv", "mean"),
            pgv_median=("target_pgv", "median"),
            pgv_p90=("target_pgv", lambda x: x.quantile(0.90)),
            pgv_p95=("target_pgv", lambda x: x.quantile(0.95)),
            pgv_max=("target_pgv", "max"),
        )
        .reset_index()
        .to_csv(out_dir / "00_summary_by_track_number.csv", index=False)
    )

    (
        df.groupby("train_type_plot")
        .agg(
            n=("target_pgv", "size"),
            events=("event_id", "nunique") if "event_id" in df.columns else ("target_pgv", "size"),
            pgv_mean=("target_pgv", "mean"),
            pgv_median=("target_pgv", "median"),
            pgv_p90=("target_pgv", lambda x: x.quantile(0.90)),
            pgv_p95=("target_pgv", lambda x: x.quantile(0.95)),
            pgv_max=("target_pgv", "max"),
            speed_median=("train_speed_kmh_plot", "median"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
        .to_csv(out_dir / "00_summary_by_train_type.csv", index=False)
    )

    if "train_type_group" in df.columns:
        group_summary = (
            df.groupby("train_type_group")
            .agg(
                n=("target_pgv", "size"),
                events=("event_id", "nunique") if "event_id" in df.columns else ("target_pgv", "size"),
                original_train_types=("train_type_plot", lambda x: "; ".join(sorted(map(str, x.dropna().unique()))[:25])),
                pgv_mean=("target_pgv", "mean"),
                pgv_median=("target_pgv", "median"),
                pgv_p90=("target_pgv", lambda x: x.quantile(0.90)),
                pgv_p95=("target_pgv", lambda x: x.quantile(0.95)),
                pgv_max=("target_pgv", "max"),
                speed_median=("train_speed_kmh_plot", "median"),
            )
            .reset_index()
        )
        group_summary["_order"] = group_summary["train_type_group"].map(
            {name: i for i, name in enumerate(TRAIN_TYPE_GROUP_ORDER)}
        )
        group_summary = group_summary.sort_values(["_order", "n"], ascending=[True, False]).drop(columns="_order")
        group_summary.to_csv(out_dir / "00_summary_by_train_type_group.csv", index=False)

        mapping = (
            df.groupby(["train_type_plot", "train_type_group"])
            .agg(n=("target_pgv", "size"), events=("event_id", "nunique") if "event_id" in df.columns else ("target_pgv", "size"))
            .reset_index()
            .sort_values(["train_type_group", "n"], ascending=[True, False])
        )
        mapping["_order"] = mapping["train_type_group"].map(
            {name: i for i, name in enumerate(TRAIN_TYPE_GROUP_ORDER)}
        )
        mapping = mapping.sort_values(["_order", "n"], ascending=[True, False]).drop(columns="_order")
        mapping.to_csv(out_dir / "00_train_type_group_mapping.csv", index=False)

    # Cross summaries: train type/category separated by sensor.
    def _sensor_category_summary(cat_col: str, out_name: str) -> None:
        if cat_col not in df.columns:
            return
        tmp = (
            df.groupby(["sensor", cat_col])
            .agg(
                n=("target_pgv", "size"),
                events=("event_id", "nunique") if "event_id" in df.columns else ("target_pgv", "size"),
                pgv_mean=("target_pgv", "mean"),
                pgv_median=("target_pgv", "median"),
                pgv_p75=("target_pgv", lambda x: x.quantile(0.75)),
                pgv_p90=("target_pgv", lambda x: x.quantile(0.90)),
                pgv_p95=("target_pgv", lambda x: x.quantile(0.95)),
                pgv_max=("target_pgv", "max"),
                log_mean=("target_log", "mean"),
                log_median=("target_log", "median"),
                speed_median=("train_speed_kmh_plot", "median"),
                distance_m_median=("distance_m", "median"),
            )
            .reset_index()
        )
        tmp["_sensor_sort"] = tmp["sensor"].map(sensor_plot_key)
        if cat_col == "train_type_group":
            tmp["_cat_order"] = tmp[cat_col].map({name: i for i, name in enumerate(TRAIN_TYPE_GROUP_ORDER)}).fillna(999)
            tmp = tmp.sort_values(["_sensor_sort", "_cat_order", "n"], ascending=[True, True, False])
            tmp = tmp.drop(columns=["_sensor_sort", "_cat_order"])
        else:
            tmp = tmp.sort_values(["_sensor_sort", "n"], ascending=[True, False]).drop(columns=["_sensor_sort"])
        tmp.to_csv(out_dir / out_name, index=False)

    _sensor_category_summary("train_type_group", "00_summary_by_sensor_and_train_type_group.csv")
    _sensor_category_summary("train_type_plot", "00_summary_by_sensor_and_train_type.csv")

    # Numeric correlations with target_log and target_pgv.
    numeric = df.select_dtypes(include=[np.number]).copy()
    leak_like = [c for c in numeric.columns if c.lower() in {"target_pgv", "target_log"}]
    corr_rows = []
    for target in ["target_log", "target_pgv"]:
        if target not in numeric.columns:
            continue
        corr = numeric.corr(numeric_only=True)[target].drop(labels=leak_like, errors="ignore")
        corr = corr.replace([np.inf, -np.inf], np.nan).dropna().sort_values(key=lambda s: s.abs(), ascending=False)
        for feature, value in corr.items():
            corr_rows.append({"target": target, "feature": feature, "pearson_corr": float(value), "abs_corr": float(abs(value))})
    pd.DataFrame(corr_rows).to_csv(out_dir / "00_numeric_correlations.csv", index=False)


# =============================================================================
# Plot helpers
# =============================================================================

def save_fig(path: Path, dpi: int = 150) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def clipped_values(x: pd.Series, q: float = 0.995) -> pd.Series:
    x = x.dropna()
    if x.empty:
        return x
    hi = x.quantile(q)
    return x[x <= hi]


def grouped_arrays(df: pd.DataFrame, group_col: str, value_col: str, min_n: int = 1, top_n: Optional[int] = None):
    counts = df[group_col].value_counts()
    valid = counts[counts >= min_n]

    if group_col == "train_type_group":
        cats = [cat for cat in TRAIN_TYPE_GROUP_ORDER if cat in valid.index]
    else:
        cats = valid.index.tolist()
        if top_n is not None:
            cats = cats[:top_n]
        # Natural sorting for sensors and tracks; count sorting for detailed train type if top_n used.
        if group_col in {"sensor", "track_number_plot"}:
            cats = sorted(cats, key=sensor_plot_key)

    arrays = [df.loc[df[group_col] == cat, value_col].dropna().values for cat in cats]
    labels = [f"{cat}\n(n={len(arr)})" for cat, arr in zip(cats, arrays)]
    return cats, arrays, labels


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    # Raw PGV full
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["target_pgv"].dropna(), bins=60)
    ax.set_xlabel("target_pgv_z_mms [mm/s]")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of target PGV")
    save_fig(out_dir / "01_hist_target_pgv_full.png")

    # Raw PGV clipped
    x = clipped_values(df["target_pgv"], 0.995)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x, bins=60)
    ax.set_xlabel("target_pgv_z_mms [mm/s]")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of target PGV, clipped at p99.5")
    save_fig(out_dir / "01b_hist_target_pgv_clipped_p995.png")

    # Log target
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["target_log"].dropna(), bins=60)
    ax.set_xlabel("target_log = ln(target_pgv_z_mms)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of target_log")
    save_fig(out_dir / "02_hist_target_log.png")


def plot_box_and_violin(df: pd.DataFrame, group_col: str, group_label: str, prefix: str, out_dir: Path, top_n: Optional[int] = None) -> None:
    cats, arrays_pgv, labels = grouped_arrays(df, group_col, "target_pgv", min_n=3, top_n=top_n)
    if not arrays_pgv:
        print(f"[WARN] No groups to plot for {group_col}")
        return

    # PGV boxplot, clipped y-axis to make central structure visible.
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels)), 5))
    ax.boxplot(arrays_pgv, tick_labels=labels, showfliers=True)
    ax.set_ylabel("target_pgv_z_mms [mm/s]")
    ax.set_title(f"PGV distribution by {group_label}")
    ax.tick_params(axis="x", rotation=45)
    save_fig(out_dir / f"{prefix}_pgv_boxplot_by_{group_col}.png")

    # PGV violin.
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels)), 5))
    ax.violinplot(arrays_pgv, showmedians=True, showextrema=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("target_pgv_z_mms [mm/s]")
    ax.set_title(f"PGV violin by {group_label}")
    save_fig(out_dir / f"{prefix}b_pgv_violin_by_{group_col}.png")

    # Log-target violin usually more interpretable.
    _, arrays_log, labels_log = grouped_arrays(df, group_col, "target_log", min_n=3, top_n=top_n)
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels_log)), 5))
    ax.violinplot(arrays_log, showmedians=True, showextrema=True)
    ax.set_xticks(np.arange(1, len(labels_log) + 1))
    ax.set_xticklabels(labels_log, rotation=45, ha="right")
    ax.set_ylabel("target_log = ln(target_pgv_z_mms)")
    ax.set_title(f"target_log violin by {group_label}")
    save_fig(out_dir / f"{prefix}c_target_log_violin_by_{group_col}.png")


def scatter_by_sensor(df: pd.DataFrame, x_col: str, y_col: str, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    sub = df[[x_col, y_col, "sensor"]].dropna()
    if sub.empty:
        print(f"[WARN] Empty scatter data for {x_col} vs {y_col}")
        return

    sensors = sorted(sub["sensor"].unique(), key=sensor_plot_key)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sensor in sensors:
        ss = sub[sub["sensor"] == sensor]
        ax.scatter(ss[x_col], ss[y_col], s=10, alpha=0.35, label=sensor)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(sensors) <= 15:
        ax.legend(fontsize=8, ncols=2)
    ax.grid(True, ls=":", alpha=0.4)
    save_fig(out_path)


def plot_speed_and_distance(df: pd.DataFrame, out_dir: Path) -> None:
    scatter_by_sensor(
        df,
        "train_speed_kmh_plot",
        "target_pgv",
        "Train speed [km/h]",
        "target_pgv_z_mms [mm/s]",
        "PGV vs train speed",
        out_dir / "06_pgv_vs_train_speed_by_sensor.png",
    )
    scatter_by_sensor(
        df,
        "train_speed_kmh_plot",
        "target_log",
        "Train speed [km/h]",
        "target_log = ln(target_pgv_z_mms)",
        "target_log vs train speed",
        out_dir / "06b_target_log_vs_train_speed_by_sensor.png",
    )

    scatter_by_sensor(
        df,
        "distance_m",
        "target_pgv",
        "Effective distance to active track [m]",
        "target_pgv_z_mms [mm/s]",
        "PGV vs distance",
        out_dir / "07_pgv_vs_distance_by_sensor.png",
    )
    scatter_by_sensor(
        df,
        "distance_m",
        "target_log",
        "Effective distance to active track [m]",
        "target_log = ln(target_pgv_z_mms)",
        "target_log vs distance",
        out_dir / "07b_target_log_vs_distance_by_sensor.png",
    )


def plot_top_correlations(df: pd.DataFrame, out_dir: Path, top_n: int = 25) -> None:
    numeric = df.select_dtypes(include=[np.number])
    if "target_log" not in numeric.columns:
        return

    corr = numeric.corr(numeric_only=True)["target_log"].drop(labels=["target_log", "target_pgv"], errors="ignore")
    corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
    corr = corr.sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(corr))))
    ax.barh(corr.index[::-1], corr.values[::-1])
    ax.set_xlabel("Pearson correlation with target_log")
    ax.set_title(f"Top {len(corr)} numeric correlations with target_log")
    ax.axvline(0.0, lw=0.8)
    save_fig(out_dir / "09_top_numeric_correlations_with_target_log.png")


def plot_highest_event_profiles(df: pd.DataFrame, out_dir: Path, n_events: int = 12) -> None:
    if "event_id" not in df.columns:
        return
    sub = df.dropna(subset=["distance_m", "target_pgv"])
    if sub.empty:
        return

    top_events = (
        sub.groupby("event_id")["target_pgv"]
        .max()
        .sort_values(ascending=False)
        .head(n_events)
        .index.tolist()
    )
    if not top_events:
        return

    ncols = 3
    nrows = int(math.ceil(len(top_events) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3, 2.8 * nrows)))
    axes = np.array(axes).reshape(-1)

    for ax, eid in zip(axes, top_events):
        ev = sub[sub["event_id"] == eid].sort_values("distance_m")
        ax.plot(ev["distance_m"], ev["target_pgv"], marker="o")
        for _, row in ev.iterrows():
            ax.annotate(str(row["sensor"]), (row["distance_m"], row["target_pgv"]), fontsize=7)
        ax.set_title(str(eid)[:28], fontsize=8)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("PGV [mm/s]")
        ax.grid(True, ls=":", alpha=0.4)

    for ax in axes[len(top_events):]:
        ax.axis("off")

    plt.suptitle("Highest-PGV event profiles", fontsize=11)
    save_fig(out_dir / "08_highest_pgv_event_profiles.png")


def plot_speed_context_by_category(
    df: pd.DataFrame,
    group_col: str,
    group_label: str,
    out_dir: Path,
    out_name: str,
    top_n: Optional[int] = 15,
) -> None:
    """Helps identify confounding: train type can simply proxy speed and track."""
    if group_col not in df.columns:
        return

    if group_col == "train_type_group":
        cats = [cat for cat in TRAIN_TYPE_GROUP_ORDER if cat in set(df[group_col])]
        sub = df[df[group_col].isin(cats)].copy()
        plot_top_n = None
    else:
        counts = df[group_col].value_counts().head(top_n)
        cats = counts.index.tolist()
        sub = df[df[group_col].isin(cats)].copy()
        plot_top_n = top_n

    if sub.empty:
        return

    _, arrays, labels = grouped_arrays(sub, group_col, "train_speed_kmh_plot", min_n=3, top_n=plot_top_n)
    if not arrays:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels)), 5))
    ax.boxplot(arrays, tick_labels=labels, showfliers=True)
    ax.set_ylabel("Train speed [km/h]")
    ax.set_title(f"Train speed distribution by {group_label}")
    ax.tick_params(axis="x", rotation=45)
    save_fig(out_dir / out_name)


def plot_train_type_speed_context(df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    plot_speed_context_by_category(
        df,
        group_col="train_type_plot",
        group_label="train type",
        out_dir=out_dir,
        out_name="10_train_speed_by_train_type.png",
        top_n=top_n,
    )
    plot_speed_context_by_category(
        df,
        group_col="train_type_group",
        group_label="train type group",
        out_dir=out_dir,
        out_name="10b_train_speed_by_train_type_group.png",
        top_n=None,
    )




def category_order_for_plot(df: pd.DataFrame, group_col: str, top_n: Optional[int] = None) -> list[str]:
    """Return category order for plots, using priority order for grouped train type."""
    counts = df[group_col].value_counts()
    if group_col == "train_type_group":
        return [cat for cat in TRAIN_TYPE_GROUP_ORDER if cat in counts.index]
    cats = counts.index.tolist()
    if top_n is not None:
        cats = cats[:top_n]
    return cats


def plot_category_by_sensor_grid(
    df: pd.DataFrame,
    category_col: str,
    category_label: str,
    y_col: str,
    y_label: str,
    title: str,
    out_path: Path,
    top_n: Optional[int] = None,
    kind: str = "box",
    min_n: int = 3,
) -> None:
    """
    Faceted categorical distribution: one panel per sensor.

    This is the most useful view for your current question because it separates
    the train-type effect from the very strong sensor/distance effect.
    """
    needed = ["sensor", category_col, y_col]
    sub = df[needed].dropna().copy()
    if sub.empty:
        print(f"[WARN] Empty category-by-sensor data for {category_col} / {y_col}")
        return

    sensors = sorted(sub["sensor"].unique(), key=sensor_plot_key)
    cats = category_order_for_plot(sub, category_col, top_n=top_n)
    if not cats:
        return

    n_sensors = len(sensors)
    ncols = 2 if n_sensors <= 6 else 3
    nrows = int(math.ceil(n_sensors / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(10, 1.15 * len(cats) * ncols), max(4, 3.2 * nrows)),
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for ax, sensor in zip(axes, sensors):
        ss = sub[sub["sensor"] == sensor]
        arrays = []
        labels = []
        for cat in cats:
            vals = ss.loc[ss[category_col] == cat, y_col].dropna().values
            if len(vals) >= min_n:
                arrays.append(vals)
                labels.append(f"{cat}\n(n={len(vals)})")
            else:
                arrays.append(np.array([]))
                labels.append(f"{cat}\n(n={len(vals)})")

        # Matplotlib cannot draw empty arrays in box/violin plots, so draw only valid positions.
        valid_positions = [i + 1 for i, a in enumerate(arrays) if len(a) >= min_n]
        valid_arrays = [a for a in arrays if len(a) >= min_n]
        if valid_arrays:
            if kind == "violin":
                ax.violinplot(valid_arrays, positions=valid_positions, showmedians=True, showextrema=True)
            else:
                ax.boxplot(valid_arrays, positions=valid_positions, showfliers=True)
        ax.set_title(str(sensor))
        ax.set_xticks(np.arange(1, len(cats) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(True, axis="y", ls=":", alpha=0.35)
        ax.set_xlabel(category_label)
        ax.set_ylabel(y_label)

    for ax in axes[n_sensors:]:
        ax.axis("off")

    plt.suptitle(title, fontsize=12)
    save_fig(out_path)


def plot_sensor_category_heatmap(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    agg: str,
    title: str,
    out_path: Path,
    top_n: Optional[int] = None,
) -> None:
    """Simple sensor × train-type heatmap for summary statistics."""
    if category_col not in df.columns or value_col not in df.columns:
        return
    sub = df[["sensor", category_col, value_col]].dropna().copy()
    if sub.empty:
        return

    sensors = sorted(sub["sensor"].unique(), key=sensor_plot_key)
    cats = category_order_for_plot(sub, category_col, top_n=top_n)
    if not cats:
        return

    if agg == "count":
        piv = sub.pivot_table(index="sensor", columns=category_col, values=value_col, aggfunc="size")
    elif agg == "median":
        piv = sub.pivot_table(index="sensor", columns=category_col, values=value_col, aggfunc="median")
    elif agg == "p90":
        piv = sub.pivot_table(index="sensor", columns=category_col, values=value_col, aggfunc=lambda x: x.quantile(0.90))
    else:
        raise ValueError(f"Unsupported agg: {agg}")

    piv = piv.reindex(index=sensors, columns=cats)
    arr = piv.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(cats)), max(4, 0.55 * len(sensors))))
    im = ax.imshow(masked, aspect="auto")
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(sensors)))
    ax.set_yticklabels(sensors)
    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel("sensor")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{agg}({value_col})")

    # Annotate values for quick interpretation.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                if agg == "count":
                    txt = f"{int(val)}"
                else:
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    save_fig(out_path)


def plot_train_type_by_sensor(df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    """PGV/train-type plots explicitly separated by sensor."""
    # Grouped train type: preferred interpretability view.
    plot_category_by_sensor_grid(
        df,
        category_col="train_type_group",
        category_label="train type group",
        y_col="target_pgv",
        y_label="target_pgv_z_mms [mm/s]",
        title="PGV by train type group, separated by sensor",
        out_path=out_dir / "11_pgv_by_train_type_group_per_sensor_boxplot.png",
        top_n=None,
        kind="box",
    )
    plot_category_by_sensor_grid(
        df,
        category_col="train_type_group",
        category_label="train type group",
        y_col="target_log",
        y_label="target_log = ln(target_pgv_z_mms)",
        title="target_log by train type group, separated by sensor",
        out_path=out_dir / "11b_target_log_by_train_type_group_per_sensor_violin.png",
        top_n=None,
        kind="violin",
    )

    # Original detailed train type: useful, but can be crowded.
    plot_category_by_sensor_grid(
        df,
        category_col="train_type_plot",
        category_label="train type",
        y_col="target_pgv",
        y_label="target_pgv_z_mms [mm/s]",
        title=f"PGV by detailed train type, separated by sensor — top {top_n}",
        out_path=out_dir / "12_pgv_by_train_type_per_sensor_boxplot.png",
        top_n=top_n,
        kind="box",
    )
    plot_category_by_sensor_grid(
        df,
        category_col="train_type_plot",
        category_label="train type",
        y_col="target_log",
        y_label="target_log = ln(target_pgv_z_mms)",
        title=f"target_log by detailed train type, separated by sensor — top {top_n}",
        out_path=out_dir / "12b_target_log_by_train_type_per_sensor_violin.png",
        top_n=top_n,
        kind="violin",
    )

    # Compact summary heatmaps for presentations/debugging.
    plot_sensor_category_heatmap(
        df,
        category_col="train_type_group",
        value_col="target_pgv",
        agg="median",
        title="Median PGV by sensor and train type group",
        out_path=out_dir / "13_median_pgv_heatmap_sensor_x_train_type_group.png",
        top_n=None,
    )
    plot_sensor_category_heatmap(
        df,
        category_col="train_type_group",
        value_col="target_pgv",
        agg="p90",
        title="P90 PGV by sensor and train type group",
        out_path=out_dir / "13b_p90_pgv_heatmap_sensor_x_train_type_group.png",
        top_n=None,
    )
    plot_sensor_category_heatmap(
        df,
        category_col="train_type_group",
        value_col="target_pgv",
        agg="count",
        title="Sample count by sensor and train type group",
        out_path=out_dir / "13c_count_heatmap_sensor_x_train_type_group.png",
        top_n=None,
    )

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Explicit path to sensor-level parquet_v002 dataset.parquet.",
    )
    parser.add_argument(
        "--parquet-root",
        type=str,
        default=str(default_root() / "holten_parquet"),
        help="Root containing parquet_v002_* folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(default_root() / "holten_models" / "outputs" / "eda_sensor_level_v2"),
        help="Root where the EDA output folder is created.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="linec_side_minus1",
        choices=["all", "linec", "linec_side_minus1", "linec_side_-1", "side_minus1"],
        help=(
            "Data subset. Use 'all' for all sensors, 'linec' for line-C sensors, "
            "or 'linec_side_minus1' for the modelling subset used in recent line-C work."
        ),
    )
    parser.add_argument(
        "--top-train-types",
        type=int,
        default=15,
        help="Maximum number of train types shown in train-type plots.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for the output folder name.",
    )
    args = parser.parse_args()

    repo_root = discover_repo_root()
    add_repo_to_path(repo_root)

    parquet_root = Path(args.parquet_root)
    output_root = Path(args.output_root)
    tag = args.tag or f"eda_{args.subset}"

    out_dir = make_output_dir(output_root, tag)
    print(f"[INFO] Repo root  : {repo_root}")
    print(f"[INFO] Output dir : {out_dir}")

    if args.data:
        data_path = Path(args.data)
    else:
        data_path = find_latest_sensor_parquet(parquet_root)

    print(f"[INFO] Loading sensor-level dataset: {data_path}")
    df_raw = pd.read_parquet(data_path)
    print(f"[INFO] Raw shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]:,} cols")

    # Save metadata
    pd.DataFrame(
        [
            {
                "repo_root": str(repo_root),
                "data_path": str(data_path),
                "raw_rows": len(df_raw),
                "raw_cols": len(df_raw.columns),
                "subset": args.subset,
            }
        ]
    ).to_csv(out_dir / "00_run_metadata.csv", index=False)

    df = prepare_dataframe(df_raw, repo_root=repo_root, out_dir=out_dir)
    df = subset_dataframe(df, args.subset, repo_root=repo_root)
    print(f"[INFO] After subset={args.subset!r}: {len(df):,} rows")

    if df.empty:
        raise ValueError(
            f"No rows left after subset={args.subset!r}. "
            "Try --subset all, or inspect 00_available_columns_raw.csv."
        )

    # Save the plotting dataframe head to make debugging easy.
    cols_debug = [
        c
        for c in [
            "event_id",
            "sensor",
            "target_pgv",
            "target_log",
            "train_type_plot",
            "train_type_group",
            "train_speed_kmh_plot",
            "track_number_plot",
            "distance_m",
            "acc_side_of_track",
        ]
        if c in df.columns
    ]
    df[cols_debug].head(200).to_csv(out_dir / "00_plotting_dataframe_head.csv", index=False)

    save_summaries(df, out_dir)
    plot_histograms(df, out_dir)
    plot_box_and_violin(df, "sensor", "sensor", "03", out_dir)
    plot_box_and_violin(df, "track_number_plot", "track number", "04", out_dir)
    # Detailed/original train-type plots.
    plot_box_and_violin(df, "train_type_plot", "train type", "05", out_dir, top_n=args.top_train_types)

    # Grouped train-type plots using the priority order requested for interpretation.
    plot_box_and_violin(df, "train_type_group", "train type group", "05g", out_dir, top_n=None)

    plot_speed_and_distance(df, out_dir)
    plot_highest_event_profiles(df, out_dir)
    plot_top_correlations(df, out_dir)
    plot_train_type_speed_context(df, out_dir, top_n=args.top_train_types)
    plot_train_type_by_sensor(df, out_dir, top_n=args.top_train_types)

    print("\n[DONE] Initial EDA plots written to:")
    print(f"  {out_dir}")
    print("\nStart with:")
    print("  00_overview.csv")
    print("  00_summary_by_sensor.csv")
    print("  01_hist_target_pgv_full.png")
    print("  02_hist_target_log.png")
    print("  03_pgv_boxplot_by_sensor.png")
    print("  05_pgv_boxplot_by_train_type_plot.png")
    print("  05g_pgv_boxplot_by_train_type_group.png")
    print("  11_pgv_by_train_type_group_per_sensor_boxplot.png")
    print("  13_median_pgv_heatmap_sensor_x_train_type_group.png")
    print("  00_train_type_group_mapping.csv")
    print("  06b_target_log_vs_train_speed_by_sensor.png")
    print("  07b_target_log_vs_distance_by_sensor.png")


if __name__ == "__main__":
    main()
