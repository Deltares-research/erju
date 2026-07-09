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
# Attenuation / event-level exploration
# =============================================================================

def parse_track_number_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")


def fit_track_attenuation_exponents(df: pd.DataFrame, r0: float = 10.0) -> tuple[dict[int, float], pd.DataFrame]:
    """Fit track-specific attenuation exponents with the event-centred method."""
    needed = ["event_id", "distance_m", "target_log", "track_number_plot"]
    sub = df[[c for c in needed if c in df.columns]].dropna().copy()
    if sub.empty:
        return {}, pd.DataFrame(columns=["track_number", "n_track", "n_events_used", "n_rows_used"])

    sub = sub[(sub["distance_m"] > 0)].copy()
    sub["track_number_num"] = parse_track_number_numeric(sub["track_number_plot"])
    sub = sub[sub["track_number_num"].isin([1, 2])].copy()

    results: dict[int, float] = {}
    rows = []
    for track_id in [1, 2]:
        st = sub[sub["track_number_num"] == track_id].copy()
        num = 0.0
        den = 0.0
        n_events_used = 0
        n_rows_used = 0

        for _, ev in st.groupby("event_id"):
            if len(ev) < 2:
                continue
            x = np.log(ev["distance_m"].to_numpy(dtype=float) / float(r0))
            y = ev["target_log"].to_numpy(dtype=float)
            if len(np.unique(np.round(x, 10))) < 2:
                continue
            x_c = x - x.mean()
            y_c = y - y.mean()
            denom_ev = np.sum(x_c ** 2)
            if denom_ev <= 1e-12:
                continue
            num += -np.sum(x_c * y_c)
            den += denom_ev
            n_events_used += 1
            n_rows_used += len(ev)

        n_track = float(num / den) if den > 1e-12 else np.nan
        results[track_id] = n_track
        rows.append(
            {
                "track_number": track_id,
                "n_track": n_track,
                "n_events_used": n_events_used,
                "n_rows_used": n_rows_used,
                "r0_m": float(r0),
            }
        )
    return results, pd.DataFrame(rows)


def build_event_level_dataframe(df: pd.DataFrame, track_n_map: dict[int, float], r0: float = 10.0) -> pd.DataFrame:
    if "event_id" not in df.columns:
        return pd.DataFrame()

    # Candidate event-level feature columns carried over from the event-wise repeated rows.
    carry_cols = []
    explicit = [
        "train_speed_kmh_plot",
        "train_type_code",
        "train_type_plot",
        "train_type_group",
        "track_number_plot",
    ]
    for c in explicit:
        if c in df.columns:
            carry_cols.append(c)
    carry_cols += [c for c in df.columns if c.startswith("wf_")]
    carry_cols += [c for c in df.columns if c.startswith("fo_td_")]
    carry_cols += [c for c in df.columns if c.startswith("fo_oct_")]
    # preserve order, remove duplicates
    carry_cols = list(dict.fromkeys(carry_cols))

    rows = []
    for event_id, ev in df.groupby("event_id"):
        ev = ev.copy()
        ev = ev[(ev["target_pgv"] > 0) & (ev["distance_m"] > 0)].copy()
        if ev.empty:
            continue
        base = ev.iloc[0]
        row = {"event_id": str(event_id)}
        row["n_sensors"] = int(len(ev))
        row["event_max_pgv"] = float(ev["target_pgv"].max())
        row["event_mean_log_pgv"] = float(ev["target_log"].mean())
        row["event_mean_pgv"] = float(ev["target_pgv"].mean())

        if "track_number_plot" in ev.columns:
            track_num = parse_track_number_numeric(pd.Series([base["track_number_plot"]])).iloc[0]
        else:
            track_num = np.nan
        row["track_number_num"] = float(track_num) if pd.notna(track_num) else np.nan

        mp4 = ev[ev["sensor"].astype(str) == "MP4"]
        if not mp4.empty:
            mp4_row = mp4.iloc[0]
            row["event_mp4_pgv"] = float(mp4_row["target_pgv"])
            row["event_mp4_log_pgv"] = float(mp4_row["target_log"])
            row["event_mp4_distance_m"] = float(mp4_row["distance_m"])
        else:
            row["event_mp4_pgv"] = np.nan
            row["event_mp4_log_pgv"] = np.nan
            row["event_mp4_distance_m"] = np.nan

        n_track = track_n_map.get(int(track_num), np.nan) if pd.notna(track_num) else np.nan
        row["n_track_used"] = n_track
        if np.isfinite(n_track):
            c_rows = ev["target_log"].to_numpy(dtype=float) + n_track * np.log(ev["distance_m"].to_numpy(dtype=float) / float(r0))
            row["c_target"] = float(np.mean(c_rows))
            row["c_target_std"] = float(np.std(c_rows))
            if not mp4.empty:
                row["c_target_mp4"] = float(mp4_row["target_log"] + n_track * np.log(float(mp4_row["distance_m"]) / float(r0)))
            else:
                row["c_target_mp4"] = np.nan
        else:
            row["c_target"] = np.nan
            row["c_target_std"] = np.nan
            row["c_target_mp4"] = np.nan

        if "pred_pgv" in ev.columns:
            pred = ev["pred_pgv"].to_numpy(dtype=float)
            true = ev["target_pgv"].to_numpy(dtype=float)
            mask = np.isfinite(pred) & np.isfinite(true)
            if mask.any():
                row["event_rmse_pgv"] = float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))
                row["event_bias_pgv"] = float(np.mean(pred[mask] - true[mask]))
                row["event_max_abs_error_pgv"] = float(np.max(np.abs(pred[mask] - true[mask])))
            else:
                row["event_rmse_pgv"] = np.nan
                row["event_bias_pgv"] = np.nan
                row["event_max_abs_error_pgv"] = np.nan
            if not mp4.empty and np.isfinite(mp4.iloc[0].get("pred_pgv", np.nan)):
                row["mp4_underprediction"] = float(mp4.iloc[0]["target_pgv"] - mp4.iloc[0]["pred_pgv"])
            else:
                row["mp4_underprediction"] = np.nan
        
        for c in carry_cols:
            row[c] = base.get(c, np.nan)
        rows.append(row)

    ev_df = pd.DataFrame(rows)
    if ev_df.empty:
        return ev_df

    if "train_speed_kmh_plot" in ev_df.columns:
        ev_df["train_speed_kmh_plot"] = pd.to_numeric(ev_df["train_speed_kmh_plot"], errors="coerce")
    if "train_type_code" in ev_df.columns:
        ev_df["train_type_code"] = pd.to_numeric(ev_df["train_type_code"], errors="coerce")
    return ev_df


def maybe_merge_prediction_data(df: pd.DataFrame, pred_data_path: Optional[str], out_dir: Path) -> pd.DataFrame:
    """Optional merge of model predictions for error-based event profile exploration."""
    # If predictions already exist, just standardise names.
    existing_pred = first_existing(df, ["pred_pgv", "predicted_pgv", "pgv_pred", "y_pred_pgv"])
    if existing_pred is not None and existing_pred != "pred_pgv":
        df = df.copy()
        df["pred_pgv"] = pd.to_numeric(df[existing_pred], errors="coerce")
        return df
    if existing_pred == "pred_pgv" or not pred_data_path:
        return df

    pred_path = Path(pred_data_path)
    if not pred_path.exists():
        print(f"[WARN] Prediction file not found, skipping error-based explorations: {pred_path}")
        return df

    if pred_path.suffix.lower() == ".csv":
        pred_df = pd.read_csv(pred_path)
    else:
        pred_df = pd.read_parquet(pred_path)

    event_col = first_existing(pred_df, ["event_id"])
    sensor_col = first_existing(pred_df, ["sensor", "sensor_id"])
    pred_col = first_existing(pred_df, ["pred_pgv", "predicted_pgv", "pgv_pred", "y_pred_pgv", "prediction_pgv"])
    if event_col is None or sensor_col is None or pred_col is None:
        print(
            f"[WARN] Could not infer prediction columns from {pred_path}. "
            f"Need event_id, sensor/sensor_id, and pred_pgv-like column."
        )
        return df

    pred_keep = pred_df[[event_col, sensor_col, pred_col]].copy()
    pred_keep[event_col] = pred_keep[event_col].astype(str)
    pred_keep[sensor_col] = pred_keep[sensor_col].astype(str)
    pred_keep["pred_pgv"] = pd.to_numeric(pred_keep[pred_col], errors="coerce")
    pred_keep = pred_keep.rename(columns={event_col: "event_id", sensor_col: "sensor"})[["event_id", "sensor", "pred_pgv"]]
    pred_keep.to_csv(out_dir / "00_prediction_merge_preview.csv", index=False)

    out = df.merge(pred_keep, on=["event_id", "sensor"], how="left")
    matched = out["pred_pgv"].notna().sum()
    print(f"[INFO] Merged prediction data: matched {matched:,} sensor rows")
    return out


def _track_or_sensor_categories(sub: pd.DataFrame, color_by: str) -> list[str]:
    if color_by == "sensor":
        return sorted(sub["sensor"].astype(str).unique(), key=sensor_plot_key)
    if color_by == "track":
        vals = sub["track_number_plot"].astype(str).unique().tolist()
        return sorted(vals, key=natural_sensor_key)
    raise ValueError(color_by)


def plot_log_attenuation(
    df: pd.DataFrame,
    out_path: Path,
    r0: float = 10.0,
    color_by: str = "sensor",
    title: str = "log(PGV) vs log(distance/r0)",
    filter_track: Optional[int] = None,
    filter_event_ids: Optional[set[str]] = None,
    alpha: float = 0.12,
) -> None:
    needed = ["distance_m", "target_log"]
    if color_by == "sensor":
        needed.append("sensor")
    else:
        needed.append("track_number_plot")
    sub = df[[c for c in set(needed + ["event_id", "track_number_plot", "sensor"]) if c in df.columns]].dropna().copy()
    sub = sub[(sub["distance_m"] > 0)].copy()
    if filter_track is not None:
        sub["track_number_num"] = parse_track_number_numeric(sub["track_number_plot"])
        sub = sub[sub["track_number_num"] == filter_track].copy()
    if filter_event_ids is not None:
        sub = sub[sub["event_id"].astype(str).isin(set(map(str, filter_event_ids)))].copy()
    if sub.empty:
        print(f"[WARN] Empty attenuation plot data: {out_path.name}")
        return

    sub["log_distance_ratio"] = np.log(sub["distance_m"].astype(float) / float(r0))
    cats = _track_or_sensor_categories(sub, color_by=color_by)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for cat in cats:
        if color_by == "sensor":
            ss = sub[sub["sensor"].astype(str) == str(cat)].copy()
        else:
            ss = sub[sub["track_number_plot"].astype(str) == str(cat)].copy()
        if ss.empty:
            continue
        ax.scatter(ss["log_distance_ratio"], ss["target_log"], s=10, alpha=alpha, label=str(cat))
        med = (
            ss.groupby(ss["log_distance_ratio"].round(6))["target_log"]
            .median()
            .reset_index()
            .sort_values("log_distance_ratio")
        )
        ax.plot(med["log_distance_ratio"], med["target_log"], lw=2.0)

    ax.set_xlabel(f"log(distance / {r0:g} m)")
    ax.set_ylabel("log(PGV)")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.35)
    if len(cats) <= 10:
        ax.legend(fontsize=8, ncols=2)
    save_fig(out_path)


def plot_distance_attenuation_exploration(df: pd.DataFrame, event_df: pd.DataFrame, out_dir: Path, r0: float = 10.0, high_pgv_quantile: float = 0.90) -> None:
    plot_log_attenuation(
        df,
        out_path=out_dir / "14_logatten_all_events_by_sensor.png",
        r0=r0,
        color_by="sensor",
        title="All events: log(PGV) vs log(distance / r0), coloured by sensor",
        alpha=0.10,
    )
    plot_log_attenuation(
        df,
        out_path=out_dir / "14b_logatten_all_events_by_track.png",
        r0=r0,
        color_by="track",
        title="All events: log(PGV) vs log(distance / r0), coloured by track",
        alpha=0.10,
    )

    if not event_df.empty and "event_max_pgv" in event_df.columns:
        thr = event_df["event_max_pgv"].quantile(high_pgv_quantile)
        high_ids = set(event_df.loc[event_df["event_max_pgv"] >= thr, "event_id"].astype(str))
        plot_log_attenuation(
            df,
            out_path=out_dir / "14c_logatten_high_pgv_events_by_sensor.png",
            r0=r0,
            color_by="sensor",
            title=f"High-PGV events only (>= q{100*high_pgv_quantile:.0f}): log(PGV) vs log(distance / r0)",
            filter_event_ids=high_ids,
            alpha=0.18,
        )

    plot_log_attenuation(
        df,
        out_path=out_dir / "14d_logatten_track1_by_sensor.png",
        r0=r0,
        color_by="sensor",
        title="Track 1 only: log(PGV) vs log(distance / r0)",
        filter_track=1,
        alpha=0.14,
    )
    plot_log_attenuation(
        df,
        out_path=out_dir / "14e_logatten_track2_by_sensor.png",
        r0=r0,
        color_by="sensor",
        title="Track 2 only: log(PGV) vs log(distance / r0)",
        filter_track=2,
        alpha=0.14,
    )


def plot_event_profile_grid(
    df: pd.DataFrame,
    event_ids: list[str],
    out_path: Path,
    title: str,
    x_mode: str = "distance",
    y_mode: str = "pgv",
    r0: float = 10.0,
) -> None:
    if not event_ids:
        return
    sub = df[df["event_id"].astype(str).isin(list(map(str, event_ids)))].copy()
    if sub.empty:
        return

    n = len(event_ids)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4, 3.2 * nrows)))
    axes = np.array(axes).reshape(-1)

    for ax, event_id in zip(axes, event_ids):
        ev = sub[sub["event_id"].astype(str) == str(event_id)].copy()
        ev = ev[(ev["distance_m"] > 0) & (ev["target_pgv"] > 0)].sort_values("distance_m")
        if ev.empty:
            ax.axis("off")
            continue

        if x_mode == "distance":
            x = ev["distance_m"].to_numpy(dtype=float)
            xlabel = "Distance [m]"
        elif x_mode == "log_distance":
            x = np.log(ev["distance_m"].to_numpy(dtype=float))
            xlabel = "log(Distance [m])"
        elif x_mode == "log_distance_ratio":
            x = np.log(ev["distance_m"].to_numpy(dtype=float) / float(r0))
            xlabel = f"log(distance / {r0:g} m)"
        else:
            raise ValueError(x_mode)

        if y_mode == "pgv":
            y = ev["target_pgv"].to_numpy(dtype=float)
            ylabel = "PGV [mm/s]"
        elif y_mode == "log_pgv":
            y = ev["target_log"].to_numpy(dtype=float)
            ylabel = "log(PGV)"
        else:
            raise ValueError(y_mode)

        ax.plot(x, y, marker="o")
        for _, row in ev.iterrows():
            xi = np.log(float(row["distance_m"])) if x_mode == "log_distance" else (
                np.log(float(row["distance_m"]) / float(r0)) if x_mode == "log_distance_ratio" else float(row["distance_m"])
            )
            yi = float(row["target_log"]) if y_mode == "log_pgv" else float(row["target_pgv"])
            ax.annotate(str(row["sensor"]), (xi, yi), fontsize=7)
        ax.set_title(str(event_id)[:28], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, ls=":", alpha=0.35)

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle(title, fontsize=12)
    save_fig(out_path)


def plot_event_profiles_exploration(
    df: pd.DataFrame,
    event_df: pd.DataFrame,
    out_dir: Path,
    r0: float = 10.0,
    n_events: int = 20,
    random_seed: int = 42,
) -> None:
    if event_df.empty or "event_id" not in event_df.columns:
        return

    # Top high-PGV events.
    top_high = event_df.sort_values("event_max_pgv", ascending=False)["event_id"].astype(str).head(n_events).tolist()
    plot_event_profile_grid(
        df, top_high, out_dir / "15_event_profiles_top20_high_pgv_linear.png",
        "Top high-PGV events: PGV vs distance", x_mode="distance", y_mode="pgv", r0=r0,
    )
    plot_event_profile_grid(
        df, top_high, out_dir / "15b_event_profiles_top20_high_pgv_loglog.png",
        "Top high-PGV events: log(PGV) vs log(distance)", x_mode="log_distance", y_mode="log_pgv", r0=r0,
    )

    # Random normal events: middle band of event_max_pgv.
    q_lo = event_df["event_max_pgv"].quantile(0.30)
    q_hi = event_df["event_max_pgv"].quantile(0.70)
    normal_pool = event_df[(event_df["event_max_pgv"] >= q_lo) & (event_df["event_max_pgv"] <= q_hi)].copy()
    if not normal_pool.empty:
        rng = np.random.default_rng(random_seed)
        normal_ids = normal_pool["event_id"].astype(str).tolist()
        if len(normal_ids) > n_events:
            normal_ids = rng.choice(normal_ids, size=n_events, replace=False).tolist()
        plot_event_profile_grid(
            df, normal_ids, out_dir / "16_event_profiles_random_normal_linear.png",
            "Random normal events: PGV vs distance", x_mode="distance", y_mode="pgv", r0=r0,
        )
        plot_event_profile_grid(
            df, normal_ids, out_dir / "16b_event_profiles_random_normal_loglog.png",
            "Random normal events: log(PGV) vs log(distance)", x_mode="log_distance", y_mode="log_pgv", r0=r0,
        )

    # Optional model-error based event sets.
    if "event_rmse_pgv" in event_df.columns and event_df["event_rmse_pgv"].notna().any():
        worst_err = event_df.sort_values("event_rmse_pgv", ascending=False)["event_id"].astype(str).head(n_events).tolist()
        plot_event_profile_grid(
            df, worst_err, out_dir / "17_event_profiles_largest_model_error_linear.png",
            "Events with largest model RMSE: PGV vs distance", x_mode="distance", y_mode="pgv", r0=r0,
        )
        plot_event_profile_grid(
            df, worst_err, out_dir / "17b_event_profiles_largest_model_error_loglog.png",
            "Events with largest model RMSE: log(PGV) vs log(distance)", x_mode="log_distance", y_mode="log_pgv", r0=r0,
        )

    if "mp4_underprediction" in event_df.columns and event_df["mp4_underprediction"].notna().any():
        worst_mp4 = event_df.sort_values("mp4_underprediction", ascending=False)["event_id"].astype(str).head(n_events).tolist()
        plot_event_profile_grid(
            df, worst_mp4, out_dir / "18_event_profiles_largest_mp4_underprediction_linear.png",
            "Events with largest MP4 underprediction: PGV vs distance", x_mode="distance", y_mode="pgv", r0=r0,
        )
        plot_event_profile_grid(
            df, worst_mp4, out_dir / "18b_event_profiles_largest_mp4_underprediction_loglog.png",
            "Events with largest MP4 underprediction: log(PGV) vs log(distance)", x_mode="log_distance", y_mode="log_pgv", r0=r0,
        )




def find_latest_waveform_build(waveform_root: Path, waveform_glob: str = "holten_waveform_v0*") -> Optional[Path]:
    """Find latest waveform build containing waveforms.npy and event_index.parquet."""
    if not waveform_root.exists():
        return None
    builds = [p for p in sorted(waveform_root.glob(waveform_glob), key=lambda p: p.name) if p.is_dir()]
    builds = [p for p in builds if (p / "waveforms.npy").exists() and (p / "event_index.parquet").exists()]
    if not builds:
        return None
    return builds[-1]


def _safe_feature_ratio(a: float, b: float, eps: float = 1e-30) -> float:
    return float(a / (b + eps))


def compute_waveform_features_for_block(block: np.ndarray, fs_hz: float = 250.0) -> dict:
    """
    Compute compact event-level descriptors from one FO waveform tensor.

    Accepts v1 single-channel shape (T,) or v2/v3 multi-channel shape (C, T).
    The features are deliberately simple and physically interpretable:
    amplitude/energy, channel-energy concentration, spectral centroid, and
    high/low frequency energy ratio.
    """
    x = np.asarray(block, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected waveform shape (T,) or (C,T), got {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    n_ch, n_t = x.shape
    if n_t < 4:
        return {}

    x2 = x * x
    absx = np.abs(x)
    ch_energy = x2.sum(axis=1)
    ch_rms = np.sqrt(np.mean(x2, axis=1))
    total_energy = float(ch_energy.sum())
    global_rms = float(np.sqrt(np.mean(x2)))
    center_pos = n_ch // 2

    # Temporal envelope from all channels combined.
    env = np.sqrt(np.mean(x2, axis=0))
    env_sum = float(env.sum())
    if env_sum > 0:
        t_idx = np.arange(n_t, dtype=np.float64)
        temporal_centroid_s = float(np.dot(t_idx, env) / env_sum / fs_hz)
        temporal_spread_s = float(np.sqrt(np.dot((t_idx / fs_hz - temporal_centroid_s) ** 2, env) / env_sum))
    else:
        temporal_centroid_s = np.nan
        temporal_spread_s = np.nan

    # Spatial energy distribution across channels. Channel pitch differs between
    # builds, but index-based features still capture whether energy is centred
    # or shifted within the stored aperture.
    ch_pos = np.arange(n_ch, dtype=np.float64) - float(center_pos)
    if total_energy > 0:
        spatial_centroid_ch = float(np.dot(ch_pos, ch_energy) / total_energy)
        spatial_spread_ch = float(np.sqrt(np.dot((ch_pos - spatial_centroid_ch) ** 2, ch_energy) / total_energy))
    else:
        spatial_centroid_ch = np.nan
        spatial_spread_ch = np.nan

    # Spectrum aggregated across channels. The waveforms are already bandpassed
    # and resampled in the build scripts, but these features check whether high
    # intensity is associated with distinct frequency content.
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fs_hz)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    p_freq = spec.mean(axis=0)
    valid = (freqs >= 1.0) & (freqs <= min(100.0, 0.5 * fs_hz))
    total_spec = float(p_freq[valid].sum()) if valid.any() else 0.0

    def band_energy(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(p_freq[m].sum()) if m.any() else 0.0

    low_1_20 = band_energy(1.0, 20.0)
    mid_20_40 = band_energy(20.0, 40.0)
    high_40_100 = band_energy(40.0, min(100.0, 0.5 * fs_hz) + 1e-9)
    high_20_100 = mid_20_40 + high_40_100
    if total_spec > 0:
        spectral_centroid = float(np.sum(freqs[valid] * p_freq[valid]) / total_spec)
        p_norm = p_freq[valid] / total_spec
        spectral_entropy = float(-np.sum(p_norm * np.log(np.clip(p_norm, 1e-30, None))) / np.log(len(p_norm))) if len(p_norm) > 1 else np.nan
    else:
        spectral_centroid = np.nan
        spectral_entropy = np.nan

    return {
        "wf_n_channels": int(n_ch),
        "wf_n_samples": int(n_t),
        "wf_global_energy": total_energy,
        "wf_global_mean_square": float(np.mean(x2)),
        "wf_global_rms": global_rms,
        "wf_global_abs_mean": float(absx.mean()),
        "wf_global_abs_max": float(absx.max()),
        "wf_global_crest_factor": _safe_feature_ratio(float(absx.max()), global_rms),
        "wf_ch_energy_max": float(ch_energy.max()),
        "wf_ch_energy_mean": float(ch_energy.mean()),
        "wf_ch_energy_std": float(ch_energy.std()),
        "wf_ch_energy_max_over_mean": _safe_feature_ratio(float(ch_energy.max()), float(ch_energy.mean())),
        "wf_ch_rms_max": float(ch_rms.max()),
        "wf_ch_rms_mean": float(ch_rms.mean()),
        "wf_ch_rms_std": float(ch_rms.std()),
        "wf_center_ch_rms": float(ch_rms[center_pos]),
        "wf_center_ch_energy": float(ch_energy[center_pos]),
        "wf_max_energy_channel_offset": float(ch_pos[int(np.argmax(ch_energy))]),
        "wf_spatial_centroid_ch": spatial_centroid_ch,
        "wf_spatial_spread_ch": spatial_spread_ch,
        "wf_temporal_centroid_s": temporal_centroid_s,
        "wf_temporal_spread_s": temporal_spread_s,
        "wf_spectral_centroid": spectral_centroid,
        "wf_spectral_entropy": spectral_entropy,
        "wf_band_energy_1_20hz": low_1_20,
        "wf_band_energy_20_40hz": mid_20_40,
        "wf_band_energy_40_100hz": high_40_100,
        "wf_high_low_energy_ratio": _safe_feature_ratio(high_20_100, low_1_20),
        "wf_mid_low_energy_ratio": _safe_feature_ratio(mid_20_40, low_1_20),
        "wf_high_total_energy_fraction": _safe_feature_ratio(high_20_100, total_spec),
    }


def load_waveform_feature_dataframe(
    waveform_build: Path,
    fs_hz: float = 250.0,
    max_events: int = 0,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load waveforms.npy/event_index.parquet and compute event-level features."""
    waveform_build = Path(waveform_build)
    waveforms_path = waveform_build / "waveforms.npy"
    index_path = waveform_build / "event_index.parquet"
    if not waveforms_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"Waveform build must contain waveforms.npy and event_index.parquet: {waveform_build}")

    print(f"[INFO] Loading waveform build: {waveform_build}")
    waveforms = np.load(waveforms_path, mmap_mode="r")
    index_df = pd.read_parquet(index_path)
    if "event_id" not in index_df.columns:
        raise KeyError(f"event_index.parquet has no event_id column: {index_path}")

    if "build_status" in index_df.columns:
        index_df = index_df[index_df["build_status"].astype(str).str.lower() == "ok"].copy()
    if "waveform_row_idx" not in index_df.columns:
        index_df = index_df.copy()
        index_df["waveform_row_idx"] = np.arange(len(index_df))

    index_df["event_id"] = index_df["event_id"].astype(str)
    index_df["waveform_row_idx"] = pd.to_numeric(index_df["waveform_row_idx"], errors="coerce")
    index_df = index_df.dropna(subset=["waveform_row_idx"]).copy()
    index_df["waveform_row_idx"] = index_df["waveform_row_idx"].astype(int)
    index_df = index_df[(index_df["waveform_row_idx"] >= 0) & (index_df["waveform_row_idx"] < waveforms.shape[0])].copy()
    if max_events and max_events > 0:
        index_df = index_df.head(max_events).copy()

    rows = []
    total = len(index_df)
    for i, row in enumerate(index_df.itertuples(index=False), start=1):
        if i % 250 == 0 or i == total:
            print(f"[INFO] Waveform features: {i:,}/{total:,}", end="\r")
        eid = str(getattr(row, "event_id"))
        ridx = int(getattr(row, "waveform_row_idx"))
        try:
            feats = compute_waveform_features_for_block(waveforms[ridx], fs_hz=fs_hz)
            if not feats:
                continue
            feats["event_id"] = eid
            feats["waveform_row_idx"] = ridx
            rows.append(feats)
        except Exception as exc:
            rows.append({"event_id": eid, "waveform_row_idx": ridx, "waveform_feature_error": str(exc)})
    print()

    wf_df = pd.DataFrame(rows)
    if out_dir is not None:
        wf_df.to_csv(out_dir / "20_waveform_event_features.csv", index=False)
        pd.DataFrame([
            {
                "waveform_build": str(waveform_build),
                "waveforms_shape": str(tuple(waveforms.shape)),
                "event_index_rows_ok": int(len(index_df)),
                "features_rows": int(len(wf_df)),
                "fs_hz_assumed": float(fs_hz),
            }
        ]).to_csv(out_dir / "20_waveform_feature_metadata.csv", index=False)
    return wf_df


def merge_waveform_features_into_events(
    event_df: pd.DataFrame,
    waveform_root: Path,
    waveform_build_arg: Optional[str],
    waveform_glob: str,
    fs_hz: float,
    max_events: int,
    out_dir: Path,
) -> pd.DataFrame:
    """Merge computed waveform descriptors into the event-level dataframe."""
    if event_df.empty:
        return event_df

    if waveform_build_arg is None or str(waveform_build_arg).lower() in {"", "none", "skip", "false"}:
        print("[INFO] Waveform feature extraction skipped.")
        return event_df

    if str(waveform_build_arg).lower() in {"latest", "auto"}:
        build = find_latest_waveform_build(waveform_root, waveform_glob=waveform_glob)
        if build is None:
            print(f"[WARN] No waveform build found under {waveform_root} with glob {waveform_glob!r}; continuing without waveform features.")
            return event_df
    else:
        build = Path(waveform_build_arg)

    wf_df = load_waveform_feature_dataframe(build, fs_hz=fs_hz, max_events=max_events, out_dir=out_dir)
    if wf_df.empty:
        print("[WARN] Waveform feature dataframe is empty; continuing without waveform features.")
        return event_df

    out = event_df.merge(wf_df, on="event_id", how="left", suffixes=("", "_wf"))
    matched = out["wf_global_rms"].notna().sum() if "wf_global_rms" in out.columns else 0
    print(f"[INFO] Merged waveform features into event table: {matched:,}/{len(out):,} events matched")
    return out

def event_feature_candidates(ev_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    explicit_numeric = [
        c for c in [
            "train_speed_kmh_plot",
            "track_number_num",
            "train_type_code",
            "wf_global_energy",
            "wf_ch_energy_max",
            "wf_global_rms",
            "wf_high_low_energy_ratio",
            "wf_spectral_centroid",
        ] if c in ev_df.columns
    ]
    wf_cols = [c for c in ev_df.columns if c.startswith("wf_")]
    fo_td_cols = [c for c in ev_df.columns if c.startswith("fo_td_")]
    fo_oct_cols = [c for c in ev_df.columns if c.startswith("fo_oct_")]
    numeric_candidates = list(dict.fromkeys(explicit_numeric + wf_cols + fo_td_cols + fo_oct_cols))
    numeric_candidates = [c for c in numeric_candidates if pd.api.types.is_numeric_dtype(ev_df[c])]
    category_candidates = [c for c in ["track_number_plot", "train_type_group", "train_type_plot"] if c in ev_df.columns]
    return numeric_candidates, category_candidates, fo_oct_cols


def plot_top_correlations_for_target(ev_df: pd.DataFrame, target_col: str, feature_cols: list[str], out_path: Path, top_n: int = 25) -> pd.Series:
    valid = [c for c in feature_cols if c in ev_df.columns]
    if target_col not in ev_df.columns or not valid:
        return pd.Series(dtype=float)
    data = ev_df[[target_col] + valid].copy()
    corr = data.corr(numeric_only=True)[target_col].drop(labels=[target_col], errors="ignore")
    corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
    corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)
    top = corr.head(top_n)
    if top.empty:
        return corr
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(top))))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_xlabel(f"Pearson correlation with {target_col}")
    ax.set_title(f"Top {len(top)} correlations with {target_col}")
    ax.axvline(0.0, lw=0.8)
    save_fig(out_path)
    return corr


def plot_event_target_vs_features_grid(
    ev_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    out_path: Path,
    high_flag_col: str = "high_intensity_flag",
) -> None:
    if target_col not in ev_df.columns or not feature_cols:
        return
    n = len(feature_cols)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 3.4 * nrows)))
    axes = np.array(axes).reshape(-1)
    for ax, feat in zip(axes, feature_cols):
        sub = ev_df[[feat, target_col, high_flag_col]].dropna().copy() if high_flag_col in ev_df.columns else ev_df[[feat, target_col]].dropna().copy()
        if sub.empty:
            ax.axis("off")
            continue
        if high_flag_col in sub.columns:
            normal = sub[~sub[high_flag_col].astype(bool)]
            high = sub[sub[high_flag_col].astype(bool)]
            ax.scatter(normal[feat], normal[target_col], s=14, alpha=0.35, label="normal")
            ax.scatter(high[feat], high[target_col], s=18, alpha=0.55, label="high-intensity")
        else:
            ax.scatter(sub[feat], sub[target_col], s=14, alpha=0.35)
        ax.set_xlabel(feat)
        ax.set_ylabel(target_col)
        ax.grid(True, ls=":", alpha=0.35)
    for ax in axes[n:]:
        ax.axis("off")
    if high_flag_col in ev_df.columns and n > 0:
        axes[0].legend(fontsize=8)
    plt.suptitle(f"{target_col} vs selected event-level features", fontsize=12)
    save_fig(out_path)


def plot_high_intensity_feature_boxplots(ev_df: pd.DataFrame, feature_cols: list[str], out_path: Path, high_flag_col: str = "high_intensity_flag") -> None:
    if high_flag_col not in ev_df.columns or not feature_cols:
        return
    n = len(feature_cols)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 3.0 * nrows)))
    axes = np.array(axes).reshape(-1)
    for ax, feat in zip(axes, feature_cols):
        sub = ev_df[[feat, high_flag_col]].dropna().copy()
        if sub.empty:
            ax.axis("off")
            continue
        normal = sub.loc[~sub[high_flag_col].astype(bool), feat].to_numpy(dtype=float)
        high = sub.loc[sub[high_flag_col].astype(bool), feat].to_numpy(dtype=float)
        arrays = [normal, high]
        ax.boxplot(arrays, tick_labels=[f"normal\n(n={len(normal)})", f"high\n(n={len(high)})"], showfliers=True)
        ax.set_title(feat, fontsize=9)
        ax.grid(True, axis="y", ls=":", alpha=0.35)
    for ax in axes[n:]:
        ax.axis("off")
    plt.suptitle("Feature distributions: normal vs high-intensity events", fontsize=12)
    save_fig(out_path)


def plot_event_intensity_exploration(ev_df: pd.DataFrame, out_dir: Path, high_pgv_quantile: float = 0.90) -> None:
    if ev_df.empty:
        return
    ev_df = ev_df.copy()
    # Define high intensity flag for separation plots.
    thr = ev_df["event_max_pgv"].quantile(high_pgv_quantile)
    ev_df["high_intensity_flag"] = ev_df["event_max_pgv"] >= thr
    ev_df.to_csv(out_dir / "20_event_level_dataframe.csv", index=False)

    numeric_features, category_features, fo_oct_cols = event_feature_candidates(ev_df)
    pd.DataFrame({"numeric_feature": numeric_features}).to_csv(out_dir / "20_available_event_numeric_features.csv", index=False)
    pd.DataFrame({"category_feature": category_features}).to_csv(out_dir / "20_available_event_category_features.csv", index=False)

    target_cols = [c for c in ["event_max_pgv", "event_mp4_pgv", "event_mean_log_pgv", "c_target"] if c in ev_df.columns]
    corr_frames = []
    top_features_union = set()
    for target_col in target_cols:
        corr = plot_top_correlations_for_target(
            ev_df,
            target_col=target_col,
            feature_cols=numeric_features,
            out_path=out_dir / f"20_corr_top_{target_col}.png",
            top_n=25,
        )
        if not corr.empty:
            corr_frames.append(pd.DataFrame({"target_metric": target_col, "feature": corr.index, "pearson_corr": corr.values}))
            top_features_union.update(corr.head(8).index.tolist())

    if corr_frames:
        pd.concat(corr_frames, ignore_index=True).to_csv(out_dir / "20_event_feature_correlations_long.csv", index=False)

    # Use the union of top-correlated features for scatter/separation views.
    priority_for_grids = [
        "train_speed_kmh_plot",
        "track_number_num",
        "train_type_code",
        "wf_global_energy",
        "wf_ch_energy_max",
        "wf_global_rms",
        "wf_high_low_energy_ratio",
        "wf_spectral_centroid",
        "fo_td_max_abs_mean",
        "fo_td_rms_mean",
        "fo_td_std_mean",
    ]
    selected = [c for c in priority_for_grids if c in ev_df.columns and c in set(numeric_features)]
    # Fill up with top-correlated features not already selected.
    for c in list(top_features_union):
        if c not in selected and c in numeric_features:
            selected.append(c)
    selected = selected[:10]

    if selected:
        if "event_max_pgv" in ev_df.columns:
            plot_event_target_vs_features_grid(ev_df, "event_max_pgv", selected, out_dir / "21_event_max_pgv_vs_selected_features.png")
        if "c_target" in ev_df.columns:
            plot_event_target_vs_features_grid(ev_df, "c_target", selected, out_dir / "21b_c_target_vs_selected_features.png")
        plot_high_intensity_feature_boxplots(ev_df, selected, out_dir / "22_high_intensity_feature_boxplots.png")

    # Simple contextual boxplots for categorical event descriptors.
    for cat_col in category_features:
        for target_col in ["event_max_pgv", "event_mp4_pgv", "c_target"]:
            if target_col not in ev_df.columns or cat_col not in ev_df.columns:
                continue
            sub = ev_df[[cat_col, target_col]].dropna().copy()
            if sub.empty:
                continue
            counts = sub[cat_col].value_counts()
            cats = counts[counts >= 3].index.tolist()
            if cat_col == "train_type_group":
                cats = [c for c in TRAIN_TYPE_GROUP_ORDER if c in cats]
            elif cat_col == "track_number_plot":
                cats = sorted(cats, key=natural_sensor_key)
            else:
                cats = cats[:12]
            arrays = [sub.loc[sub[cat_col] == c, target_col].to_numpy(dtype=float) for c in cats]
            labels = [f"{c}\n(n={len(a)})" for c, a in zip(cats, arrays)]
            fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels)), 5))
            ax.boxplot(arrays, tick_labels=labels, showfliers=True)
            ax.set_ylabel(target_col)
            ax.set_title(f"{target_col} by {cat_col}")
            ax.tick_params(axis="x", rotation=45)
            save_fig(out_dir / f"23_{target_col}_by_{cat_col}.png")

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
    parser.add_argument(
        "--r0",
        type=float,
        default=10.0,
        help="Reference distance r0 [m] used in log(distance / r0) and c_target.",
    )
    parser.add_argument(
        "--high-pgv-quantile",
        type=float,
        default=0.90,
        help="Quantile threshold used to define high-PGV / high-intensity events.",
    )
    parser.add_argument(
        "--profile-n-events",
        type=int,
        default=20,
        help="Number of events shown in each event-profile family.",
    )
    parser.add_argument(
        "--pred-data",
        type=str,
        default=None,
        help="Optional CSV/Parquet with event_id, sensor/sensor_id, and pred_pgv to enable error-based event-profile plots.",
    )
    parser.add_argument(
        "--waveform-build",
        type=str,
        default="latest",
        help=(
            "Waveform build folder to use for event-level waveform features. "
            "Use 'latest' to auto-discover, or 'skip' to disable."
        ),
    )
    parser.add_argument(
        "--waveform-root",
        type=str,
        default=str(default_root() / "holten_waveform"),
        help="Root containing holten_waveform_v* build folders.",
    )
    parser.add_argument(
        "--waveform-glob",
        type=str,
        default="holten_waveform_v0*",
        help="Glob used when --waveform-build latest is selected.",
    )
    parser.add_argument(
        "--waveform-fs",
        type=float,
        default=250.0,
        help="Sampling frequency [Hz] of stored waveform tensors after preprocessing.",
    )
    parser.add_argument(
        "--waveform-max-events",
        type=int,
        default=0,
        help="Optional debug limit for waveform feature extraction. 0 = all events.",
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
    df = maybe_merge_prediction_data(df, pred_data_path=args.pred_data, out_dir=out_dir)
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

    # Physics-linked attenuation exploration and event-level intensity analysis.
    track_n_map, track_n_df = fit_track_attenuation_exponents(df, r0=args.r0)
    track_n_df.to_csv(out_dir / "14_track_attenuation_exponents.csv", index=False)
    event_df = build_event_level_dataframe(df, track_n_map=track_n_map, r0=args.r0)
    if not event_df.empty:
        event_df = merge_waveform_features_into_events(
            event_df=event_df,
            waveform_root=Path(args.waveform_root),
            waveform_build_arg=args.waveform_build,
            waveform_glob=args.waveform_glob,
            fs_hz=args.waveform_fs,
            max_events=args.waveform_max_events,
            out_dir=out_dir,
        )
        plot_distance_attenuation_exploration(df, event_df, out_dir, r0=args.r0, high_pgv_quantile=args.high_pgv_quantile)
        plot_event_profiles_exploration(df, event_df, out_dir, r0=args.r0, n_events=args.profile_n_events)
        plot_event_intensity_exploration(event_df, out_dir, high_pgv_quantile=args.high_pgv_quantile)

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
    print("  14_track_attenuation_exponents.csv")
    print("  14_logatten_all_events_by_sensor.png")
    print("  15_event_profiles_top20_high_pgv_linear.png")
    print("  20_event_level_dataframe.csv")
    print("  20_waveform_event_features.csv")
    print("  20_waveform_feature_metadata.csv")
    print("  20_corr_top_c_target.png")
    print("  21b_c_target_vs_selected_features.png")
    print("  00_train_type_group_mapping.csv")
    print("  06b_target_log_vs_train_speed_by_sensor.png")
    print("  07b_target_log_vs_distance_by_sensor.png")


if __name__ == "__main__":
    main()
