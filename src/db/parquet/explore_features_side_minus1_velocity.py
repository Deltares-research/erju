"""explore_features.py — Feature distribution & analysis script for parquet v2.

Run from the project root:
    python src/db/parquet/explore_features_side_minus1_velocity.py

Analyses produced:
  1. Distributions
     1A  target raw + log
     1B  geometry features
     1C  train metadata
     1D  FO time-domain (3x3 grid)
     1E  FO octave bands (3 files: mean / max / std)
  2. Correlation analysis (Spearman)
     2A  feature-target bar chart (top N)
     2B  feature-feature heatmap (top N by importance)
  3. Feature importance vs Spearman correlation scatter
  4. Target vs distance scatter (colored by train_type / train_speed)
  5. Seaborn pair plot — top N features by importance
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import matplotlib

matplotlib.use("Agg")  # headless backend – must be before pyplot import

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns


# ── Paths ─────────────────────────────────────────────────────────────────────
PARQUET_PATH = Path(
    r"P:\11210978-erju-ai\holten_parquet\parquet_v002_20260408_151746\dataset.parquet"
)
IMPORTANCE_CSV = Path(
    r"P:\11210978-erju-ai\holten_models\xgb_v004_20260408_165448\feature_importance.csv"
)
TRAIN_TYPE_GROUPING_JSON = Path(
    r"d:\codes\erju\docs\TRAIN_TYPE_GROUPING_PROPOSAL_V1.json"
)
SITE_CONFIG_JSON = Path(r"d:\codes\erju\sites\holten.json")

# Filter for this variant
FILTER_SIDE_OF_TRACK = -1
FILTER_SENSOR_UNIT = "velocity_mms"

# Analysis flags — set False to skip
RUN_1_DISTRIBUTIONS = True
RUN_2_CORRELATIONS = True
RUN_3_IMPORTANCE_CORR = True
RUN_4_TARGET_DISTANCE = True
RUN_5_PAIRPLOT = True
RUN_6_MP4_DIAGNOSTICS = True

# Pair-plot / heatmap top-N
TOP_N = 8

# ── Output dirs ──────────────────────────────────────────────────────────────
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUT = PARQUET_PATH.parent / "plots" / "side_minus1_velocity_mms" / f"run_{RUN_TAG}"
DIST_DIR = BASE_OUT / "distributions"
CORR_DIR = BASE_OUT / "correlations"
TGTD_DIR = BASE_OUT / "target_analysis"

for _d in (DIST_DIR, CORR_DIR, TGTD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
print(f"Output run folder: {BASE_OUT}")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading parquet …")
df = pd.read_parquet(PARQUET_PATH)
TARGET = "target_pgv_z_mms"
print(f"  {df.shape[0]:,} rows  x  {df.shape[1]} columns")

if not SITE_CONFIG_JSON.exists():
    raise FileNotFoundError(f"Site config not found: {SITE_CONFIG_JSON}")

with open(SITE_CONFIG_JSON, "r", encoding="utf-8") as f:
    _site_cfg = json.load(f)

sensor_unit_map = _site_cfg.get("accelerometer", {}).get("sensor_data_unit", {})
velocity_sensor_ids = {
    sid for sid, unit in sensor_unit_map.items() if str(unit) == FILTER_SENSOR_UNIT
}

rows_before_filter = len(df)
df = df[
    (df["acc_side_of_track"] == FILTER_SIDE_OF_TRACK)
    & (df["sensor_id"].isin(velocity_sensor_ids))
].copy()
print(
    "Applied filters: "
    f"acc_side_of_track == {FILTER_SIDE_OF_TRACK}, "
    f"sensor unit == {FILTER_SENSOR_UNIT}"
)
print(
    f"  Rows after filter: {len(df):,} "
    f"(kept {100.0 * len(df) / max(rows_before_filter, 1):.1f}% of parquet rows)"
)
if len(df) == 0:
    raise ValueError(
        "Filtered dataset is empty. Check filter values and source parquet."
    )

# Derive event datetime/week from event_id (e.g. 20240829_080626.mat).
event_id_clean = df["event_id"].astype(str).str.replace(".mat", "", regex=False)
event_ts_str = event_id_clean.str.slice(0, 15)
df["event_dt"] = pd.to_datetime(event_ts_str, format="%Y%m%d_%H%M%S", errors="coerce")
iso_cal = df["event_dt"].dt.isocalendar()
df["event_week"] = (
    iso_cal["year"].astype(str) + "-W" + iso_cal["week"].astype(str).str.zfill(2)
)


def _load_train_type_group_mapping(path: Path) -> tuple[dict[str, str], list[str]]:
    """Return raw train_type -> group mapping and preferred group order."""
    if not path.exists():
        print(f"WARNING: grouping JSON not found: {path}")
        return {}, []

    payload = pd.read_json(path, typ="series")
    if "group_to_train_types" not in payload:
        print(f"WARNING: 'group_to_train_types' missing in grouping JSON: {path}")
        return {}, []

    group_to_types = payload["group_to_train_types"]
    mapping: dict[str, str] = {}
    for group_name, values in group_to_types.items():
        for raw_type in values:
            mapping[str(raw_type)] = str(group_name)

    group_order = [str(g) for g in payload.get("group_priority_order", [])]
    return mapping, group_order


TRAIN_TYPE_TO_GROUP, TRAIN_GROUP_ORDER = _load_train_type_group_mapping(
    TRAIN_TYPE_GROUPING_JSON
)
if TRAIN_TYPE_TO_GROUP:
    df["train_type_group"] = (
        df["train_type"].astype(str).map(TRAIN_TYPE_TO_GROUP).fillna("OTHER")
    )
    print(
        f"Loaded train_type grouping: {len(TRAIN_TYPE_TO_GROUP)} labels -> "
        f"{df['train_type_group'].nunique()} groups"
    )
else:
    df["train_type_group"] = "OTHER"
    print("No train_type grouping loaded; defaulting all rows to group 'OTHER'.")

PRESENT_GROUPS = sorted(df["train_type_group"].dropna().unique())
if TRAIN_GROUP_ORDER:
    TRAIN_GROUPS_PRESENT = [g for g in TRAIN_GROUP_ORDER if g in PRESENT_GROUPS]
    TRAIN_GROUPS_PRESENT += sorted(set(PRESENT_GROUPS).difference(TRAIN_GROUPS_PRESENT))
else:
    TRAIN_GROUPS_PRESENT = PRESENT_GROUPS
TRAIN_GROUP_COLOR_MAP = dict(
    zip(TRAIN_GROUPS_PRESENT, sns.color_palette("tab10", len(TRAIN_GROUPS_PRESENT)))
)

# Distance-group metadata (exact sensor distances in this filtered set).
DISTANCE_GROUP_VALUES = sorted(df["acc_distance_to_track_m"].dropna().unique())
DISTANCE_GROUP_LABELS = [f"{float(v):g}m" for v in DISTANCE_GROUP_VALUES]
DISTANCE_VALUE_TO_LABEL = dict(zip(DISTANCE_GROUP_VALUES, DISTANCE_GROUP_LABELS))
df["distance_group"] = df["acc_distance_to_track_m"].map(DISTANCE_VALUE_TO_LABEL)
DISTANCE_GROUP_COLOR_MAP = dict(
    zip(DISTANCE_GROUP_LABELS, sns.color_palette("viridis", len(DISTANCE_GROUP_LABELS)))
)

# ── Octave band ordering (low to high frequency) ─────────────────────────────
BAND_ORDER = [
    "1_00hz",
    "1_25hz",
    "1_60hz",
    "2_00hz",
    "2_50hz",
    "3_15hz",
    "4_00hz",
    "5_00hz",
    "6_30hz",
    "8_00hz",
    "010hz",
    "012hz",
    "016hz",
    "020hz",
    "025hz",
    "031hz",
    "040hz",
    "050hz",
    "063hz",
    "080hz",
    "100hz",
]
BAND_LABELS = [
    "1",
    "1.25",
    "1.6",
    "2",
    "2.5",
    "3.15",
    "4",
    "5",
    "6.3",
    "8",
    "10",
    "12.5",
    "16",
    "20",
    "25",
    "31.5",
    "40",
    "50",
    "63",
    "80",
    "100",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _savefig(fig: plt.Figure, subdir: Path, name: str) -> None:
    p = subdir / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.relative_to(BASE_OUT)}")


def _log_hist(
    ax: plt.Axes,
    vals: pd.Series,
    color: str,
    xlabel: str,
    title: str,
    log_x: bool = False,
) -> None:
    v = vals.dropna()
    if log_x:
        v = v[v > 0]
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 50)
        ax.set_xscale("log")
    else:
        bins = 50
    ax.hist(v, bins=bins, color=color, edgecolor="white", linewidth=0.2)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def _group_hist(
    ax: plt.Axes,
    data: pd.DataFrame,
    col: str,
    xlabel: str,
    title: str,
    log_x: bool = False,
    legend: bool = False,
    hue_col: str = "train_type_group",
    hue_order: list[str] | None = None,
    palette: dict[str, tuple[float, float, float]] | None = None,
    legend_title: str = "train_type_group",
) -> None:
    sub = data[[col, hue_col]].dropna().copy()
    if log_x:
        sub = sub[sub[col] > 0]
    if sub.empty:
        ax.set_title(f"{title} (no data)", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        return

    sns.histplot(
        data=sub,
        x=col,
        hue=hue_col,
        hue_order=hue_order,
        palette=palette,
        element="step",
        fill=False,
        stat="density",
        common_norm=False,
        bins=50,
        ax=ax,
        legend=legend,
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    if ax.legend_ is not None:
        ax.legend_.set_title(legend_title)
        for txt in ax.legend_.get_texts():
            txt.set_fontsize(7)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Distributions
# ══════════════════════════════════════════════════════════════════════════════
if RUN_1_DISTRIBUTIONS:
    print("\n=== Analysis 1: Distributions ===")

    # ── 1A  Target ────────────────────────────────────────────────────────────
    print("[1A] Target distribution …")
    vals = df[TARGET].dropna()

    # 1A-1: Overall (single combined distribution)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Target: PGV-Z (mm/s) — overall",
        fontsize=13,
        fontweight="bold",
    )
    _log_hist(axes[0], vals, "steelblue", "PGV-Z (mm/s)", "Raw values")
    log_vals = np.log1p(vals)
    axes[1].hist(
        log_vals, bins=50, color="darkorange", edgecolor="white", linewidth=0.2
    )
    axes[1].set_xlabel("log1p(PGV-Z)", fontsize=8)
    axes[1].set_ylabel("Count", fontsize=8)
    axes[1].set_title("log1p-transformed", fontsize=9)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(axis="y", alpha=0.3)
    for ax, v in zip(axes, [vals, log_vals]):
        ax.axvline(
            v.median(),
            color="red",
            linewidth=1.2,
            linestyle="--",
            label=f"median={v.median():.3f}",
        )
        ax.legend(fontsize=7)
    fig.tight_layout()
    _savefig(fig, DIST_DIR, "01A_target_distribution_overall.png")

    # 1A-2: All groups overlaid in one plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Target: PGV-Z (mm/s) — train_type_group overlay",
        fontsize=13,
        fontweight="bold",
    )
    _group_hist(
        axes[0],
        df,
        TARGET,
        "PGV-Z (mm/s)",
        "Raw values by train_type_group",
        legend=True,
    )
    tmp = df[[TARGET, "train_type_group", "distance_group"]].dropna().copy()
    tmp["log_target"] = np.log1p(tmp[TARGET])
    _group_hist(
        axes[1],
        tmp,
        "log_target",
        "log1p(PGV-Z)",
        "log1p-transformed by train_type_group",
        legend=False,
    )
    fig.tight_layout()
    _savefig(fig, DIST_DIR, "01A_target_distribution_group_overlay.png")

    # 1A-3: One panel per group (readable comparison)
    n_groups = len(TRAIN_GROUPS_PRESENT)
    all_raw = df[TARGET].dropna()
    all_log = np.log1p(all_raw)

    raw_xmin, raw_xmax = float(all_raw.min()), float(all_raw.max())
    log_xmin, log_xmax = float(all_log.min()), float(all_log.max())
    raw_bins = np.linspace(raw_xmin, raw_xmax, 50)
    log_bins = np.linspace(log_xmin, log_xmax, 50)

    # Compute global y-limits so all panels are directly comparable.
    raw_ymax = 1.0
    log_ymax = 1.0
    for grp in TRAIN_GROUPS_PRESENT:
        gvals = df.loc[df["train_type_group"] == grp, TARGET].dropna().to_numpy()
        if gvals.size == 0:
            continue
        raw_counts, _ = np.histogram(gvals, bins=raw_bins)
        log_counts, _ = np.histogram(np.log1p(gvals), bins=log_bins)
        raw_ymax = max(raw_ymax, float(raw_counts.max()))
        log_ymax = max(log_ymax, float(log_counts.max()))

    fig, axes = plt.subplots(
        n_groups,
        2,
        figsize=(12, max(3.0 * n_groups, 8.0)),
        sharex="col",
        sharey="col",
    )
    if n_groups == 1:
        axes = np.array([axes])
    fig.suptitle(
        "Target: PGV-Z (mm/s) — one row per train_type_group",
        fontsize=13,
        fontweight="bold",
    )
    for row_idx, grp in enumerate(TRAIN_GROUPS_PRESENT):
        gvals = df.loc[df["train_type_group"] == grp, TARGET].dropna()
        clr = TRAIN_GROUP_COLOR_MAP[grp]

        axes[row_idx, 0].hist(
            gvals,
            bins=raw_bins,
            color=clr,
            edgecolor="white",
            linewidth=0.2,
        )
        axes[row_idx, 0].set_xlabel("PGV-Z (mm/s)", fontsize=8)
        axes[row_idx, 0].set_ylabel("Count", fontsize=8)
        axes[row_idx, 0].set_title(f"{grp} — raw (n={len(gvals):,})", fontsize=9)
        axes[row_idx, 0].tick_params(labelsize=7)
        axes[row_idx, 0].grid(axis="y", alpha=0.3)
        axes[row_idx, 0].set_xlim(raw_xmin, raw_xmax)
        axes[row_idx, 0].set_ylim(0, raw_ymax * 1.05)

        glog = np.log1p(gvals)
        axes[row_idx, 1].hist(
            glog,
            bins=log_bins,
            color=clr,
            edgecolor="white",
            linewidth=0.2,
        )
        axes[row_idx, 1].set_xlabel("log1p(PGV-Z)", fontsize=8)
        axes[row_idx, 1].set_ylabel("Count", fontsize=8)
        axes[row_idx, 1].set_title(f"{grp} — log1p", fontsize=9)
        axes[row_idx, 1].tick_params(labelsize=7)
        axes[row_idx, 1].grid(axis="y", alpha=0.3)
        axes[row_idx, 1].set_xlim(log_xmin, log_xmax)
        axes[row_idx, 1].set_ylim(0, log_ymax * 1.05)

    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    _savefig(fig, DIST_DIR, "01A_target_distribution_group_panels.png")

    # 1A-4: All distance groups overlaid in one plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Target: PGV-Z (mm/s) — distance_group overlay",
        fontsize=13,
        fontweight="bold",
    )
    _group_hist(
        axes[0],
        df,
        TARGET,
        "PGV-Z (mm/s)",
        "Raw values by distance_group",
        legend=True,
        hue_col="distance_group",
        hue_order=DISTANCE_GROUP_LABELS,
        palette=DISTANCE_GROUP_COLOR_MAP,
        legend_title="distance_group",
    )
    _group_hist(
        axes[1],
        tmp,
        "log_target",
        "log1p(PGV-Z)",
        "log1p-transformed by distance_group",
        legend=False,
        hue_col="distance_group",
        hue_order=DISTANCE_GROUP_LABELS,
        palette=DISTANCE_GROUP_COLOR_MAP,
        legend_title="distance_group",
    )
    fig.tight_layout()
    _savefig(fig, DIST_DIR, "01A_target_distribution_distance_overlay.png")

    # 1A-5: One panel per distance group
    n_dist_groups = len(DISTANCE_GROUP_LABELS)
    raw_ymax_dist = 1.0
    log_ymax_dist = 1.0
    for dlabel in DISTANCE_GROUP_LABELS:
        dvals = df.loc[df["distance_group"] == dlabel, TARGET].dropna().to_numpy()
        if dvals.size == 0:
            continue
        raw_counts, _ = np.histogram(dvals, bins=raw_bins)
        log_counts, _ = np.histogram(np.log1p(dvals), bins=log_bins)
        raw_ymax_dist = max(raw_ymax_dist, float(raw_counts.max()))
        log_ymax_dist = max(log_ymax_dist, float(log_counts.max()))

    fig, axes = plt.subplots(
        n_dist_groups,
        2,
        figsize=(12, max(2.8 * n_dist_groups, 8.0)),
        sharex="col",
        sharey="col",
    )
    if n_dist_groups == 1:
        axes = np.array([axes])
    fig.suptitle(
        "Target: PGV-Z (mm/s) — one row per distance_group",
        fontsize=13,
        fontweight="bold",
    )
    for row_idx, dlabel in enumerate(DISTANCE_GROUP_LABELS):
        dvals = df.loc[df["distance_group"] == dlabel, TARGET].dropna()
        clr = DISTANCE_GROUP_COLOR_MAP[dlabel]

        axes[row_idx, 0].hist(
            dvals,
            bins=raw_bins,
            color=clr,
            edgecolor="white",
            linewidth=0.2,
        )
        axes[row_idx, 0].set_xlabel("PGV-Z (mm/s)", fontsize=8)
        axes[row_idx, 0].set_ylabel("Count", fontsize=8)
        axes[row_idx, 0].set_title(f"{dlabel} — raw (n={len(dvals):,})", fontsize=9)
        axes[row_idx, 0].tick_params(labelsize=7)
        axes[row_idx, 0].grid(axis="y", alpha=0.3)
        axes[row_idx, 0].set_xlim(raw_xmin, raw_xmax)
        axes[row_idx, 0].set_ylim(0, raw_ymax_dist * 1.05)

        dlog = np.log1p(dvals)
        axes[row_idx, 1].hist(
            dlog,
            bins=log_bins,
            color=clr,
            edgecolor="white",
            linewidth=0.2,
        )
        axes[row_idx, 1].set_xlabel("log1p(PGV-Z)", fontsize=8)
        axes[row_idx, 1].set_ylabel("Count", fontsize=8)
        axes[row_idx, 1].set_title(f"{dlabel} — log1p", fontsize=9)
        axes[row_idx, 1].tick_params(labelsize=7)
        axes[row_idx, 1].grid(axis="y", alpha=0.3)
        axes[row_idx, 1].set_xlim(log_xmin, log_xmax)
        axes[row_idx, 1].set_ylim(0, log_ymax_dist * 1.05)

    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    _savefig(fig, DIST_DIR, "01A_target_distribution_distance_panels.png")

    # 1A-6: 2m subset only, separated by track number
    print("[1A-6] Target distribution for 2m subset by track_number …")
    df_2m = df[df["acc_distance_to_track_m"] == 2.0].dropna(subset=[TARGET]).copy()
    track_vals = sorted(df_2m["track_number"].dropna().astype(int).unique().tolist())

    if len(track_vals) == 0:
        print("  WARNING: no 2m data found for track-separated target plot.")
    else:
        if len(track_vals) != 2:
            print(
                f"  WARNING: expected 2 track classes at 2m, found {len(track_vals)}: {track_vals}"
            )

        all_raw_2m = df_2m[TARGET].to_numpy()
        all_log_2m = np.log1p(all_raw_2m)
        raw_2m_xmin, raw_2m_xmax = float(all_raw_2m.min()), float(all_raw_2m.max())
        log_2m_xmin, log_2m_xmax = float(all_log_2m.min()), float(all_log_2m.max())
        raw_2m_bins = np.linspace(raw_2m_xmin, raw_2m_xmax, 50)
        log_2m_bins = np.linspace(log_2m_xmin, log_2m_xmax, 50)

        raw_2m_ymax = 1.0
        log_2m_ymax = 1.0
        for trk in track_vals:
            tvals = df_2m.loc[df_2m["track_number"] == trk, TARGET].to_numpy()
            if tvals.size == 0:
                continue
            rc, _ = np.histogram(tvals, bins=raw_2m_bins)
            lc, _ = np.histogram(np.log1p(tvals), bins=log_2m_bins)
            raw_2m_ymax = max(raw_2m_ymax, float(rc.max()))
            log_2m_ymax = max(log_2m_ymax, float(lc.max()))

        track_palette = sns.color_palette("Set2", max(len(track_vals), 2))
        fig, axes = plt.subplots(
            len(track_vals),
            2,
            figsize=(12, max(2.8 * len(track_vals), 6.0)),
            sharex="col",
            sharey="col",
        )
        if len(track_vals) == 1:
            axes = np.array([axes])
        fig.suptitle(
            "Target: PGV-Z at 2m — one row per track_number",
            fontsize=13,
            fontweight="bold",
        )

        for i, trk in enumerate(track_vals):
            clr = track_palette[i % len(track_palette)]
            tvals = df_2m.loc[df_2m["track_number"] == trk, TARGET].to_numpy()

            axes[i, 0].hist(
                tvals,
                bins=raw_2m_bins,
                color=clr,
                edgecolor="white",
                linewidth=0.2,
            )
            axes[i, 0].set_xlabel("PGV-Z (mm/s)", fontsize=8)
            axes[i, 0].set_ylabel("Count", fontsize=8)
            axes[i, 0].set_title(f"track {trk} — raw (n={len(tvals):,})", fontsize=9)
            axes[i, 0].tick_params(labelsize=7)
            axes[i, 0].grid(axis="y", alpha=0.3)
            axes[i, 0].set_xlim(raw_2m_xmin, raw_2m_xmax)
            axes[i, 0].set_ylim(0, raw_2m_ymax * 1.05)

            tlog = np.log1p(tvals)
            axes[i, 1].hist(
                tlog,
                bins=log_2m_bins,
                color=clr,
                edgecolor="white",
                linewidth=0.2,
            )
            axes[i, 1].set_xlabel("log1p(PGV-Z)", fontsize=8)
            axes[i, 1].set_ylabel("Count", fontsize=8)
            axes[i, 1].set_title(f"track {trk} — log1p", fontsize=9)
            axes[i, 1].tick_params(labelsize=7)
            axes[i, 1].grid(axis="y", alpha=0.3)
            axes[i, 1].set_xlim(log_2m_xmin, log_2m_xmax)
            axes[i, 1].set_ylim(0, log_2m_ymax * 1.05)

        fig.tight_layout(rect=[0, 0.01, 1, 0.98])
        _savefig(fig, DIST_DIR, "01A_target_distribution_2m_by_track_panels.png")

    # 1A-7: All filtered data, separated by ISO week
    print("[1A-7] Target distribution by event_week (all filtered rows) …")
    week_df = df.dropna(subset=["event_week", TARGET]).copy()
    week_order = sorted(week_df["event_week"].unique().tolist())

    if len(week_order) == 0:
        print("  WARNING: no parsable event_week values found.")
    else:
        week_palette = dict(zip(week_order, sns.color_palette("mako", len(week_order))))

        # Overlay view
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            "Target: PGV-Z (mm/s) — event_week overlay",
            fontsize=13,
            fontweight="bold",
        )
        _group_hist(
            axes[0],
            week_df,
            TARGET,
            "PGV-Z (mm/s)",
            "Raw values by event_week",
            legend=True,
            hue_col="event_week",
            hue_order=week_order,
            palette=week_palette,
            legend_title="event_week",
        )

        week_tmp = week_df[[TARGET, "event_week"]].copy()
        week_tmp["log_target"] = np.log1p(week_tmp[TARGET])
        _group_hist(
            axes[1],
            week_tmp,
            "log_target",
            "log1p(PGV-Z)",
            "log1p-transformed by event_week",
            legend=False,
            hue_col="event_week",
            hue_order=week_order,
            palette=week_palette,
            legend_title="event_week",
        )
        fig.tight_layout()
        _savefig(fig, DIST_DIR, "01A_target_distribution_week_overlay.png")

        # One row per week, shared scales
        raw_w = week_df[TARGET].to_numpy()
        log_w = np.log1p(raw_w)
        raw_w_xmin, raw_w_xmax = float(raw_w.min()), float(raw_w.max())
        log_w_xmin, log_w_xmax = float(log_w.min()), float(log_w.max())
        raw_w_bins = np.linspace(raw_w_xmin, raw_w_xmax, 50)
        log_w_bins = np.linspace(log_w_xmin, log_w_xmax, 50)

        raw_w_ymax = 1.0
        log_w_ymax = 1.0
        for w in week_order:
            wvals = week_df.loc[week_df["event_week"] == w, TARGET].to_numpy()
            if wvals.size == 0:
                continue
            rc, _ = np.histogram(wvals, bins=raw_w_bins)
            lc, _ = np.histogram(np.log1p(wvals), bins=log_w_bins)
            raw_w_ymax = max(raw_w_ymax, float(rc.max()))
            log_w_ymax = max(log_w_ymax, float(lc.max()))

        fig, axes = plt.subplots(
            len(week_order),
            2,
            figsize=(12, max(2.8 * len(week_order), 6.0)),
            sharex="col",
            sharey="col",
        )
        if len(week_order) == 1:
            axes = np.array([axes])
        fig.suptitle(
            "Target: PGV-Z (mm/s) — one row per event_week",
            fontsize=13,
            fontweight="bold",
        )
        for i, w in enumerate(week_order):
            clr = week_palette[w]
            wvals = week_df.loc[week_df["event_week"] == w, TARGET].to_numpy()

            axes[i, 0].hist(
                wvals,
                bins=raw_w_bins,
                color=clr,
                edgecolor="white",
                linewidth=0.2,
            )
            axes[i, 0].set_xlabel("PGV-Z (mm/s)", fontsize=8)
            axes[i, 0].set_ylabel("Count", fontsize=8)
            axes[i, 0].set_title(f"{w} — raw (n={len(wvals):,})", fontsize=9)
            axes[i, 0].tick_params(labelsize=7)
            axes[i, 0].grid(axis="y", alpha=0.3)
            axes[i, 0].set_xlim(raw_w_xmin, raw_w_xmax)
            axes[i, 0].set_ylim(0, raw_w_ymax * 1.05)

            wlog = np.log1p(wvals)
            axes[i, 1].hist(
                wlog,
                bins=log_w_bins,
                color=clr,
                edgecolor="white",
                linewidth=0.2,
            )
            axes[i, 1].set_xlabel("log1p(PGV-Z)", fontsize=8)
            axes[i, 1].set_ylabel("Count", fontsize=8)
            axes[i, 1].set_title(f"{w} — log1p", fontsize=9)
            axes[i, 1].tick_params(labelsize=7)
            axes[i, 1].grid(axis="y", alpha=0.3)
            axes[i, 1].set_xlim(log_w_xmin, log_w_xmax)
            axes[i, 1].set_ylim(0, log_w_ymax * 1.05)

        fig.tight_layout(rect=[0, 0.01, 1, 0.98])
        _savefig(fig, DIST_DIR, "01A_target_distribution_week_panels.png")

    # ── 1B  Geometry features ─────────────────────────────────────────────────
    print("[1B] Geometry features …")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Geometry features", fontsize=13, fontweight="bold")

    # distance histogram
    _group_hist(
        axes[0],
        df,
        "acc_distance_to_track_m",
        "Distance to track (m)",
        "acc_distance_to_track_m by train_type_group",
        legend=False,
    )

    # side of track bar
    side_cts = (
        df.groupby(["acc_side_of_track", "train_type_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TRAIN_GROUPS_PRESENT, fill_value=0)
    )
    side_cts.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color=[TRAIN_GROUP_COLOR_MAP[g] for g in side_cts.columns],
        width=0.8,
    )
    axes[1].set_xlabel("acc_side_of_track", fontsize=8)
    axes[1].set_ylabel("Count", fontsize=8)
    axes[1].set_title("acc_side_of_track by train_type_group", fontsize=9)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(axis="y", alpha=0.3)
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()

    # track number bar
    trk_cts = (
        df.groupby(["track_number", "train_type_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TRAIN_GROUPS_PRESENT, fill_value=0)
    )
    trk_cts.plot(
        kind="bar",
        stacked=True,
        ax=axes[2],
        color=[TRAIN_GROUP_COLOR_MAP[g] for g in trk_cts.columns],
        width=0.8,
    )
    axes[2].set_xlabel("track_number", fontsize=8)
    axes[2].set_ylabel("Count", fontsize=8)
    axes[2].set_title("track_number by train_type_group", fontsize=9)
    axes[2].tick_params(labelsize=7)
    axes[2].grid(axis="y", alpha=0.3)
    if axes[2].legend_ is not None:
        axes[2].legend_.remove()

    fig.tight_layout()
    _savefig(fig, DIST_DIR, "01B_geometry_features.png")

    # ── 1C  Train metadata ────────────────────────────────────────────────────
    print("[1C] Train metadata …")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Train metadata", fontsize=13, fontweight="bold")

    # train_type bar (use unique events to avoid per-sensor duplication)
    ev_df = df.drop_duplicates("event_id")
    grp_cts = (
        ev_df["train_type_group"]
        .value_counts()
        .reindex(TRAIN_GROUPS_PRESENT, fill_value=0)
    )
    bars = axes[0].barh(
        grp_cts.index,
        grp_cts.values,
        color=[TRAIN_GROUP_COLOR_MAP[g] for g in grp_cts.index],
    )
    axes[0].set_xlabel("Unique events", fontsize=8)
    axes[0].set_title("train_type_group (unique events)", fontsize=9)
    axes[0].tick_params(labelsize=7)
    axes[0].grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, grp_cts.values):
        axes[0].text(
            v + 2, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", fontsize=7
        )

    # train_speed histogram
    _group_hist(
        axes[1],
        ev_df,
        "train_speed_kmh",
        "Speed (km/h)",
        "train_speed_kmh by train_type_group (unique events)",
        legend=False,
    )

    # track_number per train_type stacked bar
    pivot = (
        ev_df.groupby(["train_type_group", "track_number"])
        .size()
        .unstack(fill_value=0)
        .reindex(TRAIN_GROUPS_PRESENT, fill_value=0)
    )
    pivot.plot(kind="bar", ax=axes[2], colormap="tab10", width=0.7)
    axes[2].set_xlabel("train_type_group", fontsize=8)
    axes[2].set_ylabel("Row count", fontsize=8)
    axes[2].set_title("Unique events per train_type_group × track", fontsize=9)
    axes[2].tick_params(axis="x", labelsize=7, rotation=30)
    axes[2].tick_params(axis="y", labelsize=7)
    axes[2].legend(title="track", fontsize=6, title_fontsize=7)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, DIST_DIR, "01C_train_metadata.png")

    # ── 1D  FO time-domain features ───────────────────────────────────────────
    print("[1D] FO time-domain features …")
    td_cols = [
        "fo_td_max_abs_mean",
        "fo_td_max_abs_max",
        "fo_td_max_abs_std",
        "fo_td_rms_mean",
        "fo_td_rms_max",
        "fo_td_rms_std",
        "fo_td_std_mean",
        "fo_td_std_max",
        "fo_td_std_std",
    ]
    colors = ["steelblue"] * 3 + ["darkorange"] * 3 + ["mediumseagreen"] * 3
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(
        "FO time-domain features (log-x scale)", fontsize=13, fontweight="bold"
    )
    td_sample = df.sample(min(len(df), 8000), random_state=42)
    for ax, col, clr in zip(axes.flat, td_cols, colors):
        _group_hist(
            ax,
            td_sample,
            col,
            col,
            col.replace("fo_td_", "") + " by train_type_group",
            log_x=True,
            legend=False,
        )
    fig.tight_layout()
    _savefig(fig, DIST_DIR, "01D_fo_timedomain.png")

    # ── 1E  FO octave bands (one file per reduction) ──────────────────────────
    print("[1E] FO octave bands …")
    REDUCTION_COLORS = {
        "mean": "steelblue",
        "max": "darkorange",
        "std": "mediumseagreen",
    }
    NCOLS, NROWS = 3, 7  # 3×7 = 21 subplots per figure

    for red, clr in REDUCTION_COLORS.items():
        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(13, 18))
        fig.suptitle(
            f"FO octave bands — {red} (log-x scale)", fontsize=13, fontweight="bold"
        )
        for ax, band, lbl in zip(axes.flat, BAND_ORDER, BAND_LABELS):
            col = f"fo_oct_{band}_{red}"
            if col in df.columns:
                _group_hist(
                    ax,
                    td_sample,
                    col,
                    col,
                    f"{lbl} Hz by train_type_group",
                    log_x=True,
                    legend=False,
                )
            else:
                ax.set_visible(False)
        fig.tight_layout()
        _savefig(fig, DIST_DIR, f"01E_fo_octave_bands_{red}.png")

    print("[1] All distribution plots done.")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Correlation analysis (Spearman)
# ══════════════════════════════════════════════════════════════════════════════
if RUN_2_CORRELATIONS:
    print("\n=== Analysis 2: Correlation analysis ===")

    # Feature columns (numeric, excluding target and metadata)
    EXCLUDE_COLS = {
        TARGET,
        "event_id",
        "site_id",
        "sensor_id",
        "train_type",
        "train_type_code",
    }
    feat_cols = [
        c
        for c in df.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    # ── 2A  Feature-target Spearman bar chart ─────────────────────────────────
    print("[2A] Feature-target Spearman correlations …")
    y = df[TARGET].dropna()
    valid_idx = y.index

    rho_vals: dict[str, float] = {}
    for col in feat_cols:
        x = df.loc[valid_idx, col].dropna()
        common = x.index.intersection(valid_idx)
        if len(common) < 30:
            continue
        rho, _ = spearmanr(x.loc[common], y.loc[common])
        rho_vals[col] = rho

    rho_series = pd.Series(rho_vals).sort_values(key=abs, ascending=False)

    # Full-list plot (all features, sorted by |rho|)
    TOP_CORR = 40
    fig, ax = plt.subplots(figsize=(10, max(6, TOP_CORR * 0.3)))
    top_rho = rho_series.iloc[:TOP_CORR]
    colors = ["steelblue" if v >= 0 else "tomato" for v in top_rho.values]
    ax.barh(top_rho.index[::-1], top_rho.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman rho", fontsize=9)
    ax.set_title(
        f"Feature-target Spearman correlation (top {TOP_CORR} by |rho|)",
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, CORR_DIR, "02A_feature_target_spearman.png")

    # ── 2B  Feature-feature heatmap (top N by importance) ─────────────────────
    print("[2B] Feature-feature heatmap (top N by importance) …")
    fi = pd.read_csv(IMPORTANCE_CSV)
    top_feats = fi.sort_values("gain", ascending=False).head(TOP_N)["feature"].tolist()
    # keep only those that are in the dataset
    top_feats = [f for f in top_feats if f in df.columns]

    sub = df[top_feats].dropna()
    # Compute pairwise Spearman matrix
    n = len(top_feats)
    rho_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(sub.iloc[:, i], sub.iloc[:, j])
            rho_mat[i, j] = r
            rho_mat[j, i] = r
    rho_df = pd.DataFrame(rho_mat, index=top_feats, columns=top_feats)

    fig, ax = plt.subplots(figsize=(max(8, TOP_N * 0.7), max(7, TOP_N * 0.65)))
    sns.heatmap(
        rho_df,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Spearman rho", "shrink": 0.7},
    )
    ax.set_title(
        f"Feature-feature Spearman correlation (top {TOP_N} by XGB gain)",
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(labelsize=7, axis="both")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _savefig(fig, CORR_DIR, "02B_feature_feature_heatmap.png")

    print("[2] Correlation analysis done.")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Feature importance vs Spearman correlation scatter
# ══════════════════════════════════════════════════════════════════════════════
if RUN_3_IMPORTANCE_CORR:
    print("\n=== Analysis 3: Importance vs correlation ===")

    # Build Spearman rho for every numeric feature if not already done
    EXCLUDE_COLS = {
        TARGET,
        "event_id",
        "site_id",
        "sensor_id",
        "train_type",
        "train_type_code",
    }
    feat_cols = [
        c
        for c in df.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    y = df[TARGET].dropna()
    valid_idx = y.index
    rho_vals: dict[str, float] = {}
    for col in feat_cols:
        x = df.loc[valid_idx, col].dropna()
        common = x.index.intersection(valid_idx)
        if len(common) < 30:
            continue
        rho, _ = spearmanr(x.loc[common], y.loc[common])
        rho_vals[col] = rho
    rho_series = pd.Series(rho_vals)

    # Load importance
    fi = pd.read_csv(IMPORTANCE_CSV)
    fi = fi[fi["feature"].isin(rho_series.index)].copy()
    fi["rho"] = fi["feature"].map(rho_series)

    # Normalise gain to 0-1 for point sizing
    fi["gain_norm"] = fi["gain"] / fi["gain"].max()

    # Colour by feature group
    def _feat_group(name: str) -> str:
        if "distance" in name or "side" in name or "track" in name:
            return "geometry"
        if "fo_td" in name:
            return "FO time-domain"
        if "fo_oct" in name:
            return "FO octave band"
        if "train_speed" in name:
            return "train metadata"
        return "other"

    fi["group"] = fi["feature"].apply(_feat_group)
    GROUP_COLORS = {
        "geometry": "steelblue",
        "FO time-domain": "darkorange",
        "FO octave band": "mediumseagreen",
        "train metadata": "orchid",
        "other": "gray",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for grp, gdf in fi.groupby("group"):
        ax.scatter(
            gdf["rho"],
            gdf["gain"],
            s=gdf["gain_norm"] * 300 + 20,
            color=GROUP_COLORS.get(grp, "gray"),
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label=grp,
            zorder=3,
        )

    # Label top features by gain
    top_label = fi.nlargest(15, "gain")
    for _, row in top_label.iterrows():
        ax.annotate(
            row["feature"],
            xy=(row["rho"], row["gain"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Spearman rho (vs target)", fontsize=10)
    ax.set_ylabel("XGB gain (normalised feature importance)", fontsize=10)
    ax.set_title(
        "XGBoost gain vs Spearman correlation with target\n"
        "(point size proportional to gain; top-15 annotated)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(title="Feature group", fontsize=8, title_fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, CORR_DIR, "03_importance_vs_spearman.png")
    print("[3] Importance vs correlation done.")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Target vs distance scatter
# ══════════════════════════════════════════════════════════════════════════════
if RUN_4_TARGET_DISTANCE:
    print("\n=== Analysis 4: Target vs distance ===")

    sub = df[
        [
            TARGET,
            "acc_distance_to_track_m",
            "train_type",
            "train_type_group",
            "train_speed_kmh",
            "track_number",
            "sensor_id",
        ]
    ].dropna()

    # ── 4A  Coloured by train_type_group ──────────────────────────────────
    print("[4A] Scatter coloured by train_type_group …")
    present_groups = set(sub["train_type_group"].unique())
    train_groups = [g for g in TRAIN_GROUPS_PRESENT if g in present_groups]
    color_map = {g: TRAIN_GROUP_COLOR_MAP[g] for g in train_groups}

    fig, ax = plt.subplots(figsize=(10, 6))
    for grp in train_groups:
        mask = sub["train_type_group"] == grp
        ax.scatter(
            sub.loc[mask, "acc_distance_to_track_m"],
            sub.loc[mask, TARGET],
            s=10,
            alpha=0.35,
            color=color_map[grp],
            label=grp,
            rasterized=True,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance to track (m)  [log]", fontsize=10)
    ax.set_ylabel("PGV-Z (mm/s)  [log]", fontsize=10)
    ax.set_title(
        "Target vs distance — coloured by train_type_group",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        title="train_type_group",
        fontsize=7,
        title_fontsize=8,
        markerscale=2,
        framealpha=0.8,
    )
    ax.grid(which="both", alpha=0.2)
    fig.tight_layout()
    _savefig(fig, TGTD_DIR, "04A_target_vs_distance_train_type_group.png")

    # ── 4B  Coloured by train_speed ───────────────────────────────────────
    print("[4B] Scatter coloured by train_speed …")
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        sub["acc_distance_to_track_m"],
        sub[TARGET],
        c=sub["train_speed_kmh"],
        cmap="plasma",
        s=10,
        alpha=0.4,
        rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Train speed (km/h)", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance to track (m)  [log]", fontsize=10)
    ax.set_ylabel("PGV-Z (mm/s)  [log]", fontsize=10)
    ax.set_title(
        "Target vs distance — coloured by train speed", fontsize=12, fontweight="bold"
    )
    ax.grid(which="both", alpha=0.2)
    fig.tight_layout()
    _savefig(fig, TGTD_DIR, "04B_target_vs_distance_speed.png")

    # ── 4C  One facet per sensor_id (shows per-sensor attenuation curves) ──
    print("[4C] Facet plot per sensor_id …")
    sensors = sorted(sub["sensor_id"].unique())
    ncols = 4
    nrows = int(np.ceil(len(sensors) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4, nrows * 3.2), sharex=True, sharey=True
    )
    fig.suptitle("PGV-Z vs distance — per sensor", fontsize=13, fontweight="bold")
    x_min = float(sub["acc_distance_to_track_m"].min())
    x_max = float(sub["acc_distance_to_track_m"].max())
    y_pos = sub[TARGET][sub[TARGET] > 0]
    y_min = float(y_pos.min()) if not y_pos.empty else 1e-6
    y_max = float(sub[TARGET].max())
    for ax, sid in zip(axes.flat, sensors):
        sdf = sub[sub["sensor_id"] == sid]
        for grp in train_groups:
            mask = sdf["train_type_group"] == grp
            if mask.sum() == 0:
                continue
            ax.scatter(
                sdf.loc[mask, "acc_distance_to_track_m"],
                sdf.loc[mask, TARGET],
                s=8,
                alpha=0.4,
                color=color_map[grp],
                label=grp,
                rasterized=True,
            )
        dist_val = sdf["acc_distance_to_track_m"].median()
        ax.set_title(f"{sid}  (d={dist_val:.1f} m)", fontsize=8, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlabel("dist (m)", fontsize=7)
        ax.set_ylabel("PGV-Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    # hide unused axes
    for ax in axes.flat[len(sensors) :]:
        ax.set_visible(False)
    # shared legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[grp],
            markersize=6,
            label=grp,
        )
        for grp in train_groups
    ]
    fig.legend(
        handles=handles,
        title="train_type_group",
        fontsize=7,
        title_fontsize=8,
        loc="lower right",
        ncol=2,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _savefig(fig, TGTD_DIR, "04C_target_vs_distance_per_sensor_grouped.png")

    print("[4] Target vs distance done.")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5 — Seaborn pair plot (top-N features by XGB gain + target)
# ══════════════════════════════════════════════════════════════════════════════
if RUN_5_PAIRPLOT:
    print("\n=== Analysis 5: Pair plot ===")

    fi = pd.read_csv(IMPORTANCE_CSV)
    top_feats = fi.sort_values("gain", ascending=False).head(TOP_N)["feature"].tolist()
    top_feats = [f for f in top_feats if f in df.columns]

    # Always include distance + specific FO means even if not in top N
    FORCE_INCLUDE = [
        "acc_distance_to_track_m",
        "fo_oct_1_60hz_mean",
        "fo_oct_2_00hz_mean",
        "fo_oct_040hz_mean",
    ]
    extra_cols = [c for c in FORCE_INCLUDE if c in df.columns and c not in top_feats]

    # Build working frame including grouped train-type label for colouring.
    pair_df = df[top_feats + extra_cols + ["train_type_group"]].copy()

    # Log-transform positive FO features for readability
    for col in top_feats + extra_cols:
        if col in pair_df.columns and pair_df[col].gt(0).all():
            pair_df[col] = np.log1p(pair_df[col])
            pair_df.rename(columns={col: f"log_{col}"}, inplace=True)

    # Subsample to keep the plot fast (seaborn pairplot is O(n^2) per cell)
    MAX_ROWS = 3000
    if len(pair_df) > MAX_ROWS:
        pair_df = pair_df.sample(MAX_ROWS, random_state=42)

    # Keep only log_acc_distance_to_track + top 3 FO octave band _mean features
    TOP3_FO_MEANS = {
        "log_fo_oct_1_60hz_mean",
        "log_fo_oct_2_00hz_mean",
        "log_fo_oct_040hz_mean",
    }
    feat_cols_plot = [
        c
        for c in pair_df.columns
        if c != "train_type_group"
        and (c == "log_acc_distance_to_track_m" or c in TOP3_FO_MEANS)
    ]
    print(
        f"  Plotting pair plot with {len(feat_cols_plot)} variables "
        f"on {len(pair_df):,} rows …"
    )

    present_groups = sorted(pair_df["train_type_group"].dropna().unique())
    group_order = [g for g in TRAIN_GROUPS_PRESENT if g in present_groups]
    group_palette = {g: TRAIN_GROUP_COLOR_MAP[g] for g in group_order}

    pg = sns.pairplot(
        pair_df,
        vars=feat_cols_plot,
        hue="train_type_group",
        hue_order=group_order,
        palette=group_palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.3, "s": 10, "rasterized": True},
        diag_kws={"fill": True, "alpha": 0.4},
    )
    pg.figure.suptitle(
        f"Pair plot — top {TOP_N} features by XGB gain  │  coloured by train_type_group",
        y=1.01,
        fontsize=12,
        fontweight="bold",
    )
    pg._legend.set_title("train_type_group")
    out_path = BASE_OUT / "05_pairplot_top_features.png"
    pg.figure.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(pg.figure)
    print(f"  Saved: {out_path.relative_to(BASE_OUT)}")
    print("[5] Pair plot done.")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6 — MP4-specific diagnostics (requested follow-up)
# ══════════════════════════════════════════════════════════════════════════════
if RUN_6_MP4_DIAGNOSTICS:
    print("\n=== Analysis 6: MP4 diagnostics ===")

    mp4 = df[(df["sensor_id"] == "MP4") & df[TARGET].notna()].copy()
    if len(mp4) == 0:
        print("[6] WARNING: no MP4 rows available after filters; skipping.")
    else:
        print(
            f"[6] MP4 subset: rows={len(mp4):,} | "
            f"unique events={mp4['event_id'].nunique():,}"
        )

        # 6A — grouped by speed bins (quantile bins)
        print("[6A] MP4 target distribution by speed bins …")
        mp4_speed = mp4.dropna(subset=["train_speed_kmh"]).copy()
        if len(mp4_speed) < 20:
            print(
                "  WARNING: insufficient rows with train_speed_kmh for speed-bin plot."
            )
        else:
            q = np.quantile(
                mp4_speed["train_speed_kmh"].to_numpy(), [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            )
            q = np.unique(np.round(q, 3))
            if len(q) < 3:
                print(
                    "  WARNING: speed values too concentrated to build quantile bins."
                )
            else:
                labels = [f"{q[i]:.0f}-{q[i+1]:.0f} km/h" for i in range(len(q) - 1)]
                mp4_speed["speed_bin"] = pd.cut(
                    mp4_speed["train_speed_kmh"],
                    bins=q,
                    labels=labels,
                    include_lowest=True,
                    duplicates="drop",
                )
                speed_order = [
                    x for x in labels if x in set(mp4_speed["speed_bin"].astype(str))
                ]
                mp4_speed = mp4_speed.dropna(subset=["speed_bin"])

                raw = mp4_speed[TARGET].to_numpy()
                log = np.log1p(raw)
                raw_bins = np.linspace(float(raw.min()), float(raw.max()), 50)
                log_bins = np.linspace(float(log.min()), float(log.max()), 50)

                raw_ymax, log_ymax = 1.0, 1.0
                for b in speed_order:
                    vals = mp4_speed.loc[
                        mp4_speed["speed_bin"].astype(str) == b, TARGET
                    ].to_numpy()
                    if vals.size == 0:
                        continue
                    rc, _ = np.histogram(vals, bins=raw_bins)
                    lc, _ = np.histogram(np.log1p(vals), bins=log_bins)
                    raw_ymax = max(raw_ymax, float(rc.max()))
                    log_ymax = max(log_ymax, float(lc.max()))

                fig, axes = plt.subplots(
                    len(speed_order),
                    2,
                    figsize=(12, max(2.8 * len(speed_order), 7.0)),
                    sharex="col",
                    sharey="col",
                )
                if len(speed_order) == 1:
                    axes = np.array([axes])
                palette = sns.color_palette("crest", len(speed_order))
                fig.suptitle(
                    "MP4 (2m) target distribution — by speed bin",
                    fontsize=13,
                    fontweight="bold",
                )
                for i, b in enumerate(speed_order):
                    vals = mp4_speed.loc[
                        mp4_speed["speed_bin"].astype(str) == b, TARGET
                    ].to_numpy()
                    clr = palette[i]
                    axes[i, 0].hist(
                        vals, bins=raw_bins, color=clr, edgecolor="white", linewidth=0.2
                    )
                    axes[i, 1].hist(
                        np.log1p(vals),
                        bins=log_bins,
                        color=clr,
                        edgecolor="white",
                        linewidth=0.2,
                    )
                    axes[i, 0].set_title(f"{b} — raw (n={len(vals):,})", fontsize=9)
                    axes[i, 1].set_title(f"{b} — log1p", fontsize=9)
                    axes[i, 0].set_ylabel("Count", fontsize=8)
                    axes[i, 0].grid(axis="y", alpha=0.3)
                    axes[i, 1].grid(axis="y", alpha=0.3)
                    axes[i, 0].tick_params(labelsize=7)
                    axes[i, 1].tick_params(labelsize=7)
                    axes[i, 0].set_ylim(0, raw_ymax * 1.05)
                    axes[i, 1].set_ylim(0, log_ymax * 1.05)
                for ax in axes[:, 0]:
                    ax.set_xlabel("PGV-Z (mm/s)", fontsize=8)
                for ax in axes[:, 1]:
                    ax.set_xlabel("log1p(PGV-Z)", fontsize=8)
                fig.tight_layout(rect=[0, 0.01, 1, 0.98])
                _savefig(fig, DIST_DIR, "06A_mp4_target_distribution_speed_bins.png")

        # 6B — grouped by raw train_type (top labels)
        print("[6B] MP4 target distribution by raw train_type …")
        mp4_tt = mp4.copy()
        mp4_tt["train_type_raw"] = mp4_tt["train_type"].fillna("unknown").astype(str)
        tt_counts = mp4_tt["train_type_raw"].value_counts()
        top_tt = tt_counts.head(10).index.tolist()
        mp4_top = mp4_tt[mp4_tt["train_type_raw"].isin(top_tt)].copy()

        raw = mp4_top[TARGET].to_numpy()
        log = np.log1p(raw)
        raw_bins = np.linspace(float(raw.min()), float(raw.max()), 50)
        log_bins = np.linspace(float(log.min()), float(log.max()), 50)

        raw_ymax, log_ymax = 1.0, 1.0
        for tt in top_tt:
            vals = mp4_top.loc[mp4_top["train_type_raw"] == tt, TARGET].to_numpy()
            if vals.size == 0:
                continue
            rc, _ = np.histogram(vals, bins=raw_bins)
            lc, _ = np.histogram(np.log1p(vals), bins=log_bins)
            raw_ymax = max(raw_ymax, float(rc.max()))
            log_ymax = max(log_ymax, float(lc.max()))

        fig, axes = plt.subplots(
            len(top_tt),
            2,
            figsize=(12, max(2.7 * len(top_tt), 8.0)),
            sharex="col",
            sharey="col",
        )
        if len(top_tt) == 1:
            axes = np.array([axes])
        palette = sns.color_palette("rocket", len(top_tt))
        fig.suptitle(
            "MP4 (2m) target distribution — by raw train_type (top 10)",
            fontsize=13,
            fontweight="bold",
        )
        for i, tt in enumerate(top_tt):
            vals = mp4_top.loc[mp4_top["train_type_raw"] == tt, TARGET].to_numpy()
            clr = palette[i]
            axes[i, 0].hist(
                vals, bins=raw_bins, color=clr, edgecolor="white", linewidth=0.2
            )
            axes[i, 1].hist(
                np.log1p(vals),
                bins=log_bins,
                color=clr,
                edgecolor="white",
                linewidth=0.2,
            )
            axes[i, 0].set_title(f"{tt} — raw (n={len(vals):,})", fontsize=8)
            axes[i, 1].set_title(f"{tt} — log1p", fontsize=8)
            axes[i, 0].set_ylabel("Count", fontsize=8)
            axes[i, 0].grid(axis="y", alpha=0.3)
            axes[i, 1].grid(axis="y", alpha=0.3)
            axes[i, 0].tick_params(labelsize=7)
            axes[i, 1].tick_params(labelsize=7)
            axes[i, 0].set_ylim(0, raw_ymax * 1.05)
            axes[i, 1].set_ylim(0, log_ymax * 1.05)
        for ax in axes[:, 0]:
            ax.set_xlabel("PGV-Z (mm/s)", fontsize=8)
        for ax in axes[:, 1]:
            ax.set_xlabel("log1p(PGV-Z)", fontsize=8)
        fig.tight_layout(rect=[0, 0.01, 1, 0.98])
        _savefig(fig, DIST_DIR, "06B_mp4_target_distribution_raw_train_type_top10.png")

        # 6B2 — grouped by train_type_group (same grouping system, MP4 only)
        print("[6B2] MP4 target distribution by train_type_group …")
        mp4_grp = mp4.copy()
        grp_order = [
            g
            for g in TRAIN_GROUPS_PRESENT
            if g in set(mp4_grp["train_type_group"].dropna().unique())
        ]

        if len(grp_order) == 0:
            print("  WARNING: no train_type_group labels found for MP4.")
        else:
            raw = mp4_grp[TARGET].to_numpy()
            log = np.log1p(raw)
            raw_bins = np.linspace(float(raw.min()), float(raw.max()), 50)
            log_bins = np.linspace(float(log.min()), float(log.max()), 50)

            raw_ymax, log_ymax = 1.0, 1.0
            for grp in grp_order:
                vals = mp4_grp.loc[
                    mp4_grp["train_type_group"] == grp, TARGET
                ].to_numpy()
                if vals.size == 0:
                    continue
                rc, _ = np.histogram(vals, bins=raw_bins)
                lc, _ = np.histogram(np.log1p(vals), bins=log_bins)
                raw_ymax = max(raw_ymax, float(rc.max()))
                log_ymax = max(log_ymax, float(lc.max()))

            fig, axes = plt.subplots(
                len(grp_order),
                2,
                figsize=(12, max(2.7 * len(grp_order), 8.0)),
                sharex="col",
                sharey="col",
            )
            if len(grp_order) == 1:
                axes = np.array([axes])

            fig.suptitle(
                "MP4 (2m) target distribution — by train_type_group",
                fontsize=13,
                fontweight="bold",
            )
            for i, grp in enumerate(grp_order):
                vals = mp4_grp.loc[
                    mp4_grp["train_type_group"] == grp, TARGET
                ].to_numpy()
                clr = TRAIN_GROUP_COLOR_MAP.get(grp, "gray")
                axes[i, 0].hist(
                    vals,
                    bins=raw_bins,
                    color=clr,
                    edgecolor="white",
                    linewidth=0.2,
                )
                axes[i, 1].hist(
                    np.log1p(vals),
                    bins=log_bins,
                    color=clr,
                    edgecolor="white",
                    linewidth=0.2,
                )
                axes[i, 0].set_title(f"{grp} — raw (n={len(vals):,})", fontsize=8)
                axes[i, 1].set_title(f"{grp} — log1p", fontsize=8)
                axes[i, 0].set_ylabel("Count", fontsize=8)
                axes[i, 0].grid(axis="y", alpha=0.3)
                axes[i, 1].grid(axis="y", alpha=0.3)
                axes[i, 0].tick_params(labelsize=7)
                axes[i, 1].tick_params(labelsize=7)
                axes[i, 0].set_ylim(0, raw_ymax * 1.05)
                axes[i, 1].set_ylim(0, log_ymax * 1.05)
            for ax in axes[:, 0]:
                ax.set_xlabel("PGV-Z (mm/s)", fontsize=8)
            for ax in axes[:, 1]:
                ax.set_xlabel("log1p(PGV-Z)", fontsize=8)
            fig.tight_layout(rect=[0, 0.01, 1, 0.98])
            _savefig(fig, DIST_DIR, "06B2_mp4_target_distribution_train_type_group.png")

        # 6C — time-of-day and day-of-week effects
        print("[6C] MP4 target diagnostics by hour/day …")
        mp4_time = mp4.dropna(subset=["event_dt"]).copy()
        mp4_time["log_target"] = np.log1p(mp4_time[TARGET])
        mp4_time["hour"] = mp4_time["event_dt"].dt.hour
        mp4_time["hour_bin"] = pd.cut(
            mp4_time["hour"],
            bins=[0, 4, 8, 12, 16, 20, 24],
            labels=["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"],
            right=False,
            include_lowest=True,
        )
        mp4_time["day_of_week"] = mp4_time["event_dt"].dt.day_name()
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        fig.suptitle(
            "MP4 (2m) log-target diagnostics — time patterns",
            fontsize=13,
            fontweight="bold",
        )

        sns.boxplot(
            data=mp4_time,
            x="hour_bin",
            y="log_target",
            ax=axes[0],
            color="#4c78a8",
            fliersize=1.5,
        )
        axes[0].set_title("log1p(PGV-Z) by hour bin", fontsize=10)
        axes[0].set_xlabel("hour bin", fontsize=9)
        axes[0].set_ylabel("log1p(PGV-Z)", fontsize=9)
        axes[0].tick_params(axis="x", rotation=25, labelsize=8)
        axes[0].grid(axis="y", alpha=0.25)

        sns.boxplot(
            data=mp4_time,
            x="day_of_week",
            y="log_target",
            order=day_order,
            ax=axes[1],
            color="#f58518",
            fliersize=1.5,
        )
        axes[1].set_title("log1p(PGV-Z) by day of week", fontsize=10)
        axes[1].set_xlabel("day of week", fontsize=9)
        axes[1].set_ylabel("log1p(PGV-Z)", fontsize=9)
        axes[1].tick_params(axis="x", rotation=30, labelsize=8)
        axes[1].grid(axis="y", alpha=0.25)

        fig.tight_layout(rect=[0, 0.01, 1, 0.95])
        _savefig(fig, DIST_DIR, "06C_mp4_logtarget_time_patterns.png")

        print("[6] MP4 diagnostics done.")

print("\nDone.")
