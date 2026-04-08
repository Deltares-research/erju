"""explore_features.py — Feature distribution & analysis script for parquet v2.

Run from the project root:
    python src/db/parquet/explore_features.py

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

# Analysis flags — set False to skip
RUN_1_DISTRIBUTIONS = False
RUN_2_CORRELATIONS = False
RUN_3_IMPORTANCE_CORR = False
RUN_4_TARGET_DISTANCE = False
RUN_5_PAIRPLOT = True

# Pair-plot / heatmap top-N
TOP_N = 8

# ── Output dirs ──────────────────────────────────────────────────────────────
BASE_OUT = PARQUET_PATH.parent / "plots"
DIST_DIR = BASE_OUT / "distributions"
CORR_DIR = BASE_OUT / "correlations"
TGTD_DIR = BASE_OUT / "target_analysis"

for _d in (DIST_DIR, CORR_DIR, TGTD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading parquet …")
df = pd.read_parquet(PARQUET_PATH)
TARGET = "target_pgv_z_mms"
print(f"  {df.shape[0]:,} rows  x  {df.shape[1]} columns")

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


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Distributions
# ══════════════════════════════════════════════════════════════════════════════
if RUN_1_DISTRIBUTIONS:
    print("\n=== Analysis 1: Distributions ===")

    # ── 1A  Target ────────────────────────────────────────────────────────────
    print("[1A] Target distribution …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Target: PGV-Z (mm/s)", fontsize=13, fontweight="bold")

    vals = df[TARGET].dropna()
    _log_hist(axes[0], vals, "steelblue", "PGV-Z (mm/s)", "Raw values")
    _log_hist(axes[1], vals, "darkorange", "log1p(PGV-Z)", "log1p-transformed")
    # override x on second panel — log1p is already fine linear
    axes[1].clear()
    log_vals = np.log1p(vals)
    axes[1].hist(
        log_vals, bins=50, color="darkorange", edgecolor="white", linewidth=0.2
    )
    axes[1].set_xlabel("log1p(PGV-Z)", fontsize=8)
    axes[1].set_ylabel("Count", fontsize=8)
    axes[1].set_title("log1p-transformed", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    # stats annotation
    for ax, v, lbl in zip(axes, [vals, log_vals], ["raw", "log1p"]):
        ax.axvline(
            v.median(),
            color="red",
            linewidth=1.2,
            linestyle="--",
            label=f"median={v.median():.3f}",
        )
        ax.legend(fontsize=7)

    fig.tight_layout()
    _savefig(fig, DIST_DIR, "01A_target_distribution.png")

    # ── 1B  Geometry features ─────────────────────────────────────────────────
    print("[1B] Geometry features …")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Geometry features", fontsize=13, fontweight="bold")

    # distance histogram
    _log_hist(
        axes[0],
        df["acc_distance_to_track_m"],
        "mediumseagreen",
        "Distance to track (m)",
        "acc_distance_to_track_m",
    )

    # side of track bar
    side_cts = df["acc_side_of_track"].value_counts().sort_index()
    axes[1].bar(side_cts.index.astype(str), side_cts.values, color="slateblue")
    axes[1].set_xlabel("acc_side_of_track", fontsize=8)
    axes[1].set_ylabel("Count", fontsize=8)
    axes[1].set_title("acc_side_of_track", fontsize=9)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, v in zip(axes[1].patches, side_cts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # track number bar
    trk_cts = df["track_number"].value_counts().sort_index()
    axes[2].bar(trk_cts.index.astype(str), trk_cts.values, color="coral")
    axes[2].set_xlabel("track_number", fontsize=8)
    axes[2].set_ylabel("Count", fontsize=8)
    axes[2].set_title("track_number", fontsize=9)
    axes[2].tick_params(labelsize=7)
    axes[2].grid(axis="y", alpha=0.3)
    for bar, v in zip(axes[2].patches, trk_cts.values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    _savefig(fig, DIST_DIR, "01B_geometry_features.png")

    # ── 1C  Train metadata ────────────────────────────────────────────────────
    print("[1C] Train metadata …")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Train metadata", fontsize=13, fontweight="bold")

    # train_type bar (use unique events to avoid per-sensor duplication)
    ev_df = df.drop_duplicates("event_id")
    tt_cts = ev_df["train_type"].value_counts()
    bars = axes[0].barh(tt_cts.index, tt_cts.values, color="steelblue")
    axes[0].set_xlabel("Unique events", fontsize=8)
    axes[0].set_title("train_type (unique events)", fontsize=9)
    axes[0].tick_params(labelsize=7)
    axes[0].grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, tt_cts.values):
        axes[0].text(
            v + 2, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", fontsize=7
        )

    # train_speed histogram
    _log_hist(
        axes[1],
        ev_df["train_speed_kmh"],
        "mediumseagreen",
        "Speed (km/h)",
        "train_speed_kmh (unique events)",
    )

    # track_number per train_type stacked bar
    pivot = df.groupby(["train_type", "track_number"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", ax=axes[2], colormap="tab10", width=0.7)
    axes[2].set_xlabel("train_type", fontsize=8)
    axes[2].set_ylabel("Row count", fontsize=8)
    axes[2].set_title("Rows per train_type × track", fontsize=9)
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
    for ax, col, clr in zip(axes.flat, td_cols, colors):
        _log_hist(ax, df[col], clr, col, col.replace("fo_td_", ""), log_x=True)
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
                _log_hist(ax, df[col], clr, col, f"{lbl} Hz", log_x=True)
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
            "train_speed_kmh",
            "track_number",
            "sensor_id",
        ]
    ].dropna()

    # ── 4A  Coloured by train_type ───────────────────────────────────────
    print("[4A] Scatter coloured by train_type …")
    train_types = sorted(sub["train_type"].unique())
    palette = sns.color_palette("tab10", len(train_types))
    color_map = dict(zip(train_types, palette))

    fig, ax = plt.subplots(figsize=(10, 6))
    for tt in train_types:
        mask = sub["train_type"] == tt
        ax.scatter(
            sub.loc[mask, "acc_distance_to_track_m"],
            sub.loc[mask, TARGET],
            s=10,
            alpha=0.35,
            color=color_map[tt],
            label=tt,
            rasterized=True,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance to track (m)  [log]", fontsize=10)
    ax.set_ylabel("PGV-Z (mm/s)  [log]", fontsize=10)
    ax.set_title(
        "Target vs distance — coloured by train_type", fontsize=12, fontweight="bold"
    )
    ax.legend(
        title="train_type", fontsize=7, title_fontsize=8, markerscale=2, framealpha=0.8
    )
    ax.grid(which="both", alpha=0.2)
    fig.tight_layout()
    _savefig(fig, TGTD_DIR, "04A_target_vs_distance_traintype.png")

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
        nrows, ncols, figsize=(ncols * 4, nrows * 3.2), sharex=False, sharey=False
    )
    fig.suptitle("PGV-Z vs distance — per sensor", fontsize=13, fontweight="bold")
    for ax, sid in zip(axes.flat, sensors):
        sdf = sub[sub["sensor_id"] == sid]
        for tt in train_types:
            mask = sdf["train_type"] == tt
            if mask.sum() == 0:
                continue
            ax.scatter(
                sdf.loc[mask, "acc_distance_to_track_m"],
                sdf.loc[mask, TARGET],
                s=8,
                alpha=0.4,
                color=color_map[tt],
                label=tt,
                rasterized=True,
            )
        dist_val = sdf["acc_distance_to_track_m"].median()
        ax.set_title(f"{sid}  (d={dist_val:.1f} m)", fontsize=8, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlabel("dist (m)", fontsize=7)
        ax.set_ylabel("PGV-Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
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
            markerfacecolor=color_map[tt],
            markersize=6,
            label=tt,
        )
        for tt in train_types
    ]
    fig.legend(
        handles=handles,
        title="train_type",
        fontsize=7,
        title_fontsize=8,
        loc="lower right",
        ncol=2,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _savefig(fig, TGTD_DIR, "04C_target_vs_distance_per_sensor.png")

    print("[4] Target vs distance done.")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5 — Seaborn pair plot (top-N features by XGB gain + target)
# ══════════════════════════════════════════════════════════════════════════════
if RUN_5_PAIRPLOT:
    print("\n=== Analysis 5: Pair plot ===")

    fi = pd.read_csv(IMPORTANCE_CSV)
    top_feats = fi.sort_values("gain", ascending=False).head(TOP_N)["feature"].tolist()
    top_feats = [f for f in top_feats if f in df.columns]

    # Build working frame including raw target for binning
    pair_df = df[top_feats + [TARGET]].copy()

    # PGV-Z colour bins
    BIN_EDGES = [0, 1, 2, 3, 4, np.inf]
    BIN_LABELS = ["0-1", "1-2", "2-3", "3-4", "4+"]
    pair_df["pgvz_bin"] = pd.cut(
        pair_df[TARGET], bins=BIN_EDGES, labels=BIN_LABELS, right=False
    ).astype(str)
    pair_df = pair_df.drop(columns=[TARGET])

    # Log-transform positive FO features for readability
    for col in top_feats:
        if pair_df[col].gt(0).all():
            pair_df[col] = np.log1p(pair_df[col])
            pair_df.rename(columns={col: f"log_{col}"}, inplace=True)

    # Subsample to keep the plot fast (seaborn pairplot is O(n^2) per cell)
    MAX_ROWS = 3000
    if len(pair_df) > MAX_ROWS:
        pair_df = pair_df.sample(MAX_ROWS, random_state=42)

    feat_cols_plot = [c for c in pair_df.columns if c != "pgvz_bin"]
    print(
        f"  Plotting pair plot with {len(feat_cols_plot)} variables "
        f"on {len(pair_df):,} rows …"
    )

    BIN_PALETTE = {
        "0-1": "#4daf4a",  # green
        "1-2": "#377eb8",  # blue
        "2-3": "#ff7f00",  # orange
        "3-4": "#e41a1c",  # red
        "4+": "#984ea3",  # purple
    }

    pg = sns.pairplot(
        pair_df,
        vars=feat_cols_plot,
        hue="pgvz_bin",
        hue_order=BIN_LABELS,
        palette=BIN_PALETTE,
        diag_kind="kde",
        plot_kws={"alpha": 0.3, "s": 10, "rasterized": True},
        diag_kws={"fill": True, "alpha": 0.4},
    )
    pg.figure.suptitle(
        f"Pair plot — top {TOP_N} features by XGB gain  │  coloured by PGV-Z bin (mm/s)",
        y=1.01,
        fontsize=12,
        fontweight="bold",
    )
    pg._legend.set_title("PGV-Z (mm/s)")
    out_path = BASE_OUT / "05_pairplot_top_features.png"
    pg.figure.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(pg.figure)
    print(f"  Saved: {out_path.relative_to(BASE_OUT)}")
    print("[5] Pair plot done.")

print("\nDone.")
