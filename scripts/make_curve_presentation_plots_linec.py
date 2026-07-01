"""
Attenuation-curve presentation plots for Holten Line-C PGV_z.

Generates publication-ready figures from completed model predictions.

Usage:
    python scripts/make_curve_presentation_plots_linec.py

Inputs (auto-discovered, most recent preferred):
    1. P3_corrected_n    : cnn_curveprior_p3_savepred_linec_v001_*/predictions.parquet
       (fallback: none — skip P3 plots if not available)
    2. Q3_all_initfix    : cnn_curvequery_initfix_linec_v001_vQ3_all_*/predictions.parquet
       (fallback: cnn_curvequery_stability_linec_v001_vQ3_all_*/predictions.parquet)
    3. Q3_holdout_MP2    : *_vQ3_holdout_MP2_*/predictions.parquet + held_predictions.parquet
    4. Q3_holdout_MP4    : *_vQ3_holdout_MP4_*/predictions.parquet + held_predictions.parquet

Outputs:
    plots/curve_presentation_YYYYMMDD/
        01_measured_vs_predicted_loglog.png
        02_per_sensor_error_bar.png
        03_residual_vs_distance.png
        04_attenuation_curve_representative.png
        05_attenuation_curve_high_pgv.png
        06_attenuation_curve_failures.png
        07_heldout_MP2_examples.png
        08_heldout_MP4_examples.png
        summary_metric_table.csv
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ============================================================================
# Paths
# ============================================================================

if os.name == "nt":
    MODELS_ROOT = Path(r"P:\11210978-erju-ai\holten_models")
else:
    MODELS_ROOT = Path("/p/11210978-erju-ai/holten_models")

OUTPUT_DIR = _REPO_ROOT / "plots" / f"curve_presentation_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensor order (near → far)
SENSOR_ORDER  = ["MP4", "MP8", "MP10", "MP1", "MP2"]
SENSOR_DIST_T1 = [2.5, 4.0, 8.0, 16.0, 23.0]
SENSOR_DIST_T2 = [6.5, 8.0, 12.0, 20.0, 27.0]

# Colour palette
COLOURS = {
    "P3":    "#1f77b4",   # blue
    "Q3":    "#2ca02c",   # green
    "Q3_lf": "#7fba7a",   # light green (stability fallback)
    "measured": "#333333",
    "physics": "#d62728",  # red
}

# ============================================================================
# Discovery helpers
# ============================================================================

def _find_latest(pattern: str) -> Optional[Path]:
    """Return most recent matching directory under MODELS_ROOT."""
    hits = sorted(MODELS_ROOT.glob(pattern), key=lambda p: p.name)
    return hits[-1] if hits else None


def _load_preds(dir_path: Path, filename: str = "predictions.parquet") -> Optional[pd.DataFrame]:
    if dir_path is None:
        return None
    p = dir_path / filename
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df


def discover_datasets() -> Dict[str, Optional[pd.DataFrame]]:
    """Auto-discover all prediction files, newest first."""
    datasets: Dict[str, Optional[pd.DataFrame]] = {}

    # P3 savepred (Task 1 output)
    d = _find_latest("cnn_curveprior_p3_savepred_linec_v001_vP3_*")
    datasets["P3"] = _load_preds(d) if d else None
    datasets["P3_dir"] = d

    # Q3 all-sensor: prefer initfix, fallback to stability
    d = _find_latest("cnn_curvequery_initfix_linec_v001_vQ3_all_*")
    if d is None:
        d = _find_latest("cnn_curvequery_stability_linec_v001_vQ3_all_*")
        label = "Q3_stability_fallback"
    else:
        label = "Q3_initfix"
    datasets["Q3_label"] = label
    datasets["Q3"] = _load_preds(d) if d else None
    datasets["Q3_dir"] = d

    # Q3 holdout MP2
    d_q3_mp2 = _find_latest("cnn_curvequery_initfix_linec_v001_vQ3_holdout_MP2_*")
    if d_q3_mp2 is None:
        d_q3_mp2 = _find_latest("cnn_curvequery_stability_linec_v001_vQ3_holdout_MP2_*")
    datasets["Q3_holdout_MP2_nonheld"] = _load_preds(d_q3_mp2) if d_q3_mp2 else None
    datasets["Q3_holdout_MP2_held"]    = _load_preds(d_q3_mp2, "held_predictions.parquet") if d_q3_mp2 else None

    # Q3 holdout MP4
    d_q3_mp4 = _find_latest("cnn_curvequery_initfix_linec_v001_vQ3_holdout_MP4_*")
    if d_q3_mp4 is None:
        d_q3_mp4 = _find_latest("cnn_curvequery_stability_linec_v001_vQ3_holdout_MP4_*")
    datasets["Q3_holdout_MP4_nonheld"] = _load_preds(d_q3_mp4) if d_q3_mp4 else None
    datasets["Q3_holdout_MP4_held"]    = _load_preds(d_q3_mp4, "held_predictions.parquet") if d_q3_mp4 else None

    return datasets


# ============================================================================
# Event selection helpers
# ============================================================================

def select_representative_events(df: pd.DataFrame, n: int = 6) -> List[str]:
    """Select events with low median absolute log error, target PGV ≥ 0.5 mm/s."""
    ev_err = (
        df.assign(abs_err=lambda d: np.abs(d["pred_log"] - d["target_log"]))
          .groupby("event_id")["abs_err"].median()
          .sort_values()
    )
    # Exclude pure near-zero events
    ev_maxpgv = df.groupby("event_id")["target_pgv"].max()
    good = ev_maxpgv[ev_maxpgv >= 1.0].index
    candidates = ev_err[ev_err.index.isin(good)]
    return list(candidates.head(n).index)


def select_high_pgv_events(df: pd.DataFrame, n: int = 6) -> List[str]:
    """Select events by maximum target PGV."""
    ev_max = df.groupby("event_id")["target_pgv"].max().sort_values(ascending=False)
    return list(ev_max.head(n).index)


def select_failure_events(df: pd.DataFrame, n: int = 6) -> List[str]:
    """Select events with highest RMSE(PGV)."""
    ev_rmse = (
        df.assign(sq=(lambda d: (d["pred_pgv"] - d["target_pgv"]) ** 2))
          .groupby("event_id")["sq"].mean()
          .apply(np.sqrt)
          .sort_values(ascending=False)
    )
    return list(ev_rmse.head(n).index)


# ============================================================================
# Metric summary
# ============================================================================

def compute_metrics(df: pd.DataFrame, label: str) -> Dict:
    p = df["pred_log"].values
    t = df["target_log"].values
    pp = df["pred_pgv"].values
    tp = df["target_pgv"].values
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return {
        "model": label,
        "n_events": df["event_id"].nunique(),
        "n_rows": len(df),
        "rmse_log": float(np.sqrt(np.mean((p - t) ** 2))),
        "rmse_pgv": float(np.sqrt(np.mean((pp - tp) ** 2))),
        "r2_log":   float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias_pgv": float(np.mean(pp - tp)),
    }


# ============================================================================
# Plot helpers
# ============================================================================

def _style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 120,
    })


def plot_profile_panel(ax, df: pd.DataFrame, event_id: str,
                       label: str, colour: str,
                       n_track1: float = 1.0655, n_track2: float = 1.3246,
                       r0: float = 10.0,
                       show_physics: bool = True) -> None:
    """Plot one attenuation profile (measured + predicted + physics) on ax."""
    ev = df[df["event_id"] == event_id].sort_values("distance")
    if ev.empty:
        ax.set_visible(False)
        return

    dist  = ev["distance"].values
    tpgv  = ev["target_pgv"].values
    ppgv  = ev["pred_pgv"].values
    track = int(ev["track"].iloc[0])
    n     = n_track1 if track == 1 else n_track2

    ax.semilogy(dist, tpgv, "ko-", ms=5, lw=1.2, label="Measured", zorder=3)
    ax.semilogy(dist, ppgv, color=colour, marker="^", linestyle="--",
                ms=5, lw=1.2, label=label, zorder=3)

    if show_physics and "c_hat" in ev.columns:
        c_hat = ev["c_hat"].iloc[0]
        r_smooth = np.linspace(2.0, 28.0, 120)
        pgv_curve = np.exp(c_hat - n * np.log(r_smooth / r0))
        ax.semilogy(r_smooth, pgv_curve, color=COLOURS["physics"],
                    linestyle=":", lw=1.0, alpha=0.7, label="Physics curve")

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("PGV (mm/s)")
    ax.set_title(f"Event: {str(event_id)[-12:-4]}\nTrack {track}", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, 30)


# ============================================================================
# Individual plot functions
# ============================================================================

def plot_loglog(datasets: Dict, output_dir: Path) -> None:
    """Plot 1: measured vs predicted log-log scatter."""
    _style()
    items = []
    if datasets.get("P3") is not None:
        items.append(("P3_corrected_n", datasets["P3"], COLOURS["P3"]))
    if datasets.get("Q3") is not None:
        lbl = datasets.get("Q3_label", "Q3")
        items.append((lbl, datasets["Q3"], COLOURS["Q3"]))

    if not items:
        print("  [skip] plot_loglog: no data")
        return

    n_plots = len(items)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), squeeze=False)
    for ax, (label, df, colour) in zip(axes[0], items):
        ax.scatter(df["target_pgv"], df["pred_pgv"], s=4, alpha=0.3,
                   color=colour, rasterized=True)
        lim = max(df["target_pgv"].max(), df["pred_pgv"].max()) * 1.05
        ax.plot([0.05, lim], [0.05, lim], "r--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Measured PGV (mm/s)")
        ax.set_ylabel("Predicted PGV (mm/s)")
        rmse = np.sqrt(np.mean((df["pred_pgv"] - df["target_pgv"]) ** 2))
        ax.set_title(f"{label}\nRMSE = {rmse:.3f} mm/s")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Measured vs Predicted — Line-C side -1 (test set)", fontsize=12)
    fig.tight_layout()
    p = output_dir / "01_measured_vs_predicted_loglog.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_per_sensor_error_bar(datasets: Dict, output_dir: Path) -> None:
    """Plot 2: per-sensor RMSE(log) bar chart."""
    _style()
    items = []
    if datasets.get("P3") is not None:
        items.append(("P3_corrected_n", datasets["P3"], COLOURS["P3"]))
    if datasets.get("Q3") is not None:
        lbl = datasets.get("Q3_label", "Q3")
        items.append((lbl, datasets["Q3"], COLOURS["Q3"]))
    if not items:
        print("  [skip] plot_per_sensor_error_bar: no data")
        return

    x = np.arange(len(SENSOR_ORDER))
    width = 0.35 / max(1, len(items))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, df, colour) in enumerate(items):
        rmse_vals = []
        for s in SENSOR_ORDER:
            sub = df[df["sensor"] == s] if "sensor" in df.columns else pd.DataFrame()
            if len(sub) > 0:
                rmse_vals.append(float(np.sqrt(np.mean((sub["pred_log"] - sub["target_log"]) ** 2))))
            else:
                rmse_vals.append(np.nan)
        offset = (i - len(items) / 2 + 0.5) * width
        ax.bar(x + offset, rmse_vals, width, label=label, color=colour, alpha=0.8)

    ax.set_xlabel("Sensor"); ax.set_ylabel("RMSE (log)")
    ax.set_title("Per-Sensor RMSE(log) — Test set")
    ax.set_xticks(x); ax.set_xticklabels(SENSOR_ORDER)
    ax.legend(); ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(0, None)
    fig.tight_layout()
    p = output_dir / "02_per_sensor_error_bar.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_residual_vs_distance(datasets: Dict, output_dir: Path) -> None:
    """Plot 3: residual (pred - measured) PGV vs distance."""
    _style()
    items = []
    if datasets.get("P3") is not None:
        items.append(("P3_corrected_n", datasets["P3"], COLOURS["P3"]))
    if datasets.get("Q3") is not None:
        lbl = datasets.get("Q3_label", "Q3")
        items.append((lbl, datasets["Q3"], COLOURS["Q3"]))
    if not items:
        print("  [skip] plot_residual_vs_distance: no data")
        return

    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4), squeeze=False)
    for ax, (label, df, colour) in zip(axes[0], items):
        resid = df["pred_pgv"] - df["target_pgv"]
        ax.scatter(df["distance"], resid, s=4, alpha=0.3, color=colour, rasterized=True)
        ax.axhline(0, color="k", lw=0.8)
        for d_nom, s in zip(SENSOR_DIST_T1, SENSOR_ORDER):
            ax.axvline(d_nom, color="grey", lw=0.4, linestyle=":")
            ax.text(d_nom + 0.2, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 5, s,
                    fontsize=7, color="grey")
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Residual (mm/s)")
        ax.set_title(label); ax.grid(True, alpha=0.3)
    fig.suptitle("Residual (predicted − measured) vs Distance", fontsize=12)
    fig.tight_layout()
    p = output_dir / "03_residual_vs_distance.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def _plot_event_grid(df: pd.DataFrame, event_ids: List[str], title: str,
                     label: str, colour: str, output_path: Path) -> None:
    """Generic 2×3 attenuation curve grid."""
    _style()
    n = min(6, len(event_ids))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    for ax_i in range(n):
        plot_profile_panel(axes[ax_i], df, event_ids[ax_i],
                           label=label, colour=colour, show_physics=True)
        if ax_i == 0:
            axes[ax_i].legend(fontsize=7, loc="upper right")
    for ax_i in range(n, 6):
        axes[ax_i].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


def plot_attenuation_grids(datasets: Dict, output_dir: Path) -> None:
    """Plots 4, 5, 6: representative / high-PGV / failure grids."""
    # Choose the best available all-sensor dataset
    df = None
    if datasets.get("Q3") is not None and not datasets["Q3"].empty:
        df     = datasets["Q3"]
        label  = datasets.get("Q3_label", "Q3")
        colour = COLOURS["Q3"]
    elif datasets.get("P3") is not None and not datasets["P3"].empty:
        df     = datasets["P3"]
        label  = "P3_corrected_n"
        colour = COLOURS["P3"]

    rep_ev  = select_representative_events(df, n=6)
    hi_ev   = select_high_pgv_events(df, n=6)
    fail_ev = select_failure_events(df, n=6)

    _plot_event_grid(df, rep_ev,
                     f"Representative Attenuation Profiles — {label}",
                     label, colour,
                     output_dir / "04_attenuation_curve_representative.png")
    _plot_event_grid(df, hi_ev,
                     f"High-PGV Attenuation Profiles — {label}",
                     label, colour,
                     output_dir / "05_attenuation_curve_high_pgv.png")
    _plot_event_grid(df, fail_ev,
                     f"Failure Cases (Highest RMSE) — {label}",
                     label, colour,
                     output_dir / "06_attenuation_curve_failures.png")


def plot_heldout(held_df: Optional[pd.DataFrame],
                 nonheld_df: Optional[pd.DataFrame],
                 sensor_name: str,
                 output_path: Path) -> None:
    """Plot 7/8: held-out sensor scatter + example profiles."""
    if held_df is None or len(held_df) == 0:
        print(f"  [skip] held-out {sensor_name}: no data")
        return
    _style()
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # Left: scatter
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_sc.scatter(held_df["target_pgv"], held_df["pred_pgv"],
                  s=6, alpha=0.5, color=COLOURS["Q3"], rasterized=True)
    lim = max(held_df["target_pgv"].max(), held_df["pred_pgv"].max()) * 1.1
    ax_sc.plot([0.05, lim], [0.05, lim], "r--", lw=1)
    ax_sc.set_xscale("log"); ax_sc.set_yscale("log")
    rmse = np.sqrt(np.mean((held_df["pred_pgv"] - held_df["target_pgv"]) ** 2))
    bias = np.mean(held_df["pred_log"] - held_df["target_log"])
    ax_sc.set_xlabel("Measured PGV"); ax_sc.set_ylabel("Predicted PGV")
    ax_sc.set_title(f"Held-out {sensor_name}\nRMSE={rmse:.3f} bias_log={bias:+.3f}")
    ax_sc.grid(True, which="both", alpha=0.3)

    # Right: example profiles (need full event data from non-held part)
    if nonheld_df is not None:
        # Combine held + non-held to reconstruct full events
        # Held data has all 5 sensor types in some runs, or just the holdout sensor
        all_df = pd.concat([nonheld_df, held_df], ignore_index=True)
        # Pick events that appear in held_df (test events)
        held_events = held_df["event_id"].unique()
        # Select 3 representative events from held data
        held_err = (
            held_df.assign(abs_err=lambda d: np.abs(d["pred_log"] - d["target_log"]))
                   .groupby("event_id")["abs_err"].median()
                   .sort_values()
        )
        n_good = min(3, len(held_err))
        sel_evs = list(held_err.head(n_good).index)
    else:
        all_df  = held_df
        sel_evs = list(held_df["event_id"].unique()[:3])

    for ax_i, ev in enumerate(sel_evs[:3]):
        ax = fig.add_subplot(gs[0, ax_i + 1])
        # Plot held-out sensor point
        ev_held = held_df[held_df["event_id"] == ev]
        if not ev_held.empty:
            ax.semilogy(ev_held["distance"].values,
                        ev_held["target_pgv"].values,
                        "ko", ms=8, label=f"{sensor_name} measured", zorder=5)
            ax.semilogy(ev_held["distance"].values,
                        ev_held["pred_pgv"].values,
                        "r^", ms=8, label=f"{sensor_name} predicted", zorder=5)
        # Plot other sensors if available
        ev_others = all_df[(all_df["event_id"] == ev) &
                           (all_df["sensor"] != sensor_name)].sort_values("distance")
        if not ev_others.empty:
            ax.semilogy(ev_others["distance"].values,
                        ev_others["target_pgv"].values,
                        "k--", alpha=0.4, lw=0.8, ms=3, marker="o")
            ax.semilogy(ev_others["distance"].values,
                        ev_others["pred_pgv"].values,
                        color=COLOURS["Q3"], alpha=0.5, lw=0.8, linestyle="--",
                        ms=3, marker="^")
        # Physics curve
        if "c_hat" in all_df.columns:
            c_ev = all_df[all_df["event_id"] == ev]
            if not c_ev.empty:
                track = int(c_ev["track"].iloc[0])
                c_hat = c_ev["c_hat"].iloc[0]
                n     = 1.0655 if track == 1 else 1.3246
                r_sm  = np.linspace(2.0, 28.0, 120)
                ax.semilogy(r_sm, np.exp(c_hat - n * np.log(r_sm / 10.0)),
                            color=COLOURS["physics"], linestyle=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("PGV (mm/s)")
        ax.set_title(f"Event {str(ev)[-12:-4]}", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(1, 30)
        if ax_i == 0:
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(f"Held-out {sensor_name} — Zero-shot prediction (never seen in training)",
                 fontsize=11)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


def save_metric_table(datasets: Dict, output_dir: Path) -> None:
    """Save summary_metric_table.csv."""
    rows = []
    known_baselines = [
        {"model": "Row-wise CNN v2 line-C",   "rmse_log": 0.595,  "rmse_pgv": 2.348, "r2_log": 0.654, "notes": "validated baseline"},
        {"model": "P1_corrected_n",            "rmse_log": 0.6146, "rmse_pgv": 2.483, "r2_log": 0.643, "notes": "fixed-output, no residual"},
        {"model": "P3_corrected_n (original)", "rmse_log": 0.5993, "rmse_pgv": 2.401, "r2_log": 0.660, "notes": "fixed-output, best structured model"},
        {"model": "Q2_all_lr1e4",              "rmse_log": 0.6128, "rmse_pgv": 2.416, "r2_log": 0.646, "notes": "stability run, chaotic"},
        {"model": "Q3_all_lr1e4",              "rmse_log": 0.6196, "rmse_pgv": 2.375, "r2_log": None,  "notes": "stability run, chaotic"},
    ]
    rows.extend(known_baselines)

    for key, label in [("P3", "P3_savepred"), ("Q3", datasets.get("Q3_label", "Q3_initfix"))]:
        df = datasets.get(key)
        if df is not None:
            m = compute_metrics(df, label)
            m["notes"] = "from predictions.parquet"
            rows.append(m)

    # Held-out metrics
    for sensor_key, label in [("Q3_holdout_MP2_held", "Q3_holdout_MP2 (held)"),
                               ("Q3_holdout_MP4_held", "Q3_holdout_MP4 (held)")]:
        df = datasets.get(sensor_key)
        if df is not None and len(df) > 0:
            m = compute_metrics(df, label)
            m["notes"] = "held-out sensor only"
            rows.append(m)

    out_df = pd.DataFrame(rows)
    p = output_dir / "summary_metric_table.csv"
    out_df.to_csv(p, index=False)
    print(f"  Saved {p.name}")
    print(out_df[["model", "rmse_log", "rmse_pgv", "r2_log"]].to_string(index=False))


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Attenuation Curve Presentation Plots — Holten Line-C")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    print()

    print("Discovering prediction files...")
    ds = discover_datasets()
    for key in ["P3_dir", "Q3_dir"]:
        d = ds.get(key)
        print(f"  {key}: {d.name if d else 'NOT FOUND'}")
    print()

    print("1. Measured vs Predicted log-log")
    plot_loglog(ds, OUTPUT_DIR)

    print("2. Per-sensor error bar")
    plot_per_sensor_error_bar(ds, OUTPUT_DIR)

    print("3. Residual vs distance")
    plot_residual_vs_distance(ds, OUTPUT_DIR)

    print("4-6. Attenuation curve grids (representative / high-PGV / failures)")
    plot_attenuation_grids(ds, OUTPUT_DIR)

    print("7. Held-out MP2 examples")
    plot_heldout(
        held_df=ds.get("Q3_holdout_MP2_held"),
        nonheld_df=ds.get("Q3_holdout_MP2_nonheld"),
        sensor_name="MP2",
        output_path=OUTPUT_DIR / "07_heldout_MP2_examples.png",
    )

    print("8. Held-out MP4 examples")
    plot_heldout(
        held_df=ds.get("Q3_holdout_MP4_held"),
        nonheld_df=ds.get("Q3_holdout_MP4_nonheld"),
        sensor_name="MP4",
        output_path=OUTPUT_DIR / "08_heldout_MP4_examples.png",
    )

    print("9. Summary metric table")
    save_metric_table(ds, OUTPUT_DIR)

    print()
    print(f"Done. Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
