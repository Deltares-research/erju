"""PGV_z attenuation exploration — side=-1 (left) sensors only.

Uses the full Parquet dataset (all events, not just the held-out test split).

Outputs:
  <OUTPUT_DIR>/individual/<event_id>.png   — one plot per event
  <OUTPUT_DIR>/summary_all.png             — all events pooled, mean ± 1 std
  <OUTPUT_DIR>/summary_by_traintype.png    — grouped by train_type
  <OUTPUT_DIR>/summary_by_track.png        — grouped by track_number

Usage:
  python src/ml/explore_pgvz_attenuation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# USER INPUT
# =============================================================================

PARQUET = Path(
    r"P:\11210978-erju-ai\holten_parquet\parquet_v001_20260406_031129\dataset.parquet"
)
OUTPUT_DIR = Path(r"P:\11210978-erju-ai\holten_other\atenuation_plots")

SIDE_FILTER = -1  # -1 = left, 0 = unknown, +1 = right

# Exclude in-track sensors (acceleration_g, not velocity_mms)
EXCLUDE_SENSOR_IDS = ["MP14", "MP15", "MP16", "MP17", "MP18", "MP19"]

# =============================================================================

_TINY = 1e-4  # lower floor for log scale (mm/s)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = df[~df["sensor_id"].isin(EXCLUDE_SENSOR_IDS)]
    df = df[df["acc_side_of_track"] == SIDE_FILTER]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Individual event plots
# ---------------------------------------------------------------------------


def _plot_individual(event_id: str, event_df: pd.DataFrame, output_dir: Path) -> None:
    # Average within (distance, sensor) in case of duplicates, then sort
    plot_df = (
        event_df.groupby(["sensor_id", "acc_distance_to_track_m"])
        .agg(pgv_z=("target_pgv_z_mms", "mean"))
        .reset_index()
        .sort_values("acc_distance_to_track_m")
    )

    if plot_df.empty or plot_df["pgv_z"].le(0).all():
        return

    # Event metadata from first row
    meta = event_df.iloc[0]
    train_type = meta.get("train_type", "?")
    raw_speed = meta.get("train_speed_kmh", float("nan"))
    speed_str = f"{raw_speed:.0f} km/h" if pd.notna(raw_speed) else "?"
    track_num = meta.get("track_number", "?")

    # Connecting line uses per-distance mean across sensors
    dist_mean = (
        plot_df.groupby("acc_distance_to_track_m")["pgv_z"]
        .mean()
        .reset_index()
        .sort_values("acc_distance_to_track_m")
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        dist_mean["acc_distance_to_track_m"],
        dist_mean["pgv_z"],
        "--",
        color="#4878CF",
        lw=1.0,
        alpha=0.55,
        zorder=2,
    )
    ax.scatter(
        plot_df["acc_distance_to_track_m"],
        plot_df["pgv_z"],
        s=80,
        color="#4878CF",
        edgecolors="white",
        linewidths=0.6,
        zorder=5,
        label="Measured PGV$_z$",
    )

    # Annotate sensor IDs
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["sensor_id"],
            xy=(row["acc_distance_to_track_m"], row["pgv_z"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            color="#555555",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance to track (m)", fontsize=10)
    ax.set_ylabel("PGV$_z$ (mm/s)", fontsize=10)
    ax.set_title(
        f"PGV$_z$ attenuation — {event_id}\n"
        f"Train type: {train_type}  |  Speed: {speed_str}  |  Track: {track_num}  |  Side: left",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe = event_id.replace(":", "-").replace("/", "-").replace("\\", "-")
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot helpers
# ---------------------------------------------------------------------------


def _agg_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    """One value per event × distance (average sensors at same dist), then stats across events."""
    per_event = (
        df.groupby(["event_id", "acc_distance_to_track_m"])["target_pgv_z_mms"]
        .mean()
        .reset_index()
    )
    stats = (
        per_event.groupby("acc_distance_to_track_m")["target_pgv_z_mms"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("acc_distance_to_track_m")
    )
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def _draw_band(ax, stats: pd.DataFrame, color: str, label: str) -> None:
    x = stats["acc_distance_to_track_m"].values
    mean = stats["mean"].values
    std = stats["std"].values

    lo = np.maximum(mean - std, _TINY)
    hi = mean + std

    ax.fill_between(x, lo, hi, alpha=0.22, color=color, zorder=2)
    ax.plot(
        x,
        mean,
        "o-",
        color=color,
        lw=1.8,
        ms=6,
        zorder=5,
        label=f"{label}  (n={int(stats['count'].mean())} ev/dist)",
    )
    # Invisible patch for the band legend entry
    ax.fill_between([], [], [], alpha=0.4, color=color, label=f"{label} ±1 std")


def _finalise_ax(ax, title: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance to track (m)", fontsize=10)
    ax.set_ylabel("PGV$_z$ (mm/s)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------


def _plot_summary_all(df: pd.DataFrame, output_dir: Path) -> None:
    stats = _agg_by_distance(df)
    n_events = df["event_id"].nunique()

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_band(ax, stats, color="#4878CF", label="All events")
    _finalise_ax(
        ax,
        f"PGV$_z$ attenuation — all events pooled  (N={n_events}, side: left)\n"
        "Mean ± 1 std across events per distance",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_by_group(
    df: pd.DataFrame,
    group_col: str,
    title_label: str,
    output_dir: Path,
) -> list[str]:
    """One PNG per group value, saved to output_dir/."""
    groups = sorted(df[group_col].dropna().unique())
    saved: list[str] = []
    for grp in groups:
        sub = df[df[group_col] == grp]
        if sub.empty:
            continue
        n_grp = sub["event_id"].nunique()
        stats = _agg_by_distance(sub)

        fig, ax = plt.subplots(figsize=(8, 5))
        _draw_band(ax, stats, color="#4878CF", label=str(grp))
        _finalise_ax(
            ax,
            f"PGV$_z$ attenuation — {title_label}: {grp}  (N={n_grp} events, side: left)\n"
            "Mean ± 1 std across events per distance",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_grp = str(grp).replace(" ", "_").replace("/", "-")
        fname = f"summary_{group_col}_{safe_grp}.png"
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("  PGV_z attenuation explorer")
    print("=" * 60)
    print("Loading data ...")
    # Load full dataset first to report how many events exist before side filter
    df_all = pd.read_parquet(PARQUET)
    df_all = df_all[~df_all["sensor_id"].isin(EXCLUDE_SENSOR_IDS)]
    n_total = df_all["event_id"].nunique()

    df = _load_data()
    n_events = df["event_id"].nunique()
    n_dropped = n_total - n_events
    print(f"  Total events (all sides) : {n_total}")
    print(f"  Events with side={SIDE_FILTER} sensors: {n_events}")
    if n_dropped:
        print(f"  Note: {n_dropped} events had no side={SIDE_FILTER} sensor → excluded from all plots")
    print(f"  Rows used: {len(df)}")

    # --- Individual plots ---
    indiv_dir = OUTPUT_DIR / "individual"
    print(f"\nGenerating {n_events} individual plots → {indiv_dir}")
    events = sorted(df["event_id"].unique())
    for idx, eid in enumerate(events, 1):
        if idx % 100 == 0 or idx == n_events:
            print(f"  {idx}/{n_events} ...")
        _plot_individual(eid, df[df["event_id"] == eid], indiv_dir)
    print("  Done.")

    # --- Summary plots ---
    print(f"\nGenerating summary plots → {OUTPUT_DIR}")
    _plot_summary_all(df, OUTPUT_DIR)
    print("  summary_all.png saved.")

    saved_tt = _plot_summary_by_group(df, "train_type", "train type", OUTPUT_DIR)
    for f in saved_tt:
        print(f"  {f} saved.")

    saved_tr = _plot_summary_by_group(df, "track_number", "track number", OUTPUT_DIR)
    for f in saved_tr:
        print(f"  {f} saved.")

    print(f"\nAll done. Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
