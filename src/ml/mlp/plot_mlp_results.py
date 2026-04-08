"""plot_mlp_results.py — Generate diagnostic plots for a trained MLP build folder.

Run from project root:
    python src/ml/mlp/plot_mlp_results.py

Edit BUILD_DIR to point at any mlp build folder.
All plots are saved into BUILD_DIR/plots/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Target build folder ───────────────────────────────────────────────────────
BUILD_DIR = Path(r"P:\11210978-erju-ai\holten_models\mlp_v005_20260408_183514")

# ── Load artefacts ────────────────────────────────────────────────────────────
hist = pd.read_csv(BUILD_DIR / "training_history.csv")
summ = json.load(open(BUILD_DIR / "summary.json"))
pred = pd.read_parquet(BUILD_DIR / "predictions.parquet")

PLOTS_DIR = BUILD_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

best_epoch = summ["best_epoch"]
val_rmse = summ["val_rmse_mms"]
test_rmse = summ["test_rmse_mms"]
version = summ["version_name"]
n_features = summ["n_features"]
n_params = summ["n_params"]

val_pred = pred[pred["split"] == "val"].copy()
test_pred = pred[pred["split"] == "test"].copy()


def _savefig(fig: plt.Figure, name: str) -> None:
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: plots/{name}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Spearman": float(rho)}


print(f"\nGenerating plots for {version} ...")


# ── 1. Learning curves ────────────────────────────────────────────────────────
print("[1] Learning curves ...")
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    hist["epoch"],
    hist["train_loss"],
    color="steelblue",
    linewidth=1.5,
    label="Train loss (MSE, log-space)",
)
ax.plot(
    hist["epoch"],
    hist["val_loss"],
    color="darkorange",
    linewidth=1.5,
    label="Val loss (MSE, log-space)",
)

# Best epoch marker
best_val = hist.loc[hist["epoch"] == best_epoch, "val_loss"].values[0]
ax.axvline(best_epoch, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
ax.scatter(
    [best_epoch],
    [best_val],
    color="red",
    zorder=5,
    s=60,
    label=f"Best epoch {best_epoch}  (val={best_val:.5f})",
)

# Early-stop epoch (last epoch trained)
last_epoch = int(hist["epoch"].max())
if last_epoch != best_epoch:
    ax.axvline(
        last_epoch,
        color="gray",
        linewidth=1.0,
        linestyle=":",
        alpha=0.7,
        label=f"Early stop epoch {last_epoch}",
    )

ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("MSE loss (log1p space)", fontsize=10)
ax.set_title(
    f"{version} — Learning curves\n"
    f"Arch: 79 -> 64 -> 1  |  {n_params:,} params  |  "
    f"Val RMSE {val_rmse:.3f} mm/s  |  Test RMSE {test_rmse:.3f} mm/s",
    fontsize=11,
    fontweight="bold",
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(left=1)
fig.tight_layout()
_savefig(fig, "01_learning_curves.png")


# ── 2. Predicted vs Actual (val + test, log-log axes) ─────────────────────────
print("[2] Predicted vs Actual ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, df_split, label, color in zip(
    axes,
    [val_pred, test_pred],
    ["Validation", "Test"],
    ["steelblue", "darkorange"],
):
    y_true = df_split["target_pgv_z_mms"].values
    y_pred_v = df_split["y_pred"].values
    m = _metrics(y_true, y_pred_v)

    ax.scatter(y_true, y_pred_v, s=8, alpha=0.35, color=color, rasterized=True)

    # Perfect-prediction line
    lim_lo = min(y_true.min(), y_pred_v.min()) * 0.9
    lim_hi = max(y_true.max(), y_pred_v.max()) * 1.1
    ax.plot(
        [lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1.0, alpha=0.7, label="y=x"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual PGV-Z (mm/s)", fontsize=9)
    ax.set_ylabel("Predicted PGV-Z (mm/s)", fontsize=9)
    ax.set_title(
        f"{label} set\n"
        f"RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  "
        f"R²={m['R2']:.3f}  ρ={m['Spearman']:.3f}",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(which="both", alpha=0.2)

fig.suptitle(f"{version} — Predicted vs Actual", fontsize=12, fontweight="bold")
fig.tight_layout()
_savefig(fig, "02_predicted_vs_actual.png")


# ── 3. Residuals (y_pred - y_true) vs Actual ─────────────────────────────────
print("[3] Residuals ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, df_split, label, color in zip(
    axes,
    [val_pred, test_pred],
    ["Validation", "Test"],
    ["steelblue", "darkorange"],
):
    y_true = df_split["target_pgv_z_mms"].values
    y_pred_v = df_split["y_pred"].values
    residuals = y_pred_v - y_true

    ax.scatter(y_true, residuals, s=8, alpha=0.35, color=color, rasterized=True)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.axhline(
        residuals.mean(),
        color="red",
        linewidth=1.2,
        linestyle="-",
        label=f"Mean bias {residuals.mean():.3f} mm/s",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Actual PGV-Z (mm/s)", fontsize=9)
    ax.set_ylabel("Residual (pred - actual)  [mm/s]", fontsize=9)
    ax.set_title(f"{label} residuals", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

fig.suptitle(f"{version} — Residuals vs Actual", fontsize=12, fontweight="bold")
fig.tight_layout()
_savefig(fig, "03_residuals.png")


# ── 4. Residual histogram (val + test) ────────────────────────────────────────
print("[4] Residual histogram ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, df_split, label, color in zip(
    axes,
    [val_pred, test_pred],
    ["Validation", "Test"],
    ["steelblue", "darkorange"],
):
    y_true = df_split["target_pgv_z_mms"].values
    y_pred_v = df_split["y_pred"].values
    residuals = y_pred_v - y_true

    ax.hist(
        residuals, bins=60, color=color, edgecolor="white", linewidth=0.3, alpha=0.85
    )
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.axvline(
        residuals.mean(),
        color="red",
        linewidth=1.2,
        linestyle="-",
        label=f"Mean={residuals.mean():.3f}",
    )
    ax.axvline(
        np.median(residuals),
        color="green",
        linewidth=1.2,
        linestyle="-",
        label=f"Median={np.median(residuals):.3f}",
    )

    ax.set_xlabel("Residual (pred - actual)  [mm/s]", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(
        f"{label} — std={residuals.std():.3f} mm/s", fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

fig.suptitle(f"{version} — Residual distributions", fontsize=12, fontweight="bold")
fig.tight_layout()
_savefig(fig, "04_residual_histograms.png")


# ── 5. Error by PGV-Z bin ─────────────────────────────────────────────────────
print("[5] Error by PGV-Z bin ...")
BIN_EDGES = [0, 1, 2, 3, 4, np.inf]
BIN_LABELS = ["0-1", "1-2", "2-3", "3-4", "4+"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, df_split, label in zip(axes, [val_pred, test_pred], ["Validation", "Test"]):
    df_split = df_split.copy()
    df_split["bin"] = pd.cut(
        df_split["target_pgv_z_mms"], bins=BIN_EDGES, labels=BIN_LABELS, right=False
    )
    rmse_by_bin = (
        df_split.groupby("bin", observed=True)
        .apply(
            lambda g: np.sqrt(mean_squared_error(g["target_pgv_z_mms"], g["y_pred"])),
            include_groups=False,
        )
        .rename("rmse")
    )
    counts = df_split["bin"].value_counts().reindex(BIN_LABELS)

    colors = ["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c", "#984ea3"]
    bars = ax.bar(
        rmse_by_bin.index, rmse_by_bin.values, color=colors, edgecolor="white"
    )
    for bar, cnt in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={cnt}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xlabel("PGV-Z bin (mm/s)", fontsize=9)
    ax.set_ylabel("RMSE (mm/s)", fontsize=9)
    ax.set_title(f"{label} — RMSE per PGV-Z bin", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    f"{version} — Error by vibration magnitude bin", fontsize=12, fontweight="bold"
)
fig.tight_layout()
_savefig(fig, "05_error_by_bin.png")


# ── 6. Summary card (text-based metrics table) ────────────────────────────────
print("[6] Summary metrics card ...")
val_m = _metrics(val_pred["target_pgv_z_mms"].values, val_pred["y_pred"].values)
test_m = _metrics(test_pred["target_pgv_z_mms"].values, test_pred["y_pred"].values)

fig, ax = plt.subplots(figsize=(9, 5))
ax.axis("off")

rows = [
    ["Metric", "Validation", "Test"],
    ["RMSE (mm/s)", f"{val_m['RMSE']:.4f}", f"{test_m['RMSE']:.4f}"],
    ["MAE (mm/s)", f"{val_m['MAE']:.4f}", f"{test_m['MAE']:.4f}"],
    ["R²", f"{val_m['R2']:.4f}", f"{test_m['R2']:.4f}"],
    ["Spearman ρ", f"{val_m['Spearman']:.4f}", f"{test_m['Spearman']:.4f}"],
    ["Best epoch", str(best_epoch), "—"],
    ["Total epochs run", str(int(hist["epoch"].max())), "—"],
    ["# features", str(n_features), "—"],
    ["# params", f"{n_params:,}", "—"],
    ["Architecture", "79 -> 64 -> 1", "ReLU, no dropout"],
]

table = ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.4, 1.8)

# Header style
for j in range(3):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternating row colours
for i in range(1, len(rows)):
    clr = "#f0f4f8" if i % 2 == 0 else "white"
    for j in range(3):
        table[i, j].set_facecolor(clr)

ax.set_title(f"{version} — Training Summary", fontsize=14, fontweight="bold", pad=20)
fig.tight_layout()
_savefig(fig, "06_summary_card.png")


print(f"\nAll plots saved to: {PLOTS_DIR}")
