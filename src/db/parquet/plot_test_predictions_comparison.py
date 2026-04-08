"""Plot held-out test set predictions for XGBoost v001, v002, v003, v004.

For each version:
  - Reconstructs the exact test set using the saved split_manifest.json
    (test_event_ids) — no randomness, guaranteed identical to training run
  - Applies the same feature engineering used at training time
  - Runs model.predict() on the held-out test features
  - Back-transforms predictions if log1p target transform was used

Note: v001-v003 use Parquet v1 (linear 5-Hz FO bands).
      v004 uses Parquet v2 (1/3-octave FO bands) — loaded separately.

Output: one PNG saved to the latest v004 build folder under plots/
  test_predictions_comparison_v001_v002_v003_v004.png

Layout: 4 columns (one per version), 2 rows:
  Row 1 — Predicted vs Actual scatter (with 1:1 line + +-2 mm/s band)
  Row 2 — Residuals vs Actual (with zero line)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ml.xgboost.xgb_utils import engineer_features, prepare_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_ROOT = Path(r"P:\11210978-erju-ai\holten_models")

# Parquet v1 — used by v001, v002, v003
PARQUET_V1 = Path(
    r"P:\11210978-erju-ai\holten_parquet\parquet_v001_20260406_031129\dataset.parquet"
)
# Parquet v2 — used by v004 (1/3-octave bands)
PARQUET_V2 = Path(
    r"P:\11210978-erju-ai\holten_parquet\parquet_v002_20260408_151746\dataset.parquet"
)

VERSIONS = [
    ("xgb_v001_20260406_131814", PARQUET_V1),
    ("xgb_v002_20260406_134240", PARQUET_V1),
    ("xgb_v003_20260406_194334", PARQUET_V1),
    ("xgb_v004_20260408_165448", PARQUET_V2),
]

VERSION_LABELS = [
    "v001\n(baseline)",
    "v002\n(log + geom)",
    "v003\n(MP1-13 only)",
    "v004\n(octave bands)",
]
COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#9B59B6"]

# MLP builds to append (one entry per run)
MLP_VERSIONS = [
    "mlp_v001_20260408_180508",
]
MLP_LABELS = [
    "MLP v1\n(79→64→1)",
]
MLP_COLORS = ["#E67E22"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics_str(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return f"RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}"


def load_version(build_dir: Path, df_full: pd.DataFrame):
    """Return (y_true, y_pred, label_str) for one build folder."""
    cfg = json.loads((build_dir / "config_snapshot.json").read_text())
    mfst = json.loads((build_dir / "split_manifest.json").read_text())

    test_event_ids = set(mfst["test_event_ids"])
    exclude_ids = cfg.get("exclude_sensor_ids", [])
    fe_cfg_dict = cfg.get("fe", {})

    # Reconstruct the test DataFrame exactly as during training
    df = df_full.copy()
    if exclude_ids:
        df = df[~df["sensor_id"].isin(exclude_ids)]
    test_df = df[df["event_id"].isin(test_event_ids)].reset_index(drop=True)

    feat_cfg = cfg.get("features", {})
    identifier_cols = feat_cfg.get(
        "identifier_cols", ["event_id", "site_id", "sensor_id"]
    )
    string_cols = feat_cfg.get("string_cols", ["train_type"])
    target_col = feat_cfg.get("target_col", "target_pgv_z_mms")

    X_test, y_test, _ = prepare_features(
        df=test_df,
        target_col=target_col,
        identifier_cols=identifier_cols,
        string_cols=string_cols,
    )

    # Feature engineering — build a minimal namespace object from dict
    class _FECfg:
        pass

    fe_cfg = _FECfg()
    fe_cfg.log_transform_target = fe_cfg_dict.get("log_transform_target", False)
    fe_cfg.add_geometry_features = fe_cfg_dict.get("add_geometry_features", False)

    X_test = engineer_features(X_test, fe_cfg)

    model = xgb.XGBRegressor()
    model.load_model(str(build_dir / "model_final.ubj"))

    y_pred_model = model.predict(X_test)

    if fe_cfg.log_transform_target:
        y_pred = np.expm1(y_pred_model)
    else:
        y_pred = y_pred_model

    return y_test.values, y_pred, test_df["sensor_id"].values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading Parquet files ...")
    df_v1 = pd.read_parquet(PARQUET_V1)
    df_v2 = pd.read_parquet(PARQUET_V2)

    results = []
    for vdir, parquet_path in VERSIONS:
        build_dir = MODELS_ROOT / vdir
        df_full = df_v1 if parquet_path == PARQUET_V1 else df_v2
        print(
            f"Loading {vdir} (parquet {'v1' if parquet_path == PARQUET_V1 else 'v2'}) ..."
        )
        y_true, y_pred, sensor_ids = load_version(build_dir, df_full)
        results.append((y_true, y_pred, sensor_ids))

    # Load MLP predictions from saved predictions.parquet (test split only)
    for vdir in MLP_VERSIONS:
        build_dir = MODELS_ROOT / vdir
        print(f"Loading {vdir} (MLP predictions.parquet) ...")
        pred_df = pd.read_parquet(build_dir / "predictions.parquet")
        test_df = pred_df[pred_df["split"] == "test"].reset_index(drop=True)
        y_true = test_df["target_pgv_z_mms"].values
        y_pred = test_df["y_pred"].values
        sids = test_df["sensor_id"].values
        results.append((y_true, y_pred, sids))

    all_labels = VERSION_LABELS + MLP_LABELS
    all_colors = COLORS + MLP_COLORS
    n_versions = len(all_labels)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        n_versions,
        figsize=(n_versions * 4.5, 9),
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    fig.suptitle(
        "XGBoost & MLP — Held-out Test Set: Predicted vs Actual PGV$_z$\n"
        "(one point per sensor-event pair)",
        fontsize=12,
        y=0.98,
    )

    for col, (label, color, (y_true, y_pred, _)) in enumerate(
        zip(all_labels, all_colors, results)
    ):
        ax_scatter = axes[0, col]
        ax_resid = axes[1, col]

        resid = y_pred - y_true
        vmax = max(y_true.max(), y_pred.max()) * 1.05
        vmax = max(vmax, 5.0)

        # --- Row 0: Predicted vs Actual ---
        ax_scatter.scatter(y_true, y_pred, s=6, alpha=0.35, color=color, linewidths=0)

        # 1:1 line
        ax_scatter.plot([0, vmax], [0, vmax], "k--", lw=1.0, label="1:1")
        # ±2 mm/s band
        ax_scatter.fill_between(
            [0, vmax],
            [-2, vmax - 2],
            [2, vmax + 2],
            alpha=0.08,
            color="grey",
            label="±2 mm/s",
        )

        ax_scatter.set_xlim(0, vmax)
        ax_scatter.set_ylim(0, vmax)
        ax_scatter.set_xlabel("Actual PGV$_z$ (mm/s)", fontsize=8)
        ax_scatter.set_ylabel("Predicted PGV$_z$ (mm/s)", fontsize=8)
        ax_scatter.set_title(label, fontsize=10, fontweight="bold")
        ax_scatter.set_aspect("equal", adjustable="box")
        ax_scatter.tick_params(labelsize=7)
        ax_scatter.text(
            0.04,
            0.96,
            _metrics_str(y_true, y_pred),
            transform=ax_scatter.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )
        if col == 0:
            ax_scatter.legend(fontsize=7, loc="lower right")

        # --- Row 1: Residuals vs Actual ---
        ax_resid.scatter(y_true, resid, s=6, alpha=0.35, color=color, linewidths=0)
        ax_resid.axhline(0, color="black", lw=1.0, ls="--")
        ax_resid.axhline(2, color="grey", lw=0.7, ls=":")
        ax_resid.axhline(-2, color="grey", lw=0.7, ls=":")
        ax_resid.set_xlim(0, vmax)
        ax_resid.set_xlabel("Actual PGV$_z$ (mm/s)", fontsize=8)
        ax_resid.set_ylabel("Residual (pred − actual, mm/s)", fontsize=8)
        ax_resid.set_title(f"Residuals — {label.split(chr(10))[0]}", fontsize=9)
        ax_resid.tick_params(labelsize=7)

        # Annotate median / std of residuals
        ax_resid.text(
            0.04,
            0.96,
            f"median={np.median(resid):+.2f}  std={np.std(resid):.2f} mm/s",
            transform=ax_resid.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # Save to the latest model's build folder (MLP if present, else last XGB)
    latest_build = MLP_VERSIONS[-1] if MLP_VERSIONS else VERSIONS[-1][0]
    latest_dir = MODELS_ROOT / latest_build / "plots"
    latest_dir.mkdir(parents=True, exist_ok=True)
    # Build filename from all version names so it's always unique and traceable
    xgb_tag = "_".join(v[0].split("_")[1] for v in VERSIONS)  # e.g. v001_v002_v003_v004
    mlp_tag = "_".join(v.split("_")[1] for v in MLP_VERSIONS)  # e.g. v001
    out_name = f"test_predictions_comparison_xgb_{xgb_tag}_mlp_{mlp_tag}.png"
    out_path = latest_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
