"""report_event_grid.py — 3×2 PGV prediction grid for the report.

Layout: 3 rows (random events) × 2 columns
  Left  column: XGBoost v4
  Right column: MLP v2

Linear scale only.  Figure height ≤ 5 inches.

Output: report_event_grid.png  (in the repo root)
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
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ml.mlp.mlp_utils import (
    MLP,
    fit_imputer,
    fit_scaler,
    make_event_level_val_split,
)
from src.ml.xgboost.xgb_utils import (
    engineer_features,
    make_event_level_test_split,
    prepare_features,
)
from src.utils.geometry_utils import apply_corrected_distances

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODELS_ROOT = Path(r"P:\11210978-erju-ai\holten_models")
PARQUET_ROOT = Path(r"P:\11210978-erju-ai\holten_parquet")

RANDOM_SEED = 7  # change to get different events
N_EVENTS = 3
SENSOR_LINES = ["C"]  # same filter used in training / predict_pgv.py
SENSOR_SIDE = [-1]


# ---------------------------------------------------------------------------
# Helpers — auto-discover latest build
# ---------------------------------------------------------------------------
def _latest(root: Path, pattern: str) -> Path:
    builds = sorted(root.glob(pattern), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No {pattern} found under {root}")
    return builds[-1]


# ---------------------------------------------------------------------------
# Load parquet v2
# ---------------------------------------------------------------------------
v2_dir = _latest(PARQUET_ROOT, "parquet_v002_*")
df_full = pd.read_parquet(v2_dir / "dataset.parquet")
if "effective_distance_to_active_track_m" not in df_full.columns:
    df_full = apply_corrected_distances(df_full)

# ---------------------------------------------------------------------------
# Load holten.json sensor→line map
# ---------------------------------------------------------------------------
with open("sites/holten.json") as f:
    site = json.load(f)
sensor_line_map: dict[str, str] = {}
for mp, line in site["accelerometer"]["line_geometry"]["sensor_line_map"].items():
    sensor_line_map[mp] = line[-1].upper()  # "line_C" → "C"

# ---------------------------------------------------------------------------
# Filter to line C, side -1 (same as training)
# ---------------------------------------------------------------------------
df_full["_line"] = df_full["sensor_id"].map(sensor_line_map)
df_use = (
    df_full[
        df_full["_line"].isin(SENSOR_LINES)
        & df_full["acc_side_of_track"].isin(SENSOR_SIDE)
    ]
    .drop(columns=["_line"])
    .copy()
)

# ---------------------------------------------------------------------------
# Pick N_EVENTS from the held-out test split
# ---------------------------------------------------------------------------
test_df, _ = make_event_level_test_split(
    df_use,
    group_col="event_id",
    test_fraction=0.20,
    random_seed=42,
)
rng = np.random.default_rng(RANDOM_SEED)
chosen_events = rng.choice(test_df["event_id"].unique(), size=N_EVENTS, replace=False)
print(f"Selected events: {list(chosen_events)}")

# ---------------------------------------------------------------------------
# Load XGBoost v4
# ---------------------------------------------------------------------------
xgb_dir = _latest(MODELS_ROOT, "xgb_v004_*")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(str(xgb_dir / "model_final.ubj"))
print(f"XGBoost build : {xgb_dir.name}")


def predict_xgb(df_sensor: pd.DataFrame) -> np.ndarray:
    X = df_sensor.copy()
    dist_col = (
        "effective_distance_to_active_track_m"
        if "effective_distance_to_active_track_m" in X.columns
        else "acc_distance_to_track_m"
    )
    d = X[dist_col].clip(lower=0.0)
    X["feat_log1p_distance"] = np.log1p(d)
    X["feat_inv_distance_sq"] = 1.0 / (d**2 + 1.0)
    X = X[list(xgb_model.feature_names_in_)]
    return np.expm1(xgb_model.predict(X))


# ---------------------------------------------------------------------------
# Load MLP v2 — rebuild imputer/scaler from training data, then load weights
# ---------------------------------------------------------------------------
mlp_dir = _latest(MODELS_ROOT, "mlp_v002_*")
print(f"MLP build     : {mlp_dir.name}")

with open(mlp_dir / "summary.json") as f:
    summary = json.load(f)
with open(mlp_dir / "config_snapshot.json") as f:
    cfg = json.load(f)
data_cfg = cfg["data"]
model_cfg = cfg["model"]
fe_cfg = cfg.get("fe", {})

# Rebuild train split to re-fit imputer + scaler
_, train_val_df = make_event_level_test_split(
    df_use,
    group_col=data_cfg["group_col"],
    test_fraction=data_cfg["test_size"],
    random_seed=data_cfg["random_seed"],
)
train_df, _ = make_event_level_val_split(
    train_val_df,
    group_col=data_cfg["group_col"],
    val_fraction=data_cfg["val_size"],
    random_seed=data_cfg["random_seed"],
)


class _FE:
    add_geometry_features = fe_cfg.get("add_geometry_features", True)
    log_transform_target = fe_cfg.get("log_transform_target", True)


fe_obj = _FE()

X_train, _, _ = prepare_features(
    train_df,
    target_col=data_cfg["target_col"],
    identifier_cols=data_cfg["identifier_cols"],
    string_cols=data_cfg["string_cols"],
)
X_train = engineer_features(X_train, fe_obj)
mlp_feature_cols = list(X_train.columns)
imputer = fit_imputer(X_train.values)
X_ti = imputer.transform(X_train.values)
scaler = fit_scaler(X_ti)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_net = MLP(
    input_size=summary["n_features"],
    hidden_sizes=model_cfg["hidden_sizes"],
    activation=model_cfg["activation"],
    dropout=model_cfg.get("dropout", 0.0),
    batch_norm=model_cfg.get("batch_norm", False),
).to(device)
ckpt = torch.load(mlp_dir / "best_model.pt", map_location=device, weights_only=True)
mlp_net.load_state_dict(ckpt["model_state_dict"])
mlp_net.eval()


def predict_mlp(df_sensor: pd.DataFrame) -> np.ndarray:
    X, _, _ = prepare_features(
        df_sensor,
        target_col=data_cfg["target_col"],
        identifier_cols=data_cfg["identifier_cols"],
        string_cols=data_cfg["string_cols"],
    )
    X = engineer_features(X, fe_obj)
    for c in mlp_feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[mlp_feature_cols]
    Xi = imputer.transform(X.values)
    Xs = scaler.transform(Xi)
    with torch.no_grad():
        t = torch.tensor(Xs, dtype=torch.float32).to(device)
        yp = mlp_net(t).cpu().numpy().ravel()
    return np.expm1(yp) if fe_obj.log_transform_target else yp


# ---------------------------------------------------------------------------
# Plot — 3 rows × 2 cols, height=5
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    N_EVENTS,
    2,
    figsize=(7, 5),
    sharey=False,
)
fig.subplots_adjust(hspace=0.55, wspace=0.35)

for row, ev in enumerate(chosen_events):
    df_ev = test_df[test_df["event_id"] == ev].copy()
    df_ev = df_ev.dropna(
        subset=["effective_distance_to_active_track_m", "target_pgv_z_mms"]
    )
    df_ev = df_ev.sort_values("effective_distance_to_active_track_m").reset_index(
        drop=True
    )

    r_real = df_ev["effective_distance_to_active_track_m"].values.astype(float)
    pgv_real = df_ev["target_pgv_z_mms"].values.astype(float)

    pred_xgb = predict_xgb(df_ev)
    pred_mlp = predict_mlp(df_ev)

    short_id = ev[:15] + "…" if len(ev) > 15 else ev

    for col, (preds, model_label, color) in enumerate(
        [
            (pred_xgb, "XGBoost v4", "steelblue"),
            (pred_mlp, "MLP v2", "darkorange"),
        ]
    ):
        ax = axes[row][col]

        ax.scatter(
            r_real,
            pgv_real,
            color="black",
            s=30,
            zorder=5,
            marker="x",
            linewidths=1.5,
            label="Measured",
        )
        ax.plot(
            r_real,
            preds,
            color=color,
            lw=1.5,
            marker="o",
            markersize=4,
            label=model_label,
            alpha=0.85,
        )

        ax.set_xlabel("Distance (m)", fontsize=7)
        ax.set_ylabel("PGV (mm/s)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        title_prefix = f"Event {row+1}"
        ax.set_title(f"{title_prefix} — {model_label}", fontsize=7, pad=3)

        if row == 0:
            ax.legend(fontsize=6, loc="upper right")

out = Path("report_event_grid.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
