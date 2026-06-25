"""report_attenuation_curve.py — Mean attenuation curve: real vs predicted.

Runs XGBoost v4 and MLP v2 on all test-set events, groups by sensor distance,
and plots mean ± 1 std for real measurements, XGBoost, and MLP on one axes.

Output: report_attenuation_curve.png
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
# CONFIG  (identical to report_event_grid.py)
# ---------------------------------------------------------------------------
MODELS_ROOT = Path(r"P:\11210978-erju-ai\holten_models")
PARQUET_ROOT = Path(r"P:\11210978-erju-ai\holten_parquet")
SENSOR_LINES = ["C"]
SENSOR_SIDE = [-1]

# Set to a train_type string to restrict the curve to one train type,
# or None to use all trains.
# Use exact match; run with None first to see available values.
TRAIN_TYPE_FILTER = "SPR(A)"  # e.g. "SPR(A)" | "ICM" | "SNG" | None


def _latest(root: Path, pattern: str) -> Path:
    builds = sorted(root.glob(pattern), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No {pattern} found under {root}")
    return builds[-1]


# ---------------------------------------------------------------------------
# Load & filter data
# ---------------------------------------------------------------------------
v2_dir = _latest(PARQUET_ROOT, "parquet_v002_*")
df_full = pd.read_parquet(v2_dir / "dataset.parquet")
if "effective_distance_to_active_track_m" not in df_full.columns:
    df_full = apply_corrected_distances(df_full)

with open("sites/holten.json") as f:
    site = json.load(f)
sensor_line_map = {
    mp: line[-1].upper()
    for mp, line in site["accelerometer"]["line_geometry"]["sensor_line_map"].items()
}
df_full["_line"] = df_full["sensor_id"].map(sensor_line_map)
df_use = (
    df_full[
        df_full["_line"].isin(SENSOR_LINES)
        & df_full["acc_side_of_track"].isin(SENSOR_SIDE)
    ]
    .drop(columns=["_line"])
    .copy()
)

test_df, _ = make_event_level_test_split(
    df_use, group_col="event_id", test_fraction=0.20, random_seed=42
)
test_df = test_df.dropna(
    subset=["effective_distance_to_active_track_m", "target_pgv_z_mms"]
).reset_index(drop=True)

# Optional train-type filter
if TRAIN_TYPE_FILTER is not None:
    test_df = test_df[test_df["train_type"] == TRAIN_TYPE_FILTER].reset_index(drop=True)
    print(f"Train type filter: '{TRAIN_TYPE_FILTER}'")

print(f"Test rows: {len(test_df)}  |  events: {test_df['event_id'].nunique()}")

# ---------------------------------------------------------------------------
# Load XGBoost v4
# ---------------------------------------------------------------------------
xgb_dir = _latest(MODELS_ROOT, "xgb_v004_*")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(str(xgb_dir / "model_final.ubj"))
print(f"XGBoost : {xgb_dir.name}")


def predict_xgb(df: pd.DataFrame) -> np.ndarray:
    X = df.copy()
    d = X["effective_distance_to_active_track_m"].clip(lower=0.0)
    X["feat_log1p_distance"] = np.log1p(d)
    X["feat_inv_distance_sq"] = 1.0 / (d**2 + 1.0)
    return np.expm1(xgb_model.predict(X[list(xgb_model.feature_names_in_)]))


# ---------------------------------------------------------------------------
# Load MLP v2
# ---------------------------------------------------------------------------
mlp_dir = _latest(MODELS_ROOT, "mlp_v002_*")
print(f"MLP     : {mlp_dir.name}")

with open(mlp_dir / "summary.json") as f:
    summary = json.load(f)
with open(mlp_dir / "config_snapshot.json") as f:
    cfg = json.load(f)
data_cfg = cfg["data"]
model_cfg = cfg["model"]
fe_cfg = cfg.get("fe", {})

_, train_val = make_event_level_test_split(
    df_use,
    group_col=data_cfg["group_col"],
    test_fraction=data_cfg["test_size"],
    random_seed=data_cfg["random_seed"],
)
train_df, _ = make_event_level_val_split(
    train_val,
    group_col=data_cfg["group_col"],
    val_fraction=data_cfg["val_size"],
    random_seed=data_cfg["random_seed"],
)


class _FE:
    add_geometry_features = fe_cfg.get("add_geometry_features", True)
    log_transform_target = fe_cfg.get("log_transform_target", True)


fe_obj = _FE()
X_tr, _, _ = prepare_features(
    train_df,
    target_col=data_cfg["target_col"],
    identifier_cols=data_cfg["identifier_cols"],
    string_cols=data_cfg["string_cols"],
)
X_tr = engineer_features(X_tr, fe_obj)
mlp_feature_cols = list(X_tr.columns)  # store exact column order
imputer = fit_imputer(X_tr.values)
X_tri = imputer.transform(X_tr.values)
scaler = fit_scaler(X_tri)

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


def predict_mlp(df: pd.DataFrame) -> np.ndarray:
    X, _, _ = prepare_features(
        df,
        target_col=data_cfg["target_col"],
        identifier_cols=data_cfg["identifier_cols"],
        string_cols=data_cfg["string_cols"],
    )
    X = engineer_features(X, fe_obj)
    # Align to training columns (handles missing dummies, drops extras)
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
# Run predictions on the full test set
# ---------------------------------------------------------------------------
print("Running predictions on full test set ...")
test_df = test_df.copy()
test_df["pred_xgb"] = predict_xgb(test_df)
test_df["pred_mlp"] = predict_mlp(test_df)

# ---------------------------------------------------------------------------
# Aggregate by distance: mean ± 1 std
# ---------------------------------------------------------------------------
dist_col = "effective_distance_to_active_track_m"
grp = test_df.groupby(dist_col)

stats = pd.DataFrame(
    {
        "mean_real": grp["target_pgv_z_mms"].mean(),
        "std_real": grp["target_pgv_z_mms"].std(),
        "mean_xgb": grp["pred_xgb"].mean(),
        "std_xgb": grp["pred_xgb"].std(),
        "mean_mlp": grp["pred_mlp"].mean(),
        "std_mlp": grp["pred_mlp"].std(),
        "n": grp["target_pgv_z_mms"].count(),
    }
).reset_index()

print("\nAttenuation statistics per distance:")
print(
    stats[[dist_col, "n", "mean_real", "mean_xgb", "mean_mlp"]].to_string(index=False)
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 2.5))

r = stats[dist_col].values

# Slight horizontal jitter so error bars from the three series don't overlap
JITTER = 0.25  # metres — invisible at the scale of the axis

SERIES = [
    ("mean_real", "std_real", "Measured", "#222222", "x", 0.0, 2.0),
    ("mean_xgb", "std_xgb", "XGBoost v4", "#1565C0", "o", -JITTER, 1.6),
    ("mean_mlp", "std_mlp", "MLP v2", "#C62828", "s", +JITTER, 1.6),
]

for mean_col, std_col, label, color, marker, dx, lw in SERIES:
    mean = stats[mean_col].values
    std = stats[std_col].values
    # Error bars (semi-transparent)
    eb = ax.errorbar(
        r + dx,
        mean,
        yerr=std,
        fmt=marker,
        color=color,
        ecolor=color,
        lw=lw,
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        markersize=6,
        alpha=0.55,
        label=f"{label} (mean ± 1σ)",
        zorder=3,
    )
    # Connecting line (more transparent)
    ax.plot(r + dx, mean, color=color, lw=lw, alpha=0.35, zorder=2)


ax.set_xlabel("Distance to track (m)", fontsize=7)
ax.set_ylabel("PGV$_z$ (mm/s)", fontsize=7)
type_label = f" — {TRAIN_TYPE_FILTER}" if TRAIN_TYPE_FILTER else ""
ax.set_title(f"Attenuation curve — test set average{type_label}", fontsize=9)
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=7)

plt.tight_layout()
out = Path("report_attenuation_curve.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved → {out}")
