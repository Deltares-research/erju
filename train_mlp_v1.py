"""train_mlp_v1.py — Entry point for MLP v1 training.

Run from project root:
    python train_mlp_v1.py

Monitor training live:
    tensorboard --logdir <build_folder>/runs
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.ml.mlp.config_mlp_v1 import CONFIG
from src.ml.mlp.mlp_utils import (
    MLP,
    fit_imputer,
    fit_scaler,
    make_event_level_val_split,
    make_tensor_dataset,
    predict_numpy,
    rmse_original_units,
    save_checkpoint,
    train_mlp,
)
from src.ml.xgboost.xgb_utils import (
    engineer_features,
    make_event_level_test_split,
    prepare_features,
)


# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(CONFIG.data.random_seed)
np.random.seed(CONFIG.data.random_seed)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Build folder ──────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
build_dir = CONFIG.models_root / f"{CONFIG.version_name}_{ts}"
build_dir.mkdir(parents=True, exist_ok=True)
print(f"Build folder: {build_dir}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading parquet ...")
df = pd.read_parquet(CONFIG.parquet_path)
print(f"  {df.shape[0]:,} rows  x  {df.shape[1]} columns")

# ── Event-level test split (identical strategy to XGBoost) ───────────────────
train_val_df, test_df = make_event_level_test_split(
    df,
    group_col=CONFIG.data.group_col,
    test_fraction=CONFIG.data.test_size,
    random_seed=CONFIG.data.random_seed,
)

# ── Event-level validation split (from remaining train_val) ──────────────────
train_df, val_df = make_event_level_val_split(
    train_val_df,
    group_col=CONFIG.data.group_col,
    val_fraction=CONFIG.data.val_size,
    random_seed=CONFIG.data.random_seed,
)

print(f"\nSplit summary (by event):")
print(
    f"  Train : {train_df[CONFIG.data.group_col].nunique():>4} events | {len(train_df):>6,} rows"
)
print(
    f"  Val   : {val_df[CONFIG.data.group_col].nunique():>4} events | {len(val_df):>6,} rows"
)
print(
    f"  Test  : {test_df[CONFIG.data.group_col].nunique():>4} events | {len(test_df):>6,} rows"
)


# ── Prepare features (same pipeline as XGBoost) ──────────────────────────────
def _prep(frame: pd.DataFrame):
    X, y, _ = prepare_features(
        frame,
        target_col=CONFIG.data.target_col,
        identifier_cols=CONFIG.data.identifier_cols,
        string_cols=CONFIG.data.string_cols,
    )
    X = engineer_features(X, CONFIG.fe)
    return X, y


X_train, y_train = _prep(train_df)
X_val, y_val = _prep(val_df)
X_test, y_test = _prep(test_df)

print(f"\nFeature matrix shape: {X_train.shape[1]} features")

# ── Log1p-transform target ────────────────────────────────────────────────────
if CONFIG.fe.log_transform_target:
    y_train_t = np.log1p(y_train.values)
    y_val_t = np.log1p(y_val.values)
    y_test_t = np.log1p(y_test.values)
else:
    y_train_t = y_train.values
    y_val_t = y_val.values
    y_test_t = y_test.values

# ── Feature scaling (fit on train only — no leakage) ─────────────────────────
imputer = fit_imputer(X_train.values)
X_train_i = imputer.transform(X_train.values)
X_val_i = imputer.transform(X_val.values)
X_test_i = imputer.transform(X_test.values)

scaler = fit_scaler(X_train_i)
X_train_s = scaler.transform(X_train_i)
X_val_s = scaler.transform(X_val_i)
X_test_s = scaler.transform(X_test_i)

print(f"Scaler fitted on training data only (mean/std of {X_train.shape[1]} features).")

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_ds = make_tensor_dataset(X_train_s, y_train_t, device)
val_ds = make_tensor_dataset(X_val_s, y_val_t, device)
test_ds = make_tensor_dataset(X_test_s, y_test_t, device)

train_loader = DataLoader(train_ds, batch_size=CONFIG.train.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG.train.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CONFIG.train.batch_size, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model = MLP(
    input_size=X_train.shape[1],
    hidden_sizes=CONFIG.model.hidden_sizes,
    activation=CONFIG.model.activation,
    dropout=CONFIG.model.dropout,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {model}")
print(f"  Trainable parameters: {n_params:,}")

# ── Train ─────────────────────────────────────────────────────────────────────
history = train_mlp(model, train_loader, val_loader, CONFIG.train, build_dir, device)

# ── Load best model and evaluate on test set ──────────────────────────────────
print("\nLoading best model checkpoint for test evaluation ...")
best_ckpt = torch.load(build_dir / "best_model.pt", map_location=device)
model.load_state_dict(best_ckpt["model_state_dict"])

y_pred_test_log = predict_numpy(model, test_loader)
y_pred_val_log = predict_numpy(model, val_loader)

test_rmse = rmse_original_units(
    y_test_t, y_pred_test_log, CONFIG.fe.log_transform_target
)
val_rmse = rmse_original_units(y_val_t, y_pred_val_log, CONFIG.fe.log_transform_target)

print(f"\n  Val  RMSE: {val_rmse:.4f} mm/s  (best epoch {history['best_epoch']})")
print(f"  Test RMSE: {test_rmse:.4f} mm/s")

# ── Save artefacts ────────────────────────────────────────────────────────────
# Config snapshot
cfg_dict = dataclasses.asdict(CONFIG)
cfg_dict["parquet_path"] = str(CONFIG.parquet_path)
cfg_dict["models_root"] = str(CONFIG.models_root)
with open(build_dir / "config_snapshot.json", "w") as f:
    json.dump(cfg_dict, f, indent=2, default=str)

# Summary
summary = {
    "version_name": CONFIG.version_name,
    "build_dir": str(build_dir),
    "n_features": int(X_train.shape[1]),
    "feature_names": list(X_train.columns),
    "n_params": n_params,
    "best_epoch": history["best_epoch"],
    "best_val_loss_log": history["best_val_loss"],
    "val_rmse_mms": val_rmse,
    "test_rmse_mms": test_rmse,
    "train_events": int(train_df[CONFIG.data.group_col].nunique()),
    "val_events": int(val_df[CONFIG.data.group_col].nunique()),
    "test_events": int(test_df[CONFIG.data.group_col].nunique()),
}
with open(build_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# OOF-style predictions on val+test for later comparison
pred_df = pd.concat(
    [
        val_df[[CONFIG.data.group_col, "sensor_id", CONFIG.data.target_col]].assign(
            y_pred_log=y_pred_val_log,
            y_pred=(
                np.expm1(y_pred_val_log)
                if CONFIG.fe.log_transform_target
                else y_pred_val_log
            ),
            split="val",
        ),
        test_df[[CONFIG.data.group_col, "sensor_id", CONFIG.data.target_col]].assign(
            y_pred_log=y_pred_test_log,
            y_pred=(
                np.expm1(y_pred_test_log)
                if CONFIG.fe.log_transform_target
                else y_pred_test_log
            ),
            split="test",
        ),
    ],
    ignore_index=True,
)
pred_df.to_parquet(build_dir / "predictions.parquet", index=False)

# Training history CSV
hist_df = pd.DataFrame(
    {
        "epoch": range(1, len(history["train_losses"]) + 1),
        "train_loss": history["train_losses"],
        "val_loss": history["val_losses"],
    }
)
hist_df.to_csv(build_dir / "training_history.csv", index=False)

print(f"\nArtefacts saved to: {build_dir}")
print(f"  config_snapshot.json")
print(f"  summary.json")
print(f"  predictions.parquet")
print(f"  training_history.csv")
print(f"  best_model.pt")
print(f"  checkpoints/  (every {CONFIG.train.checkpoint_every_n_epochs} epochs)")
print(f"  runs/         (TensorBoard logs)")
print(f"\nDone.")
