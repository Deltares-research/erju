"""Train CNN v1 — single-channel FO waveform baseline for PGV_z.

Pipeline
--------
1. Discover the latest waveform_v1 build (waveforms.npy + event_index.parquet)
   and the latest Parquet v2 sensor-level dataset.
2. Build sensor-level samples: each row maps to its event waveform by index and
   carries corrected-distance features + optional train metadata + PGV_z.
3. Event-level splits identical to XGBoost v4:
       test  = 15% events (seed 42)            <- same events as v4
       val   = 15% of remaining events (seed 43)
4. Standardize scalar features on the training fold only.  The waveform is NOT
   normalized (amplitude preserved).
5. Train a 1D CNN encoder + MLP head on log(PGV_z), early stop on val MSE.
6. Evaluate on the held-out test set in BOTH log-space and PGV-space.
7. Save artefacts + diagnostic plots and compare to the v4 benchmark (1.79).

Run:
    python train_cnn_v1.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.cnn.config_cnn_v1 import CONFIG
from src.ml.cnn.cnn_v1_utils import (
    WaveformCNN,
    WaveformDataset,
    compute_metrics,
    predict,
    train_cnn,
)
from src.ml.mlp.mlp_utils import make_event_level_val_split
from src.ml.xgboost.xgb_utils import make_event_level_test_split
from src.utils.geometry_utils import apply_corrected_distances


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _find_latest_waveform_build(cfg) -> Path:
    if cfg.data.waveform_build_dir:
        return Path(cfg.data.waveform_build_dir)
    root = Path(cfg.data.waveform_root)
    builds = sorted(root.glob("holten_waveform_v001_*"), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No holten_waveform_v001_* builds in {root}")
    return builds[-1]


def _find_latest_v2(cfg) -> Path:
    if cfg.data.parquet_v2_path:
        return Path(cfg.data.parquet_v2_path)
    root = Path(cfg.data.parquet_root)
    builds = sorted(root.glob("parquet_v002_*"), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No parquet_v002_* builds in {root}")
    return builds[-1] / "dataset.parquet"


# ---------------------------------------------------------------------------
# Scalar feature engineering
# ---------------------------------------------------------------------------


def _build_scalar_matrix(
    df: pd.DataFrame,
    cfg,
    speed_median: float,
) -> Tuple[np.ndarray, List[str]]:
    """Assemble the raw (un-standardized) scalar feature matrix for a split."""
    fcfg = cfg.features
    r0 = fcfg.r0_m
    r = df[cfg.data.distance_col].to_numpy(dtype=np.float64)
    r = np.clip(r, 1e-3, None)

    cols: List[np.ndarray] = []
    names: List[str] = []

    cols.append(r)
    names.append("r")
    cols.append(np.log(r / r0))
    names.append("log_r_ratio")
    if fcfg.use_inverse_distance:
        cols.append(1.0 / r)
        names.append("inv_r")
        cols.append(1.0 / np.sqrt(r))
        names.append("inv_sqrt_r")

    if fcfg.use_metadata:
        speed_raw = df[fcfg.speed_col].to_numpy(dtype=np.float64)
        if fcfg.add_missing_speed_flag:
            flag = (~np.isfinite(speed_raw)).astype(np.float64)
        speed_filled = np.where(np.isfinite(speed_raw), speed_raw, speed_median)
        cols.append(speed_filled)
        names.append("train_speed_kmh")
        if fcfg.add_missing_speed_flag:
            cols.append(flag)
            names.append("train_speed_kmh_is_missing")
        for c in ("train_type_code", "track_number"):
            if c in df.columns:
                cols.append(df[c].to_numpy(dtype=np.float64))
                names.append(c)

    X = np.column_stack(cols).astype(np.float64)
    return X, names


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_measured_vs_predicted(pgv_true, pgv_pred, metrics, out_path, benchmark):
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = max(min(pgv_true.min(), pgv_pred.min()) * 0.5, 1e-3)
    hi = max(pgv_true.max(), pgv_pred.max()) * 2.0
    ax.scatter(pgv_true, pgv_pred, s=8, alpha=0.3, color="steelblue")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1")
    ax.plot([lo, hi], [lo * 2, hi * 2], "r:", lw=0.8, label="x2 / /2")
    ax.plot([lo, hi], [lo / 2, hi / 2], "r:", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured PGV_z [mm/s]")
    ax.set_ylabel("Predicted PGV_z [mm/s]")
    ax.set_title(
        f"CNN v1 test  RMSE={metrics['rmse_mms']:.3f} mm/s  "
        f"RMSE(log)={metrics['rmse_log']:.3f}  R2(log)={metrics['r2_log']:.3f}\n"
        f"v4 benchmark = {benchmark:.2f} mm/s"
    )
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_residuals_vs(x, resid, xlabel, title, out_path, logx=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, resid, s=8, alpha=0.3, color="darkorange")
    ax.axhline(0, color="k", lw=1.2, ls="--")
    ax.axhline(np.log(2), color="r", lw=0.8, ls=":", label="+/- log(2)")
    ax.axhline(-np.log(2), color="r", lw=0.8, ls=":")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("residual log(pred/true)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = CONFIG
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CNN v1 — single-channel FO waveform baseline")
    print("=" * 70)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load waveform dataset + Parquet v2
    # ------------------------------------------------------------------
    wave_dir = _find_latest_waveform_build(cfg)
    v2_path = _find_latest_v2(cfg)
    print(f"Waveform build : {wave_dir}")
    print(f"Parquet v2     : {v2_path}")

    waveforms = np.load(wave_dir / "waveforms.npy")
    index_df = pd.read_parquet(wave_dir / "event_index.parquet")
    index_ok = index_df[index_df["build_status"] == "ok"].copy()
    index_ok["waveform_row_idx"] = index_ok["waveform_row_idx"].astype(int)
    row_by_event = dict(
        zip(index_ok["event_id"].astype(str), index_ok["waveform_row_idx"])
    )
    # Fixed global amplitude scale (strain -> microstrain). One constant for the
    # whole dataset; preserves relative amplitude (not per-event normalization).
    waveforms = (waveforms.astype(np.float32) * np.float32(cfg.data.waveform_scale))
    print(f"Waveforms      : {waveforms.shape}  ({waveforms.dtype})  "
          f"x{cfg.data.waveform_scale:g} scale")
    print(f"Events with wf : {len(row_by_event):,}")

    # ------------------------------------------------------------------
    # 2. Sensor-level rows
    # ------------------------------------------------------------------
    df = pd.read_parquet(v2_path)
    if cfg.data.exclude_sensor_ids:
        df = df[~df[cfg.data.sensor_col].isin(cfg.data.exclude_sensor_ids)]
    if "effective_distance_to_active_track_m" not in df.columns:
        df = apply_corrected_distances(df)

    df = df.dropna(subset=[cfg.data.distance_col, cfg.data.pgv_col])
    df = df[(df[cfg.data.pgv_col] > 0) & (df[cfg.data.distance_col] > 0)]
    df["event_id"] = df["event_id"].astype(str)
    df = df[df["event_id"].isin(row_by_event)].reset_index(drop=True)
    df["waveform_row_idx"] = df["event_id"].map(row_by_event).astype(int)
    print(f"Sensor rows    : {len(df):,}  |  events: {df['event_id'].nunique():,}")

    # ------------------------------------------------------------------
    # 3. Event-level splits (same as XGBoost v4)
    # ------------------------------------------------------------------
    df_tv, df_test = make_event_level_test_split(
        df, group_col="event_id",
        test_fraction=cfg.split.test_fraction,
        random_seed=cfg.split.test_seed,
    )
    df_train, df_val = make_event_level_val_split(
        df_tv, group_col="event_id",
        val_fraction=cfg.split.val_fraction,
        random_seed=cfg.split.val_seed,
    )
    print(
        f"Split events   : train {df_train['event_id'].nunique():,} | "
        f"val {df_val['event_id'].nunique():,} | test {df_test['event_id'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # 4. Scalar features (impute + standardize on train fold only)
    # ------------------------------------------------------------------
    speed_median = float(df_train[cfg.features.speed_col].median())
    X_train, feat_names = _build_scalar_matrix(df_train, cfg, speed_median)
    X_val, _ = _build_scalar_matrix(df_val, cfg, speed_median)
    X_test, _ = _build_scalar_matrix(df_test, cfg, speed_median)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    print(f"Scalar features ({len(feat_names)}): {feat_names}")
    print(f"  metadata mode : {'B (with metadata)' if cfg.features.use_metadata else 'A (distance only)'}")

    # ------------------------------------------------------------------
    # 5. Targets (log) + datasets
    # ------------------------------------------------------------------
    eps = cfg.train.target_eps

    def _logt(d):
        return np.log(np.clip(d[cfg.data.pgv_col].to_numpy(np.float64), eps, None)).astype(np.float32)

    y_train, y_val, y_test = _logt(df_train), _logt(df_val), _logt(df_test)

    ds_train = WaveformDataset(waveforms, df_train["waveform_row_idx"].to_numpy(), X_train, y_train)
    ds_val = WaveformDataset(waveforms, df_val["waveform_row_idx"].to_numpy(), X_val, y_val)
    ds_test = WaveformDataset(waveforms, df_test["waveform_row_idx"].to_numpy(), X_test, y_test)

    bs = cfg.train.batch_size
    nw = cfg.train.num_workers
    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=nw)

    # ------------------------------------------------------------------
    # 6. Model + build folder
    # ------------------------------------------------------------------
    mc = cfg.model
    model = WaveformCNN(
        n_scalars=len(feat_names),
        conv_channels=mc.conv_channels,
        conv_kernels=mc.conv_kernels,
        conv_strides=mc.conv_strides,
        use_batchnorm=mc.use_batchnorm,
        conv_dropout=mc.conv_dropout,
        head_hidden=mc.head_hidden,
        head_dropout=mc.head_dropout,
        activation=mc.activation,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params   : {n_params:,}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_dir = cfg.output_root_path() / f"{cfg.output.version_name}_{ts}"
    plots_dir = build_dir / cfg.output.plots_subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if cfg.train.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(build_dir / cfg.train.tensorboard_subdir))

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("\nTraining ...")
    best_model_path = build_dir / cfg.output.best_model_filename
    artifacts = train_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        patience=cfg.train.patience,
        best_model_path=best_model_path,
        use_lr_scheduler=cfg.train.lr_scheduler,
        writer=writer,
        verbose=True,
    )
    if writer is not None:
        writer.close()

    # ------------------------------------------------------------------
    # 8. Evaluate (load best)
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    log_pred_test = predict(model, test_loader, device)
    log_pred_test = np.clip(log_pred_test, -30.0, 30.0)  # guard exp() overflow
    pgv_pred_test = np.exp(log_pred_test)
    pgv_true_test = df_test[cfg.data.pgv_col].to_numpy(np.float64)
    metrics_test = compute_metrics(pgv_true_test, pgv_pred_test, eps=eps)

    # Val + train metrics for context
    log_pred_val = predict(model, val_loader, device)
    log_pred_val = np.clip(log_pred_val, -30.0, 30.0)
    metrics_val = compute_metrics(
        df_val[cfg.data.pgv_col].to_numpy(np.float64), np.exp(log_pred_val), eps=eps
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"  RMSE(PGV_z)   = {metrics_test['rmse_mms']:.4f} mm/s   "
          f"(v4 benchmark {cfg.benchmark_v4_rmse:.2f})")
    print(f"  MAE(PGV_z)    = {metrics_test['mae_mms']:.4f} mm/s")
    print(f"  RMSE(log)     = {metrics_test['rmse_log']:.4f}")
    print(f"  MAE(log)      = {metrics_test['mae_log']:.4f}")
    print(f"  R2(log)       = {metrics_test['r2_log']:.4f}")
    print(f"  R2(PGV_z)     = {metrics_test['r2_mms']:.4f}")
    print(f"  delta vs v4   = {metrics_test['rmse_mms'] - cfg.benchmark_v4_rmse:+.4f} mm/s")

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    resid_log = np.log(np.clip(pgv_pred_test, eps, None)) - np.log(
        np.clip(pgv_true_test, eps, None)
    )
    _plot_measured_vs_predicted(
        pgv_true_test, pgv_pred_test, metrics_test,
        plots_dir / "measured_vs_predicted.png", cfg.benchmark_v4_rmse,
    )
    _plot_residuals_vs(
        df_test[cfg.data.distance_col].to_numpy(np.float64), resid_log,
        "distance to track [m]", "CNN v1 — residuals vs distance",
        plots_dir / "residuals_vs_distance.png",
    )
    _plot_residuals_vs(
        pgv_true_test, resid_log,
        "measured PGV_z [mm/s]", "CNN v1 — residuals vs PGV",
        plots_dir / "residuals_vs_pgv.png", logx=True,
    )
    # Learning curve
    hist_df = pd.DataFrame(artifacts.history)
    if not hist_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist_df["epoch"], hist_df["train_loss"], label="train")
        ax.plot(hist_df["epoch"], hist_df["val_loss"], label="val")
        ax.axvline(artifacts.best_epoch, color="k", ls="--", lw=0.8,
                   label=f"best ep {artifacts.best_epoch}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (log space)")
        ax.set_title("CNN v1 learning curve")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(plots_dir / "learning_curve.png", dpi=140)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 10. Save artefacts
    # ------------------------------------------------------------------
    hist_df.to_csv(build_dir / cfg.output.history_filename, index=False)

    pred_df = df_test[["event_id", cfg.data.sensor_col, cfg.data.distance_col,
                       cfg.data.pgv_col]].copy()
    pred_df["pgv_pred_mms"] = pgv_pred_test
    pred_df["log_pgv_true"] = np.log(np.clip(pgv_true_test, eps, None))
    pred_df["log_pgv_pred"] = log_pred_test
    pred_df.to_parquet(build_dir / cfg.output.predictions_filename, index=False)

    split_manifest = {
        "train_event_ids": sorted(df_train["event_id"].unique().tolist()),
        "val_event_ids": sorted(df_val["event_id"].unique().tolist()),
        "test_event_ids": sorted(df_test["event_id"].unique().tolist()),
        "n_train_rows": int(len(df_train)),
        "n_val_rows": int(len(df_val)),
        "n_test_rows": int(len(df_test)),
    }
    with open(build_dir / cfg.output.split_manifest_filename, "w") as fh:
        json.dump(split_manifest, fh, indent=2)

    summary = {
        "version": cfg.output.version_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "waveform_build": str(wave_dir),
        "parquet_v2": str(v2_path),
        "n_params": int(n_params),
        "feature_names": feat_names,
        "metadata_mode": "B" if cfg.features.use_metadata else "A",
        "best_epoch": artifacts.best_epoch,
        "best_val_loss": artifacts.best_val_loss,
        "metrics_test": metrics_test,
        "metrics_val": metrics_val,
        "benchmark_v4_rmse": cfg.benchmark_v4_rmse,
        "delta_vs_v4_mms": metrics_test["rmse_mms"] - cfg.benchmark_v4_rmse,
    }
    with open(build_dir / cfg.output.summary_filename, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    with open(build_dir / cfg.output.config_snapshot_filename, "w") as fh:
        json.dump(cfg.as_dict(), fh, indent=2, default=str)

    print(f"\nBuild folder: {build_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
