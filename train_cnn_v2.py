"""Train CNN v2 — multi-channel FO waveform 2D CNN for PGV_z.

Same pipeline as cnn_v1 but the input is a (21, T) channel x time block and the
encoder is a compact 2D CNN.  Event-level splits, preprocessing, metrics and
diagnostics are identical to cnn_v1 so results are directly comparable.

Ablations (CLI):
    python train_cnn_v2.py                  # waveform + distance + metadata
    python train_cnn_v2.py --no-metadata    # waveform + distance only
    python train_cnn_v2.py --no-waveform    # scalar-only (distance + metadata)
    python train_cnn_v2.py --no-batchnorm
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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

from src.ml.cnn.config_cnn_v2 import CONFIG
from src.ml.cnn.cnn_v1_utils import compute_metrics, predict, train_cnn
from src.ml.cnn.cnn_v2_utils import WaveformCNN2D, WaveformDataset2D
from src.ml.mlp.mlp_utils import make_event_level_val_split
from src.ml.xgboost.xgb_utils import make_event_level_test_split
from src.utils.geometry_utils import apply_corrected_distances

# Reuse the scalar-feature builder and plot helpers from cnn_v1's train script.
from train_cnn_v1 import (
    _build_scalar_matrix,
    _plot_measured_vs_predicted,
    _plot_residuals_by_cropped,
    _plot_residuals_vs,
    _plot_residuals_vs_duration,
)


def _find_latest_waveform_build(cfg) -> Path:
    if cfg.data.waveform_build_dir:
        return Path(cfg.data.waveform_build_dir)
    root = Path(cfg.data.waveform_root)
    builds = sorted(root.glob(cfg.data.waveform_glob), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No {cfg.data.waveform_glob} builds in {root}")
    return builds[-1]


def _find_latest_v2(cfg) -> Path:
    if cfg.data.parquet_v2_path:
        return Path(cfg.data.parquet_v2_path)
    root = Path(cfg.data.parquet_root)
    builds = sorted(root.glob("parquet_v002_*"), key=lambda p: p.name)
    if not builds:
        raise FileNotFoundError(f"No parquet_v002_* builds in {root}")
    return builds[-1] / "dataset.parquet"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN v2 multi-channel FO CNN.")
    p.add_argument("--no-metadata", action="store_true")
    p.add_argument("--no-waveform", action="store_true")
    p.add_argument("--no-batchnorm", action="store_true")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--waveform-build", type=str, default="",
                   help="Path to a specific waveform build dir (overrides discovery).")
    p.add_argument("--waveform-glob", type=str, default="",
                   help="Glob for the waveform build to auto-discover (e.g. holten_waveform_v003_ch51_*).")
    return p.parse_args()


def _apply_overrides(cfg, args) -> str:
    parts: List[str] = []
    if args.waveform_build:
        cfg.data.waveform_build_dir = args.waveform_build
    if args.waveform_glob:
        cfg.data.waveform_glob = args.waveform_glob
    if args.no_metadata:
        cfg.features.use_metadata = False
        parts.append("modeA")
    if args.no_waveform:
        cfg.model.use_waveform = False
        parts.append("scalaronly")
    if args.no_batchnorm:
        cfg.model.use_batchnorm = False
        parts.append("nobn")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def main() -> None:
    cfg = CONFIG
    args = _parse_args()
    run_tag = _apply_overrides(cfg, args)

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CNN v2 — multi-channel FO waveform 2D CNN")
    print("=" * 70)
    print(f"Device: {device}")

    # 1. Load data
    wave_dir = _find_latest_waveform_build(cfg)
    v2_path = _find_latest_v2(cfg)
    print(f"Waveform build : {wave_dir}")
    print(f"Parquet v2     : {v2_path}")

    waveforms = np.load(wave_dir / "waveforms.npy")
    if waveforms.dtype != np.float32:
        waveforms = waveforms.astype(np.float32)
    waveforms *= np.float32(cfg.data.waveform_scale)        # in-place global scale (no copy)
    index_df = pd.read_parquet(wave_dir / "event_index.parquet")
    index_ok = index_df[index_df["build_status"] == "ok"].copy()
    index_ok["waveform_row_idx"] = index_ok["waveform_row_idx"].astype(int)
    index_ok["event_id"] = index_ok["event_id"].astype(str)
    row_by_event = dict(zip(index_ok["event_id"], index_ok["waveform_row_idx"]))
    cropped_by_event = dict(zip(index_ok["event_id"], index_ok["was_cropped"]))
    duration_by_event = dict(zip(index_ok["event_id"], index_ok["duration_original_s"]))
    print(f"Waveforms      : {waveforms.shape}  ({waveforms.dtype})  "
          f"x{cfg.data.waveform_scale:g} scale")
    print(f"Events with wf : {len(row_by_event):,}")

    # 2. Sensor rows
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
    df["was_cropped"] = df["event_id"].map(cropped_by_event).astype(bool)
    df["duration_original_s"] = df["event_id"].map(duration_by_event).astype(float)
    print(f"Sensor rows    : {len(df):,}  |  events: {df['event_id'].nunique():,}")

    # 3. Splits (identical to v4 / cnn_v1)
    df_tv, df_test = make_event_level_test_split(
        df, group_col="event_id",
        test_fraction=cfg.split.test_fraction, random_seed=cfg.split.test_seed,
    )
    df_train, df_val = make_event_level_val_split(
        df_tv, group_col="event_id",
        val_fraction=cfg.split.val_fraction, random_seed=cfg.split.val_seed,
    )
    print(f"Split events   : train {df_train['event_id'].nunique():,} | "
          f"val {df_val['event_id'].nunique():,} | test {df_test['event_id'].nunique():,}")

    # 4. Scalar features
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

    # 5. Targets + datasets
    eps = cfg.train.target_eps

    def _logt(d):
        return np.log(np.clip(d[cfg.data.pgv_col].to_numpy(np.float64), eps, None)).astype(np.float32)

    y_train, y_val, y_test = _logt(df_train), _logt(df_val), _logt(df_test)
    ds_train = WaveformDataset2D(waveforms, df_train["waveform_row_idx"].to_numpy(), X_train, y_train)
    ds_val = WaveformDataset2D(waveforms, df_val["waveform_row_idx"].to_numpy(), X_val, y_val)
    ds_test = WaveformDataset2D(waveforms, df_test["waveform_row_idx"].to_numpy(), X_test, y_test)

    bs, nw = cfg.train.batch_size, cfg.train.num_workers
    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=nw)

    # 6. Model + build folder
    mc = cfg.model
    model = WaveformCNN2D(
        n_scalars=len(feat_names),
        conv_channels=mc.conv_channels, kernel_ch=mc.kernel_ch,
        kernel_time=mc.kernel_time, stride_ch=mc.stride_ch, stride_time=mc.stride_time,
        use_batchnorm=mc.use_batchnorm, conv_dropout=mc.conv_dropout,
        head_hidden=mc.head_hidden, head_dropout=mc.head_dropout,
        activation=mc.activation, use_waveform=mc.use_waveform,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params   : {n_params:,}")
    print(f"  waveform      : {mc.use_waveform}  |  batchnorm: {mc.use_batchnorm}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = cfg.output.version_name + (f"_{run_tag}" if run_tag else "") + f"_{ts}"
    build_dir = cfg.output_root_path() / folder_name
    plots_dir = build_dir / cfg.output.plots_subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if cfg.train.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(build_dir / cfg.train.tensorboard_subdir))

    # 7. Train
    print("\nTraining ...")
    best_model_path = build_dir / cfg.output.best_model_filename
    artifacts = train_cnn(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
        epochs=cfg.train.epochs, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        patience=cfg.train.patience, best_model_path=best_model_path,
        use_lr_scheduler=cfg.train.lr_scheduler, writer=writer, verbose=True,
    )
    if writer is not None:
        writer.close()

    # 8. Evaluate
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    log_pred_test = np.clip(predict(model, test_loader, device), -30.0, 30.0)
    pgv_pred_test = np.exp(log_pred_test)
    pgv_true_test = df_test[cfg.data.pgv_col].to_numpy(np.float64)
    metrics_test = compute_metrics(pgv_true_test, pgv_pred_test, eps=eps)
    log_pred_val = np.clip(predict(model, val_loader, device), -30.0, 30.0)
    metrics_val = compute_metrics(df_val[cfg.data.pgv_col].to_numpy(np.float64),
                                  np.exp(log_pred_val), eps=eps)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"  RMSE(PGV_z)   = {metrics_test['rmse_mms']:.4f} mm/s")
    print(f"  MAE(PGV_z)    = {metrics_test['mae_mms']:.4f} mm/s")
    print(f"  RMSE(log)     = {metrics_test['rmse_log']:.4f}")
    print(f"  MAE(log)      = {metrics_test['mae_log']:.4f}")
    print(f"  R2(log)       = {metrics_test['r2_log']:.4f}")
    print(f"  R2(PGV_z)     = {metrics_test['r2_mms']:.4f}")
    print(f"  vs v4 (1.79)        : {metrics_test['rmse_mms'] - cfg.benchmark_v4_rmse:+.4f} mm/s")
    print(f"  vs cnn_v1 (1.876)   : {metrics_test['rmse_mms'] - cfg.benchmark_cnn_v1_rmse:+.4f} mm/s")
    print(f"  vs scalar (1.879)   : {metrics_test['rmse_mms'] - cfg.benchmark_scalar_only_rmse:+.4f} mm/s")

    was_cropped_test = df_test["was_cropped"].to_numpy(bool)
    duration_test = df_test["duration_original_s"].to_numpy(float)
    crop_breakdown: Dict[str, Dict[str, float]] = {}
    for label, mask in [("not_cropped", ~was_cropped_test), ("cropped", was_cropped_test)]:
        if mask.sum() > 1:
            m = compute_metrics(pgv_true_test[mask], pgv_pred_test[mask], eps=eps)
            crop_breakdown[label] = {"n": int(mask.sum()), **m}
    print("\n  Residual breakdown by crop status:")
    for label, m in crop_breakdown.items():
        print(f"    {label:12s} n={m['n']:4d}  RMSE={m['rmse_mms']:.4f} mm/s  "
              f"RMSE(log)={m['rmse_log']:.4f}")

    # 9. Plots
    resid_log = np.log(np.clip(pgv_pred_test, eps, None)) - np.log(np.clip(pgv_true_test, eps, None))
    _plot_measured_vs_predicted(pgv_true_test, pgv_pred_test, metrics_test,
                                plots_dir / "measured_vs_predicted.png", cfg.benchmark_v4_rmse)
    _plot_residuals_vs(df_test[cfg.data.distance_col].to_numpy(np.float64), resid_log,
                       "distance to track [m]", "CNN v2 — residuals vs distance",
                       plots_dir / "residuals_vs_distance.png")
    _plot_residuals_vs(pgv_true_test, resid_log, "measured PGV_z [mm/s]",
                       "CNN v2 — residuals vs PGV", plots_dir / "residuals_vs_pgv.png", logx=True)
    _plot_residuals_by_cropped(resid_log, was_cropped_test, plots_dir / "residuals_by_cropped.png")
    _plot_residuals_vs_duration(duration_test, resid_log, was_cropped_test,
                                plots_dir / "residuals_vs_duration.png")
    hist_df = pd.DataFrame(artifacts.history)
    if not hist_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist_df["epoch"], hist_df["train_loss"], label="train")
        ax.plot(hist_df["epoch"], hist_df["val_loss"], label="val")
        ax.axvline(artifacts.best_epoch, color="k", ls="--", lw=0.8, label=f"best ep {artifacts.best_epoch}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (log space)")
        ax.set_title("CNN v2 learning curve")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(plots_dir / "learning_curve.png", dpi=140)
        plt.close(fig)

    # 10. Save artefacts
    hist_df.to_csv(build_dir / cfg.output.history_filename, index=False)
    pred_df = df_test[["event_id", cfg.data.sensor_col, cfg.data.distance_col,
                       cfg.data.pgv_col, "was_cropped", "duration_original_s"]].copy()
    pred_df["pgv_pred_mms"] = pgv_pred_test
    pred_df["log_pgv_true"] = np.log(np.clip(pgv_true_test, eps, None))
    pred_df["log_pgv_pred"] = log_pred_test
    pred_df["resid_log"] = resid_log
    pred_df.to_parquet(build_dir / cfg.output.predictions_filename, index=False)

    split_manifest = {
        "train_event_ids": sorted(df_train["event_id"].unique().tolist()),
        "val_event_ids": sorted(df_val["event_id"].unique().tolist()),
        "test_event_ids": sorted(df_test["event_id"].unique().tolist()),
        "n_train_rows": int(len(df_train)), "n_val_rows": int(len(df_val)),
        "n_test_rows": int(len(df_test)),
    }
    with open(build_dir / cfg.output.split_manifest_filename, "w") as fh:
        json.dump(split_manifest, fh, indent=2)

    summary = {
        "version": cfg.output.version_name, "run_tag": run_tag,
        "created": datetime.now().isoformat(timespec="seconds"), "device": str(device),
        "waveform_build": str(wave_dir), "parquet_v2": str(v2_path),
        "n_params": int(n_params), "feature_names": feat_names,
        "use_waveform": cfg.model.use_waveform, "use_batchnorm": cfg.model.use_batchnorm,
        "metadata_mode": "B" if cfg.features.use_metadata else "A",
        "best_epoch": artifacts.best_epoch, "best_val_loss": artifacts.best_val_loss,
        "metrics_test": metrics_test, "metrics_val": metrics_val,
        "metrics_test_by_crop": crop_breakdown,
        "benchmark_v4_rmse": cfg.benchmark_v4_rmse,
        "benchmark_cnn_v1_rmse": cfg.benchmark_cnn_v1_rmse,
        "benchmark_scalar_only_rmse": cfg.benchmark_scalar_only_rmse,
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
