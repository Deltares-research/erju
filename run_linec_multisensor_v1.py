#!/usr/bin/env python3
"""run_linec_multisensor_v1.py
==============================
Train + evaluate ONE of {M0, M1, M2} for one seed on the 5-sensor Line-C
spectral target (MP4/MP8/MP10/MP1/MP2, 19 bands), using the validated
51-channel waveform product and the fixed 1,697-event split.

Examples
--------
Local deterministic smoke test:
    venv\\Scripts\\python.exe run_linec_multisensor_v1.py --model M1 --smoke-n 24 --epochs 3 --num-workers 0

Full cluster run (one model, one seed):
    python run_linec_multisensor_v1.py --model M2 --seed 42 \\
        --output-root /p/11210978-erju-ai/holten_models/outputs/linec_multisensor_v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.ml.linec_multisensor.data import SENSORS, build_datasets, make_loader
from src.ml.linec_multisensor.engine import load_best_checkpoint, train_model
from src.ml.linec_multisensor.metrics import (
    bootstrap_macro_rmse_ci, compute_full_metrics, plot_amplitude_compression,
    plot_measured_vs_predicted, plot_per_band_rmse, plot_training_history,
    run_inference, to_jsonable,
)
from src.ml.linec_multisensor.models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["M0", "M1", "M2"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--smoke-n", type=int, default=None, help="subsample N events per split for a fast smoke test")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-mixed-precision", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _default_root() -> Path:
    return (Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")) \
        / "holten_models" / "outputs" / "linec_multisensor_v1"


def _save_predictions(pred_db, true_db, r_m, tracks, events, split_name, band_nominal, out_dir: Path) -> None:
    n_events, n_sensors, n_bands = true_db.shape
    ev_rep = np.repeat(events, n_sensors * n_bands)
    sensor_rep = np.tile(np.repeat(SENSORS, n_bands), n_events)
    band_rep = np.tile(band_nominal, n_events * n_sensors)
    track_rep = np.repeat(tracks, n_sensors * n_bands)
    dist_rep = np.repeat(r_m.reshape(-1), n_bands)
    df = pd.DataFrame({
        "event_id": ev_rep, "split": split_name, "sensor_id": sensor_rep, "track_number": track_rep,
        "distance_m": dist_rep, "band_nominal_hz": band_rep,
        "target_db": true_db.reshape(-1), "predicted_db": pred_db.reshape(-1),
    })
    df.to_parquet(out_dir / f"predictions_{split_name}.parquet", index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_smoke{args.smoke_n}" if args.smoke_n else ""
    root = args.output_root or _default_root()
    out_dir = root / f"{args.model}_seed{args.seed}{suffix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 78)
    print(f"Line-C multi-sensor experiment | model={args.model} seed={args.seed} device={device}")
    print(f"output: {out_dir}")
    print("=" * 78)

    train_ds, val_ds, test_ds, stats, events_df = build_datasets(smoke_n=args.smoke_n)
    print(f"Events: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  n_meta={stats.n_meta}")

    batch_size = args.batch_size or 32
    train_loader = make_loader(train_ds, batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, stats.n_meta, n_o0=stats.n_o0)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    config = {
        "model": args.model, "seed": args.seed, "smoke_n": args.smoke_n,
        "epochs_arg": args.epochs, "patience_arg": args.patience, "batch_size": batch_size,
        "n_meta": stats.n_meta, "sensors": SENSORS, "parameter_count": n_params,
        "git_commit": git_commit(), "device": str(device), "torch_version": torch.__version__,
        "created": datetime.now().isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    stats.save(out_dir / "data_stats.pkl")

    history = train_model(
        model, train_loader, val_loader, device, out_dir,
        target_mean=stats.target_mean, target_std=stats.target_std,
        epochs=args.epochs, patience=args.patience, mixed_prec=not args.no_mixed_precision,
    )
    best_epoch = load_best_checkpoint(model, out_dir, device)
    print(f"Loaded best checkpoint from epoch {best_epoch}")
    plot_training_history(history, out_dir)

    results = {"config": config, "best_epoch": best_epoch, "parameter_count": n_params}
    for split_name, loader, ds in (("val", val_loader, val_ds), ("test", test_loader, test_ds)):
        pred_db, true_db, r_m, tracks, ev_idx = run_inference(model, loader, device)
        events = ds.events[ev_idx]
        metrics = compute_full_metrics(pred_db, true_db, r_m, stats.band_nominal_hz)
        metrics["bootstrap_macro_rmse_ci"] = bootstrap_macro_rmse_ci(pred_db, true_db, n_boot=args.n_boot, seed=args.seed)
        results[split_name] = metrics
        _save_predictions(pred_db, true_db, r_m, tracks, events, split_name, stats.band_nominal_hz, out_dir)
        plot_measured_vs_predicted(true_db, pred_db, out_dir, split_name)
        plot_per_band_rmse(stats.band_nominal_hz, metrics["per_band_rmse_db"], out_dir, split_name)
        plot_amplitude_compression(true_db, pred_db, out_dir, split_name)
        print(f"  {split_name}: macro_rmse={metrics['macro_rmse_db']:.3f} dB  "
              f"total_rms_r2={metrics['total_rms']['r2']:.3f}  "
              f"strong_event_rmse={metrics['strong_event_metrics']['macro_rmse_db']:.3f} dB")

    (out_dir / "results.json").write_text(json.dumps(to_jsonable(results), indent=2))
    print(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
