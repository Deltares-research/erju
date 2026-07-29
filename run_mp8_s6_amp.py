#!/usr/bin/env python3
"""Train and evaluate the targeted S6_amp MP8 spectral model.

Examples
--------
Local smoke test:
    python run_mp8_s6_amp.py --smoke-n 24 --epochs 8 --num-workers 0

Full cluster run:
    python run_mp8_s6_amp.py --output-root /p/11210978-erju-ai/holten_models/outputs/mp8_spectral_cnn_v002
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
import torch

from src.ml.spectral.config_spectral_v001 import get_config
from src.ml.spectral.dataset_spectral import build_datasets, make_loader
from src.ml.spectral.eval_spectral import evaluate_split, print_summary, save_eval_outputs
from src.ml.spectral.models_spectral import build_model, count_parameters
from src.ml.spectral.train_spectral import load_best_checkpoint, train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--smoke-n", type=int, default=None,
                   help="Use the same small subset for train/val/test.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-mixed-precision", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    cfg = get_config("S6_amp", seed=args.seed)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
        cfg.train.patience = min(cfg.train.patience, max(3, args.epochs))
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.train.lr = args.learning_rate
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_mixed_precision:
        cfg.train.mixed_prec = False

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_root = Path("/p/11210978-erju-ai/holten_models/outputs/mp8_spectral_cnn_v002")
    if os.name == "nt":
        default_root = Path(r"P:\11210978-erju-ai\holten_models\outputs\mp8_spectral_cnn_v002")
    root = args.output_root or default_root
    suffix = f"_smoke{args.smoke_n}" if args.smoke_n else ""
    out_dir = root / f"S6_amp_seed{cfg.seed}{suffix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 72)
    print(f"S6_amp MP8 spectral model | device={device} | output={out_dir}")
    print("=" * 72)

    train_ds, val_ds, test_ds, stats, events_df, band_cols = build_datasets(
        cfg, smoke_n=args.smoke_n
    )
    train_loader = make_loader(
        train_ds, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = make_loader(
        val_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = make_loader(
        test_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = build_model(cfg, train_ds.n_meta_features)
    print(f"Events: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Model parameters: {count_parameters(model):,}")

    metadata = {
        "config": cfg.to_dict(),
        "git_commit": git_commit(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_events": len(train_ds),
        "val_events": len(val_ds),
        "test_events": len(test_ds),
        "parameter_count": count_parameters(model),
        "band_columns": band_cols,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    stats.save(out_dir / "data_stats.pkl")

    history = train_model(model, train_loader, val_loader, device, cfg, out_dir)
    best_epoch = load_best_checkpoint(model, out_dir, device)
    print(f"Loaded best checkpoint from epoch {best_epoch}")

    val_res = evaluate_split(
        model, val_loader, stats, events_df, cfg, device, band_cols, split_name="val"
    )
    save_eval_outputs(val_res, out_dir / "metrics" / "val", cfg.name, "val", history)
    print_summary(val_res, cfg.name, "val")

    test_res = evaluate_split(
        model, test_loader, stats, events_df, cfg, device, band_cols, split_name="test"
    )
    save_eval_outputs(test_res, out_dir / "metrics" / "test", cfg.name, "test", history)
    print_summary(test_res, cfg.name, "test")

    print(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
