"""train_mp8_spectral_v001.py
============================
Entry point for the MP8 spectral / PGV CNN experiments.

Usage examples
--------------
# Full run, config S1, seed 42
python train_mp8_spectral_v001.py --config S1 --seed 42

# Smoke run (16 events, 50 epochs, no test eval)
python train_mp8_spectral_v001.py --config S1 --mode smoke

# Eval-only: load existing checkpoint and evaluate
python train_mp8_spectral_v001.py --config S1 --mode eval_only \
    --resume path/to/best_model.pt

# Custom output root
python train_mp8_spectral_v001.py --config S2 --out_root /custom/root
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.spectral.config_spectral_v001 import (
    DEFAULT_OUT_ROOT, get_config, META_BASELINE_RMSE_PER_BAND, PRIMARY_NOMINALS,
)
from src.ml.spectral.dataset_spectral import build_datasets, make_loader
from src.ml.spectral.eval_spectral import (
    evaluate_split, print_summary, save_eval_outputs,
)
from src.ml.spectral.models_spectral import build_model, count_parameters
from src.ml.spectral.train_spectral import load_best_checkpoint, train_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train / evaluate MP8 spectral prediction CNN"
    )
    p.add_argument("--config",   required=True,
                   choices=["S1", "S2", "S3", "S4", "S5", "P1"],
                   help="Experiment configuration name")
    p.add_argument("--seed",     type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--mode",     default="full",
                   choices=["full", "smoke", "eval_only"],
                   help="Run mode (default: full)")
    p.add_argument("--out_root", default=None,
                   help="Override default output root directory")
    p.add_argument("--resume",   default=None,
                   help="Path to checkpoint .pt file for resume / eval_only")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"mem={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    else:
        d = torch.device("cpu")
        print("  No GPU found; using CPU")
    return d


def _print_banner(cfg, mode: str, out_dir: Path) -> None:
    print("\n" + "=" * 70)
    print(f"  MP8 Spectral CNN PoC — {cfg.name}  (seed={cfg.seed})")
    print(f"  Description : {cfg.description}")
    print(f"  Mode        : {mode}")
    print(f"  Target type : {cfg.target_type}")
    print(f"  use_meta_residual={cfg.use_meta_residual}  "
          f"use_pgv_aux={cfg.use_pgv_aux}  "
          f"pgv_aux_weight={cfg.pgv_aux_weight}")
    arch = cfg.arch
    print(f"  Encoder     : {arch.encoder_type}  "
          f"channels={arch.conv_channels}")
    tr = cfg.train
    print(f"  Train       : lr={tr.lr}  wd={tr.weight_decay}  "
          f"bs={tr.batch_size}  epochs={tr.epochs}  patience={tr.patience}")
    print(f"  Output      : {out_dir}")
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    cfg    = get_config(args.config, seed=args.seed)
    mode   = args.mode
    device = _device()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ── Output directory ──────────────────────────────────────────────────────
    out_root = Path(args.out_root) if args.out_root else Path(DEFAULT_OUT_ROOT)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / "mp8_spectral_cnn_v001" / f"{cfg.name}_seed{cfg.seed}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("checkpoints", "predictions", "metrics", "plots", "logs"):
        (out_dir / sub).mkdir(exist_ok=True)

    _print_banner(cfg, mode, out_dir)

    # Save config
    cfg_dict = {
        "name": cfg.name, "description": cfg.description, "seed": cfg.seed,
        "target_type": cfg.target_type,
        "use_meta_residual": cfg.use_meta_residual,
        "use_pgv_aux": cfg.use_pgv_aux,
        "pgv_aux_weight": cfg.pgv_aux_weight,
        "arch": vars(cfg.arch), "train": vars(cfg.train),
    }
    (out_dir / "config.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8")

    # ── Data ──────────────────────────────────────────────────────────────────
    smoke_n = 16 if mode == "smoke" else None
    t0 = time.time()
    train_ds, val_ds, test_ds, stats, events_df, band_cols = \
        build_datasets(cfg, smoke_n=smoke_n)
    n_meta = train_ds.meta.shape[1]
    print(f"  Data loaded in {time.time() - t0:.1f}s  "
          f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
          f"n_meta={n_meta}  n_bands={len(band_cols)}")

    # Save DataStats
    with open(out_dir / "data_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    num_workers = 0 if mode == "smoke" else cfg.num_workers
    train_loader = make_loader(train_ds, cfg.train.batch_size, shuffle=True,
                               num_workers=num_workers)
    val_loader   = make_loader(val_ds, cfg.train.batch_size, shuffle=False,
                               num_workers=num_workers)
    test_loader  = make_loader(test_ds, cfg.train.batch_size, shuffle=False,
                               num_workers=num_workers)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, n_meta)
    n_params = count_parameters(model)
    print(f"  Model: {model.__class__.__name__}  trainable params={n_params:,}")

    if mode == "eval_only":
        if not args.resume:
            sys.exit("ERROR: --mode eval_only requires --resume <path>")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        history = None
        print(f"  Loaded checkpoint from {args.resume}")
    else:
        # ── Training ──────────────────────────────────────────────────────────
        # Override epochs for smoke mode
        if mode == "smoke":
            cfg.train.epochs  = 50
            cfg.train.patience = 20

        checkpoints_dir = out_dir / "checkpoints"
        history = train_model(
            model, train_loader, val_loader, device, cfg, checkpoints_dir,
        )

        # Load best checkpoint
        best_epoch = load_best_checkpoint(model, checkpoints_dir, device)
        print(f"  Loaded best checkpoint (epoch {best_epoch}) for evaluation")

        # Save best model to top-level output dir too
        import shutil
        best_src = checkpoints_dir / "best_model.pt"
        if best_src.exists():
            shutil.copy(best_src, out_dir / "model_best.pt")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics_dir = out_dir / "metrics"

    for split_name, loader in [("val", val_loader),
                                 ("test", test_loader) if mode != "smoke" else ()]:
        if loader is None:
            continue
        res = evaluate_split(
            model, loader, stats, events_df, cfg, device, band_cols,
            split_name=split_name,
        )
        print_summary(res, cfg.name, split_name)
        save_eval_outputs(res, metrics_dir, cfg.name, split_name,
                          history if split_name == "val" else None)

    # Also save history from checkpoints dir if it exists
    hist_path = out_dir / "checkpoints" / "train_history.json"
    if hist_path.exists():
        import shutil
        shutil.copy(hist_path, out_dir / "train_history.json")

    print(f"\n  All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
