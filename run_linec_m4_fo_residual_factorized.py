#!/usr/bin/env python3
"""run_linec_m4_fo_residual_factorized.py
==========================================
M4_FO_RESIDUAL_FACTORIZED: FO-signal-only residual correction on top of a
frozen M0 (metadata) baseline, factorized into (1) a single amplitude
(total-level) correction and (2) a low-rank PCA shape-residual correction.

Does NOT modify or retrain M0, M1, M2 or M3 -- their completed seed-42
checkpoints are loaded read-only (M0 frozen permanently as the baseline; M3's
FO trunk used only as the initialization for M4's trainable waveform
encoder).

Pipeline
--------
1. Load frozen M0 (metadata baseline) -- never updated.
2. Precompute B[e] = M0(e) for every TRAIN event; fit the residual PCA
   (<=20 components, >=90% train-only cumulative variance) and the
   amplitude/PCA-score standardization stats on TRAIN residuals only.
3. Build M4Model: WaveformOnlyEncoder initialized from M3's completed
   checkpoint (FO waveform + raw-amp only -- no metadata/track/distance),
   with two new zero-initialized heads (amplitude scalar, PCA shape
   coefficients) reconstructing L_hat = B + A_hat + R_hat.
4. Train: first `--freeze-epochs` epochs with the FO trunk frozen (heads
   only), then unfreeze and fine-tune the trunk. Loss = existing
   composite_loss(L_hat, L_true) + huber(A_hat_z, A_target_z) +
   huber(z_hat, z_target), fixed equal weights (not swept).
5. Evaluate on val/test with the existing metrics/plotting pipeline, plus
   amplitude-correction RMSE/correlation diagnostics.

Clean pipeline conventions preserved: FP32 only, SAFE_SQRT_EPS (inherited
via the shared ResNet2DEncoder pooling classes), finite guards, gradient
clipping, validation-based early stopping, fixed 1,697-event split, seed 42.

Examples
--------
Local smoke check (requires a local CUDA GPU):
    venv\\Scripts\\python.exe run_linec_m4_fo_residual_factorized.py --seed 42 --smoke --num-workers 0

Full cluster run:
    python run_linec_m4_fo_residual_factorized.py --seed 42 \\
        --output-root /p/11210978-erju-ai/holten_models/outputs/linec_multisensor_v1/M4
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_linec_multisensor_v1 import _default_root, _log_environment, _save_predictions, git_commit, set_seed
from src.ml.linec_multisensor.data import SENSORS, build_datasets, make_loader
from src.ml.linec_multisensor.engine import TRAIN_HP, _check_batch_finite, _evaluate, load_best_checkpoint
from src.ml.linec_multisensor.losses import composite_loss
from src.ml.linec_multisensor.metrics import (
    bootstrap_macro_rmse_ci, compute_full_metrics, plot_amplitude_compression,
    plot_mean_median_spectra, plot_measured_vs_predicted, plot_per_band_rmse,
    plot_random_events, plot_representative_events, plot_spectral_quantile_bands,
    plot_training_history, run_inference, to_jsonable,
)
from src.ml.linec_multisensor.models import build_model
from src.ml.linec_multisensor.models_m4 import AUX_HUBER_DELTA, M4Model, PCAResidualStats, fit_residual_pca
from src.ml.linec_multisensor.stability import save_diagnostic_and_abort

_ROOT = Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")
_DEFAULT_M0_CKPT = _ROOT / "holten_models/outputs/linec_multisensor_v1/M0/M0_seed42_20260803_041551/best_model.pt"
_DEFAULT_M3_CKPT = _ROOT / "holten_models/outputs/linec_multisensor_v1/M3/M3_seed42_20260803_055820/best_model.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--freeze-epochs", type=int, default=5)
    p.add_argument("--max-pca-components", type=int, default=20)
    p.add_argument("--pca-variance-threshold", type=float, default=0.90)
    p.add_argument("--m0-checkpoint", type=Path, default=_DEFAULT_M0_CKPT)
    p.add_argument("--m3-checkpoint", type=Path, default=_DEFAULT_M3_CKPT)
    return p.parse_args()


def train_m4(model: M4Model, train_loader, val_loader, device: torch.device, out_dir: Path,
             target_mean: np.ndarray, target_std: np.ndarray, epochs=None, patience=None,
             freeze_epochs: int = 5):
    """Mirrors engine.train_model, with (a) a staged trunk freeze/unfreeze
    and (b) two extra huber loss terms (fixed, equal weight) on top of the
    existing composite_loss. Validation-based early stopping uses the same
    val macro RMSE (dB) criterion as M0/M1/M2/M3."""
    hp = TRAIN_HP
    epochs = epochs if epochs is not None else hp["epochs"]
    patience = patience if patience is not None else hp["patience"]
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    model.to(device)
    t_mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    t_std = torch.as_tensor(target_std, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    history = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            model.unfreeze_trunk()
            print(f"  [epoch {epoch}] unfreezing FO trunk for fine-tuning")

        t0 = time.time()
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        comp_sums: dict = {}
        grad_norm_sum, grad_norm_max = 0.0, 0.0

        for batch_idx, (wf, meta, nv, tgt, r, track, _) in enumerate(train_loader):
            wf, meta, nv, tgt, r, track = (t.to(device) for t in (wf, meta, nv, tgt, r, track))
            _check_batch_finite("train_input", epoch, batch_idx, out_dir, wf=wf, meta=meta, tgt=tgt, r=r)

            opt.zero_grad(set_to_none=True)
            l_hat, a_hat_z, z_hat, b = model.forward_full(wf, meta, nv, r, track)
            _check_batch_finite("train_pred", epoch, batch_idx, out_dir,
                                 pred=l_hat, a_hat_z=a_hat_z, z_hat=z_hat, b=b)

            a_target_z, z_target, _ = model.target_transform(tgt, b)
            _check_batch_finite("train_target", epoch, batch_idx, out_dir,
                                 a_target_z=a_target_z, z_target=z_target)

            loss_main, comps = composite_loss(l_hat, tgt, t_mean, t_std)
            l_amp = F.huber_loss(a_hat_z, a_target_z, delta=AUX_HUBER_DELTA)
            l_shape_pca = F.huber_loss(z_hat, z_target, delta=AUX_HUBER_DELTA)
            loss = loss_main + l_amp + l_shape_pca

            if not bool(torch.isfinite(loss)):
                save_diagnostic_and_abort(out_dir, "train_loss", epoch, batch_idx,
                                           {"pred": l_hat, "tgt": tgt},
                                           message=f"loss is non-finite: {comps}")

            loss.backward()

            grad_norm_sq = 0.0
            for pname, p in model.named_parameters():
                if p.grad is not None:
                    if not bool(torch.isfinite(p.grad).all()):
                        save_diagnostic_and_abort(
                            out_dir, "train_grad", epoch, batch_idx,
                            {"pred": l_hat, "tgt": tgt, f"grad[{pname}]": p.grad},
                            message=f"non-finite gradient detected in parameter '{pname}'")
                    grad_norm_sq += float(p.grad.detach().float().pow(2).sum())
            if not np.isfinite(grad_norm_sq):
                save_diagnostic_and_abort(out_dir, "train_grad_norm", epoch, batch_idx, {},
                                           message=f"gradient norm is non-finite ({grad_norm_sq})")
            grad_norm = grad_norm_sq ** 0.5
            grad_norm_sum += grad_norm
            grad_norm_max = max(grad_norm_max, grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()

            train_loss_sum += float(loss.item())
            n_batches += 1
            for k, v in comps.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v
            comp_sums["amp_huber"] = comp_sums.get("amp_huber", 0.0) + float(l_amp.item())
            comp_sums["shape_pca_huber"] = comp_sums.get("shape_pca_huber", 0.0) + float(l_shape_pca.item())

        train_loss = train_loss_sum / max(n_batches, 1)
        val_rmse_db = _evaluate(model, val_loader, device, out_dir, epoch)
        sched.step(val_rmse_db)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss, "val_macro_rmse_db": val_rmse_db,
               "lr": opt.param_groups[0]["lr"], "elapsed_s": round(elapsed, 2),
               "grad_norm_mean": grad_norm_sum / max(n_batches, 1), "grad_norm_max": grad_norm_max,
               "trunk_frozen": epoch <= freeze_epochs}
        row.update({f"train_{k}": v / max(n_batches, 1) for k, v in comp_sums.items()})
        history.append(row)
        print(f"  epoch {epoch:4d}  train_loss={train_loss:.4f}  val_macro_rmse={val_rmse_db:.4f} dB  "
              f"lr={row['lr']:.2e}  grad_norm(mean/max)={row['grad_norm_mean']:.3f}/{row['grad_norm_max']:.3f}  "
              f"trunk_frozen={row['trunk_frozen']}  ({elapsed:.1f}s)")

        improved = val_rmse_db < best_val - 1e-4
        if improved:
            best_val = val_rmse_db
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_macro_rmse_db": val_rmse_db}, out_dir / "best_model.pt")
        else:
            epochs_since_best += 1
        if epochs_since_best >= patience:
            print(f"  early stopping at epoch {epoch} (best epoch {best_epoch}, val={best_val:.4f} dB)")
            break

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    if best_epoch == -1:
        (out_dir / "RUN_INVALID").write_text(
            "Run invalid: training loop completed without ever recording a finite "
            "validation checkpoint (best_epoch=-1).\n"
        )
    return history, best_epoch


@torch.no_grad()
def _amplitude_correction_diagnostics(model: M4Model, loader, device: torch.device) -> dict:
    """RMSE/correlation between the learned amplitude correction A_hat and
    the true amplitude correction A_target it was trained to predict."""
    model.eval()
    a_hat_list, a_target_list = [], []
    for wf, meta, nv, tgt, r, track, _ in loader:
        wf, meta, nv, tgt, r, track = (t.to(device) for t in (wf, meta, nv, tgt, r, track))
        _, a_hat_z, _, b = model.forward_full(wf, meta, nv, r, track)
        _, _, a_target = model.target_transform(tgt, b)
        a_hat = a_hat_z * model.a_std + model.a_mean
        a_hat_list.append(a_hat.cpu().numpy())
        a_target_list.append(a_target.cpu().numpy())
    a_hat_arr = np.concatenate(a_hat_list)
    a_target_arr = np.concatenate(a_target_list)
    rmse = float(np.sqrt(np.mean((a_hat_arr - a_target_arr) ** 2)))
    corr = float(np.corrcoef(a_hat_arr, a_target_arr)[0, 1]) if np.std(a_target_arr) > 1e-9 else float("nan")
    return {"rmse_db": rmse, "pearson": corr}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    smoke_n = args.smoke_n
    epochs = args.epochs
    patience = args.patience
    if args.smoke:
        smoke_n = smoke_n if smoke_n is not None else 32
        epochs = epochs if epochs is not None else 3
        patience = patience if patience is not None else 3

    device = _log_environment("M4")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_smoke{smoke_n}" if smoke_n else ""
    root = args.output_root or _default_root()
    out_dir = root / f"M4_seed{args.seed}{suffix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 78)
    print(f"Line-C multi-sensor experiment | model=M4_FO_RESIDUAL_FACTORIZED seed={args.seed} device={device}")
    print(f"output: {out_dir}")
    print("=" * 78)

    train_ds, val_ds, test_ds, stats, events_df = build_datasets(smoke_n=smoke_n)
    print(f"Events: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  n_meta={stats.n_meta}")

    batch_size = args.batch_size or 32
    train_loader = make_loader(train_ds, batch_size, shuffle=True, num_workers=args.num_workers)
    train_loader_ordered = make_loader(train_ds, batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, batch_size, shuffle=False, num_workers=args.num_workers)

    # ── 1. frozen M0 baseline ────────────────────────────────────────────────
    m0_model = build_model("M0", stats.n_meta).to(device)
    m0_ckpt = torch.load(args.m0_checkpoint, map_location=device)
    m0_model.load_state_dict(m0_ckpt["model_state"])
    m0_model.eval()
    for p_ in m0_model.parameters():
        p_.requires_grad_(False)
    print(f"Loaded frozen M0 baseline from {args.m0_checkpoint} (epoch {m0_ckpt['epoch']})")

    # ── 2. residual PCA + standardization stats, TRAIN split only ──────────
    pca_stats = fit_residual_pca(m0_model, train_loader_ordered, device,
                                  max_components=args.max_pca_components,
                                  variance_threshold=args.pca_variance_threshold)
    cum_var = float(np.cumsum(pca_stats.explained_variance_ratio_full)[pca_stats.n_components - 1])
    print(f"Residual PCA: n_components={pca_stats.n_components}  cumulative_train_variance={cum_var:.4f}  "
          f"a_mean={pca_stats.a_mean:.4f} dB  a_std={pca_stats.a_std:.4f} dB")

    # ── 3. build M4 (FO trunk initialized from M3, both new heads zero-init) ─
    m3_ckpt = torch.load(args.m3_checkpoint, map_location=device)
    m3_trunk_state = {k[len("trunk."):]: v for k, v in m3_ckpt["model_state"].items() if k.startswith("trunk.")}
    model = M4Model(stats.n_meta, m0_model.state_dict(), m3_trunk_state, pca_stats).to(device)
    model.freeze_trunk()
    print(f"Loaded M3 FO trunk from {args.m3_checkpoint} (epoch {m3_ckpt['epoch']}); trunk frozen for "
          f"the first {args.freeze_epochs} epoch(s)")

    n_params = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("frozen_m0."))
    n_params_frozen_m0 = sum(p.numel() for name, p in model.named_parameters() if name.startswith("frozen_m0."))
    print(f"Model parameters: {n_params:,} trainable-eventually (+ {n_params_frozen_m0:,} frozen M0)")

    config = {
        "model": "M4_FO_RESIDUAL_FACTORIZED", "seed": args.seed, "smoke": args.smoke, "smoke_n": smoke_n,
        "epochs_arg": epochs, "patience_arg": patience, "batch_size": batch_size,
        "n_meta": stats.n_meta, "sensors": SENSORS, "parameter_count": n_params,
        "frozen_m0_parameter_count": n_params_frozen_m0,
        "freeze_epochs": args.freeze_epochs, "pca_n_components": pca_stats.n_components,
        "pca_cumulative_variance": cum_var, "pca_variance_threshold": args.pca_variance_threshold,
        "pca_max_components": args.max_pca_components,
        "m0_checkpoint": str(args.m0_checkpoint), "m3_checkpoint": str(args.m3_checkpoint),
        "git_commit": git_commit(), "device": str(device), "torch_version": torch.__version__,
        "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
        "precision": "float32", "created": datetime.now().isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    stats.save(out_dir / "data_stats.pkl")
    pca_stats.save(out_dir / "residual_pca_stats.pkl")

    history, best_epoch = train_m4(
        model, train_loader, val_loader, device, out_dir,
        target_mean=stats.target_mean, target_std=stats.target_std,
        epochs=epochs, patience=patience, freeze_epochs=args.freeze_epochs,
    )
    plot_training_history(history, out_dir)

    if best_epoch == -1:
        print(f"[INVALID RUN] no finite validation checkpoint was ever recorded; see {out_dir / 'RUN_INVALID'}")
        return

    best_epoch = load_best_checkpoint(model, out_dir, device)
    print(f"Loaded best checkpoint from epoch {best_epoch}")

    results = {"config": config, "best_epoch": best_epoch, "parameter_count": n_params}
    for split_name, loader, ds in (("val", val_loader, val_ds), ("test", test_loader, test_ds)):
        pred_db, true_db, r_m, tracks, ev_idx = run_inference(model, loader, device)
        events = ds.events[ev_idx]
        metrics = compute_full_metrics(pred_db, true_db, r_m, stats.band_nominal_hz)
        metrics["bootstrap_macro_rmse_ci"] = bootstrap_macro_rmse_ci(pred_db, true_db, n_boot=args.n_boot, seed=args.seed)
        metrics["amplitude_correction"] = _amplitude_correction_diagnostics(model, loader, device)
        results[split_name] = metrics
        _save_predictions(pred_db, true_db, r_m, tracks, events, split_name, stats.band_nominal_hz, out_dir)
        plot_measured_vs_predicted(true_db, pred_db, out_dir, split_name)
        plot_per_band_rmse(stats.band_nominal_hz, metrics["per_band_rmse_db"], out_dir, split_name)
        plot_amplitude_compression(true_db, pred_db, out_dir, split_name)
        plot_mean_median_spectra(true_db, pred_db, stats.band_nominal_hz, out_dir, split_name)
        plot_random_events(true_db, pred_db, events, stats.band_nominal_hz, out_dir, split_name, n_events=10, seed=args.seed)
        plot_spectral_quantile_bands(true_db, pred_db, stats.band_nominal_hz, out_dir, split_name)
        plot_representative_events(true_db, pred_db, stats.band_nominal_hz, out_dir, split_name)
        print(f"  {split_name}: macro_rmse={metrics['macro_rmse_db']:.3f} dB  "
              f"total_rms_r2={metrics['total_rms']['r2']:.3f}  "
              f"amp_corr_rmse={metrics['amplitude_correction']['rmse_db']:.3f} dB  "
              f"amp_corr_pearson={metrics['amplitude_correction']['pearson']:.3f}  "
              f"strong_event_rmse={metrics['strong_event_metrics']['macro_rmse_db']:.3f} dB")

    (out_dir / "results.json").write_text(json.dumps(to_jsonable(results), indent=2))
    print(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
