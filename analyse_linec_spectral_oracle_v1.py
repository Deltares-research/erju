"""analyse_linec_spectral_oracle_v1.py
======================================
Target-only Line-C spectral oracle diagnostic (M0/M1/M2 are OUT OF SCOPE).

Fits two purely-analytic propagation decoders on the authoritative 5-sensor
spectral targets (no neural network, no waveform input) and evaluates them
in two oracle modes. Does not modify or import the S1-S6 training pipeline
except for read-only reuse of its metric / total-RMS helper functions.

Decoders (fit per track t in {1,2}, per one-third-octave band f):

    O0:  L_hat[e,j,f] = C[e,f] - 20*n_track[f]*log10(r[e,j]/r0)
    O1:  L_hat[e,j,f] = C[e,f] - 20*n_track[f]*log10(r[e,j]/r0)
                              - alpha_track_db[f]*(r[e,j]-r0)

    C[e,f]            event source spectrum at r = r0 (event+band, not sensor-specific)
    n_track[f]        smooth track-specific geometric-spreading exponent
    alpha_track_db[f] smooth track-specific attenuation [dB/m] (O1 only)

n_track / alpha_track are fit on TRAIN events only, per band, jointly across
all 19 bands with a second-difference smoothness penalty (closed-form ridge
solve, no gradient descent). C[e,f] has a closed form given fixed n/alpha
(mean of the propagation-corrected level over the sensors used) and is
re-estimated for every event/mode being evaluated -- never smoothed, never
shared across events, never sensor-specific.

Evaluation modes (both O0 and O1, on train/val/test):
    A. Five-sensor reconstruction ("representation oracle"): C[e,f] uses all
       5 sensors of that same event -- an upper bound, NOT a predictive model.
    B. Leave-one-sensor-out oracle: C[e,f] uses only the other 4 sensors; the
       held sensor's target is never used to fit C for that event.

Inputs (read-only, authoritative sources -- see NETCDF/spectral audits):
    P:/11210978-erju-ai/holten_spectral_targets_v002_corrected/spectral_targets.parquet
    sites/holten.json (via src.utils.geometry_utils.apply_corrected_distances)

Outputs (never overwritten -- new timestamped subfolder every run):
    P:/11210978-erju-ai/holten_models/outputs/linec_spectral_oracle_v1/<run_id>/
        results.json, propagation_curves.csv,
        event_source_spectrum_C_train_{O0,O1}.parquet,
        mode_b_predictions_{O0,O1}.parquet (event/sensor/band leave-one-out predictions),
        plots/*.png, run_manifest.json

Run:
    python analyse_linec_spectral_oracle_v1.py --smoke   # fast code/smoke check
    python analyse_linec_spectral_oracle_v1.py            # full diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.utils.geometry_utils import apply_corrected_distances
from src.ml.spectral.eval_spectral import (
    _rmse, _mae, _r2, _pearson, _spearman,
    total_rms_from_db,
)

# ── Config ────────────────────────────────────────────────────────────────────

def _root():
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")

SPEC_PATH = _root() / "holten_spectral_targets_v002_corrected" / "spectral_targets.parquet"
OUT_ROOT = _root() / "holten_models" / "outputs" / "linec_spectral_oracle_v1"

SENSORS = ["MP4", "MP8", "MP10", "MP1", "MP2"]
R0 = 10.0
SMOOTH_LAMBDA_REL = 0.05  # smoothness penalty weight, relative to per-band data strength
N_ALS_ITER_FULL = 8
N_ALS_ITER_SMOKE = 3
N_BOOTSTRAP_FULL = 30
N_BOOTSTRAP_SMOKE = 5
N_EVENTS_SMOKE = 200


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_tensors():
    cols = [
        "event_id", "sensor_id", "band_index", "band_nominal_hz",
        "fully_inside_valid_range", "velocity_band_level_db",
        "track_number", "split",
    ]
    df = pd.read_parquet(SPEC_PATH, columns=cols)
    df = df[df["sensor_id"].isin(SENSORS) & df["fully_inside_valid_range"]].copy()

    band_ids = sorted(df["band_index"].unique())
    n_bands = len(band_ids)
    if n_bands != 19:
        print(f"[WARN] expected 19 fully-inside bands, found {n_bands}")
    band_nominal = (
        df[["band_index", "band_nominal_hz"]]
        .drop_duplicates()
        .set_index("band_index")
        .loc[band_ids, "band_nominal_hz"]
        .to_numpy(dtype=np.float64)
    )

    ev_meta = (
        df[["event_id", "track_number", "split"]]
        .drop_duplicates(subset="event_id")
        .set_index("event_id")
    )

    pivot = df.pivot_table(index=["event_id", "sensor_id"], columns="band_index", values="velocity_band_level_db")
    pivot = pivot[band_ids]

    sensor_frames = {}
    common_events = None
    for s in SENSORS:
        sub = pivot.xs(s, level="sensor_id")
        sensor_frames[s] = sub
        ev_set = set(sub.dropna(how="any").index)
        common_events = ev_set if common_events is None else (common_events & ev_set)

    valid_track_events = set(ev_meta.index[ev_meta["track_number"].isin([1, 2])])
    common_events = common_events & valid_track_events
    events = np.array(sorted(common_events))
    n_events = len(events)
    print(f"Loaded {len(df):,} rows -> {n_events:,} events with complete 5-sensor / 19-band / known-track data "
          f"(dropped {1697 - n_events} of 1697 canonical events)")

    L = np.stack(
        [sensor_frames[s].reindex(events).to_numpy(dtype=np.float64) for s in SENSORS],
        axis=1,
    )  # (n_events, 5, 19)
    TRACK = ev_meta.loc[events, "track_number"].to_numpy(dtype=int)
    SPLIT = ev_meta.loc[events, "split"].to_numpy(dtype=str)

    combo = pd.DataFrame([(s, t) for s in SENSORS for t in (1, 2)], columns=["sensor_id", "track_number"])
    combo["acc_distance_to_track_m"] = np.nan  # required input column; overwritten by apply_corrected_distances
    combo = apply_corrected_distances(combo, sensor_col="sensor_id", track_col="track_number")
    dist_lookup = {
        (row.sensor_id, row.track_number): row.effective_distance_to_active_track_m
        for row in combo.itertuples()
    }
    R = np.zeros((n_events, len(SENSORS)), dtype=np.float64)
    for j, s in enumerate(SENSORS):
        R[:, j] = np.where(TRACK == 1, dist_lookup[(s, 1)], dist_lookup[(s, 2)])

    for split_name in ("train", "val", "test"):
        for t in (1, 2):
            n = int(np.sum((SPLIT == split_name) & (TRACK == t)))
            print(f"  split={split_name:5s} track={t}  n_events={n}")

    return {
        "L": L, "R": R, "TRACK": TRACK, "SPLIT": SPLIT,
        "events": events, "band_nominal": band_nominal,
    }


# ── Smoothness-penalized joint band fit ──────────────────────────────────────

def _second_diff_matrix(n: int) -> np.ndarray:
    D = np.zeros((n - 2, n))
    idx = np.arange(n - 2)
    D[idx, idx] = 1.0
    D[idx, idx + 1] = -2.0
    D[idx, idx + 2] = 1.0
    return D


def _penalty_matrix(n_bands: int) -> np.ndarray:
    D = _second_diff_matrix(n_bands)
    return D.T @ D


def _solve_joint_band_params(X: np.ndarray, Y: np.ndarray, penalty: np.ndarray,
                              smooth_lambda_rel: float, model: str):
    """X: (n_obs, p) shared design (p=1 O0, p=2 O1). Y: (n_obs, n_bands) targets."""
    n_bands = Y.shape[1]
    A = X.T @ X
    Bmat = X.T @ Y
    I = np.eye(n_bands)
    if model == "O0":
        lam = smooth_lambda_rel * A[0, 0]
        M = A[0, 0] * I + lam * penalty
        n_vec = np.linalg.solve(M, Bmat[0])
        return np.stack([n_vec]), A
    lam_n = smooth_lambda_rel * A[0, 0]
    lam_a = smooth_lambda_rel * A[1, 1]
    top = np.hstack([A[0, 0] * I + lam_n * penalty, A[0, 1] * I])
    bot = np.hstack([A[1, 0] * I, A[1, 1] * I + lam_a * penalty])
    M = np.vstack([top, bot])
    rhs = np.concatenate([Bmat[0], Bmat[1]])
    sol = np.linalg.solve(M, rhs)
    return np.stack([sol[:n_bands], sol[n_bands:]]), A


def _fit_single_track(L_tr: np.ndarray, R_tr: np.ndarray, model: str, n_iter: int,
                       smooth_lambda_rel: float, penalty: np.ndarray):
    n_bands = L_tr.shape[2]
    n_f = np.zeros(n_bands)
    a_f = np.zeros(n_bands)
    history = []
    A = None
    for _ in range(n_iter):
        x1 = -20.0 * np.log10(R_tr / R0)  # (n,5)
        if model == "O1":
            x2 = -(R_tr - R0)
            corr = x1[:, :, None] * n_f[None, None, :] + x2[:, :, None] * a_f[None, None, :]
        else:
            corr = x1[:, :, None] * n_f[None, None, :]
        C = np.mean(L_tr - corr, axis=1)  # (n,19)
        Y = (L_tr - C[:, None, :]).reshape(-1, n_bands)
        X1flat = x1.reshape(-1)
        if model == "O1":
            X = np.column_stack([X1flat, x2.reshape(-1)])
        else:
            X = X1flat[:, None]
        beta, A = _solve_joint_band_params(X, Y, penalty, smooth_lambda_rel, model)
        new_n = beta[0]
        new_a = beta[1] if model == "O1" else np.zeros(n_bands)
        delta = float(np.max(np.abs(new_n - n_f)))
        if model == "O1":
            delta = max(delta, float(np.max(np.abs(new_a - a_f))))
        n_f, a_f = new_n, new_a
        history.append(delta)
        if delta < 1e-5:
            break
    return n_f, a_f, A, history


def fit_propagation(L, R, TRACK, SPLIT, model: str, n_iter: int, smooth_lambda_rel: float):
    n_bands = L.shape[2]
    penalty = _penalty_matrix(n_bands)
    out = {}
    for track in (1, 2):
        sel = (SPLIT == "train") & (TRACK == track)
        n_f, a_f, A, history = _fit_single_track(L[sel], R[sel], model, n_iter, smooth_lambda_rel, penalty)
        cond_number = float(np.linalg.cond(A)) if model == "O1" else None
        corr_n_alpha = float(np.corrcoef(n_f, a_f)[0, 1]) if model == "O1" else None
        out[track] = {
            "n_track": n_f,
            "alpha_track": a_f if model == "O1" else None,
            "n_iterations": len(history),
            "convergence_history": history,
            "condition_number": cond_number,
            "n_alpha_correlation": corr_n_alpha,
            "n_train_events": int(sel.sum()),
        }
    return out


# ── Prediction / metrics ──────────────────────────────────────────────────────

def compute_corr(R_eval: np.ndarray, TRACK_eval: np.ndarray, fit: dict, model: str) -> np.ndarray:
    n_bands = len(fit[1]["n_track"])
    n_arr = np.where(TRACK_eval[:, None] == 1, fit[1]["n_track"][None, :], fit[2]["n_track"][None, :])
    x1 = -20.0 * np.log10(R_eval / R0)
    corr = x1[:, :, None] * n_arr[:, None, :]
    if model == "O1":
        a_arr = np.where(TRACK_eval[:, None] == 1, fit[1]["alpha_track"][None, :], fit[2]["alpha_track"][None, :])
        x2 = -(R_eval - R0)
        corr = corr + x2[:, :, None] * a_arr[:, None, :]
    return corr


def compute_metrics(true_db, pred_db, r_m, band_nominal) -> dict:
    true_db = np.asarray(true_db, dtype=np.float64)
    pred_db = np.asarray(pred_db, dtype=np.float64)
    r_m = np.asarray(r_m, dtype=np.float64)
    n_bands = true_db.shape[1]
    resid = true_db - pred_db

    true_tot = total_rms_from_db(true_db)
    pred_tot = total_rms_from_db(pred_db)

    true_level_db = 10.0 * np.log10(np.clip(np.sum(10.0 ** (true_db / 10.0), axis=1), 1e-300, None))
    pred_level_db = 10.0 * np.log10(np.clip(np.sum(10.0 ** (pred_db / 10.0), axis=1), 1e-300, None))
    true_shape = true_db - true_level_db[:, None]
    pred_shape = pred_db - pred_level_db[:, None]

    resid_flat = resid.reshape(-1)
    r_rep = np.repeat(r_m, n_bands)
    amp_rep = np.repeat(true_tot, n_bands)

    return {
        "n_obs_events": int(true_db.shape[0]),
        "macro_rmse_db": float(_rmse(true_db.reshape(-1), pred_db.reshape(-1))),
        "macro_mae_db": float(_mae(true_db.reshape(-1), pred_db.reshape(-1))),
        "macro_r2": float(_r2(true_db.reshape(-1), pred_db.reshape(-1))),
        "per_band_rmse_db": [float(_rmse(true_db[:, f], pred_db[:, f])) for f in range(n_bands)],
        "total_rms": {
            "rmse_mms": float(_rmse(true_tot, pred_tot)),
            "pearson": float(_pearson(true_tot, pred_tot)),
            "spearman": float(_spearman(true_tot, pred_tot)),
            "r2": float(_r2(true_tot, pred_tot)),
        },
        "shape_only_rmse_db": float(_rmse(true_shape.reshape(-1), pred_shape.reshape(-1))),
        "residual_vs_distance_pearson": float(_pearson(resid_flat, r_rep)),
        "residual_vs_amplitude_pearson": float(_pearson(resid_flat, amp_rep)),
    }


def evaluate_mode_a(L, R, TRACK, SPLIT, fit, model, split_name, band_nominal) -> dict:
    sel = SPLIT == split_name
    L_s, R_s, TRACK_s = L[sel], R[sel], TRACK[sel]
    corr = compute_corr(R_s, TRACK_s, fit, model)
    C = np.mean(L_s - corr, axis=1)
    L_hat = C[:, None, :] + corr
    overall = compute_metrics(L_s.reshape(-1, L_s.shape[2]), L_hat.reshape(-1, L_hat.shape[2]),
                               R_s.reshape(-1), band_nominal)
    per_sensor = {
        s: compute_metrics(L_s[:, j, :], L_hat[:, j, :], R_s[:, j], band_nominal)
        for j, s in enumerate(SENSORS)
    }
    return {"overall": overall, "per_sensor": per_sensor, "n_events": int(sel.sum()), "C": C}


def evaluate_mode_b(L, R, TRACK, SPLIT, fit, model, split_name, band_nominal) -> dict:
    sel = SPLIT == split_name
    L_s, R_s, TRACK_s = L[sel], R[sel], TRACK[sel]
    corr = compute_corr(R_s, TRACK_s, fit, model)
    per_sensor = {}
    all_true, all_pred, all_r = [], [], []
    for k, held in enumerate(SENSORS):
        mask = np.ones(len(SENSORS), dtype=bool)
        mask[k] = False
        C_loo = np.mean(L_s[:, mask, :] - corr[:, mask, :], axis=1)
        L_hat_held = C_loo + corr[:, k, :]
        per_sensor[held] = compute_metrics(L_s[:, k, :], L_hat_held, R_s[:, k], band_nominal)
        all_true.append(L_s[:, k, :])
        all_pred.append(L_hat_held)
        all_r.append(R_s[:, k])
    overall = compute_metrics(np.concatenate(all_true, axis=0), np.concatenate(all_pred, axis=0),
                               np.concatenate(all_r), band_nominal)
    return {"overall": overall, "per_sensor": per_sensor, "n_events": int(sel.sum())}


def run_full_eval(L, R, TRACK, SPLIT, fit, model, band_nominal) -> dict:
    out = {"modeA": {}, "modeB": {}}
    for split_name in ("train", "val", "test"):
        out["modeA"][split_name] = evaluate_mode_a(L, R, TRACK, SPLIT, fit, model, split_name, band_nominal)
        out["modeB"][split_name] = evaluate_mode_b(L, R, TRACK, SPLIT, fit, model, split_name, band_nominal)
    return out


# ── Mode-B event-level export (read-only re-derivation; same math as evaluate_mode_b) ────────

def build_mode_b_predictions_df(L, R, TRACK, SPLIT, events, fit, model, band_nominal) -> pd.DataFrame:
    """Long-format event/sensor/band Mode-B predictions, all splits, for offline inspection."""
    corr = compute_corr(R, TRACK, fit, model)
    n_events, n_sensors, n_bands = L.shape
    frames = []
    for k, held in enumerate(SENSORS):
        mask = np.ones(n_sensors, dtype=bool)
        mask[k] = False
        C_loo = np.mean(L[:, mask, :] - corr[:, mask, :], axis=1)
        pred_db = C_loo + corr[:, k, :]
        true_db = L[:, k, :]
        frames.append(pd.DataFrame({
            "event_id": np.repeat(events, n_bands),
            "split": np.repeat(SPLIT, n_bands),
            "held_sensor": held,
            "track_number": np.repeat(TRACK, n_bands),
            "distance_m": np.repeat(R[:, k], n_bands),
            "band_nominal_hz": np.tile(band_nominal, n_events),
            "target_db": true_db.reshape(-1),
            "predicted_db": pred_db.reshape(-1),
        }))
    return pd.concat(frames, ignore_index=True)


def mode_b_total_rms_per_sensor(L, R, TRACK, SPLIT, fit, model, split_name) -> dict:
    """Per-sensor (true_tot, pred_tot) mm/s arrays for Mode-B on one split, for scatter plotting only."""
    sel = SPLIT == split_name
    L_s, R_s, TRACK_s = L[sel], R[sel], TRACK[sel]
    corr = compute_corr(R_s, TRACK_s, fit, model)
    out = {}
    for k, held in enumerate(SENSORS):
        mask = np.ones(len(SENSORS), dtype=bool)
        mask[k] = False
        C_loo = np.mean(L_s[:, mask, :] - corr[:, mask, :], axis=1)
        pred_db = C_loo + corr[:, k, :]
        out[held] = (total_rms_from_db(L_s[:, k, :]), total_rms_from_db(pred_db))
    return out


# ── Bootstrap stability ───────────────────────────────────────────────────────

def bootstrap_stability(L, R, TRACK, SPLIT, model, n_boot, als_iters, smooth_lambda_rel, n_bands, seed=0):
    rng = np.random.default_rng(seed)
    penalty = _penalty_matrix(n_bands)
    train_idx = {t: np.where((SPLIT == "train") & (TRACK == t))[0] for t in (1, 2)}
    boot_n = {1: [], 2: []}
    boot_a = {1: [], 2: []}
    for _ in range(n_boot):
        for t in (1, 2):
            resample = rng.choice(train_idx[t], size=len(train_idx[t]), replace=True)
            n_f, a_f, _, _ = _fit_single_track(L[resample], R[resample], model, als_iters, smooth_lambda_rel, penalty)
            boot_n[t].append(n_f)
            if model == "O1":
                boot_a[t].append(a_f)
    result = {}
    for t in (1, 2):
        result[t] = {"n_track_boot_std": np.std(np.stack(boot_n[t]), axis=0).tolist()}
        if model == "O1":
            result[t]["alpha_track_boot_std"] = np.std(np.stack(boot_a[t]), axis=0).tolist()
    return result


# ── Shape/level identity check ───────────────────────────────────────────────

def verify_shape_normalization(C: np.ndarray) -> dict:
    T = 10.0 * np.log10(np.sum(10.0 ** (C / 10.0), axis=1))
    S = C - T[:, None]
    check = np.sum(10.0 ** (S / 10.0), axis=1)
    max_dev = float(np.max(np.abs(check - 1.0)))
    return {"max_abs_deviation_from_1": max_dev, "passed": bool(max_dev < 1e-8)}


# ── Comparison / recommendation ──────────────────────────────────────────────

def compare_o0_o1(eval_o0, eval_o1) -> dict:
    comparison = {}
    for mode in ("modeA", "modeB"):
        comparison[mode] = {}
        for split_name in ("train", "val", "test"):
            r0 = eval_o0[mode][split_name]["overall"]["macro_rmse_db"]
            r1 = eval_o1[mode][split_name]["overall"]["macro_rmse_db"]
            comparison[mode][split_name] = {
                "o0_macro_rmse_db": r0,
                "o1_macro_rmse_db": r1,
                "o1_improvement_db": r0 - r1,
                "o1_improvement_rel_pct": (100.0 * (r0 - r1) / r0) if r0 else float("nan"),
            }
    return comparison


def recommend_decoder(fit_o1, comparison) -> dict:
    test_cmp = comparison["modeB"]["test"]
    improvement_pct = test_cmp["o1_improvement_rel_pct"]
    cond_numbers = {t: fit_o1[t]["condition_number"] for t in (1, 2)}
    corr_n_alpha = {t: fit_o1[t]["n_alpha_correlation"] for t in (1, 2)}
    max_cond = max(cond_numbers.values())
    reasons = []
    recommend_o1 = True
    if improvement_pct < 2.0:
        recommend_o1 = False
        reasons.append(f"O1 leave-one-out test macro-RMSE improvement only {improvement_pct:.2f}% (<2% threshold)")
    if max_cond > 30:
        recommend_o1 = False
        reasons.append(f"O1 design-matrix condition number {max_cond:.1f} (>30) -> weak n/alpha identifiability")
    if any(abs(c) > 0.8 for c in corr_n_alpha.values() if c == c):
        recommend_o1 = False
        reasons.append(f"fitted n_track/alpha_track correlated across bands {corr_n_alpha} -> not separately identifiable")
    decision = "O1" if recommend_o1 else "O0"
    return {
        "decision": decision,
        "improvement_pct_modeB_test": improvement_pct,
        "condition_number": cond_numbers,
        "n_alpha_correlation": corr_n_alpha,
        "reasons": reasons or ["O1 met all improvement/identifiability thresholds"],
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_fig(fig, path: Path) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_propagation_curves(band_nominal, fit_o0, fit_o1, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for track, lbl in ((1, "Track 1"), (2, "Track 2")):
        axes[0].plot(band_nominal, fit_o0[track]["n_track"], "--", label=f"{lbl} O0")
        axes[0].plot(band_nominal, fit_o1[track]["n_track"], "-", label=f"{lbl} O1")
        axes[1].plot(band_nominal, fit_o1[track]["alpha_track"], "-", label=f"{lbl} O1")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("n_track[f]")
    axes[0].set_title("Geometric spreading exponent"); axes[0].legend(fontsize=8)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("alpha_track_db[f]  [dB/m]")
    axes[1].set_title("Attenuation coefficient (O1 only)"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir / "propagation_curves.png")


def plot_total_rms_scatter(rms_o0: dict, rms_o1: dict, out_dir: Path, split_name="test") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SENSORS)))
    for ax, rms_data, name in zip(axes, (rms_o0, rms_o1), ("O0", "O1")):
        all_true, all_pred = [], []
        for j, s in enumerate(SENSORS):
            true_tot, pred_tot = rms_data[s]
            ax.scatter(true_tot, pred_tot, s=14, alpha=0.6, color=colors[j], label=s, edgecolors="none")
            all_true.append(true_tot); all_pred.append(pred_tot)
        all_true = np.concatenate(all_true); all_pred = np.concatenate(all_pred)
        lo = float(min(all_true.min(), all_pred.min()))
        hi = float(max(all_true.max(), all_pred.max()))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, "k--", lw=1, label="1:1")
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
        r2 = _r2(all_true, all_pred); rmse = _rmse(all_true, all_pred)
        ax.set_title(f"{name} mode-B {split_name}  R2={r2:.3f}  RMSE={rmse:.4f} mm/s")
        ax.set_xlabel("measured total RMS (mm/s)"); ax.set_ylabel("predicted total RMS (mm/s)")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
    fig.tight_layout()
    _save_fig(fig, out_dir / "total_rms_summary.png")


def plot_bootstrap_stability(band_nominal, boot_o1, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for track, lbl in ((1, "Track 1"), (2, "Track 2")):
        axes[0].plot(band_nominal, boot_o1[track]["n_track_boot_std"], label=lbl)
        axes[1].plot(band_nominal, boot_o1[track]["alpha_track_boot_std"], label=lbl)
    axes[0].set_xscale("log"); axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("bootstrap std(n_track[f])")
    axes[0].legend(fontsize=8); axes[0].set_title("O1 event-bootstrap stability: n")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("bootstrap std(alpha_track_db[f])")
    axes[1].legend(fontsize=8); axes[1].set_title("O1 event-bootstrap stability: alpha")
    fig.tight_layout()
    _save_fig(fig, out_dir / "bootstrap_stability.png")


# ── Serialization helper ─────────────────────────────────────────────────────

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {(str(k)): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="fast code/smoke check on a small event subsample")
    p.add_argument("--n-bootstrap", type=int, default=None)
    p.add_argument("--als-iters", type=int, default=None)
    return p.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    smoke = args.smoke
    n_iter = args.als_iters or (N_ALS_ITER_SMOKE if smoke else N_ALS_ITER_FULL)
    n_boot = args.n_bootstrap if args.n_bootstrap is not None else (N_BOOTSTRAP_SMOKE if smoke else N_BOOTSTRAP_FULL)

    section("Line-C target-only spectral oracle diagnostic (O0 vs O1)")
    print(f"smoke={smoke}  als_iters={n_iter}  n_bootstrap={n_boot}  r0={R0}  "
          f"smooth_lambda_rel={SMOOTH_LAMBDA_REL}  sensors={SENSORS}")

    data = load_tensors()
    L, R, TRACK, SPLIT = data["L"], data["R"], data["TRACK"], data["SPLIT"]
    events, band_nominal = data["events"], data["band_nominal"]
    n_bands = L.shape[2]

    if smoke:
        rng = np.random.default_rng(0)
        n_take = min(N_EVENTS_SMOKE, len(events))
        idx = np.sort(rng.choice(len(events), size=n_take, replace=False))
        L, R, TRACK, SPLIT, events = L[idx], R[idx], TRACK[idx], SPLIT[idx], events[idx]
        for split_name in ("train", "val", "test"):
            for t in (1, 2):
                n = int(np.sum((SPLIT == split_name) & (TRACK == t)))
                if n < 5:
                    raise RuntimeError(
                        f"smoke subsample too small: split={split_name} track={t} n={n} (<5); "
                        f"increase N_EVENTS_SMOKE"
                    )
        print(f"[smoke] subsampled to {len(events)} events")

    section("Fitting O0 (n only) and O1 (n + alpha) on TRAIN events")
    fit_o0 = fit_propagation(L, R, TRACK, SPLIT, "O0", n_iter, SMOOTH_LAMBDA_REL)
    fit_o1 = fit_propagation(L, R, TRACK, SPLIT, "O1", n_iter, SMOOTH_LAMBDA_REL)
    for t in (1, 2):
        print(f"  track {t}: O0 n_train_events={fit_o0[t]['n_train_events']} "
              f"iters={fit_o0[t]['n_iterations']} n_track_mean={np.mean(fit_o0[t]['n_track']):.4f}")
        print(f"  track {t}: O1 n_train_events={fit_o1[t]['n_train_events']} "
              f"iters={fit_o1[t]['n_iterations']} n_track_mean={np.mean(fit_o1[t]['n_track']):.4f} "
              f"alpha_track_mean={np.mean(fit_o1[t]['alpha_track']):.5f} "
              f"cond={fit_o1[t]['condition_number']:.2f} corr(n,alpha)={fit_o1[t]['n_alpha_correlation']:.3f}")

    section("Evaluating (mode A: five-sensor reconstruction / mode B: leave-one-sensor-out)")
    eval_o0 = run_full_eval(L, R, TRACK, SPLIT, fit_o0, "O0", band_nominal)
    eval_o1 = run_full_eval(L, R, TRACK, SPLIT, fit_o1, "O1", band_nominal)
    C_train_o0 = eval_o0["modeA"]["train"].pop("C")
    C_train_o1 = eval_o1["modeA"]["train"].pop("C")
    for split_name in ("train", "val", "test"):
        for mode in ("modeA", "modeB"):
            m0 = eval_o0[mode][split_name]["overall"]["macro_rmse_db"]
            m1 = eval_o1[mode][split_name]["overall"]["macro_rmse_db"]
            print(f"  {mode} {split_name:5s}: O0 macro_rmse={m0:.3f} dB   O1 macro_rmse={m1:.3f} dB")

    section("Verifying C[e,f] -> T[e], S[e,f] normalization identity")
    check_o0 = verify_shape_normalization(C_train_o0)
    check_o1 = verify_shape_normalization(C_train_o1)
    print(f"  O0: max|sum_f 10^(S/10) - 1| = {check_o0['max_abs_deviation_from_1']:.3e}  passed={check_o0['passed']}")
    print(f"  O1: max|sum_f 10^(S/10) - 1| = {check_o1['max_abs_deviation_from_1']:.3e}  passed={check_o1['passed']}")

    section(f"Event-bootstrap stability of n_track/alpha_track (B={n_boot})")
    boot_o0 = bootstrap_stability(L, R, TRACK, SPLIT, "O0", n_boot, n_iter, SMOOTH_LAMBDA_REL, n_bands)
    boot_o1 = bootstrap_stability(L, R, TRACK, SPLIT, "O1", n_boot, n_iter, SMOOTH_LAMBDA_REL, n_bands)

    comparison = compare_o0_o1(eval_o0, eval_o1)
    recommendation = recommend_decoder(fit_o1, comparison)

    section("Recommendation")
    print(f"  decision: {recommendation['decision']}")
    for r in recommendation["reasons"]:
        print(f"  - {r}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / (("smoke_" if smoke else "run_") + ts)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "sensors": SENSORS, "r0_m": R0, "smooth_lambda_rel": SMOOTH_LAMBDA_REL,
            "als_iters": n_iter, "n_bootstrap": n_boot, "smoke": smoke,
            "n_events_used": int(len(events)), "n_bands": int(n_bands),
            "band_nominal_hz": band_nominal.tolist(),
            "created": datetime.now().isoformat(),
        },
        "shape_normalization_check": {"O0": check_o0, "O1": check_o1},
        "O0": {
            "fit": {t: {k: v for k, v in fit_o0[t].items()} for t in (1, 2)},
            "bootstrap": boot_o0,
            "eval": eval_o0,
        },
        "O1": {
            "fit": {t: {k: v for k, v in fit_o1[t].items()} for t in (1, 2)},
            "bootstrap": boot_o1,
            "eval": eval_o1,
        },
        "comparison": comparison,
        "recommendation": recommendation,
    }
    (run_dir / "results.json").write_text(json.dumps(_to_jsonable(results), indent=2))

    band_df = pd.DataFrame({
        "band_nominal_hz": band_nominal,
        "n_track1_O0": fit_o0[1]["n_track"], "n_track2_O0": fit_o0[2]["n_track"],
        "n_track1_O1": fit_o1[1]["n_track"], "n_track2_O1": fit_o1[2]["n_track"],
        "alpha_track1_O1": fit_o1[1]["alpha_track"], "alpha_track2_O1": fit_o1[2]["alpha_track"],
    })
    band_df.to_csv(run_dir / "propagation_curves.csv", index=False)

    train_events = events[SPLIT == "train"]
    for model_name, C_train in (("O0", C_train_o0), ("O1", C_train_o1)):
        df_C = pd.DataFrame(C_train, columns=[f"band_{i}_{hz:g}Hz" for i, hz in enumerate(band_nominal)])
        df_C.insert(0, "event_id", train_events)
        df_C.to_parquet(run_dir / f"event_source_spectrum_C_train_{model_name}.parquet", index=False)

    for model_name, fit in (("O0", fit_o0), ("O1", fit_o1)):
        df_modeb = build_mode_b_predictions_df(L, R, TRACK, SPLIT, events, fit, model_name, band_nominal)
        df_modeb.to_parquet(run_dir / f"mode_b_predictions_{model_name}.parquet", index=False)

    rms_o0 = mode_b_total_rms_per_sensor(L, R, TRACK, SPLIT, fit_o0, "O0", "test")
    rms_o1 = mode_b_total_rms_per_sensor(L, R, TRACK, SPLIT, fit_o1, "O1", "test")

    plot_propagation_curves(band_nominal, fit_o0, fit_o1, plots_dir)
    plot_total_rms_scatter(rms_o0, rms_o1, plots_dir)
    plot_bootstrap_stability(band_nominal, boot_o1, plots_dir)

    manifest = {
        "script": "analyse_linec_spectral_oracle_v1.py",
        "input_spec_parquet": str(SPEC_PATH),
        "geometry_source": "sites/holten.json via src.utils.geometry_utils.apply_corrected_distances",
        "args": vars(args),
        "elapsed_s": round(time.time() - t0, 2),
        "run_dir": str(run_dir),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    section("Done")
    print(f"  output directory: {run_dir}")
    print(f"  elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
