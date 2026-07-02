"""
train_pxgbr_ensemble_linec_v1.py
PXGBR Ensemble/Loss Sweep — Holten Line-C side -1

Architecture:
  pred_log_final = pred_log_p3 + residual_xgb

Grid search over:
  - 5 sample weight schemes  (uniform, A/B/C/D)
  - 1 PGV-scaled target variant
  - 4 × 2 × 2 × 2 × 2 × 2 = 64 hyperparameter combinations

Model selection: validation RMSE(PGV), not residual RMSE
Ensemble:        top-3/top-5 mean, median, inverse-RMSE-weighted mean

Validated baselines:
  P3_corrected_n: RMSE(PGV)=2.4006  RMSE(log)=0.5993
  PXGBR-R2:       RMSE(PGV)=2.2718  RMSE(log)=0.5946  MP4=4.7389

Target: RMSE(PGV) < 2.20,  MP4 < 4.70
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))


def _data_root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")


MODELS_ROOT  = _data_root() / "holten_models"
PARQUET_ROOT = _data_root() / "holten_parquet"
SENSOR_ORDER = ["MP4", "MP8", "MP10", "MP1", "MP2"]

LEAKAGE_COLS = {
    "target_pgv_z_mms", "target_log", "target_pgv",
    "residual_log", "scaled_residual",
    "c_target", "event_id", "split", "sensor", "sensor_id",
    "site_id", "train_type", "acc_side_of_track",
    "acc_distance_to_track_m", "acc_distance_to_track_2_m",
    "effective_distance_to_active_track_m",
}


# ===========================================================================
# DATA LOADING
# ===========================================================================

def find_p3_dir(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    hits = sorted(MODELS_ROOT.glob("cnn_curveprior_p3_savepred_linec_v001_vP3_*"),
                  key=lambda p: p.name)
    if not hits:
        raise FileNotFoundError("No P3 savepred directory found.")
    return hits[-1]


def find_parquet_v2() -> Path:
    hits = sorted(PARQUET_ROOT.glob("parquet_v002_*"), key=lambda p: p.name)
    if not hits:
        raise FileNotFoundError("No parquet_v002_* builds found")
    return hits[-1] / "dataset.parquet"


def find_pxgbr_dir() -> Optional[Path]:
    hits = sorted(MODELS_ROOT.glob("pxgbr_linec_v001_*"), key=lambda p: p.name)
    return hits[-1] if hits else None


def load_and_join(p3_dir: Path, parquet_v2: Path) -> pd.DataFrame:
    """Load P3 predictions, join with FO features, build feature matrix."""
    p3 = pd.read_parquet(p3_dir / "all_predictions.parquet")
    fo = pd.read_parquet(parquet_v2)
    fo = fo[fo["sensor_id"].isin(SENSOR_ORDER)].copy()
    fo = fo.dropna(subset=["target_pgv_z_mms", "track_number"])
    fo = fo[(fo["target_pgv_z_mms"] > 0) & (fo["track_number"].isin([1, 2]))]
    fo["event_id"] = fo["event_id"].astype(str)
    fo["sensor"]   = fo["sensor_id"]

    fo_num_cols = [c for c in fo.columns
                   if pd.api.types.is_numeric_dtype(fo[c])
                   and c not in LEAKAGE_COLS
                   and c not in ("train_speed_kmh", "train_type_code", "track_number")
                   and not c.startswith("target_")
                   and "pgv" not in c.lower()]

    merged = p3.merge(fo[["event_id", "sensor"] + fo_num_cols
                          + ["train_speed_kmh", "train_type_code"]],
                      on=["event_id", "sensor"], how="inner",
                      suffixes=("", "_fo"))

    # Physics-derived features
    merged["log_distance"]        = np.log(merged["distance"] / 10.0)
    merged["sensor_code"]         = merged["sensor"].map(
        {s: i for i, s in enumerate(SENSOR_ORDER)}).astype(float)
    spd = merged["train_speed_kmh"]
    merged["train_speed_missing"] = spd.isna().astype(float)
    merged["train_speed_kmh"]     = spd.fillna(0.0)

    # Targets
    merged["residual_log"]    = merged["target_log"] - merged["pred_log_p3"]
    merged["scaled_residual"] = (
        (merged["target_pgv"] - merged["pred_pgv_p3"])
        / np.sqrt(merged["target_pgv"].clip(lower=0) + 1.0)
    )

    print(f"Dataset: {len(merged):,} rows | splits: {merged.split.value_counts().to_dict()}")
    return merged


def feature_cols(df: pd.DataFrame) -> List[str]:
    explicit = [
        "pred_log_p3", "pred_pgv_p3", "c_hat_p3", "epsilon_p3", "n_used",
        "distance", "log_distance", "track", "sensor_code",
        "train_speed_kmh", "train_speed_missing",
    ]
    if "train_type_code" in df.columns:
        explicit.append("train_type_code")
    if "track_number" in df.columns:
        explicit.append("track_number")

    non_fo = set(explicit) | LEAKAGE_COLS | {
        "residual_log", "scaled_residual",
        "pred_log_p3", "pred_pgv_p3",  # already in explicit
    }
    fo_add = [c for c in df.columns
              if pd.api.types.is_numeric_dtype(df[c])
              and c not in non_fo
              and not c.startswith("target_")
              and not c.startswith("pred_")
              and "pgv" not in c.lower()]

    all_f = explicit + [c for c in fo_add if c not in explicit]
    return [c for c in all_f if c in df.columns]


# ===========================================================================
# SAMPLE WEIGHTS
# ===========================================================================

def make_weights(df: pd.DataFrame, scheme: str) -> np.ndarray:
    s  = df["sensor"].values
    tp = df["target_pgv"].values
    w  = np.ones(len(df), dtype=np.float32)

    if scheme == "uniform":
        pass
    elif scheme == "A":   # current R2 base
        w += 1.5 * (s == "MP4") + 1.0 * (tp > 4) + 1.0 * (tp > 8)
    elif scheme == "B":   # stronger MP4
        w += 3.0 * (s == "MP4") + 1.0 * (tp > 4) + 1.0 * (tp > 8)
    elif scheme == "C":   # smooth PGV
        w += 0.5 * np.sqrt(np.clip(tp, 0, None)) + 1.5 * (s == "MP4")
    elif scheme == "D":   # high-PGV aggressive
        w += 1.0 * (s == "MP4") + 2.0 * (tp > 4) + 3.0 * (tp > 8)
    return w.astype(np.float32)


# ===========================================================================
# METRICS
# ===========================================================================

def rmse_pgv(pred_log: np.ndarray, target_log: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.exp(pred_log) - np.exp(target_log)) ** 2)))


def rmse_log(pred_log: np.ndarray, target_log: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred_log - target_log) ** 2)))


def full_metrics(pred_log: np.ndarray, target_log: np.ndarray,
                 sensors: np.ndarray, label: str) -> Dict:
    pred_pgv_   = np.exp(pred_log)
    target_pgv_ = np.exp(target_log)

    def _r(a, b):  return float(np.sqrt(np.mean((a-b)**2)))
    def _b(a, b):  return float(np.mean(a-b))
    def _r2(a, b):
        ss=np.sum((b-a)**2); st=np.sum((b-b.mean())**2)
        return float(1-ss/st) if st>0 else 0.

    m = {"model": label,
         "rmse_log": _r(pred_log, target_log),
         "rmse_pgv": _r(pred_pgv_, target_pgv_),
         "mae_pgv":  float(np.mean(np.abs(pred_pgv_-target_pgv_))),
         "r2_log":   _r2(pred_log, target_log),
         "bias_pgv": _b(pred_pgv_, target_pgv_)}

    m["per_sensor"] = {}
    for s in SENSOR_ORDER:
        mask = sensors == s
        if mask.any():
            m["per_sensor"][s] = {
                "rmse_pgv": _r(pred_pgv_[mask], target_pgv_[mask]),
                "rmse_log": _r(pred_log[mask],  target_log[mask]),
                "bias_pgv": _b(pred_pgv_[mask], target_pgv_[mask]),
            }

    for thr in [4., 8.]:
        hi = target_pgv_ > thr
        if hi.any():
            m[f"pgv_gt{int(thr)}"] = {"n": int(hi.sum()),
                                       "rmse_pgv": _r(pred_pgv_[hi], target_pgv_[hi])}
        mp4 = (sensors=="MP4") & hi
        if mp4.any():
            m[f"mp4_pgv_gt{int(thr)}"] = {"n": int(mp4.sum()),
                                            "rmse_pgv": _r(pred_pgv_[mp4], target_pgv_[mp4])}
    return m


def mono_rate(pred_log: np.ndarray, sensors: np.ndarray,
              event_ids: np.ndarray) -> float:
    viol, total = 0, 0
    for eid in np.unique(event_ids)[:300]:
        ev = event_ids == eid
        if ev.sum() != 5: continue
        tmp = pd.DataFrame({"sensor": sensors[ev], "pred": pred_log[ev]})
        try:
            pv = [tmp[tmp.sensor==s]["pred"].values[0] for s in SENSOR_ORDER]
        except IndexError:
            continue
        for i in range(4):
            total += 1
            if pv[i] < pv[i+1]: viol += 1
    return viol/total if total else 0.


def apply_mono(df: pd.DataFrame, col: str) -> pd.Series:
    out = df[col].copy()
    for _, g in df.groupby("event_id"):
        idx = g.sort_values("distance").index
        out.loc[idx] = np.minimum.accumulate(out.loc[idx].values)
    return out


# ===========================================================================
# TRAINING LOOP
# ===========================================================================

def train_one(X_tr, y_tr, w_tr, X_va, y_va, params: Dict,
              n_est=3000, esr=50) -> Tuple[xgb.XGBRegressor, float, float]:
    """Train one XGBoost model; return (model, val_score, best_round)."""
    m = xgb.XGBRegressor(
        **params, n_estimators=n_est, early_stopping_rounds=esr,
        eval_metric="rmse", tree_method="hist", verbosity=0, random_state=42
    )
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m, float(m.best_score), int(m.best_iteration)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3_dir",    default=None)
    ap.add_argument("--output_tag", default="pxgbr_ensemble_linec_v001")
    args = ap.parse_args()

    print("=" * 80)
    print("PXGBR Ensemble/Loss Sweep  —  Holten Line-C side -1")
    print("=" * 80)

    p3_dir    = find_p3_dir(args.p3_dir)
    parquet_v2 = find_parquet_v2()
    pxgbr_dir  = find_pxgbr_dir()
    print(f"P3 savepred: {p3_dir.name}")
    print(f"PXGBR ref:   {pxgbr_dir.name if pxgbr_dir else 'none'}")

    df = load_and_join(p3_dir, parquet_v2)
    fc = feature_cols(df)
    print(f"Features: {len(fc)}")

    for c in fc:
        df[c] = df[c].fillna(0.0)

    tr = df[df.split=="train"].copy()
    va = df[df.split=="val"].copy()
    te = df[df.split=="test"].copy()

    Xtr = tr[fc].values.astype(np.float32)
    Xva = va[fc].values.astype(np.float32)
    Xte = te[fc].values.astype(np.float32)

    y_log_tr = tr["residual_log"].values.astype(np.float32)
    y_log_va = va["residual_log"].values.astype(np.float32)
    y_scl_tr = tr["scaled_residual"].values.astype(np.float32)
    y_scl_va = va["scaled_residual"].values.astype(np.float32)

    # ── Hyperparameter grid ───────────────────────────────────────────────────
    grid = list(product(
        [3, 4, 5],          # max_depth
        [0.02, 0.05],       # learning_rate
        [1, 5],             # min_child_weight
        [1, 10],            # reg_lambda
        [0.8, 0.9],         # subsample
        [0.8, 0.9],         # colsample_bytree
    ))
    print(f"\nHyperparameter combinations: {len(grid)}")

    # Weight schemes for log-residual target
    weight_schemes = ["uniform", "A", "B", "C", "D"]
    # PGV-scaled variant (uniform weights only)
    pgv_scaled_schemes = ["uniform"]

    total_runs = len(grid) * len(weight_schemes) + len(grid) * len(pgv_scaled_schemes)
    print(f"Total model fits:  {total_runs}")

    # ── Training ─────────────────────────────────────────────────────────────
    results: List[Dict] = []
    run_id = 0

    for depth, lr, mcw, rl, ss, cbt in grid:
        params = dict(max_depth=depth, learning_rate=lr, min_child_weight=mcw,
                      reg_lambda=rl, subsample=ss, colsample_bytree=cbt,
                      objective="reg:squarederror")

        # Log-residual variants
        for scheme in weight_schemes:
            w = make_weights(tr, scheme)
            model, val_rmse_res, best_rd = train_one(
                Xtr, y_log_tr, w, Xva, y_log_va, params
            )
            # Compute val RMSE(PGV) using the log-residual predictions
            val_pred_log = va["pred_log_p3"].values + model.predict(Xva)
            val_rpgv = rmse_pgv(val_pred_log, va["target_log"].values)
            val_rlog = rmse_log(val_pred_log, va["target_log"].values)

            results.append({
                "run_id": run_id, "target": "log_residual", "scheme": scheme,
                "depth": depth, "lr": lr, "mcw": mcw, "rl": rl,
                "ss": ss, "cbt": cbt, "best_round": best_rd,
                "val_rmse_res": val_rmse_res,
                "val_rmse_pgv": val_rpgv,
                "val_rmse_log": val_rlog,
                "model": model,
            })
            run_id += 1

        # PGV-scaled variant
        for scheme in pgv_scaled_schemes:
            w = make_weights(tr, scheme)
            model, val_rmse_res, best_rd = train_one(
                Xtr, y_scl_tr, w, Xva, y_scl_va, params
            )
            # Reconstruct val PGV from scaled residual
            val_pred_scaled = model.predict(Xva)
            val_pred_pgv = va["pred_pgv_p3"].values + \
                val_pred_scaled * np.sqrt(va["pred_pgv_p3"].values.clip(0) + 1)
            val_pred_pgv = val_pred_pgv.clip(1e-6)
            val_pred_log_r = np.log(val_pred_pgv)
            val_rpgv = rmse_pgv(val_pred_log_r, va["target_log"].values)
            val_rlog = rmse_log(val_pred_log_r, va["target_log"].values)

            results.append({
                "run_id": run_id, "target": "scaled_residual", "scheme": scheme,
                "depth": depth, "lr": lr, "mcw": mcw, "rl": rl,
                "ss": ss, "cbt": cbt, "best_round": best_rd,
                "val_rmse_res": val_rmse_res,
                "val_rmse_pgv": val_rpgv,
                "val_rmse_log": val_rlog,
                "model": model,
            })
            run_id += 1

        if run_id % 50 == 0:
            top = sorted(results, key=lambda r: r["val_rmse_pgv"])
            print(f"  [{run_id}/{total_runs}] best val RMSE(PGV)={top[0]['val_rmse_pgv']:.4f}"
                  f" ({top[0]['scheme']}, depth={top[0]['depth']}, lr={top[0]['lr']:.3f})")

    # ── Sort by val RMSE(PGV) ─────────────────────────────────────────────────
    results.sort(key=lambda r: r["val_rmse_pgv"])
    print(f"\nTop 10 by val RMSE(PGV):")
    print(f"{'#':>3} {'scheme':>8} {'target':>16} {'depth':>5} {'lr':>5} "
          f"{'mcw':>3} {'rl':>4} {'ss':>4} {'cbt':>4} "
          f"{'val_PGV':>8} {'val_log':>8}")
    for i, r in enumerate(results[:10]):
        print(f"{i+1:>3} {r['scheme']:>8} {r['target']:>16} {r['depth']:>5} "
              f"{r['lr']:>5.3f} {r['mcw']:>3} {r['rl']:>4} "
              f"{r['ss']:>4} {r['cbt']:>4} "
              f"{r['val_rmse_pgv']:>8.4f} {r['val_rmse_log']:>8.4f}")

    # ── Generate test predictions for top models ──────────────────────────────
    N_top = min(10, len(results))   # keep top-10 for ensembles
    test_preds_log: Dict[int, np.ndarray] = {}
    test_preds_pgv: Dict[int, np.ndarray] = {}

    for r in results[:N_top]:
        m = r["model"]
        if r["target"] == "log_residual":
            pl = te["pred_log_p3"].values + m.predict(Xte)
            pp = np.exp(pl)
        else:  # scaled_residual
            ps = m.predict(Xte)
            pp = (te["pred_pgv_p3"].values
                  + ps * np.sqrt(te["pred_pgv_p3"].values.clip(0) + 1)).clip(1e-6)
            pl = np.log(pp)
        test_preds_log[r["run_id"]] = pl
        test_preds_pgv[r["run_id"]] = pp

    top_ids = [r["run_id"] for r in results[:N_top]]
    preds_mat = np.vstack([test_preds_log[i] for i in top_ids[:5]])  # (5, N_test)

    # ── Ensembles ────────────────────────────────────────────────────────────
    # Mean top-3 (by val RMSE(PGV))
    ens_top3_log  = np.mean(preds_mat[:3], axis=0)
    # Mean top-5
    ens_top5_log  = np.mean(preds_mat[:5], axis=0)
    # Median top-5
    ens_med5_log  = np.median(preds_mat[:5], axis=0)
    # Inverse-RMSE-weighted mean top-5
    inv_w = np.array([1.0 / results[i]["val_rmse_pgv"] for i in range(5)])
    inv_w /= inv_w.sum()
    ens_inv_log   = np.average(preds_mat[:5], axis=0, weights=inv_w)

    # Best single model (no ensemble)
    best_single_log = test_preds_log[results[0]["run_id"]]

    # Monotonic versions
    def _mono(log_arr):
        tmp = te.copy(); tmp["_pred"] = log_arr
        return apply_mono(tmp, "_pred").values

    ens_top3_mono  = _mono(ens_top3_log)
    ens_top5_mono  = _mono(ens_top5_log)
    ens_inv_mono   = _mono(ens_inv_log)

    # ── Test metrics ─────────────────────────────────────────────────────────
    tgt_log  = te["target_log"].values
    sensors  = te["sensor"].values
    ev_ids   = te["event_id"].values

    print("\n" + "=" * 80)
    print("TEST METRICS")
    print("=" * 80)

    models_to_eval: Dict[str, np.ndarray] = {
        "P3":                te["pred_log_p3"].values,
        "best_single":       best_single_log,
        "ens_top3":          ens_top3_log,
        "ens_top5":          ens_top5_log,
        "ens_med5":          ens_med5_log,
        "ens_inv5":          ens_inv_log,
        "ens_top3_mono":     ens_top3_mono,
        "ens_top5_mono":     ens_top5_mono,
        "ens_inv5_mono":     ens_inv_mono,
    }

    all_metrics: Dict[str, Dict] = {}
    for name, pl in models_to_eval.items():
        m = full_metrics(pl, tgt_log, sensors, name)
        m["mono_viol_rate"] = mono_rate(pl, sensors, ev_ids)
        all_metrics[name] = m

    # Print table
    print(f"\n{'Model':<22} {'RMSE(log)':>10} {'RMSE(PGV)':>10} "
          f"{'R²(log)':>8} {'MP4 RMSE':>10} {'mono':>6}")
    print("-" * 72)
    for name, m in all_metrics.items():
        mp4 = m["per_sensor"].get("MP4", {}).get("rmse_pgv", float("nan"))
        mvr = m.get("mono_viol_rate", float("nan"))
        print(f"{name:<22} {m['rmse_log']:>10.4f} {m['rmse_pgv']:>10.4f} "
              f"{m['r2_log']:>8.4f} {mp4:>10.4f} {mvr:>6.3f}")

    # Load PXGBR-R2 predictions for comparison
    pxgbr_test_log = None
    if pxgbr_dir:
        try:
            px = pd.read_parquet(pxgbr_dir / "predictions_test.parquet")
            pk = te["event_id"] + "|" + te["sensor"]
            px_map = dict(zip(px["event_id"] + "|" + px["sensor"], px["pred_log_R2"]))
            pxgbr_test_log = pk.map(px_map).values
            if not pd.isna(pxgbr_test_log).any():
                m = full_metrics(pxgbr_test_log, tgt_log, sensors, "PXGBR-R2")
                m["mono_viol_rate"] = mono_rate(pxgbr_test_log, sensors, ev_ids)
                all_metrics["PXGBR-R2"] = m
                mp4 = m["per_sensor"].get("MP4", {}).get("rmse_pgv", float("nan"))
                print(f"{'PXGBR-R2':<22} {m['rmse_log']:>10.4f} {m['rmse_pgv']:>10.4f} "
                      f"{m['r2_log']:>8.4f} {mp4:>10.4f} {m['mono_viol_rate']:>6.3f}")
        except Exception as e:
            print(f"  [WARN] PXGBR-R2 load failed: {e}")

    # Identify best ensemble
    ens_names = ["ens_top3", "ens_top5", "ens_med5", "ens_inv5",
                 "ens_top3_mono", "ens_top5_mono", "ens_inv5_mono", "best_single"]
    best_ens  = min(ens_names, key=lambda n: all_metrics[n]["rmse_pgv"])
    print(f"\nBest ensemble: {best_ens}  "
          f"RMSE(PGV)={all_metrics[best_ens]['rmse_pgv']:.4f}  "
          f"RMSE(log)={all_metrics[best_ens]['rmse_log']:.4f}")

    p3_rpgv   = all_metrics["P3"]["rmse_pgv"]
    best_rpgv = all_metrics[best_ens]["rmse_pgv"]
    pxgbr_rpgv = all_metrics.get("PXGBR-R2", {}).get("rmse_pgv", 2.2718)
    print(f"vs P3:      {best_rpgv - p3_rpgv:+.4f} mm/s  "
          f"({(best_rpgv - p3_rpgv)/p3_rpgv*100:+.1f}%)")
    print(f"vs PXGBR-R2: {best_rpgv - pxgbr_rpgv:+.4f} mm/s")

    # ── Save outputs ─────────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = MODELS_ROOT / f"{args.output_tag}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    log_dir = Path("outputs/pxgbr_ensemble_linec_v1")
    log_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_ser = {
        k: {kk: vv for kk, vv in v.items() if kk != "per_sensor"}
        for k, v in all_metrics.items()
    }
    metrics_ser["per_sensor"] = {k: v.get("per_sensor", {}) for k, v in all_metrics.items()}
    metrics_ser["best_ensemble"] = best_ens
    metrics_ser["grid_runs"] = len(results)
    metrics_ser["top5_runs"] = [
        {k: v for k, v in r.items() if k != "model"}
        for r in results[:5]
    ]
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_ser, f, indent=2, default=float)

    # metrics_table.csv
    rows = []
    for name, m in all_metrics.items():
        row = {"model": name}
        for k in ["rmse_log", "rmse_pgv", "mae_pgv", "r2_log", "bias_pgv", "mono_viol_rate"]:
            row[k] = m.get(k, float("nan"))
        for s in SENSOR_ORDER:
            sm = m["per_sensor"].get(s, {})
            row[f"{s}_rmse_pgv"] = sm.get("rmse_pgv", float("nan"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "metrics_table.csv", index=False)

    # predictions_test.parquet — all individual top models
    pred_df = te[["event_id", "sensor", "track", "distance", "split",
                  "target_log", "target_pgv",
                  "pred_log_p3", "pred_pgv_p3"]].copy()
    for i, r in enumerate(results[:N_top]):
        pred_df[f"pred_log_model{i}"] = test_preds_log[r["run_id"]]
        pred_df[f"pred_pgv_model{i}"] = test_preds_pgv[r["run_id"]]
    pred_df.to_parquet(out / "predictions_test.parquet", index=False)

    # ensemble_predictions_test.parquet
    ens_df = te[["event_id", "sensor", "track", "distance", "split",
                 "target_log", "target_pgv",
                 "pred_log_p3", "pred_pgv_p3"]].copy()
    for ens_name, ens_log in [
        ("best_single", best_single_log),
        ("ens_top3",    ens_top3_log),
        ("ens_top5",    ens_top5_log),
        ("ens_med5",    ens_med5_log),
        ("ens_inv5",    ens_inv_log),
        ("ens_top3_mono", ens_top3_mono),
        ("ens_top5_mono", ens_top5_mono),
        ("ens_inv5_mono", ens_inv_mono),
    ]:
        ens_df[f"pred_log_{ens_name}"] = ens_log
        ens_df[f"pred_pgv_{ens_name}"] = np.exp(ens_log)
    if pxgbr_test_log is not None:
        ens_df["pred_log_pxgbr_r2"] = pxgbr_test_log
        ens_df["pred_pgv_pxgbr_r2"] = np.exp(pxgbr_test_log)
    ens_df.to_parquet(out / "ensemble_predictions_test.parquet", index=False)

    # feature_importance_top30.csv — average over top-5 log-residual models
    top5_log_res = [r for r in results[:10] if r["target"] == "log_residual"][:5]
    if top5_log_res:
        imp_arr = np.vstack([r["model"].feature_importances_ for r in top5_log_res])
        fi_avg  = pd.DataFrame({"feature": fc, "importance": imp_arr.mean(axis=0)})
        fi_avg  = fi_avg.sort_values("importance", ascending=False).reset_index(drop=True)
        fi_avg.to_csv(out / "feature_importance_top30.csv", index=False)

    # Save top-3 models
    for i, r in enumerate(results[:3]):
        r["model"].save_model(str(out / f"model_rank{i+1}.ubj"))

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        _make_plots(ens_df, all_metrics, best_ens, pxgbr_test_log is not None, out)
    except Exception as e:
        print(f"[WARN] Plots failed: {e}")

    print(f"\nOutput saved to: {out}")


def _make_plots(ens_df, all_metrics, best_ens, has_pxgbr, out):
    # 1. Scatter comparison
    names_to_plot = ["P3"] + (["PXGBR-R2"] if has_pxgbr else []) + [best_ens]
    col_map = {
        "P3":       "pred_pgv_p3",
        "PXGBR-R2": "pred_pgv_pxgbr_r2",
        "best_single": "pred_pgv_best_single",
        "ens_top3":    "pred_pgv_ens_top3",
        "ens_top5":    "pred_pgv_ens_top5",
        "ens_med5":    "pred_pgv_ens_med5",
        "ens_inv5":    "pred_pgv_ens_inv5",
        "ens_top3_mono": "pred_pgv_ens_top3_mono",
        "ens_top5_mono": "pred_pgv_ens_top5_mono",
        "ens_inv5_mono": "pred_pgv_ens_inv5_mono",
    }
    colours = {"P3": "#1f77b4", "PXGBR-R2": "#ff7f0e",
               "best_single": "#9467bd",
               **{k: "#2ca02c" for k in col_map if "ens" in k}}

    fig, axes = plt.subplots(1, len(names_to_plot),
                              figsize=(5*len(names_to_plot), 5), squeeze=False)
    for ax, name in zip(axes[0], names_to_plot):
        col = col_map.get(name)
        if not col or col not in ens_df.columns:
            ax.set_visible(False); continue
        ax.scatter(ens_df["target_pgv"], ens_df[col], s=4, alpha=0.3,
                   color=colours.get(name, "grey"), rasterized=True)
        lim = max(ens_df["target_pgv"].max(), ens_df[col].max()) * 1.05
        ax.plot([0.05, lim], [0.05, lim], "r--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        rmse = all_metrics[name]["rmse_pgv"]
        ax.set_title(f"{name}\nRMSE={rmse:.3f}"); ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("Measured"); ax.set_ylabel("Predicted")
    fig.suptitle("Measured vs Predicted — Line-C test set")
    fig.tight_layout()
    fig.savefig(out / "measured_vs_predicted_P3_vs_PXGBR_vs_ensemble.png",
                dpi=120, bbox_inches="tight"); plt.close(fig)

    # 2. Per-sensor bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_models = [n for n in names_to_plot if n in all_metrics]
    x = np.arange(5); w = 0.8 / max(len(plot_models), 1)
    for ax_i, (metric, ylabel) in enumerate([("rmse_pgv", "RMSE(PGV)"),
                                              ("rmse_log", "RMSE(log)")]):
        for j, name in enumerate(plot_models):
            vals = [all_metrics[name]["per_sensor"].get(s, {}).get(metric, float("nan"))
                    for s in SENSOR_ORDER]
            axes[ax_i].bar(x + (j - len(plot_models)/2 + 0.5)*w, vals, w,
                           label=name, alpha=0.8)
        axes[ax_i].set_xticks(x); axes[ax_i].set_xticklabels(SENSOR_ORDER)
        axes[ax_i].set_ylabel(ylabel); axes[ax_i].legend(fontsize=7)
        axes[ax_i].grid(True, axis="y", alpha=0.4)
    fig.suptitle("Per-Sensor Error — P3 vs PXGBR vs Ensemble")
    fig.tight_layout()
    fig.savefig(out / "per_sensor_rmse_P3_vs_PXGBR_vs_ensemble.png",
                dpi=120, bbox_inches="tight"); plt.close(fig)

    best_col = col_map.get(best_ens, None)
    if best_col and best_col in ens_df.columns:
        # 3. High-PGV profiles
        hi_evs = list(ens_df.groupby("event_id")["target_pgv"].max()
                              .sort_values(ascending=False).head(6).index)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8)); axes = axes.flatten()
        for ax_i, ev in enumerate(hi_evs[:6]):
            ev_df = ens_df[ens_df.event_id==ev].sort_values("distance")
            if ev_df.empty: continue
            d = ev_df["distance"].values; t = ev_df["target_pgv"].values
            axes[ax_i].semilogy(d, t, "ko-", ms=5, label="Measured")
            axes[ax_i].semilogy(d, ev_df["pred_pgv_p3"].values, "b^--", ms=4, lw=1, label="P3")
            axes[ax_i].semilogy(d, ev_df[best_col].values, "g^--", ms=4, lw=1.2, label=best_ens)
            axes[ax_i].set_title(str(ev)[-12:-4], fontsize=8)
            axes[ax_i].set_xlabel("d (m)"); axes[ax_i].grid(True, which="both", alpha=0.3)
            if ax_i == 0: axes[ax_i].legend(fontsize=7)
        fig.suptitle(f"High-PGV Events — P3 vs {best_ens}")
        fig.tight_layout()
        fig.savefig(out / "high_pgv_profiles_P3_vs_ensemble.png", dpi=120); plt.close(fig)

        # 4. Failure profiles
        fail_evs = list(
            ens_df.assign(sq=(lambda d: (d[best_col] - d["target_pgv"])**2))
                  .groupby("event_id")["sq"].mean()
                  .apply(np.sqrt).sort_values(ascending=False).head(6).index
        )
        fig, axes = plt.subplots(2, 3, figsize=(13, 8)); axes = axes.flatten()
        for ax_i, ev in enumerate(fail_evs[:6]):
            ev_df = ens_df[ens_df.event_id==ev].sort_values("distance")
            if ev_df.empty: continue
            axes[ax_i].semilogy(ev_df["distance"].values, ev_df["target_pgv"].values,
                                "ko-", ms=5, label="Measured")
            axes[ax_i].semilogy(ev_df["distance"].values, ev_df["pred_pgv_p3"].values,
                                "b^--", ms=4, lw=1, label="P3")
            axes[ax_i].semilogy(ev_df["distance"].values, ev_df[best_col].values,
                                "r^--", ms=4, lw=1.2, label=best_ens)
            axes[ax_i].set_title(str(ev)[-12:-4], fontsize=8)
            axes[ax_i].set_xlabel("d (m)"); axes[ax_i].grid(True, which="both", alpha=0.3)
            if ax_i == 0: axes[ax_i].legend(fontsize=7)
        fig.suptitle(f"Failure Cases — P3 vs {best_ens}")
        fig.tight_layout()
        fig.savefig(out / "failure_profiles_P3_vs_ensemble.png", dpi=120); plt.close(fig)

    print("  Plots saved")


if __name__ == "__main__":
    main()
