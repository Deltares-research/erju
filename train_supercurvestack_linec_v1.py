"""
train_supercurvestack_linec_v1.py
SuperCurveStack v1 — Multimodal stacked ensemble for Holten Line-C PGV_z

Base learners trained independently on train split:
  A — Direct log-PGV XGBoost (no physics / with physics features)
  B — Physics residual XGBoost  (pred_log_p3 + residual_hat)
  C — PGV-scaled residual XGBoost
  D — Event-level c/n attenuation model (tabular, fresh c_hat)
  E — Small tabular MLP (residual target)

Meta-learner trained on validation base predictions:
  1. Non-negative linear blend (minimise val RMSE(PGV))
  2. Ridge regression
  3. Shallow XGBoost (depth 2)

Current baselines:
  P3_corrected_n: RMSE(PGV)=2.4006  RMSE(log)=0.5993
  PXGBR-R2:       RMSE(PGV)=2.2718  RMSE(log)=0.5946
  PXGBR_ens_top5: RMSE(PGV)=2.2473  RMSE(log)=0.6223   ← beat this

Target: RMSE(PGV) < 2.20   Stretch: < 2.00
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

def _root() -> Path:
    return Path(r"P:\11210978-erju-ai") if os.name == "nt" else Path("/p/11210978-erju-ai")

MODELS_ROOT  = _root() / "holten_models"
PARQUET_ROOT = _root() / "holten_parquet"
SENSOR_ORDER = ["MP4", "MP8", "MP10", "MP1", "MP2"]
R0 = 10.0

LEAKAGE = {
    "target_pgv_z_mms", "target_log", "target_pgv",
    "c_target", "residual_log", "scaled_residual",
    "max_pgv", "mp4_pgv", "event_id", "split",
    "sensor", "sensor_id", "site_id", "train_type",
    "acc_side_of_track",
}


# ===========================================================================
# DATA LOADING
# ===========================================================================

def find_p3(override=None):
    if override: return Path(override)
    hits = sorted(MODELS_ROOT.glob("cnn_curveprior_p3_savepred_linec_v001_vP3_*"))
    if not hits: raise FileNotFoundError("No P3 savepred dir")
    return hits[-1]

def find_pq2():
    hits = sorted(PARQUET_ROOT.glob("parquet_v002_*"))
    if not hits: raise FileNotFoundError("No parquet_v002")
    return hits[-1] / "dataset.parquet"

def find_pxgbr():
    hits = sorted(MODELS_ROOT.glob("pxgbr_linec_v001_*"))
    return hits[-1] if hits else None

def find_ens():
    hits = sorted(MODELS_ROOT.glob("pxgbr_ensemble_linec_v001_*"))
    return hits[-1] if hits else None


def load_master(p3_dir, pq2_path) -> pd.DataFrame:
    p3 = pd.read_parquet(p3_dir / "all_predictions.parquet")
    fo = pd.read_parquet(pq2_path)
    fo = fo[fo["sensor_id"].isin(SENSOR_ORDER)].copy()
    fo = fo[(fo["target_pgv_z_mms"] > 0) & fo["track_number"].isin([1, 2])]
    fo["event_id"] = fo["event_id"].astype(str)
    fo["sensor"]   = fo["sensor_id"]

    # FO numeric cols (exclude leakage and metadata handled separately)
    fo_num = [c for c in fo.columns
              if pd.api.types.is_numeric_dtype(fo[c])
              and c not in LEAKAGE
              and c not in ("train_speed_kmh", "train_type_code", "track_number",
                            "acc_distance_to_track_m", "acc_distance_to_track_2_m",
                            "effective_distance_to_active_track_m")
              and not c.startswith("target_")
              and "pgv" not in c.lower()]

    meta_cols = ["train_speed_kmh", "train_type_code", "track_number"]

    merged = p3.merge(
        fo[["event_id", "sensor"] + fo_num + meta_cols],
        on=["event_id", "sensor"], how="inner", suffixes=("", "_fo")
    )

    merged["log_distance"]        = np.log(merged["distance"] / R0)
    merged["sensor_code"]         = merged["sensor"].map(
        {s: i for i, s in enumerate(SENSOR_ORDER)}).astype(float)
    spd = merged["train_speed_kmh"]
    merged["train_speed_missing"] = spd.isna().astype(float)
    merged["train_speed_kmh"]     = spd.fillna(0.0)
    merged["residual_log"]        = merged["target_log"] - merged["pred_log_p3"]
    merged["scaled_residual"]     = (
        (merged["target_pgv"] - merged["pred_pgv_p3"])
        / np.sqrt(merged["pred_pgv_p3"].clip(lower=0) + 1.0)
    )
    print(f"Master table: {len(merged):,} rows | {merged.split.value_counts().to_dict()}")
    return merged, fo_num


def fo_feature_cols(df, fo_num):
    explicit = [
        "pred_log_p3", "pred_pgv_p3", "epsilon_p3", "n_used",
        "distance", "log_distance", "track", "sensor_code",
        "train_speed_kmh", "train_speed_missing",
        "train_type_code", "track_number",
    ]
    fc = explicit + [c for c in fo_num if c in df.columns and c not in explicit]
    return [c for c in fc if c in df.columns]


def no_physics_cols(df, fo_num):
    base = [
        "distance", "log_distance", "track", "sensor_code",
        "train_speed_kmh", "train_speed_missing",
        "train_type_code", "track_number",
    ]
    return base + [c for c in fo_num if c in df.columns and c not in base]


# ===========================================================================
# HELPERS
# ===========================================================================

def weights_pgv(df, scheme):
    s  = df["sensor"].values
    tp = df["target_pgv"].values
    w  = np.ones(len(df), dtype=np.float32)
    if scheme == "A":
        w += 1.5*(s=="MP4") + 1.0*(tp>4) + 1.0*(tp>8)
    elif scheme == "D":   # best from ensemble sweep
        w += 1.0*(s=="MP4") + 2.0*(tp>4) + 3.0*(tp>8)
    elif scheme == "B":
        w += 3.0*(s=="MP4") + 1.0*(tp>4) + 1.0*(tp>8)
    return w.astype(np.float32)


def _xgbr(params, n_est=5000, esr=60):
    return xgb.XGBRegressor(
        **params, n_estimators=n_est, early_stopping_rounds=esr,
        eval_metric="rmse", tree_method="hist", verbosity=0, random_state=42
    )


def _fit_best(candidates: List[Dict], X_tr, y_tr, X_va, y_va,
              val_pgv_tr, val_pgv_va, verbose=False):
    """Train all candidates, return sorted by val RMSE(PGV)."""
    results = []
    for i, c in enumerate(candidates):
        w = c.pop("_weights", np.ones(len(y_tr)))
        tgt = c.pop("_target_key", "residual")
        m = _xgbr(c)
        m.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_va, y_va)], verbose=False)
        vp = m.predict(X_va)
        if tgt == "residual":
            val_pred_pgv = np.exp(val_pgv_va + vp)
        elif tgt == "scaled":
            p_pgv = (np.exp(val_pgv_va) + vp * np.sqrt(np.exp(val_pgv_va) + 1)).clip(1e-6)
            val_pred_pgv = p_pgv
        else:  # direct
            val_pred_pgv = np.exp(vp)
        val_rmse_pgv = float(np.sqrt(np.mean((val_pred_pgv - np.exp(val_pgv_va))**2)))
        val_rmse_log = float(m.best_score)
        results.append({
            "model": m, "val_rmse_pgv": val_rmse_pgv,
            "val_rmse_log": val_rmse_log, "best_round": m.best_iteration,
            "target_key": tgt, "val_pred": vp,
        })
        if verbose and (i+1) % 10 == 0:
            best = min(results, key=lambda r: r["val_rmse_pgv"])
            print(f"  [{i+1}/{len(candidates)}] best val RMSE(PGV)={best['val_rmse_pgv']:.4f}")
    results.sort(key=lambda r: r["val_rmse_pgv"])
    return results


# ===========================================================================
# METRICS
# ===========================================================================

def rmse_pgv(pl, tl): return float(np.sqrt(np.mean((np.exp(pl)-np.exp(tl))**2)))
def rmse_log(pl, tl): return float(np.sqrt(np.mean((pl-tl)**2)))


def full_metrics(pl, tl, sensors, event_ids, label):
    pp = np.exp(pl); tp = np.exp(tl)
    r  = lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
    b  = lambda a,b: float(np.mean(a-b))
    r2 = lambda a,b: float(1-(np.sum((b-a)**2)/max(np.sum((b-b.mean())**2),1e-12)))
    m  = {"model": label,
          "rmse_log": r(pl,tl), "rmse_pgv": r(pp,tp),
          "mae_pgv": float(np.mean(np.abs(pp-tp))),
          "r2_log": r2(pl,tl), "bias_pgv": b(pp,tp)}
    m["per_sensor"] = {}
    for s in SENSOR_ORDER:
        msk = sensors==s
        if msk.any():
            m["per_sensor"][s] = {
                "rmse_pgv": r(pp[msk],tp[msk]), "rmse_log": r(pl[msk],tl[msk]),
                "bias_pgv": b(pp[msk],tp[msk]), "bias_log": b(pl[msk],tl[msk])}
    for thr in [4.,8.]:
        hi = tp>thr
        if hi.any():
            m[f"pgv_gt{int(thr)}"] = {"n": int(hi.sum()), "rmse_pgv": r(pp[hi],tp[hi])}
        mp4hi = (sensors=="MP4")&hi
        if mp4hi.any():
            m[f"mp4_pgv_gt{int(thr)}"] = {"n": int(mp4hi.sum()), "rmse_pgv": r(pp[mp4hi],tp[mp4hi])}
    # monotonicity
    viol=0; tot=0
    for eid in np.unique(event_ids)[:300]:
        ev=event_ids==eid
        if ev.sum()!=5: continue
        tmp=pd.DataFrame({"s":sensors[ev],"p":pl[ev]}).set_index("s")
        try: pv=[tmp.loc[s,"p"] for s in SENSOR_ORDER if s in tmp.index]
        except: continue
        for i in range(len(pv)-1): tot+=1; viol+= pv[i]<pv[i+1]
    m["mono_viol_rate"] = viol/tot if tot else 0.
    return m


def apply_mono(df, col):
    out=df[col].copy()
    for _,g in df.groupby("event_id"):
        idx=g.sort_values("distance").index
        out.loc[idx]=np.minimum.accumulate(out.loc[idx].values)
    return out


# ===========================================================================
# EVENT-LEVEL TABLE FOR BASE D
# ===========================================================================

def build_event_table(df, fo_num):
    """One row per event aggregating FO features + P3 diagnostics."""
    rows=[]
    for eid, g in df.groupby("event_id"):
        if len(g)!=5: continue
        sp=g["split"].iloc[0]; tr=int(g["track"].iloc[0]); nu=float(g["n_used"].iloc[0])
        c_tgt=float(g["c_target"].mean())  # used as TARGET only
        row={"event_id":eid,"split":sp,"track":tr,"n_used":nu,
             "c_hat_p3":float(g["c_hat_p3"].iloc[0]) if "c_hat_p3" in g.columns else 0.0,
             "c_target_event":c_tgt,
             "train_speed_kmh":float(g["train_speed_kmh"].iloc[0]),
             "train_speed_missing":float(g["train_speed_missing"].iloc[0]),
             "train_type_code":float(g["train_type_code"].iloc[0]) if "train_type_code" in g.columns else -1.}
        eps=g["epsilon_p3"].values
        for k,v in zip(["eps_mean","eps_std","eps_max","eps_min"],
                       [eps.mean(),eps.std(),eps.max(),eps.min()]):
            row[k]=float(v)
        for c in fo_num:
            if c not in g.columns: continue
            v=g[c].values.astype(float)
            row[f"{c}__mean"]=float(v.mean())
            row[f"{c}__std"] =float(v.std())
            row[f"{c}__max"] =float(v.max())
            row[f"{c}__min"] =float(v.min())
        rows.append(row)
    ev=pd.DataFrame(rows)
    print(f"Event table: {len(ev):,} events")
    return ev


def event_feat_cols(ev_df):
    excluded={"event_id","split","c_target_event","c_hat_p3"}
    return [c for c in ev_df.columns
            if pd.api.types.is_numeric_dtype(ev_df[c]) and c not in excluded]


# ===========================================================================
# BASE E — SMALL MLP
# ===========================================================================

class SmallMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, w_tr, X_va, y_va, n_epochs=200, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc  = StandardScaler().fit(X_tr)
    Xtr = torch.tensor(sc.transform(X_tr), dtype=torch.float32, device=dev)
    Ytr = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    Wtr = torch.tensor(w_tr, dtype=torch.float32, device=dev)
    Xva = torch.tensor(sc.transform(X_va), dtype=torch.float32, device=dev)
    Yva = torch.tensor(y_va, dtype=torch.float32, device=dev)
    model = SmallMLP(X_tr.shape[1]).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best_val, best_state, patience, no_imp = 1e9, None, 20, 0
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(Xtr, Ytr, Wtr)
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    for ep in range(n_epochs):
        model.train()
        for xb, yb, wb in dl:
            pred = model(xb)
            loss = (wb * torch.abs(pred-yb)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xva).cpu().numpy()
        vl = float(np.sqrt(np.mean((vp-y_va)**2)))
        if vl < best_val-1e-5:
            best_val=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp+=1
        if no_imp>=patience: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(Xva).cpu().numpy()
    return model, sc, val_pred, best_val


# ===========================================================================
# META LEARNER
# ===========================================================================

def blend_nonneg(base_val, y_val, base_te, val_pgv):
    """Optimise non-negative weights summing to 1 on validation RMSE(PGV)."""
    n_base = base_val.shape[1]
    def obj(w):
        w_pos = np.maximum(w, 0); w_pos /= w_pos.sum()+1e-9
        pl = base_val @ w_pos
        return float(np.sqrt(np.mean((np.exp(pl)-val_pgv)**2)))
    best_val_rpgv = 1e9; best_w = None
    for _ in range(30):  # random restarts
        w0 = np.random.dirichlet(np.ones(n_base))
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol":1e-5})
        if res.fun < best_val_rpgv:
            best_val_rpgv = res.fun
            best_w = np.maximum(res.x, 0)
            best_w /= best_w.sum()+1e-9
    pred_val = base_val @ best_w
    pred_te  = base_te  @ best_w
    return pred_val, pred_te, best_w, best_val_rpgv


def blend_ridge(base_val, y_val, base_te):
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(base_val, y_val)
    return ridge.predict(base_val), ridge.predict(base_te), ridge


def blend_xgb(base_val, y_val, base_te, val_pgv):
    """Shallow XGBoost meta-learner."""
    Xv = base_val; Xt = base_te
    best_rpgv=1e9; best_pred_va=None; best_pred_te=None
    for depth in [2,3]:
        for lr in [0.05,0.1]:
            m=xgb.XGBRegressor(max_depth=depth,learning_rate=lr,
                                n_estimators=500,subsample=0.9,
                                tree_method="hist",verbosity=0,random_state=42)
            m.fit(Xv, y_val, eval_set=[(Xv,y_val)], verbose=False)
            pv=m.predict(Xv)
            rv=float(np.sqrt(np.mean((np.exp(pv)-val_pgv)**2)))
            if rv<best_rpgv:
                best_rpgv=rv; best_pred_va=pv; best_pred_te=m.predict(Xt)
    return best_pred_va, best_pred_te, best_rpgv


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3_dir", default=None)
    ap.add_argument("--output_tag", default="supercurvestack_linec_v001")
    args = ap.parse_args()

    print("="*80); print("SuperCurveStack v1 — Holten Line-C"); print("="*80)

    p3_dir    = find_p3(args.p3_dir)
    pq2_path  = find_pq2()
    pxgbr_dir = find_pxgbr()
    ens_dir   = find_ens()
    print(f"P3:   {p3_dir.name}")
    print(f"PXGBR: {pxgbr_dir.name if pxgbr_dir else 'n/a'}")
    print(f"Ens:  {ens_dir.name if ens_dir else 'n/a'}")

    # ── Data ────────────────────────────────────────────────────────────────
    df, fo_num = load_master(p3_dir, pq2_path)
    fc_all     = fo_feature_cols(df, fo_num)
    fc_nophys  = no_physics_cols(df, fo_num)

    for c in fc_all:
        df[c] = df[c].fillna(0.0)

    tr = df[df.split=="train"].copy()
    va = df[df.split=="val"].copy()
    te = df[df.split=="test"].copy()

    Xtr_all = tr[fc_all].values.astype(np.float32)
    Xva_all = va[fc_all].values.astype(np.float32)
    Xte_all = te[fc_all].values.astype(np.float32)
    Xtr_np  = tr[fc_nophys].values.astype(np.float32)
    Xva_np  = va[fc_nophys].values.astype(np.float32)
    Xte_np  = te[fc_nophys].values.astype(np.float32)

    y_tr_log = tr["target_log"].values.astype(np.float32)
    y_va_log = va["target_log"].values.astype(np.float32)
    y_tr_res = tr["residual_log"].values.astype(np.float32)
    y_va_res = va["residual_log"].values.astype(np.float32)
    y_tr_scl = tr["scaled_residual"].values.astype(np.float32)
    y_va_scl = va["scaled_residual"].values.astype(np.float32)

    val_pgv  = va["target_pgv"].values
    te_pgv   = te["target_pgv"].values

    # Dictionary to accumulate val/test predictions from all base models
    base_val: Dict[str, np.ndarray] = {}
    base_te:  Dict[str, np.ndarray] = {}
    fi_store: Dict[str, np.ndarray] = {}

    def _store_preds(name, val_log, te_log):
        base_val[name] = val_log
        base_te[name]  = te_log

    # ── Load external baselines ───────────────────────────────────────────────
    if pxgbr_dir:
        try:
            px=pd.read_parquet(pxgbr_dir/"predictions_test.parquet")
            pk_te=te["event_id"]+"|"+te["sensor"]
            _store_preds("P3",     va["pred_log_p3"].values, te["pred_log_p3"].values)
            px_map=dict(zip(px["event_id"]+"|"+px["sensor"], px["pred_log_R2"]))
            pl_te=pk_te.map(px_map).values
            pk_va=va["event_id"]+"|"+va["sensor"]
            px_va=pd.read_parquet(pxgbr_dir/"predictions_test.parquet")
            # Use val from test parquet proxy (not available) — skip PXGBR val
            if not pd.isna(pl_te).any():
                base_te["PXGBR-R2"]=pl_te
        except Exception as e: print(f"  [WARN] PXGBR load: {e}")
    else:
        _store_preds("P3", va["pred_log_p3"].values, te["pred_log_p3"].values)

    if "P3" not in base_val:
        _store_preds("P3", va["pred_log_p3"].values, te["pred_log_p3"].values)

    if ens_dir:
        try:
            ep=pd.read_parquet(ens_dir/"ensemble_predictions_test.parquet")
            if "pred_log_ens_top5" in ep.columns:
                pk_te=te["event_id"]+"|"+te["sensor"]
                em=dict(zip(ep["event_id"]+"|"+ep["sensor"], ep["pred_log_ens_top5"]))
                pl_te=pk_te.map(em).values
                if not pd.isna(pl_te).any(): base_te["ens_top5"]=pl_te
        except Exception as e: print(f"  [WARN] Ens load: {e}")

    # ── Base A: Direct XGBoost ────────────────────────────────────────────────
    print("\n" + "="*60 + "\nBASE A: Direct log-PGV XGBoost\n" + "="*60)

    # Best config from ensemble sweep: depth=3, lr=0.02, mcw=5, rl=10
    # Focused grid around that
    a_grid = [dict(max_depth=d, learning_rate=lr, min_child_weight=mcw,
                   reg_lambda=rl, subsample=0.9, colsample_bytree=0.8,
                   objective="reg:squarederror",
                   _weights=weights_pgv(tr, sch), _target_key="direct",
                   _name=f"A_{feat}_{sch}_d{d}_lr{lr}_m{mcw}_l{rl}")
              for d, lr, mcw, rl, sch, feat in product(
                  [3, 4, 5], [0.02, 0.05], [1, 5], [1, 10],
                  ["uniform", "D"], ["nophys", "phys"])]

    print(f"  Grid size: {len(a_grid)}")
    a_results_nop, a_results_phy = [], []

    for c in a_grid:
        feat = c.pop("_name"); use_phys = ("phys" in feat)
        X_tr_ = Xtr_all if use_phys else Xtr_np
        X_va_ = Xva_all if use_phys else Xva_np
        y_tr_ = y_tr_log; y_va_ = y_va_log
        w = c.pop("_weights"); c.pop("_target_key")
        m = _xgbr(c)
        m.fit(X_tr_, y_tr_, sample_weight=w, eval_set=[(X_va_, y_va_)], verbose=False)
        vp = m.predict(X_va_)
        rv = float(np.sqrt(np.mean((np.exp(vp)-val_pgv)**2)))
        rec = {"model":m, "val_rmse_pgv":rv, "val_rmse_log":float(m.best_score),
               "use_phys":use_phys, "feat":feat}
        if use_phys: a_results_phy.append(rec)
        else:        a_results_nop.append(rec)

    a_results_nop.sort(key=lambda r: r["val_rmse_pgv"])
    a_results_phy.sort(key=lambda r: r["val_rmse_pgv"])
    best_A1 = a_results_nop[0]
    best_A2 = a_results_phy[0]
    print(f"  A1 (no physics): best val RMSE(PGV)={best_A1['val_rmse_pgv']:.4f}")
    print(f"  A2 (with physics): best val RMSE(PGV)={best_A2['val_rmse_pgv']:.4f}")

    _store_preds("A1", best_A1["model"].predict(Xva_np).astype(np.float32),
                       best_A1["model"].predict(Xte_np).astype(np.float32))
    _store_preds("A2", best_A2["model"].predict(Xva_all).astype(np.float32),
                       best_A2["model"].predict(Xte_all).astype(np.float32))
    fi_store["A2"] = pd.DataFrame({"feature": fc_all,
                                    "importance": best_A2["model"].feature_importances_}
                                  ).sort_values("importance",ascending=False)

    # ── Base B: Physics residual XGBoost ─────────────────────────────────────
    print("\n" + "="*60 + "\nBASE B: Physics residual XGBoost\n" + "="*60)

    # Warm start from best ensemble config: depth=3, lr=0.02, mcw=5, rl=10
    b_grid = []
    for d,lr,mcw,rl,ss,cbt,sch in product(
        [3,4], [0.02,0.05], [1,5], [5,10], [0.9], [0.8,0.9], ["uniform","A","D"]
    ):
        b_grid.append(dict(max_depth=d, learning_rate=lr, min_child_weight=mcw,
                           reg_lambda=rl, subsample=ss, colsample_bytree=cbt,
                           objective="reg:squarederror",
                           _scheme=sch))

    print(f"  Grid size: {len(b_grid)}")
    b_results = []
    for c in b_grid:
        sch=c.pop("_scheme"); w=weights_pgv(tr, sch)
        m=_xgbr(c); m.fit(Xtr_all, y_tr_res, sample_weight=w,
                           eval_set=[(Xva_all,y_va_res)], verbose=False)
        vp=m.predict(Xva_all)
        rv=float(np.sqrt(np.mean((np.exp(va["pred_log_p3"].values+vp)-val_pgv)**2)))
        b_results.append({"model":m,"val_rmse_pgv":rv,"scheme":sch,"residual_pred_va":vp})

    b_results.sort(key=lambda r: r["val_rmse_pgv"])
    best_B = b_results[0]
    print(f"  Best val RMSE(PGV)={best_B['val_rmse_pgv']:.4f} (scheme={best_B['scheme']})")
    _store_preds("B",
        va["pred_log_p3"].values + best_B["model"].predict(Xva_all).astype(np.float32),
        te["pred_log_p3"].values + best_B["model"].predict(Xte_all).astype(np.float32))
    fi_store["B"] = pd.DataFrame({"feature":fc_all,
                                   "importance":best_B["model"].feature_importances_}
                                 ).sort_values("importance",ascending=False)

    # Ensemble of top-5 B models
    top5_B_val = np.mean(
        [va["pred_log_p3"].values + r["model"].predict(Xva_all) for r in b_results[:5]], axis=0)
    top5_B_te  = np.mean(
        [te["pred_log_p3"].values + r["model"].predict(Xte_all) for r in b_results[:5]], axis=0)
    _store_preds("B_ens5", top5_B_val.astype(np.float32), top5_B_te.astype(np.float32))

    # ── Base C: PGV-scaled residual ────────────────────────────────────────────
    print("\n" + "="*60 + "\nBASE C: PGV-scaled residual\n" + "="*60)

    c_grid=[]
    for d,lr,mcw,rl in product([3,4],[0.02,0.05],[1,5],[5,10]):
        c_grid.append(dict(max_depth=d,learning_rate=lr,min_child_weight=mcw,
                           reg_lambda=rl,subsample=0.9,colsample_bytree=0.8,
                           objective="reg:squarederror"))

    c_results=[]
    for c in c_grid:
        w=weights_pgv(tr,"D"); m=_xgbr(c)
        m.fit(Xtr_all,y_tr_scl,sample_weight=w,eval_set=[(Xva_all,y_va_scl)],verbose=False)
        vp=m.predict(Xva_all)
        pp=(np.exp(va["pred_log_p3"].values)+vp*np.sqrt(np.exp(va["pred_log_p3"].values)+1)).clip(1e-6)
        rv=float(np.sqrt(np.mean((pp-val_pgv)**2)))
        c_results.append({"model":m,"val_rmse_pgv":rv,"scl_pred_va":vp})

    c_results.sort(key=lambda r: r["val_rmse_pgv"])
    best_C=c_results[0]
    print(f"  Best val RMSE(PGV)={best_C['val_rmse_pgv']:.4f}")

    def _c_reconstruct(p3_log, scl_pred):
        pp=(np.exp(p3_log)+scl_pred*np.sqrt(np.exp(p3_log)+1)).clip(1e-6)
        return np.log(pp)

    _store_preds("C",
        _c_reconstruct(va["pred_log_p3"].values, best_C["model"].predict(Xva_all)).astype(np.float32),
        _c_reconstruct(te["pred_log_p3"].values, best_C["model"].predict(Xte_all)).astype(np.float32))

    # ── Base D: Event-level c/n model ─────────────────────────────────────────
    print("\n" + "="*60 + "\nBASE D: Event-level c/n attenuation\n" + "="*60)

    ev_df = build_event_table(df, fo_num)
    efc   = event_feat_cols(ev_df)
    for c in efc:
        ev_df[c] = ev_df[c].fillna(0.0)

    ev_tr = ev_df[ev_df.split=="train"]
    ev_va = ev_df[ev_df.split=="val"]
    ev_te = ev_df[ev_df.split=="test"]

    Xev_tr = ev_tr[efc].values.astype(np.float32)
    Xev_va = ev_va[efc].values.astype(np.float32)
    Xev_te = ev_te[efc].values.astype(np.float32)
    y_c_tr = ev_tr["c_target_event"].values.astype(np.float32)
    y_c_va = ev_va["c_target_event"].values.astype(np.float32)

    # Compute per-event n values (for D2)
    n_map_tr = dict(zip(ev_tr["event_id"], zip(
        [float(g["n_used"].iloc[0]) for _,g in df[df.split=="train"].groupby("event_id")],
        ev_tr["event_id"]
    )))

    d_results=[]
    for d,lr,rl in product([3,4],[0.03,0.05],[5,10]):
        m=_xgbr(dict(max_depth=d,learning_rate=lr,reg_lambda=rl,
                     subsample=0.9,colsample_bytree=0.8,objective="reg:squarederror"))
        m.fit(Xev_tr,y_c_tr,eval_set=[(Xev_va,y_c_va)],verbose=False)
        # Reconstruct on validation
        c_hat_va = m.predict(Xev_va)
        c_hat_te = m.predict(Xev_te)
        c_map_va = dict(zip(ev_va["event_id"], c_hat_va))
        c_map_te = dict(zip(ev_te["event_id"], c_hat_te))
        # Use track-specific n from event table
        def _recon(df_split, c_map, n_col_default=1.0655):
            preds=[]
            for _,row in df_split.iterrows():
                eid=row["event_id"]; nu=row["n_used"]; d=row["distance"]
                c=c_map.get(eid, 0.0)
                preds.append(c - nu*np.log(d/R0))
            return np.array(preds, dtype=np.float32)
        pl_va = _recon(va, c_map_va)
        pl_te = _recon(te, c_map_te)
        rv=float(np.sqrt(np.mean((np.exp(pl_va)-val_pgv)**2)))
        d_results.append({"model":m,"val_rmse_pgv":rv,"c_hat_va":c_hat_va,
                          "pl_va":pl_va,"pl_te":pl_te})

    d_results.sort(key=lambda r: r["val_rmse_pgv"])
    best_D=d_results[0]
    print(f"  Best val RMSE(PGV)={best_D['val_rmse_pgv']:.4f}")
    _store_preds("D1", best_D["pl_va"], best_D["pl_te"])
    fi_store["D"] = pd.DataFrame({"feature":efc,
                                   "importance":best_D["model"].feature_importances_}
                                 ).sort_values("importance",ascending=False)

    # ── Base E: Small MLP ─────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nBASE E: Small tabular MLP (residual target)\n" + "="*60)

    X_mlp_tr = Xtr_all; X_mlp_va = Xva_all; X_mlp_te = Xte_all
    w_mlp    = weights_pgv(tr, "D")
    best_mlp_rpgv=1e9; best_mlp_va=None; best_mlp_te=None

    for seed in [42, 7, 13]:
        try:
            mlp, sc_mlp, vp_mlp, vl_mlp = train_mlp(
                X_mlp_tr, y_tr_res, w_mlp, X_mlp_va, y_va_res, n_epochs=150, seed=seed)
            pl_va = va["pred_log_p3"].values + vp_mlp
            rv    = float(np.sqrt(np.mean((np.exp(pl_va)-val_pgv)**2)))
            print(f"  seed={seed}: val_res_rmse={vl_mlp:.4f}  val_RMSE(PGV)={rv:.4f}")
            if rv<best_mlp_rpgv:
                best_mlp_rpgv=rv
                with torch.no_grad():
                    mlp.eval()
                    dev = next(mlp.parameters()).device
                    Xte_t = torch.tensor(sc_mlp.transform(X_mlp_te),dtype=torch.float32,device=dev)
                    res_te = mlp(Xte_t).cpu().numpy()
                best_mlp_va = pl_va.astype(np.float32)
                best_mlp_te = (te["pred_log_p3"].values + res_te).astype(np.float32)
        except Exception as e:
            print(f"  seed={seed} failed: {e}")

    if best_mlp_va is not None:
        print(f"  Best MLP val RMSE(PGV)={best_mlp_rpgv:.4f}")
        _store_preds("E_mlp", best_mlp_va, best_mlp_te)

    # ── Collect all base predictions ──────────────────────────────────────────
    print("\n" + "="*60 + "\nMETA-LEARNER\n" + "="*60)

    # Only use base models that have both val and test predictions
    common = sorted(set(base_val) & set(base_te))
    print(f"  Base models with val+test preds: {common}")

    mat_va = np.column_stack([base_val[k] for k in common]).astype(np.float32)
    mat_te = np.column_stack([base_te[k]  for k in common]).astype(np.float32)

    # Add distance + sensor as meta features
    meta_extra_va = np.column_stack([va["log_distance"].values,
                                     va["sensor_code"].values]).astype(np.float32)
    meta_extra_te = np.column_stack([te["log_distance"].values,
                                     te["sensor_code"].values]).astype(np.float32)
    mat_va_ext = np.hstack([mat_va, meta_extra_va])
    mat_te_ext = np.hstack([mat_te, meta_extra_te])

    y_va_log_m = va["target_log"].values.astype(np.float32)

    # 1. Non-negative linear blend
    pred_blend_va, pred_blend_te, blend_w, blend_rpgv = blend_nonneg(
        mat_va, y_va_log_m, mat_te, val_pgv)
    print(f"  Blend val RMSE(PGV)={blend_rpgv:.4f}  weights={blend_w.round(3)}")

    # 2. Ridge
    pred_ridge_va, pred_ridge_te, ridge_model = blend_ridge(mat_va_ext, y_va_log_m, mat_te_ext)
    rv_ridge = float(np.sqrt(np.mean((np.exp(pred_ridge_va)-val_pgv)**2)))
    print(f"  Ridge val RMSE(PGV)={rv_ridge:.4f}")

    # 3. Shallow XGBoost meta
    pred_xgb_va, pred_xgb_te, rv_xgb = blend_xgb(mat_va_ext, y_va_log_m, mat_te_ext, val_pgv)
    print(f"  XGB meta val RMSE(PGV)={rv_xgb:.4f}")

    # Select best meta by val RMSE(PGV)
    meta_candidates = [
        ("blend",  pred_blend_va.astype(np.float32), pred_blend_te.astype(np.float32), blend_rpgv),
        ("ridge",  pred_ridge_va.astype(np.float32), pred_ridge_te.astype(np.float32), rv_ridge),
        ("xgb",    pred_xgb_va.astype(np.float32),   pred_xgb_te.astype(np.float32),   rv_xgb),
    ]
    meta_candidates.sort(key=lambda x: x[3])
    best_meta_name, best_meta_va, best_meta_te, best_meta_rv = meta_candidates[0]
    print(f"  Best meta: {best_meta_name} val RMSE(PGV)={best_meta_rv:.4f}")

    _store_preds("meta_best", best_meta_va, best_meta_te)
    _store_preds("meta_blend", pred_blend_va.astype(np.float32), pred_blend_te.astype(np.float32))
    _store_preds("meta_ridge", pred_ridge_va.astype(np.float32), pred_ridge_te.astype(np.float32))

    # ── Monotonic post-processing ─────────────────────────────────────────────
    best_pred_te_log = best_meta_te
    meta_mono_te = apply_mono(te, "_tmp").values  # placeholder
    # Apply properly
    tmp_df = te.copy(); tmp_df["_meta"] = best_pred_te_log
    meta_mono_te = apply_mono(tmp_df, "_meta").values

    # ── Test metrics ──────────────────────────────────────────────────────────
    print("\n" + "="*80 + "\nTEST METRICS\n" + "="*80)

    tgt_log   = te["target_log"].values
    sensors_te = te["sensor"].values
    ev_ids_te  = te["event_id"].values

    all_metrics: Dict[str, Dict] = {}
    eval_preds = {k: base_te[k] for k in sorted(base_te)}
    eval_preds["meta_best_mono"] = meta_mono_te

    for name, pl in eval_preds.items():
        if pl is not None and not np.isnan(pl).any():
            all_metrics[name] = full_metrics(pl, tgt_log, sensors_te, ev_ids_te, name)

    print(f"\n{'Model':<20} {'RMSE(log)':>10} {'RMSE(PGV)':>10} {'R²(log)':>8} {'MP4':>8} {'mono':>6}")
    print("-"*70)
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1]["rmse_pgv"]):
        mp4 = m["per_sensor"].get("MP4", {}).get("rmse_pgv", float("nan"))
        mvr = m.get("mono_viol_rate", float("nan"))
        print(f"{name:<20} {m['rmse_log']:>10.4f} {m['rmse_pgv']:>10.4f} "
              f"{m['r2_log']:>8.4f} {mp4:>8.4f} {mvr:>6.3f}")

    best_name = min(all_metrics, key=lambda n: all_metrics[n]["rmse_pgv"])
    print(f"\nBest: {best_name}  RMSE(PGV)={all_metrics[best_name]['rmse_pgv']:.4f}")

    # Base model correlations on val
    corr = pd.DataFrame({k: base_val[k] for k in common}).corr()

    # ── Save outputs ─────────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = MODELS_ROOT / f"{args.output_tag}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    log_dir = Path("outputs/supercurvestack_linec_v1")
    log_dir.mkdir(parents=True, exist_ok=True)

    # master_table.parquet
    df.to_parquet(out / "master_table.parquet", index=False)
    (out/"feature_columns_all.txt").write_text("\n".join(fc_all), encoding="utf-8")
    (out/"leakage_audit.txt").write_text(
        "Excluded leakage columns:\n"+"\n".join(sorted(LEAKAGE)), encoding="utf-8")

    # metrics
    with open(out/"metrics.json","w") as f:
        json.dump({k:{kk:vv for kk,vv in v.items() if kk!="per_sensor"}
                   for k,v in all_metrics.items()}, f, indent=2, default=float)
    pd.DataFrame([{
        "model": n,
        **{k:v for k,v in m.items() if k not in ("per_sensor",)},
        **{f"{s}_rmse_pgv": m["per_sensor"].get(s,{}).get("rmse_pgv",float("nan"))
           for s in SENSOR_ORDER}
    } for n,m in all_metrics.items()]).to_csv(out/"metrics_table.csv", index=False)

    # predictions
    pred_out = te[["event_id","sensor","track","distance","split",
                   "target_log","target_pgv","pred_log_p3","pred_pgv_p3"]].copy()
    for name, pl in eval_preds.items():
        if pl is not None:
            pred_out[f"pred_log_{name}"] = pl
            pred_out[f"pred_pgv_{name}"] = np.exp(pl)
    pred_out.to_parquet(out/"predictions_test.parquet", index=False)

    # val base predictions
    va_out = va[["event_id","sensor","target_log","target_pgv"]].copy()
    for k in common: va_out[f"pred_log_{k}"] = base_val[k]
    va_out.to_parquet(out/"base_predictions_val.parquet", index=False)

    # feature importances
    for name, fi_df in fi_store.items():
        fi_df.to_csv(out/f"feature_importance_{name}.csv", index=False)

    # meta weights
    with open(out/"meta_weights.json","w") as f:
        json.dump({"blend_weights": dict(zip(common, blend_w.tolist())),
                   "best_meta": best_meta_name,
                   "blend_val_rpgv": blend_rpgv,
                   "ridge_val_rpgv": rv_ridge,
                   "xgb_meta_val_rpgv": rv_xgb}, f, indent=2, default=float)

    # base model correlation
    corr.to_csv(out/"base_model_correlation_val.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        _make_plots(pred_out, all_metrics, best_name, out)
    except Exception as e:
        print(f"[WARN] Plots failed: {e}")

    print(f"\nOutput saved to: {out}")
    print("\n" + "="*40 + " FINAL SUMMARY " + "="*40)
    p3_r = all_metrics.get("P3", {}).get("rmse_pgv", 2.4006)
    pxgbr_r = all_metrics.get("PXGBR-R2", {}).get("rmse_pgv", 2.2718)
    ens_r  = 2.2473  # from prior run
    best_r = all_metrics[best_name]["rmse_pgv"]
    print(f"P3:            RMSE(PGV)={p3_r:.4f}")
    print(f"PXGBR-R2:      RMSE(PGV)={pxgbr_r:.4f}")
    print(f"PXGBR ens_top5:RMSE(PGV)={ens_r:.4f}")
    print(f"Best stack:    RMSE(PGV)={best_r:.4f}  ({best_name})")
    print(f"vs P3:         {best_r-p3_r:+.4f} ({(best_r-p3_r)/p3_r*100:+.1f}%)")
    print(f"vs ens_top5:   {best_r-ens_r:+.4f}")


def _make_plots(pred_df, all_metrics, best_name, out):
    colours = {"P3":"#1f77b4","PXGBR-R2":"#ff7f0e","B_ens5":"#9467bd",
               "meta_best":"#2ca02c","meta_best_mono":"#17becf"}

    # 1. Scatter: top-3 + P3
    top3 = sorted(all_metrics, key=lambda n: all_metrics[n]["rmse_pgv"])[:3]
    plot_models = list(dict.fromkeys(["P3"] + top3))[:4]
    fig, axes = plt.subplots(1, len(plot_models), figsize=(5*len(plot_models),5), squeeze=False)
    for ax, name in zip(axes[0], plot_models):
        col = f"pred_pgv_{name}"
        if col not in pred_df.columns: ax.set_visible(False); continue
        ax.scatter(pred_df["target_pgv"], pred_df[col], s=4, alpha=0.3,
                   color=colours.get(name,"grey"), rasterized=True)
        lim=max(pred_df["target_pgv"].max(), pred_df[col].max())*1.05
        ax.plot([0.05,lim],[0.05,lim],"r--",lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{name}\nRMSE={all_metrics[name]['rmse_pgv']:.3f}")
        ax.set_xlabel("Measured"); ax.set_ylabel("Predicted"); ax.grid(True,which="both",alpha=0.3)
    fig.suptitle("Measured vs Predicted"); fig.tight_layout()
    fig.savefig(out/"measured_vs_predicted_all_models.png",dpi=120,bbox_inches="tight")
    plt.close(fig)

    # 2. Per-sensor RMSE bar chart
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    x=np.arange(5); w=0.8/max(len(plot_models),1)
    for metric,ylabel,ax in [("rmse_pgv","RMSE(PGV)",axes[0]),("rmse_log","RMSE(log)",axes[1])]:
        for j,name in enumerate(plot_models):
            if name not in all_metrics: continue
            vals=[all_metrics[name]["per_sensor"].get(s,{}).get(metric,float("nan")) for s in SENSOR_ORDER]
            ax.bar(x+(j-len(plot_models)/2+0.5)*w,vals,w,label=name,alpha=0.8,color=colours.get(name,"grey"))
        ax.set_xticks(x); ax.set_xticklabels(SENSOR_ORDER)
        ax.set_ylabel(ylabel); ax.legend(fontsize=7); ax.grid(True,axis="y",alpha=0.4)
    fig.suptitle("Per-Sensor Error — All Models"); fig.tight_layout()
    fig.savefig(out/"per_sensor_rmse_all_models.png",dpi=120,bbox_inches="tight")
    plt.close(fig)

    best_col = f"pred_pgv_{best_name}"
    if best_col in pred_df.columns:
        # 3. High-PGV profiles
        hi_evs = list(pred_df.groupby("event_id")["target_pgv"].max()
                              .sort_values(ascending=False).head(6).index)
        for tag, evs, fname in [("high_pgv", hi_evs, "high_pgv_profiles_best_stack.png"),
                                 ("failure",
                                  list(pred_df.assign(sq=lambda d:(d[best_col]-d["target_pgv"])**2)
                                              .groupby("event_id")["sq"].mean()
                                              .apply(np.sqrt).sort_values(ascending=False).head(6).index),
                                  "failure_profiles_best_stack.png")]:
            fig,axes=plt.subplots(2,3,figsize=(13,8)); axes=axes.flatten()
            for ax_i,ev in enumerate(evs[:6]):
                ev_df=pred_df[pred_df.event_id==ev].sort_values("distance")
                if ev_df.empty: continue
                d=ev_df["distance"].values; t=ev_df["target_pgv"].values
                axes[ax_i].semilogy(d,t,"ko-",ms=5,label="Measured")
                axes[ax_i].semilogy(d,ev_df["pred_pgv_P3"].values if "pred_pgv_P3" in ev_df.columns
                                    else ev_df["pred_pgv_p3"].values,"b^--",ms=4,lw=1,label="P3")
                axes[ax_i].semilogy(d,ev_df[best_col].values,"g^--",ms=4,lw=1.2,label=best_name)
                axes[ax_i].set_title(str(ev)[-12:-4],fontsize=8); axes[ax_i].set_xlabel("d (m)")
                axes[ax_i].grid(True,which="both",alpha=0.3)
                if ax_i==0: axes[ax_i].legend(fontsize=7)
            fig.suptitle(f"{tag.replace('_',' ').title()} — P3 vs {best_name}")
            fig.tight_layout(); fig.savefig(out/fname,dpi=120); plt.close(fig)

    # 4. Residual vs distance — best stack
    if best_col in pred_df.columns:
        fig,ax=plt.subplots(figsize=(7,5))
        ax.scatter(pred_df["distance"],pred_df[best_col]-pred_df["target_pgv"],
                   s=4,alpha=0.3,rasterized=True)
        ax.axhline(0,color="k",lw=0.8)
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Residual (mm/s)")
        ax.set_title(f"Residual vs Distance — {best_name}"); ax.grid(True,alpha=0.3)
        fig.tight_layout(); fig.savefig(out/"residual_vs_distance_stack.png",dpi=120); plt.close(fig)

    print("  Plots saved")


if __name__ == "__main__":
    main()
