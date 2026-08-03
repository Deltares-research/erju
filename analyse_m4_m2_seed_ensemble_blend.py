"""analyse_m4_m2_seed_ensemble_blend.py
==========================================
Read-only, no-training diagnostic (does not modify or retrain any model).

1. Forms M4 and M2 two-seed (42+43) dB-space prediction ensembles
   (predicted_db averaged across the two seeds, event-by-event).
2. Selects one scalar blend weight w for
       pred = w*M4_ensemble + (1-w)*M2_ensemble
   on VALIDATION macro RMSE only (grid search, w=0.00..1.00 step 0.05).
3. Applies the validation-selected w unchanged to TEST and reports the full
   metric suite.
4. Runs paired event-level bootstrap comparisons of the final blend against
   the M4 ensemble, the M2 ensemble, the M1 seed ensemble, and M0 (seed 42
   only -- no M0 seed-43 run exists).
5. Saves final val/test prediction parquet files and a comparison JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.ml.linec_multisensor.data import SENSORS
from src.ml.linec_multisensor.metrics import compute_full_metrics, to_jsonable
from src.ml.spectral.eval_spectral import total_rms_from_db

ROOT = Path(r"P:\11210978-erju-ai\holten_models\outputs\linec_multisensor_v1")
PATHS = {
    "M0": ROOT / "M0" / "M0_seed42_20260803_041551",
    "M1_s42": ROOT / "M1" / "M1_seed42_20260803_050054",
    "M1_s43": ROOT / "M1" / "M1_seed43_20260803_052725",
    "M2_s42": ROOT / "M2" / "M2_seed42_20260803_050057",
    "M2_s43": ROOT / "M2" / "M2_seed43_20260803_052722",
    "M4_s42": ROOT / "M4" / "M4_seed42_20260803_064511",
    "M4_s43": ROOT / "M4" / "M4_seed43_20260803_123914",
}
KEYS = ["event_id", "sensor_id", "band_nominal_hz", "track_number", "distance_m"]
W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
N_BOOT = 1000
BOOT_SEED = 42
OUT_DIR = ROOT / "ENSEMBLE_M4_M2_seed_blend"


def load(name: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(PATHS[name] / f"predictions_{split}.parquet")


def seed_ensemble_db(name_a: str, name_b: str, split: str) -> pd.DataFrame:
    """Average predicted_db of two seed runs of the SAME model, in dB space."""
    da = load(name_a, split)
    db = load(name_b, split)[KEYS + ["predicted_db"]].rename(columns={"predicted_db": "predicted_db_b"})
    m = da.merge(db, on=KEYS, validate="one_to_one")
    m["predicted_db"] = 0.5 * (m["predicted_db"] + m["predicted_db_b"])
    return m.drop(columns=["predicted_db_b"])


def to_3d(df: pd.DataFrame, col: str):
    """Pivot to (N,5,19). Sensor axis is forced to SENSORS order (pivot_table
    sorts sensor_id alphabetically by default, which does NOT match SENSORS =
    ["MP4","MP8","MP10","MP1","MP2"] -- the order compute_full_metrics's
    per_sensor dict assumes via enumerate(SENSORS))."""
    piv = df.pivot_table(index=["event_id", "sensor_id"], columns="band_nominal_hz", values=col)
    band_cols = piv.columns.values
    events = sorted(df["event_id"].unique())
    idx = pd.MultiIndex.from_product([events, SENSORS], names=["event_id", "sensor_id"])
    piv = piv.reindex(idx)
    assert not piv.isna().any().any(), "missing (event,sensor) rows after reindex to SENSORS order"
    return piv.values.reshape(len(events), 5, 19), piv.index, band_cols


def event_mean_total_rms(true_db: np.ndarray, pred_db: np.ndarray) -> dict:
    true_tot = total_rms_from_db(true_db.reshape(-1, 19)).reshape(true_db.shape[:2])
    pred_tot = total_rms_from_db(pred_db.reshape(-1, 19)).reshape(pred_db.shape[:2])
    true_ev = true_tot.mean(axis=1)
    pred_ev = pred_tot.mean(axis=1)
    return {
        "r2": float(r2_score(true_ev, pred_ev)),
        "rmse_mms": float(np.sqrt(np.mean((true_ev - pred_ev) ** 2))),
        "pearson": float(np.corrcoef(true_ev, pred_ev)[0, 1]),
    }


def bootstrap_compare(true_db: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                       n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired event-level bootstrap: mean/95% CI of (macro_rmse_a - macro_rmse_b)
    and the fraction of resamples in which a has the lower (better) RMSE."""
    n_events = true_db.shape[0]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_events, size=n_events)
        t = true_db[idx]
        rmse_a = float(np.sqrt(np.mean((t - pred_a[idx]) ** 2)))
        rmse_b = float(np.sqrt(np.mean((t - pred_b[idx]) ** 2)))
        diffs[i] = rmse_a - rmse_b
    return {
        "mean_diff_db": float(diffs.mean()), "ci95_lo": float(np.percentile(diffs, 2.5)),
        "ci95_hi": float(np.percentile(diffs, 97.5)), "frac_final_blend_better": float(np.mean(diffs < 0)),
    }


def main() -> None:
    m4_val = seed_ensemble_db("M4_s42", "M4_s43", "val")
    m4_test = seed_ensemble_db("M4_s42", "M4_s43", "test")
    m2_val = seed_ensemble_db("M2_s42", "M2_s43", "val")
    m2_test = seed_ensemble_db("M2_s42", "M2_s43", "test")
    m1_test = seed_ensemble_db("M1_s42", "M1_s43", "test")
    m0_test = load("M0", "test")

    # ── weight search on VAL only ────────────────────────────────────────────
    val_merged = m4_val.merge(m2_val[KEYS + ["predicted_db"]].rename(columns={"predicted_db": "pred_m2"}),
                               on=KEYS, validate="one_to_one").rename(columns={"predicted_db": "pred_m4"})
    target_v, m4_v, m2_v = (val_merged[c].to_numpy() for c in ("target_db", "pred_m4", "pred_m2"))

    sweep = []
    for w in W_GRID:
        pred = w * m4_v + (1 - w) * m2_v
        rmse = float(np.sqrt(np.mean((target_v - pred) ** 2)))
        sweep.append({"w": float(w), "val_macro_rmse_db": rmse})
    best = min(sweep, key=lambda r: r["val_macro_rmse_db"])
    best_w = best["w"]
    print(f"val weight sweep: best w={best_w:.2f}  val_macro_rmse={best['val_macro_rmse_db']:.4f} dB  "
          f"(M4_ens alone={sweep[-1]['val_macro_rmse_db']:.4f} dB, M2_ens alone={sweep[0]['val_macro_rmse_db']:.4f} dB)")
    val_merged["predicted_db"] = best_w * val_merged["pred_m4"] + (1 - best_w) * val_merged["pred_m2"]

    # ── apply w unchanged to TEST ────────────────────────────────────────────
    test_merged = m4_test.merge(m2_test[KEYS + ["predicted_db"]].rename(columns={"predicted_db": "pred_m2"}),
                                 on=KEYS, validate="one_to_one").rename(columns={"predicted_db": "pred_m4"})
    test_merged["predicted_db"] = best_w * test_merged["pred_m4"] + (1 - best_w) * test_merged["pred_m2"]

    final_true_3d, final_idx, bands = to_3d(test_merged, "target_db")
    final_pred_3d, _, _ = to_3d(test_merged, "predicted_db")
    m = compute_full_metrics(final_pred_3d, final_true_3d, np.zeros(final_true_3d.shape[:2]), band_nominal=bands)
    ev_mean = event_mean_total_rms(final_true_3d, final_pred_3d)

    print("\ntest @ final blend:")
    print(f"  macro_rmse_db={m['macro_rmse_db']:.4f}  macro_mae_db={m['macro_mae_db']:.4f}  macro_r2={m['macro_r2']:.4f}")
    print(f"  total_rms (global)     rmse_mms={m['total_rms']['rmse_mms']:.6f}  r2={m['total_rms']['r2']:.4f}")
    print(f"  total_rms (event-mean) rmse_mms={ev_mean['rmse_mms']:.6f}  r2={ev_mean['r2']:.4f}  pearson={ev_mean['pearson']:.4f}")
    print(f"  amplitude_compression slope={m['amplitude_compression']['slope']:.4f}")
    print(f"  shape_only_rmse_db={m['shape_only_rmse_db']:.4f}")
    print(f"  strong_event_metrics macro_rmse_db={m['strong_event_metrics']['macro_rmse_db']:.4f}")
    print(f"  per_sensor_rmse_db={ {k: round(v['rmse_db'], 3) for k, v in m['per_sensor'].items()} }")

    # ── paired event-level bootstrap comparisons on TEST ────────────────────
    m4_true_3d, m4_idx, _ = to_3d(m4_test, "target_db")
    m4_pred_3d, _, _ = to_3d(m4_test, "predicted_db")
    m2_true_3d, m2_idx, _ = to_3d(m2_test, "target_db")
    m2_pred_3d, _, _ = to_3d(m2_test, "predicted_db")
    m1_true_3d, m1_idx, _ = to_3d(m1_test, "target_db")
    m1_pred_3d, _, _ = to_3d(m1_test, "predicted_db")
    m0_true_3d, m0_idx, _ = to_3d(m0_test, "target_db")
    m0_pred_3d, _, _ = to_3d(m0_test, "predicted_db")
    for idx in (m4_idx, m2_idx, m1_idx, m0_idx):
        assert list(idx) == list(final_idx), "event/sensor index mismatch across models"

    comparisons = {}
    for label, pred_c in (("vs_M4_ensemble", m4_pred_3d), ("vs_M2_ensemble", m2_pred_3d),
                           ("vs_M1_seed_ensemble", m1_pred_3d), ("vs_M0", m0_pred_3d)):
        comparisons[label] = bootstrap_compare(final_true_3d, final_pred_3d, pred_c, n_boot=N_BOOT, seed=BOOT_SEED)
        c = comparisons[label]
        print(f"  bootstrap {label}: mean_diff={c['mean_diff_db']:+.4f} dB  "
              f"95% CI=[{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}]  "
              f"P(final blend better)={c['frac_final_blend_better']:.3f}")

    # ── save artifacts ───────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["event_id", "sensor_id", "track_number", "distance_m", "band_nominal_hz", "target_db", "predicted_db"]
    for split_name, merged in (("val", val_merged), ("test", test_merged)):
        out = merged[cols].copy()
        out.insert(1, "split", split_name)
        out.to_parquet(OUT_DIR / f"predictions_{split_name}.parquet", index=False)

    report = {
        "description": "M4 (seed42+43) + M2 (seed42+43) dB-space seed-ensemble blend, "
                        "weight selected on validation macro RMSE only",
        "selected_weight_w_on_M4_ensemble": best_w,
        "val_weight_sweep": sweep,
        "test": {
            "macro_rmse_db": m["macro_rmse_db"], "macro_mae_db": m["macro_mae_db"], "macro_r2": m["macro_r2"],
            "total_rms_global": m["total_rms"], "total_rms_event_mean": ev_mean,
            "amplitude_compression": m["amplitude_compression"], "shape_only_rmse_db": m["shape_only_rmse_db"],
            "strong_event_metrics": m["strong_event_metrics"],
            "per_sensor_rmse_db": {k: v["rmse_db"] for k, v in m["per_sensor"].items()},
            "per_band_rmse_db": dict(zip([float(b) for b in bands], m["per_band_rmse_db"])),
        },
        "bootstrap_comparisons_test": comparisons,
        "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED,
        "sources": {k: str(v) for k, v in PATHS.items()},
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(to_jsonable(report), indent=2))
    print(f"\nSaved: {OUT_DIR / 'predictions_val.parquet'}")
    print(f"Saved: {OUT_DIR / 'predictions_test.parquet'}")
    print(f"Saved: {OUT_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
