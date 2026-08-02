#!/usr/bin/env python3
"""analyse_linec_multisensor_compare_v1.py
===========================================
Cross-run comparison for the Line-C multi-sensor experiment:

    M0 vs M1  (does the waveform add anything over metadata alone?)
    M1 vs M2  (does the O0 physics decode help or hurt vs direct regression?)
    M2 vs the O0 Mode-B oracle ceiling (best-case analytic upper bound)

Paired event-level bootstrap CI is used for M1 vs M2 (same resample applied
to both models' predictions, from predictions_test.parquet of each run).

Run AFTER individual run_linec_multisensor_v1.py runs have completed (one
run directory per model/seed). Does not train or modify anything.

Usage
-----
    python analyse_linec_multisensor_compare_v1.py \\
        --m0-dir <path> --m1-dir <path> --m2-dir <path> \\
        --oracle-run-dir <path to linec_spectral_oracle_v1/run_YYYYMMDD_HHMMSS>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.linec_multisensor.data import ORACLE_RUN_ID, _root
from src.ml.linec_multisensor.metrics import paired_bootstrap_diff_ci
from src.ml.spectral.eval_spectral import _rmse


def _load_pred_tensor(run_dir: Path, split_name: str):
    df = pd.read_parquet(run_dir / f"predictions_{split_name}.parquet")
    events = sorted(df["event_id"].unique())
    sensors = sorted(df["sensor_id"].unique())
    bands = sorted(df["band_nominal_hz"].unique())
    ev_pos = {e: i for i, e in enumerate(events)}
    s_pos = {s: i for i, s in enumerate(sensors)}
    b_pos = {b: i for i, b in enumerate(bands)}
    n, s, f = len(events), len(sensors), len(bands)
    true_db = np.full((n, s, f), np.nan)
    pred_db = np.full((n, s, f), np.nan)
    for row in df.itertuples():
        i, j, k = ev_pos[row.event_id], s_pos[row.sensor_id], b_pos[row.band_nominal_hz]
        true_db[i, j, k] = row.target_db
        pred_db[i, j, k] = row.predicted_db
    assert not np.isnan(true_db).any() and not np.isnan(pred_db).any(), "incomplete prediction grid"
    return true_db, pred_db, events, sensors, bands


def _results_json(run_dir: Path) -> dict:
    return json.loads((run_dir / "results.json").read_text())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m0-dir", type=Path, required=True)
    p.add_argument("--m1-dir", type=Path, required=True)
    p.add_argument("--m2-dir", type=Path, required=True)
    p.add_argument("--oracle-run-dir", type=Path, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    split = args.split
    oracle_dir = args.oracle_run_dir or (
        _root() / "holten_models" / "outputs" / "linec_spectral_oracle_v1" / ORACLE_RUN_ID
    )

    res = {"m0": _results_json(args.m0_dir), "m1": _results_json(args.m1_dir), "m2": _results_json(args.m2_dir)}
    macro = {k: v[split]["macro_rmse_db"] for k, v in res.items()}

    oracle_res = json.loads((oracle_dir / "results.json").read_text())
    oracle_o0_modeb_test = oracle_res["O0"]["eval"]["modeB"]["test"]["overall"]["macro_rmse_db"]

    print("=" * 78)
    print(f"Line-C multi-sensor comparison ({split} split)")
    print("=" * 78)
    print(f"  M0 macro_rmse_db = {macro['m0']:.3f}")
    print(f"  M1 macro_rmse_db = {macro['m1']:.3f}")
    print(f"  M2 macro_rmse_db = {macro['m2']:.3f}")
    print(f"  O0 mode-B oracle ceiling (test) = {oracle_o0_modeb_test:.3f}")
    print(f"  M0 vs M1: {'M1 better' if macro['m1'] < macro['m0'] else 'M0 better/equal'} "
          f"(delta={macro['m0'] - macro['m1']:+.3f} dB)")
    print(f"  M1 vs M2: {'M2 better' if macro['m2'] < macro['m1'] else 'M1 better/equal'} "
          f"(delta={macro['m1'] - macro['m2']:+.3f} dB)")
    print(f"  M2 vs oracle ceiling: gap = {macro['m2'] - oracle_o0_modeb_test:+.3f} dB "
          f"(0 = M2 matches the analytic best case)")

    true1, pred1, ev1, *_ = _load_pred_tensor(args.m1_dir, split)
    true2, pred2, ev2, *_ = _load_pred_tensor(args.m2_dir, split)
    assert ev1 == ev2, "M1/M2 must be evaluated on the identical event set for paired comparison"
    paired_ci = paired_bootstrap_diff_ci(pred1, pred2, true1, n_boot=args.n_boot, seed=0)
    print(f"  M1 vs M2 paired bootstrap (rmse_M1 - rmse_M2): mean={paired_ci['mean_diff_db']:+.3f} dB "
          f"95% CI=[{paired_ci['ci_lo']:+.3f}, {paired_ci['ci_hi']:+.3f}]  significant={paired_ci['significant']}")

    out = {
        "split": split, "macro_rmse_db": macro, "oracle_o0_modeb_test_rmse_db": oracle_o0_modeb_test,
        "m1_vs_m2_paired_bootstrap": paired_ci,
        "run_dirs": {"m0": str(args.m0_dir), "m1": str(args.m1_dir), "m2": str(args.m2_dir), "oracle": str(oracle_dir)},
    }
    out_path = args.out or (args.m2_dir.parent / f"comparison_{split}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
