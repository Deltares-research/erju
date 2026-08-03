"""publish_m4_m2_ensemble_results.py
======================================
Freezes the M4+M2 two-seed (42+43) dB-space ensemble blend experiment and
produces publication-ready outputs. Read-only: does not train, tune, or
modify any model, checkpoint, or existing prediction parquet file.

Reads the already-saved ENSEMBLE_M4_M2_seed_blend/{predictions_val,
predictions_test,comparison}.json produced by analyse_m4_m2_seed_ensemble_blend.py
and the canonical seed-42/43 predictions of M0/M1/M2/M3/M4.

Primary comparison table uses consistent model units: M0 and M3 are
single-seed (seed42, no seed-43 run exists for either), M1/M2/M4 are
two-seed (42+43) dB-space prediction ensembles -- matching the units that
feed the final M4+M2 blend. A secondary table reports the individual-seed
results plus mean/std for the two-seed models.

Writes (all under ENSEMBLE_M4_M2_seed_blend/, nothing outside it, nothing
inside model checkpoint directories):
  - manifest.json                     reproducibility manifest
  - final_results.json                short machine-readable summary
  - comparison_table_models.csv        primary table (consistent units)
  - comparison_table_seeds.csv         secondary table (per-seed + mean/std)
  - comparison_table_bootstrap.csv     paired-bootstrap comparison table
  - per_band_rmse_comparison.png
  - per_sensor_rmse_comparison.png
  - measured_vs_predicted_test.png     (final blend)
  - amplitude_compression_test.png     (final blend)
  - representative_events_test.png     (final blend, low/medium/high)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse_m4_m2_seed_ensemble_blend import OUT_DIR, PATHS, ROOT, event_mean_total_rms, load, seed_ensemble_db, to_3d
from src.ml.linec_multisensor.metrics import (
    compute_full_metrics, plot_amplitude_compression, plot_measured_vs_predicted,
    plot_representative_events, to_jsonable,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


BASELINE_DIRS = {
    "M0": ROOT / "M0" / "M0_seed42_20260803_041551",
    "M1": ROOT / "M1" / "M1_seed42_20260803_050054",
    "M2": ROOT / "M2" / "M2_seed42_20260803_050057",
    "M3": ROOT / "M3" / "M3_seed42_20260803_055820",
    "M4": ROOT / "M4" / "M4_seed42_20260803_064511",
}
GIT_COMMITS = {
    "M0": "c8aaa23fb1a036f9dec5f92fc4ddd6e8a98cd484",
    "M1": "600c902e9f50b13306440dd3e72f5c649305a1a8",
    "M2": "600c902e9f50b13306440dd3e72f5c649305a1a8",
    "M3": "cd708906f227cabb6d8eb0037d9734c6830d86ae",
    "M4": "225e219546db2d45674d39d8a227ff5c839509b5",
}
SEEDS = {"M0": [42], "M1": [42, 43], "M2": [42, 43], "M3": [42], "M4": [42, 43]}
SPLIT_SIZES = {"train": 1103, "val": 254, "test": 340}
BAND_NOMINAL = np.array([1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5,
                          16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0])

NOTES = [
    "The ensemble weight was selected exclusively on validation and applied unchanged to "
    "test. Earlier test evaluations informed the broader model development sequence, so "
    "the final test result is an internal benchmark, not an estimate from a completely "
    "untouched prospective holdout.",
    "Pooled (global) total-RMS R2 and event-mean total-RMS R2 measure different things: "
    "pooled R2 scores every (event,sensor) pair independently, while event-mean R2 first "
    "averages the 5 sensors per event and scores across events. The two can diverge "
    "substantially; neither supersedes the other.",
    "O0 Mode B estimates the event reference spectrum from four measured accelerometer "
    "sensors and predicts the held-out fifth sensor. It is a target-informed, "
    "non-deployable oracle reference, not a strict upper bound.",
    "No model was trained or tuned to produce this blend; all results are derived "
    "read-only from the already-completed M0-M4 training runs' saved predictions.",
]


def model_metrics(true_3d: np.ndarray, pred_3d: np.ndarray):
    m = compute_full_metrics(pred_3d, true_3d, np.zeros(true_3d.shape[:2]), band_nominal=BAND_NOMINAL)
    ev = event_mean_total_rms(true_3d, pred_3d)
    return m, ev


def metrics_row(label: str, df: pd.DataFrame) -> tuple[dict, dict, dict]:
    true_3d, _, _ = to_3d(df, "target_db")
    pred_3d, _, _ = to_3d(df, "predicted_db")
    m, ev = model_metrics(true_3d, pred_3d)
    row = {
        "model": label, "macro_rmse_db": m["macro_rmse_db"], "macro_mae_db": m["macro_mae_db"],
        "macro_r2": m["macro_r2"], "total_rms_global_r2": m["total_rms"]["r2"],
        "total_rms_global_rmse_mms": m["total_rms"]["rmse_mms"],
        "total_rms_event_mean_r2": ev["r2"], "total_rms_event_mean_rmse_mms": ev["rmse_mms"],
        "amplitude_compression_slope": m["amplitude_compression"]["slope"],
        "shape_only_rmse_db": m["shape_only_rmse_db"],
        "strong_event_macro_rmse_db": m["strong_event_metrics"]["macro_rmse_db"],
    }
    return row, m["per_band_rmse_db"], {k: v["rmse_db"] for k, v in m["per_sensor"].items()}


def main() -> None:
    # ── primary table: consistent units (M0/M3 single-seed42, M1/M2/M4 seed42+43 ensembles) ─
    primary_labels = {
        "M0_single_seed42": pd.read_parquet(BASELINE_DIRS["M0"] / "predictions_test.parquet"),
        "M1_seed_ensemble": seed_ensemble_db("M1_s42", "M1_s43", "test"),
        "M2_seed_ensemble": seed_ensemble_db("M2_s42", "M2_s43", "test"),
        "M3_single_seed42": pd.read_parquet(BASELINE_DIRS["M3"] / "predictions_test.parquet"),
        "M4_seed_ensemble": seed_ensemble_db("M4_s42", "M4_s43", "test"),
    }
    rows, per_band, per_sensor = [], {}, {}
    for label, df in primary_labels.items():
        row, pb, ps = metrics_row(label, df)
        rows.append(row); per_band[label] = pb; per_sensor[label] = ps

    blend_df = pd.read_parquet(OUT_DIR / "predictions_test.parquet")
    blend_true_3d, _, blend_bands = to_3d(blend_df, "target_db")
    blend_pred_3d, _, _ = to_3d(blend_df, "predicted_db")
    row, pb, ps = metrics_row("Final_blend", blend_df)
    rows.append(row); per_band["Final_blend"] = pb; per_sensor["Final_blend"] = ps

    table1 = pd.DataFrame(rows).set_index("model")
    table1.to_csv(OUT_DIR / "comparison_table_models.csv")
    print("=== Table 1 (primary, consistent units): single-seed M0/M3, seed42+43 ensembles M1/M2/M4, final blend (test) ===")
    print(table1.round(4).to_string())

    # ── secondary table: individual seeds + mean/std where two seeds exist ──
    seed_rows = []
    single_seed_dfs = {"M0_s42": primary_labels["M0_single_seed42"], "M3_s42": primary_labels["M3_single_seed42"]}
    for label, df in single_seed_dfs.items():
        row, _, _ = metrics_row(label, df)
        seed_rows.append(row)

    numeric_cols = None
    for model_name, (name_a, name_b) in (("M1", ("M1_s42", "M1_s43")), ("M2", ("M2_s42", "M2_s43")),
                                          ("M4", ("M4_s42", "M4_s43"))):
        r_a, _, _ = metrics_row(name_a, load(name_a, "test"))
        r_b, _, _ = metrics_row(name_b, load(name_b, "test"))
        seed_rows.extend([r_a, r_b])
        numeric_cols = [c for c in r_a if c != "model"]
        mean_row = {"model": f"{model_name}_mean", **{c: float(np.mean([r_a[c], r_b[c]])) for c in numeric_cols}}
        std_row = {"model": f"{model_name}_std", **{c: float(np.std([r_a[c], r_b[c]], ddof=1)) for c in numeric_cols}}
        seed_rows.extend([mean_row, std_row])

    table_seeds = pd.DataFrame(seed_rows).set_index("model")
    table_seeds.to_csv(OUT_DIR / "comparison_table_seeds.csv")
    print("\n=== Secondary table: individual seeds + mean/std (test) ===")
    print(table_seeds.round(4).to_string())

    # ── table 2: paired-bootstrap comparison (reuse already-computed results) ─
    comparison = json.loads((OUT_DIR / "comparison.json").read_text())
    boot = comparison["bootstrap_comparisons_test"]
    boot_rows = []
    for label, r in boot.items():
        boot_rows.append({
            "comparator": label.replace("vs_", ""), "mean_diff_db": r["mean_diff_db"],
            "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"],
            "frac_final_blend_better": r["frac_final_blend_better"],
            "significant": bool(r["ci95_lo"] > 0 or r["ci95_hi"] < 0),
        })
    table2 = pd.DataFrame(boot_rows).set_index("comparator")
    table2.to_csv(OUT_DIR / "comparison_table_bootstrap.csv")
    print("\n=== Table 2: paired event-level bootstrap (final blend - comparator) ===")
    print(table2.round(4).to_string())

    # ── figure: per-band RMSE comparison ─────────────────────────────────────
    models = list(primary_labels) + ["Final_blend"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in models:
        ax.plot(BAND_NOMINAL, per_band[name], "o-", lw=1.3, ms=3, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("nominal band (Hz)"); ax.set_ylabel("RMSE (dB)")
    ax.set_title("Per-band macro RMSE comparison (test)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_band_rmse_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── figure: per-sensor RMSE comparison ───────────────────────────────────
    sensors = list(per_sensor["M0_single_seed42"].keys())
    x = np.arange(len(sensors)); width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(models):
        vals = [per_sensor[name][s] for s in sensors]
        ax.bar(x + i * width, vals, width=width, label=name)
    ax.set_xticks(x + width * (len(models) - 1) / 2); ax.set_xticklabels(sensors)
    ax.set_ylabel("RMSE (dB)"); ax.set_title("Per-sensor macro RMSE comparison (test)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_sensor_rmse_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── final-blend diagnostic figures (reuse existing plotting functions) ──
    plot_measured_vs_predicted(blend_true_3d, blend_pred_3d, OUT_DIR, "test")
    plot_amplitude_compression(blend_true_3d, blend_pred_3d, OUT_DIR, "test")
    plot_representative_events(blend_true_3d, blend_pred_3d, blend_bands, OUT_DIR, "test")

    # ── reproducibility manifest ──────────────────────────────────────────────
    manifest = {
        "experiment": "M4 (seed42+43) + M2 (seed42+43) dB-space seed-ensemble blend",
        "source_run_directories": {
            **{k: str(v) for k, v in PATHS.items()},  # M0, M1_s42/s43, M2_s42/s43, M4_s42/s43
            "M3_s42": str(BASELINE_DIRS["M3"]),
            "final_blend_output": str(OUT_DIR),
        },
        "git_commits": GIT_COMMITS,
        "seeds": SEEDS,
        "selected_validation_weight_w_on_M4_ensemble": comparison["selected_weight_w_on_M4_ensemble"],
        "validation_weight_sweep": comparison["val_weight_sweep"],
        "split_sizes": SPLIT_SIZES,
        "prediction_averaging_convention": "arithmetic mean of predicted_db in dB space "
                                            "(not power/linear domain), computed row-wise "
                                            "after merging seeds on "
                                            "[event_id, sensor_id, band_nominal_hz, track_number, distance_m]",
        "bootstrap": {"seed": comparison["bootstrap_seed"], "n_boot": comparison["n_boot"],
                      "method": "paired event-level resampling with replacement"},
        "publication_script_git_commit": git_commit(),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2))

    # ── final_results.json ───────────────────────────────────────────────────
    final_results = {
        "experiment": "M4+M2 two-seed dB-space ensemble blend",
        "status": "frozen -- no further training or tuning",
        "selected_weight_w_on_M4_ensemble": comparison["selected_weight_w_on_M4_ensemble"],
        "test_metrics_final_blend": comparison["test"],
        "test_macro_rmse_db_by_model": {r["model"]: r["macro_rmse_db"] for r in rows},
        "bootstrap_comparisons_test": boot,
        "notes": NOTES,
    }
    (OUT_DIR / "final_results.json").write_text(json.dumps(to_jsonable(final_results), indent=2))

    print(f"\nSaved manifest.json, final_results.json, comparison_table_models.csv, "
          f"comparison_table_seeds.csv, comparison_table_bootstrap.csv, per_band_rmse_comparison.png, "
          f"per_sensor_rmse_comparison.png, measured_vs_predicted_test.png, "
          f"amplitude_compression_test.png, representative_events_test.png -> {OUT_DIR}")


if __name__ == "__main__":
    main()
