"""
Extract diagnostic report from P3_corrected_n model.

Usage:
    python extract_curveprior_diagnostics.py \
        --model_dir /p/11210978-erju-ai/holten_models/cnn_curveprior_linec_v001_vP3_fit_corrected_20260627_211829 \
        --output_report CURVEPRIOR_P3_DIAGNOSTICS_20260627.md
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def extract_diagnostics(model_dir, output_report):
    """Extract and format diagnostic metrics from model output."""
    
    model_dir = Path(model_dir)
    
    # Load predictions and metadata
    predictions_file = model_dir / "predictions.parquet"
    config_file = model_dir / "config_snapshot.json"
    metrics_file = model_dir / "metrics.json"
    
    if not predictions_file.exists():
        print(f"ERROR: {predictions_file} not found")
        return
    
    print(f"Loading predictions from {predictions_file}")
    preds_df = pd.read_parquet(predictions_file)
    
    # Load config and metrics
    config = {}
    metrics = {}
    
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    
    # Build diagnostics report
    report_lines = [
        "# P3_corrected_n Model Diagnostics Report",
        f"**Generated:** {pd.Timestamp.now()}",
        f"**Model Directory:** {model_dir}",
        "",
        "## Configuration",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Summary Metrics",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
        "",
        "## Detailed Per-Sensor Analysis",
        "",
    ]
    
    # Compute per-sensor metrics if predictions available
    if len(preds_df) > 0:
        report_lines.extend([
            "### Combined Test Set",
            f"- Total samples: {len(preds_df)}",
            f"- Unique events: {preds_df['event_id'].nunique() if 'event_id' in preds_df.columns else 'N/A'}",
            f"- Unique sensors: {preds_df['sensor_name'].nunique() if 'sensor_name' in preds_df.columns else 'N/A'}",
            "",
        ])
        
        # Per-sensor metrics
        if 'sensor_name' in preds_df.columns and 'pred_log' in preds_df.columns:
            report_lines.append("### Per-Sensor RMSE and Bias")
            report_lines.append("| Sensor | Count | RMSE(log) | RMSE(PGV) | Mean Bias(log) |")
            report_lines.append("|--------|-------|-----------|-----------|----------------|")
            
            for sensor in sorted(preds_df['sensor_name'].unique()):
                sensor_data = preds_df[preds_df['sensor_name'] == sensor]
                if 'target_log' in sensor_data.columns:
                    rmse_log = np.sqrt(np.mean((sensor_data['pred_log'] - sensor_data['target_log']) ** 2))
                    bias_log = np.mean(sensor_data['pred_log'] - sensor_data['target_log'])
                    if 'target_pgv' in sensor_data.columns:
                        rmse_pgv = np.sqrt(np.mean((sensor_data['pred_pgv'] - sensor_data['target_pgv']) ** 2))
                    else:
                        rmse_pgv = np.nan
                    report_lines.append(
                        f"| {sensor} | {len(sensor_data)} | {rmse_log:.4f} | {rmse_pgv:.4f} | {bias_log:.4f} |"
                    )
            report_lines.append("")
        
        # Epsilon analysis if available
        if 'epsilon_hat' in preds_df.columns:
            report_lines.extend([
                "### Residual (epsilon) Analysis",
                f"- Mean epsilon: {preds_df['epsilon_hat'].mean():.6f}",
                f"- Std epsilon: {preds_df['epsilon_hat'].std():.6f}",
                f"- Min epsilon: {preds_df['epsilon_hat'].min():.6f}",
                f"- Max epsilon: {preds_df['epsilon_hat'].max():.6f}",
                f"- RMS epsilon: {np.sqrt(np.mean(preds_df['epsilon_hat'] ** 2)):.6f}",
                "",
            ])
        
        # PGV > 4 mm/s subset
        if 'target_pgv' in preds_df.columns:
            high_pgv = preds_df[preds_df['target_pgv'] > 4.0]
            if len(high_pgv) > 0:
                report_lines.extend([
                    "### High-PGV Subset (PGV > 4.0 mm/s)",
                    f"- Count: {len(high_pgv)} ({100*len(high_pgv)/len(preds_df):.1f}%)",
                    f"- RMSE(log): {np.sqrt(np.mean((high_pgv['pred_log'] - high_pgv['target_log']) ** 2)):.4f}",
                    f"- RMSE(PGV): {np.sqrt(np.mean((high_pgv['pred_pgv'] - high_pgv['target_pgv']) ** 2)):.4f}",
                    f"- Bias(log): {np.mean(high_pgv['pred_log'] - high_pgv['target_log']):.4f}",
                    "",
                ])
        
        # Track-wise analysis if available
        if 'track' in preds_df.columns:
            report_lines.append("### Per-Track Analysis")
            for track in sorted(preds_df['track'].unique()):
                track_data = preds_df[preds_df['track'] == track]
                if 'target_log' in track_data.columns:
                    rmse_log = np.sqrt(np.mean((track_data['pred_log'] - track_data['target_log']) ** 2))
                    r2 = 1 - np.sum((track_data['pred_log'] - track_data['target_log']) ** 2) / np.sum((track_data['target_log'] - track_data['target_log'].mean()) ** 2)
                    report_lines.append(f"- Track {track}: RMSE(log)={rmse_log:.4f}, R²={r2:.4f}, N={len(track_data)}")
            report_lines.append("")
    
    # Key findings
    report_lines.extend([
        "## Key Findings",
        "",
        "### Baseline Performance",
        f"- RMSE(log): {metrics.get('test_rmse_log', 'N/A')}",
        f"- R²(log): {metrics.get('test_r2_log', 'N/A')}",
        f"- RMSE(PGV): {metrics.get('test_rmse_pgv', 'N/A')} mm/s",
        f"- Best epoch: {metrics.get('best_epoch', 'N/A')}",
        "",
        "### Fitted Attenuation Exponents",
        f"- Track 1 n: {config.get('features', {}).get('n_track1', 'N/A')}",
        f"- Track 2 n: {config.get('features', {}).get('n_track2', 'N/A')}",
        "",
        "### Model Configuration",
        f"- Variant: {config.get('train', {}).get('variant', 'N/A')}",
        f"- n_mode: {config.get('train', {}).get('n_mode', 'N/A')}",
        f"- lambda_residual: {config.get('train', {}).get('lambda_residual', 'N/A')}",
        f"- MP4 weight: {config.get('train', {}).get('mp4_weight', 1.0)}",
        "",
        "## Baseline Established",
        f"This report documents the baseline P3_corrected_n performance.",
        f"Compare query-conditioned models Q1/Q2/Q3/Q4 against these metrics.",
        "",
    ])
    
    # Write report
    report_text = "\n".join(report_lines)
    
    with open(output_report, 'w') as f:
        f.write(report_text)
    
    print(f"Diagnostics report saved to {output_report}")
    print("\n" + "="*80)
    print(report_text)
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract P3 diagnostics")
    parser.add_argument("--model_dir", required=True, help="Path to P3 model directory")
    parser.add_argument("--output_report", default="CURVEPRIOR_P3_DIAGNOSTICS_20260627.md",
                        help="Output report filename")
    args = parser.parse_args()
    
    extract_diagnostics(args.model_dir, args.output_report)
