"""plot_mlp_architecture.py — Visualise MLP architecture from a build folder.

Reads config_snapshot.json from any mlp build folder and draws:
  - Left panel: network diagram (nodes per layer, annotated with sizes)
  - Right panel: layer-by-layer text table (type, shape, params)

Run from project root:
    python src/ml/mlp/plot_mlp_architecture.py

Edit BUILD_DIR below, or set to None to show all 5 builds in one figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
# Set to a single build folder, or None to plot all builds side-by-side
BUILD_DIR: Path | None = None

MODELS_ROOT = Path(r"P:\11210978-erju-ai\holten_models")

# All MLP builds in order
ALL_BUILDS = [
    "mlp_v001_20260408_180508",
    "mlp_v002_20260408_181957",
    "mlp_v003_20260408_182326",
    "mlp_v004_20260408_183030",
    "mlp_v005_20260408_183514",
]

OUTPUT_DIR = MODELS_ROOT / "architecture_plots"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Architecture parser ───────────────────────────────────────────────────────


def parse_architecture(cfg: dict, build_dir: Path) -> dict:
    """Extract architecture info from a config_snapshot dict."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    fe_cfg = cfg.get("fe", {})
    summ_path = build_dir / "summary.json"
    n_params = 0
    test_rmse = None
    n_features = 0
    if summ_path.exists():
        s = json.load(open(summ_path))
        n_params = s.get("n_params", 0)
        test_rmse = s.get("test_rmse_mms")
        n_features = s.get("n_features", 0)

    hidden = model_cfg.get("hidden_sizes", [64])
    layer_sizes = [n_features] + hidden + [1]

    return {
        "version": cfg.get("version_name", "?"),
        "layer_sizes": layer_sizes,
        "activation": model_cfg.get("activation", "relu"),
        "dropout": model_cfg.get("dropout", 0.0),
        "batch_norm": model_cfg.get("batch_norm", False),
        "lr": train_cfg.get("learning_rate", 1e-3),
        "lr_scheduler": train_cfg.get("lr_scheduler", False),
        "log_target": fe_cfg.get("log_transform_target", True),
        "n_params": n_params,
        "test_rmse": test_rmse,
    }


def count_params(layer_sizes: list[int]) -> list[int]:
    """Params per linear layer (weights + biases)."""
    params = []
    for i in range(len(layer_sizes) - 1):
        params.append(layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1])
    return params


# ── Single-architecture plotter ───────────────────────────────────────────────

# Visual constants
NODE_RADIUS = 0.35
MAX_NODES_VIS = 12  # max nodes to draw per layer before switching to "summarised" style
H_SPACING = 3.2  # horizontal gap between layers
V_SPACING = 1.0  # vertical gap between nodes

LAYER_COLORS = {
    "input": "#2C3E50",
    "hidden": "#2980B9",
    "output": "#27AE60",
}
EDGE_COLOR = "#BDC3C7"
TEXT_COLOR = "white"
BG_COLOR = "#F8F9FA"


def _draw_layer(
    ax,
    x: float,
    layer_sizes: list[int],
    layer_idx: int,
    n_vis_nodes: int,
    label: str,
    color: str,
    extra_label: str = "",
) -> list[tuple[float, float]]:
    """Draw one layer column, return list of (x, y) node centres."""
    n = layer_sizes[layer_idx]
    n_vis = min(n, n_vis_nodes)
    summarised = n > n_vis_nodes

    # Centre the column vertically
    total_h = (n_vis - 1) * V_SPACING
    y_start = total_h / 2

    centres = []
    for i in range(n_vis):
        y = y_start - i * V_SPACING
        if summarised and i == n_vis // 2:
            # draw "..." node
            ax.text(
                x,
                y,
                "⋮",
                ha="center",
                va="center",
                fontsize=14,
                color=color,
                fontweight="bold",
            )
            centres.append((x, y))
        else:
            circle = plt.Circle(
                (x, y),
                NODE_RADIUS,
                facecolor=color,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            ax.add_patch(circle)
            centres.append((x, y))

    # Layer label below
    ax.text(
        x,
        -y_start - V_SPACING * 0.9,
        label,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#2C3E50",
    )
    ax.text(
        x,
        -y_start - V_SPACING * 1.55,
        f"n={n}",
        ha="center",
        va="top",
        fontsize=7,
        color="#7F8C8D",
    )
    if extra_label:
        ax.text(
            x,
            -y_start - V_SPACING * 2.1,
            extra_label,
            ha="center",
            va="top",
            fontsize=6.5,
            color="#E74C3C",
        )

    return centres


def draw_architecture(arch: dict, ax_net: plt.Axes, ax_table: plt.Axes) -> None:
    """Draw network diagram + layer table for one architecture."""
    sizes = arch["layer_sizes"]
    n_layers = len(sizes)
    act = arch["activation"].upper()
    do = arch["dropout"]
    bn = arch["batch_norm"]
    params_per_layer = count_params(sizes)

    ax_net.set_facecolor(BG_COLOR)
    ax_net.set_aspect("equal")
    ax_net.axis("off")

    layer_centres: list[list[tuple[float, float]]] = []

    for li in range(n_layers):
        x = li * H_SPACING
        is_in = li == 0
        is_out = li == n_layers - 1
        color = (
            LAYER_COLORS["input"]
            if is_in
            else (LAYER_COLORS["output"] if is_out else LAYER_COLORS["hidden"])
        )

        if is_in:
            label = "Input"
        elif is_out:
            label = "Output"
        else:
            label = f"Hidden {li}"

        extras = []
        if not is_out and not is_in:
            extras.append(act)
            if bn:
                extras.append("BN")
            if do > 0:
                extras.append(f"Drop {do}")
        extra_str = " | ".join(extras)

        centres = _draw_layer(
            ax_net, x, sizes, li, MAX_NODES_VIS, label, color, extra_str
        )
        layer_centres.append(centres)

    # Draw edges between consecutive layers (skip "..." nodes for clarity)
    for li in range(n_layers - 1):
        for x0, y0 in layer_centres[li]:
            for x1, y1 in layer_centres[li + 1]:
                ax_net.plot(
                    [x0 + NODE_RADIUS, x1 - NODE_RADIUS],
                    [y0, y1],
                    color=EDGE_COLOR,
                    linewidth=0.25,
                    alpha=0.5,
                    zorder=1,
                )

    # Title
    rmse_str = (
        f"  |  Test RMSE = {arch['test_rmse']:.4f} mm/s" if arch["test_rmse"] else ""
    )
    ax_net.set_title(
        f"{arch['version']}   ({arch['n_params']:,} params){rmse_str}",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )

    # ── Layer table (right panel) ─────────────────────────────────────────────
    ax_table.axis("off")
    rows = [["#", "Type", "In", "Out", "Params"]]
    row_idx = 0
    for li in range(n_layers - 1):
        row_idx += 1
        rows.append(
            [
                str(row_idx),
                "Linear",
                str(sizes[li]),
                str(sizes[li + 1]),
                f"{params_per_layer[li]:,}",
            ]
        )
        if li < n_layers - 2:  # not output layer
            if bn:
                row_idx += 1
                rows.append(
                    [
                        str(row_idx),
                        "BatchNorm1d",
                        str(sizes[li + 1]),
                        str(sizes[li + 1]),
                        "—",
                    ]
                )
            row_idx += 1
            rows.append([str(row_idx), act, "—", "—", "—"])
            if do > 0:
                row_idx += 1
                rows.append([str(row_idx), f"Dropout({do})", "—", "—", "—"])

    rows.append(["", "━━━ Total ━━━", "", "", f"{arch['n_params']:,}"])

    tbl = ax_table.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.6)

    for j in range(5):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)):
        clr = "#EBF5FB" if i % 2 == 0 else "white"
        for j in range(5):
            tbl[i, j].set_facecolor(clr)

    # Hyper-param summary below table
    hp_lines = [
        f"Activation: {arch['activation']}",
        f"Dropout: {arch['dropout']}",
        f"Batch Norm: {arch['batch_norm']}",
        f"LR: {arch['lr']}  |  Scheduler: {arch['lr_scheduler']}",
        f"Log1p target: {arch['log_target']}",
    ]
    ax_table.text(
        0.5,
        0.02,
        "\n".join(hp_lines),
        transform=ax_table.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor="#AED6F1"),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def plot_single(build_dir: Path) -> None:
    cfg_path = build_dir / "config_snapshot.json"
    if not cfg_path.exists():
        print(f"  No config_snapshot.json in {build_dir}")
        return
    cfg = json.load(open(cfg_path))
    arch = parse_architecture(cfg, build_dir)

    fig, (ax_net, ax_table) = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.patch.set_facecolor(BG_COLOR)
    draw_architecture(arch, ax_net, ax_table)
    fig.suptitle("MLP Architecture", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = OUTPUT_DIR / f"{arch['version']}_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_all(builds: list[str]) -> None:
    """One subplot pair per version, stacked vertically."""
    archs = []
    for b in builds:
        p = MODELS_ROOT / b / "config_snapshot.json"
        if p.exists():
            archs.append(parse_architecture(json.load(open(p)), MODELS_ROOT / b))
        else:
            print(f"  Skipping {b} (no config_snapshot.json)")

    n = len(archs)
    fig = plt.figure(figsize=(18, 7 * n))
    fig.patch.set_facecolor(BG_COLOR)

    for i, arch in enumerate(archs):
        ax_net = fig.add_subplot(n, 2, 2 * i + 1)
        ax_table = fig.add_subplot(n, 2, 2 * i + 2)
        draw_architecture(arch, ax_net, ax_table)

    fig.suptitle(
        "MLP Architecture — All Versions", fontsize=15, fontweight="bold", y=1.005
    )
    fig.tight_layout()

    out = OUTPUT_DIR / "all_versions_architecture.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print(f"Output folder: {OUTPUT_DIR}\n")
    if BUILD_DIR is not None:
        print(f"Plotting single build: {BUILD_DIR.name}")
        plot_single(BUILD_DIR)
    else:
        print("Plotting all builds …")
        for b in ALL_BUILDS:
            print(f"  Processing {b} …")
            plot_single(MODELS_ROOT / b)
        print("\nGenerating combined all-versions plot …")
        plot_all(ALL_BUILDS)
    print("\nDone.")
