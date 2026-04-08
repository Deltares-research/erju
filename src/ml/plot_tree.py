"""Plot a single decision tree from the XGBoost v003 model.

Pure matplotlib — no Graphviz required.
Edit TREE_INDEX / MAX_DEPTH to control which tree and how deep to draw.
Output saved to the model's plots/ folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb

# =============================================================================
# USER INPUT
# =============================================================================

BUILD_DIR = Path(r"P:\11210978-erju-ai\holten_models\xgb_v003_20260406_194334")
TREE_INDEX = 0  # which tree (0 = first boosting round)
MAX_DEPTH = 6  # max depth to draw; full depth=6 is very wide (use 3 or 4)
DPI = 180

# =============================================================================

# ---- colour scheme ----------------------------------------------------------
CLR_SPLIT = "#4878CF"  # internal node
CLR_LEAF = "#6ACC65"  # leaf node
CLR_EDGE = "#888888"

# ---- node layout parameters -------------------------------------------------
H_GAP = 1.0  # horizontal spacing multiplier
V_STEP = 2.0  # vertical distance per depth level


def _parse_tree(node: dict, depth: int, max_depth: int) -> dict:
    """Recursively parse the JSON tree, capping at max_depth."""
    result = {
        "id": node["nodeid"],
        "depth": depth,
        "is_leaf": "leaf" in node,
    }
    if "leaf" in node:
        result["value"] = node["leaf"]
    else:
        result["split"] = node.get("split", "?")
        result["threshold"] = node.get("split_condition", 0.0)
        result["yes"] = node.get("yes")
        result["no"] = node.get("no")
        children = {c["nodeid"]: c for c in node.get("children", [])}
        if depth < max_depth:
            result["left"] = _parse_tree(children[result["yes"]], depth + 1, max_depth)
            result["right"] = _parse_tree(children[result["no"]], depth + 1, max_depth)
        else:
            # Truncate — show as pseudo-leaf
            result["is_leaf"] = True
            result["value"] = 0.0
            result["truncated"] = True
    return result


def _assign_positions(node: dict, depth: int = 0, counter: list | None = None) -> None:
    """In-order traversal to assign x positions (leaves get integer slots)."""
    if counter is None:
        counter = [0]
    if node["is_leaf"]:
        node["x"] = counter[0]
        counter[0] += 1
    else:
        _assign_positions(node["left"], depth + 1, counter)
        node["x"] = counter[0]
        counter[0] += 1
        _assign_positions(node["right"], depth + 1, counter)
    node["y"] = -depth * V_STEP


def _draw_node(ax, node: dict, box_w: float = 1.6, box_h: float = 0.7) -> None:
    x, y = node["x"] * H_GAP, node["y"]
    is_leaf = node["is_leaf"]
    color = CLR_LEAF if is_leaf else CLR_SPLIT

    fancy = mpatches.FancyBboxPatch(
        (x - box_w / 2, y - box_h / 2),
        box_w,
        box_h,
        boxstyle="round,pad=0.05",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color + "33",  # 20% opacity fill
        zorder=3,
    )
    ax.add_patch(fancy)

    if is_leaf:
        val = node.get("value", 0.0)
        lbl = f"leaf\n{val:.4f}" if not node.get("truncated") else "…"
    else:
        feat = node["split"]
        thr = node["threshold"]
        lbl = f"{feat}\n< {thr:.3g}"

    ax.text(
        x,
        y,
        lbl,
        ha="center",
        va="center",
        fontsize=6.5,
        zorder=4,
        wrap=True,
    )


def _draw_edges(ax, node: dict, box_h: float = 0.7) -> None:
    if node["is_leaf"]:
        return
    px, py = node["x"] * H_GAP, node["y"]
    for child, label in [(node["left"], "yes"), (node["right"], "no")]:
        cx, cy = child["x"] * H_GAP, child["y"]
        ax.annotate(
            "",
            xy=(cx, cy + box_h / 2),
            xytext=(px, py - box_h / 2),
            arrowprops=dict(arrowstyle="-|>", color=CLR_EDGE, lw=0.8),
            zorder=2,
        )
        mx, my = (px + cx) / 2, (py + cy) / 2
        ax.text(
            mx + 0.05, my, label, fontsize=6, color=CLR_EDGE, ha="left", va="center"
        )
        _draw_edges(ax, child, box_h)


def _walk(node: dict, fn):
    fn(node)
    if not node["is_leaf"]:
        _walk(node["left"], fn)
        _walk(node["right"], fn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_DIR = BUILD_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = xgb.XGBRegressor()
model.load_model(str(BUILD_DIR / "model_final.ubj"))

raw = model.get_booster().get_dump(dump_format="json")
tree_json = json.loads(raw[TREE_INDEX])

tree = _parse_tree(tree_json, depth=0, max_depth=MAX_DEPTH)
_assign_positions(tree)

# Compute canvas size from node positions
all_x, all_y = [], []
_walk(tree, lambda n: (all_x.append(n["x"] * H_GAP), all_y.append(n["y"])))
width = max(all_x) - min(all_x) + 4
height = max(all_y) - min(all_y) + 3

fig, ax = plt.subplots(figsize=(max(width, 10), max(height, 6)))
ax.set_xlim(min(all_x) - 2, max(all_x) + 2)
ax.set_ylim(min(all_y) - 1.5, max(all_y) + 1.5)
ax.axis("off")

_walk(tree, lambda n: _draw_edges(ax, n))
_walk(tree, lambda n: _draw_node(ax, n))

ax.set_title(
    f"XGBoost v003 — Tree #{TREE_INDEX}  (depth capped at {MAX_DEPTH})",
    fontsize=11,
    pad=8,
)

out = OUTPUT_DIR / f"tree_{TREE_INDEX:04d}_d{MAX_DEPTH}.png"
fig.tight_layout()
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
