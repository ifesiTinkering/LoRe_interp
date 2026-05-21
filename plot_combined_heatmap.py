"""
Reproduce Rachel's Figure 2 (top) and add a larger PRISM-only heatmap below.

Top panel: re-uses her ``fig_user_weights_grid`` from ``experiments/figs.py``
unchanged (we just monkey-patch her ``MODELS_DIR`` and ``FIGS_DIR`` so the
function reads the checkpoints she ships in the repo instead of her internal
NAS path).

Bottom panel: all 182 PRISM seen users from her ``W_seen_K8.pt``, sorted by
their argmax basis, drawn in the same Blues palette and same per-row
normalisation she uses, so the two panels are visually consistent.

Run locally:
    python plot_combined_heatmap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from matplotlib.colors import Normalize
from PIL import Image

HERE        = Path(__file__).resolve().parent
REPO        = HERE / "apa"
CHECKPOINTS = REPO / "experiments" / "checkpoints"
OUT_DIR     = HERE / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Render the top panel by calling her function
# ---------------------------------------------------------------------------

# Patch her config so her code reads the bundled checkpoints instead of
# /nas/XXXX-9/.../models. Do this BEFORE importing experiments.figs.
sys.path.insert(0, str(REPO))
import apa.config as _cfg
_cfg.MODELS_DIR = CHECKPOINTS

# Force her figs module to write into our directory, not her experiments/figs/
import importlib
_figs = importlib.import_module("experiments.figs")
_figs.FIGS_DIR = OUT_DIR
_figs.FIGS_DIR.mkdir(parents=True, exist_ok=True)

top_pdf = _figs.fig_user_weights_grid(K=8, save=True)
top_png = top_pdf.with_suffix(".png")
print(f"top panel saved to {top_png}")

# ---------------------------------------------------------------------------
# Render the bottom panel: all 182 PRISM users from W_seen_K8
# ---------------------------------------------------------------------------

W = torch.load(CHECKPOINTS / "W_seen_K8.pt", map_location="cpu", weights_only=False)
W = W.float().numpy()                                          # (182, 8)
n_users, K = W.shape

# Sort users by their dominant basis so users sharing a dominant basis cluster
order = np.lexsort(keys=(-np.max(np.abs(W), axis=1), np.argmax(np.abs(W), axis=1)))
W_sorted = W[order]

# Same per-row normalisation Rachel uses in fig_user_weights_grid: each row is
# scaled to [0, 1] by its own max |w|, and 0 is mapped to a pale tint.
PALE_FLOOR = 0.12
norm = Normalize(vmin=-PALE_FLOOR / (1 - PALE_FLOOR), vmax=1.0)

row_scale = np.maximum(np.max(np.abs(W_sorted), axis=1, keepdims=True), 1e-12)
cell_colors = mpl.colormaps["Blues"](norm(np.abs(W_sorted) / row_scale))

fig_bot, ax_bot = plt.subplots(figsize=(7, 14))
ax_bot.imshow(cell_colors, aspect="auto", interpolation="nearest")
ax_bot.set_xticks(range(K))
ax_bot.set_xticklabels([f"$V_{{{i}}}$" for i in range(K)])
ax_bot.set_yticks([])
ax_bot.set_xlabel("Basis function")
ax_bot.set_title(
    f"All {n_users} PRISM seen users (W_seen_K8.pt), sorted by dominant basis",
    pad=10,
)
ax_bot.tick_params(axis="both", which="both", length=0)
for spine in ax_bot.spines.values():
    spine.set_visible(False)

# Add a thin black line between argmax bands so the 8 clusters are visible
argmax_sorted = np.argmax(np.abs(W_sorted), axis=1)
boundaries = np.where(np.diff(argmax_sorted) != 0)[0] + 1
for b in boundaries:
    ax_bot.axhline(b - 0.5, color="black", linewidth=0.5)

fig_bot.tight_layout()
bot_png = OUT_DIR / "bottom_all_prism.png"
fig_bot.savefig(bot_png, bbox_inches="tight", dpi=200)
plt.close(fig_bot)
print(f"bottom panel saved to {bot_png}")

# ---------------------------------------------------------------------------
# Stack the two panels into one PNG
# ---------------------------------------------------------------------------

top_img = Image.open(top_png).convert("RGBA")
bot_img = Image.open(bot_png).convert("RGBA")

# Make widths equal (scale narrower one up to wider one's width)
W_target = max(top_img.width, bot_img.width)
def _resize_to_width(img, w_target):
    if img.width == w_target:
        return img
    h_new = round(img.height * w_target / img.width)
    return img.resize((w_target, h_new), Image.LANCZOS)
top_img = _resize_to_width(top_img, W_target)
bot_img = _resize_to_width(bot_img, W_target)

combined = Image.new("RGBA", (W_target, top_img.height + bot_img.height), (255, 255, 255, 255))
combined.paste(top_img, (0, 0))
combined.paste(bot_img, (0, top_img.height))
combined_png = OUT_DIR / "combined_heatmap.png"
combined.convert("RGB").save(combined_png, dpi=(200, 200))
print(f"combined figure saved to {combined_png}")
