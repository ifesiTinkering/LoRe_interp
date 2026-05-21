"""
Reproduce Rachel's Figure 2 (top) and add a larger PRISM-only heatmap below.

Top panel: re-uses her ``fig_user_weights_grid`` from ``experiments/figs.py``
unchanged (we just monkey-patch her ``MODELS_DIR`` and ``FIGS_DIR`` so the
function reads the checkpoints she ships in the repo instead of her internal
NAS path).

Bottom panel: all 182 PRISM seen users from her ``W_seen_K8.pt``, grouped
by each user's top stated preference (from PRISM ``survey.jsonl``). User
ids are recovered by calling her ``apa.load_prism`` functions to replay
her seed=123 split, then alphabetically sorting (which matches the row
ordering her ``group_embeddings_by_user`` produces). Same Blues palette
and per-row normalisation she uses, so the two panels look consistent.

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
# Bottom panel: all 182 PRISM users from W_seen_K8, grouped by stated preference
#
# W_seen_K8.pt has 182 anonymous rows. We call her own load_prism functions
# to download PRISM, parse it, and run her deterministic split — then sort
# the seen_user_ids alphabetically (the row ordering her
# group_embeddings_by_user produces) so row i of W corresponds to that user.
# ---------------------------------------------------------------------------

from collections import Counter
from apa.load_prism import (
    _download_prism_raw,
    _parse_prism_data,
    _split_users_and_dialogs,
)

W = torch.load(CHECKPOINTS / "W_seen_K8.pt", map_location="cpu", weights_only=False)
W = W.float().numpy()                                          # (182, 8)
n_users, K = W.shape

PRISM_CACHE = HERE / "prism_cache"
PRISM_CACHE.mkdir(exist_ok=True)

conv_path, survey_path = _download_prism_raw(PRISM_CACHE)
user_data, _dialog_data = _parse_prism_data(conv_path, survey_path)
split_ids = _split_users_and_dialogs(user_data, seed=123)
seen_user_ids_sorted = sorted(split_ids["seen_user_ids"])

assert len(seen_user_ids_sorted) == n_users, (
    f"Got {len(seen_user_ids_sorted)} seen users from her split but W has "
    f"{n_users} rows. Row ordering assumption is wrong."
)

# Group W rows by stated preference (largest group first)
row_pref = [
    (user_data[uid].demographics.preference[0]
     if user_data[uid].demographics.preference else "unknown")
    for uid in seen_user_ids_sorted
]
pref_counts = Counter(row_pref)
pref_order = [p for p, _ in pref_counts.most_common()]

group_indices: list[int] = []
group_boundaries: list[int] = []
group_labels: list[str] = []
for pref in pref_order:
    idxs = [i for i, p in enumerate(row_pref) if p == pref]
    group_indices.extend(idxs)
    group_boundaries.append(len(group_indices))
    group_labels.append(f"{pref} (n={len(idxs)})")

W_grouped = W[group_indices]

# Same per-row normalisation Rachel uses in fig_user_weights_grid
PALE_FLOOR = 0.12
norm = Normalize(vmin=-PALE_FLOOR / (1 - PALE_FLOOR), vmax=1.0)
row_scale = np.maximum(np.max(np.abs(W_grouped), axis=1, keepdims=True), 1e-12)
cell_colors = mpl.colormaps["Blues"](norm(np.abs(W_grouped) / row_scale))

fig_bot, ax_bot = plt.subplots(figsize=(8, 14))
ax_bot.imshow(cell_colors, aspect="auto", interpolation="nearest")
ax_bot.set_xticks(range(K))
ax_bot.set_xticklabels([f"$V_{{{i}}}$" for i in range(K)])
ax_bot.set_yticks([])
ax_bot.set_xlabel("Basis function")
ax_bot.set_title(
    f"All {n_users} PRISM seen users, grouped by top stated preference",
    pad=10,
)
ax_bot.tick_params(axis="both", which="both", length=0)
for spine in ax_bot.spines.values():
    spine.set_visible(False)

# Black lines between groups + group labels on the left
prev_boundary = 0
for label, b in zip(group_labels, group_boundaries):
    if b != group_boundaries[-1]:
        ax_bot.axhline(b - 0.5, color="black", linewidth=1.0)
    mid = (prev_boundary + b) / 2 - 0.5
    ax_bot.text(-0.7, mid, label, ha="right", va="center", fontsize=9)
    prev_boundary = b

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
