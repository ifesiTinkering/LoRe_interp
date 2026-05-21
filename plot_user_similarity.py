"""
User-similarity clustermap on Rachel's published W_seen_K8.pt.

Compute the raw dot product between every pair of the 182 PRISM users in
``W_seen_K8.pt``, then reorder rows/columns by hierarchical clustering so
naturally similar users sit next to each other. Magnitudes are preserved
(no L2 normalization), so the score reflects both *direction* (do these
users prefer the same combination of bases?) and *emphasis* (how strongly
do they each weight those bases?). If LoRe discovered discrete user
types we expect visible diagonal blocks; if the user population lives on
a continuous spectrum we expect a smooth gradient with no blocks.

Run locally:
    python plot_user_similarity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist

HERE        = Path(__file__).resolve().parent
W_PATH      = HERE / "apa" / "experiments" / "checkpoints" / "W_seen_K8.pt"
OUT_DIR     = HERE / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same K Rachel used. Cutting the dendrogram at 8 clusters lets us see
# whether the hierarchical structure naturally lines up with her K.
N_CUTS = 8

# ---------------------------------------------------------------------------
# Load W and compute user-user raw dot product (Gram matrix)
# ---------------------------------------------------------------------------

W = torch.load(W_PATH, map_location="cpu", weights_only=False).float().numpy()
n_users, K = W.shape                                          # (182, 8)
print(f"loaded W: shape {W.shape}")
print(f"||w|| stats: min={np.linalg.norm(W, axis=1).min():.4g}, "
      f"median={np.median(np.linalg.norm(W, axis=1)):.4g}, "
      f"max={np.linalg.norm(W, axis=1).max():.4g}")

S = W @ W.T                                                   # (182, 182) raw dot product

# ---------------------------------------------------------------------------
# Hierarchical clustering on Euclidean distance between W rows, average linkage
# (||w_i - w_j||^2 = ||w_i||^2 + ||w_j||^2 - 2 w_i·w_j — the natural distance
# when magnitudes are kept.)
# ---------------------------------------------------------------------------

Z = linkage(pdist(W, metric="euclidean"), method="average")
order = leaves_list(Z)

# Cut the dendrogram into N_CUTS flat clusters so we can draw boundaries
flat_labels = fcluster(Z, t=N_CUTS, criterion="maxclust")
labels_reordered = flat_labels[order]
boundaries = np.where(np.diff(labels_reordered) != 0)[0] + 1
n_clusters_found = len(np.unique(flat_labels))
print(f"hierarchical cut at maxclust={N_CUTS} → {n_clusters_found} clusters")

S_reordered = S[np.ix_(order, order)]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

vmax = float(np.max(np.abs(S_reordered)))

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(
    S_reordered, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    aspect="equal", interpolation="nearest",
)
for b in boundaries:
    ax.axhline(b - 0.5, color="black", linewidth=0.5)
    ax.axvline(b - 0.5, color="black", linewidth=0.5)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(f"PRISM user (reordered, n={n_users})")
ax.set_ylabel(f"PRISM user (reordered, n={n_users})")
ax.set_title(
    "User similarity on LoRe basis (K=8)\n"
    f"Dot product of W_seen_K8.pt rows (magnitudes preserved), "
    f"hierarchical reorder, cut into {n_clusters_found} clusters",
    pad=10,
)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$w_i \cdot w_j$")

fig.tight_layout()
out_path = OUT_DIR / "user_similarity_clustermap.png"
fig.savefig(out_path, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"saved {out_path}")
