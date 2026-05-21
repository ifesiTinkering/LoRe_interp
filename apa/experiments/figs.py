"""
Figures for the APA paper.

Each figure is produced by a top-level ``fig_*`` function and saved to
``experiments/figs/``. Run via the CLI:

    uv run python -m experiments.figs user_weights_grid
    uv run python -m experiments.figs all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Custom white→pink colormap, distinct from "Purples" used for C016
_PINK_CMAP = LinearSegmentedColormap.from_list(
    "apa_pink", ["#ffffff", "#fbd5e3", "#f48fb1", "#ec407a", "#c2185b"]
)
try:
    mpl.colormaps.register(_PINK_CMAP)
except ValueError:
    pass

from apa.config import MODELS_DIR

FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Loading helpers
# =============================================================================

def _load_prism_W(K: int = 8) -> tuple[np.ndarray, list[str]]:
    """Load PRISM seen-user W matrix and corresponding user IDs."""
    W = torch.load(MODELS_DIR / f"W_seen_K{K}.pt", map_location="cpu", weights_only=False)
    mapping_path = MODELS_DIR / "user_to_idx.json"
    if mapping_path.exists():
        user_to_idx = json.loads(mapping_path.read_text())
        idx_to_user = {v: k for k, v in user_to_idx.items()}
        ids = [idx_to_user.get(i, f"prism_user_{i}") for i in range(W.shape[0])]
    else:
        ids = [f"prism_user_{i}" for i in range(W.shape[0])]
    return W.float().numpy(), ids


def _load_adapted_W(path: Path) -> dict[str, np.ndarray]:
    """Return {user_id: w_vector} from a W_adapted_*.pt checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {uid: data["w"].float().numpy() for uid, data in ckpt["users"].items()}


def _top_second_place_users(
    adapted: dict[str, np.ndarray],
    prefix: str,
    n: int = 2,
) -> list[str]:
    """Pick the ``n`` users (by id prefix) with the largest second-place |w|.

    Adapted hist-llama vectors are essentially one-hot; ranking by the
    runner-up coordinate's absolute value highlights users with the most
    visible secondary structure (which is what makes the figure rows look
    interesting). Falls back to lexicographic order on ties.
    """
    candidates = [(uid, w) for uid, w in adapted.items() if uid.startswith(prefix)]
    if not candidates:
        raise ValueError(f"No adapted users with prefix {prefix!r} in checkpoint")

    def _second_place(w: np.ndarray) -> float:
        absw = np.sort(np.abs(w))[::-1]
        return float(absw[1]) if len(absw) >= 2 else 0.0

    candidates.sort(key=lambda kv: (-_second_place(kv[1]), kv[0]))
    return [uid for uid, _ in candidates[:n]]


def _draw_group_brackets(
    fig,
    ax,
    group_spans: list[tuple[str, int, int, str]],
) -> None:
    """Draw colored vertical brackets + bold labels to the left of yticklabels.

    Shared between figures so they stay visually consistent. ``group_spans``
    is a list of ``(label, start_row, end_row_exclusive, cmap_name)`` tuples.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_left_disp = min(
        t.get_window_extent(renderer=renderer).x0
        for t in ax.get_yticklabels()
    )
    inv = ax.transAxes.inverted()
    label_left_ax = inv.transform((label_left_disp, 0))[0]
    bracket_x = label_left_ax - 0.04
    label_x = bracket_x - 0.02
    trans = ax.get_yaxis_transform()
    for label, start, end, cmap_name in group_spans:
        y0, y1 = start - 0.45, (end - 1) + 0.45
        ax.plot([bracket_x, bracket_x], [y0, y1],
                transform=trans, clip_on=False,
                color=mpl.colormaps[cmap_name](0.8), linewidth=2.8)
        ax.text(label_x, (y0 + y1) / 2, label,
                transform=trans, ha="right", va="center",
                fontsize=11, fontweight="bold",
                color=mpl.colormaps[cmap_name](0.9))


# =============================================================================
# Figure 1: user-weights grid
# =============================================================================

def fig_user_weights_grid(
    K: int = 8,
    adapted_path: Path | None = None,
    save: bool = True,
) -> Path:
    """
    Render a 6 x K grid of user weight vectors.

    Rows 0-1: first two PRISM users (blue palette).
    Rows 2-3: first two C016 historical users (purple palette).
    Rows 4-5: first two C020 historical users (pink palette).

    Color intensity scales with weight magnitude on a shared global
    vmin/vmax so that magnitudes are comparable across users.
    """
    adapted_path = adapted_path or (MODELS_DIR / "W_adapted_hist_C016_C020_filtered.pt")

    W_prism, prism_ids = _load_prism_W(K=K)
    adapted = _load_adapted_W(adapted_path)

    prism_rows = [(prism_ids[i], W_prism[i]) for i in range(2)]
    # Adapted hist-llama vectors are essentially one-hot; pick the two
    # users per century with the largest second-place weight so their
    # rows show *some* secondary structure.
    c016_picks = _top_second_place_users(adapted, "hist_C016_", n=2)
    c020_picks = _top_second_place_users(adapted, "hist_C020_", n=2)
    c016_rows = [(uid, adapted[uid]) for uid in c016_picks]
    c020_rows = [(uid, adapted[uid]) for uid in c020_picks]

    groups = [
        ("PRISM", prism_rows, "Blues"),
        ("C016",  c016_rows,  "Purples"),
        ("C020",  c020_rows,  "apa_pink"),
    ]

    # Per-row normalization on |w|: weight magnitudes vary by orders of
    # magnitude across PRISM (tiny continuous) vs. adapted (near one-hot)
    # users, so a shared scale collapses one group to white. Each row is
    # mapped to [0, 1] by its own max |w|.
    # Map 0 to a pale tint instead of pure white so empty cells still
    # carry the row's group color.
    PALE_FLOOR = 0.12
    norm = Normalize(vmin=-PALE_FLOOR / (1 - PALE_FLOOR), vmax=1.0)

    n_rows = sum(len(rows) for _, rows, _ in groups)
    n_cols = K

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })

    fig, ax = plt.subplots(figsize=(0.7 * n_cols + 2.5, 0.6 * n_rows + 1.2))

    cell_colors = np.zeros((n_rows, n_cols, 4))
    row_labels: list[str] = []
    row_idx = 0
    group_spans: list[tuple[str, int, int, str]] = []  # (label, start, end, cmap)
    for label, rows, cmap_name in groups:
        cmap = mpl.colormaps[cmap_name]
        start = row_idx
        for uid, w in rows:
            scale = float(np.max(np.abs(w))) or 1.0
            cell_colors[row_idx] = cmap(norm(np.abs(w) / scale))
            row_labels.append(uid)
            row_idx += 1
        group_spans.append((label, start, row_idx, cmap_name))

    ax.imshow(cell_colors, aspect="equal", interpolation="nearest")

    # Cell borders
    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor="white", linewidth=1.2,
            ))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"$V_{{{i}}}$" for i in range(n_cols)])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Basis function")
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    _draw_group_brackets(fig, ax, group_spans)

    ax.set_title("User weight vectors over LoRe basis", pad=12)
    fig.tight_layout()

    out = FIGS_DIR / "user_weights_grid.pdf"
    if save:
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
        plt.close(fig)
    return out


# =============================================================================
# Figure 2: jury agreement heatmap
# =============================================================================


def _spearman_matrix(rankings: np.ndarray) -> np.ndarray:
    """Pairwise Spearman ρ between rows of a ranking matrix (n_voters × n_items).

    Each row is a preference order (response indices); we convert to position
    vectors and take the Pearson correlation of those, which is equivalent
    to Spearman ρ when there are no ties.
    """
    n_voters, n_items = rankings.shape
    positions = np.empty_like(rankings, dtype=float)
    for v in range(n_voters):
        for p, idx in enumerate(rankings[v]):
            positions[v, idx] = p
    return np.corrcoef(positions)


def fig_jury_agreement_heatmap(
    audit_log_path: Path | None = None,
    save: bool = True,
    **_: object,
) -> Path:
    """
    Pairwise Spearman ρ heatmap across all 30 jurors of the `complex` variant
    on Q1.

    Voters are ordered 16C → 20C → PRISM ("original"), matching the
    user_weights_grid figure. Cell color is ρ on the Q1 case from the audit
    log (the first case with query_id == 1).
    """
    audit_log_path = audit_log_path or (
        Path(__file__).parent / "vote_C016_C020" / "audit_log.json"
    )
    log = json.loads(audit_log_path.read_text())

    # Restrict to Q1 only.
    q1_cases = [c for c in log if c.get("query_id") == 1]
    if not q1_cases:
        raise ValueError(f"No case with query_id=1 in {audit_log_path}")
    case = q1_cases[0]

    # Stable voter order: 16C → 20C → original (PRISM).
    period_order = {"16C": 0, "20C": 1, "original": 2}
    voter_periods = {
        uid: meta.get("period", "unknown")
        for uid, meta in case["sampled_user_metadata"].items()
    }
    voters = sorted(
        case["sampled_user_ids"],
        key=lambda u: (period_order.get(voter_periods[u], 99), u),
    )

    rankings = np.array(
        [case["per_voter_rankings"][u] for u in voters], dtype=int
    )
    rho = _spearman_matrix(rankings)
    np.fill_diagonal(rho, 1.0)

    n = len(voters)

    # Group spans for the bracket labels, in the order they appear.
    period_for_voter = [voter_periods[u] for u in voters]
    group_cmaps = {"16C": "Purples", "20C": "apa_pink", "original": "Blues"}
    group_labels = {"16C": "C016", "20C": "C020", "original": "PRISM"}
    group_spans: list[tuple[str, int, int, str]] = []
    i = 0
    while i < n:
        p = period_for_voter[i]
        j = i
        while j < n and period_for_voter[j] == p:
            j += 1
        group_spans.append(
            (group_labels.get(p, p), i, j, group_cmaps.get(p, "Greys"))
        )
        i = j

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   aspect="equal", interpolation="nearest")

    short_labels = []
    for u in voters:
        if u.startswith("hist_C016_"):
            short_labels.append(f"C16-{u.split('_')[-1]}")
        elif u.startswith("hist_C020_"):
            short_labels.append(f"C20-{u.split('_')[-1]}")
        elif u.startswith("prism_user_"):
            short_labels.append(f"P{u.split('_')[-1]}")
        else:
            short_labels.append(u)

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Color tick labels by group so the century is readable at a glance.
    for axis_labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for tick, p in zip(axis_labels, period_for_voter):
            tick.set_color(mpl.colormaps[group_cmaps.get(p, "Greys")](0.85))

    # Group dividers
    for _, _, end, _ in group_spans[:-1]:
        ax.axhline(end - 0.5, color="black", linewidth=1.0)
        ax.axvline(end - 0.5, color="black", linewidth=1.0)

    _draw_group_brackets(fig, ax, group_spans)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ")

    ax.set_title("Pairwise jury rank agreement (Q1)", pad=12)
    fig.tight_layout()

    out = FIGS_DIR / "jury_agreement_heatmap.pdf"
    if save:
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
        plt.close(fig)
    return out


# =============================================================================
# CLI
# =============================================================================

FIGURES = {
    "user_weights_grid": fig_user_weights_grid,
    "jury_agreement_heatmap": fig_jury_agreement_heatmap,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("name", choices=list(FIGURES) + ["all"])
    parser.add_argument("--K", type=int, default=8)
    args = parser.parse_args()

    names = list(FIGURES) if args.name == "all" else [args.name]
    for name in names:
        out = FIGURES[name](K=args.K)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
