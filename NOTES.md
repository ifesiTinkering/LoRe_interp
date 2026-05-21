# Fork notes

Originally forked from [facebookresearch/LoRe](https://github.com/facebookresearch/LoRe) to experiment with reproducing Rachel Freedman's PRISM Figure 2. The Meta LoRe source files have since been removed; the surviving content is:

- **`apa/`** — Rachel Freedman's APA code (from the anonymous repo linked in her paper), used unmodified.
- **`plot_combined_heatmap.py`** — small wrapper that calls her `fig_user_weights_grid` (no new plot logic) and adds one extra panel showing all 182 PRISM users from her published `W_seen_K8.pt`.

## Changes from upstream LoRe

- Removed `PRISM/`, `PersonalLLM/`, `RedditTLDR/`, `utils.py`, `requirements.txt`, `README.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md` — we no longer use Meta's training code; Rachel's `apa/` package supersedes it for our purpose (reproducing her published heatmap from her published checkpoints).
- Kept `LICENSE` — applies to the historical git history.

## Why not just use her published code as-is?

Two things her code does not do out of the box on a fresh machine:

1. `apa.config.MODELS_DIR` points to `/nas/XXXX-9/XXXX-1/APA/models/`, which doesn't exist outside her cluster. `plot_combined_heatmap.py` monkey-patches `MODELS_DIR` to point at the bundled `apa/experiments/checkpoints/` folder so her `fig_user_weights_grid` can find the checkpoints.
2. She publishes the **top-panel** figure but not the larger-population view of her PRISM W matrix. The bottom panel of our script is that view: all 182 rows of her `W_seen_K8.pt`, sorted by dominant basis, same per-row normalisation and same Blues palette so the two panels are visually consistent.

The only new plotting logic added by us is the bottom panel (~15 lines).
