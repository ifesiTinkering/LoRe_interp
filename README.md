# LoRe basis interpretability — Rachel Freedman's APA code + larger-PRISM heatmap

This repo holds:

1. **`apa/`** — Rachel Freedman's Adaptive Pluralistic Alignment code (from
   the anonymous repository linked in her paper). Includes her pre-computed
   LoRe basis (`apa/experiments/checkpoints/V_K8.pt`), PRISM jury weights
   (`W_seen_K8.pt`), and simulated-jury weights
   (`W_adapted_hist_C016_C020_filtered.pt`).
2. **`plot_combined_heatmap.py`** — single extra script that calls her
   `fig_user_weights_grid` for the top panel (her exact Figure 2) and adds
   a bottom panel showing **all 182 PRISM seen users** from her
   `W_seen_K8.pt`, sorted by dominant basis. Output: `out/combined_heatmap.png`.

The point: reproduce her published Figure 2 exactly and visualise the full
PRISM user population on the same basis at the same time, to see whether the
patterns visible in her 2-user subset hold in the larger group.

## Running it

```bash
# Install minimal deps (CPU is fine, plotting takes ~30 sec)
pip install torch matplotlib numpy pillow

# Plot
python plot_combined_heatmap.py
# → out/user_weights_grid.png        (her top figure, regenerated)
# → out/bottom_all_prism.png         (our larger panel)
# → out/combined_heatmap.png         (stacked)
```

The script only uses her checkpoints; nothing is retrained.

## Citations

This repository combines two code sources. Cite both:

```bibtex
@misc{2026apa,
  title  = {Adaptive Pluralistic Alignment: A pipeline for dynamic artificial democracy},
  author = {Freedman, R. and others},
  year   = {2026},
  eprint = {2605.01642},
}

@misc{bose2025lore,
  title  = {LoRe: Personalizing LLMs via Low-Rank Reward Modeling},
  author = {Bose, A. and Xiong, Z. and Chi, Y. and Du, S. S. and Xiao, L. and Fazel, M.},
  year   = {2025},
  eprint = {2504.14439},
}
```

## License

Licensing is whatever each upstream piece carries. The `apa/` directory is
distributed unmodified; the LICENSE file in this repo is the upstream LoRe
license (CC-BY-NC 4.0), preserved because earlier commit history in this
fork included files from `facebookresearch/LoRe`.
