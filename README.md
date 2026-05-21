# LoRe basis interpretability — APA code + extensions

This repository builds on **Adaptive Pluralistic Alignment** (Freedman et al., 2026) — the APA paper and its published K=8 LoRe checkpoint trained on the PRISM preference dataset. Original code: https://anonymous.4open.science/r/apa. I reproduce Freedman's Figure 2 from the published checkpoint and then extend the analysis to all 182 PRISM users in her weight matrix, adding interpretability views (clustermap, PCA scatter, force-directed graph, basis-dominance) that the paper does not include.

## What I added on top of the APA code

This fork's contribution sits at the top so it's easy to find:

1. **`lore_basis_interpretability.ipynb`** — Colab notebook that runs all of the analyses below end-to-end. Loads the APA-published `W_seen_K8.pt` and produces every figure in this README. Self-contained; uses only `torch`, `numpy`, `matplotlib`, `sklearn`, `scipy`.
2. **`plot_combined_heatmap.py`** — calls APA's `fig_user_weights_grid` to reproduce Freedman's Figure 2 exactly (top panel), and stacks an extended bottom panel below it covering more users.
3. **`plot_user_similarity.py`** — pairwise dot-product heatmap across all 182 PRISM users from the APA checkpoint, with rows/cols reordered by hierarchical clustering on Euclidean distance.

The notebook also contains a PCA scatter of users in basis-coefficient space, a force-directed graph view, and the basis-dominance analysis described next.

## What I found: LoRe at K=8 on PRISM produces mixed users, not specialists

![Basis dominance across 182 PRISM users](figures/basis_dominance.png)

The most actionable finding came from the basis-dominance plot above. Across the 182 published PRISM users in `W_seen_K8.pt`:

- **All 8 bases are used.** Each is the dominant basis (largest `|w_k|`) for between 9% (V6) and 15% (V4) of users. No basis is dead; no basis is runaway-dominant. Freedman's K=8 choice is doing real work.
- **But almost nobody is a specialist.** Define dominance ratio = `|w_top| / |w_2nd|`. Across all 182 users:
  - **0 users (0%)** have dominance ratio ≥ 10× (no near-one-hot users).
  - **39 users (21%)** have a "clear winner" (≥2×).
  - **45 users (25%)** are **effectively tied** between their top two bases (<1.2× — second-favorite basis is more than 83% as influential as the favorite).
  - Median dominance ratio across the whole population: **1.41×**.
- **Per-basis decisiveness varies.** V4 stands out: it is both the most popular dominant basis AND has the highest median dominance ratio (~1.8× vs. ~1.3× elsewhere). When users care about V4 they commit to it. V7 has the most extreme outliers (one user near 7×). V5 is a "weakly dominant" basis (median ~1.3× even when winning). V0/V1/V3 are nearly indistinguishable in their dominance profiles — possibly a sign that K could be reduced without major loss.

**Takeaway:** the per-row normalization in Freedman's Figure 2 (each row scaled to its own max) makes weight differences look visually crushing, but the actual ratios are small. LoRe at K=8 on PRISM is best read as discovering a continuous mixture over 8 axes — not 8 discrete user archetypes.

## Future work

The natural next experiment: **does this basis-dominance distribution look different within demographic groups?** For example, do users who self-report caring most about "creativity" have a different dominance distribution than users who care most about "factuality"? If yes, that links Freedman's discovered basis to human-readable preference categories.

The blocker is that the published checkpoint doesn't ship a `user_id → row` mapping, so the 182 rows of `W_seen_K8.pt` are anonymous. Closing this gap requires re-running Freedman's training pipeline (`apa.load_prism` + `apa.train_lore_bases` at K=8) end-to-end on an A100 — roughly 6-8 hours for embedding generation plus ~30 minutes of training — and saving the in-memory ordering. Worth doing over a weekend; not committed to.

Once that mapping exists, the per-group dominance plots and a labeled version of the clustermap should follow easily.

## Repository contents

- **`apa/`** — Adaptive Pluralistic Alignment (APA) code, distributed unmodified from https://anonymous.4open.science/r/apa. Includes the pre-computed LoRe basis (`apa/experiments/checkpoints/V_K8.pt`), PRISM jury weights (`W_seen_K8.pt`), and simulated-jury weights (`W_adapted_hist_C016_C020_filtered.pt`).
- **`lore_basis_interpretability.ipynb`** — my notebook (see above).
- **`plot_combined_heatmap.py`** — my combined-heatmap script.
- **`plot_user_similarity.py`** — my user-similarity clustermap script.
- **`figures/`** — figures embedded in this README.

## Running it

```bash
pip install torch matplotlib numpy pillow scikit-learn scipy networkx

python plot_combined_heatmap.py
# → out/user_weights_grid.png        (Freedman Figure 2, regenerated)
# → out/bottom_all_prism.png         (my extension)
# → out/combined_heatmap.png         (stacked)

python plot_user_similarity.py
# → out/user_similarity_clustermap.png
```

Or open `lore_basis_interpretability.ipynb` in Colab for the full walkthrough including the dominance analysis. Everything uses the APA-published checkpoints; nothing is retrained.

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

Licensing is whatever each upstream piece carries. The `apa/` directory is distributed unmodified; the LICENSE file in this repo is the upstream LoRe license (CC-BY-NC 4.0), preserved because earlier commit history in this fork included files from `facebookresearch/LoRe`.
