"""
Compare eval_prefs suitability metrics across synthetic, PRISM, and random baselines.

PRISM and Random baselines use randomly sampled users from pre-computed
PRISM embeddings (no re-embedding needed).

Usage:
    uv run python scripts/compare_metrics.py --synth-path path/to/hist_prefs_all.jsonl
    uv run python scripts/compare_metrics.py --synth-path path/to/hist_prefs_all.jsonl --n-baseline-users 90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from apa.config import MODELS_DIR
from apa.synthetic_prefs.eval_prefs import (
    embed_preferences,
    evaluate_suitability,
    load_prefs,
)
from apa.synthetic_prefs.sample_data import (
    load_prism_embeddings,
    random_embeddings,
    sample_embeddings,
)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


METRICS = [
    ("annotation density (median)",        ("annotation_density", "median_pairs"),                  ">= 5"),
    ("label balance (norm consistency)",    ("label_balance", "mean_normalized_consistency"),        "> 1.3"),
    ("Krippendorff proxy",                 ("krippendorff_alpha_proxy", "corrected_ratio"),         "> 0.03"),
    ("NN accuracy",                        ("nearest_neighbor_accuracy", "mean_nn_accuracy"),       "> 0.6"),
    ("inter-user agreement (mean sim)",    ("inter_user_agreement", "mean_pairwise_similarity"),    "low=diverse"),
    ("inter-user agreement (std sim)",     ("inter_user_agreement", "std_pairwise_similarity"),     "high=diverse"),
    ("basis coherence",                    ("basis_space_coherence", "corrected_ratio"),             "> 0.005"),
    ("population accuracy",                ("population_accuracy", "accuracy"),                     "> 0.6"),
    ("held-out accuracy",                  ("held_out_accuracy", "mean_accuracy"),                  "> 0.6"),
    ("user vec mean dist",                 ("user_vector_diversity", "mean_pairwise_distance"),     "INFO"),
    ("effective rank",                     ("user_vector_diversity", "effective_rank"),              "INFO"),
    ("basis entropy (norm)",               ("basis_utilization_entropy", "normalized_mean_entropy"), "INFO"),
]


def _get(results: dict, keys: tuple) -> object:
    v = results
    for k in keys:
        v = v[k]
    return v


def main():
    parser = argparse.ArgumentParser(description="Compare suitability metrics across datasets.")
    parser.add_argument("--synth-path", type=Path, required=True,
                        help="Path to synthetic preferences JSONL.")
    parser.add_argument("--n-baseline-users", type=int, default=None,
                        help="Number of users for PRISM/Random baselines (default: match synth count).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=8)
    args = parser.parse_args()

    V = torch.load(MODELS_DIR / f"V_K{args.K}.pt", map_location="cpu", weights_only=True).float()

    # --- Synthetic: embed ---
    print(f"Embedding synthetic preferences from {args.synth_path}...", flush=True)
    synth_prefs = load_prefs(args.synth_path)
    from apa.train_lore_bases import get_embedding_model
    model, tokenizer = get_embedding_model()
    synth_embs = embed_preferences(synth_prefs, model, tokenizer)
    del model
    torch.cuda.empty_cache()

    n_synth = len(synth_embs)
    synth_results = evaluate_suitability(synth_embs, V=V, K=args.K)

    # --- PRISM & Random baselines (pre-computed embeddings, random users) ---
    n = args.n_baseline_users if args.n_baseline_users is not None else n_synth
    print(f"Sampling {n} PRISM and {n} Random users from pre-computed embeddings...", flush=True)
    prism_all = load_prism_embeddings()
    prism_n = sample_embeddings(prism_all, n, seed=args.seed)
    rand_n = random_embeddings(prism_all, n, seed=args.seed)

    prism_results = evaluate_suitability(prism_n, V=V, K=args.K)
    rand_results = evaluate_suitability(rand_n, V=V, K=args.K)

    # --- Table ---
    synth_label = f"Synth ({n_synth})"
    prism_label = f"PRISM ({n})"
    rand_label = f"Random ({n})"

    w_metric = 40
    w_col = 16
    header = f"{'Metric':<{w_metric}} {'Threshold':<{w_col}} {synth_label:<{w_col}} {prism_label:<{w_col}} {rand_label:<{w_col}}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, keys, thresh in METRICS:
        sv = _fmt(_get(synth_results, keys))
        pv = _fmt(_get(prism_results, keys))
        rv = _fmt(_get(rand_results, keys))
        print(f"{name:<{w_metric}} {thresh:<{w_col}} {sv:<{w_col}} {pv:<{w_col}} {rv:<{w_col}}")
    print(sep)


if __name__ == "__main__":
    main()
