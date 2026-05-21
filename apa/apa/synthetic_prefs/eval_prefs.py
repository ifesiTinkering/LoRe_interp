"""
LoRe dataset suitability evaluation.

Usage::

    from apa.synthetic_prefs.eval_prefs import embed_preferences, evaluate_suitability
    import torch

    # Step 1: embed raw preferences (skip if you already have embeddings)
    user_pref_embeddings = embed_preferences(user_prefs, model, tokenizer)

    # Step 2: run all metrics
    V = torch.load("models/V_K8.pt")
    results = evaluate_suitability(user_pref_embeddings, V=V)

CLI usage::

    python -m apa.synthetic_prefs.eval_prefs path/to/prefs.jsonl
    python -m apa.synthetic_prefs.eval_prefs path/to/prefs.parquet

Accepts a path to raw preference data in one of two formats:

  JSONL — one JSON object per line with fields:
      {"user_id": "u1", "prompt": "...", "chosen": "...", "rejected": "..."}

  Parquet (PRISM format) — with columns including prompt (list of chat dicts)
      and extra_info containing user_id, chosen_utterance, rejected_utterance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

PreferencePair = namedtuple("PreferencePair", ["prompt", "chosen", "rejected"])
"""A single pairwise preference: prompt text, chosen response text, rejected response text."""


# ---------------------------------------------------------------------------
# Raw text metrics (no model needed)
# ---------------------------------------------------------------------------

def annotation_density(
    user_pref_embeddings: list[torch.Tensor] | list[int],
    K: int,
) -> dict:
    """
    Check whether users have enough preference pairs to reliably fit a K-dim user vector.

    Rule of thumb: a user needs at least 2*K pairs to constrain a K-dimensional vector.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors, or a plain
            list of per-user pair counts (ints).
        K: Rank (number of basis vectors) of the LoRe model.

    Returns:
        Dict with per-user counts and a warning flag.
    """
    counts = [u if isinstance(u, int) else len(u) for u in user_pref_embeddings]
    median_count = float(np.median(counts))
    fraction_below = float(np.mean([c < 2 * K for c in counts]))
    return {
        "n_users": len(counts),
        "min_pairs": int(min(counts)),
        "median_pairs": median_count,
        "mean_pairs": float(np.mean(counts)),
        "fraction_below_2K": fraction_below,
        "warn": median_count < 2 * K,
    }


def prompt_diversity_surface(
    user_prefs: dict[str, list[PreferencePair]],
) -> dict:
    """
    Count unique prompts across the dataset (surface-level, no embedding).

    Low diversity means most users answer the same few questions — user vectors
    will reflect idiosyncratic question reactions, not generalizable preferences.

    Args:
        user_prefs: Mapping from user_id to list of PreferencePair.

    Returns:
        Dict with unique prompt counts and per-user statistics.
    """
    all_prompts: list[str] = []
    per_user_unique: list[int] = []

    for pairs in user_prefs.values():
        prompts = [p.prompt for p in pairs]
        all_prompts.extend(prompts)
        per_user_unique.append(len(set(prompts)))

    n_total_pairs = len(all_prompts)
    n_unique_prompts = len(set(all_prompts))

    return {
        "n_unique_prompts": n_unique_prompts,
        "n_total_pairs": n_total_pairs,
        "prompt_reuse_rate": 1.0 - n_unique_prompts / max(n_total_pairs, 1),
        "mean_unique_prompts_per_user": float(np.mean(per_user_unique)) if per_user_unique else 0.0,
        "median_unique_prompts_per_user": float(np.median(per_user_unique)) if per_user_unique else 0.0,
    }


# ---------------------------------------------------------------------------
# Embedding-based metrics (no V needed)
# ---------------------------------------------------------------------------

def label_balance(user_pref_embeddings: list[torch.Tensor]) -> dict:
    """
    Measure per-user preference direction consistency, normalised against the
    random-data baseline.

    Raw consistency is ||mean(pref_vecs)|| / mean(||pref_vecs||).  For iid
    random vectors in D dimensions with n samples per user, the expected value
    is 1/√n — landing in the green zone for any realistic n regardless of
    whether preferences are learnable.  We therefore normalise each user's
    score by its expected random value, giving:

        normalised_consistency_i = raw_i * √(n_i)

    Interpretation:
        ≈ 1.0  random / uninformative preferences (baseline)
        > 1.0  user's preferences point more consistently than chance
        >> 1   strong, learnable preference direction

    Returns:
        Dict with normalised and raw consistency statistics.
    """
    raw_scores = []
    norm_scores = []
    for X in user_pref_embeddings:
        X = X.float()
        n = len(X)
        mean_norm = X.mean(dim=0).norm().item()
        mean_of_norms = X.norm(dim=1).mean().item()
        raw = mean_norm / (mean_of_norms + 1e-12)
        raw_scores.append(raw)
        norm_scores.append(raw * math.sqrt(n))

    norm_arr = np.array(norm_scores)
    return {
        "mean_normalized_consistency": float(norm_arr.mean()),  # ~1.0 random, >1 structured
        "std_normalized_consistency": float(norm_arr.std()),
        "mean_raw_consistency": float(np.mean(raw_scores)),
        "per_user_normalized": norm_scores,
    }


def inter_user_agreement(user_pref_embeddings: list[torch.Tensor]) -> dict:
    """
    Measure pairwise agreement between users via cosine similarity of their
    mean preference vectors.

    High mean similarity: users mostly agree — little room for personalisation.
    Low mean similarity with high variance: mixed bag, some learnable structure.
    Very low mean similarity: users disagree broadly — LoRe must work hard.

    Returns:
        Dict with off-diagonal similarity statistics and clustering proxy.
    """
    n = len(user_pref_embeddings)
    if n < 2:
        return {
            "mean_pairwise_similarity": float("nan"),
            "std_pairwise_similarity": float("nan"),
            "min_pairwise_similarity": float("nan"),
            "max_pairwise_similarity": float("nan"),
            "fraction_high_agreement": float("nan"),
            "n_users": n,
        }

    means = torch.stack([X.float().mean(dim=0) for X in user_pref_embeddings])
    means_norm = F.normalize(means, dim=1)
    sim = means_norm @ means_norm.T  # [n_users, n_users]

    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag = sim[mask].cpu().numpy()

    return {
        "mean_pairwise_similarity": float(off_diag.mean()),
        "std_pairwise_similarity": float(off_diag.std()),
        "min_pairwise_similarity": float(off_diag.min()),
        "max_pairwise_similarity": float(off_diag.max()),
        "fraction_high_agreement": float(np.mean(off_diag > 0.5)),
        "n_users": n,
    }


def _noise_corrected_variance_ratio(
    groups: list[torch.Tensor],
    user_pref_embeddings: list[torch.Tensor],
) -> tuple[float, float, float]:
    """Shared helper: noise-corrected between-group variance ratio.

    Returns (corrected_ratio, raw_ratio, sampling_noise_fraction).
    """
    all_data = torch.cat(groups, dim=0)
    grand_mean = all_data.mean(dim=0)
    user_means = torch.stack([g.mean(dim=0) for g in groups])
    between_var = ((user_means - grand_mean) ** 2).mean().item()
    total_var = ((all_data - grand_mean) ** 2).mean().item()
    mean_recip_n = float(np.mean([1.0 / len(X) for X in user_pref_embeddings]))
    raw_ratio = between_var / (total_var + 1e-12)
    corrected_ratio = raw_ratio - mean_recip_n
    return corrected_ratio, raw_ratio, mean_recip_n


def krippendorff_alpha_proxy(user_pref_embeddings: list[torch.Tensor]) -> dict:
    """
    Noise-corrected proxy for inter-annotator reliability (ICC-style).

    The raw between-user variance ratio equals mean(1/n_i) for random data,
    because the variance of a user mean over n iid samples is total_var/n.
    This sampling noise is indistinguishable from genuine user-to-user
    differences at the raw ratio level.

    We subtract the expected sampling noise to get a corrected ratio that is
    ≈ 0 for random data by construction:

        corrected_ratio = (between_var - total_var * mean(1/n_i)) / total_var
                        = raw_ratio - mean(1/n_i)

    Interpretation:
        ≈ 0      no genuine between-user signal beyond sampling noise
        > 0      users are more distinct than noise would predict
        > 0.03   meaningful user diversity (calibrated on PRISM)

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.

    Returns:
        Dict with corrected ratio, raw ratio, and the sampling noise fraction.
    """
    groups = [X.float() for X in user_pref_embeddings]
    corrected, raw, noise = _noise_corrected_variance_ratio(groups, user_pref_embeddings)
    return {
        "corrected_ratio": corrected,        # ~0 for random, >0 for structured
        "raw_ratio": raw,
        "sampling_noise_fraction": noise,    # expected raw_ratio under null
    }


# ---------------------------------------------------------------------------
# Metrics requiring pretrained V
# ---------------------------------------------------------------------------

def basis_space_coherence(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor,
) -> dict:
    """
    Test whether users cluster meaningfully in the pretrained basis space.

    Applies the same noise-corrected variance decomposition as
    krippendorff_alpha_proxy, but in the K-dimensional V-projected space
    rather than the full D-dimensional embedding space.

    This specifically tests basis alignment: users may be distinct in raw
    embedding space yet all look identical once projected onto V (if V is
    misaligned with this domain's preference dimensions).  A positive
    corrected_ratio here means user identity predicts variation *within the
    basis space LoRe actually uses*.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V: Pretrained basis matrix [D, K].

    Returns:
        Dict with corrected and raw variance ratios in V-space.
    """
    V = V.float()
    groups = [X.float() @ V for X in user_pref_embeddings]  # list of [n_i, K]
    corrected, raw, noise = _noise_corrected_variance_ratio(groups, user_pref_embeddings)
    return {
        "corrected_ratio": corrected,        # ~0 for random, >0 for structured
        "raw_ratio": raw,
        "sampling_noise_fraction": noise,
    }


def population_accuracy(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    """
    Held-out accuracy of a single user vector fitted on ALL pairs pooled.

    This tests two things simultaneously:
      (a) Is there a universal preference direction in this domain at all?
      (b) Does the pretrained V capture it?

    A domain-agnostic basis V learned on a different dataset will fail (b) even
    if (a) is true.  Random data fails (a).  Both failures produce accuracy ≈ 0.5.

    Unlike fit_quality (which always returns 1.0 due to overfitting on training
    data), this uses a held-out split so the score reflects genuine generalisation.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V: Pretrained basis matrix [D, K].
        test_frac: Fraction of pooled pairs to hold out.
        seed: RNG seed for the pool shuffle.

    Returns:
        Dict with held-out accuracy, train/test counts.
    """
    V = V.float()
    all_X = torch.cat([X.float() for X in user_pref_embeddings])
    N = len(all_X)
    n_test = max(1, int(N * test_frac))

    rng = torch.Generator()
    rng.manual_seed(seed)
    idx = torch.randperm(N, generator=rng)
    X_train, X_test = all_X[idx[:-n_test]], all_X[idx[-n_test:]]

    XV_train = X_train @ V
    w = torch.linalg.lstsq(XV_train, torch.ones(len(XV_train), device=XV_train.device)).solution

    accuracy = (X_test @ V @ w > 0).float().mean().item()
    return {
        "accuracy": accuracy,      # ~0.5 for random/misaligned, >0.55 for good domain fit
        "n_train": len(X_train),
        "n_test": n_test,
        "warn": n_test < 5,
    }


def nearest_neighbor_accuracy(
    user_pref_embeddings: list[torch.Tensor],
    seed: int = 0,
) -> dict:
    """
    Test whether geometrically similar users actually share preferences.
    Does not require V.

    Each user's data is split in half: the first half computes means (used
    for NN lookup), the second half is scored using the NN's mean direction.
    This data-splitting prevents a subtle bias: without it, NN selection
    creates a transitive correlation (x_i → μ_i → NN selection → μ_j) that
    pushes accuracy above 0.5 even for random data.

    Interpretation:
        ≈ 0.5   random / no shared structure between similar users
        > 0.55  users with similar mean preferences share individual pairs
        >> 0.6  strong learnable structure (LoRe's core assumption holds)

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        seed: RNG seed for the train/test split.

    Returns:
        Dict with mean/std nearest-neighbour accuracy.
    """
    n_users = len(user_pref_embeddings)
    if n_users < 2:
        return {
            "mean_nn_accuracy": float("nan"),
            "std_nn_accuracy": float("nan"),
            "n_users": n_users,
        }

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Split each user: first half for means/NN graph, second half for scoring
    train_halves = []
    test_halves = []
    for X in user_pref_embeddings:
        n = len(X)
        idx = torch.randperm(n, generator=rng)
        mid = max(1, n // 2)
        train_halves.append(X[idx[:mid]].float())
        test_halves.append(X[idx[mid:]].float())

    # NN graph from train halves only
    means = torch.stack([X.mean(dim=0) for X in train_halves])
    means_norm = F.normalize(means, dim=1)
    sim = means_norm @ means_norm.T                   # [n_users, n_users]
    sim.fill_diagonal_(-1.0)
    nn_idx = sim.argmax(dim=1)

    # Score held-out halves using NN's train mean
    accuracies = []
    for i, X_test in enumerate(test_halves):
        if len(X_test) == 0:
            continue
        nn_mean = means[nn_idx[i]]                    # NN's mean from train half
        acc = (X_test @ nn_mean > 0).float().mean().item()
        accuracies.append(acc)

    return {
        "mean_nn_accuracy": float(np.mean(accuracies)),   # ~0.5 random, >0.55 structured
        "std_nn_accuracy": float(np.std(accuracies)),
    }


def fit_user_vectors(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor,
) -> torch.Tensor:
    """
    Fit per-user weight vectors via closed-form least squares.

    For each user, solves:  argmin_w ||XV @ w - 1||²
    where XV = X @ V is the projection of preference embeddings into basis space.

    This is a fast closed-form approximation of PersonalizeBatch (which optimises
    NLL via gradient descent). The approximation is sufficient for diagnostic
    metrics; use PersonalizeBatch from train_lore_bases.py for production fitting.

    Users with fewer than 2 pairs are given zero vectors.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V: Pretrained basis matrix [D, K].

    Returns:
        W: Tensor of shape [n_users, K] — raw (pre-softmax) user weight vectors.
    """
    V = V.float()
    K = V.shape[1]
    user_ws = []

    for X in user_pref_embeddings:
        X = X.float()
        if len(X) < 2:
            user_ws.append(torch.zeros(K))
            continue
        XV = X @ V  # [n_prefs, K]
        target = torch.ones(len(XV), device=XV.device)
        # Closed-form least squares: w = (XV^T XV)^{-1} XV^T 1
        result = torch.linalg.lstsq(XV, target)
        user_ws.append(result.solution.cpu())

    return torch.stack(user_ws)  # [n_users, K]


def user_vector_diversity(W: torch.Tensor) -> dict:
    """
    Measure how spread out the fitted user vectors are in basis space.

    High diversity means LoRe has learned to distinguish users. Low diversity
    means all users have similar weights — personalisation isn't helping.

    Args:
        W: Raw user weight matrix [n_users, K] from fit_user_vectors().

    Returns:
        Dict with mean pairwise distance, effective rank of user vector space,
        and eigenvalue spectrum of the user vector covariance.
    """
    W_soft = F.softmax(W.float(), dim=1)  # [n_users, K]
    W_norm = F.normalize(W_soft, dim=1)
    sim = W_norm @ W_norm.T  # [n_users, n_users]

    n = W.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag_sim = sim[mask].cpu().numpy()
    mean_pairwise_distance = 1.0 - float(off_diag_sim.mean())

    # Effective rank of user vector covariance
    cov = torch.cov(W_soft.T)  # [K, K]
    eigenvalues = torch.linalg.eigvalsh(cov).cpu().float()
    eigenvalues = eigenvalues.clamp(min=0)
    max_eig = eigenvalues.max().item()
    eff_rank = int((eigenvalues > 0.01 * max_eig).sum().item()) if max_eig > 0 else 0

    return {
        "mean_pairwise_distance": mean_pairwise_distance,
        "std_pairwise_distance": float(off_diag_sim.std()),
        "effective_rank": eff_rank,
        "eigenvalues": eigenvalues.tolist(),
    }


def basis_utilization_entropy(W: torch.Tensor) -> dict:
    """
    Measure how uniformly users spread their weight across the K bases.

    High entropy (close to log K): users leverage many different bases —
    the full rank is being utilised.
    Low entropy: most users concentrate on 1-2 bases — effective rank is low,
    and the pretrained bases may not cover the new domain's preference dimensions.

    Args:
        W: Raw user weight matrix [n_users, K] from fit_user_vectors().

    Returns:
        Dict with mean entropy, max possible entropy (log K), and normalised mean.
    """
    W_soft = F.softmax(W.float(), dim=1)  # [n_users, K]
    # Shannon entropy per user: -sum(w_i * log(w_i))
    entropies = -(W_soft * (W_soft + 1e-12).log()).sum(dim=1).cpu().numpy()
    max_entropy = math.log(W.shape[1]) if W.shape[1] > 1 else 1.0

    return {
        "mean_entropy": float(entropies.mean()),
        "std_entropy": float(entropies.std()),
        "max_entropy": max_entropy,
        "normalized_mean_entropy": float(entropies.mean()) / max_entropy,
        "per_user_entropy": entropies.tolist(),
    }


# ---------------------------------------------------------------------------
# Metrics requiring held-out splits
# ---------------------------------------------------------------------------

def held_out_accuracy(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    """
    Cross-validate closed-form user vector fitting against held-out preferences.

    For each user, randomly holds out `test_frac` of pairs, fits a user vector
    on the rest, and evaluates accuracy on the held-out pairs.  Users with
    fewer than 4 pairs are skipped (need at least 1 train + 1 test).

    This is the most faithful fast proxy for what LoRe will achieve in
    production few-shot adaptation.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V: Pretrained basis matrix [D, K].
        test_frac: Fraction of pairs to hold out for evaluation (default 0.2).
        seed: RNG seed for the shuffle.

    Returns:
        Dict with mean/std held-out accuracy and number of users evaluated.
    """
    V = V.float()
    rng = torch.Generator()
    rng.manual_seed(seed)
    accuracies = []

    for X in user_pref_embeddings:
        X = X.float()
        n = len(X)
        if n < 4:
            continue
        idx = torch.randperm(n, generator=rng)
        X = X[idx]
        n_test = max(1, int(n * test_frac))
        n_train = n - n_test
        X_train, X_test = X[:n_train], X[n_train:]

        XV_train = X_train @ V
        target_train = torch.ones(n_train, device=XV_train.device)
        result = torch.linalg.lstsq(XV_train, target_train)
        w = result.solution

        XV_test = X_test @ V
        acc = (XV_test @ w > 0).float().mean().item()
        accuracies.append(acc)

    return {
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "std_accuracy": float(np.std(accuracies)) if accuracies else float("nan"),
        "n_users_evaluated": len(accuracies),
    }


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_preferences(
    user_prefs: dict[str, list[PreferencePair]],
    model: Any,
    tokenizer: Any,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """
    Embed raw user preferences using the reward model.

    For each pair, embeds f(prompt + chosen) and f(prompt + rejected) and
    stores the difference (chosen - rejected) as the preference vector.
    This matches the representation used during LoRe training.

    Args:
        user_prefs: Mapping from user_id to list of PreferencePair.
        model: Skywork-Reward (or compatible) model with hidden_states output.
        tokenizer: Corresponding tokenizer.
        device: Device string for inference.

    Returns:
        Per-user list of [n_prefs, D] float32 tensors (sorted by user_id).
    """
    from apa.train_lore_bases import _format_for_embedding, _extract_embedding

    result = []
    for user_id in sorted(user_prefs.keys()):
        pairs = user_prefs[user_id]
        diffs = []
        for pair in pairs:
            chosen_text = _format_for_embedding(pair.prompt, pair.chosen, tokenizer)
            rejected_text = _format_for_embedding(pair.prompt, pair.rejected, tokenizer)
            chosen_emb = torch.tensor(
                _extract_embedding(model, tokenizer, chosen_text, device),
                dtype=torch.float32,
            )
            rejected_emb = torch.tensor(
                _extract_embedding(model, tokenizer, rejected_text, device),
                dtype=torch.float32,
            )
            diffs.append(chosen_emb - rejected_emb)
        if diffs:
            result.append(torch.stack(diffs))
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_suitability(
    user_pref_embeddings: list[torch.Tensor],
    V: torch.Tensor | None = None,
    K: int = 8,
    user_prefs: dict[str, list[PreferencePair]] | None = None,
) -> dict:
    """
    Run all applicable suitability metrics and print a summary table.

    Args:
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors (required).
        V: Pretrained LoRe basis [D, K]. If None, only embedding-based metrics run.
        K: Rank value for annotation_density threshold check.
        user_prefs: Raw preference pairs (dict[user_id, list[PreferencePair]]).
                    Required only for prompt_diversity_surface.

    Returns:
        Flat dict of all computed metric results.
    """
    results: dict = {}

    # --- Raw text ---
    results["annotation_density"] = annotation_density(user_pref_embeddings, K)
    if user_prefs is not None:
        results["prompt_diversity"] = prompt_diversity_surface(user_prefs)

    # --- Embedding-based ---
    results["label_balance"] = label_balance(user_pref_embeddings)
    results["inter_user_agreement"] = inter_user_agreement(user_pref_embeddings)
    results["krippendorff_alpha_proxy"] = krippendorff_alpha_proxy(user_pref_embeddings)
    results["nearest_neighbor_accuracy"] = nearest_neighbor_accuracy(user_pref_embeddings)

    if V is not None:
        # --- V-dependent ---
        results["basis_space_coherence"] = basis_space_coherence(user_pref_embeddings, V)
        results["population_accuracy"] = population_accuracy(user_pref_embeddings, V)
        W = fit_user_vectors(user_pref_embeddings, V)
        results["user_vector_diversity"] = user_vector_diversity(W)
        results["basis_utilization_entropy"] = basis_utilization_entropy(W)

        # --- Held-out ---
        results["held_out_accuracy"] = held_out_accuracy(user_pref_embeddings, V)

    return results


# ---------------------------------------------------------------------------
# Load raw preferences from file
# ---------------------------------------------------------------------------

def load_prefs_jsonl(path: Path) -> dict[str, list[PreferencePair]]:
    """Load preferences from JSONL (one JSON object per line).

    Expected fields: user_id, prompt, chosen, rejected.
    """
    prefs: dict[str, list[PreferencePair]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prefs[obj["user_id"]].append(
                PreferencePair(
                    prompt=obj["prompt"],
                    chosen=obj["chosen"],
                    rejected=obj["rejected"],
                )
            )
    return dict(prefs)


def load_prefs_parquet(path: Path) -> dict[str, list[PreferencePair]]:
    """Load preferences from a PRISM-format parquet file.

    Expects columns: prompt (list of chat dicts), extra_info (dict with
    user_id, chosen_utterance, rejected_utterance).
    """
    import pandas as pd

    df = pd.read_parquet(path)
    prefs: dict[str, list[PreferencePair]] = defaultdict(list)

    for _, row in df.iterrows():
        extra = row["extra_info"]
        user_id = extra["user_id"]
        chosen = extra["chosen_utterance"]
        rejected = extra.get("rejected_utterance", "")

        # rejected_utterance may be a list/array of alternatives; take first
        if isinstance(rejected, (list, tuple)):
            if len(rejected) == 0:
                continue
            rejected = rejected[0]
        elif hasattr(rejected, '__len__') and not isinstance(rejected, str):
            # numpy array
            if len(rejected) == 0:
                continue
            rejected = str(rejected[0])

        if not rejected:
            continue

        # prompt is a list/array of chat dicts, e.g. [{"role": "user", "content": "..."}]
        prompt = row["prompt"]
        if hasattr(prompt, '__iter__') and not isinstance(prompt, str):
            # Extract the user's message text from chat turns
            prompt_text = " ".join(
                turn["content"] for turn in prompt
                if isinstance(turn, dict) and turn.get("role") == "user"
            )
        else:
            prompt_text = str(prompt)

        prefs[user_id].append(
            PreferencePair(prompt=prompt_text, chosen=chosen, rejected=rejected)
        )

    return dict(prefs)


def load_prefs(path: Path) -> dict[str, list[PreferencePair]]:
    """Load preferences from a file, auto-detecting format by extension."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_prefs_jsonl(path)
    elif suffix == ".parquet":
        return load_prefs_parquet(path)
    elif suffix == ".json":
        return load_prefs_jsonl(path)  # treat .json as JSONL
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .jsonl or .parquet."
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _D = "\033[2m"; _0 = "\033[0m"


def _status(val, green, warn=None):
    """Return (colour+label, raw) for a value given (lo, hi) green/warn ranges."""
    if math.isnan(val):
        return f"{_Y}N/A{_0}", val
    if green[0] <= val <= green[1]:
        return f"{_G}PASS{_0}", val
    if warn and warn[0] <= val <= warn[1]:
        return f"{_Y}WARN{_0}", val
    return f"{_R}FAIL{_0}", val


def report(name: str, user_pref_embeddings: list[torch.Tensor],
           V: torch.Tensor, K: int = 8) -> dict:
    """
    Run all suitability metrics and print a formatted report.

    Args:
        name:                 Dataset label shown in the header.
        user_pref_embeddings: Per-user list of [n_prefs, D] tensors.
        V:                    Pretrained LoRe basis [D, K].
        K:                    Rank (for annotation density threshold).

    Returns:
        Dict of raw metric results.
    """
    results = evaluate_suitability(user_pref_embeddings, V=V, K=K)

    ad  = results["annotation_density"]
    lb  = results["label_balance"]
    kap = results["krippendorff_alpha_proxy"]
    nna = results["nearest_neighbor_accuracy"]
    bsc = results.get("basis_space_coherence", {})
    pa  = results.get("population_accuracy", {})
    hoa = results.get("held_out_accuracy", {})
    uvd = results.get("user_vector_diversity", {})
    bue = results.get("basis_utilization_entropy", {})

    # --- thresholds ---
    rows = [
        # (metric, display_value, threshold_label, val, green_range, warn_range)
        ("annotation density",
            f"{ad['median_pairs']:.0f} pairs/user",
            "median >= 5",
            ad["median_pairs"],                (5, 1e9),   (2, 4.9)),
        ("label balance",
            f"{lb['mean_normalized_consistency']:.3f} norm consistency (1.0=random)",
            "> 1.3",
            lb["mean_normalized_consistency"], (1.3, 1e9), (1.1, 1.3)),
        ("Krippendorff proxy",
            f"{kap['corrected_ratio']:.4f} corrected ratio (0=random)",
            "> 0.03",
            kap["corrected_ratio"],            (0.03, 1.0), (0.01, 0.03)),
        ("NN accuracy",
            f"{nna['mean_nn_accuracy']:.3f} mean accuracy (0.5=random)",
            "> 0.6",
            nna["mean_nn_accuracy"],           (0.6, 1.0), (0.55, 0.6)),
    ]

    if bsc:
        rows.extend([
            ("basis coherence",
                f"{bsc['corrected_ratio']:.4f} corrected ratio (0=random)",
                "> 0.005",
                bsc["corrected_ratio"],            (0.005, 1.0), None),
            ("population accuracy",
                f"{pa['accuracy']:.3f} held-out accuracy (0.5=random)",
                "> 0.6",
                pa["accuracy"],                    (0.6, 1.0), (0.55, 0.6)),
            ("held-out accuracy",
                f"{hoa['mean_accuracy']:.3f}  (n={hoa['n_users_evaluated']})",
                "> 0.6",
                hoa["mean_accuracy"],              (0.6, 1.0), (0.55, 0.6)),
        ])

    info_rows = []
    if uvd:
        info_rows.append(("user vec mean dist",   f"{uvd['mean_pairwise_distance']:.3f}"))
    if bue:
        info_rows.append(("basis entropy (norm)", f"{bue['normalized_mean_entropy']:.3f}"))

    # --- print ---
    W2 = 54        # value column width
    sep = "─" * (24 + W2 + 16 + 8)
    print(f"\n{'─'*len(sep)}")
    print(f"  Dataset: {name}  ({len(user_pref_embeddings)} users, K={K})")
    print(sep)
    print(f"  {'Metric':<22} {'Value':<{W2}} {'Threshold':<16} Status")
    print(sep)
    for metric, display, threshold, val, green, warn in rows:
        status, _ = _status(val, green, warn)
        print(f"  {metric:<22} {display:<{W2}} {threshold:<16} {status}")
    if info_rows:
        print(f"  {_D}{'─'*(len(sep)-2)}{_0}")
        for metric, display in info_rows:
            print(f"  {_D}{metric:<22} {display:<{W2}} {'—':<16} INFO{_0}")
    print(sep)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_embeddings_from_file(path: Path) -> list[torch.Tensor]:
    """Load a list of per-user embedding tensors from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return [t.float() for t in data]
    raise ValueError(f"Expected a list of tensors in {path}, got {type(data).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="Run LoRe suitability evaluation on a preference dataset."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to preference data (.jsonl, .parquet) or pre-computed embeddings (.pt).",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Treat data_path as a .pt file of pre-computed per-user embeddings "
             "(list of [n_prefs, D] tensors). Skips the embedding model entirely.",
    )
    parser.add_argument(
        "--basis",
        type=Path,
        default=None,
        dest="basis_path",
        help="Path to pretrained basis V_K*.pt. Default: auto-detect from config.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=8,
        help="Rank of the LoRe model (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for embedding model (default: cuda if available).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset label for the report header (default: filename stem).",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"Error: {args.data_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    name = args.name or args.data_path.stem

    if args.embeddings:
        # --- Pre-computed embeddings path (no model needed) ---
        print(f"Loading embeddings from {args.data_path}...", flush=True)
        user_pref_embeddings = _load_embeddings_from_file(args.data_path)
        n_users = len(user_pref_embeddings)
        n_pairs = sum(len(t) for t in user_pref_embeddings)
        print(f"  {n_users} users, {n_pairs} preference pairs", flush=True)
    else:
        # --- Raw text path (needs embedding model) ---
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading preferences from {args.data_path}...", flush=True)
        user_prefs = load_prefs(args.data_path)
        n_users = len(user_prefs)
        n_pairs = sum(len(v) for v in user_prefs.values())
        print(f"  {n_users} users, {n_pairs} preference pairs", flush=True)

        print("Loading embedding model...", flush=True)
        from apa.train_lore_bases import get_embedding_model
        model, tokenizer = get_embedding_model(device=device)

        print("Embedding preferences...", flush=True)
        user_pref_embeddings = embed_preferences(user_prefs, model, tokenizer, device=device)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- Load V ---
    if args.basis_path is not None:
        V_path = args.basis_path
    else:
        from apa.config import MODELS_DIR
        V_path = MODELS_DIR / f"V_K{args.K}.pt"

    if not V_path.exists():
        print(f"Error: basis file {V_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading basis from {V_path}...", flush=True)
    V = torch.load(V_path, map_location="cpu", weights_only=True).float()

    # --- Run report ---
    report(name, user_pref_embeddings, V, K=args.K)


if __name__ == "__main__":
    main()
