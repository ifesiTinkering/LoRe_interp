"""
Few-shot LoRe adaptation and scoring for new users.

This module provides:
- Few-shot adaptation: fit per-user weight vectors given preference data and
  pre-trained LoRe bases (V).
- LoReScorer: unified scoring API for any user (PRISM, historical, or newly adapted).
- CLI for running adaptation on JSONL preference files.

CLI Usage:
    python -m apa.lore_adapt prefs.jsonl --K 8
    python -m apa.lore_adapt prefs.jsonl --K 8 --test_frac 0.2 --name my_experiment
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from apa._logging import log


# =============================================================================
# Adaptation Functions
# =============================================================================


def embed_and_index_preferences(
    user_prefs: dict[str, list],
    model: Any,
    tokenizer: Any,
    device: str = "cuda",
) -> tuple[list[str], list[torch.Tensor]]:
    """
    Embed user preferences and return user IDs alongside embeddings.

    Wraps eval_prefs.embed_preferences to also return the sorted user_id list
    so the caller knows which index corresponds to which user.

    Args:
        user_prefs: Mapping from user_id to list of PreferencePair.
        model: Skywork-Reward model.
        tokenizer: Corresponding tokenizer.
        device: Device for inference.

    Returns:
        (sorted_user_ids, per_user_embeddings) where each embedding tensor
        has shape [n_prefs, embed_dim].
    """
    from apa.synthetic_prefs.eval_prefs import embed_preferences

    user_ids = sorted(user_prefs.keys())
    embeddings = embed_preferences(user_prefs, model, tokenizer, device=device)
    assert len(embeddings) == len(user_ids), (
        f"embed_preferences returned {len(embeddings)} users but expected {len(user_ids)}; "
        "iteration order may have changed"
    )
    return user_ids, embeddings


def split_train_test(
    user_embeddings: list[torch.Tensor],
    test_frac: float = 0.2,
    min_total: int = 4,
    seed: int = 42,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """
    Split per-user embeddings into train and test sets.

    Users with fewer than min_total preferences get all prefs in train
    and None for test.

    Args:
        user_embeddings: Per-user list of [n_prefs, D] tensors.
        test_frac: Fraction of prefs to hold out for test.
        min_total: Minimum prefs needed to create a test split.
        seed: RNG seed for reproducibility.

    Returns:
        (train_list, test_list) where test_list[i] is None if user i
        had fewer than min_total prefs.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    train_list = []
    test_list: list[torch.Tensor | None] = []

    for X in user_embeddings:
        n = len(X)
        if n < min_total:
            train_list.append(X)
            test_list.append(None)
            continue

        idx = torch.randperm(n, generator=rng)
        X_shuffled = X[idx]
        n_test = max(1, int(n * test_frac))
        train_list.append(X_shuffled[n_test:])
        test_list.append(X_shuffled[:n_test])

    return train_list, test_list


def adapt_users(
    user_ids: list[str],
    train_embeddings: list[torch.Tensor],
    V: torch.Tensor,
    num_iterations: int = 500,
    learning_rate: float = 0.5,
) -> dict[str, dict]:
    """
    Few-shot adapt per-user weight vectors using pre-trained bases.

    Args:
        user_ids: List of user IDs (parallel to train_embeddings).
        train_embeddings: Per-user list of [n_prefs, D] diff tensors.
        V: Pre-trained basis matrix [D, K].
        num_iterations: Gradient descent iterations per user.
        learning_rate: Adam learning rate.

    Returns:
        Dict mapping user_id to {'w': tensor[K], 'n_train_prefs': int}.
    """
    from apa.train_lore_bases import get_device, learn_multiple_few_shot

    device = get_device()
    V = V.to(device)
    train_embeddings = [x.to(device) for x in train_embeddings]

    W_list = learn_multiple_few_shot(
        train_embeddings, V, num_iterations=num_iterations, learning_rate=learning_rate,
    )

    results = {}
    for i, user_id in enumerate(user_ids):
        results[user_id] = {
            'w': W_list[i].detach().cpu(),
            'n_train_prefs': len(train_embeddings[i]),
        }
    return results


def evaluate_adapted(
    results: dict[str, dict],
    user_ids: list[str],
    train_embeddings: list[torch.Tensor],
    test_embeddings: list[torch.Tensor | None],
    V: torch.Tensor,
) -> dict:
    """
    Evaluate adapted user vectors on train and test data.

    Mutates results in-place to add 'train_accuracy' and 'test_accuracy' fields.

    Returns:
        Aggregate stats dict with mean/std for train and test accuracy.
    """
    from apa.train_lore_bases import evaluate_model

    # Ensure everything is on CPU for evaluation
    V = V.detach().cpu().float()

    train_accs = []
    test_accs = []

    for i, user_id in enumerate(user_ids):
        w = results[user_id]['w'].cpu().float()

        train_acc = evaluate_model(train_embeddings[i].cpu(), V, w)
        results[user_id]['train_accuracy'] = train_acc
        train_accs.append(train_acc)

        if test_embeddings[i] is not None:
            test_acc = evaluate_model(test_embeddings[i].cpu(), V, w)
            results[user_id]['test_accuracy'] = test_acc
            test_accs.append(test_acc)

    stats = {
        'train_accuracy_mean': float(np.mean(train_accs)),
        'train_accuracy_std': float(np.std(train_accs)),
        'n_users': len(user_ids),
    }
    if test_accs:
        stats['test_accuracy_mean'] = float(np.mean(test_accs))
        stats['test_accuracy_std'] = float(np.std(test_accs))
        stats['n_users_with_test'] = len(test_accs)

    return stats


def save_adapted(
    results: dict[str, dict],
    output_path: Path,
    metadata: dict | None = None,
) -> None:
    """Save adapted user vectors to a checkpoint file."""
    checkpoint = {
        'users': results,
        'metadata': metadata or {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_adapted(checkpoint_path: Path | str) -> dict:
    """Load an adapted user vectors checkpoint."""
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)


# =============================================================================
# LoRe Scorer
# =============================================================================


class LoReScorer:
    """
    Unified scoring API for LoRe user models.

    Loads user vectors from any source (adapted, PRISM, historical) into a
    single registry, then scores prompt+response pairs for any registered user.

    Methods:
        score(user_id, prompt, response) — embed text and score (lazy-loads model)
        score_embedding(user_id, embedding) — score a pre-computed embedding

    Usage:
        scorer = LoReScorer.from_checkpoint("/path/to/V_K8.pt")
        scorer.load_adapted_users("/path/to/W_adapted_hist_top3.pt")
        score = scorer.score("hist_C013_01", "What is justice?", "Justice is...")
    """

    def __init__(self, V: torch.Tensor):
        self.V = V.float()
        self.user_registry: dict[str, torch.Tensor] = {}
        self._embedding_model = None
        self._embedding_tokenizer = None

    @classmethod
    def from_checkpoint(cls, V_path: str | Path) -> "LoReScorer":
        """Create a scorer from a V basis checkpoint."""
        from apa.train_lore_bases import LoReRewardModel

        model = LoReRewardModel.load(str(V_path), device='cpu')
        return cls(model.V.data.clone())

    def load_adapted_users(self, checkpoint_path: str | Path) -> int:
        """Load users from an adapted checkpoint (W_adapted_*.pt)."""
        checkpoint = load_adapted(checkpoint_path)
        users = checkpoint['users']
        count = 0
        for user_id, data in users.items():
            self.user_registry[user_id] = data['w'].float()
            count += 1
        return count

    def load_prism_users(
        self,
        W_path: str | Path,
        user_mapping_path: str | Path | None = None,
    ) -> int:
        """Load PRISM users from W_seen_K*.pt."""
        W = torch.load(W_path, map_location='cpu', weights_only=False)

        if user_mapping_path and Path(user_mapping_path).exists():
            with open(user_mapping_path, 'r') as f:
                user_to_idx = json.load(f)
            idx_to_user = {v: k for k, v in user_to_idx.items()}
        else:
            idx_to_user = {i: f"prism_user_{i}" for i in range(W.shape[0])}

        for idx in range(W.shape[0]):
            user_id = idx_to_user.get(idx, f"prism_user_{idx}")
            self.user_registry[user_id] = W[idx].float()

        return W.shape[0]

    def _ensure_embedding_model(self) -> None:
        """Lazy-load the Skywork reward embedding model + tokenizer."""
        if self._embedding_model is None:
            from apa.train_lore_bases import get_embedding_model
            self._embedding_model, self._embedding_tokenizer = get_embedding_model()

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Batched-embed a list of texts using the scorer's embedding model.

        Lazy-loads the embedding model on first call. Returns a float32
        tensor of shape (len(texts), embedding_dim). 
        """
        from apa.train_lore_bases import embed_texts as _embed_texts

        self._ensure_embedding_model()
        arr = _embed_texts(
            texts,
            model=self._embedding_model,
            tokenizer=self._embedding_tokenizer,
            show_progress=False,
        )
        return torch.tensor(arr, dtype=torch.float32)

    def score(self, user_id: str, prompt: str, response: str) -> float:
        """
        Score a prompt+response pair for a given user.

        Lazily loads the embedding model on first call.

        Args:
            user_id: Registered user ID.
            prompt: The user prompt / question.
            response: The LLM response to score.

        Returns:
            Scalar reward score (higher = more preferred by this user).

        Raises:
            KeyError: If user_id is not in the registry.
        """
        if user_id not in self.user_registry:
            raise KeyError(
                f"User '{user_id}' not found. "
                f"Registered users: {len(self.user_registry)}"
            )

        from apa.train_lore_bases import _format_for_embedding, _extract_embedding

        self._ensure_embedding_model()

        text = _format_for_embedding(prompt, response, self._embedding_tokenizer)
        device = str(next(self._embedding_model.parameters()).device)
        embedding = _extract_embedding(
            self._embedding_model, self._embedding_tokenizer, text, device,
        )
        embedding = torch.tensor(embedding, dtype=torch.float32)

        return self.score_embedding(user_id, embedding)

    def score_embedding(self, user_id: str, embedding: torch.Tensor) -> float:
        """Score a pre-computed embedding for a given user."""
        if user_id not in self.user_registry:
            raise KeyError(
                f"User '{user_id}' not found. "
                f"Registered users: {len(self.user_registry)}"
            )
        w = self.user_registry[user_id]
        V = self.V
        result = embedding @ V @ w
        return float(result)

    def get_user_ids(self) -> list[str]:
        """Return all registered user IDs."""
        return list(self.user_registry.keys())

    def has_user(self, user_id: str) -> bool:
        """Check if a user is registered."""
        return user_id in self.user_registry


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point for few-shot LoRe adaptation."""
    from apa.config import configure_environment, MODELS_DIR
    from apa.synthetic_prefs.eval_prefs import load_prefs_jsonl
    from apa.train_lore_bases import get_embedding_model

    parser = argparse.ArgumentParser(
        description="Few-shot LoRe adaptation for new users",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prefs_path", type=Path, help="Path to JSONL preferences file")
    parser.add_argument("--K", type=int, default=8, help="LoRe rank (must match a V_K*.pt checkpoint)")
    parser.add_argument("--basis", type=Path, default=None, dest="basis_path",
                        help="Path to V basis checkpoint. Default: MODELS_DIR/V_K{K}.pt")
    parser.add_argument("--num_iterations", type=int, default=500, help="Few-shot iterations")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Few-shot learning rate")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction of prefs held out for test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory. Default: MODELS_DIR")
    parser.add_argument("--device", type=str, default=None, help="Device for embedding model")
    parser.add_argument("--name", type=str, default=None, help="Dataset label for output filename")
    args = parser.parse_args()

    configure_environment()

    script_start = time.time()
    log("=" * 60)
    log("Few-shot LoRe adaptation")
    log("=" * 60)

    # --- Load preferences ---
    log(f"Loading preferences from {args.prefs_path}...")
    user_prefs = load_prefs_jsonl(args.prefs_path)
    n_users = len(user_prefs)
    n_pairs = sum(len(v) for v in user_prefs.values())
    log(f"  {n_users} users, {n_pairs} preference pairs")

    # --- Embed ---
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading embedding model (device={device})...")
    model, tokenizer = get_embedding_model(device=device)

    log(f"Embedding preferences ({n_pairs * 2} texts)...")
    embed_start = time.time()
    user_ids, user_embeddings = embed_and_index_preferences(user_prefs, model, tokenizer, device=device)
    embed_time = time.time() - embed_start
    log(f"  Embedding done in {embed_time:.1f}s")

    # Free embedding model memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Split train/test ---
    log(f"Splitting train/test (test_frac={args.test_frac}, seed={args.seed})...")
    train_embs, test_embs = split_train_test(user_embeddings, test_frac=args.test_frac, seed=args.seed)
    n_with_test = sum(1 for t in test_embs if t is not None)
    log(f"  {n_with_test}/{n_users} users have test data")

    # --- Load V ---
    output_dir = args.output_dir or MODELS_DIR
    V_path = args.basis_path or (MODELS_DIR / f"V_K{args.K}.pt")
    log(f"Loading basis V from {V_path}...")
    V = torch.load(V_path, map_location='cpu')
    if isinstance(V, dict):
        V = V.get('V', V.get('basis_matrix'))
    log(f"  V shape: {V.shape} (K={V.shape[1]})")

    # --- Adapt ---
    log(f"Adapting {n_users} users ({args.num_iterations} iterations, lr={args.learning_rate})...")
    adapt_start = time.time()
    results = adapt_users(user_ids, train_embs, V, args.num_iterations, args.learning_rate)
    adapt_time = time.time() - adapt_start
    log(f"  Adaptation done in {adapt_time:.1f}s")

    # --- Evaluate ---
    log("Evaluating...")
    stats = evaluate_adapted(results, user_ids, train_embs, test_embs, V)

    # --- Print report ---
    log("")
    log("Per-user results:")
    log("-" * 80)
    log(f"{'User ID':<20} {'Train Prefs':>11} {'Train Acc':>10} {'Test Acc':>10}")
    log("-" * 80)
    for user_id in user_ids:
        r = results[user_id]
        test_str = f"{r['test_accuracy']*100:>9.1f}%" if 'test_accuracy' in r else "       N/A"
        log(f"{user_id:<20} {r['n_train_prefs']:>11} {r['train_accuracy']*100:>9.1f}% {test_str}")
    log("-" * 80)
    test_mean_str = f"{stats['test_accuracy_mean']*100:>9.1f}%" if 'test_accuracy_mean' in stats else "       N/A"
    log(f"{'MEAN':<20} {'':>11} {stats['train_accuracy_mean']*100:>9.1f}% {test_mean_str}")
    log("")

    # --- Save ---
    name = args.name or f"{args.prefs_path.stem}_K{args.K}"
    output_path = Path(output_dir) / f"W_adapted_{name}.pt"
    metadata = {
        'source': str(args.prefs_path),
        'K': args.K,
        'num_iterations': args.num_iterations,
        'learning_rate': args.learning_rate,
        'test_frac': args.test_frac,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'stats': stats,
    }
    save_adapted(results, output_path, metadata=metadata)
    log(f"Saved adapted vectors to {output_path}")

    log("=" * 60)
    log(f"Done! Total runtime: {time.time() - script_start:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
