"""
Sample or synthesize preference data for suitability evaluation baselines.

Usage (raw text):
    python -m apa.synthetic_prefs.sample_data sample path/to/prefs.jsonl -n 100
    python -m apa.synthetic_prefs.sample_data random path/to/prefs.jsonl -n 100

Usage (embeddings):
    python -m apa.synthetic_prefs.sample_data sample-emb -n 50
    python -m apa.synthetic_prefs.sample_data random-emb -n 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from apa.synthetic_prefs.eval_prefs import PreferencePair, load_prefs


def _flatten_prefs(
    user_prefs: dict[str, list[PreferencePair]],
) -> list[tuple[str, PreferencePair]]:
    """Flatten grouped prefs into a list of (user_id, pair) tuples."""
    flat = []
    for user_id, pairs in user_prefs.items():
        for pair in pairs:
            flat.append((user_id, pair))
    return flat


def sample_prefs(
    user_prefs: dict[str, list[PreferencePair]],
    n: int,
    seed: int = 0,
) -> dict[str, list[PreferencePair]]:
    """Randomly sample *n* preference pairs, preserving user IDs.

    If *n* exceeds the total number of pairs, all pairs are returned.
    """
    flat = _flatten_prefs(user_prefs)
    rng = random.Random(seed)
    n = min(n, len(flat))
    selected = rng.sample(flat, n)

    grouped: dict[str, list[PreferencePair]] = {}
    for user_id, pair in selected:
        grouped.setdefault(user_id, []).append(pair)
    return grouped


def random_prefs(
    user_prefs: dict[str, list[PreferencePair]],
    n: int,
    seed: int = 0,
) -> dict[str, list[PreferencePair]]:
    """Generate *n* preference pairs with randomly flipped labels.

    Pairs are sampled with replacement from the source data so *n* may
    exceed the original dataset size.
    """
    flat = _flatten_prefs(user_prefs)
    if not flat:
        return {}

    rng = random.Random(seed)
    result: dict[str, list[PreferencePair]] = {}

    for _ in range(n):
        user_id, pair = rng.choice(flat)
        if rng.random() < 0.5:
            new_pair = PreferencePair(pair.prompt, pair.rejected, pair.chosen)
        else:
            new_pair = PreferencePair(pair.prompt, pair.chosen, pair.rejected)
        result.setdefault(user_id, []).append(new_pair)

    return result


# ---------------------------------------------------------------------------
# Question-matched operations (filter to specific prompts)
# ---------------------------------------------------------------------------


def sample_prefs_by_questions(
    user_prefs: dict[str, list[PreferencePair]],
    question_prompts: set[str],
    n_users: int | None = None,
    seed: int = 0,
) -> dict[str, list[PreferencePair]]:
    """Filter preferences to only include pairs matching *question_prompts*,
    then optionally subsample to *n_users*.

    Args:
        user_prefs: Grouped preferences keyed by user_id.
        question_prompts: Set of prompt strings to keep.
        n_users: If given, randomly sample this many users from the result.
        seed: Random seed for user subsampling.

    Returns:
        Filtered (and optionally subsampled) preferences.
    """
    filtered: dict[str, list[PreferencePair]] = {}
    for uid, pairs in user_prefs.items():
        kept = [p for p in pairs if p.prompt in question_prompts]
        if kept:
            filtered[uid] = kept

    if n_users is not None and n_users < len(filtered):
        rng = random.Random(seed)
        selected = rng.sample(sorted(filtered.keys()), n_users)
        filtered = {uid: filtered[uid] for uid in selected}

    return filtered


def random_prefs_by_questions(
    user_prefs: dict[str, list[PreferencePair]],
    question_prompts: set[str],
    n_users: int | None = None,
    seed: int = 0,
) -> dict[str, list[PreferencePair]]:
    """Like :func:`sample_prefs_by_questions` but randomly flip chosen/rejected.

    Destroys preference signal while keeping the same text distribution —
    useful as a null baseline on the same questions.
    """
    filtered = sample_prefs_by_questions(user_prefs, question_prompts, n_users=None, seed=seed)
    rng = random.Random(seed)

    result: dict[str, list[PreferencePair]] = {}
    for uid, pairs in filtered.items():
        flipped = []
        for p in pairs:
            if rng.random() < 0.5:
                flipped.append(PreferencePair(p.prompt, p.rejected, p.chosen))
            else:
                flipped.append(p)
        result[uid] = flipped

    if n_users is not None and n_users < len(result):
        selected = rng.sample(sorted(result.keys()), n_users)
        result = {uid: result[uid] for uid in selected}

    return result


# ---------------------------------------------------------------------------
# Embedding-level operations (no re-embedding needed)
# ---------------------------------------------------------------------------

def load_prism_embeddings(device: str = "cpu") -> list[torch.Tensor]:
    """Load pre-computed PRISM train_seen embeddings."""
    from apa.config import EMBEDDINGS_DIR
    from apa.load_prism import group_embeddings_by_user

    train_emb = torch.load(EMBEDDINGS_DIR / "train.pkl", weights_only=False)
    test_emb = torch.load(EMBEDDINGS_DIR / "test.pkl", weights_only=False)
    train_seen, _, _, _ = group_embeddings_by_user(train_emb, test_emb, device=device)
    return train_seen


def sample_embeddings(
    user_embeddings: list[torch.Tensor],
    n_users: int,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Randomly select *n_users* from a list of per-user embedding tensors."""
    rng = random.Random(seed)
    n_users = min(n_users, len(user_embeddings))
    return rng.sample(user_embeddings, n_users)


def random_embeddings(
    user_embeddings: list[torch.Tensor],
    n_users: int,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Create *n_users* synthetic users by randomly flipping embedding signs.

    Each synthetic user is a copy of a randomly chosen real user, but each
    preference vector's sign is independently flipped with probability 0.5.
    This destroys label signal while preserving magnitude structure.
    """
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    result = []
    for _ in range(n_users):
        src = rng.choice(user_embeddings)
        signs = torch.where(
            torch.rand(len(src), generator=gen) < 0.5,
            torch.tensor(1.0), torch.tensor(-1.0),
        )
        result.append(src * signs.unsqueeze(1))
    return result


def _write_jsonl(
    user_prefs: dict[str, list[PreferencePair]],
    out: Path,
) -> None:
    """Write grouped preferences to a JSONL file."""
    with open(out, "w") as f:
        for user_id, pairs in sorted(user_prefs.items()):
            for pair in pairs:
                json.dump(
                    {
                        "user_id": user_id,
                        "prompt": pair.prompt,
                        "chosen": pair.chosen,
                        "rejected": pair.rejected,
                    },
                    f,
                )
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sample or synthesize preference data for evaluation baselines."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- raw text subcommands ---
    for name, help_text in [
        ("sample", "Randomly sample N pairs from the source file."),
        ("random", "Generate N pairs with randomly flipped labels."),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("data_path", type=Path, help="Source preference file (.jsonl or .parquet).")
        sp.add_argument("-n", type=int, required=True, help="Number of preference pairs to produce.")
        sp.add_argument("-o", "--output", type=Path, default=None, help="Output JSONL path (default: stdout).")
        sp.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")

    # --- embedding subcommands ---
    for name, help_text in [
        ("sample-emb", "Sample N users from pre-computed PRISM embeddings."),
        ("random-emb", "Generate N random-label users from PRISM embeddings."),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("-n", type=int, required=True, help="Number of users.")
        sp.add_argument("-o", "--output", type=Path, required=True, help="Output .pt file.")
        sp.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")

    args = parser.parse_args()

    # --- embedding commands ---
    if args.command in ("sample-emb", "random-emb"):
        embeddings = load_prism_embeddings()

        if args.command == "sample-emb":
            data = sample_embeddings(embeddings, args.n, seed=args.seed)
        else:
            data = random_embeddings(embeddings, args.n, seed=args.seed)

        torch.save(data, args.output)
        n_pairs = sum(len(t) for t in data)
        print(f"Saved {len(data)} users ({n_pairs} pairs) to {args.output}", file=sys.stderr)
        return

    # --- raw text commands ---
    if not args.data_path.exists():
        print(f"Error: {args.data_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    user_prefs = load_prefs(args.data_path)

    if args.command == "sample":
        result = sample_prefs(user_prefs, args.n, seed=args.seed)
    else:
        result = random_prefs(user_prefs, args.n, seed=args.seed)

    n_pairs = sum(len(v) for v in result.values())
    n_users = len(result)

    if args.output:
        _write_jsonl(result, args.output)
        print(f"Wrote {n_pairs} pairs ({n_users} users) to {args.output}", file=sys.stderr)
    else:
        for user_id, pairs in sorted(result.items()):
            for pair in pairs:
                json.dump(
                    {
                        "user_id": user_id,
                        "prompt": pair.prompt,
                        "chosen": pair.chosen,
                        "rejected": pair.rejected,
                    },
                    sys.stdout,
                )
                sys.stdout.write("\n")
        print(f"# {n_pairs} pairs ({n_users} users)", file=sys.stderr)


if __name__ == "__main__":
    main()
