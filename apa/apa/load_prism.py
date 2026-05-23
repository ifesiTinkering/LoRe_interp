"""
PRISM dataset loading, embedding generation, and checkpoint management.

CLI Usage:
    uv run python -m apa.load_prism              # Prepare data and generate all embeddings
    uv run python -m apa.load_prism --split train --n_samples 100  # Limit to train split
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from apa._logging import log
from apa._logging import log as _log


# =============================================================================
# Data Models (from LoRe/PRISM/prepare.py)
# =============================================================================

@dataclass
class Demographics:
    """User demographic information from survey."""
    self_description: str = ""
    preference: list[str] = field(default_factory=list)
    age: str = ""
    gender: str = ""
    education: str = ""
    employment: str = ""
    marital: str = ""
    english_proficiency: str = ""


@dataclass
class UserInfo:
    """User info with demographics and dialog IDs."""
    user_id: str
    dialog_ids: list[str] = field(default_factory=list)
    demographics: Demographics = field(default_factory=Demographics)
    system_string: str = ""


@dataclass
class Turn:
    """A single turn in a dialog with chosen/rejected utterances."""
    turn_nb: int
    user_utterance: list[str] = field(default_factory=list)
    chosen_utterance: list[str] = field(default_factory=list)
    rejected_utterance: list[str] = field(default_factory=list)


@dataclass
class DialogInfo:
    """Dialog info with turns and user ID."""
    dialog_id: str
    user_id: str
    turns: list[Turn] = field(default_factory=list)


# =============================================================================
# PRISM Data Preparation (from LoRe/PRISM/prepare.py)
# =============================================================================

def _download_prism_raw(output_dir: Path) -> tuple[Path, Path]:
    """Download raw PRISM JSONL files from HuggingFace."""
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)

    conv_path = output_dir / "conversations.jsonl"
    survey_path = output_dir / "survey.jsonl"

    if not conv_path.exists():
        _log("Downloading conversations.jsonl from HuggingFace...")
        downloaded = hf_hub_download(
            repo_id="HannahRoseKirk/prism-alignment",
            filename="conversations.jsonl",
            repo_type="dataset",
            local_dir=output_dir,
        )
        _log(f"Downloaded to {downloaded}")

    if not survey_path.exists():
        _log("Downloading survey.jsonl from HuggingFace...")
        downloaded = hf_hub_download(
            repo_id="HannahRoseKirk/prism-alignment",
            filename="survey.jsonl",
            repo_type="dataset",
            local_dir=output_dir,
        )
        _log(f"Downloaded to {downloaded}")

    return conv_path, survey_path


def _parse_prism_data(conv_path: Path, survey_path: Path) -> tuple[dict[str, UserInfo], dict[str, DialogInfo]]:
    """
    Parse PRISM JSONL files into structured user and dialog data.

    Returns:
        (user_data, dialog_data) - dicts keyed by user_id and dialog_id
    """
    _log("Parsing PRISM survey data...")
    user_data: dict[str, UserInfo] = {}

    with open(survey_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            user_id = entry.get("user_id", "")
            if not user_id:
                continue

            demographics = Demographics(
                self_description=entry.get("self_description", ""),
                preference=entry.get("preference", []),
                age=entry.get("age", ""),
                gender=entry.get("gender", ""),
                education=entry.get("education", ""),
                employment=entry.get("employment", ""),
                marital=entry.get("marital", ""),
                english_proficiency=entry.get("english_proficiency", ""),
            )

            user_data[user_id] = UserInfo(
                user_id=user_id,
                demographics=demographics,
                system_string=entry.get("system_string", ""),
            )

    _log(f"Parsed {len(user_data)} users from survey")

    _log("Parsing PRISM conversation data...")
    dialog_data: dict[str, DialogInfo] = {}

    with open(conv_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            dialog_id = entry.get("conversation_id", "")
            user_id = entry.get("user_id", "")
            if not dialog_id or not user_id:
                continue

            # Parse conversation history into turns
            conv_history = entry.get("conversation_history", [])
            turns_by_nb: dict[int, Turn] = {}

            for msg in conv_history:
                turn_nb = msg.get("turn", 0)
                role = msg.get("role", "")
                content = msg.get("content", "")
                if_chosen = msg.get("if_chosen", False)

                if turn_nb not in turns_by_nb:
                    turns_by_nb[turn_nb] = Turn(turn_nb=turn_nb)

                turn = turns_by_nb[turn_nb]
                if role == "user":
                    if content not in turn.user_utterance:
                        turn.user_utterance.append(content)
                elif role == "model":
                    if if_chosen:
                        if content not in turn.chosen_utterance:
                            turn.chosen_utterance.append(content)
                    else:
                        if content not in turn.rejected_utterance:
                            turn.rejected_utterance.append(content)

            # Sort turns by turn number
            turns = [turns_by_nb[nb] for nb in sorted(turns_by_nb.keys())]

            dialog_data[dialog_id] = DialogInfo(
                dialog_id=dialog_id,
                user_id=user_id,
                turns=turns,
            )

            # Add dialog_id to user's dialog_ids
            if user_id in user_data:
                user_data[user_id].dialog_ids.append(dialog_id)

    _log(f"Parsed {len(dialog_data)} dialogs from conversations")

    return user_data, dialog_data


def _split_users_and_dialogs(
    user_data: dict[str, UserInfo],
    seed: int = 123,
    seen_ratio: float = 0.8,
    train_ratio: float = 0.5,
    min_dialogs: int = 5,
) -> dict[str, Any]:
    """
    Split users 80/20 into seen/unseen, then split each user's dialogs 50/50.

    Args:
        user_data: Dict of user_id -> UserInfo
        seed: Random seed (default 123 for reproducibility)
        seen_ratio: Fraction of users that are "seen" (default 0.8)
        train_ratio: Fraction of each user's dialogs for training (default 0.5)
        min_dialogs: Minimum dialogs required per user (default 5)

    Returns:
        Dict with train_dialog_ids, test_dialog_ids, seen_user_ids, unseen_user_ids
    """
    _log(f"Splitting users with seed={seed}, seen_ratio={seen_ratio}, min_dialogs>{min_dialogs}")

    # Filter to users with >min_dialogs
    valid_users = [
        uid for uid, uinfo in user_data.items()
        if len(uinfo.dialog_ids) > min_dialogs
    ]
    _log(f"Users with >{min_dialogs} dialogs: {len(valid_users)}")

    # Split users into seen/unseen
    random.seed(seed)
    random.shuffle(valid_users)

    n_seen = int(len(valid_users) * seen_ratio)
    seen_user_ids = valid_users[:n_seen]
    unseen_user_ids = valid_users[n_seen:]

    _log(f"Seen users: {len(seen_user_ids)}, Unseen users: {len(unseen_user_ids)}")

    # Split dialogs per user
    train_dialog_ids = []
    test_dialog_ids = []

    for user_id in valid_users:
        user_dialogs = user_data[user_id].dialog_ids.copy()
        random.shuffle(user_dialogs)

        n_train = int(len(user_dialogs) * train_ratio)
        train_dialog_ids.extend(user_dialogs[:n_train])
        test_dialog_ids.extend(user_dialogs[n_train:])

    _log(f"Train dialogs: {len(train_dialog_ids)}, Test dialogs: {len(test_dialog_ids)}")

    return {
        "train_dialog_ids": train_dialog_ids,
        "test_dialog_ids": test_dialog_ids,
        "seen_user_ids": seen_user_ids,
        "unseen_user_ids": unseen_user_ids,
    }


def _create_comparison_dataset(
    dialog_data: dict[str, DialogInfo],
    split_ids: dict[str, Any],
    split: str,  # "train" or "test"
) -> list[dict]:
    """
    Create comparison dataset from dialogs for a given split.

    Each row represents one turn with a chosen/rejected pair.
    Note: Turns without rejected_utterance are kept (with empty list) to match
    the original LoRe data format.
    """
    dialog_ids = split_ids[f"{split}_dialog_ids"]
    seen_user_ids = set(split_ids["seen_user_ids"])

    rows = []
    for dialog_id in dialog_ids:
        if dialog_id not in dialog_data:
            continue

        dialog = dialog_data[dialog_id]
        user_id = dialog.user_id
        is_seen = user_id in seen_user_ids

        # Build conversation prompt up to each turn
        prompt_messages = []
        for turn in dialog.turns:
            # Skip turns without chosen utterance (need at least a chosen response)
            if not turn.chosen_utterance:
                continue

            # The prompt is the conversation history up to this turn
            current_prompt = prompt_messages.copy()

            # Add user utterance to prompt
            if turn.user_utterance:
                current_prompt.append({
                    "content": turn.user_utterance[0],
                    "role": "user",
                })

            # Create row for this turn
            # Keep rejected_utterance as list (may be empty) to match LoRe format
            row = {
                "data_source": "prism",
                "prompt": current_prompt,
                "ability": "chat",
                "reward_model": "human",
                "extra_info": {
                    "chosen_utterance": turn.chosen_utterance[0],
                    "dialog_id": dialog_id,
                    "rejected_utterance": turn.rejected_utterance,  # Keep as list for RM eval
                    "seen": is_seen,
                    "split": split,
                    "total_turn_nb": len(dialog.turns),
                    "turn_nb": turn.turn_nb,
                    "user_id": user_id,
                },
            }
            rows.append(row)

            # Add chosen response to prompt for next turn
            prompt_messages.append({
                "content": turn.user_utterance[0] if turn.user_utterance else "",
                "role": "user",
            })
            prompt_messages.append({
                "content": turn.chosen_utterance[0],
                "role": "assistant",
            })

    return rows


def prepare_prism_data(
    output_dir: Path | str | None = None,
    raw_data_dir: Path | str | None = None,
    seed: int = 123,
) -> tuple[Path, Path]:
    """
    Prepare PRISM parquet files with proper train/test splits.

    Args:
        output_dir: Directory for output files (default: NAS prism data dir)
        raw_data_dir: Directory for raw JSONL files (optional)
        seed: Random seed for reproducibility (default 123)

    Returns:
        (train_parquet_path, test_parquet_path)
    """
    from apa.config import PRISM_DATA_DIR

    if output_dir is None:
        output_dir = PRISM_DATA_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use output_dir for raw data if not specified
    if raw_data_dir is None:
        raw_data_dir = output_dir

    raw_data_dir = Path(raw_data_dir)

    _log("=" * 60)
    _log("PRISM Data Preparation")
    _log("=" * 60)
    _log(f"Output directory: {output_dir}")
    _log(f"Raw data directory: {raw_data_dir}")
    _log(f"Seed: {seed}")

    # Step 1: Download raw data if needed
    conv_path = raw_data_dir / "conversations.jsonl"
    survey_path = raw_data_dir / "survey.jsonl"

    if not conv_path.exists() or not survey_path.exists():
        conv_path, survey_path = _download_prism_raw(raw_data_dir)

    # Step 2: Parse data
    user_data, dialog_data = _parse_prism_data(conv_path, survey_path)

    # Step 3: Split users and dialogs
    split_ids = _split_users_and_dialogs(user_data, seed=seed)

    # Save intermediate files
    user_data_path = output_dir / "prism_data_user.json"
    dialog_data_path = output_dir / "prism_data_dialog.json"
    split_ids_path = output_dir / "prism_split_ids_50.json"

    _log("Saving intermediate JSON files...")

    # Convert dataclasses to dicts for JSON serialization
    user_dict = {
        uid: {
            "user_id": u.user_id,
            "dialog_ids": u.dialog_ids,
            "demographics": {
                "self_description": u.demographics.self_description,
                "preference": u.demographics.preference,
                "age": u.demographics.age,
                "gender": u.demographics.gender,
                "education": u.demographics.education,
                "employment": u.demographics.employment,
                "marital": u.demographics.marital,
                "english_proficiency": u.demographics.english_proficiency,
            },
            "system_string": u.system_string,
        }
        for uid, u in user_data.items()
    }

    dialog_dict = {
        did: {
            "dialog_id": d.dialog_id,
            "user_id": d.user_id,
            "turns": [
                {
                    "turn_nb": t.turn_nb,
                    "user_utterance": t.user_utterance,
                    "chosen_utterance": t.chosen_utterance,
                    "rejected_utterance": t.rejected_utterance,
                }
                for t in d.turns
            ],
        }
        for did, d in dialog_data.items()
    }

    with open(user_data_path, "w") as f:
        json.dump(user_dict, f, indent=4)
    with open(dialog_data_path, "w") as f:
        json.dump(dialog_dict, f, indent=4)
    with open(split_ids_path, "w") as f:
        json.dump(split_ids, f, indent=4)

    _log(f"Saved user data to {user_data_path}")
    _log(f"Saved dialog data to {dialog_data_path}")
    _log(f"Saved split IDs to {split_ids_path}")

    # Step 4: Create comparison datasets
    _log("Creating train comparison dataset...")
    train_rows = _create_comparison_dataset(dialog_data, split_ids, "train")
    _log(f"Train dataset: {len(train_rows)} rows")

    _log("Creating test comparison dataset...")
    test_rows = _create_comparison_dataset(dialog_data, split_ids, "test")
    _log(f"Test dataset: {len(test_rows)} rows")

    # Step 5: Save as parquet
    train_parquet = output_dir / "train.parquet"
    test_parquet = output_dir / "test.parquet"

    _log("Saving parquet files...")
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    train_df.to_parquet(train_parquet, index=False)
    test_df.to_parquet(test_parquet, index=False)

    _log(f"Saved train.parquet: {len(train_df)} rows")
    _log(f"Saved test.parquet: {len(test_df)} rows")

    _log("=" * 60)
    _log("PRISM data preparation complete!")
    _log("=" * 60)

    return train_parquet, test_parquet


# =============================================================================
# Data Loading
# =============================================================================

def get_user_column(df: pd.DataFrame) -> str | None:
    """Find the user identifier column ('user_id' or 'interaction_id')."""
    if 'user_id' in df.columns:
        return 'user_id'
    if 'interaction_id' in df.columns:
        return 'interaction_id'
    return None


def get_unique_users(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique user identifiers from DataFrame."""
    user_col = get_user_column(df)
    if user_col is None:
        return []
    return sorted(df[user_col].unique().tolist())


def load_prism_pairwise(
    path: Path | str | None = None,
    min_pairs_per_user: int = 0,
) -> pd.DataFrame:
    """
    Load PRISM pairwise preference data.

    Args:
        path: Path to CSV. If None, uses default from config.
        min_pairs_per_user: Filter to users with at least this many pairs

    Returns:
        DataFrame with user_id, question_id, prompt, response_1, response_2, etc.
    """
    if path is None:
        from apa.config import HISTORICAL_PREFS_DATA
        path = HISTORICAL_PREFS_DATA / "prism" / "questions_pairwise.csv"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PRISM pairwise data not found at {path}")

    df = pd.read_csv(path, sep="\t")

    if 'user_id' not in df.columns and 'interaction_id' in df.columns:
        df['user_id'] = df['interaction_id']

    if min_pairs_per_user > 0:
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_pairs_per_user].index
        df = df[df['user_id'].isin(valid_users)]
        print(f"Filtered to {len(valid_users)} users with >= {min_pairs_per_user} pairs")

    return df


# =============================================================================
# Dataset Class
# =============================================================================

class PRISMDataset(Dataset):
    """PyTorch Dataset for PRISM pairwise preference data."""

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        labels: np.ndarray,
        user_ids: np.ndarray | None = None,
    ):
        """
        Args:
            embeddings: Dict with 'response_1_embeddings' and 'response_2_embeddings'
            labels: Array of shape (n_samples,) with 0 or 1 indicating preference
            user_ids: Optional array of user IDs for each sample
        """
        self.response_1 = torch.tensor(embeddings['response_1_embeddings'], dtype=torch.float32)
        self.response_2 = torch.tensor(embeddings['response_2_embeddings'], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        assert len(self.response_1) == len(self.response_2) == len(self.labels)

        if user_ids is not None:
            unique_users = sorted(set(user_ids))
            self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
            self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
            self.user_indices = torch.tensor(
                [self.user_to_idx[uid] for uid in user_ids],
                dtype=torch.long
            )
            self._n_users = len(unique_users)
        else:
            self.user_to_idx = None
            self.idx_to_user = None
            self.user_indices = None
            self._n_users = 1

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            'response_1_embedding': self.response_1[idx],
            'response_2_embedding': self.response_2[idx],
            'label': self.labels[idx],
        }
        if self.user_indices is not None:
            item['user_idx'] = self.user_indices[idx]
        return item

    @property
    def embedding_dim(self) -> int:
        return self.response_1.shape[1]

    @property
    def n_users(self) -> int:
        return self._n_users

    def get_user_data(self, user_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all data for a specific user."""
        if self.user_to_idx is None:
            raise ValueError("Dataset was not initialized with user_ids")
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            raise KeyError(f"User {user_id} not found in dataset")
        mask = self.user_indices == user_idx
        return self.response_1[mask], self.response_2[mask], self.labels[mask]


# =============================================================================
# Embedding Grouping
# =============================================================================

def group_embeddings_by_user(
    train_embeddings: list[dict],
    test_embeddings: list[dict],
    device: str | torch.device = "cuda:0",
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Group embeddings by user and compute difference (chosen - rejected).

    Returns:
        (train_seen, train_unseen, test_seen, test_unseen) - lists of per-user tensors
    """
    def process_dataset(dataset, seen_value, split_name):
        split_label = "seen" if seen_value else "unseen"
        log(f"Processing {split_name} {split_label} dataset ({len(dataset)} examples)...")
        start_time = time.time()
        grouped = defaultdict(lambda: {"embeddings": []})
        skipped = 0
        processed = 0

        for idx, example in enumerate(dataset):
            extra_info = example.get("extra_info", {})
            if extra_info.get("seen") == seen_value and extra_info.get("split") == split_name:
                user_id = extra_info.get("user_id")
                if user_id:
                    chosen_emb = extra_info.get("chosen_conv_embedding")
                    rejected_emb = extra_info.get("rejected_conv_embedding")
                    if chosen_emb is None or rejected_emb is None:
                        skipped += 1
                        continue
                    chosen = torch.tensor(chosen_emb, dtype=torch.float32, device=device)
                    rejected = torch.tensor(rejected_emb, dtype=torch.float32, device=device)
                    grouped[user_id]["embeddings"].append(chosen - rejected)
                    processed += 1

            if (idx + 1) % max(1, len(dataset) // 10) == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(dataset) - idx - 1) / rate if rate > 0 else 0
                log(f"  Progress: {idx+1}/{len(dataset)} ({100*(idx+1)/len(dataset):.1f}%) | "
                    f"Processed: {processed} | Skipped: {skipped} | ETA: {remaining:.1f}s")

        log(f"  Stacking embeddings for {len(grouped)} users...")
        sorted_grouped = []
        count = 0
        for user_id in sorted(grouped.keys()):
            count += len(grouped[user_id]["embeddings"])
            sorted_grouped.append(torch.stack(grouped[user_id]["embeddings"]))

        elapsed = time.time() - start_time
        log(f"  Completed {split_name} {split_label}: {count} embeddings from {len(grouped)} users "
            f"({processed} processed, {skipped} skipped) in {elapsed:.1f}s")
        return sorted_grouped

    log("=" * 60)
    log("Grouping embeddings by user...")
    log("=" * 60)
    grouping_start = time.time()

    train_seen = process_dataset(train_embeddings, seen_value=True, split_name="train")
    train_unseen = process_dataset(train_embeddings, seen_value=False, split_name="train")
    test_seen = process_dataset(test_embeddings, seen_value=True, split_name="test")
    test_unseen = process_dataset(test_embeddings, seen_value=False, split_name="test")

    grouping_time = time.time() - grouping_start
    log(f"Embedding grouping completed in {grouping_time:.1f}s ({grouping_time/60:.1f} min)")
    log("=" * 60)

    return train_seen, train_unseen, test_seen, test_unseen


# =============================================================================
# Checkpoint Management
# =============================================================================

class CheckpointManager:
    """Manages checkpointing for long-running training processes."""

    def __init__(
        self,
        checkpoint_dir: Path,
        name: str,
        checkpoint_interval: int = 100,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.name = name
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self.name}_checkpoint.pt"

    def save_checkpoint(self, state: dict[str, Any], iteration: int) -> None:
        checkpoint = {'iteration': iteration, 'state': state}
        torch.save(checkpoint, self.checkpoint_path)
        print(f"[Checkpoint] Saved at iteration {iteration}")

    def load_checkpoint(self) -> tuple[dict[str, Any], int] | None:
        if not self.checkpoint_path.exists():
            return None
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        print(f"[Checkpoint] Loaded from iteration {checkpoint['iteration']}")
        return checkpoint['state'], checkpoint['iteration']

    def maybe_save(self, state: dict[str, Any], iteration: int, force: bool = False) -> None:
        if force or (iteration > 0 and iteration % self.checkpoint_interval == 0):
            self.save_checkpoint(state, iteration)

    def cleanup(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"[Checkpoint] Removed: {self.checkpoint_path}")


def save_with_symlink(
    data: pd.DataFrame | torch.Tensor | dict,
    nas_path: Path,
    local_path: Path | None = None,
    sep: str = '\t',
) -> None:
    """Save data to NAS and optionally create symlink in local directory."""
    nas_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_csv(nas_path, sep=sep, index=False)
    elif isinstance(data, (torch.Tensor, dict)):
        torch.save(data, nas_path)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    print(f"Saved to {nas_path}")

    if local_path is not None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() or local_path.is_symlink():
            local_path.unlink()
        local_path.symlink_to(nas_path)
        print(f"Created symlink: {local_path} -> {nas_path}")


def load_from_nas(nas_path: Path) -> Any:
    """Load data from NAS path (auto-detects file type)."""
    if not nas_path.exists():
        raise FileNotFoundError(f"File not found: {nas_path}")

    suffix = nas_path.suffix.lower()
    if suffix in ['.pt', '.pth']:
        return torch.load(nas_path, map_location='cpu')
    elif suffix == '.pkl':
        import pickle
        with open(nas_path, 'rb') as f:
            return pickle.load(f)
    elif suffix in ['.csv', '.tsv']:
        sep = '\t' if suffix == '.tsv' else ','
        return pd.read_csv(nas_path, sep=sep)
    else:
        raise ValueError(f"Unknown file extension: {suffix}")


# =============================================================================
# CLI: Embedding Generation
# =============================================================================

def _generate_embeddings(dataset, model, tokenizer, device: str, output_path: Path, n_samples: int | None = None) -> list[dict]:
    """Generate embeddings for the dataset (internal CLI helper)."""
    from tqdm import tqdm

    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    dataset_size = len(dataset)
    _log(f"Generating embeddings for {dataset_size} examples...")
    start_time = time.time()

    # CHECKPOINT PATCH (ifesiTinkering fork): resume from partial save on disconnect.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.pt")
    if partial_path.exists():
        embeddings_data = torch.load(partial_path, weights_only=False)
        resume_from = len(embeddings_data)
        _log(f"Resuming from checkpoint {partial_path}: {resume_from}/{dataset_size} already done")
    else:
        embeddings_data = []
        resume_from = 0
    CHECKPOINT_EVERY = 200

    for idx, entry in enumerate(tqdm(dataset, desc="Generating embeddings", total=dataset_size), 1):
        if idx <= resume_from:
            continue
        if not isinstance(entry, dict):
            entry = dict(entry)
        else:
            entry = dict(entry)
            if "extra_info" in entry and isinstance(entry["extra_info"], dict):
                entry["extra_info"] = dict(entry["extra_info"])

        extra_info = entry.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        entry["extra_info"] = extra_info

        prompt = entry.get("prompt", [])
        chosen_utterance = extra_info.get("chosen_utterance", "")
        rejected_utterance = extra_info.get("rejected_utterance", "")

        chosen = [{"content": chosen_utterance, "role": "assistant"}]
        rejected = [{"content": rejected_utterance, "role": "assistant"}]
        chosen_conv = prompt + chosen
        rejected_conv = prompt + rejected

        # Generate chosen embedding
        try:
            tokenized = tokenizer.apply_chat_template(chosen_conv, tokenize=True, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(tokenized)
                embedding = output.last_hidden_state[0, -1].cpu()
            entry["extra_info"]["chosen_conv_embedding"] = embedding
            del tokenized, output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str and "memory" in error_str:
                _log(f"CUDA OOM at example {idx} (chosen), skipping...")
                entry["extra_info"]["chosen_conv_embedding"] = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                raise

        # Generate rejected embedding
        try:
            tokenized = tokenizer.apply_chat_template(rejected_conv, tokenize=True, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(tokenized)
                embedding = output.last_hidden_state[0, -1].cpu()
            entry["extra_info"]["rejected_conv_embedding"] = embedding
            del tokenized, output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str and "memory" in error_str:
                _log(f"CUDA OOM at example {idx} (rejected), skipping...")
                entry["extra_info"]["rejected_conv_embedding"] = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                raise

        embeddings_data.append(entry)

        # CHECKPOINT PATCH: periodic partial save so a Colab disconnect at hour N doesn't lose hours of work.
        if idx % CHECKPOINT_EVERY == 0:
            torch.save(embeddings_data, partial_path)
            _log(f"Checkpoint saved at {idx}/{dataset_size} -> {partial_path}")

        if idx % 100 == 0 and device.startswith("cuda"):
            torch.cuda.empty_cache()

        if idx % max(1, dataset_size // 10) == 0 or idx == dataset_size:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (dataset_size - idx) / rate if rate > 0 else 0
            _log(f"Progress: {idx}/{dataset_size} ({100*idx/dataset_size:.1f}%) | Rate: {rate:.1f} ex/s | ETA: {remaining:.1f}s")

    total_time = time.time() - start_time
    _log(f"Completed {dataset_size} examples in {total_time:.1f}s ({total_time/60:.1f} min)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings_data, output_path)
    _log(f"Saved embeddings to {output_path}")

    # CHECKPOINT PATCH: clean up partial after final save succeeds.
    if partial_path.exists():
        partial_path.unlink()
        _log(f"Removed partial checkpoint {partial_path}")

    return embeddings_data


def main() -> None:
    """CLI entry point for data preparation and embedding generation."""
    from apa.config import configure_environment, EMBEDDINGS_DIR, PRISM_DATA_DIR

    parser = argparse.ArgumentParser(
        description="Prepare PRISM data and embeddings for LoRe training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_samples", type=int, default=None, help="Limit to first N samples (for testing)")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"], help="Which split(s) to process")
    parser.add_argument("--model", type=str, default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", help="Embedding model to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run model on")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (uses EMBEDDINGS_DIR if not specified)")
    parser.add_argument("--data_dir", type=str, default=None, help="PRISM data directory (uses PRISM_DATA_DIR if not specified)")
    args = parser.parse_args()

    configure_environment()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    data_dir = Path(args.data_dir) if args.data_dir else PRISM_DATA_DIR

    # Embedding generation mode
    _log("=" * 60)
    _log("PRISM Embedding Generation")
    _log("=" * 60)
    _log(f"Model: {args.model}")
    _log(f"Device: {args.device}")
    _log(f"Split: {args.split}")
    if args.n_samples:
        _log(f"Limiting to {args.n_samples} samples per split")

    output_dir = Path(args.output_dir) if args.output_dir else EMBEDDINGS_DIR

    _log(f"Output directory: {output_dir}")

    # Prepare data (download raw JSONL and create parquet files)
    train_path, test_path = prepare_prism_data(output_dir=data_dir)

    _log(f"Loading PRISM data from {train_path.parent}")

    _log("Loading model and tokenizer...")
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        _log(f"GPU Memory: {total_mem:.2f} GB total")

    try:
        model = AutoModel.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto" if args.device.startswith("cuda") else None,
            attn_implementation="eager", num_labels=1, low_cpu_mem_usage=True,
        )
        _log(f"Model loaded on {args.device}")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        _log(f"CUDA error during model loading: {e}")
        _log("Falling back to CPU...")
        args.device = "cpu"
        model = AutoModel.from_pretrained(
            args.model, torch_dtype=torch.float32, device_map=None,
            attn_implementation="eager", num_labels=1, low_cpu_mem_usage=True,
        )
        model = model.to(args.device)
        _log("Model loaded on CPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    from datasets import load_dataset

    if args.split in ["train", "both"]:
        _log("=" * 60)
        _log("Processing TRAIN split")
        _log("=" * 60)
        train_dataset = load_dataset("parquet", data_files=str(train_path))["train"]
        _log(f"Train dataset: {len(train_dataset)} examples")
        _generate_embeddings(train_dataset, model, tokenizer, args.device, output_dir / "train.pkl", args.n_samples)

    if args.split in ["test", "both"]:
        _log("=" * 60)
        _log("Processing TEST split")
        _log("=" * 60)
        test_dataset = load_dataset("parquet", data_files=str(test_path))["train"]
        _log(f"Test dataset: {len(test_dataset)} examples")
        _generate_embeddings(test_dataset, model, tokenizer, args.device, output_dir / "test.pkl", args.n_samples)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log("=" * 60)
    _log("Done!")
    _log("=" * 60)


if __name__ == "__main__":
    main()
