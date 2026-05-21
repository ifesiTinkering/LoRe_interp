"""
Question selection strategies — methods for picking which PRISM questions
to put to a historical (e.g. HistLlama) persona during synthetic
preference generation.

Strategies:
- random_subset: Uniformly random sample of N questions.
- select_by_ids: Filter by an explicit list of ``question_id``s
  (used to consume the curated question set in
  ``experiments/chosen_questions.jsonl``).
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


def random_subset(
    all_questions: pd.DataFrame,
    n_questions: int,
    config: dict,
) -> pd.DataFrame:
    """Select questions uniformly at random."""
    seed = config.get('seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_questions = min(n_questions, len(all_questions))
    indices = random.sample(range(len(all_questions)), n_questions)
    return all_questions.iloc[indices].reset_index(drop=True)


def select_by_ids(
    all_questions: pd.DataFrame,
    question_ids: list[int],
) -> pd.DataFrame:
    """Select specific questions by their ``question_id``.

    Args:
        all_questions: DataFrame with a ``question_id`` column.
        question_ids: List of question IDs to select.

    Returns:
        Filtered DataFrame containing only the requested questions.

    Raises:
        ValueError: If any requested IDs are not found.
    """
    available = set(all_questions["question_id"])
    missing = set(question_ids) - available
    if missing:
        raise ValueError(f"Question IDs not found in data: {sorted(missing)}")
    mask = all_questions["question_id"].isin(question_ids)
    result = all_questions[mask].drop_duplicates(subset="question_id")
    return result.reset_index(drop=True)
