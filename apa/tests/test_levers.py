"""
Unit tests for strategy modules.
"""

import pytest
import numpy as np
import pandas as pd

from apa.levers.voter_aggregation import (
    borda_count,
    plurality,
    copeland,
    instant_runoff,
)
from apa.levers.query_selection import random_subset, select_by_ids
from apa.levers.voter_sampling import (
    _normalize_source_label,
    parse_jury_source_spec,
    per_group_sampling,
    random_sampling,
    stratified_sampling,
    temporal_mix_sampling,
)


class TestVoterAggregation:
    """Tests for ranking aggregation strategies."""

    def test_borda_count_basic(self):
        """Test basic Borda count."""
        rankings = {
            'user_1': [0, 1, 2],
            'user_2': [0, 2, 1],
            'user_3': [1, 0, 2],
        }

        result = borda_count(rankings, {})

        assert len(result) == 3
        assert result[0] == 0  # 0 has highest Borda score (2+2+1=5)
        assert set(result) == {0, 1, 2}

    def test_borda_count_empty(self):
        """Test Borda count with empty input."""
        result = borda_count({}, {})
        assert result == []

    def test_plurality_basic(self):
        """Test basic plurality voting."""
        rankings = {
            'user_1': [0, 1, 2],
            'user_2': [0, 2, 1],
            'user_3': [1, 0, 2],
            'user_4': [0, 1, 2],
        }

        result = plurality(rankings, {})

        assert result[0] == 0  # 0 has most first-place votes (3)

    def test_copeland_basic(self):
        """Test basic Copeland method."""
        rankings = {
            'user_1': [0, 1, 2],
            'user_2': [0, 2, 1],
            'user_3': [0, 1, 2],
        }

        result = copeland(rankings, {})

        assert result[0] == 0  # 0 beats both 1 and 2 pairwise

    def test_instant_runoff_basic(self):
        """Test basic instant runoff voting."""
        rankings = {
            'user_1': [0, 1, 2],
            'user_2': [1, 0, 2],
            'user_3': [1, 2, 0],
            'user_4': [0, 2, 1],
        }

        result = instant_runoff(rankings, {})

        assert len(result) == 3
        assert set(result) == {0, 1, 2}


class TestVoterSampling:
    """Tests for user sampling strategies."""

    def test_random_sampling_basic(self):
        """Test basic random sampling."""
        all_users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5']

        result = random_sampling(all_users, None, 3, {})

        assert len(result) == 3
        assert all(u in all_users for u in result)
        assert len(set(result)) == 3  # No duplicates

    def test_stratified_sampling_basic(self):
        """Test basic stratified sampling."""
        all_users = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
        metadata = {
            'u1': {'century': 'C013'},
            'u2': {'century': 'C013'},
            'u3': {'century': 'C017'},
            'u4': {'century': 'C017'},
            'u5': {'century': 'C021'},
            'u6': {'century': 'C021'},
        }

        result = stratified_sampling(all_users, metadata, 6, {'stratify_by': 'century'})

        assert len(result) == 6
        # Should have some from each group
        centuries = [metadata[u]['century'] for u in result if u in metadata]
        assert len(set(centuries)) >= 2

    def test_stratified_no_metadata(self):
        """Test stratified falls back to random without metadata."""
        all_users = ['u1', 'u2', 'u3']
        result = stratified_sampling(all_users, None, 2, {})
        assert len(result) == 2

    def test_temporal_mix_sampling(self):
        """Test temporal mix sampling."""
        all_users = ['modern_1', 'modern_2', 'hist_1', 'hist_2']
        metadata = {
            'modern_1': {'century': 'C021'},
            'modern_2': {'century': 'C021'},
            'hist_1': {'century': 'C013'},
            'hist_2': {'century': 'C017'},
        }

        result = temporal_mix_sampling(all_users, metadata, 4, {'historical_ratio': 0.5})

        assert len(result) == 4


class TestQuerySelection:
    """Tests for question selection strategies."""

    def test_random_subset_basic(self):
        """Test basic random subset selection."""
        df = pd.DataFrame({
            'question_id': range(10),
            'prompt': [f'q{i}' for i in range(10)],
        })

        result = random_subset(df, 5, {'seed': 42})

        assert len(result) == 5
        assert list(result.columns) == list(df.columns)

    def test_random_subset_reproducible(self):
        """Test random subset is reproducible with seed."""
        df = pd.DataFrame({
            'question_id': range(100),
            'prompt': [f'q{i}' for i in range(100)],
        })

        result1 = random_subset(df, 10, {'seed': 42})
        result2 = random_subset(df, 10, {'seed': 42})

        assert result1['question_id'].tolist() == result2['question_id'].tolist()

    def test_random_subset_caps_n(self):
        """Test random subset caps n_questions at available."""
        df = pd.DataFrame({
            'question_id': range(5),
            'prompt': [f'q{i}' for i in range(5)],
        })

        result = random_subset(df, 100, {})

        assert len(result) == 5


class TestSelectByIds:
    """Tests for select_by_ids question selection."""

    def test_basic_selection(self):
        df = pd.DataFrame({
            'question_id': [10, 20, 30, 40, 50],
            'prompt': ['a', 'b', 'c', 'd', 'e'],
        })
        result = select_by_ids(df, [20, 40])
        assert len(result) == 2
        assert set(result['question_id']) == {20, 40}

    def test_preserves_all_columns(self):
        df = pd.DataFrame({
            'question_id': [1, 2, 3],
            'prompt': ['a', 'b', 'c'],
            'response_1': ['r1a', 'r1b', 'r1c'],
        })
        result = select_by_ids(df, [2])
        assert list(result.columns) == list(df.columns)
        assert result.iloc[0]['prompt'] == 'b'

    def test_missing_ids_raises(self):
        df = pd.DataFrame({'question_id': [1, 2, 3], 'prompt': ['a', 'b', 'c']})
        with pytest.raises(ValueError, match="not found"):
            select_by_ids(df, [1, 99])

    def test_duplicate_rows_deduped(self):
        """If a question_id appears multiple times, only one row is kept."""
        df = pd.DataFrame({
            'question_id': [1, 1, 2],
            'prompt': ['a', 'a', 'b'],
            'interaction_id': ['u1', 'u2', 'u3'],
        })
        result = select_by_ids(df, [1])
        assert len(result) == 1


class TestPerGroupSampling:
    """Tests for the per_group_sampling lever (backs --jury_sources)."""

    @staticmethod
    def _jury_setup():
        all_users = [f"prism_{i}" for i in range(20)] + \
                    [f"hist_C016_{i:02d}" for i in range(10)] + \
                    [f"hist_C020_{i:02d}" for i in range(10)]
        metadata = {}
        for u in all_users:
            if u.startswith("prism_"):
                metadata[u] = {"period": "original"}
            elif "C016" in u:
                metadata[u] = {"period": "16C"}
            else:
                metadata[u] = {"period": "20C"}
        return all_users, metadata

    def test_explicit_counts_per_group(self):
        all_users, metadata = self._jury_setup()
        import random
        random.seed(42)
        sampled, strategy, cfg = per_group_sampling(
            all_users, metadata,
            jury_sources=[("16C", None), ("20C", None), ("original", 10)],
            m_voters_fallback=10,
        )
        assert strategy == "per_group"
        assert cfg["per_group_counts"] == {"16C": 10, "20C": 10, "original": 10}
        assert len(sampled) == 30
        # All-of-group selections must include every member.
        c16 = [u for u in sampled if metadata[u]["period"] == "16C"]
        c20 = [u for u in sampled if metadata[u]["period"] == "20C"]
        prism = [u for u in sampled if metadata[u]["period"] == "original"]
        assert len(c16) == 10 and len(c20) == 10 and len(prism) == 10

    def test_fallback_to_stratified_when_no_counts(self):
        all_users, metadata = self._jury_setup()
        import random
        random.seed(42)
        sampled, strategy, cfg = per_group_sampling(
            all_users, metadata,
            jury_sources=[("16C", None), ("20C", None)],
            m_voters_fallback=6,
        )
        assert strategy == "stratified"
        assert cfg == {"stratify_by": "period"}
        assert len(sampled) == 6

    def test_missing_group_raises(self):
        all_users, metadata = self._jury_setup()
        with pytest.raises(ValueError, match="No voters in jury"):
            per_group_sampling(
                all_users, metadata,
                jury_sources=[("99C", None)],
                m_voters_fallback=5,
            )

    def test_count_exceeds_available_raises(self):
        all_users, metadata = self._jury_setup()
        with pytest.raises(ValueError, match="only 10 are available"):
            per_group_sampling(
                all_users, metadata,
                jury_sources=[("16C", 11)],
                m_voters_fallback=10,
            )

    def test_normalize_source_label(self):
        assert _normalize_source_label("prism") == "original"
        assert _normalize_source_label("original") == "original"
        assert _normalize_source_label("C21") == "21C"
        assert _normalize_source_label("c21") == "21C"
        assert _normalize_source_label("C017") == "17C"
        assert _normalize_source_label("21C") == "21C"
        assert _normalize_source_label("13c") == "13C"
        assert _normalize_source_label("weird") == "weird"

    def test_parse_jury_source_spec(self):
        assert parse_jury_source_spec("C16") == ("16C", None)
        assert parse_jury_source_spec("prism") == ("original", None)
        assert parse_jury_source_spec("prism:10") == ("original", 10)
        assert parse_jury_source_spec("C16=3") == ("16C", 3)
        assert parse_jury_source_spec("C16:all") == ("16C", None)
        assert parse_jury_source_spec("prism:*") == ("original", None)
        with pytest.raises(ValueError, match="Invalid"):
            parse_jury_source_spec("prism:abc")
        with pytest.raises(ValueError, match=">= 0"):
            parse_jury_source_spec("prism:-1")
