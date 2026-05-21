"""
Tests for the democratic inference orchestrator.

Avoids loading any real models — jury and embedding step are stubbed out.
"""

from __future__ import annotations

import json

import pytest
import torch

from apa.democratic_response import (
    DemocraticInference,
    InferenceResult,
    QueryCase,
    build_default_jury,
    load_query_cases,
)
from apa.lore_adapt import LoReScorer


# =============================================================================
# Helpers
# =============================================================================

def _fake_scorer(n_users: int = 5, D: int = 32, K: int = 4) -> LoReScorer:
    V = torch.randn(D, K)
    scorer = LoReScorer(V)
    for i in range(n_users):
        scorer.user_registry[f"fake_user_{i}"] = torch.randn(K)
    return scorer


def _fake_meta(scorer: LoReScorer, origin: str = "prism") -> dict[str, dict]:
    period = "original" if origin == "prism" else "13C"
    return {
        uid: {"period": period, "ID": uid, "origin": origin, "checkpoint": "fake"}
        for uid in scorer.get_user_ids()
    }


def _patch_embedding(monkeypatch, D: int = 32):
    """Replace _embed_responses with a deterministic stub."""
    from apa import democratic_response as dr

    def fake_embed(query, responses, scorer):
        return torch.stack(
            [torch.tensor([float(i + 1)] * D) for i in range(len(responses))]
        )

    monkeypatch.setattr(dr, "_embed_responses", fake_embed)


# =============================================================================
# Response file loaders
# =============================================================================

class TestLoadQueryCases:
    def test_rich_jsonl_multiple_queries(self, tmp_path):
        """Format used in experiments/query_responses.jsonl."""
        path = tmp_path / "r.jsonl"
        path.write_text(
            json.dumps({
                "query_id": 1,
                "query": "Q1?",
                "responses": [
                    {"response_id": 1, "text": "A1"},
                    {"response_id": 2, "text": "A2"},
                ],
            }) + "\n"
            + json.dumps({
                "query_id": 2,
                "query": "Q2?",
                "responses": [
                    {"response_id": 1, "text": "B1"},
                    {"response_id": 2, "text": "B2"},
                    {"response_id": 3, "text": "B3"},
                ],
            }) + "\n"
        )
        cases = load_query_cases(path)
        assert len(cases) == 2
        assert cases[0].query_id == 1
        assert cases[0].query == "Q1?"
        assert cases[0].responses == ["A1", "A2"]
        assert cases[1].query_id == 2
        assert cases[1].responses == ["B1", "B2", "B3"]

    def test_flat_jsonl(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text(
            json.dumps({"query": "Q?", "response": "A1"}) + "\n"
            + json.dumps({"response": "A2"}) + "\n"
            + json.dumps({"response": "A3"}) + "\n"
        )
        cases = load_query_cases(path)
        assert len(cases) == 1
        assert cases[0].query == "Q?"
        assert cases[0].responses == ["A1", "A2", "A3"]

    def test_json_list_of_strings(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps(["A1", "A2"]))
        cases = load_query_cases(path)
        assert len(cases) == 1
        assert cases[0].responses == ["A1", "A2"]
        assert cases[0].query is None

    def test_json_wrapping_dict(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "query": "Q?",
            "responses": [{"response": "A1"}, "A2"],
        }))
        cases = load_query_cases(path)
        assert len(cases) == 1
        assert cases[0].query == "Q?"
        assert cases[0].responses == ["A1", "A2"]

    def test_json_list_of_wrapping_dicts(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps([
            {"query_id": 1, "query": "Q1?", "responses": [{"text": "A1"}]},
            {"query_id": 2, "query": "Q2?", "responses": [{"text": "B1"}]},
        ]))
        cases = load_query_cases(path)
        assert [c.query_id for c in cases] == [1, 2]

    def test_txt(self, tmp_path):
        path = tmp_path / "r.txt"
        path.write_text("A1\n\nA2\nA3\n")
        cases = load_query_cases(path)
        assert len(cases) == 1
        assert cases[0].responses == ["A1", "A2", "A3"]
        assert cases[0].query is None

    def test_unsupported(self, tmp_path):
        path = tmp_path / "r.xml"
        path.write_text("<x/>")
        with pytest.raises(ValueError):
            load_query_cases(path)

    def test_real_experiments_file(self):
        """Sanity-check the experiments/query_responses.jsonl fixture parses."""
        from pathlib import Path
        path = Path(__file__).parent.parent / "experiments" / "query_responses.jsonl"
        if not path.exists():
            pytest.skip(f"{path} not present in this checkout")
        cases = load_query_cases(path)
        assert len(cases) >= 1
        for c in cases:
            assert isinstance(c, QueryCase)
            assert c.query
            assert c.responses
            assert all(isinstance(r, str) and r for r in c.responses)


# =============================================================================
# Orchestrator
# =============================================================================

class TestDemocraticInference:
    def test_validates_methods(self):
        scorer = _fake_scorer()
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            DemocraticInference(
                scorer=scorer,
                user_metadata=_fake_meta(scorer),
                jury_manifest={},
                methods=["not_a_method"],
            )

    def test_validates_sampling(self):
        scorer = _fake_scorer()
        with pytest.raises(ValueError, match="Unknown sampling method"):
            DemocraticInference(
                scorer=scorer,
                user_metadata=_fake_meta(scorer),
                jury_manifest={},
                sampling="bogus",
            )

    def test_requires_query_or_responses(self):
        scorer = _fake_scorer()
        inf = DemocraticInference(
            scorer=scorer,
            user_metadata=_fake_meta(scorer),
            jury_manifest={},
        )
        with pytest.raises(ValueError, match="query or responses"):
            inf()

    def test_empty_jury_raises(self, monkeypatch):
        _patch_embedding(monkeypatch)
        V = torch.randn(32, 4)
        scorer = LoReScorer(V)
        inf = DemocraticInference(scorer=scorer, user_metadata={}, jury_manifest={})
        with pytest.raises(RuntimeError, match="Jury is empty"):
            inf(responses=["A", "B"], responses_source="test")

    def test_runs_with_supplied_responses(self, monkeypatch):
        _patch_embedding(monkeypatch)
        scorer = _fake_scorer(n_users=6)
        inf = DemocraticInference(
            scorer=scorer,
            user_metadata=_fake_meta(scorer),
            jury_manifest={"V": "fake", "sources": []},
            m_voters=4,
            methods=["borda_count", "copeland"],
            sampling="random",  # bypass stratification for this fake fixture
            seed=42,
        )
        result = inf(responses=["A", "B", "C"], responses_source="unit-test")

        assert isinstance(result, InferenceResult)
        assert result.query is None
        assert result.query_id is None
        assert result.responses == ["A", "B", "C"]
        assert result.responses_source == "unit-test"
        assert len(result.sampled_user_ids) == 4
        # Per-voter structures cover sampled voters.
        assert set(result.per_voter_scores) == set(result.sampled_user_ids)
        assert set(result.per_voter_rankings) == set(result.sampled_user_ids)
        # Each ranking is a permutation of response indices.
        for ranking in result.per_voter_rankings.values():
            assert sorted(ranking) == [0, 1, 2]
        # Both methods produced aggregations.
        assert set(result.aggregations) == {"borda_count", "copeland"}
        for method in ("borda_count", "copeland"):
            agg = result.aggregations[method]
            assert sorted(agg["ranking"]) == [0, 1, 2]
            assert agg["winner_idx"] == agg["ranking"][0]
        # Audit log has the reproducibility-relevant fields.
        assert result.response_embeddings_hash
        assert result.config["seed"] == 42
        assert result.config["methods"] == ["borda_count", "copeland"]
        # inference_llm is None when we didn't call the LLM.
        assert result.config["inference_llm"] is None
        # Average rank: length matches responses, values in [1, n_responses].
        assert len(result.average_ranks) == 3
        assert all(1.0 <= r <= 3.0 for r in result.average_ranks)
        # And the sum of (average_rank * n_voters) over all responses equals
        # the total rank mass: 1+2+3 per voter, times n_voters.
        total = sum(result.average_ranks) * len(result.sampled_user_ids)
        assert abs(total - len(result.sampled_user_ids) * (1 + 2 + 3)) < 1e-6
        # Per-voter metadata uses the new period/ID schema.
        first_meta = next(iter(result.sampled_user_metadata.values()))
        assert "period" in first_meta and "ID" in first_meta

    def test_vote_is_reconstructable_from_audit_log(self, monkeypatch, tmp_path):
        """The persisted audit log should let us re-run the aggregation and match."""
        _patch_embedding(monkeypatch)
        scorer = _fake_scorer(n_users=7)
        inf = DemocraticInference(
            scorer=scorer,
            user_metadata=_fake_meta(scorer),
            jury_manifest={"V": "fake", "sources": []},
            m_voters=5,
            methods=["borda_count", "plurality"],
            sampling="random",
            seed=7,
        )
        result = inf(query="Q?", responses=["A", "B", "C", "D"], responses_source="test", query_id=42)
        assert result.query_id == 42

        log_path = tmp_path / "vote.json"
        result.save(log_path)
        with open(log_path) as f:
            reloaded = json.load(f)

        from apa.levers.voter_aggregation import borda_count, plurality
        rankings = {uid: r for uid, r in reloaded["per_voter_rankings"].items()}
        replay_borda = borda_count(rankings, {})
        replay_plurality = plurality(rankings, {})
        assert replay_borda[0] == reloaded["aggregations"]["borda_count"]["winner_idx"]
        assert replay_plurality[0] == reloaded["aggregations"]["plurality"]["winner_idx"]

    def test_default_method_from_config(self, monkeypatch):
        _patch_embedding(monkeypatch)
        scorer = _fake_scorer()
        inf = DemocraticInference(
            scorer=scorer,
            user_metadata=_fake_meta(scorer),
            jury_manifest={},
            m_voters=3,
            sampling="random",
        )
        result = inf(responses=["A", "B", "C"])
        # Default from InferenceConfig is borda_count.
        assert list(result.aggregations) == ["borda_count"]


# =============================================================================
# build_default_jury
# =============================================================================

class TestBuildDefaultJury:
    def test_missing_v_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_default_jury(K=99, V_checkpoint=tmp_path / "nope.pt")

    def test_loads_from_checkpoints(self, tmp_path, monkeypatch):
        # Synthesise a V checkpoint in the format LoReRewardModel.load expects.
        D, K = 16, 4
        V = torch.randn(D, K)

        # Patch from_checkpoint to avoid depending on the real LoReRewardModel format.
        class FakeModel:
            def __init__(self, V):
                class _V: pass
                self.V = _V()
                self.V.data = V

        def fake_load(cls, path):
            return LoReScorer(V.clone())

        monkeypatch.setattr(LoReScorer, "from_checkpoint", classmethod(fake_load))

        # Make the V file exist so the existence check passes.
        V_path = tmp_path / f"V_K{K}.pt"
        V_path.write_bytes(b"x")

        # PRISM users checkpoint in the tensor-of-W format.
        prism_W = torch.randn(3, K)
        prism_path = tmp_path / f"W_seen_K{K}.pt"
        torch.save(prism_W, prism_path)

        # Adapted users checkpoint in lore_adapt save format.
        from apa.lore_adapt import save_adapted
        adapted_results = {f"adapted_{i}": {"w": torch.randn(K)} for i in range(2)}
        adapted_path = tmp_path / f"W_adapted_fake_K{K}_K{K}.pt"
        save_adapted(adapted_results, adapted_path, metadata={"test": True})

        scorer, meta, summary = build_default_jury(
            K=K,
            V_checkpoint=V_path,
            prism_users=prism_path,
            adapted_users=adapted_path,
        )

        assert len(scorer.get_user_ids()) == 5  # 3 PRISM + 2 adapted
        # Metadata tags each user with origin, period, and ID.
        origins = {v["origin"] for v in meta.values()}
        assert origins == {"prism", "adapted"}
        for uid, m in meta.items():
            assert m["ID"] == uid
            assert "period" in m
        # PRISM users have period='original'; adapted users have derived period.
        prism_periods = {m["period"] for m in meta.values() if m["origin"] == "prism"}
        assert prism_periods == {"original"}
        # Summary records origin (no longer 'name').
        summary_origins = [s["origin"] for s in summary["sources"]]
        assert "prism" in summary_origins
        assert "adapted" in summary_origins

    def test_adapted_user_period_from_id(self):
        """hist_CNNN_ID → 'NC' period."""
        from apa.democratic_response import _period_for_user
        assert _period_for_user("hist_C013_01", "adapted") == "13C"
        assert _period_for_user("hist_C017_02", "adapted") == "17C"
        assert _period_for_user("hist_C021_09", "adapted") == "21C"
        assert _period_for_user("prism_user_42", "prism") == "original"
        assert _period_for_user("weird_id", "adapted") == "other"


class TestStratifiedJury:
    """Sanity-check the default half-PRISM/half-adapted sampling."""

    def test_stratified_by_origin_gives_balanced_jury(self, monkeypatch):
        _patch_embedding(monkeypatch)
        # 10 PRISM + 10 adapted fake voters.
        V = torch.randn(32, 4)
        scorer = LoReScorer(V)
        for i in range(10):
            scorer.user_registry[f"prism_user_{i}"] = torch.randn(4)
        for i in range(10):
            scorer.user_registry[f"hist_C013_{i:02d}"] = torch.randn(4)
        meta = {}
        for uid in scorer.get_user_ids():
            origin = "prism" if uid.startswith("prism") else "adapted"
            meta[uid] = {
                "period": "original" if origin == "prism" else "13C",
                "ID": uid,
                "origin": origin,
            }

        # Default sampling is stratified on 'origin' with m_voters=10 →
        # exactly 5 prism + 5 adapted.
        inf = DemocraticInference(
            scorer=scorer,
            user_metadata=meta,
            jury_manifest={},
            m_voters=10,
            seed=0,
        )
        result = inf(responses=["A", "B"], responses_source="test")
        sampled_origins = [meta[uid]["origin"] for uid in result.sampled_user_ids]
        assert sampled_origins.count("prism") == 5
        assert sampled_origins.count("adapted") == 5


class TestJurySourcesSelection:
    """User-facing --jury_sources filter: draws evenly from named groups."""

    def _pool(self):
        V = torch.randn(16, 4)
        scorer = LoReScorer(V)
        meta = {}
        # 8 PRISM, 4 C21, 3 C17, 6 C13 fake voters.
        def _add(uid, period, origin):
            scorer.user_registry[uid] = torch.randn(4)
            meta[uid] = {"period": period, "ID": uid, "origin": origin}
        for i in range(8):
            _add(f"prism_{i}", "original", "prism")
        for i in range(4):
            _add(f"hist_C021_{i:02d}", "21C", "adapted")
        for i in range(3):
            _add(f"hist_C017_{i:02d}", "17C", "adapted")
        for i in range(6):
            _add(f"hist_C013_{i:02d}", "13C", "adapted")
        return scorer, meta

    # Note: _normalize_source_label / parse_jury_source_spec are unit-tested
    # in tests/test_levers.py alongside the rest of the lever surface.

    def test_filters_and_balances(self, monkeypatch):
        _patch_embedding(monkeypatch, D=16)
        scorer, meta = self._pool()
        inf = DemocraticInference(
            scorer=scorer,
            user_metadata=meta,
            jury_manifest={},
            m_voters=6,
            jury_sources=["prism", "C21", "C17"],
            seed=0,
        )
        result = inf(responses=["A", "B"], responses_source="test")
        periods = [meta[uid]["period"] for uid in result.sampled_user_ids]
        # Exactly 6 voters, 2 from each of the three named groups.
        assert len(result.sampled_user_ids) == 6
        assert periods.count("original") == 2
        assert periods.count("21C") == 2
        assert periods.count("17C") == 2
        assert "13C" not in periods
        # Audit log records the normalised jury_sources as list[{label,count}].
        assert result.config["jury_sources"] == [
            {"label": "original", "count": None},
            {"label": "21C", "count": None},
            {"label": "17C", "count": None},
        ]

    def test_empty_group_raises(self, monkeypatch):
        _patch_embedding(monkeypatch, D=16)
        scorer, meta = self._pool()
        inf = DemocraticInference(
            scorer=scorer,
            user_metadata=meta,
            jury_manifest={},
            m_voters=4,
            jury_sources=["prism", "C15"],  # no C15 voters available
            seed=0,
        )
        with pytest.raises(ValueError, match="No voters in jury"):
            inf(responses=["A", "B"], responses_source="test")

    def test_default_unchanged_when_none(self, monkeypatch):
        """With jury_sources=None, audit log's jury_sources is null."""
        _patch_embedding(monkeypatch)
        scorer = _fake_scorer()
        inf = DemocraticInference(
            scorer=scorer,
            user_metadata=_fake_meta(scorer),
            jury_manifest={},
            m_voters=3,
            sampling="random",
        )
        result = inf(responses=["A", "B", "C"])
        assert result.config["jury_sources"] is None
