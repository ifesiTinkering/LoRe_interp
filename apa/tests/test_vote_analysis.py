"""Tests for apa.vote_analysis and the per-group jury_sources extension."""

from __future__ import annotations

import json
import math

import pytest
import torch

from apa.democratic_response import DemocraticInference
from apa.levers.voter_sampling import parse_jury_source_spec
from apa.lore_adapt import LoReScorer
from apa.vote_analysis import (
    analyze_audit_log,
    analyze_case,
    kendall_tau,
    render_report,
    spearman,
)


# =============================================================================
# Rank-agreement metric tests
# =============================================================================


def test_spearman_identical():
    assert spearman([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)


def test_spearman_reversed():
    assert spearman([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)


def test_kendall_identical_and_reversed():
    assert kendall_tau([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert kendall_tau([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)


def test_kendall_partial():
    # Three concordant pairs, three discordant → 0.
    val = kendall_tau([0, 1, 2, 3], [1, 0, 3, 2])
    assert val == pytest.approx(1 / 3)  # 4 concordant, 2 discordant out of 6.


# =============================================================================
# parse_jury_source_spec
# =============================================================================


def test_parse_bare_label():
    assert parse_jury_source_spec("C16") == ("16C", None)
    assert parse_jury_source_spec("prism") == ("original", None)


def test_parse_with_count():
    assert parse_jury_source_spec("prism:10") == ("original", 10)
    assert parse_jury_source_spec("C16=3") == ("16C", 3)


def test_parse_all_keyword():
    assert parse_jury_source_spec("C16:all") == ("16C", None)
    assert parse_jury_source_spec("prism:*") == ("original", None)


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_jury_source_spec("prism:abc")
    with pytest.raises(ValueError):
        parse_jury_source_spec("prism:-1")


# =============================================================================
# Per-group jury sampling in DemocraticInference
# =============================================================================


def _patch_embedding(monkeypatch, D: int = 16) -> None:
    """Stub LoReScorer.embed_texts so __call__ doesn't need a real model."""
    from apa import democratic_response as dr

    def _fake_embed(query, responses, scorer):
        n = len(responses)
        torch.manual_seed(0)
        return torch.randn(n, D)

    monkeypatch.setattr(dr, "_embed_responses", _fake_embed)


def _build_pool(D: int = 16):
    V = torch.randn(D, 4)
    scorer = LoReScorer(V)
    meta: dict[str, dict] = {}

    def _add(uid, period, origin):
        scorer.user_registry[uid] = torch.randn(4)
        meta[uid] = {"period": period, "ID": uid, "origin": origin}

    for i in range(8):
        _add(f"prism_{i}", "original", "prism")
    for i in range(10):
        _add(f"hist_C016_{i:02d}", "16C", "adapted")
    for i in range(10):
        _add(f"hist_C020_{i:02d}", "20C", "adapted")
    return scorer, meta


def test_per_group_counts_take_all_and_sample(monkeypatch):
    _patch_embedding(monkeypatch)
    scorer, meta = _build_pool()
    inf = DemocraticInference(
        scorer=scorer,
        user_metadata=meta,
        jury_manifest={},
        jury_sources=["C16", "C20", "prism:5"],
        seed=0,
    )
    result = inf(responses=["A", "B", "C", "D"], responses_source="test")
    periods = [meta[uid]["period"] for uid in result.sampled_user_ids]

    assert periods.count("16C") == 10  # all C16
    assert periods.count("20C") == 10  # all C20
    assert periods.count("original") == 5  # 5 sampled PRISM
    assert len(result.sampled_user_ids) == 25

    # Audit log always records list[{label,count}] (count=None means "all").
    js = result.config["jury_sources"]
    assert isinstance(js, list) and all(isinstance(e, dict) for e in js)
    counts = {entry["label"]: entry["count"] for entry in js}
    assert counts == {"16C": None, "20C": None, "original": 5}


def test_per_group_overflow_raises(monkeypatch):
    _patch_embedding(monkeypatch)
    scorer, meta = _build_pool()
    inf = DemocraticInference(
        scorer=scorer,
        user_metadata=meta,
        jury_manifest={},
        jury_sources=["prism:99"],  # only 8 PRISM voters available
        seed=0,
    )
    with pytest.raises(ValueError, match="only 8 are available"):
        inf(responses=["A", "B"], responses_source="test")


def test_bare_labels_preserve_legacy_stratified(monkeypatch):
    """No explicit counts → existing stratified-by-period split (regression)."""
    _patch_embedding(monkeypatch)
    scorer, meta = _build_pool()
    inf = DemocraticInference(
        scorer=scorer,
        user_metadata=meta,
        jury_manifest={},
        m_voters=6,
        jury_sources=["prism", "C16", "C20"],
        seed=0,
    )
    result = inf(responses=["A", "B"], responses_source="test")
    periods = [meta[uid]["period"] for uid in result.sampled_user_ids]
    assert len(result.sampled_user_ids) == 6
    assert periods.count("original") == 2
    assert periods.count("16C") == 2
    assert periods.count("20C") == 2
    # Even with no explicit counts, log uses uniform list[{label,count=None}].
    assert result.config["jury_sources"] == [
        {"label": "original", "count": None},
        {"label": "16C", "count": None},
        {"label": "20C", "count": None},
    ]


# =============================================================================
# analyze_case + render_report
# =============================================================================


def _synthetic_case() -> dict:
    """Two groups whose voters disagree across groups but agree within."""
    return {
        "query_id": 1,
        "query": "synthetic",
        "responses": ["A", "B", "C", "D"],
        "sampled_user_ids": ["a1", "a2", "b1", "b2"],
        "sampled_user_metadata": {
            "a1": {"period": "16C"},
            "a2": {"period": "16C"},
            "b1": {"period": "20C"},
            "b2": {"period": "20C"},
        },
        "per_voter_rankings": {
            "a1": [0, 1, 2, 3],
            "a2": [0, 1, 2, 3],
            "b1": [3, 2, 1, 0],
            "b2": [3, 2, 1, 0],
        },
        "per_voter_scores": {},
    }


def test_analyze_case_groups_and_aggregations():
    case = _synthetic_case()
    analysis = analyze_case(case)

    assert analysis["groups"] == {
        "16C": ["a1", "a2"],
        "20C": ["b1", "b2"],
    }

    # Per-group winners: 16C prefers #1, 20C prefers #4. Full jury ties on
    # Borda, but the ranking is a valid permutation.
    pg = analysis["per_group_aggregations"]
    assert pg["16C"]["borda_count"]["winner_idx"] == 0
    assert pg["20C"]["borda_count"]["winner_idx"] == 3
    assert sorted(pg["__all__"]["borda_count"]["ranking"]) == [0, 1, 2, 3]


def test_analyze_case_agreement_metrics():
    case = _synthetic_case()
    analysis = analyze_case(case)
    agree = analysis["agreement"]

    def find_intra(scope):
        return next(e for e in agree if e["kind"] == "intra" and e["scope"] == scope)

    def find_inter(g1, g2):
        return next(
            e for e in agree
            if e["kind"] == "inter" and set(e["groups"]) == {g1, g2}
        )

    # Identical rankings within each group ⇒ ρ=τ=+1.
    assert find_intra("16C")["mean_spearman"] == pytest.approx(1.0)
    assert find_intra("20C")["mean_kendall_tau"] == pytest.approx(1.0)

    # Reversed rankings between groups ⇒ ρ=τ=-1.
    inter = find_inter("16C", "20C")
    assert inter["mean_spearman"] == pytest.approx(-1.0)
    assert inter["mean_kendall_tau"] == pytest.approx(-1.0)
    assert inter["n_pairs"] == 4

    # Full jury (all voters together): 4 voters ⇒ C(4,2)=6 pairs. With two
    # +1-correlated pairs intra-group and four −1-correlated cross-group
    # pairs, mean ρ = (2*1 + 4*(-1)) / 6 = -1/3.
    full = find_intra("__all__")
    assert full["n_pairs"] == 6
    assert full["mean_spearman"] == pytest.approx(-1.0 / 3.0)
    assert full["mean_kendall_tau"] == pytest.approx(-1.0 / 3.0)


def test_render_report_smoke():
    case = _synthetic_case()
    analysis = analyze_audit_log([case])
    report = render_report(analysis)
    assert "Vote analysis — 1 case(s)" in report
    assert "intra[16C]" in report
    assert "inter[16C ↔ 20C]" in report
    # No NaNs leaking into output for fully-defined inputs.
    assert "nan" not in report.lower()


def test_audit_log_round_trip(tmp_path):
    """Save synthetic audit log → load → analyze."""
    case = _synthetic_case()
    log_path = tmp_path / "audit.json"
    log_path.write_text(json.dumps([case]))
    log = json.loads(log_path.read_text())
    analysis = analyze_audit_log(log)
    assert analysis["n_cases"] == 1
    assert analysis["cases"][0]["query_id"] == 1
