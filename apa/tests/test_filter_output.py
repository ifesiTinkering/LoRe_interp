"""Unit tests for the three-stage preference filter."""

from __future__ import annotations

from experiments.filter_output import filter_prefs


def _raw(user_id, qid, pref, consistency=1.0, *, prompt="Q?", r1="A", r2="B"):
    """Build a minimal raw-record dict matching the historical_prefs schema."""
    return {
        "user_id": user_id,
        "question_id": qid,
        "prompt": prompt,
        "response_1": r1,
        "response_2": r2,
        "final_preference": pref,
        "consistency": consistency,
        "prob_1_original": 0.9 if pref == "1" else 0.1,
        "prob_2_original": 0.1 if pref == "1" else 0.9,
        "prob_1_reversed": 0.1 if pref == "1" else 0.9,
        "prob_2_reversed": 0.9 if pref == "1" else 0.1,
        "soft_preference_1": 0.9 if pref == "1" else 0.1,
    }


class TestFilterA:
    """Filter A: drop records with consistency != 1.0."""

    def test_drops_inconsistent(self):
        raw = [
            _raw("u0", 1, "1", consistency=1.0),
            _raw("u0", 2, "-1", consistency=0.0),
            _raw("u0", 3, "1", consistency=1.0),
            _raw("u1", 1, "2", consistency=0.0),
            _raw("u1", 2, "1", consistency=1.0),
        ]
        # Add disagreement on q1 so something survives B.
        raw.append(_raw("u2", 1, "2", consistency=1.0))
        records, stats = filter_prefs(raw, min_records_per_user=1)
        assert stats["input"] == 6
        assert stats["after_consistency"] == 4  # the two consistency=0.0 dropped


class TestFilterB:
    """Filter B: drop questions where every surviving record agreed."""

    def test_drops_unanimous_question(self):
        # q1: 3 users all pick "1" -> unanimous -> dropped.
        # q2: 2 users pick "1", 1 picks "2" -> kept.
        raw = [
            _raw("u0", 1, "1"), _raw("u1", 1, "1"), _raw("u2", 1, "1"),
            _raw("u0", 2, "1"), _raw("u1", 2, "1"), _raw("u2", 2, "2"),
        ]
        records, stats = filter_prefs(raw, min_records_per_user=1)
        assert stats["divisive_questions"] == 1
        # All 3 records that survive belong to q2.
        assert all(r["prompt"] == "Q?" for r in records)
        assert stats["after_divisive"] == 3

    def test_keeps_only_divisive(self):
        raw = [
            _raw("u0", 1, "1"), _raw("u1", 1, "2"),  # q1 divisive
            _raw("u0", 2, "1"), _raw("u1", 2, "1"),  # q2 unanimous
        ]
        _, stats = filter_prefs(raw, min_records_per_user=1)
        assert stats["divisive_questions"] == 1
        assert stats["after_divisive"] == 2


class TestFilterC:
    """Filter C: drop users with < min_records_per_user records."""

    def test_drops_low_coverage_user_at_threshold(self):
        # u0 has 4 records, u1 has 5; threshold=5 → keep only u1.
        # All on questions where there's at least one disagreement.
        raw = []
        for q in [1, 2, 3, 4]:
            raw.append(_raw("u0", q, "1"))
            raw.append(_raw("u1", q, "2"))
        # extra record for u1 on q5; u2 disagrees on q5 to keep it divisive.
        raw.append(_raw("u1", 5, "2"))
        raw.append(_raw("u2", 5, "1"))

        records, stats = filter_prefs(raw, min_records_per_user=5)
        kept_users = {r["user_id"] for r in records}
        assert "u0" not in kept_users  # only 4 records
        assert "u1" in kept_users  # 5 records
        assert stats["users_kept"] == len(kept_users)

    def test_keeps_user_at_exact_threshold(self):
        # u0 with exactly 5 records, all on divisive Qs.
        raw = []
        for q in range(1, 6):
            raw.append(_raw("u0", q, "1"))
            raw.append(_raw("u1", q, "2"))  # makes each Q divisive
        records, stats = filter_prefs(raw, min_records_per_user=5)
        kept_users = {r["user_id"] for r in records}
        assert "u0" in kept_users and "u1" in kept_users


class TestChain:
    """End-to-end behaviour of the A→B→C chain."""

    def test_idempotent_when_already_clean(self):
        # 3 users, 5 questions, all consistent, all questions have at least one
        # disagreement, all users have >=5 records → nothing should drop.
        raw = []
        for q in range(1, 6):
            raw.append(_raw("u0", q, "1"))
            raw.append(_raw("u1", q, "2"))
            raw.append(_raw("u2", q, "1"))
        records, stats = filter_prefs(raw, min_records_per_user=5)
        assert stats["input"] == 15
        assert stats["after_consistency"] == 15
        assert stats["after_divisive"] == 15
        assert stats["after_user_coverage"] == 15
        assert stats["output"] == 15

    def test_output_is_eval_prefs_schema(self):
        raw = [
            _raw("u0", 1, "1"), _raw("u1", 1, "2"),
            _raw("u0", 2, "1"), _raw("u1", 2, "2"),
            _raw("u0", 3, "1"), _raw("u1", 3, "2"),
            _raw("u0", 4, "1"), _raw("u1", 4, "2"),
            _raw("u0", 5, "1"), _raw("u1", 5, "2"),
        ]
        records, _ = filter_prefs(raw, min_records_per_user=5)
        assert records, "expected non-empty output"
        for r in records:
            for key in ("user_id", "prompt", "chosen", "rejected"):
                assert key in r, f"missing required key {key!r}"
