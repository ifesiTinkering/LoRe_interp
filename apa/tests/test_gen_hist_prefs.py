"""
Unit tests for synthetic historical preference generation.

Tests pure functions only — no GPU or model loading required.
"""

import json
import tempfile
from pathlib import Path

import pytest

from apa.synthetic_prefs.historical_prefs import (
    VALID_CENTURIES,
    _build_comparison_messages,
    _render_stage1_prompt,
    _render_stage2_prompt,
    _resolve_choice_token_ids,
    load_curated_question_ids,
    load_profiles,
    results_to_jsonl_records,
    write_jsonl,
)
from apa.synthetic_prefs.eval_prefs import PreferencePair, load_prefs_jsonl
from apa.synthetic_prefs.sample_data import (
    sample_prefs_by_questions,
    random_prefs_by_questions,
)


# ---------------------------------------------------------------------------
# load_profiles
# ---------------------------------------------------------------------------

class TestLoadProfiles:
    """Test loading user profiles from JSONL."""

    def test_bundled_profiles(self):
        """Bundled profiles.jsonl loads correctly with expected structure."""
        profiles = load_profiles()
        assert "C013" in profiles
        assert "C019" in profiles
        for century in ["C013", "C014", "C015", "C016", "C017", "C018", "C019", "C020", "C021"]:
            assert century in profiles, f"Missing century {century}"
            assert len(profiles[century]) == 10, f"{century} has {len(profiles[century])} profiles, expected 10"

    def test_all_profiles_nonempty(self):
        """Every profile is a non-empty string."""
        profiles = load_profiles()
        for century, profs in profiles.items():
            assert century in VALID_CENTURIES
            for p in profs:
                assert isinstance(p, str)
                assert len(p) > 0

    def test_custom_profiles_file(self, tmp_path):
        """Can load profiles from a custom JSONL file."""
        path = tmp_path / "custom.jsonl"
        path.write_text(
            '{"century": "C017", "profile": "A 17th-century Dutch painter."}\n'
            '{"century": "C017", "profile": "A 17th-century Ottoman merchant."}\n'
        )
        profiles = load_profiles(path)
        assert profiles == {
            "C017": [
                "A 17th-century Dutch painter.",
                "A 17th-century Ottoman merchant.",
            ]
        }

    def test_empty_file(self, tmp_path):
        """Empty file returns empty dict."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        profiles = load_profiles(path)
        assert profiles == {}


# ---------------------------------------------------------------------------
# results_to_jsonl_records
# ---------------------------------------------------------------------------

def _make_result(user_id, prompt, r1, r2, pref, *,
                 prob_1_original=0.0, prob_2_original=0.0,
                 prob_1_reversed=0.0, prob_2_reversed=0.0,
                 soft_preference_1=0.5, consistency=1.0):
    return {
        "user_id": user_id,
        "prompt": prompt,
        "response_1": r1,
        "response_2": r2,
        "final_preference": pref,
        "prob_1_original": prob_1_original,
        "prob_2_original": prob_2_original,
        "prob_1_reversed": prob_1_reversed,
        "prob_2_reversed": prob_2_reversed,
        "soft_preference_1": soft_preference_1,
        "consistency": consistency,
    }


class TestResultsToJsonlRecords:
    """Test conversion from raw results to eval_prefs JSONL format."""

    def test_preference_1(self):
        """Preference '1' maps chosen=response_1, rejected=response_2."""
        results = [_make_result("u0", "Q?", "A", "B", "1")]
        records = results_to_jsonl_records(results)
        assert len(records) == 1
        rec = records[0]
        assert rec["user_id"] == "u0"
        assert rec["prompt"] == "Q?"
        assert rec["chosen"] == "A"
        assert rec["rejected"] == "B"

    def test_preference_2(self):
        """Preference '2' maps chosen=response_2, rejected=response_1."""
        results = [_make_result("u0", "Q?", "A", "B", "2")]
        records = results_to_jsonl_records(results)
        assert len(records) == 1
        assert records[0]["chosen"] == "B"
        assert records[0]["rejected"] == "A"

    def test_ambiguous_skipped(self):
        """Preference '-1' (ambiguous) is skipped."""
        results = [_make_result("u0", "Q?", "A", "B", "-1")]
        records = results_to_jsonl_records(results)
        assert len(records) == 0

    def test_mixed(self):
        """Only valid preferences make it through."""
        results = [
            _make_result("u0", "Q1", "A", "B", "1"),
            _make_result("u0", "Q2", "C", "D", "-1"),
            _make_result("u1", "Q3", "E", "F", "2"),
        ]
        records = results_to_jsonl_records(results)
        assert len(records) == 2
        assert records[0]["user_id"] == "u0"
        assert records[1]["user_id"] == "u1"

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert results_to_jsonl_records([]) == []

    def test_logprobs_carried_for_pref_1(self):
        """When preference is '1', prob_chosen comes from prob_1, prob_rejected from prob_2."""
        results = [_make_result(
            "u0", "Q?", "A", "B", "1",
            prob_1_original=0.85, prob_2_original=0.15,
            prob_1_reversed=0.10, prob_2_reversed=0.90,
            soft_preference_1=0.875, consistency=1.0,
        )]
        rec = results_to_jsonl_records(results)[0]
        assert rec["prob_chosen_original"] == 0.85
        assert rec["prob_rejected_original"] == 0.15
        # In reversed prompt, "1" tokenized prob is for physical response 2 (rejected).
        assert rec["prob_chosen_reversed"] == 0.90
        assert rec["prob_rejected_reversed"] == 0.10
        assert rec["soft_preference_chosen"] == 0.875
        assert rec["consistency"] == 1.0

    def test_logprobs_swapped_for_pref_2(self):
        """When preference is '2', chosen=response_2 so prob_chosen comes from prob_2."""
        results = [_make_result(
            "u0", "Q?", "A", "B", "2",
            prob_1_original=0.20, prob_2_original=0.80,
            prob_1_reversed=0.75, prob_2_reversed=0.25,
            soft_preference_1=0.225, consistency=1.0,
        )]
        rec = results_to_jsonl_records(results)[0]
        assert rec["prob_chosen_original"] == 0.80
        assert rec["prob_rejected_original"] == 0.20
        assert rec["prob_chosen_reversed"] == 0.75
        assert rec["prob_rejected_reversed"] == 0.25
        # soft_preference_chosen for '2' is 1 - soft_preference_1.
        assert abs(rec["soft_preference_chosen"] - (1 - 0.225)) < 1e-9


# ---------------------------------------------------------------------------
# write_jsonl + round-trip with load_prefs_jsonl
# ---------------------------------------------------------------------------

class TestWriteJsonl:
    """Test JSONL writing and round-trip compatibility with eval_prefs."""

    def test_round_trip(self, tmp_path):
        """Written JSONL can be loaded by eval_prefs.load_prefs_jsonl."""
        records = [
            {"user_id": "u0", "prompt": "Q1?", "chosen": "A", "rejected": "B"},
            {"user_id": "u0", "prompt": "Q2?", "chosen": "C", "rejected": "D"},
            {"user_id": "u1", "prompt": "Q1?", "chosen": "E", "rejected": "F"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(records, path)

        loaded = load_prefs_jsonl(path)
        assert set(loaded.keys()) == {"u0", "u1"}
        assert len(loaded["u0"]) == 2
        assert len(loaded["u1"]) == 1
        assert loaded["u0"][0].prompt == "Q1?"
        assert loaded["u0"][0].chosen == "A"
        assert loaded["u0"][0].rejected == "B"

    def test_creates_parent_dirs(self, tmp_path):
        """write_jsonl creates parent directories if needed."""
        path = tmp_path / "sub" / "dir" / "test.jsonl"
        write_jsonl([{"user_id": "u", "prompt": "Q", "chosen": "A", "rejected": "B"}], path)
        assert path.exists()

    def test_empty_records(self, tmp_path):
        """Empty records list produces empty file."""
        path = tmp_path / "empty.jsonl"
        write_jsonl([], path)
        assert path.read_text() == ""


# ---------------------------------------------------------------------------
# load_curated_question_ids
# ---------------------------------------------------------------------------

class TestLoadCuratedQuestionIds:

    def test_bundled_file(self):
        ids = load_curated_question_ids()
        assert len(ids) == 20
        assert all(isinstance(i, int) for i in ids)

    def test_custom_file(self, tmp_path):
        path = tmp_path / "qs.txt"
        path.write_text("# comment\n100\n200\n\n300\n")
        ids = load_curated_question_ids(path)
        assert ids == [100, 200, 300]


# ---------------------------------------------------------------------------
# sample_prefs_by_questions / random_prefs_by_questions
# ---------------------------------------------------------------------------

def _make_prefs():
    return {
        "u0": [
            PreferencePair("Q1", "A", "B"),
            PreferencePair("Q2", "C", "D"),
            PreferencePair("Q3", "E", "F"),
        ],
        "u1": [
            PreferencePair("Q1", "G", "H"),
            PreferencePair("Q4", "I", "J"),
        ],
        "u2": [
            PreferencePair("Q2", "K", "L"),
        ],
    }


class TestSamplePrefsByQuestions:

    def test_filters_to_matching_prompts(self):
        prefs = _make_prefs()
        result = sample_prefs_by_questions(prefs, {"Q1", "Q2"})
        # u0 has Q1 and Q2, u1 has Q1, u2 has Q2
        assert set(result.keys()) == {"u0", "u1", "u2"}
        assert len(result["u0"]) == 2
        assert len(result["u1"]) == 1
        assert len(result["u2"]) == 1

    def test_drops_users_with_no_matches(self):
        prefs = _make_prefs()
        result = sample_prefs_by_questions(prefs, {"Q4"})
        # Only u1 has Q4
        assert set(result.keys()) == {"u1"}

    def test_subsample_users(self):
        prefs = _make_prefs()
        result = sample_prefs_by_questions(prefs, {"Q1", "Q2"}, n_users=2, seed=42)
        assert len(result) == 2

    def test_empty_prompt_set(self):
        prefs = _make_prefs()
        result = sample_prefs_by_questions(prefs, set())
        assert result == {}


class TestRandomPrefsByQuestions:

    def test_preserves_prompts(self):
        prefs = _make_prefs()
        result = random_prefs_by_questions(prefs, {"Q1"})
        for pairs in result.values():
            for p in pairs:
                assert p.prompt == "Q1"

    def test_flips_some_labels(self):
        """With enough data, at least one pair should be flipped."""
        # Use many pairs to make flipping statistically certain
        prefs = {"u0": [PreferencePair("Q1", "A", "B") for _ in range(50)]}
        result = random_prefs_by_questions(prefs, {"Q1"}, seed=42)
        chosen_values = {p.chosen for p in result["u0"]}
        # Should have both A and B as chosen (some flipped, some not)
        assert chosen_values == {"A", "B"}


# ---------------------------------------------------------------------------
# Prompt construction (system-role persona, X/Y labels, two-stage CoT)
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal tokenizer stub for prompt-rendering tests (no HF dependency)."""

    chat_template = "stub"
    _encode_map = {"X": [88], "Y": [89], "1": [16], "2": [17]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            continue_final_message=False):
        # Render each turn as "[ROLE]\ncontent" so tests can grep substrings.
        parts = [f"[{m['role'].upper()}]\n{m['content']}" for m in messages]
        rendered = "\n\n".join(parts)
        if add_generation_prompt:
            rendered += "\n\n[ASSISTANT]\n"
        if continue_final_message:
            rendered += "<<CONTINUE>>"
        return rendered

    def encode(self, text, add_special_tokens=False):
        if text not in self._encode_map:
            raise KeyError(f"stub tokenizer has no mapping for {text!r}")
        return self._encode_map[text]


class TestComparisonMessages:
    """Persona must be in the system role, labels must be X/Y, no legacy 1/2."""

    def test_uses_system_role_with_persona(self):
        messages = _build_comparison_messages(
            prompt="What should I do?",
            response_first="resp one",
            response_second="resp two",
            user_profile="A 17th-century Dutch jurist",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Dutch jurist" in messages[0]["content"]

    def test_uses_xy_labels_in_user_turn(self):
        messages = _build_comparison_messages(
            prompt="Q?", response_first="r1", response_second="r2", user_profile="P",
        )
        user_text = messages[1]["content"]
        assert "Response X:" in user_text
        assert "Response Y:" in user_text
        assert "Answer: X" in user_text
        assert "Answer: Y" in user_text

    def test_no_legacy_label_leak(self):
        """Regression guard: no Option 1/2 or A/B labels remain anywhere."""
        messages = _build_comparison_messages(
            prompt="Q?", response_first="r1", response_second="r2", user_profile="P",
        )
        joined = "\n".join(m["content"] for m in messages)
        for forbidden in ("Option 1", "Option 2", "number 1 or 2",
                          "Response A", "Response B", "Answer: A", "Answer: B"):
            assert forbidden not in joined, f"Found legacy label {forbidden!r}"

    def test_no_profile_uses_impartial_system(self):
        messages = _build_comparison_messages(
            prompt="Q?", response_first="r1", response_second="r2", user_profile=None,
        )
        assert messages[0]["role"] == "system"
        assert "impartial" in messages[0]["content"].lower()



class TestRenderPrompts:
    """Both stage-1 and stage-2 renderers should round-trip through the chat template."""

    def test_stage1_includes_assistant_marker(self):
        tok = _StubTokenizer()
        messages = _build_comparison_messages("Q?", "r1", "r2", "P")
        rendered = _render_stage1_prompt(tok, messages)
        assert "[SYSTEM]" in rendered and "[USER]" in rendered
        assert rendered.endswith("[ASSISTANT]\n")

    def test_stage2_appends_assistant_continuation(self):
        tok = _StubTokenizer()
        messages = _build_comparison_messages("Q?", "r1", "r2", "P")
        stage1_text = "Response Y matches my values better.\nAnswer: Y"
        rendered, recovered = _render_stage2_prompt(tok, messages, stage1_text)
        # Continuation flag passed through stub tokenizer.
        assert "<<CONTINUE>>" in rendered
        # The stage-1 text appears in the assistant turn.
        assert "matches my values" in rendered
        # Stage 1 emitted "Answer:", so no recovery was needed.
        assert recovered is False

    def test_stage2_appends_answer_prefix_when_missing(self):
        tok = _StubTokenizer()
        messages = _build_comparison_messages("Q?", "r1", "r2", "P")
        # Reasoning that did not produce an "Answer:" line.
        stage1_text = "I prefer the second response on balance."
        rendered, recovered = _render_stage2_prompt(tok, messages, stage1_text)
        assert "Answer:" in rendered
        assert recovered is True


class TestResolveChoiceTokenIds:
    """Token resolver returns IDs for X/Y, not 1/2."""

    def test_returns_xy_ids(self):
        tok = _StubTokenizer()
        ids = _resolve_choice_token_ids(tok)
        assert ids == (88, 89)

    def test_raises_when_not_single_token(self):
        class MultiTokenTokenizer(_StubTokenizer):
            _encode_map = {"X": [88, 1], "Y": [89]}

        with pytest.raises(RuntimeError, match="single tokens"):
            _resolve_choice_token_ids(MultiTokenTokenizer())
