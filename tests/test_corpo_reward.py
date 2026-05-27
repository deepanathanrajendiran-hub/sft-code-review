"""Unit tests for corpo_reward.py components."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from corpo_reward import (
    BaseSampleCache,
    composite_reward,
    hallucination_score,
    length_sanity_score,
    load_base_sample_cache,
    pairwise_score,
)


class TestLengthSanity:
    def test_within_band_returns_one(self):
        # Plateau [150, 1000] anchored on v4 OOD extracted-review distribution (median ~600 chars)
        assert length_sanity_score("x" * 150) == 1.0
        assert length_sanity_score("x" * 600) == 1.0   # v4 OOD median
        assert length_sanity_score("x" * 1000) == 1.0

    def test_too_short_linear_taper(self):
        # n=75: 75/150 = 0.5
        assert length_sanity_score("x" * 75) == 0.5
        assert length_sanity_score("") == 0.0

    def test_too_long_linear_taper(self):
        # n=1750: 1 - (1750 - 1000) / 1500 = 0.5
        assert length_sanity_score("x" * 1750) == 0.5
        # n=2500: floor
        assert length_sanity_score("x" * 2500) == 0.0
        # n=5000: stays at floor
        assert length_sanity_score("x" * 5000) == 0.0

    def test_just_below_band(self):
        # n=149: 149/150
        assert length_sanity_score("x" * 149) == pytest.approx(149 / 150)

    def test_just_above_band(self):
        # n=1001: 1 - (1001 - 1000) / 1500 = 1499/1500
        assert length_sanity_score("x" * 1001) == pytest.approx(1499 / 1500)


class TestHallucinationScore:
    def test_clean_review_scores_one(self, sample_diff):
        """Review referencing only identifiers actually in the diff scores 1.0."""
        # sample_diff is the auth.py diff with `user` mentioned
        clean_review = "The change to `user` on line 11 is correct."
        score = hallucination_score(sample_diff, clean_review)
        assert score == 1.0

    def test_hallucinated_identifiers_lower_score(self, sample_diff):
        """Review mentioning identifiers NOT in the diff lowers the score."""
        # `validate_token` is not in sample_diff
        bad_review = "The `validate_token` function should also check session age."
        score = hallucination_score(sample_diff, bad_review)
        assert score < 1.0

    def test_no_identifiers_returns_one(self, sample_diff):
        """Review with no backticked identifiers scores 1.0 (nothing to hallucinate)."""
        review = "Looks fine to me."
        assert hallucination_score(sample_diff, review) == 1.0

    def test_score_is_in_unit_interval(self, sample_diff):
        """Score is always in [0.0, 1.0]."""
        # Many hallucinations
        bad_review = "The `foo` and `bar` and `baz_qux` functions are wrong."
        score = hallucination_score(sample_diff, bad_review)
        assert 0.0 <= score <= 1.0


class TestPairwiseScore:
    def test_rollout_wins_returns_one(self, sample_diff):
        """When judge says rollout (passed as A) wins, score is 1.0."""
        with patch("corpo_reward._judge_fn") as judge:
            judge.return_value = "A"
            score = pairwise_score(
                diff=sample_diff,
                rollout="rollout text",
                base_sample="base text",
                reference="ref",
            )
            assert score == 1.0

    def test_rollout_loses_returns_zero(self, sample_diff):
        with patch("corpo_reward._judge_fn") as judge:
            judge.return_value = "B"
            score = pairwise_score(sample_diff, "rollout", "base", "ref")
            assert score == 0.0

    def test_tie_returns_half(self, sample_diff):
        with patch("corpo_reward._judge_fn") as judge:
            judge.return_value = "TIE"
            score = pairwise_score(sample_diff, "rollout", "base", "ref")
            assert score == 0.5

    def test_judge_called_with_rollout_as_a(self, sample_diff):
        """Verify we pass rollout=A, base=B to the judge (judge handles A/B swap internally)."""
        with patch("corpo_reward._judge_fn") as judge:
            judge.return_value = "A"
            pairwise_score(sample_diff, "ROLLOUT_X", "BASE_Y", "REF")
            args = judge.call_args
            # judge signature: (review_a, review_b, diff, reference)
            assert args.args[0] == "ROLLOUT_X"
            assert args.args[1] == "BASE_Y"
            assert args.args[2] == sample_diff
            assert args.args[3] == "REF"

class TestCompositeReward:
    def _wrap_as_rollout(self, review_body: str) -> str:
        """Wrap a review body in <think>+<review> structure for testing."""
        return f"<think>analysis</think><review>{review_body}</review>"

    def test_perfect_rollout_scores_one(self, sample_diff):
        """Wins pairwise + no halluc + length OK -> R = 0.7 + 0.2 + 0.1 = 1.0."""
        with patch("corpo_reward._judge_fn", return_value="A"):
            # 600 chars: in [150, 1000] band, length_sanity = 1.0
            rollout = self._wrap_as_rollout("x" * 600)
            r = composite_reward(
                diff=sample_diff,
                rollout=rollout,
                base_sample="base",
                reference="ref",
            )
            assert r == pytest.approx(1.0)

    def test_total_loss_scores_above_zero_due_to_unhalluc(self, sample_diff):
        """Lose pairwise (0) + no halluc (1.0) + tiny length -> ~0.2.

        Uses a 1-char extracted review: in-band-extracted but well below the
        length plateau, scoring 1/150 ≈ 0.00667 on length.
        """
        with patch("corpo_reward._judge_fn", return_value="B"):
            rollout = self._wrap_as_rollout("x")  # 1 char extracted, no idents
            r = composite_reward(sample_diff, rollout, "base", "ref")
            # pair=0, halluc=1.0, length = 1/150 = 0.00667
            # R = 0*0.7 + 1*0.2 + 0.00667*0.1 = 0.2007
            assert r == pytest.approx(0.2 + 0.00067, abs=0.001)

    def test_tie_with_clean_review_correct_weighting(self, sample_diff):
        """Tie pairwise (0.5) + no halluc (1.0) + length OK (1.0) -> 0.5*0.7 + 1*0.2 + 1*0.1 = 0.65."""
        with patch("corpo_reward._judge_fn", return_value="TIE"):
            rollout = self._wrap_as_rollout("x" * 600)
            r = composite_reward(sample_diff, rollout, "base", "ref")
            assert r == pytest.approx(0.65)

    def test_length_hacked_rollout_loses_reward(self, sample_diff):
        """A 75-char (extracted) length-hacked review scores below v4-OOD-length.

        Win pairwise (0.7) + no halluc (0.2) + length_sanity(75)=0.5 (0.05) = 0.95
        vs v4-OOD-length: 0.7 + 0.2 + 1.0*0.1 = 1.0
        """
        with patch("corpo_reward._judge_fn", return_value="A"):
            r_short = composite_reward(
                sample_diff, self._wrap_as_rollout("x" * 75), "base", "ref"
            )
            r_v4 = composite_reward(
                sample_diff, self._wrap_as_rollout("x" * 600), "base", "ref"
            )
            assert r_v4 - r_short == pytest.approx(0.05, abs=0.001)

    def test_returns_value_in_unit_interval(self, sample_diff):
        """Output always in [0.0, 1.0]."""
        with patch("corpo_reward._judge_fn", return_value="A"):
            r = composite_reward(sample_diff, self._wrap_as_rollout("x" * 50), "base", "ref")
            assert 0.0 <= r <= 1.0
        with patch("corpo_reward._judge_fn", return_value="B"):
            r = composite_reward(sample_diff, self._wrap_as_rollout("x" * 3000), "base", "ref")
            assert 0.0 <= r <= 1.0


class TestBaseSampleCache:
    def test_load_returns_cache_with_lookup(self, tmp_path):
        cache_path = tmp_path / "base.jsonl"
        with cache_path.open("w") as fh:
            fh.write(json.dumps({"instance_id": "id_1", "base_output": "B1"}) + "\n")
            fh.write(json.dumps({"instance_id": "id_2", "base_output": "B2"}) + "\n")
        cache: BaseSampleCache = load_base_sample_cache(cache_path)
        assert cache.get("id_1") == "B1"
        assert cache.get("id_2") == "B2"

    def test_missing_id_raises(self, tmp_path):
        cache_path = tmp_path / "base.jsonl"
        cache_path.write_text(json.dumps({"instance_id": "id_1", "base_output": "B1"}) + "\n")
        cache = load_base_sample_cache(cache_path)
        with pytest.raises(KeyError, match="id_99"):
            cache.get("id_99")

class TestReviewExtraction:
    """composite_reward should extract the <review> block before scoring length and hallucination."""

    def test_empty_rollout_returns_zero(self, sample_diff):
        """A rollout with no parseable review is a deployment failure → reward 0.0."""
        with patch("corpo_reward._judge_fn", return_value="A"):
            r = composite_reward(sample_diff, "", "base", "ref")
            assert r == 0.0

    def test_placeholder_only_rollout_returns_zero(self, sample_diff):
        """A rollout where the only <review> block is a literal '...' placeholder → reward 0.0.

        Matches the v4 deployment behavior where the extractor returns empty for
        placeholder reviews and the extracted string is unusable.
        """
        rollout = "<think>I'll write the review: <review>...</review></think>"
        with patch("corpo_reward._judge_fn", return_value="A"):
            r = composite_reward(sample_diff, rollout, "base", "ref")
            assert r == 0.0

    def test_valid_rollout_extracts_and_scores(self, sample_diff):
        """A rollout with <think> + valid <review> scores on the extracted review only."""
        review_body = "x" * 600  # in v4 OOD band → length=1.0
        rollout = f"<think>analysis here</think><review>{review_body}</review>"
        with patch("corpo_reward._judge_fn", return_value="A"):
            r = composite_reward(sample_diff, rollout, "base", "ref")
            # 0.7 (pair=1) + 0.2 (halluc=1, no idents in review_body) + 0.1 (length=1) = 1.0
            assert r == pytest.approx(1.0)

    def test_think_block_identifiers_ignored(self, sample_diff):
        """Identifiers in <think> that aren't in the diff should NOT count as hallucinations.

        Pre-Run-#3, the <think> block was scored for hallucinations, penalizing
        exploratory reasoning that referenced library APIs. Now only the
        extracted review is scored.
        """
        rollout = (
            "<think>maybe `validate_token` is involved here</think>"
            "<review>The change looks fine.</review>"
        )
        with patch("corpo_reward._judge_fn", return_value="A"):
            r = composite_reward(sample_diff, rollout, "base", "ref")
            # halluc on "The change looks fine." should be 1.0 (no backticked identifiers in review)
            assert r > 0.9  # only the small length penalty applies
