"""Unit tests for corpo_reward.py components."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from corpo_reward import hallucination_score, length_sanity_score, pairwise_score


class TestLengthSanity:
    def test_within_band_returns_one(self):
        assert length_sanity_score("x" * 200) == 1.0
        assert length_sanity_score("x" * 1000) == 1.0
        assert length_sanity_score("x" * 4000) == 1.0

    def test_too_short_linear_taper(self):
        assert length_sanity_score("x" * 100) == 0.5
        assert length_sanity_score("") == 0.0

    def test_too_long_linear_taper(self):
        # n=6000: 1 - (6000-4000)/4000 = 0.5
        assert length_sanity_score("x" * 6000) == 0.5
        # n=8000 floor
        assert length_sanity_score("x" * 8000) == 0.0
        # n=12000 stays at floor
        assert length_sanity_score("x" * 12000) == 0.0

    def test_just_below_band(self):
        # n=199: 199/200
        assert length_sanity_score("x" * 199) == pytest.approx(199 / 200)


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
