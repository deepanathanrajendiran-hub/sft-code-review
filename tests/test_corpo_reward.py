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
        # Plateau [2500, 5000] anchored on v4's 3500-char mean ±30%
        assert length_sanity_score("x" * 2500) == 1.0
        assert length_sanity_score("x" * 3500) == 1.0
        assert length_sanity_score("x" * 5000) == 1.0

    def test_too_short_linear_taper(self):
        # n=1250: 1250/2500 = 0.5
        assert length_sanity_score("x" * 1250) == 0.5
        # n=750: length-hacked output gets ~0.30
        assert length_sanity_score("x" * 750) == pytest.approx(0.30, rel=0.01)
        assert length_sanity_score("") == 0.0

    def test_too_long_linear_taper(self):
        # n=7500: 1 - (7500-5000)/5000 = 0.5
        assert length_sanity_score("x" * 7500) == 0.5
        # n=10000 floor
        assert length_sanity_score("x" * 10000) == 0.0
        # n=12000 stays at floor
        assert length_sanity_score("x" * 12000) == 0.0

    def test_just_below_band(self):
        # n=2499: 2499/2500
        assert length_sanity_score("x" * 2499) == pytest.approx(2499 / 2500)


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
    def test_perfect_rollout_scores_one(self, sample_diff):
        """Wins pairwise + no halluc + length OK -> R = 0.6 + 0.3 + 0.1 = 1.0."""
        with patch("corpo_reward._judge_fn", return_value="A"):
            r = composite_reward(
                diff=sample_diff,
                # 3500 chars = v4-baseline length, length_sanity = 1.0
                rollout="x" * 3500,
                base_sample="base",
                reference="ref",
            )
            assert r == pytest.approx(1.0)

    def test_total_loss_scores_above_zero_due_to_unhalluc(self, sample_diff):
        """Lose pairwise (0) + no halluc (1.0) + bad length (0) -> 0*0.6 + 1*0.3 + 0*0.1 = 0.3."""
        with patch("corpo_reward._judge_fn", return_value="B"):
            r = composite_reward(sample_diff, "", "base", "ref")
            assert r == pytest.approx(0.3)

    def test_tie_with_clean_review_correct_weighting(self, sample_diff):
        """Tie pairwise (0.5) + no halluc (1.0) + length OK (1.0) -> 0.5*0.6 + 1*0.3 + 1*0.1 = 0.7."""
        with patch("corpo_reward._judge_fn", return_value="TIE"):
            # 3500 chars = v4-baseline, length_sanity = 1.0
            r = composite_reward(sample_diff, "x" * 3500, "base", "ref")
            assert r == pytest.approx(0.7)

    def test_length_hacked_rollout_loses_reward(self, sample_diff):
        """A 750-char length-hacked output (run #1 collapse mode) scores below v4-length output.

        Win pairwise (0.6) + no halluc (0.3) + length_sanity(750)=0.30 (0.03) = 0.93
        vs v4-length: Win pairwise (0.6) + no halluc (0.3) + length_sanity(3500)=1.0 (0.1) = 1.0
        The 0.07 gap is the active disincentive against length collapse.
        """
        with patch("corpo_reward._judge_fn", return_value="A"):
            r_short = composite_reward(sample_diff, "x" * 750, "base", "ref")
            r_v4 = composite_reward(sample_diff, "x" * 3500, "base", "ref")
            assert r_v4 - r_short == pytest.approx(0.07, abs=0.001)

    def test_returns_value_in_unit_interval(self, sample_diff):
        """Output always in [0.0, 1.0]."""
        with patch("corpo_reward._judge_fn", return_value="A"):
            r = composite_reward(sample_diff, "x" * 100, "base", "ref")
            assert 0.0 <= r <= 1.0
        with patch("corpo_reward._judge_fn", return_value="B"):
            r = composite_reward(sample_diff, "x" * 12000, "base", "ref")
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
