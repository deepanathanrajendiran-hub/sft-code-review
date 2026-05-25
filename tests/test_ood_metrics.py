import pytest

from ood_metrics import (
    extract_locations,
    iou_strict,
    iou_lenient, breakdown_by_difficulty, breakdown_by_problem_domain, pairwise_win,
    haiku_pairwise_judge_3vote,
    bootstrap_winrate_ci,
)
from ood_metrics import hit_rate, hit_rate_strict, hallucination_rate, STOPWORDS
from unittest.mock import patch, MagicMock
import os


class TestExtractLocations:
    def test_exact_line_citation(self):
        review = "The bug is in `auth.py:42` where the check fails."
        locs = extract_locations(review)
        assert ("auth.py", 42) in [(loc["file"], loc["line"]) for loc in locs]

    def test_path_with_subdirs(self):
        review = "Issue at `src/api/auth.py:100`."
        locs = extract_locations(review)
        assert any(loc["file"] == "src/api/auth.py" and loc["line"] == 100 for loc in locs)

    def test_identifier_only(self):
        review = "The `validate_token` function is missing a null check."
        locs = extract_locations(review)
        assert any(loc.get("identifier") == "validate_token" for loc in locs)

    def test_empty_review(self):
        assert extract_locations("") == []

    def test_no_code_references(self):
        review = "This looks fine to me."
        assert extract_locations(review) == []

    def test_multiple_defects(self):
        review = (
            "Two issues: `auth.py:10` has a typo, and `validators.py:55` "
            "checks the wrong field."
        )
        locs = extract_locations(review)
        files = {(loc["file"], loc["line"]) for loc in locs if loc.get("line")}
        assert ("auth.py", 10) in files
        assert ("validators.py", 55) in files

    def test_identifier_dedup(self):
        review = "`validate_token` here, `validate_token` again, and `validate_token` once more"
        locs = extract_locations(review)
        idents = [loc.get("identifier") for loc in locs if loc.get("identifier")]
        assert idents.count("validate_token") == 1, (
            f"identifier 'validate_token' should be deduplicated; got {idents}"
        )

    def test_jsx_tsx_extensions(self):
        review = "Issue at `App.tsx:42` and `Component.jsx:10`."
        locs = extract_locations(review)
        files = {(loc["file"], loc["line"]) for loc in locs if loc.get("file")}
        assert ("App.tsx", 42) in files
        assert ("Component.jsx", 10) in files


class TestIouStrict:
    def test_perfect_match(self):
        pred = {"v4_pred": "`auth.py:11` is wrong"}
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert iou_strict(pred, labels) == 1.0

    def test_no_overlap(self):
        pred = {"v4_pred": "`other.py:50`"}
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert iou_strict(pred, labels) == 0.0

    def test_partial_overlap(self):
        # pred mentions 2 lines, label has 1, intersection = 1, union = 2
        pred = {"v4_pred": "`auth.py:11` and `auth.py:22`"}
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert iou_strict(pred, labels) == 0.5

    def test_empty_pred(self):
        pred = {"v4_pred": ""}
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert iou_strict(pred, labels) == 0.0

    def test_empty_labels(self):
        pred = {"v4_pred": "`auth.py:11`"}
        labels = []
        assert iou_strict(pred, labels) == 0.0

    def test_both_empty_returns_one(self):
        pred = {"v4_pred": ""}
        labels = []
        assert iou_strict(pred, labels) == 1.0


class TestIouLenient:
    def test_within_5_lines_counts(self):
        pred = {"v4_pred": "`auth.py:14`"}  # label at line 11
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert iou_lenient(pred, labels) == 1.0

    def test_outside_5_lines_does_not_count(self):
        pred = {"v4_pred": "`auth.py:20`"}  # label at line 11
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert iou_lenient(pred, labels) == 0.0

    def test_same_identifier_counts_without_line(self):
        pred = {"v4_pred": "The `validate_token` function has a bug."}
        labels = [{
            "path": "auth.py",
            "line": 11,
            "text": "validate_token check should be moved up",
        }]
        assert iou_lenient(pred, labels) == 1.0

    def test_short_identifier_does_not_substring_match(self):
        # Short identifier "err" should NOT match "error" via substring.
        pred = {"v4_pred": "`err` is the wrong variable"}
        labels = [{
            "path": "x.py",
            "line": 1,
            "text": "the error handling here is wrong",
        }]
        # "err" should NOT word-boundary-match within "error"
        assert iou_lenient(pred, labels) == 0.0

    def test_both_empty_returns_one(self):
        pred = {"v4_pred": ""}
        labels = []
        assert iou_lenient(pred, labels) == 1.0


class TestHitRate:
    def test_catches_all_labels(self):
        pred = {"v4_pred": "`auth.py:11` and `auth.py:22`"}
        labels = [
            {"path": "auth.py", "line": 11, "text": "..."},
            {"path": "auth.py", "line": 22, "text": "..."},
        ]
        assert hit_rate(pred, labels) == 1.0

    def test_catches_half(self):
        pred = {"v4_pred": "`auth.py:11`"}
        labels = [
            {"path": "auth.py", "line": 11, "text": "..."},
            {"path": "auth.py", "line": 22, "text": "..."},
        ]
        assert hit_rate(pred, labels) == 0.5

    def test_catches_none(self):
        pred = {"v4_pred": "`auth.py:99`"}
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert hit_rate(pred, labels) == 0.0

    def test_empty_labels_returns_one(self):
        pred = {"v4_pred": "`auth.py:11`"}
        assert hit_rate(pred, []) == 1.0  # vacuous — nothing was supposed to be flagged

    def test_empty_pred_returns_zero(self):
        pred = {"v4_pred": ""}
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert hit_rate(pred, labels) == 0.0


class TestHallucinationRate:
    def test_no_hallucination(self, sample_diff):
        # mentions only identifiers from the diff
        review = "The check at `user` should also handle empty strings."
        pred = {"v4_pred": review, "diff": sample_diff}
        assert hallucination_rate(pred) == 0.0

    def test_hallucinated_identifier(self, sample_diff):
        review = "The `nonexistent_function` is broken."
        pred = {"v4_pred": review, "diff": sample_diff}
        assert hallucination_rate(pred) > 0.0

    def test_stopwords_not_flagged(self, sample_diff):
        review = "This `returns` `True` `where` valid."
        pred = {"v4_pred": review, "diff": sample_diff}
        # all are in STOPWORDS, none should be flagged
        assert hallucination_rate(pred) == 0.0

    def test_stopwords_populated(self):
        # sanity: stopword list should include tokens from the synced eval.ipynb _STOP
        for token in ["Optional", "True", "False", "None", "async", "await", "returns"]:
            assert token in STOPWORDS, f"{token} missing from STOPWORDS"

    def test_no_review_returns_zero(self, sample_diff):
        pred = {"v4_pred": "", "diff": sample_diff}
        assert hallucination_rate(pred) == 0.0

    def test_no_backticks_returns_zero(self, sample_diff):
        pred = {"v4_pred": "Looks fine to me.", "diff": sample_diff}
        assert hallucination_rate(pred) == 0.0

    def test_denominator_excludes_stopwords(self, sample_diff):
        # review has: 2 stopwords (True, None), 1 real diff identifier (user),
        # 1 hallucinated identifier (nonexistent_function)
        # candidates = {nonexistent_function, user} (stopwords excluded)
        # hallucinated = {nonexistent_function}
        # Expected: 1/2 = 0.5 (not 1/4 = 0.25 if denominator used all backticked)
        review = "Issues with `True`, `None`, `user`, and `nonexistent_function`."
        pred = {"v4_pred": review, "diff": sample_diff}
        result = hallucination_rate(pred)
        assert abs(result - 0.5) < 0.01, f"Expected 0.5, got {result}"


class TestHitRateStrict:
    def test_strict_exact_match(self):
        pred = {"v4_pred": "`auth.py:11`"}
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert hit_rate_strict(pred, labels) == 1.0

    def test_strict_off_by_one_fails(self):
        # ±5 lenient would match; strict does not
        pred = {"v4_pred": "`auth.py:12`"}
        labels = [{"path": "auth.py", "line": 11, "text": "..."}]
        assert hit_rate_strict(pred, labels) == 0.0

    def test_strict_empty_labels(self):
        pred = {"v4_pred": "`auth.py:11`"}
        assert hit_rate_strict(pred, []) == 1.0


class TestBreakdownByDifficulty:
    def test_groups_by_field(self):
        preds = [
            {"difficulty": "low", "v4_pred": "`auth.py:11`"},
            {"difficulty": "low", "v4_pred": "`auth.py:20`"},
            {"difficulty": "hard", "v4_pred": "`auth.py:11`"},
        ]
        labels_by_instance = [
            [{"path": "auth.py", "line": 11, "text": "..."}],
            [{"path": "auth.py", "line": 11, "text": "..."}],
            [{"path": "auth.py", "line": 11, "text": "..."}],
        ]
        result = breakdown_by_difficulty(preds, labels_by_instance, hit_rate)
        assert "low" in result
        assert "hard" in result
        assert result["low"] == 0.5  # 1/2 caught
        assert result["hard"] == 1.0  # 1/1 caught


class TestBreakdownByProblemDomain:
    def test_groups_by_field(self):
        preds = [
            {"problem_domain": "Bug Fixes", "v4_pred": "`auth.py:11`"},
            {"problem_domain": "Feature", "v4_pred": "`auth.py:99`"},
        ]
        labels_by_instance = [
            [{"path": "auth.py", "line": 11, "text": "..."}],
            [{"path": "auth.py", "line": 11, "text": "..."}],
        ]
        result = breakdown_by_problem_domain(preds, labels_by_instance, hit_rate)
        assert result["Bug Fixes"] == 1.0
        assert result["Feature"] == 0.0


class TestPairwiseWin:
    @patch("ood_metrics.haiku_pairwise_judge_3vote")
    def test_majority_outcome(self, mock_vote):
        mock_vote.return_value = "A"  # v4 wins 1/1
        preds = [{
            "v4_pred": "good", "base_pred": "bad",
            "diff": "...", "reference_text": "expert"
        }]
        result = pairwise_win(preds)
        assert result["v4_wins"] == 1
        assert result["total"] == 1
        assert result["win_rate"] == 1.0
        assert "win_rate_ci_lo" in result
        assert "win_rate_ci_hi" in result


class TestBootstrapWinrateCi:
    def test_zero_verdicts(self):
        mean, lo, hi = bootstrap_winrate_ci([], which="A")
        assert mean == 0.0 and lo == 0.0 and hi == 0.0

    def test_all_a_winrate_is_one(self):
        mean, lo, hi = bootstrap_winrate_ci(["A"] * 50, which="A")
        assert mean == 1.0
        assert hi == 1.0
        assert lo == 1.0  # zero variance

    def test_half_and_half(self):
        verdicts = ["A"] * 50 + ["B"] * 50
        mean, lo, hi = bootstrap_winrate_ci(verdicts, which="A", n_iter=500)
        assert abs(mean - 0.5) < 0.01
        assert lo < 0.5 < hi


class TestDeepSeekV4FlashJudge:
    def test_returns_a_when_no_swap_and_model_says_a(self, sample_diff):
        """random < 0.5 → no swap; model says 'A' → return 'A' unchanged."""
        with patch("ood_metrics.random.random", return_value=0.4), \
             patch("ood_metrics._get_deepseek_client") as get_client:
            mock = MagicMock()
            mock.chat.completions.create.return_value.choices = [
                MagicMock(message=MagicMock(content="A"))
            ]
            get_client.return_value = mock
            from ood_metrics import deepseek_v4flash_pairwise_judge
            verdict = deepseek_v4flash_pairwise_judge(
                review_a="good review",
                review_b="bad review",
                diff=sample_diff,
                reference="ref",
            )
            assert verdict == "A"

    def test_returns_b_when_swap_and_model_says_a(self, sample_diff):
        """random > 0.5 → swap (B becomes A internally); model says 'A' → flip back to 'B'."""
        with patch("ood_metrics.random.random", return_value=0.6), \
             patch("ood_metrics._get_deepseek_client") as get_client:
            mock = MagicMock()
            mock.chat.completions.create.return_value.choices = [
                MagicMock(message=MagicMock(content="A"))
            ]
            get_client.return_value = mock
            from ood_metrics import deepseek_v4flash_pairwise_judge
            verdict = deepseek_v4flash_pairwise_judge(
                review_a="good review",
                review_b="bad review",
                diff=sample_diff,
                reference="ref",
            )
            assert verdict == "B"

    def test_returns_tie_on_garbage(self, sample_diff):
        with patch("ood_metrics._get_deepseek_client") as get_client:
            mock = MagicMock()
            mock.chat.completions.create.return_value.choices = [
                MagicMock(message=MagicMock(content="???"))
            ]
            get_client.return_value = mock
            from ood_metrics import deepseek_v4flash_pairwise_judge
            verdict = deepseek_v4flash_pairwise_judge("x", "y", sample_diff, "")
            assert verdict == "TIE"

    def test_client_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        # Reset module-level client cache
        import ood_metrics
        ood_metrics._deepseek_client = None
        with patch("openai.OpenAI") as openai_ctor:
            ood_metrics._get_deepseek_client()
            openai_ctor.assert_called_once()
            kwargs = openai_ctor.call_args.kwargs
            assert kwargs["api_key"] == "sk-test"
            assert "deepseek" in kwargs["base_url"].lower()


class TestDeepSeekV4ProJudge3Vote:
    def test_majority_a(self, sample_diff):
        with patch("ood_metrics.deepseek_v4pro_pairwise_judge") as one_vote:
            one_vote.side_effect = ["A", "A", "B"]
            from ood_metrics import deepseek_v4pro_pairwise_judge_3vote
            verdict = deepseek_v4pro_pairwise_judge_3vote("x", "y", sample_diff, "ref")
            assert verdict == "A"

    def test_no_majority_returns_tie(self, sample_diff):
        with patch("ood_metrics.deepseek_v4pro_pairwise_judge") as one_vote:
            one_vote.side_effect = ["A", "B", "TIE"]
            from ood_metrics import deepseek_v4pro_pairwise_judge_3vote
            verdict = deepseek_v4pro_pairwise_judge_3vote("x", "y", sample_diff, "ref")
            assert verdict == "TIE"

    def test_single_call_uses_v4pro_model(self, sample_diff):
        with patch("ood_metrics._get_deepseek_client") as get_client:
            mock = MagicMock()
            mock.chat.completions.create.return_value.choices = [
                MagicMock(message=MagicMock(content="A"))
            ]
            get_client.return_value = mock
            from ood_metrics import deepseek_v4pro_pairwise_judge, DEEPSEEK_V4_PRO
            deepseek_v4pro_pairwise_judge("x", "y", sample_diff, "ref")
            kwargs = mock.chat.completions.create.call_args.kwargs
            assert kwargs["model"] == DEEPSEEK_V4_PRO
