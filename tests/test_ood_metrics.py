import pytest

from ood_metrics import (
    extract_locations,
    iou_strict,
    iou_lenient, breakdown_by_difficulty, breakdown_by_problem_domain, pairwise_win,
)
from ood_metrics import hit_rate, hallucination_rate, STOPWORDS
from unittest.mock import MagicMock, patch


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

    def test_empty_labels_returns_zero(self):
        pred = {"v4_pred": "`auth.py:11`"}
        assert hit_rate(pred, []) == 0.0

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
        review = "This `Returns` `True` `where` valid."
        pred = {"v4_pred": review, "diff": sample_diff}
        # all are in STOPWORDS, none should be flagged
        assert hallucination_rate(pred) == 0.0

    def test_stopwords_populated(self):
        # sanity: stopword list should include common Python keywords
        for token in ["Optional", "True", "False", "None", "Returns"]:
            assert token in STOPWORDS, f"{token} missing from STOPWORDS"

    def test_no_review_returns_zero(self, sample_diff):
        pred = {"v4_pred": "", "diff": sample_diff}
        assert hallucination_rate(pred) == 0.0

    def test_no_backticks_returns_zero(self, sample_diff):
        pred = {"v4_pred": "Looks fine to me.", "diff": sample_diff}
        assert hallucination_rate(pred) == 0.0

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
    @patch("ood_metrics._haiku_vote")
    def test_majority_voting(self, mock_vote):
        # 2/3 votes prefer v4
        mock_vote.side_effect = ["v4", "base", "v4"]
        preds = [{"v4_pred": "good", "base_pred": "bad", "diff": "..."}]
        result = pairwise_win(preds, api_key="fake")
        assert result["v4_wins"] == 1
        assert result["total"] == 1
        assert result["win_rate"] == 1.0
