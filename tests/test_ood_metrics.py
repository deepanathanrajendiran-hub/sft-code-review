import pytest

from ood_metrics import (
    extract_locations,
    iou_strict,
    iou_lenient,
)


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
