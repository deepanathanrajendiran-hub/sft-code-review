import json
import pytest
from pathlib import Path

from swecare_loader import (
    get_train_repo_names,
    matches_train_repo,
    map_swecare_row,
)


class TestMatchesTrainRepo:
    @pytest.mark.parametrize(
        "swecare_repo,train_names,expected",
        [
            # exact name match
            ("huggingface/transformers", {"transformers"}, True),
            # different owner, same name → still match
            ("forked/transformers", {"transformers"}, True),
            # substring should NOT match
            ("acme/transformers-tutorial", {"transformers"}, False),
            ("acme/old-transformers", {"transformers"}, False),
            # case insensitive
            ("HuggingFace/Transformers", {"transformers"}, True),
            # no match
            ("django/django", {"transformers", "sklearn"}, False),
            # scikit-learn alias
            ("scikit-learn/scikit-learn", {"scikit-learn", "sklearn"}, True),
        ],
    )
    def test_matches(self, swecare_repo, train_names, expected):
        assert matches_train_repo(swecare_repo, train_names) is expected


class TestMapSwecareRow:
    def test_extracts_diff(self, sample_swecare_row):
        out = map_swecare_row(sample_swecare_row)
        assert out["diff"].startswith("diff --git a/auth.py")
        assert "if user is not None" in out["diff"]

    def test_carries_metadata(self, sample_swecare_row):
        out = map_swecare_row(sample_swecare_row)
        assert out["instance_id"] == "django__django-12345@abc"
        assert out["repo"] == "django/django"
        assert out["difficulty"] == "low"
        assert out["problem_domain"] == "Bug Fixes"

    def test_carries_reference_comments(self, sample_swecare_row):
        out = map_swecare_row(sample_swecare_row)
        assert len(out["reference_comments"]) == 1
        assert out["reference_comments"][0]["path"] == "auth.py"
        assert out["reference_comments"][0]["line"] == 11

    def test_remaps_line_from_original_line(self):
        row = {
            "instance_id": "x", "repo": "x/y", "commit_to_review": {"patch_to_review": ""},
            "reference_review_comments": [
                {"path": "a.py", "line": None, "original_line": 42, "text": "t1"},
                {"path": "b.py", "line": 7, "original_line": 99, "text": "t2"},
            ],
            "metadata": {"difficulty": "low", "problem_domain": "Bug"},
        }
        out = map_swecare_row(row)
        assert out["reference_comments"][0]["line"] == 42  # remapped from original_line
        assert out["reference_comments"][1]["line"] == 7   # kept (not None)

    def test_emits_reference_text(self):
        row = {
            "instance_id": "x", "repo": "x/y", "commit_to_review": {"patch_to_review": ""},
            "reference_review_comments": [
                {"path": "a.py", "line": 1, "text": "alpha"},
                {"path": "a.py", "line": 2, "text": "beta"},
            ],
            "metadata": {},
        }
        out = map_swecare_row(row)
        assert "alpha" in out["reference_text"]
        assert "beta" in out["reference_text"]


class TestGetTrainRepoNames:
    def test_extracts_unique_names(self, tmp_path):
        jsonl = tmp_path / "train.jsonl"
        jsonl.write_text(
            json.dumps({"repo": "huggingface/transformers"}) + "\n"
            + json.dumps({"repo": "scikit-learn/scikit-learn"}) + "\n"
            + json.dumps({"repo": "huggingface/transformers"}) + "\n"  # dup
            + json.dumps({"repo": "pydantic/pydantic"}) + "\n"
            + json.dumps({"repo": "tiangolo/fastapi"}) + "\n"
        )
        names = get_train_repo_names(jsonl)
        assert names == {"transformers", "scikit-learn", "pydantic", "fastapi"}

    def test_missing_file_returns_default(self, tmp_path):
        names = get_train_repo_names(tmp_path / "does_not_exist.jsonl")
        assert names == {"transformers", "sklearn", "scikit-learn", "pydantic", "fastapi"}
