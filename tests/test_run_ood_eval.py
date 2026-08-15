"""Unit tests for the pure functions in run_ood_eval.py.

Only `_extract_review` is tested here — `_generate` requires vLLM/GPU
and is exercised by the Colab smoke run (Task 8).
"""
from pathlib import Path

import pytest

from run_ood_eval import _extract_review


class TestExtractReview:
    def test_placeholder_inside_think_skipped(self):
        raw = (
            "<think>Will write: `<review>...</review>` next.</think>"
            "<review>The bug is in auth.py</review>"
        )
        assert _extract_review(raw) == "The bug is in auth.py"

    def test_plain_review_no_think(self):
        raw = "<review>Looks fine.</review>"
        assert _extract_review(raw) == "Looks fine."

    def test_no_tags_base_model(self):
        raw = "This change is correct."
        assert _extract_review(raw) == "This change is correct."

    def test_only_dots_review_returns_empty(self):
        # Case 7: only a placeholder `<review>...</review>` with no real review
        raw = "<review>...</review>"
        assert _extract_review(raw) == ""

    def test_multiple_reviews_takes_last_non_dots(self):
        # Multiple review blocks; last non-placeholder wins (no </think>)
        raw = (
            "<review>...</review> filler "
            "<review>Real review here</review>"
        )
        assert _extract_review(raw) == "Real review here"

    def test_review_after_last_think_preferred(self):
        # Two `<review>` blocks, one inside think, one after — the AFTER wins
        raw = (
            "<think>Earlier review: <review>this is wrong</review></think>"
            "<review>The actual review</review>"
        )
        assert _extract_review(raw) == "The actual review"

    def test_truncated_output_no_review_at_all(self):
        # Output got cut off mid-think
        raw = "<think>Thinking about it"
        # Falls through to fallback: strip `<think>...</think>` (no match) → returns raw
        result = _extract_review(raw)
        # The raw string is returned unmodified since there's no closing </think>
        assert result == "<think>Thinking about it"

    def test_empty_string(self):
        assert _extract_review("") == ""


def test_skip_base_flag_in_argparser():
    """run_ood_eval.py exposes --skip-base flag."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "run_ood_eval.py", "--help"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent)
    )
    assert "--skip-base" in result.stdout, f"--skip-base missing from help; got:\n{result.stdout}"
