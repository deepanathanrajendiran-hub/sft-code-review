import pytest

@pytest.fixture
def sample_swecare_row():
    """A representative row matching inclusionAI/SWE-CARE schema."""
    return {
        "instance_id": "django__django-12345@abc",
        "repo": "django/django",
        "language": "Python",
        "pull_number": 12345,
        "title": "Fix auth bug",
        "body": "Closes #100",
        "created_at": "2024-01-01T00:00:00Z",
        "problem_statement": "auth fails on empty session",
        "hints_text": "",
        "resolved_issues": [],
        "base_commit": "a" * 40,
        "commit_to_review": {
            "head_commit": "b" * 40,
            "head_commit_message": "fix",
            "patch_to_review": (
                "diff --git a/auth.py b/auth.py\n"
                "--- a/auth.py\n"
                "+++ b/auth.py\n"
                "@@ -10,3 +10,3 @@\n"
                "-    if user:\n"
                "+    if user is not None:\n"
                "         return True\n"
            ),
        },
        "reference_review_comments": [
            {
                "text": "consider also handling empty string",
                "path": "auth.py",
                "diff_hunk": "@@ -10,3 +10,3 @@",
                "line": 11,
                "start_line": 11,
                "original_line": 10,
                "original_start_line": 10,
            }
        ],
        "merged_commit": "c" * 40,
        "merged_patch": "",
        "metadata": {
            "problem_domain": "Bug Fixes",
            "difficulty": "low",
            "estimated_review_effort": 2,
        },
    }


@pytest.fixture
def sample_prediction():
    """A v4-style review output with `<think>` and `<review>` blocks."""
    return (
        "<think>\nThe diff changes `if user:` to `if user is not None:` "
        "on line 11 of auth.py. This is a defensive fix.\n</think>\n"
        "<review>\nThe change to `auth.py:11` is correct. Consider also "
        "handling empty string case for `user`.\n</review>"
    )


@pytest.fixture
def sample_diff():
    """Just the patch_to_review string from sample_swecare_row."""
    return (
        "diff --git a/auth.py b/auth.py\n"
        "--- a/auth.py\n"
        "+++ b/auth.py\n"
        "@@ -10,3 +10,3 @@\n"
        "-    if user:\n"
        "+    if user is not None:\n"
        "         return True\n"
    )
