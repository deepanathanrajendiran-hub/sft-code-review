"""Tests for label_defects: PR-thread -> clean grounded defect tuples.

The LLM classifier is injected (extract_fn) so all assembly/grounding logic is
tested with no API call. The default classifier (DeepSeek) is integration-only.
"""
import label_defects as ld

DIFF = (
    "diff --git a/auth.py b/auth.py\n"
    "--- a/auth.py\n"
    "+++ b/auth.py\n"
    "@@ -10,3 +10,3 @@\n"
    "-    if user:\n"
    "+    if user is not None:\n"
)


def test_diff_paths_extracts_changed_files():
    assert "auth.py" in ld.diff_paths(DIFF)


def test_parse_extraction_valid_defect():
    raw = '{"is_defect": true, "issue_type": "bug", "canonical_desc": "empty string not handled"}'
    d = ld.parse_extraction(raw)
    assert d["issue_type"] == "bug"
    assert d["canonical_desc"] == "empty string not handled"


def test_parse_extraction_non_defect_returns_none():
    assert ld.parse_extraction('{"is_defect": false}') is None


def test_parse_extraction_handles_fenced_json():
    raw = '```json\n{"is_defect": true, "issue_type": "security", "canonical_desc": "x"}\n```'
    assert ld.parse_extraction(raw)["issue_type"] == "security"


def test_parse_extraction_malformed_returns_none():
    assert ld.parse_extraction("not json at all") is None


def test_extract_record_keeps_grounded_defect():
    rec = {"instance_id": "i1", "diff": DIFF, "reference_comments": [
        {"path": "auth.py", "line": 11, "text": "this breaks on empty string"}]}

    def fake(diff, comment):
        return {"is_defect": True, "issue_type": "bug", "canonical_desc": "empty string"}

    out = ld.extract_defects_for_record(rec, extract_fn=fake)
    assert out["instance_id"] == "i1"
    assert out["defects"] == [
        {"path": "auth.py", "line": 11, "issue_type": "bug", "canonical_desc": "empty string"}]


def test_extract_record_drops_non_defect():
    rec = {"instance_id": "i1", "diff": DIFF, "reference_comments": [
        {"path": "auth.py", "line": 11, "text": "why did you do this?"}]}
    out = ld.extract_defects_for_record(rec, extract_fn=lambda d, c: None)
    assert out["defects"] == []


def test_extract_record_drops_ungrounded_path():
    rec = {"instance_id": "i1", "diff": DIFF, "reference_comments": [
        {"path": "not_in_diff.py", "line": 3, "text": "real bug"}]}

    def fake(diff, comment):
        return {"is_defect": True, "issue_type": "bug", "canonical_desc": "x"}

    out = ld.extract_defects_for_record(rec, extract_fn=fake)
    assert out["defects"] == []  # path absent from diff -> uncatchable -> dropped


def test_extract_record_drops_pathless_comment():
    rec = {"instance_id": "i1", "diff": DIFF, "reference_comments": [
        {"path": None, "line": None, "text": "general comment"}]}

    def fake(diff, comment):
        return {"is_defect": True, "issue_type": "bug", "canonical_desc": "x"}

    out = ld.extract_defects_for_record(rec, extract_fn=fake)
    assert out["defects"] == []
