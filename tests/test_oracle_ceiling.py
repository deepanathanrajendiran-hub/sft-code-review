"""Tests for the diff-only oracle ceiling harness (no API: review_fn injected)."""
import json

import oracle_ceiling as oc


def test_filter_style_drops_style_tuples_only():
    labels = {
        "a": [{"issue_type": "bug", "canonical_desc": "x"},
              {"issue_type": "style", "canonical_desc": "typo"}],
        "b": [{"issue_type": "style", "canonical_desc": "nit"}],
        "c": [],
    }
    out = oc.filter_style(labels)
    assert [d["issue_type"] for d in out["a"]] == ["bug"]
    assert out["b"] == []  # style-only record becomes clean, not dropped
    assert out["c"] == []
    # input not mutated
    assert len(labels["a"]) == 2


def test_generate_oracle_writes_schema(tmp_path):
    rows = [{"instance_id": "i1", "diff": "d1"}, {"instance_id": "i2", "diff": "d2"}]
    out = tmp_path / "oracle.jsonl"
    oc.generate_oracle(rows, out, review_fn=lambda diff, max_tokens: f"review of {diff}")
    got = [json.loads(l) for l in out.open()]
    assert [(r["instance_id"], r["oracle_pred"]) for r in got] == [
        ("i1", "review of d1"), ("i2", "review of d2")]
    assert got[0]["diff"] == "d1"


def test_generate_oracle_resumes_without_regenerating(tmp_path):
    out = tmp_path / "oracle.jsonl"
    out.write_text(json.dumps({"instance_id": "i1", "diff": "d1", "oracle_pred": "old"}) + "\n")
    calls = []

    def fake(diff, max_tokens):
        calls.append(diff)
        return "new"

    rows = [{"instance_id": "i1", "diff": "d1"}, {"instance_id": "i2", "diff": "d2"}]
    oc.generate_oracle(rows, out, review_fn=fake)
    assert calls == ["d2"]  # i1 skipped
    got = {json.loads(l)["instance_id"]: json.loads(l)["oracle_pred"] for l in out.open()}
    assert got == {"i1": "old", "i2": "new"}
