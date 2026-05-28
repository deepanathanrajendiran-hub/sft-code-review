"""Tests for score_v5: judge-independent recall + hallucination scorer (the v5 gate/eval).

Semantic matcher injected (no API). Reuses corpo_reward.recall_or_restraint +
ood_metrics.hallucination_rate, which are themselves tested.
"""
import score_v5


def test_score_aggregates_recall_and_halluc():
    preds = [
        {"instance_id": "i1", "diff": "@@\n+x = `foo`\n", "v4_pred": "the `foo` bug is real"},
        {"instance_id": "i2", "diff": "@@\n+y = 1\n", "v4_pred": "looks fine, no issues"},  # clean
    ]
    labels = {
        "i1": [{"path": "a.py", "line": 1, "issue_type": "bug", "canonical_desc": "foo bug"}],
        "i2": [],
    }
    mf = lambda review, defect: "foo" in review  # catches i1's defect

    out = score_v5.score(preds, labels, "v4_pred", match_fn=mf)
    assert out["n_total"] == 2
    assert out["n_labeled"] == 1
    # i1 recall=1.0 (matcher hit); i2 restraint=grounding=1.0 (no backticks -> no halluc)
    assert abs(out["recall_mean"] - 1.0) < 1e-9
    assert 0.0 <= out["halluc_mean"] <= 1.0


def test_recall_drops_when_matcher_misses():
    preds = [{"instance_id": "i1", "diff": "@@\n+x=1\n", "v4_pred": "unrelated remark"}]
    labels = {"i1": [{"path": "a.py", "line": 1, "issue_type": "bug", "canonical_desc": "z"}]}
    out = score_v5.score(preds, labels, "v4_pred", match_fn=lambda r, d: False)
    assert out["recall_mean"] == 0.0
    assert out["n_labeled"] == 1


def test_missing_instance_treated_as_clean():
    preds = [{"instance_id": "i9", "diff": "@@\n+x=1\n", "v4_pred": "fine"}]
    out = score_v5.score(preds, {}, "v4_pred", match_fn=lambda r, d: True)
    assert out["n_labeled"] == 0
    assert abs(out["recall_mean"] - 1.0) < 1e-9  # restraint: grounded clean review
