"""Tests for score_v5: precision-aware judge-independent scorer (the v5 gate/eval).

Reports defect-recall + precision on labeled records, false-positive rate on clean
records, and hallucination — so the verdict reads the SAME signal the reward optimizes.
Matcher + claim-counter injected (no API).
"""
import score_v5


def test_reports_recall_precision_fp_halluc():
    preds = [
        {"instance_id": "i1", "diff": "@@\n+x = `foo`\n", "v4_pred": "the `foo` bug is real"},
        {"instance_id": "i2", "diff": "@@\n+y = 1\n", "v4_pred": "looks fine, no issues"},  # clean, quiet
    ]
    labels = {
        "i1": [{"path": "a.py", "line": 1, "issue_type": "bug", "canonical_desc": "foo bug"}],
        "i2": [],
    }
    mf = lambda review, d: "foo" in review
    cf = lambda review: 1 if "bug" in review else 0  # i1 asserts 1 defect, i2 asserts 0
    out = score_v5.score(preds, labels, "v4_pred", match_fn=mf, count_fn=cf)
    assert out["n_total"] == 2
    assert out["n_labeled"] == 1
    assert abs(out["defect_recall_labeled"] - 1.0) < 1e-9   # caught foo
    assert abs(out["precision_labeled"] - 1.0) < 1e-9       # 1 caught / 1 claim
    assert abs(out["fp_rate_clean"] - 0.0) < 1e-9           # clean record made 0 claims
    assert 0.0 <= out["halluc_mean"] <= 1.0


def test_clean_invented_defect_is_a_false_positive():
    preds = [{"instance_id": "i2", "diff": "@@\n+y = 1\n", "v4_pred": "the `y` variable is unsafe"}]
    out = score_v5.score(preds, {"i2": []}, "v4_pred",
                         match_fn=lambda r, d: False, count_fn=lambda r: 1)
    assert out["n_labeled"] == 0
    assert out["fp_rate_clean"] == 1.0  # asserted a defect on a clean diff
    assert out["defect_recall_labeled"] is None


def test_shotgun_lowers_precision():
    preds = [{"instance_id": "i1", "diff": "@@\n+x=1\n", "v4_pred": "many issues here"}]
    labels = {"i1": [{"path": "a.py", "line": 1, "issue_type": "bug", "canonical_desc": "z"}]}
    out = score_v5.score(preds, labels, "v4_pred",
                         match_fn=lambda r, d: True, count_fn=lambda r: 4)  # caught 1, claimed 4
    assert abs(out["defect_recall_labeled"] - 1.0) < 1e-9
    assert abs(out["precision_labeled"] - 0.25) < 1e-9   # 1/4


def test_missing_instance_treated_as_clean():
    preds = [{"instance_id": "i9", "diff": "@@\n+x=1\n", "v4_pred": "fine"}]
    out = score_v5.score(preds, {}, "v4_pred", match_fn=lambda r, d: True, count_fn=lambda r: 0)
    assert out["n_labeled"] == 0
    assert out["defect_recall_labeled"] is None
    assert out["fp_rate_clean"] == 0.0
