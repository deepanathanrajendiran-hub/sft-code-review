"""Tests for PAIRED significance of v5-vs-v4 recall / fp / halluc deltas.

The goal ("v5 beats v4 on recall, no worse on halluc") is currently judged by
comparing two NOISY means (mean v4 recall vs mean v5 recall). A raw difference of
means has no error bar: with per-record recall in {0, 0.5, 1.0} (std ~0.4) and only
~150 labeled records, the standard error on the mean is ~0.033 — so a 0.093->0.070
"drop" can be pure per-record noise. Comparing two independent means is the wrong
test.

The correct test is PAIRED: on the SAME labeled records, bootstrap the per-record
delta (recall_b - recall_a). Pairing cancels per-record difficulty variance, giving
far more power. `recall_significant` is True only when the 95% bootstrap CI on the
delta excludes 0 on the improvement side (lower bound > 0).

No API: the semantic matcher (match_fn) and claim-counter (count_fn) are injected.
Encoding used by these tests:
  - a review is a whitespace-token string; a token equal to a defect's canonical_desc
    means that defect was CAUGHT; each "C" token is one CLAIM.
"""
import compare_recall as cmp

DEFECTS = [
    {"path": "a.py", "line": 1, "issue_type": "bug", "canonical_desc": "d1"},
    {"path": "a.py", "line": 2, "issue_type": "bug", "canonical_desc": "d2"},
]

# review-dependent matcher/counter so models A and B can genuinely differ by their text
MATCH = lambda review, defect: defect["canonical_desc"] in review.split()
COUNT = lambda review: review.split().count("C")


def _preds(rows, field):
    """rows: list of (instance_id, review_text)."""
    return [{"instance_id": iid, "diff": "x", field: txt} for iid, txt in rows]


def test_identical_models_zero_delta_not_significant():
    labels = {"r1": DEFECTS, "r2": DEFECTS}
    a = _preds([("r1", "d1 d2"), ("r2", "d1 d2")], "v4_pred")
    b = _preds([("r1", "d1 d2"), ("r2", "d1 d2")], "v5_pred")
    res = cmp.paired_delta(a, b, labels, "v4_pred", "v5_pred",
                           match_fn=MATCH, count_fn=COUNT, n_boot=500, seed=0)
    assert abs(res["recall_delta"]) < 1e-9
    assert res["recall_significant"] is False
    assert res["recall_ci"][0] <= 0.0 <= res["recall_ci"][1]


def test_strictly_better_b_is_significant():
    # b catches more on every record -> all per-record deltas > 0 -> CI lo > 0
    labels = {"r1": DEFECTS, "r2": DEFECTS}
    a = _preds([("r1", "d1"), ("r2", "")], "v4_pred")        # recall 0.5, 0.0 -> mean 0.25
    b = _preds([("r1", "d1 d2"), ("r2", "d1 d2")], "v5_pred")  # recall 1.0, 1.0 -> mean 1.0
    res = cmp.paired_delta(a, b, labels, "v4_pred", "v5_pred",
                           match_fn=MATCH, count_fn=COUNT, n_boot=500, seed=0)
    assert abs(res["recall_a"] - 0.25) < 1e-9
    assert abs(res["recall_b"] - 1.0) < 1e-9
    assert abs(res["recall_delta"] - 0.75) < 1e-9
    assert res["recall_significant"] is True
    assert res["recall_ci"][0] > 0.0


def test_worse_b_is_not_an_improvement():
    labels = {"r1": DEFECTS, "r2": DEFECTS}
    a = _preds([("r1", "d1 d2"), ("r2", "d1 d2")], "v4_pred")  # recall 1.0
    b = _preds([("r1", ""), ("r2", "")], "v5_pred")            # recall 0.0
    res = cmp.paired_delta(a, b, labels, "v4_pred", "v5_pred",
                           match_fn=MATCH, count_fn=COUNT, n_boot=500, seed=0)
    assert res["recall_delta"] < 0
    assert res["recall_significant"] is False   # not a significant improvement
    assert res["recall_ci"][1] < 0.0            # CI sits entirely below 0


def test_fewer_false_positives_is_significant():
    # clean records: b stops inventing defects -> fp delta negative and significant
    labels = {"c1": [], "c2": []}
    a = _preds([("c1", "C C"), ("c2", "C C")], "v4_pred")  # claims on both -> fp 1.0
    b = _preds([("c1", ""), ("c2", "")], "v5_pred")        # no claims -> fp 0.0
    res = cmp.paired_delta(a, b, labels, "v4_pred", "v5_pred",
                           match_fn=MATCH, count_fn=COUNT, n_boot=500, seed=0)
    assert abs(res["fp_a"] - 1.0) < 1e-9
    assert abs(res["fp_b"] - 0.0) < 1e-9
    assert res["fp_delta"] < 0
    assert res["fp_significant"] is True
    assert res["fp_ci"][1] < 0.0


def test_counts_labeled_and_clean_split():
    labels = {"r1": DEFECTS, "c1": [], "c2": []}
    a = _preds([("r1", "d1"), ("c1", ""), ("c2", "C")], "v4_pred")
    b = _preds([("r1", "d1 d2"), ("c1", ""), ("c2", "")], "v5_pred")
    res = cmp.paired_delta(a, b, labels, "v4_pred", "v5_pred",
                           match_fn=MATCH, count_fn=COUNT, n_boot=200, seed=0)
    assert res["n_compared"] == 3
    assert res["n_labeled"] == 1
    assert res["n_clean"] == 2


def test_only_compares_shared_instance_ids():
    labels = {"r1": DEFECTS, "r2": DEFECTS}
    a = _preds([("r1", "d1 d2"), ("r2", "d1 d2")], "v4_pred")
    b = _preds([("r1", "d1 d2")], "v5_pred")  # missing r2
    res = cmp.paired_delta(a, b, labels, "v4_pred", "v5_pred",
                           match_fn=MATCH, count_fn=COUNT, n_boot=200, seed=0)
    assert res["n_compared"] == 1   # only r1 is in both


def test_no_backtick_reviews_have_zero_halluc_delta():
    labels = {"r1": DEFECTS}
    a = _preds([("r1", "d1 d2")], "v4_pred")
    b = _preds([("r1", "d1 d2")], "v5_pred")
    res = cmp.paired_delta(a, b, labels, "v4_pred", "v5_pred",
                           match_fn=MATCH, count_fn=COUNT, n_boot=200, seed=0)
    assert abs(res["halluc_delta"]) < 1e-9


def test_bootstrap_is_reproducible_with_seed():
    labels = {"r1": DEFECTS, "r2": DEFECTS, "r3": DEFECTS}
    a = _preds([("r1", "d1"), ("r2", ""), ("r3", "d1 d2")], "v4_pred")
    b = _preds([("r1", "d1 d2"), ("r2", "d1"), ("r3", "d1 d2")], "v5_pred")
    r1 = cmp.paired_delta(a, b, labels, "v4_pred", "v5_pred",
                          match_fn=MATCH, count_fn=COUNT, n_boot=500, seed=7)
    r2 = cmp.paired_delta(a, b, labels, "v4_pred", "v5_pred",
                          match_fn=MATCH, count_fn=COUNT, n_boot=500, seed=7)
    assert r1["recall_ci"] == r2["recall_ci"]
