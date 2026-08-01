"""Tests for the v5 PRECISION-AWARE verifiable reward.

reward = 0.6*quality + 0.3*grounding + 0.1*length where
  - labeled record: quality = F1(recall, precision), precision = caught / claims
                    -> over-claiming (shotgun) LOWERS the score
  - clean record:   quality = 1/(1+claims)  -> any invented defect LOWERS the score

No LLM/API: the matcher (match_fn) and claim-counter (count_fn) are injected.
"""
import corpo_reward as cr


def _review(body: str) -> str:
    return f"<think>reasoning</think>\n<review>{body}</review>"


GROUNDED_BODY = ("The `user` handling looks correct and defensive here. " * 4)[:200]

DEFECTS = [
    {"path": "auth.py", "line": 11, "issue_type": "bug", "canonical_desc": "empty string not handled"},
    {"path": "auth.py", "line": 11, "issue_type": "edge", "canonical_desc": "None vs empty conflated"},
]


def _matcher(caught_descs):
    return lambda review, defect: defect["canonical_desc"] in caught_descs


def _claims(n):
    return lambda review: n


def test_empty_extraction_returns_zero(sample_diff):
    assert cr.verifiable_reward(sample_diff, "<review></review>", DEFECTS,
                                match_fn=_matcher(set()), count_fn=_claims(0)) == 0.0


def test_labeled_honest_catch_scores_high(sample_diff):
    # catches both defects, claims exactly 2 -> recall=1, precision=1, F1=1
    r = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), DEFECTS,
                             match_fn=_matcher({d["canonical_desc"] for d in DEFECTS}),
                             count_fn=_claims(2))
    assert abs(r - 1.0) < 1e-9  # 0.6*1 + 0.3*1(grounded) + 0.1*1(in-band)


def test_labeled_shotgun_scores_lower_than_honest(sample_diff):
    # SAME catches (both) but claims 6 -> precision=2/6=0.333. v5.2 uses F1.5 (beta=1.5):
    # still recall-favoring, but precision bites harder than v5.1's beta=2 so the
    # over-flagging v5.1 exhibited (11 fabrications vs 3 catches on disagreements) costs more.
    honest = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), DEFECTS,
                                  match_fn=_matcher({d["canonical_desc"] for d in DEFECTS}),
                                  count_fn=_claims(2))
    shotgun = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), DEFECTS,
                                   match_fn=_matcher({d["canonical_desc"] for d in DEFECTS}),
                                   count_fn=_claims(6))
    assert shotgun < honest
    # F1.5(recall=1, precision=2/6=0.333) = 3.25*0.333/(2.25*0.333+1) = 0.6190 -> 0.6*0.6190 + 0.3 + 0.1 = 0.7714
    assert abs(shotgun - 0.7714) < 1e-3


def test_labeled_miss_all_low(sample_diff):
    r = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), DEFECTS,
                             match_fn=_matcher(set()), count_fn=_claims(2))
    # recall=0, precision=0 -> F1=0 -> 0 + 0.3 + 0.1
    assert abs(r - 0.4) < 1e-9


def test_clean_no_claims_scores_high(sample_diff):
    # clean record, review asserts nothing -> quality = 1/(1+0) = 1.0
    r = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), [],
                             match_fn=_matcher(set()), count_fn=_claims(0))
    assert abs(r - 1.0) < 1e-9


def test_clean_invented_defects_punished(sample_diff):
    # clean record: linear penalty (1 - 0.35*claims) so 1 minor note isn't crushed, but
    # invented defects cost more than v5.1's 0.25 (suppresses over-flagging on clean diffs).
    clean_quiet = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), [],
                                       match_fn=_matcher(set()), count_fn=_claims(0))
    clean_shotgun = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), [],
                                         match_fn=_matcher(set()), count_fn=_claims(2))
    assert clean_shotgun < clean_quiet
    # quality = 1 - 0.35*2 = 0.30 -> 0.6*0.30 + 0.3*1 + 0.1*1 = 0.58
    assert abs(clean_shotgun - 0.58) < 1e-9


def test_reward_favors_recall_over_precision(sample_diff):
    # 1 defect, caught, but 3 claims -> recall=1.0, precision=1/3. F1.5 should reward this
    # MORE than plain F1 would (0.5), because recall is favored ~2.25x (still, less than v5.1's 4x).
    c = cr.verifiable_components(GROUNDED_BODY, DEFECTS[:1], sample_diff,
                                 match_fn=_matcher({DEFECTS[0]["canonical_desc"]}), count_fn=_claims(3))
    assert abs(c["recall"] - 1.0) < 1e-9
    assert abs(c["precision"] - (1 / 3)) < 1e-9
    # F1.5(1, 1/3) = 3.25*(1/3)/(2.25*(1/3)+1) = 0.6190  (> F1 of 0.5)
    assert abs(c["quality"] - 0.6190) < 1e-3


def test_components_exposed_for_eval(sample_diff):
    c = cr.verifiable_components(GROUNDED_BODY, DEFECTS, sample_diff,
                                 match_fn=_matcher({DEFECTS[0]["canonical_desc"]}), count_fn=_claims(2))
    assert abs(c["recall"] - 0.5) < 1e-9          # caught 1 of 2
    assert abs(c["precision"] - 0.5) < 1e-9       # 1 caught / 2 claims
    assert c["n_claims"] == 2
    assert 0.0 <= c["reward"] <= 1.0


def test_components_clean_reports_fp_rate(sample_diff):
    c = cr.verifiable_components(GROUNDED_BODY, [], sample_diff,
                                 match_fn=_matcher(set()), count_fn=_claims(3))
    assert c["recall"] is None
    assert c["fp_rate"] == 1.0   # clean record with >=1 asserted defect = false positive
    assert c["n_claims"] == 3


def test_reward_in_unit_interval(sample_diff):
    for defects, caught, n in [
        (DEFECTS, {DEFECTS[0]["canonical_desc"]}, 3),
        ([], set(), 0),
        ([], set(), 5),
        (DEFECTS, set(), 0),
    ]:
        r = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), defects,
                                 match_fn=_matcher(caught), count_fn=_claims(n))
        assert 0.0 <= r <= 1.0

def test_truncated_think_scores_zero(sample_diff):
    # max_completion_length can cut a rollout before </think>; the extractor's
    # fallback would otherwise pass raw reasoning text through as the "review"
    truncated = "<think>step 1: looking at the diff, the `user` handling"
    assert cr.verifiable_reward(sample_diff, truncated, DEFECTS,
                                match_fn=_matcher(set()), count_fn=_claims(0)) == 0.0


def test_closed_think_not_treated_as_truncated(sample_diff):
    # a normal rollout with a closed think block still scores via the review
    r = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), DEFECTS,
                             match_fn=_matcher({d["canonical_desc"] for d in DEFECTS}),
                             count_fn=_claims(2))
    assert r > 0.9
