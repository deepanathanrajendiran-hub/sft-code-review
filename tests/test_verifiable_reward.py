"""Tests for the v5 verifiable reward: 0.6*recall + 0.3*grounding + 0.1*length.

No LLM/API: the semantic matcher is injected. Grounding (anti-hallucination)
and length reuse the existing pure-Python scorers in corpo_reward.
"""
import corpo_reward as cr


def _review(body: str) -> str:
    """Wrap a body in a <review> block (what the policy emits)."""
    return f"<think>reasoning here</think>\n<review>{body}</review>"


# ~200-char bodies. Only `user` is backticked and `user` IS in sample_diff -> grounded.
GROUNDED_BODY = (
    "The change to `user` handling looks correct and defensive. " * 4
)[:200]
# a backticked identifier NOT in the diff and not a stopword -> hallucination
HALLUCINATED_BODY = (
    "This will break because `frobnicate_widget` is never initialized here. " * 3
)[:200]

DEFECTS = [
    {"path": "auth.py", "line": 11, "issue_type": "bug", "canonical_desc": "empty string not handled"},
    {"path": "auth.py", "line": 11, "issue_type": "edge", "canonical_desc": "None vs empty conflated"},
]


def _matcher(caught_descs):
    def mf(review, defect):
        return defect["canonical_desc"] in caught_descs
    return mf


def test_empty_extraction_returns_zero(sample_diff):
    assert cr.verifiable_reward(sample_diff, "<review></review>", DEFECTS, match_fn=_matcher(set())) == 0.0


def test_placeholder_extraction_returns_zero(sample_diff):
    assert cr.verifiable_reward(sample_diff, "<review>...</review>", DEFECTS, match_fn=_matcher(set())) == 0.0


def test_all_defects_caught_grounded_inband_scores_near_one(sample_diff):
    mf = _matcher({d["canonical_desc"] for d in DEFECTS})
    r = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), DEFECTS, match_fn=mf)
    # recall=1.0, grounding=1.0, length=1.0 (200 chars in [150,1000]) -> 0.6+0.3+0.1
    assert abs(r - 1.0) < 1e-9


def test_missed_all_defects_drops_recall_term(sample_diff):
    mf = _matcher(set())  # catches nothing
    r = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), DEFECTS, match_fn=mf)
    # recall=0, grounding=1, length=1 -> 0.0 + 0.3 + 0.1
    assert abs(r - 0.4) < 1e-9


def test_half_recall(sample_diff):
    mf = _matcher({DEFECTS[0]["canonical_desc"]})  # 1 of 2
    r = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), DEFECTS, match_fn=mf)
    # recall=0.5 -> 0.6*0.5 + 0.3 + 0.1 = 0.7
    assert abs(r - 0.7) < 1e-9


def test_clean_record_grounded_review_rewards_restraint(sample_diff):
    # no labeled defects -> recall_or_restraint == grounding; a grounded review should score high
    r = cr.verifiable_reward(sample_diff, _review(GROUNDED_BODY), [], match_fn=_matcher(set()))
    # restraint=grounding=1, grounding=1, length=1 -> 0.6 + 0.3 + 0.1
    assert abs(r - 1.0) < 1e-9


def test_clean_record_hallucination_is_punished(sample_diff):
    # no labeled defects, review invents `frobnicate_widget` -> grounding low -> reward low
    r = cr.verifiable_reward(sample_diff, _review(HALLUCINATED_BODY), [], match_fn=_matcher(set()))
    # grounding=0 -> restraint=0 -> 0.6*0 + 0.3*0 + 0.1*length(=1.0) = 0.1
    assert abs(r - 0.1) < 1e-9


def test_reward_in_unit_interval(sample_diff):
    for body, defects, caught in [
        (GROUNDED_BODY, DEFECTS, {DEFECTS[0]["canonical_desc"]}),
        (HALLUCINATED_BODY, [], set()),
        (GROUNDED_BODY, [], set()),
    ]:
        r = cr.verifiable_reward(sample_diff, _review(body), defects, match_fn=_matcher(caught))
        assert 0.0 <= r <= 1.0


def test_recall_or_restraint_uses_recall_when_labels_present(sample_diff):
    mf = _matcher({DEFECTS[0]["canonical_desc"]})
    val = cr.recall_or_restraint(GROUNDED_BODY, DEFECTS, sample_diff, match_fn=mf)
    assert abs(val - 0.5) < 1e-9


def test_recall_or_restraint_uses_grounding_when_no_labels(sample_diff):
    val = cr.recall_or_restraint(GROUNDED_BODY, [], sample_diff, match_fn=_matcher(set()))
    assert abs(val - 1.0) < 1e-9
