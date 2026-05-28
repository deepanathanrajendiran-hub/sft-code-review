"""Tests for defect_match: semantic 'was this defect caught' matcher + recall.

The LLM matcher is injected (match_fn) so the aggregation logic is tested
without any API call. The default _match_fn is the DeepSeek constrained
yes/no judge, exercised only in integration (not here).
"""
import defect_match as dm


def _fake_matcher(caught_descs):
    """Return a match_fn that reports 'caught' iff the defect's desc is in the set."""
    def match_fn(review, defect):
        return defect["canonical_desc"] in caught_descs
    return match_fn


DEFECTS = [
    {"path": "a.py", "line": 10, "issue_type": "bug", "canonical_desc": "off-by-one in loop"},
    {"path": "a.py", "line": 22, "issue_type": "security", "canonical_desc": "unvalidated path param"},
    {"path": "b.py", "line": 3, "issue_type": "resource_leak", "canonical_desc": "file not closed"},
]


def test_defect_caught_delegates_to_match_fn():
    review = "the loop is off by one"
    mf = _fake_matcher({"off-by-one in loop"})
    assert dm.defect_caught(review, DEFECTS[0], match_fn=mf) is True
    assert dm.defect_caught(review, DEFECTS[1], match_fn=mf) is False


def test_recall_fraction_caught():
    # review catches 2 of 3
    mf = _fake_matcher({"off-by-one in loop", "file not closed"})
    assert dm.recall("...", DEFECTS, match_fn=mf) == 2 / 3


def test_recall_all_caught_is_one():
    mf = _fake_matcher({d["canonical_desc"] for d in DEFECTS})
    assert dm.recall("...", DEFECTS, match_fn=mf) == 1.0


def test_recall_none_caught_is_zero():
    mf = _fake_matcher(set())
    assert dm.recall("...", DEFECTS, match_fn=mf) == 0.0


def test_recall_empty_defects_raises():
    # recall over an empty label set is undefined; the reward layer must special-case it,
    # so recall() must refuse rather than silently return 1.0 (which would reward verbosity).
    import pytest
    with pytest.raises(ValueError):
        dm.recall("anything", [], match_fn=_fake_matcher(set()))


def test_recall_does_not_call_matcher_when_review_empty():
    # an empty review catches nothing; must not waste an API call per defect
    calls = []
    def counting_mf(review, defect):
        calls.append(defect)
        return True
    assert dm.recall("", DEFECTS, match_fn=counting_mf) == 0.0
    assert calls == []
