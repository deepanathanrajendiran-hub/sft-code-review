"""Semantic 'was this defect caught' matcher + recall, for the v5 verifiable reward.

The recall reward (v5) replaces the gameable pairwise-LLM-judge of Runs #1-#3.
A clean defect tuple is {path, line, issue_type, canonical_desc} extracted from the
human PR review thread by label_defects.py. `recall` asks, per defect, whether the
model's review *identifies that specific defect* — a constrained, grounded yes/no
judgment, far less gameable than 'which review is better'.

The matcher (`match_fn`) is injectable so the aggregation logic is unit-tested with
no API call. The default `_match_fn` is the DeepSeek constrained yes/no judge.
"""
from __future__ import annotations


def defect_caught(review: str, defect: dict, match_fn=None) -> bool:
    """True iff `review` identifies `defect` (per the matcher)."""
    fn = match_fn or _match_fn
    return bool(fn(review, defect))


def recall(review: str, defects: list[dict], match_fn=None) -> float:
    """Fraction of clean defect tuples the review catches.

    Raises ValueError on an empty label set — recall over no labels is undefined,
    and silently returning 1.0 would reward verbosity on clean diffs. The reward
    layer (corpo_reward.recall_or_restraint) special-cases the no-label case.
    An empty/whitespace review catches nothing and short-circuits with NO matcher
    calls (one API call per defect is the cost driver).
    """
    if not defects:
        raise ValueError("recall() over an empty defect set is undefined")
    if not review or not review.strip():
        return 0.0
    fn = match_fn or _match_fn
    caught = sum(1 for d in defects if fn(review, d))
    return caught / len(defects)


def deepseek_defect_match_judge(review: str, defect: dict) -> bool:
    from ood_metrics import _get_deepseek_client, DEEPSEEK_V4_PRO

    client = _get_deepseek_client()
    prompt = (
        "You are checking whether a code review caught a SPECIFIC known defect.\n\n"
        f"KNOWN DEFECT:\n"
        f"  file: {defect.get('path')}\n"
        f"  line: {defect.get('line')}\n"
        f"  type: {defect.get('issue_type')}\n"
        f"  description: {defect.get('canonical_desc')}\n\n"
        f"CODE REVIEW:\n{review[:1500]}\n\n"
        "Does the review identify THIS defect (same underlying issue, location need not be exact)? "
        "Answer with exactly one word: YES or NO."
    )
    # Mirror ood_metrics' DeepSeek convention: V4-Pro, thinking disabled (~3x faster).
    resp = client.chat.completions.create(
        model=DEEPSEEK_V4_PRO,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4,
        extra_body={"thinking": {"type": "disabled"}},
    )
    answer = (resp.choices[0].message.content or "").strip().upper()
    return answer.startswith("Y")


# Module-level binding so tests can patch _match_fn (mirrors corpo_reward._judge_fn).
_match_fn = deepseek_defect_match_judge
