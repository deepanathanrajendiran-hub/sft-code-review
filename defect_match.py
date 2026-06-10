"""Semantic "was this defect caught" matcher and recall for the verifiable reward.

A clean defect tuple is {path, line, issue_type, canonical_desc} extracted from a
human PR review thread. `recall` asks, per defect, whether the model's review
identifies that specific defect — a constrained, grounded yes/no judgment rather
than the looser "which review is better".

The matcher (`match_fn`) is injectable so the aggregation logic can be unit-tested
with no API call. The default `_match_fn` is the DeepSeek constrained yes/no judge.
"""
from __future__ import annotations


def defect_caught(review: str, defect: dict, match_fn=None) -> bool:
    """True iff `review` identifies `defect` (per the matcher)."""
    fn = match_fn or _match_fn
    return bool(fn(review, defect))


def caught_count(review: str, defects: list[dict], match_fn=None) -> int:
    """Number of known defect tuples the review catches.

    An empty/whitespace review catches nothing and makes no matcher calls.
    """
    if not defects or not review or not review.strip():
        return 0
    fn = match_fn or _match_fn
    return sum(1 for d in defects if fn(review, d))


def count_claims(review: str, count_fn=None) -> int:
    """Number of distinct defects the review asserts, for the precision term.

    Used to penalize over-claiming: precision = caught / claims. An empty review
    asserts nothing and makes no API call. count_fn is injectable for tests.
    """
    if not review or not review.strip():
        return 0
    fn = count_fn or _count_fn
    return int(fn(review))


def recall(review: str, defects: list[dict], match_fn=None) -> float:
    if not defects:
        raise ValueError("recall() over an empty defect set is undefined")
    return caught_count(review, defects, match_fn=match_fn) / len(defects)


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
    # match ood_metrics' convention: V4-Pro with thinking disabled (~3x faster).
    resp = _with_retries(lambda: client.chat.completions.create(
        model=DEEPSEEK_V4_PRO,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4,
        extra_body={"thinking": {"type": "disabled"}},
    ))
    answer = (resp.choices[0].message.content or "").strip().upper()
    return answer.startswith("Y")


# module-level binding so tests can patch the matcher (mirrors corpo_reward._judge_fn)
_match_fn = deepseek_defect_match_judge

def deepseek_count_claims(review: str) -> int:
    """Count distinct asserted code defects via DeepSeek V4-Pro (thinking disabled)."""
    from ood_metrics import _get_deepseek_client, DEEPSEEK_V4_PRO

    client = _get_deepseek_client()
    prompt = (
        "Count the number of DISTINCT, specific code defects this review asserts "
        "(a bug/security/correctness/perf issue the author should fix). Do NOT count "
        "praise, questions, or general remarks. Reply with ONLY an integer.\n\n"
        f"REVIEW:\n{review[:1500]}"
    )
    resp = _with_retries(lambda: client.chat.completions.create(
        model=DEEPSEEK_V4_PRO,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4,
        extra_body={"thinking": {"type": "disabled"}},
    ))
    import re as _re
    m = _re.search(r"\d+", resp.choices[0].message.content or "")
    return int(m.group()) if m else 0


# module-level binding so tests can patch the claim-counter (mirrors _match_fn)
_count_fn = deepseek_count_claims

def _with_retries(call, attempts: int = 5, base_delay: float = 2.0):
    """Run an API call with exponential backoff (2s, 4s, 8s, 16s between tries).

    The reward path fires thousands of matcher/counter calls from inside the
    training loop; before this, one transient DeepSeek error propagated through
    the ThreadPoolExecutor and killed a multi-hour run. Persistent failure
    (all attempts exhausted) still raises — a silent default would corrupt
    rewards instead.
    """
    import time

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 — provider SDK errors vary
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc
