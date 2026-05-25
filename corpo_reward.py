"""Composite reward function for CoRPO training.

Reward = 0.6 * pairwise_score + 0.3 * hallucination_score + 0.1 * length_sanity_score
All components return values in [0.0, 1.0]. Composite is also [0.0, 1.0].

This module is built incrementally:
  Task 7  — length_sanity_score (this task)
  Task 8  — hallucination_score
  Task 9  — pairwise_score
  Task 10 — composite_reward (combines 7+8+9)
  Task 11 — BaseSampleCache + --build-base-cache CLI
"""
from __future__ import annotations
from ood_metrics import hallucination_rate as _ood_hallucination_rate
from ood_metrics import deepseek_v4flash_pairwise_judge


def length_sanity_score(text: str) -> float:
    """Linear-taper reward for output length.

    Returns 1.0 if 200 <= len(text) <= 4000 (the typical v4 review band).
    Linear taper to 0.0 outside on both sides:
      - below: linear from 0.0 at len=0 to 1.0 at len=200
      - above: linear from 1.0 at len=4000 to 0.0 at len=8000
    """
    n = len(text)
    if 200 <= n <= 4000:
        return 1.0
    if n < 200:
        return max(0.0, n / 200)
    # n > 4000
    return max(0.0, 1.0 - (n - 4000) / 4000)

def hallucination_score(diff: str, review_text: str) -> float:
    """Score in [0, 1] = 1.0 - hallucination_rate.

    Wraps ood_metrics.hallucination_rate which counts backticked identifiers
    in the review that don't appear in the diff (and aren't on the STOPWORDS
    allow-list). Higher score = fewer hallucinations.

    Returns 1.0 if no backticked identifiers to check (or all are grounded).
    """
    # ood_metrics.hallucination_rate takes a dict with 'diff' and 'v4_pred'
    pred = {"diff": diff, "v4_pred": review_text}
    rate = _ood_hallucination_rate(pred)
    return max(0.0, 1.0 - rate)


# Module-level binding so tests can patch _judge_fn
# (Production: V4-Flash binary judge; tests: MagicMock)
_judge_fn = deepseek_v4flash_pairwise_judge


def pairwise_score(
    diff: str,
    rollout: str,
    base_sample: str,
    reference: str,
) -> float:
    """Binary pairwise reward via DeepSeek V4-Flash judge.

    Returns:
        1.0 if rollout beats base_sample (judge says A)
        0.0 if rollout loses (judge says B)
        0.5 if TIE
    """
    verdict = _judge_fn(rollout, base_sample, diff, reference)
    if verdict == "A":
        return 1.0
    if verdict == "B":
        return 0.0
    return 0.5  # TIE

PAIRWISE_WEIGHT = 0.6
HALLUC_WEIGHT = 0.3
LENGTH_WEIGHT = 0.1


def composite_reward(
    diff: str,
    rollout: str,
    base_sample: str,
    reference: str,
) -> float:
    """Composite reward in [0, 1] for CoRPO training.

    R = PAIRWISE_WEIGHT * pairwise_score
      + HALLUC_WEIGHT   * hallucination_score
      + LENGTH_WEIGHT   * length_sanity_score

    With current weights (0.6, 0.3, 0.1), R = 1.0 iff:
      - rollout beats base in pairwise (pairwise = 1.0)
      - no hallucinated identifiers (halluc = 1.0)
      - length in [200, 4000] (length = 1.0)

    R_min_correct = 0.5 in the CoRPO trainer config: a rollout that wins
    pairwise (0.6) clears the bar even if length/halluc are zero, but a
    rollout that only has clean output without winning pairwise (0.3+0.1=0.4)
    does not.
    """
    r_pair = pairwise_score(diff, rollout, base_sample, reference)
    r_halluc = hallucination_score(diff, rollout)
    r_length = length_sanity_score(rollout)
    return (
        PAIRWISE_WEIGHT * r_pair
        + HALLUC_WEIGHT * r_halluc
        + LENGTH_WEIGHT * r_length
    )
