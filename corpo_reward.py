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
