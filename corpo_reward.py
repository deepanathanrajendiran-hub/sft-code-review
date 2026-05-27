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
from ood_metrics import deepseek_v4pro_pairwise_judge
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


def length_sanity_score(text: str) -> float:
    """Reward outputs in the v4 OOD extracted-review distribution.

    Calibration:
      - v4-deployed extracted-review median ~600 chars (per 2026-05-22 journal)
      - Plateau [150, 1000] covers terse-to-thorough reviews
      - Linear taper to 0 at 0 chars (collapse) and 2500 chars (explosion)

    The band is anchored to the *extracted review* string (post-_extract_review),
    NOT the raw rollout. This matches what composite_reward now scores and what
    production deployment evaluates.

    Shape:
      - 0 chars     -> 0.0
      - 75 chars    -> 0.50
      - 150-1000    -> 1.0    (v4 OOD band)
      - 1750 chars  -> 0.50
      - 2500+ chars -> 0.0
    """
    n = len(text)
    if 150 <= n <= 1000:
        return 1.0
    if n < 150:
        return n / 150
    return max(0.0, 1.0 - (n - 1000) / 1500)

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
# (Production: V4-Pro binary judge — V4-Flash was too conservative, calling TIE on >95%
# of v4-vs-base comparisons during the variance gate. V4-Pro discriminates better at
# ~3× cost which is still negligible.)
_judge_fn = deepseek_v4pro_pairwise_judge


def pairwise_score(
    diff: str,
    rollout: str,
    base_sample: str,
    reference: str,
) -> float:
    """Binary pairwise reward via DeepSeek V4-Pro judge.

    Returns:
        1.0 if rollout beats base_sample (judge says A)
        0.0 if rollout loses (judge says B)
        0.5 if TIE

    Switched from V4-Flash on 2026-05-25 — variance gate showed V4-Flash
    called TIE on >95% of v4-vs-base comparisons (std=0.054 vs threshold 0.10).
    V4-Pro is more decisive at ~$12/epoch (was ~$3.60 for V4-Flash).
    """
    verdict = _judge_fn(rollout, base_sample, diff, reference)
    if verdict == "A":
        return 1.0
    if verdict == "B":
        return 0.0
    return 0.5  # TIE

# Run #2 weight rebalance (2026-05-26 after run #1 -22% lift):
#   - Bumped pairwise from 0.6 → 0.7. Pairwise vs v4 IS the production metric;
#     the run #1 reward had pairwise contribute only 0.6 of total, so winning
#     pairwise was worth ~0.6 reward while gaming length+halluc was ~0.4.
#     Model rationally chose the easier path. Boosting pairwise weight makes
#     it the dominant signal.
#   - Reduced halluc from 0.3 → 0.2. Hallucination is a guardrail, not a
#     primary objective. 0.3 weight gave it too much pull, and the cheapest
#     way to keep halluc-score high is to be terse — contributing to length
#     collapse.
#   - Kept length at 0.1 (small disincentive against runaway/collapsed length).
PAIRWISE_WEIGHT = 0.7
HALLUC_WEIGHT = 0.2
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
    # NOTE: length is scored on the full rollout (think + review), NOT on extracted review.
    # The 200-4000 char band reflects typical review-only sizing; well-formed v4-style
    # outputs with think blocks may exceed it. If reward is consistently capped at
    # zero from length, either bump the upper bound here or apply _extract_review first.
    r_length = length_sanity_score(rollout)
    return (
        PAIRWISE_WEIGHT * r_pair
        + HALLUC_WEIGHT * r_halluc
        + LENGTH_WEIGHT * r_length
    )

@dataclass
class BaseSampleCache:
    """In-memory base-sample lookup (instance_id -> base_output)."""
    _data: dict[str, str]

    def get(self, instance_id: str) -> str:
        if instance_id not in self._data:
            raise KeyError(f"no cached base sample for {instance_id!r}")
        return self._data[instance_id]

def load_base_sample_cache(path: Path | str) -> BaseSampleCache:
    """Load base-sample cache from JSONL (one {instance_id, base_output} per line)."""
    data: dict[str, str] = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            data[row["instance_id"]] = row["base_output"]
    return BaseSampleCache(_data=data)

def _build_base_cache_cli():
    """CLI: generate one base-model rollout per training prompt, save to disk.

    Called via:
        python corpo_reward.py --build-base-cache \\
            --input train_prompts.jsonl \\
            --output cache/base_samples.jsonl \\
            --base-model unsloth/Qwen2.5-Coder-7B-Instruct

    Uses the same generation params as run_ood_eval.py (temp=0, max_tokens=4096,
    rep_pen=1.1) for protocol parity with the production v4 inference.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-base-cache", action="store_true", required=True)
    ap.add_argument("--input", required=True, help="train_prompts.jsonl from swecare_split")
    ap.add_argument("--output", required=True, help="cache/base_samples.jsonl")
    ap.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct")
    args = ap.parse_args()

    # Reuse run_ood_eval._generate exactly — same chat template, same sampling.
    from run_ood_eval import _generate
    from transformers import AutoTokenizer

    rows = []
    with Path(args.input).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    diffs = [r["diff"] for r in rows]
    print(f"[corpo_reward] generating {len(diffs)} base samples", file=sys.stderr)
    base_outputs = _generate(args.base_model, diffs, tokenizer)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row, out in zip(rows, base_outputs):
            fh.write(json.dumps({"instance_id": row["instance_id"], "base_output": out}) + "\n")
    print(f"[corpo_reward] wrote {len(rows)} base samples to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    _build_base_cache_cli()
