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
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


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
