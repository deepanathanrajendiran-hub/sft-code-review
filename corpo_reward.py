"""Reward functions for CoRPO training.

Two reward paths live here:

  - composite_reward: the original pairwise-judge reward (0.7 pairwise + 0.2
    hallucination + 0.1 length). Gameable, kept for reference.
  - verifiable_reward: the v5 path actually used in training. No judge, no
    opponent — quality comes from recall/precision against labeled defects
    (or restraint on clean records), so Goodhart can't apply.

Both score the EXTRACTED REVIEW (post-_extract_review), not the raw rollout —
that's what production deploys. An empty or placeholder extraction scores 0.0.
"""
from __future__ import annotations
from ood_metrics import hallucination_rate as _ood_hallucination_rate
from ood_metrics import deepseek_v4pro_pairwise_judge
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from run_ood_eval import _extract_review
from defect_match import caught_count, count_claims


def length_sanity_score(text: str) -> float:
    """Band reward for review length, anchored on the v4 OOD distribution.

    Full credit in [150, 1000] chars (v4-deployed median ~600). Linear taper to
    0 at 0 chars (collapse) and 2500 chars (explosion). Measured against the
    extracted review, not the raw rollout.
    """
    n = len(text)
    if 150 <= n <= 1000:
        return 1.0
    if n < 150:
        return n / 150
    return max(0.0, 1.0 - (n - 1000) / 1500)

def hallucination_score(diff: str, review_text: str) -> float:
    """1.0 - hallucination_rate; higher means fewer ungrounded identifiers.

    Counts backticked identifiers in the review that don't appear in the diff
    and aren't on the STOPWORDS allow-list. Returns 1.0 when there's nothing to
    flag.
    """
    pred = {"diff": diff, "v4_pred": review_text}
    rate = _ood_hallucination_rate(pred)
    return max(0.0, 1.0 - rate)


# module-level so tests can patch it. V4-Pro, not Flash: Flash called TIE on
# >95% of v4-vs-base comparisons, which kills the variance gate.
_judge_fn = deepseek_v4pro_pairwise_judge


def pairwise_score(
    diff: str,
    rollout: str,
    base_sample: str,
    reference: str,
) -> float:
    """Binary pairwise reward: 1.0 rollout wins, 0.0 loses, 0.5 tie."""
    verdict = _judge_fn(rollout, base_sample, diff, reference)
    if verdict == "A":
        return 1.0
    if verdict == "B":
        return 0.0
    return 0.5

PAIRWISE_WEIGHT = 0.7
HALLUC_WEIGHT = 0.2
LENGTH_WEIGHT = 0.1


def composite_reward(
    diff: str,
    rollout: str,
    base_sample: str,
    reference: str,
) -> float:
    """Weighted pairwise + hallucination + length reward, in [0, 1].

    Scores the extracted review. An empty/placeholder extraction returns 0.0 —
    a failed extractor is a failed deployment.
    """
    review = _extract_review(rollout)
    # a "..." placeholder is a deployment failure, treat it as no review
    if not review or review.strip() in ("", "..."):
        return 0.0

    r_pair = pairwise_score(diff, review, base_sample, reference)
    r_halluc = hallucination_score(diff, review)
    r_length = length_sanity_score(review)
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
    """Load the cache from JSONL (one {instance_id, base_output} per line)."""
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
    """CLI: generate one base-model rollout per training prompt and save it.

    Uses run_ood_eval's generation params (temp=0, max_tokens=4096, rep_pen=1.1)
    so the cached base samples match production v4 inference exactly.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-base-cache", action="store_true", required=True)
    ap.add_argument("--input", required=True, help="train_prompts.jsonl from swecare_split")
    ap.add_argument("--output", required=True, help="cache/base_samples.jsonl")
    ap.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct")
    args = ap.parse_args()

    # reuse run_ood_eval._generate so the chat template and sampling stay in sync
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

# v5 verifiable reward: 0.6*quality + 0.3*grounding + 0.1*length.
# quality is F-beta(recall, precision) on labeled records (over-claiming lowers
# precision -> lowers score) and a restraint term on clean records (any invented
# defect lowers it). No judge/opponent means there's nothing to game; the
# precision term is what pushes recall up and hallucination down.
RECALL_WEIGHT = 0.6
GROUND_WEIGHT = 0.3
LEN_WEIGHT = 0.1

# beta>1 favors recall; 1.5 weights recall ~2.25x precision. beta=2 over-flagged
# (fabricated bugs were reward-positive), so it's dialed back. The clean penalty
# is the cost of asserting an unsupported defect on a clean record.
RECALL_BETA = 1.5
CLEAN_CLAIM_PENALTY = 0.35  # clean-record quality = max(0, 1 - 0.35*claims)


def _fbeta(recall: float, precision: float, beta: float = 1.0) -> float:
    """F-beta of recall & precision; beta>1 favors recall (beta^2 weight)."""
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def verifiable_components(
    review: str, defects: list[dict], diff: str, match_fn=None, count_fn=None
) -> dict:
    n_claims = count_claims(review, count_fn=count_fn)
    grounding = hallucination_score(diff, review)
    length = length_sanity_score(review)
    if defects:
        caught = caught_count(review, defects, match_fn=match_fn)
        recall = caught / len(defects)
        precision = min(1.0, caught / n_claims) if n_claims > 0 else 0.0
        quality = _fbeta(recall, precision, RECALL_BETA)
        fp_rate = None
    else:
        caught = None
        recall = None
        precision = None
        quality = max(0.0, 1.0 - CLEAN_CLAIM_PENALTY * n_claims)
        fp_rate = 1.0 if n_claims > 0 else 0.0
    reward = RECALL_WEIGHT * quality + GROUND_WEIGHT * grounding + LEN_WEIGHT * length
    return {
        "n_claims": n_claims, "caught": caught, "recall": recall, "precision": precision,
        "quality": quality, "grounding": grounding, "length": length,
        "fp_rate": fp_rate, "reward": reward,
    }


def verifiable_reward(diff: str, rollout: str, defects: list[dict], match_fn=None, count_fn=None) -> float:
    # A rollout that opens <think> but never closes it was truncated by
    # max_completion_length. _extract_review's fallback would pass the raw
    # reasoning text through as the "review" and score garbage; treat truncation
    # as a hard failure (0.0) so the policy learns to finish within budget.
    if "<think>" in rollout and "</think>" not in rollout:
        return 0.0
    review = _extract_review(rollout)
    # _extract_review's fallback yields the literal "<review></review>" for an
    # empty block — strip the tags so that lands as no review (score 0)
    review = review.replace("<review>", "").replace("</review>", "").strip()
    if not review or review == "...":
        return 0.0
    return verifiable_components(
        review, defects, diff, match_fn=match_fn, count_fn=count_fn
    )["reward"]
