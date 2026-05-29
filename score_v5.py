"""Judge-independent v5 scorer: defect-recall + hallucination on clean labels.

This is BOTH the success metric for the goal ("beat v4 with <= v4 hallucination")
and the RL variance pre-flight precursor — it answers "does v4 have recall headroom,
and is v5 actually better?" without any gameable quality judge.

  defect_recall_labeled : caught/known on records WITH defects (the "beat v4" half)
  precision_labeled     : caught/claims on those records (anti-shotgun)
  fp_rate_clean         : fraction of CLEAN records where the model invented a defect
  halluc_mean           : backticked-identifier hallucination (the "less hallucination" half)

CLI:
    # 1) build clean labels (needs DEEPSEEK_API_KEY)
    python label_defects.py --input ood_preds_v4.jsonl --output cache/defect_labels.jsonl
    # 2) score v4 and base against them (needs DEEPSEEK_API_KEY for the matcher)
    python score_v5.py --preds ood_preds_v4.jsonl --labels cache/defect_labels.jsonl \
        --pred-fields v4_pred base_pred
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from corpo_reward import verifiable_components


def score(preds: list[dict], labels: dict[str, list[dict]], pred_field: str,
          match_fn=None, count_fn=None) -> dict:
    """Precision-aware judge-independent scoring against clean defect labels.

    Returns the goal-relevant signals, separated (never a single gameable blend):
      defect_recall_labeled : caught/known on records WITH defects (the "beat v4" half)
      precision_labeled     : caught/claims on those records (anti-shotgun)
      fp_rate_clean         : fraction of CLEAN records where the model invented a defect
      halluc_mean           : backticked-identifier hallucination over all records
      reward_mean           : the composite v5 reward (what training optimizes)
    """
    recalls, precisions, fps, hallucs, rewards = [], [], [], [], []
    for r in preds:
        defects = labels.get(r["instance_id"], [])
        review = r.get(pred_field, "") or ""
        diff = r.get("diff", "")
        c = verifiable_components(review, defects, diff, match_fn=match_fn, count_fn=count_fn)
        rewards.append(c["reward"])
        hallucs.append(1.0 - c["grounding"])
        if defects:
            recalls.append(c["recall"])
            precisions.append(c["precision"])
        else:
            fps.append(c["fp_rate"])

    def _mean(xs):
        return float(np.mean(xs)) if xs else None

    return {
        "pred_field": pred_field,
        "n_total": len(preds),
        "n_labeled": len(recalls),
        "defect_recall_labeled": _mean(recalls),
        "precision_labeled": _mean(precisions),
        "fp_rate_clean": _mean(fps),
        "halluc_mean": float(np.mean(hallucs)) if hallucs else 0.0,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="JSONL with instance_id, diff, and pred fields")
    ap.add_argument("--labels", required=True, help="defect_labels.jsonl from label_defects.py")
    ap.add_argument("--pred-fields", nargs="+", default=["v4_pred", "base_pred"])
    args = ap.parse_args()

    preds = [json.loads(l) for l in Path(args.preds).open() if l.strip()]
    labels = {}
    for l in Path(args.labels).open():
        if l.strip():
            row = json.loads(l)
            labels[row["instance_id"]] = row.get("defects", [])

    print(f"[score_v5] {len(preds)} preds; labels for {len(labels)} instances", file=sys.stderr)

    def _f(x):
        return f"{x:.3f}" if x is not None else "n/a"

    for field in args.pred_fields:
        s = score(preds, labels, field)
        print(
            f"  {field:12s}  defect_recall={_f(s['defect_recall_labeled'])}  "
            f"precision={_f(s['precision_labeled'])}  fp_rate(clean)={_f(s['fp_rate_clean'])}  "
            f"halluc={s['halluc_mean']:.3f}  reward={s['reward_mean']:.3f}  "
            f"(labeled {s['n_labeled']}/{s['n_total']})"
        )
    print("\nGOAL: v5 beats v4 iff defect_recall UP and (fp_rate, halluc) NOT worse than v4.")


if __name__ == "__main__":
    main()
