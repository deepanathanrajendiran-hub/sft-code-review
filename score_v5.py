"""Judge-independent v5 scorer: defect-recall + hallucination on clean labels.

This is BOTH the success metric for the goal ("beat v4 with <= v4 hallucination")
and the RL variance pre-flight precursor — it answers "does v4 have recall headroom,
and is v5 actually better?" without any gameable quality judge.

  recall_mean : mean recall_or_restraint over predictions (defect recall on labeled
                records via the semantic matcher; grounding-as-restraint on clean ones)
  halluc_mean : mean hallucination_rate vs the diff (the "less hallucination" half)

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

from corpo_reward import recall_or_restraint
from ood_metrics import hallucination_rate


def score(preds: list[dict], labels: dict[str, list[dict]], pred_field: str, match_fn=None) -> dict:
    """Aggregate recall + hallucination for one prediction field against clean labels.

    preds: rows with instance_id, diff, and the pred_field (an extracted review string).
    labels: instance_id -> list of clean defect tuples (missing => clean diff).
    match_fn: semantic matcher (defect_match._match_fn default; inject in tests).
    """
    recalls, hallucs = [], []
    n_labeled = 0
    for r in preds:
        defects = labels.get(r["instance_id"], [])
        review = r.get(pred_field, "") or ""
        diff = r.get("diff", "")
        recalls.append(recall_or_restraint(review, defects, diff, match_fn=match_fn))
        hallucs.append(hallucination_rate({"diff": diff, "v4_pred": review}))
        if defects:
            n_labeled += 1
    return {
        "pred_field": pred_field,
        "n_total": len(preds),
        "n_labeled": n_labeled,
        "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
        "halluc_mean": float(np.mean(hallucs)) if hallucs else 0.0,
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
    for field in args.pred_fields:
        s = score(preds, labels, field)
        print(
            f"  {field:12s}  recall={s['recall_mean']:.3f}  halluc={s['halluc_mean']:.3f}  "
            f"(labeled {s['n_labeled']}/{s['n_total']})"
        )


if __name__ == "__main__":
    main()
