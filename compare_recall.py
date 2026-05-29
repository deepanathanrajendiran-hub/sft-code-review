"""Paired bootstrap significance for v5-vs-v4 recall / fp / hallucination.

WHY THIS EXISTS
---------------
score_v5.py reports the MEAN recall of each model independently. The goal then
asks "is v5 recall > v4 recall?" — but comparing two independent noisy means has
no error bar. Per-record recall is ~{0, 0.5, 1.0} (std ~0.4); with ~150 labeled
records the standard error on the mean is ~0.033, so a 0.093 -> 0.070 swing can be
pure per-record noise. The v5.0 "recall regressed" verdict (and any similarly-sized
v5.1 "win") is not trustworthy from two means alone.

This module does the correct PAIRED test: on the SAME records scored by BOTH models,
take the per-record delta (b - a) and bootstrap its mean. Pairing cancels per-record
difficulty variance, so it detects far smaller real differences than an unpaired
comparison. A difference is called significant only when the 95% bootstrap CI on the
delta excludes 0 on the improvement side.

Reuses corpo_reward.verifiable_components for per-record (caught / recall / fp_rate /
grounding) so semantics are IDENTICAL to score_v5 and the training reward. match_fn /
count_fn are injectable (DeepSeek matcher in production, fakes in tests).

CLI:
    python compare_recall.py \\
        --preds-a ood_preds_v4.jsonl --field-a v4_pred \\
        --preds-b v5_preds.jsonl     --field-b corpo_pred \\
        --labels  cache/defect_labels.jsonl
(Both files may be the same path with two different fields.)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from corpo_reward import verifiable_components


def _mean(xs):
    return float(np.mean(xs)) if xs else None


def paired_delta(
    preds_a: list[dict],
    preds_b: list[dict],
    labels: dict[str, list[dict]],
    field_a: str,
    field_b: str,
    match_fn=None,
    count_fn=None,
    n_boot: int = 2000,
    seed: int = 0,
    max_workers: int = 16,
) -> dict:
    """Paired bootstrap of (model_b - model_a) on recall (labeled), fp_rate (clean), halluc (all).

    Only instance_ids present in BOTH prediction sets are compared. Each record is
    scored for both models with verifiable_components (same matcher/counter), then the
    per-record delta is bootstrapped. `*_significant` is True iff the 95% CI shows a
    real IMPROVEMENT: recall CI lower bound > 0; fp/halluc CI upper bound < 0.
    """
    a_by = {r["instance_id"]: r for r in preds_a}
    b_by = {r["instance_id"]: r for r in preds_b}
    ids = [i for i in a_by if i in b_by]

    def _one(iid):
        defects = labels.get(iid, [])
        ra, rb = a_by[iid], b_by[iid]
        ca = verifiable_components(ra.get(field_a, "") or "", defects, ra.get("diff", ""),
                                   match_fn=match_fn, count_fn=count_fn)
        cb = verifiable_components(rb.get(field_b, "") or "", defects, rb.get("diff", ""),
                                   match_fn=match_fn, count_fn=count_fn)
        return defects, ca, cb

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        scored = list(ex.map(_one, ids))

    recalls_a, recalls_b, recall_d = [], [], []
    fps_a, fps_b, fp_d = [], [], []
    hall_a, hall_b, halluc_d = [], [], []
    for defects, ca, cb in scored:
        ha, hb = 1.0 - ca["grounding"], 1.0 - cb["grounding"]
        hall_a.append(ha); hall_b.append(hb); halluc_d.append(hb - ha)
        if defects:
            recalls_a.append(ca["recall"]); recalls_b.append(cb["recall"])
            recall_d.append(cb["recall"] - ca["recall"])
        else:
            fps_a.append(ca["fp_rate"]); fps_b.append(cb["fp_rate"])
            fp_d.append(cb["fp_rate"] - ca["fp_rate"])

    rng = np.random.RandomState(seed)

    def _boot_ci(deltas):
        """Return (observed_mean_delta, ci_lo, ci_hi) via paired bootstrap of the mean."""
        if not deltas:
            return None, None, None
        arr = np.asarray(deltas, dtype=float)
        n = len(arr)
        means = np.empty(n_boot)
        for k in range(n_boot):
            means[k] = arr[rng.randint(0, n, n)].mean()
        return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    rd, rlo, rhi = _boot_ci(recall_d)
    fd, flo, fhi = _boot_ci(fp_d)
    hd, hlo, hhi = _boot_ci(halluc_d)

    return {
        "n_compared": len(ids),
        "n_labeled": len(recall_d),
        "n_clean": len(fp_d),
        "recall_a": _mean(recalls_a),
        "recall_b": _mean(recalls_b),
        "recall_delta": rd,
        "recall_ci": (rlo, rhi),
        "recall_significant": bool(rlo is not None and rlo > 0),
        "fp_a": _mean(fps_a),
        "fp_b": _mean(fps_b),
        "fp_delta": fd,
        "fp_ci": (flo, fhi),
        "fp_significant": bool(fhi is not None and fhi < 0),
        "halluc_a": _mean(hall_a),
        "halluc_b": _mean(hall_b),
        "halluc_delta": hd,
        "halluc_ci": (hlo, hhi),
        "halluc_significant": bool(hhi is not None and hhi < 0),
    }


def _load(path):
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-a", required=True, help="JSONL with instance_id, diff, field_a (e.g. ood_preds_v4.jsonl)")
    ap.add_argument("--field-a", default="v4_pred")
    ap.add_argument("--preds-b", required=True, help="JSONL with instance_id, diff, field_b (v5 preds)")
    ap.add_argument("--field-b", default="corpo_pred")
    ap.add_argument("--labels", required=True, help="defect_labels.jsonl from label_defects.py")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    preds_a, preds_b = _load(args.preds_a), _load(args.preds_b)
    labels = {}
    for l in Path(args.labels).open():
        if l.strip():
            row = json.loads(l)
            labels[row["instance_id"]] = row.get("defects", [])

    print(f"[compare_recall] A={args.preds_a}:{args.field_a} ({len(preds_a)})  "
          f"B={args.preds_b}:{args.field_b} ({len(preds_b)})  labels={len(labels)}",
          file=sys.stderr, flush=True)

    r = paired_delta(preds_a, preds_b, labels, args.field_a, args.field_b,
                     n_boot=args.n_boot, seed=args.seed)

    def _f(x):
        return f"{x:.4f}" if x is not None else "n/a"

    def _ci(c):
        return f"[{_f(c[0])}, {_f(c[1])}]"

    print(f"compared {r['n_compared']} records  (labeled {r['n_labeled']}, clean {r['n_clean']})")
    print(f"  RECALL  A={_f(r['recall_a'])}  B={_f(r['recall_b'])}  "
          f"delta={_f(r['recall_delta'])}  95%CI={_ci(r['recall_ci'])}  "
          f"{'SIGNIFICANT IMPROVEMENT' if r['recall_significant'] else 'not significant'}")
    print(f"  FP      A={_f(r['fp_a'])}  B={_f(r['fp_b'])}  "
          f"delta={_f(r['fp_delta'])}  95%CI={_ci(r['fp_ci'])}  "
          f"{'SIGNIFICANT IMPROVEMENT' if r['fp_significant'] else 'not significant'}")
    print(f"  HALLUC  A={_f(r['halluc_a'])}  B={_f(r['halluc_b'])}  "
          f"delta={_f(r['halluc_delta'])}  95%CI={_ci(r['halluc_ci'])}  "
          f"{'SIGNIFICANT IMPROVEMENT' if r['halluc_significant'] else 'not significant'}")
    print("\nVERDICT: v5 beats v4 on recall iff the recall CI lower bound > 0 "
          "(a real paired gain, not two noisy means).")


if __name__ == "__main__":
    main()
