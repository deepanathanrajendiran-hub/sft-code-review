"""CoRPO post-training decision gate.

Implements the 8-step ordered checklist from the spec. Run after both eval
JSONs are produced (v4 re-baseline + v4-corpo) and the Haiku cross-check.

Verdicts (in priority order):
  1. ABORTED       — variance gate failed; training never started
  2. FAILED        — training diverged
  3. DO_NOT_SHIP   — Haiku cross-check shows no lift (V4-family judge bias)
  4. FAILED        — hallucination regressed by > 1 pt
  5. SUSPICIOUS    — per-domain spread > 10 pt (overfit risk)
  6. SHIP          — lift >= 1pt AND ci_lo > 50%
  7. INCONCLUSIVE  — lift >= 1pt but ci_lo <= 50% (need more samples)
  8. FAILED        — did not clear 1pt bar
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def decide(
    *,
    variance_gate_passed: bool,
    training_diverged: bool,
    v4corpo_wins_pct: float,
    ci_lo: float,
    ci_hi: float,
    halluc_v4: float,
    halluc_v4corpo: float,
    haiku_cross_check_lift: float,
    per_domain_spread: float,
) -> str:
    """Returns one of: SHIP, FAILED, DO_NOT_SHIP, INCONCLUSIVE, SUSPICIOUS, ABORTED.

    Evaluated in spec order (first match wins):
      1. variance gate failed → ABORTED
      2. training diverged → FAILED
      3. Haiku cross-check negative → DO_NOT_SHIP (judge bias)
      4. halluc regression > 1pt → FAILED
      5. per-domain spread > 10pt → SUSPICIOUS
      6. lift >= 1pt AND ci_lo > 50 → SHIP
      7. lift >= 1pt AND ci_lo <= 50 → INCONCLUSIVE
      8. else → FAILED
    """
    if not variance_gate_passed:
        return "ABORTED"
    if training_diverged:
        return "FAILED"
    if haiku_cross_check_lift < 0:
        return "DO_NOT_SHIP"
    if halluc_v4corpo - halluc_v4 > 1.0:
        return "FAILED"
    if per_domain_spread > 10.0:
        return "SUSPICIOUS"
    lift = v4corpo_wins_pct - 50.0
    if lift >= 1.0 and ci_lo > 50.0:
        return "SHIP"
    if lift >= 1.0 and ci_lo <= 50.0:
        return "INCONCLUSIVE"
    return "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-baseline-json", required=True,
                    help="ood_metrics.py output for original v4 baseline (provides halluc_v4)")
    ap.add_argument("--corpo-eval-json", required=True,
                    help="ood_metrics.py output for v4-corpo vs v4 (provides pairwise + halluc_v4corpo + per-domain)")
    ap.add_argument("--haiku-cross-check-json", required=True,
                    help="ood_metrics.py output for v4-corpo vs v4 with --judge haiku (Goodhart guard)")
    ap.add_argument("--variance-gate-passed", action="store_true",
                    help="Set if pre-training variance gate passed")
    ap.add_argument("--training-diverged", action="store_true",
                    help="Set if training loss diverged")
    args = ap.parse_args()

    corpo = json.loads(Path(args.corpo_eval_json).read_text())
    baseline = json.loads(Path(args.v4_baseline_json).read_text())
    haiku = json.loads(Path(args.haiku_cross_check_json).read_text())

    # Pairwise (all in [0, 1] from ood_metrics; convert to percent)
    pw = corpo.get("pairwise", {})
    winrate_pct = pw.get("win_rate", 0.0) * 100.0
    ci_lo = pw.get("win_rate_ci_lo", 0.0) * 100.0
    ci_hi = pw.get("win_rate_ci_hi", 0.0) * 100.0

    # Hallucination (in [0, 1] from ood_metrics; convert to percent)
    halluc_v4 = baseline.get("hallucination_rate_mean", 0.0) * 100.0
    halluc_v4corpo = corpo.get("hallucination_rate_mean", 0.0) * 100.0

    # Per-domain spread = max - min IoU across problem_domain buckets (in percent points)
    per_domain = corpo.get("iou_lenient_by_problem_domain", {}) or {}
    if per_domain:
        values_pct = [v * 100.0 for v in per_domain.values() if isinstance(v, (int, float))]
        per_domain_spread = max(values_pct) - min(values_pct) if values_pct else 0.0
    else:
        per_domain_spread = 0.0

    # Haiku cross-check lift (point estimate; no CI needed at N=100)
    haiku_winrate_pct = haiku.get("pairwise", {}).get("win_rate", 0.0) * 100.0
    haiku_cross_check_lift = haiku_winrate_pct - 50.0

    verdict = decide(
        variance_gate_passed=args.variance_gate_passed,
        training_diverged=args.training_diverged,
        v4corpo_wins_pct=winrate_pct,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        halluc_v4=halluc_v4,
        halluc_v4corpo=halluc_v4corpo,
        haiku_cross_check_lift=haiku_cross_check_lift,
        per_domain_spread=per_domain_spread,
    )
    print("=" * 60)
    print("PHASE C CORPO DECISION GATE")
    print("=" * 60)
    print(f"v4-corpo wins vs v4:   {winrate_pct:.1f}%  [CI {ci_lo:.1f}%-{ci_hi:.1f}%]")
    print(f"Haiku cross-check:     {haiku_winrate_pct:.1f}%  (lift {haiku_cross_check_lift:+.1f}pt)")
    print(f"Halluc:                v4 {halluc_v4:.1f}% -> v4-corpo {halluc_v4corpo:.1f}%")
    print(f"Per-domain spread:     {per_domain_spread:.1f}pt")
    print("-" * 60)
    print(f"VERDICT: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
