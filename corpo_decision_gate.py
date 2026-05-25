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
                    help="JSON with v4-as-A vs v4-as-B baseline win-rate by V4-Pro (sanity, ~50%%)")
    ap.add_argument("--corpo-eval-json", required=True,
                    help="JSON with v4-corpo vs v4 win-rate by V4-Pro 3-vote (includes ci_lo, ci_hi)")
    ap.add_argument("--haiku-cross-check-json", required=True,
                    help="JSON with v4-corpo vs v4 win-rate by Bedrock Haiku (100-prompt subset)")
    ap.add_argument("--variance-gate-passed", action="store_true",
                    help="Set if pre-training variance gate passed")
    ap.add_argument("--training-diverged", action="store_true",
                    help="Set if training loss diverged")
    args = ap.parse_args()

    corpo = json.loads(Path(args.corpo_eval_json).read_text())
    haiku = json.loads(Path(args.haiku_cross_check_json).read_text())

    verdict = decide(
        variance_gate_passed=args.variance_gate_passed,
        training_diverged=args.training_diverged,
        v4corpo_wins_pct=corpo["winrate_pct"],
        ci_lo=corpo["ci_lo"],
        ci_hi=corpo["ci_hi"],
        halluc_v4=corpo.get("halluc_v4", 0.0),
        halluc_v4corpo=corpo.get("halluc_v4corpo", 0.0),
        haiku_cross_check_lift=haiku["winrate_pct"] - 50.0,
        per_domain_spread=corpo.get("per_domain_spread", 0.0),
    )
    print("=" * 60)
    print("PHASE C CORPO DECISION GATE")
    print("=" * 60)
    print(f"v4-corpo wins vs v4:   {corpo['winrate_pct']:.1f}%  [CI {corpo['ci_lo']:.1f}%-{corpo['ci_hi']:.1f}%]")
    print(f"Haiku cross-check:     {haiku['winrate_pct']:.1f}%  (lift {haiku['winrate_pct']-50.0:+.1f}pt)")
    print(f"Halluc:                v4 {corpo.get('halluc_v4',0):.1f}% -> v4-corpo {corpo.get('halluc_v4corpo',0):.1f}%")
    print(f"Per-domain spread:     {corpo.get('per_domain_spread',0):.1f}pt")
    print("-" * 60)
    print(f"VERDICT: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
