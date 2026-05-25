"""Unit tests for corpo_decision_gate.decide — pure function, 8 branches."""
from __future__ import annotations

import pytest

from corpo_decision_gate import decide


class TestDecide:
    def _inputs(self, **overrides):
        defaults = dict(
            variance_gate_passed=True,
            training_diverged=False,
            v4corpo_wins_pct=53.0,
            ci_lo=51.0,
            ci_hi=55.0,
            halluc_v4=3.6,
            halluc_v4corpo=3.5,
            haiku_cross_check_lift=2.0,
            per_domain_spread=4.0,
        )
        defaults.update(overrides)
        return defaults

    def test_clean_ship(self):
        assert decide(**self._inputs()) == "SHIP"

    def test_aborted_if_variance_gate_failed(self):
        assert decide(**self._inputs(variance_gate_passed=False)) == "ABORTED"

    def test_failed_if_training_diverged(self):
        assert decide(**self._inputs(training_diverged=True)) == "FAILED"

    def test_failed_if_halluc_regressed(self):
        # halluc_v4corpo - halluc_v4 > 1.0
        assert decide(**self._inputs(halluc_v4corpo=5.0)) == "FAILED"

    def test_do_not_ship_if_haiku_check_negative(self):
        assert decide(**self._inputs(haiku_cross_check_lift=-2.0)) == "DO_NOT_SHIP"

    def test_inconclusive_if_ci_overlaps_50(self):
        # lift_pts >= 1 but ci_lo <= 50
        assert decide(**self._inputs(v4corpo_wins_pct=51.5, ci_lo=49.5)) == "INCONCLUSIVE"

    def test_failed_if_lift_below_one_pt(self):
        assert decide(**self._inputs(v4corpo_wins_pct=50.3, ci_lo=49.0)) == "FAILED"

    def test_suspicious_if_per_domain_spread_high(self):
        assert decide(**self._inputs(per_domain_spread=15.0)) == "SUSPICIOUS"

    def test_ordering_aborted_beats_diverged(self):
        # Both flagged — aborted should win (it's first in spec order)
        assert decide(**self._inputs(
            variance_gate_passed=False,
            training_diverged=True,
        )) == "ABORTED"

    def test_ordering_diverged_beats_haiku_check(self):
        # Both flagged — diverged should win
        assert decide(**self._inputs(
            training_diverged=True,
            haiku_cross_check_lift=-5.0,
        )) == "FAILED"

    def test_halluc_regression_boundary_exactly_one_pt_not_failed(self):
        # halluc_v4corpo == halluc_v4 + 1.0 → NOT failed (rule is "> 1.0", not ">= 1.0")
        assert decide(**self._inputs(halluc_v4=3.6, halluc_v4corpo=4.6)) == "SHIP"

    def test_halluc_regression_just_above_one_pt_fails(self):
        # halluc_v4corpo > halluc_v4 + 1.0 → FAILED
        assert decide(**self._inputs(halluc_v4=3.6, halluc_v4corpo=4.61)) == "FAILED"
