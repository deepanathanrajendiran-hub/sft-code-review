"""Unit tests for corpo_reward.py components."""
from __future__ import annotations

import pytest

from corpo_reward import length_sanity_score


class TestLengthSanity:
    def test_within_band_returns_one(self):
        assert length_sanity_score("x" * 200) == 1.0
        assert length_sanity_score("x" * 1000) == 1.0
        assert length_sanity_score("x" * 4000) == 1.0

    def test_too_short_linear_taper(self):
        assert length_sanity_score("x" * 100) == 0.5
        assert length_sanity_score("") == 0.0

    def test_too_long_linear_taper(self):
        # n=6000: 1 - (6000-4000)/4000 = 0.5
        assert length_sanity_score("x" * 6000) == 0.5
        # n=8000 floor
        assert length_sanity_score("x" * 8000) == 0.0
        # n=12000 stays at floor
        assert length_sanity_score("x" * 12000) == 0.0

    def test_just_below_band(self):
        # n=199: 199/200
        assert length_sanity_score("x" * 199) == pytest.approx(199 / 200)
