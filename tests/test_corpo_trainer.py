"""Unit tests for CoRPOTrainer._compute_corpo_advantages — verify Eq. 11
from the CoRPO paper (arXiv:2511.04439).

These tests do NOT invoke trl.GRPOTrainer.__init__ (which requires a model
and tokenizer). They construct the trainer via __new__ to test the math
in isolation.

Import note: `corpo_trainer.py` does `from trl import GRPOTrainer` at module
load time. Since these tests only exercise pure-torch math on a `__new__`'d
instance, we stub a minimal `trl.GRPOTrainer` before importing the subclass
so the suite runs without the (heavy, GPU-oriented) TRL dependency installed.
The stub is a bare `object` subclass — never instantiated during the tests.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

# torch is a heavy optional dep; without this guard a bare import here aborts
# collection for the WHOLE suite on a fresh clone, not just this module.
torch = pytest.importorskip("torch")

# --- Stub `trl` so `corpo_trainer` is importable without TRL installed. ---
if "trl" not in sys.modules:
    _trl_stub = types.ModuleType("trl")

    class _GRPOTrainerStub:  # pragma: no cover - never instantiated
        pass

    _trl_stub.GRPOTrainer = _GRPOTrainerStub
    sys.modules["trl"] = _trl_stub
# -------------------------------------------------------------------------

from corpo_trainer import CoRPOTrainer  # noqa: E402  (after stub injection)


def _make_trainer(r_min_correct: float, num_generations: int) -> CoRPOTrainer:
    """Bypass __init__ to test advantage math without instantiating TRL."""
    trainer = CoRPOTrainer.__new__(CoRPOTrainer)
    trainer.r_min_correct = r_min_correct
    trainer.num_generations = num_generations
    return trainer


class TestCoRPOAdvantages:
    def test_clip_at_zero_matches_plain_grpo_when_mean_nonneg(self):
        """R_min=0, group_mean>=0 -> baseline = group_mean -> CoRPO == GRPO."""
        trainer = _make_trainer(r_min_correct=0.0, num_generations=3)
        rewards = torch.tensor([0.1, 0.5, 0.9])
        adv = trainer._compute_corpo_advantages(rewards)
        # group_mean = 0.5; clip(0.5, min=0) = 0.5; adv = R - 0.5
        torch.testing.assert_close(adv, torch.tensor([-0.4, 0.0, 0.4]))

    def test_clip_kicks_in_when_group_below_threshold(self):
        """R_min=0.5, group_mean=0.2 -> baseline clipped to 0.5; all R < 0.5 negative."""
        trainer = _make_trainer(r_min_correct=0.5, num_generations=3)
        rewards = torch.tensor([0.0, 0.2, 0.4])
        adv = trainer._compute_corpo_advantages(rewards)
        # group_mean = 0.2; clip(0.2, min=0.5) = 0.5; adv = R - 0.5
        torch.testing.assert_close(adv, torch.tensor([-0.5, -0.3, -0.1]))

    def test_clip_does_not_lower_baseline_when_group_above(self):
        """R_min=0.5, group_mean=0.7 -> baseline stays at group_mean."""
        trainer = _make_trainer(r_min_correct=0.5, num_generations=3)
        rewards = torch.tensor([0.4, 0.7, 1.0])
        adv = trainer._compute_corpo_advantages(rewards)
        # group_mean = 0.7 >= 0.5; clip(0.7, min=0.5) = 0.7; adv = R - 0.7
        torch.testing.assert_close(adv, torch.tensor([-0.3, 0.0, 0.3]))

    def test_paper_example_eight_rollouts(self):
        """Reproduces worked example from spec: 8 rollouts, R_min=3.0, all
        rewards below 3.0 must get negative advantage."""
        trainer = _make_trainer(r_min_correct=3.0, num_generations=8)
        rewards = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # mean=2.5
        adv = trainer._compute_corpo_advantages(rewards)
        # group_mean = 2.5; clip(2.5, min=3.0) = 3.0; adv = R - 3.0
        expected = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        torch.testing.assert_close(adv, expected)
        # Critically: every R < 3.0 has A < 0 (cannot receive positive update)
        assert (adv[rewards < 3.0] < 0).all()

    def test_multi_batch_independent_clips(self):
        """Two independent groups in one flat tensor; each gets its own clipped baseline."""
        trainer = _make_trainer(r_min_correct=0.5, num_generations=3)
        # Two groups of 3 rollouts each; flat layout: [g1_0, g1_1, g1_2, g2_0, g2_1, g2_2]
        rewards = torch.tensor([0.0, 0.2, 0.4,   # group 1 mean 0.2 -> clip to 0.5
                                0.6, 0.8, 1.0])  # group 2 mean 0.8 -> no clip
        adv = trainer._compute_corpo_advantages(rewards)
        expected = torch.tensor([-0.5, -0.3, -0.1,   # group 1: R - 0.5
                                 -0.2,  0.0,  0.2])  # group 2: R - 0.8
        torch.testing.assert_close(adv, expected)

    def test_accepts_2d_input(self):
        """Method should handle (B, G) shape via .view(-1) at the start."""
        trainer = _make_trainer(r_min_correct=0.5, num_generations=3)
        rewards_2d = torch.tensor([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]])
        adv = trainer._compute_corpo_advantages(rewards_2d)
        # Output is flat (B*G,) regardless of input shape
        assert adv.shape == (6,)
        torch.testing.assert_close(
            adv,
            torch.tensor([-0.5, -0.3, -0.1, -0.2, 0.0, 0.2]),
        )

class TestScaleRewardsCheck:
    def test_raises_when_scale_rewards_not_none(self):
        """Verify the safety check fires when GRPOConfig has scale_rewards != 'none'."""
        from unittest.mock import MagicMock

        # Construct CoRPOTrainer.__init__ scope without invoking super (which needs TRL+model)
        # by patching super().__init__ to a no-op
        with patch("corpo_trainer.GRPOTrainer.__init__", return_value=None):
            mock_args = MagicMock()
            mock_args.scale_rewards = "standard"  # Anything except "none"

            with pytest.raises(ValueError, match="scale_rewards"):
                # Need to set self.args before the check runs
                trainer = CoRPOTrainer.__new__(CoRPOTrainer)
                trainer.args = mock_args
                # Manually invoke just the validation block
                CoRPOTrainer.__init__(trainer, args=mock_args)

    def test_passes_when_scale_rewards_is_none(self):
        """No raise when scale_rewards is 'none'."""
        from unittest.mock import MagicMock

        with patch("corpo_trainer.GRPOTrainer.__init__", return_value=None):
            mock_args = MagicMock()
            mock_args.scale_rewards = "none"

            trainer = CoRPOTrainer.__new__(CoRPOTrainer)
            trainer.args = mock_args
            # Should not raise
            CoRPOTrainer.__init__(trainer, args=mock_args)
