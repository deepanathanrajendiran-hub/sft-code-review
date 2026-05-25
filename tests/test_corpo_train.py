"""Unit tests for corpo_train.py — arg parsing + backup verification.

The training-loop tests are deferred to integration testing on Colab (Task 20
provides a subprocess-level integration smoke test for arg threading).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from corpo_train import parse_args, verify_v4_backup


class TestParseArgs:
    def test_required_args(self):
        args = parse_args([
            "--v4-adapter", "/path/to/v4",
            "--v4-backup", "/path/to/backup",
            "--train-prompts", "train.jsonl",
            "--base-cache", "base.jsonl",
            "--output-dir", "/content/out",
        ])
        assert args.v4_adapter == "/path/to/v4"
        assert args.v4_backup == "/path/to/backup"
        assert args.train_prompts == "train.jsonl"
        assert args.base_cache == "base.jsonl"
        assert args.output_dir == "/content/out"

    def test_defaults(self):
        args = parse_args([
            "--v4-adapter", "/v4",
            "--v4-backup", "/bkup",
            "--train-prompts", "p.jsonl",
            "--base-cache", "b.jsonl",
            "--output-dir", "/o",
        ])
        assert args.r_min_correct == 0.5
        assert args.kl_beta == 0.01
        assert args.learning_rate == 5e-6
        assert args.num_generations == 8
        assert args.max_new_tokens == 2048
        assert args.epochs == 1
        assert args.variance_gate_only is False
        assert args.checkpoint_every == 50
        assert args.prompts_per_step == 4
        assert args.resume is None

    def test_variance_gate_only_flag(self):
        args = parse_args([
            "--v4-adapter", "/v4",
            "--v4-backup", "/bkup",
            "--train-prompts", "p.jsonl",
            "--base-cache", "b.jsonl",
            "--output-dir", "/o",
            "--variance-gate-only",
        ])
        assert args.variance_gate_only is True


class TestVerifyBackup:
    def test_existing_backup_passes(self, tmp_path):
        backup = tmp_path / "v4-backup"
        backup.mkdir()
        (backup / "adapter_config.json").write_text("{}")
        # Should not raise
        verify_v4_backup(str(backup))

    def test_missing_backup_raises(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="v4 backup"):
            verify_v4_backup(str(missing))

    def test_empty_directory_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="adapter_config.json"):
            verify_v4_backup(str(empty))

class TestVarianceGate:
    def test_pass_when_std_above_threshold(self):
        from corpo_train import _check_variance_gate
        # 50 prompts × 8 rollouts; rewards have std=0.2 per group → mean std > 0.10
        rewards = np.random.default_rng(0).normal(0.5, 0.2, size=(50, 8))
        passed, mean_std = _check_variance_gate(rewards, threshold=0.10)
        assert passed is True
        assert mean_std > 0.10

    def test_fail_when_std_below_threshold(self):
        from corpo_train import _check_variance_gate
        # All rewards same in each group → std=0
        rewards = np.full((50, 8), 0.5)
        passed, mean_std = _check_variance_gate(rewards, threshold=0.10)
        assert passed is False
        assert mean_std == 0.0

    def test_threshold_boundary_inclusive(self):
        from corpo_train import _check_variance_gate
        # Exactly at threshold → pass
        rewards = np.zeros((10, 8))
        rewards[:, 0] = 0.2  # introduce some spread
        passed, _ = _check_variance_gate(rewards, threshold=0.0)
        assert passed is True

    def test_wrong_dim_raises(self):
        from corpo_train import _check_variance_gate
        with pytest.raises(ValueError, match="expected 2D"):
            _check_variance_gate(np.zeros(40), threshold=0.10)
