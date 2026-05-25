"""Unit tests for corpo_train.py — arg parsing + backup verification.

The training-loop tests are deferred to integration testing on Colab (Task 20
provides a subprocess-level integration smoke test for arg threading).
"""
from __future__ import annotations

from pathlib import Path

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
