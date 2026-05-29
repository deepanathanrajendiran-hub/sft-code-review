"""Unit tests for corpo_train.py — arg parsing, backup verification, reward wiring.

v5 (verifiable reward): the opponent/base-cache is gone; training scores against
clean defect-tuple labels (--defect-labels). The training-loop itself is deferred
to Colab integration, but the reward-wiring seams (load_defect_labels,
build_reward_fn) are unit-tested here with an injected matcher (no API).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from corpo_train import (
    parse_args,
    verify_v4_backup,
    load_defect_labels,
    build_reward_fn,
)

REQUIRED = [
    "--v4-adapter", "/v4",
    "--v4-backup", "/bkup",
    "--train-prompts", "p.jsonl",
    "--defect-labels", "labels.jsonl",
    "--output-dir", "/o",
]


class TestParseArgs:
    def test_required_args(self):
        args = parse_args(REQUIRED)
        assert args.v4_adapter == "/v4"
        assert args.v4_backup == "/bkup"
        assert args.train_prompts == "p.jsonl"
        assert args.defect_labels == "labels.jsonl"
        assert args.output_dir == "/o"

    def test_defaults(self):
        args = parse_args(REQUIRED)
        assert args.r_min_correct == 0.5
        # v5: kl_beta anchors the policy to v4 (Run #3's beta=0 caused drift)
        assert args.kl_beta == 0.02
        assert args.learning_rate == 5e-6
        assert args.num_generations == 8
        assert args.max_new_tokens == 2048
        assert args.epochs == 1
        assert args.variance_gate_only is False
        assert args.checkpoint_every == 50
        assert args.prompts_per_step == 4
        assert args.resume is None

    def test_variance_gate_only_flag(self):
        args = parse_args(REQUIRED + ["--variance-gate-only"])
        assert args.variance_gate_only is True


class TestVerifyBackup:
    def test_existing_backup_passes(self, tmp_path):
        backup = tmp_path / "v4-backup"
        backup.mkdir()
        (backup / "adapter_config.json").write_text("{}")
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

    def test_same_path_as_adapter_raises(self, tmp_path):
        path = tmp_path / "v4"
        path.mkdir()
        (path / "adapter_config.json").write_text("{}")
        with pytest.raises(ValueError, match="DIFFERENT path"):
            verify_v4_backup(str(path), str(path))

    def test_different_paths_passes(self, tmp_path):
        backup = tmp_path / "v4-backup"
        backup.mkdir()
        (backup / "adapter_config.json").write_text("{}")
        verify_v4_backup(str(backup), str(tmp_path / "different-adapter"))


class TestVarianceGate:
    def test_pass_when_std_above_threshold(self):
        from corpo_train import _check_variance_gate
        rewards = np.random.default_rng(0).normal(0.5, 0.2, size=(50, 8))
        passed, mean_std = _check_variance_gate(rewards, threshold=0.10)
        assert passed is True
        assert mean_std > 0.10

    def test_fail_when_std_below_threshold(self):
        from corpo_train import _check_variance_gate
        rewards = np.full((50, 8), 0.5)
        passed, mean_std = _check_variance_gate(rewards, threshold=0.10)
        assert passed is False
        assert mean_std == 0.0

    def test_wrong_dim_raises(self):
        from corpo_train import _check_variance_gate
        with pytest.raises(ValueError, match="expected 2D"):
            _check_variance_gate(np.zeros(40), threshold=0.10)


class TestLoadDefectLabels:
    def test_parses_instance_to_defects(self, tmp_path):
        p = tmp_path / "labels.jsonl"
        p.write_text(
            json.dumps({"instance_id": "id0", "defects": [
                {"path": "a.py", "line": 1, "issue_type": "bug", "canonical_desc": "x"}]}) + "\n"
            + json.dumps({"instance_id": "id1", "defects": []}) + "\n"
        )
        labels = load_defect_labels(p)
        assert labels["id0"][0]["canonical_desc"] == "x"
        assert labels["id1"] == []

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "labels.jsonl"
        p.write_text(json.dumps({"instance_id": "id0", "defects": []}) + "\n\n")
        labels = load_defect_labels(p)
        assert labels == {"id0": []}


class TestBuildRewardFn:
    def test_routes_each_record_to_its_defects(self, monkeypatch):
        seen = []

        def spy(diff, rollout, defects, match_fn=None):
            seen.append((diff, rollout, tuple(d.get("canonical_desc") for d in defects)))
            return 0.5

        monkeypatch.setattr("corpo_train.verifiable_reward", spy)
        labels = {"id0": [{"canonical_desc": "x"}]}  # id1 missing -> clean (no defects)
        fn = build_reward_fn(labels, match_fn=lambda r, d: True)
        out = fn(
            prompts=["p0", "p1"],
            completions=["c0", "c1"],
            instance_id=["id0", "id1"],
            diff=["d0", "d1"],
        )
        assert out == [0.5, 0.5]
        assert ("d0", "c0", ("x",)) in seen
        assert ("d1", "c1", ()) in seen  # missing instance -> []

    def test_returns_one_reward_per_prompt(self, monkeypatch):
        monkeypatch.setattr("corpo_train.verifiable_reward", lambda *a, **k: 0.3)
        fn = build_reward_fn({}, match_fn=lambda r, d: False)
        out = fn(prompts=["a", "b", "c"], completions=["x", "y", "z"],
                 instance_id=["1", "2", "3"], diff=["d", "d", "d"])
        assert out == [0.3, 0.3, 0.3]


class TestResumeAndCopyFlags:
    def test_resume_path_passed_through(self):
        args = parse_args(REQUIRED + ["--resume", "/o/checkpoints/step_100"])
        assert args.resume == "/o/checkpoints/step_100"

    def test_copy_to_path_parsed(self):
        args = parse_args(REQUIRED + ["--copy-to", "/drive/v5"])
        assert args.copy_to == "/drive/v5"

    def test_copy_to_default_is_none(self):
        args = parse_args(REQUIRED)
        assert args.copy_to is None
