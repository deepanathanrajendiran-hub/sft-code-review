"""End-to-end smoke test for corpo_train.py + corpo_decision_gate.py invocation flow.

These tests do NOT train a real model. They verify:
  1. corpo_train.py fails loud when --v4-backup is missing (safety guarantee)
  2. corpo_decision_gate.py runs end-to-end on synthetic JSONs and prints the verdict
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")


def test_variance_gate_only_aborts_on_missing_backup(tmp_path):
    """corpo_train.py must refuse to run if --v4-backup path is missing."""
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"instance_id": "x", "diff": "x"}) + "\n")
    cache = tmp_path / "base.jsonl"
    cache.write_text(json.dumps({"instance_id": "x", "base_output": "y"}) + "\n")

    result = subprocess.run(
        [
            PYTHON, "corpo_train.py",
            "--v4-adapter", "/nonexistent",
            "--v4-backup", "/definitely/not/here",
            "--train-prompts", str(train),
            "--base-cache", str(cache),
            "--output-dir", str(tmp_path / "out"),
            "--variance-gate-only",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode != 0
    combined = (result.stderr + result.stdout).lower()
    assert "v4 backup" in combined, f"Expected 'v4 backup' in output, got: {combined}"


def test_decision_gate_ship_verdict(tmp_path):
    """corpo_decision_gate.py prints VERDICT: SHIP on synthetic clean-pass JSONs."""
    corpo = tmp_path / "corpo.json"
    corpo.write_text(json.dumps({
        "winrate_pct": 53.0,
        "ci_lo": 51.0,
        "ci_hi": 55.0,
        "halluc_v4": 3.6,
        "halluc_v4corpo": 3.5,
        "per_domain_spread": 4.0,
    }))
    haiku = tmp_path / "haiku.json"
    haiku.write_text(json.dumps({"winrate_pct": 52.0}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"winrate_pct": 50.0}))

    result = subprocess.run(
        [
            PYTHON, "corpo_decision_gate.py",
            "--v4-baseline-json", str(baseline),
            "--corpo-eval-json", str(corpo),
            "--haiku-cross-check-json", str(haiku),
            "--variance-gate-passed",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "VERDICT: SHIP" in result.stdout, f"stdout: {result.stdout}"


def test_decision_gate_aborted_when_variance_gate_flag_omitted(tmp_path):
    """corpo_decision_gate.py prints VERDICT: ABORTED when --variance-gate-passed is omitted."""
    corpo = tmp_path / "corpo.json"
    corpo.write_text(json.dumps({
        "winrate_pct": 53.0,
        "ci_lo": 51.0,
        "ci_hi": 55.0,
        "halluc_v4": 3.6,
        "halluc_v4corpo": 3.5,
        "per_domain_spread": 4.0,
    }))
    haiku = tmp_path / "haiku.json"
    haiku.write_text(json.dumps({"winrate_pct": 52.0}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"winrate_pct": 50.0}))

    result = subprocess.run(
        [
            PYTHON, "corpo_decision_gate.py",
            "--v4-baseline-json", str(baseline),
            "--corpo-eval-json", str(corpo),
            "--haiku-cross-check-json", str(haiku),
            # --variance-gate-passed NOT included
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0
    assert "VERDICT: ABORTED" in result.stdout, f"stdout: {result.stdout}"
