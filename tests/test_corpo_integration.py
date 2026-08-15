"""End-to-end smoke test for corpo_train.py + corpo_decision_gate.py invocation flow.

These tests do NOT train a real model. They verify:
  1. corpo_train.py fails loud when --v4-backup is missing (safety guarantee)
  2. corpo_decision_gate.py runs end-to-end on synthetic JSONs and prints the verdict
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Use the interpreter running the tests, not a hardcoded venv path — the latter
# only resolves on the author's machine.
PYTHON = sys.executable


def test_variance_gate_only_aborts_on_missing_backup(tmp_path):
    """corpo_train.py must refuse to run if --v4-backup path is missing."""
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"instance_id": "x", "diff": "x"}) + "\n")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"instance_id": "x", "defects": []}) + "\n")

    result = subprocess.run(
        [
            PYTHON, "corpo_train.py",
            "--v4-adapter", "/nonexistent",
            "--v4-backup", "/definitely/not/here",
            "--train-prompts", str(train),
            "--defect-labels", str(labels),
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
    """corpo_decision_gate.py prints VERDICT: SHIP on synthetic clean-pass JSONs (real ood_metrics schema)."""
    corpo = tmp_path / "corpo.json"
    corpo.write_text(json.dumps({
        "hallucination_rate_mean": 0.035,  # 3.5%
        "iou_lenient_by_problem_domain": {"general": 0.08, "framework": 0.07, "data": 0.09},
        "pairwise": {"win_rate": 0.53, "win_rate_ci_lo": 0.51, "win_rate_ci_hi": 0.55},
    }))
    haiku = tmp_path / "haiku.json"
    haiku.write_text(json.dumps({"pairwise": {"win_rate": 0.52}}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"hallucination_rate_mean": 0.036}))  # 3.6%

    result = subprocess.run([
        PYTHON, "corpo_decision_gate.py",
        "--v4-baseline-json", str(baseline),
        "--corpo-eval-json", str(corpo),
        "--haiku-cross-check-json", str(haiku),
        "--variance-gate-passed",
    ], capture_output=True, text=True, cwd=str(REPO_ROOT))

    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "VERDICT: SHIP" in result.stdout


def test_decision_gate_aborted_when_variance_gate_flag_omitted(tmp_path):
    """corpo_decision_gate.py prints VERDICT: ABORTED when --variance-gate-passed is omitted (real ood_metrics schema)."""
    corpo = tmp_path / "corpo.json"
    corpo.write_text(json.dumps({
        "hallucination_rate_mean": 0.035,
        "iou_lenient_by_problem_domain": {"general": 0.08, "framework": 0.07, "data": 0.09},
        "pairwise": {"win_rate": 0.53, "win_rate_ci_lo": 0.51, "win_rate_ci_hi": 0.55},
    }))
    haiku = tmp_path / "haiku.json"
    haiku.write_text(json.dumps({"pairwise": {"win_rate": 0.52}}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"hallucination_rate_mean": 0.036}))

    result = subprocess.run([
        PYTHON, "corpo_decision_gate.py",
        "--v4-baseline-json", str(baseline),
        "--corpo-eval-json", str(corpo),
        "--haiku-cross-check-json", str(haiku),
        # --variance-gate-passed NOT included
    ], capture_output=True, text=True, cwd=str(REPO_ROOT))

    assert result.returncode == 0
    assert "VERDICT: ABORTED" in result.stdout, f"stdout: {result.stdout}"
