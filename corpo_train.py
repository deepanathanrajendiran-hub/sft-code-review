"""CoRPO training entry point.

Usage:
    # Pre-training variance-gate sanity check (does NOT train)
    python corpo_train.py --variance-gate-only \\
        --v4-adapter /path/to/v4-lora \\
        --v4-backup  /path/to/v4-lora-backup \\
        --train-prompts ood_train_prompts.jsonl \\
        --base-cache cache/base_samples.jsonl \\
        --output-dir /content/corpo-out

    # Full training (~4-6h on Colab A100)
    python corpo_train.py [same args as above without --variance-gate-only]

    # Resume from checkpoint
    python corpo_train.py --resume /content/corpo-out/checkpoints/step_50 [other args]

Safety: refuses to start if --v4-backup does not contain adapter_config.json,
to prevent accidentally training over the only copy of v4.

This skeleton (Task 12) only provides parse_args + verify_v4_backup. The
variance-gate, training loop, checkpointing, and Drive sync are added by
Tasks 13-16.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. argv=None uses sys.argv[1:]."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-adapter", required=True,
                    help="Path to v4 LoRA adapter to continue training")
    ap.add_argument("--v4-backup", required=True,
                    help="Path to v4 LoRA backup (training refuses to start without it)")
    ap.add_argument("--train-prompts", required=True,
                    help="JSONL of training prompts (SWE-CARE 80%% split)")
    ap.add_argument("--base-cache", required=True,
                    help="JSONL of cached base-model rollouts (built by corpo_reward.py --build-base-cache)")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for checkpoints and final adapter")
    ap.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--r-min-correct", type=float, default=0.5)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--num-generations", type=int, default=8,
                    help="G in CoRPO; rollouts per prompt")
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--variance-gate-only", action="store_true",
                    help="Run pre-training variance check on 50 prompts and exit (Task 13)")
    ap.add_argument("--resume", default=None,
                    help="Resume from checkpoint directory")
    return ap.parse_args(argv)


def verify_v4_backup(backup_path: str) -> None:
    """Refuse to proceed unless the v4 backup contains adapter_config.json.

    Prevents mutating the only copy of v4. The CoRPO training continues v4's
    LoRA weights in-place (per the design); a backup must exist outside the
    training output directory in case the run fails or produces bad weights.
    """
    p = Path(backup_path)
    if not p.exists():
        raise FileNotFoundError(f"v4 backup directory does not exist: {backup_path}")
    if not (p / "adapter_config.json").exists():
        raise ValueError(
            f"v4 backup at {backup_path} is missing adapter_config.json — "
            f"refusing to start training in case this is not actually a LoRA backup"
        )
    print(f"[corpo_train] verified v4 backup at {backup_path}", file=sys.stderr)


def main():
    args = parse_args()
    verify_v4_backup(args.v4_backup)
    if args.variance_gate_only:
        # Variance gate implemented in Task 13
        print("[corpo_train] --variance-gate-only not yet implemented (Task 13)", file=sys.stderr)
        sys.exit(1)
    # Training loop implemented in Task 14
    print("[corpo_train] training loop not yet implemented (Task 14)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
