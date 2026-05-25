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
import numpy as np


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


def _check_variance_gate(rewards: np.ndarray, threshold: float = 0.10) -> tuple[bool, float]:
    """Return (passed, mean_within_group_std).

    rewards: shape (n_prompts, num_generations). Each row = one prompt's G rollouts.
    threshold: minimum mean within-group std required to consider reward signal usable.
    """
    if rewards.ndim != 2:
        raise ValueError(f"expected 2D rewards (n_prompts, G), got shape {rewards.shape}")
    per_group_std = rewards.std(axis=1)
    mean_std = float(per_group_std.mean())
    return mean_std >= threshold, mean_std

def main():
    args = parse_args()
    verify_v4_backup(args.v4_backup)
    if args.variance_gate_only:
        passed = run_variance_gate(args)
        sys.exit(0 if passed else 1)
    # Training loop implemented in Task 14
    print("[corpo_train] training loop not yet implemented (Task 14)", file=sys.stderr)
    sys.exit(1)

def _generate_v4_rollouts(
    adapter_path: str,
    base_model: str,
    prompts: list[dict],
    num_generations: int,
    max_new_tokens: int,
) -> list[list[str]]:
    """vLLM-generate num_generations rollouts per prompt using base+adapter.

    Returns list of length len(prompts), each containing num_generations strings.
    Uses temperature=1.0 (CoRPO paper) for diversity.

    Lazy-imports vLLM and transformers — only invoked when running on Colab GPU.
    """
    import gc

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from run_ood_eval import _extract_review

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    SYSTEM_MSG = "You are a Senior Software Engineer reviewing code changes. Provide clear, actionable feedback."
    USER_TEMPLATE = "Review the following code diff and provide feedback:\n```diff\n{diff}\n```"

    formatted = []
    for p in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(diff=p["diff"][:12000])},
        ]
        formatted.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    llm = LLM(
        model=adapter_path,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
    )
    sp = SamplingParams(
        temperature=1.0,
        max_tokens=max_new_tokens,
        n=num_generations,
        repetition_penalty=1.1,
    )
    outputs = llm.generate(formatted, sp)
    result: list[list[str]] = []
    for o in outputs:
        rollouts = [_extract_review(comp.text) for comp in o.outputs]
        result.append(rollouts)

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def run_variance_gate(args: argparse.Namespace) -> bool:
    """Generate 8 rollouts × 50 prompts with v4 policy; check reward spread.

    Returns True if mean within-group std >= 0.10. Prints histogram + verdict to stderr.
    """
    import json
    import random
    from corpo_reward import composite_reward, load_base_sample_cache

    # Load 50 random train prompts
    with open(args.train_prompts) as fh:
        all_prompts = [json.loads(line) for line in fh if line.strip()]
    rng = random.Random(42)
    sample = rng.sample(all_prompts, min(50, len(all_prompts)))

    base_cache = load_base_sample_cache(args.base_cache)

    print(f"[variance-gate] generating {len(sample)} x {args.num_generations} v4 rollouts", file=sys.stderr)
    rollouts_by_prompt = _generate_v4_rollouts(
        args.v4_adapter, args.base_model, sample, args.num_generations, args.max_new_tokens
    )

    rewards = np.zeros((len(sample), args.num_generations))
    for i, prompt in enumerate(sample):
        base = base_cache.get(prompt["instance_id"])
        for j, rollout in enumerate(rollouts_by_prompt[i]):
            rewards[i, j] = composite_reward(
                diff=prompt["diff"],
                rollout=rollout,
                base_sample=base,
                reference=prompt.get("reference_text", ""),
            )

    passed, mean_std = _check_variance_gate(rewards, threshold=0.10)
    print(f"[variance-gate] mean within-group std = {mean_std:.4f} (threshold 0.10)", file=sys.stderr)
    print(f"[variance-gate] reward histogram (bins of 0.1):", file=sys.stderr)
    hist, edges = np.histogram(rewards.ravel(), bins=10, range=(0, 1))
    for h, e in zip(hist, edges[:-1]):
        print(f"  [{e:.1f}, {e+0.1:.1f}): {'#' * min(h, 60)}  ({h})", file=sys.stderr)
    print(f"[variance-gate] verdict: {'PASS' if passed else 'FAIL'}", file=sys.stderr)
    return passed


if __name__ == "__main__":
    main()
