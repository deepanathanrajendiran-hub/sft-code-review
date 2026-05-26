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

# CRITICAL: Unsloth must be imported BEFORE any direct/indirect trl import.
# vLLM 0.12+ removed GuidedDecodingParams from vllm.sampling_params; TRL 0.22.2
# still imports it at module-load time. Unsloth's package-init monkey-patches
# vllm.sampling_params to provide a shim. Wrapped in try/except so local test
# environments (no unsloth installed) keep working.
try:
    import unsloth  # noqa: F401 — load for its import-time monkey-patches
except ImportError:
    pass

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
    ap.add_argument(
        "--copy-to",
        default=None,
        help="After training, copy final adapter to this path (e.g., Drive). Uses shutil.copytree.",
    )
    return ap.parse_args(argv)


def verify_v4_backup(backup_path: str, adapter_path: str | None = None) -> None:
    """Refuse to proceed unless backup exists AND is distinct from the adapter to be trained.

    adapter_path defaults to None for backward compat with tests that test
    backup-only checks; production main() always passes it.
    """
    p = Path(backup_path)
    if not p.exists():
        raise FileNotFoundError(f"v4 backup directory does not exist: {backup_path}")
    if not (p / "adapter_config.json").exists():
        raise ValueError(
            f"v4 backup at {backup_path} is missing adapter_config.json — "
            f"refusing to start training in case this is not actually a LoRA backup"
        )
    if adapter_path is not None and Path(backup_path).resolve() == Path(adapter_path).resolve():
        raise ValueError(
            f"--v4-backup must be a DIFFERENT path from --v4-adapter; both resolved to {Path(backup_path).resolve()}"
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

def run_training(args: argparse.Namespace) -> None:
    """Run CoRPO training: load v4 adapter, set up trainer, train num_train_epochs.

    All heavy deps (torch, transformers, peft, trl, datasets) are lazy-imported
    inside this function. The unit tests for parse_args + verify_v4_backup +
    _check_variance_gate do NOT need these installed.
    """
    import json

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from corpo_trainer import CoRPOTrainer
    from corpo_reward import composite_reward, load_base_sample_cache

    # 1. Load training prompts and the pre-built base-sample cache
    with open(args.train_prompts) as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    base_cache = load_base_sample_cache(args.base_cache)
    print(
        f"[corpo_train] loaded {len(prompts)} prompts + {len(base_cache._data)} base samples",
        file=sys.stderr,
    )

    # 2. Format prompts via chat template (matches run_ood_eval format)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    SYSTEM_MSG = (
        "You are a Senior Software Engineer reviewing code changes. "
        "Provide clear, actionable feedback."
    )
    USER_TEMPLATE = (
        "Review the following code diff and provide feedback:\n```diff\n{diff}\n```"
    )

    def _format(p):
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(diff=p["diff"][:12000])},
        ]
        p["prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return p

    dataset = Dataset.from_list([_format(p) for p in prompts])

    # 3. Load model with v4 LoRA adapter — continue-training the existing LoRA
    print(f"[corpo_train] loading {args.base_model} + v4 LoRA", file=sys.stderr)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.v4_adapter, is_trainable=True)

    # 4. Reward function — closure captures base_cache
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """TRL GRPO calls this with batched prompts+completions; we score each.

        TRL passes extra dataset columns via kwargs (instance_id, diff, reference_text).
        Parallelized over judge calls (16 workers) — each call is ~1s on V4-Pro
        non-thinking mode, so 32 calls = ~2s with parallelism vs ~32s serial.
        """
        from concurrent.futures import ThreadPoolExecutor

        instance_ids = kwargs["instance_id"]
        diffs = kwargs["diff"]
        refs = kwargs.get("reference_text", [""] * len(prompts))

        def _score_one(i: int) -> float:
            base_sample = base_cache.get(instance_ids[i])
            return composite_reward(diffs[i], completions[i], base_sample, refs[i])

        with ThreadPoolExecutor(max_workers=16) as executor:
            rewards = list(executor.map(_score_one, range(len(prompts))))
        return rewards

    # 5. Build GRPO config; CoRPOTrainer extends GRPOTrainer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        warmup_steps=50,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        beta=args.kl_beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        per_device_train_batch_size=args.prompts_per_step,
        num_train_epochs=args.epochs,
        save_steps=args.checkpoint_every,
        logging_steps=10,
        temperature=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        bf16=True,
        scale_rewards="none",  # CoRPOTrainer requires this
    )

    trainer = CoRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        r_min_correct=args.r_min_correct,
    )

    # 6. Train (resume support added by Task 15)
    print(
        f"[corpo_train] starting training; r_min_correct={args.r_min_correct}",
        file=sys.stderr,
    )
    trainer.train(resume_from_checkpoint=args.resume)

    # 7. Save final adapter to output_dir/final/
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    print(f"[corpo_train] saved final adapter to {final_path}", file=sys.stderr)
    if args.copy_to:
        import shutil
        dest = Path(args.copy_to)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(final_path, dest)
        print(f"[corpo_train] copied final adapter to {dest}", file=sys.stderr)



def main():
    args = parse_args()
    verify_v4_backup(args.v4_backup, args.v4_adapter)
    if args.variance_gate_only:
        passed = run_variance_gate(args)
        sys.exit(0 if passed else 1)
    run_training(args)

def _generate_v4_rollouts(
    adapter_path: str,
    base_model: str,
    prompts: list[dict],
    num_generations: int,
    max_new_tokens: int,
) -> list[list[str]]:
    """vLLM-generate num_generations rollouts per prompt using base + v4 LoRA via vLLM's LoRA support.

    Returns list of length len(prompts), each containing num_generations strings.
    Uses temperature=1.0 (CoRPO paper) for diversity.

    Lazy-imports vLLM and transformers — only invoked when running on Colab GPU.
    """
    import gc

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
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
        model=base_model,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enable_lora=True,
        max_lora_rank=64,  # v4 uses r=32, alpha=64; max_lora_rank must be >= rank
    )
    sp = SamplingParams(
        temperature=1.0,
        max_tokens=max_new_tokens,
        n=num_generations,
        # NOTE: repetition_penalty intentionally omitted (defaults to 1.0).
        # TRL's GRPOTrainer also uses 1.0 during training, so the variance gate
        # must match training sampling exactly to be a faithful pre-flight check.
    )
    lora_request = LoRARequest("v4", 1, adapter_path)
    outputs = llm.generate(formatted, sp, lora_request=lora_request)
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

    from concurrent.futures import ThreadPoolExecutor

    def _score_one(work_item):
        i, j, prompt, rollout, base = work_item
        return i, j, composite_reward(
            diff=prompt["diff"],
            rollout=rollout,
            base_sample=base,
            reference=prompt.get("reference_text", ""),
        )

    work = []
    for i, prompt in enumerate(sample):
        base = base_cache.get(prompt["instance_id"])
        for j, rollout in enumerate(rollouts_by_prompt[i]):
            work.append((i, j, prompt, rollout, base))

    rewards = np.zeros((len(sample), args.num_generations))
    print(f"[variance-gate] scoring {len(work)} (prompt, rollout) pairs in parallel...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=16) as executor:
        for i, j, r in executor.map(_score_one, work):
            rewards[i, j] = r

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
