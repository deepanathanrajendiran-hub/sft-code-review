"""CoRPO training entry point.

Usage:
    # Pre-training variance-gate sanity check (does NOT train)
    python corpo_train.py --variance-gate-only \\
        --v4-adapter /path/to/v4-lora \\
        --v4-backup  /path/to/v4-lora-backup \\
        --train-prompts ood_train_prompts.jsonl \\
        --defect-labels cache/defect_labels.jsonl \\
        --output-dir /content/corpo-out

    # Full training (~4-6h on Colab A100)
    python corpo_train.py [same args as above without --variance-gate-only]

    # Resume from checkpoint. Checkpoints are HF-named `checkpoint-<N>`. After a Colab
    # disconnect the local --output-dir is wiped; resume from the Drive mirror (the path
    # passed to --checkpoint-sync-dir), e.g.:
    #   python corpo_train.py --resume /content/drive/MyDrive/sft/corpo-out-v5/checkpoint-150 [other args]

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
from corpo_reward import verifiable_reward


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-adapter", required=True,
                    help="Path to v4 LoRA adapter to continue training")
    ap.add_argument("--v4-backup", required=True,
                    help="Path to v4 LoRA backup (training refuses to start without it)")
    ap.add_argument("--train-prompts", required=True,
                    help="JSONL of training prompts (SWE-CARE dev split)")
    ap.add_argument("--defect-labels", required=True,
                    help="JSONL of clean defect tuples per instance "
                         "(built by label_defects.py): {instance_id, defects:[{path,line,issue_type,canonical_desc}]}")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for checkpoints and final adapter")
    ap.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--r-min-correct", type=float, default=0.5,
                    help="CoRPO baseline-clip threshold. Calibrate from the v5 variance "
                         "gate's p33 of the v4 reward distribution (correctness boundary).")
    ap.add_argument(
        "--kl-beta",
        type=float,
        default=0.02,
        help="KL coefficient to the v4 reference. v5 restores KL>0 (Run #3's beta=0 "
             "removed the only anchor to v4 and caused capability/format drift). "
             "DeepSeek-Math used 0.04, R1 used 0.001; 0.02 is a mid anchor.",
    )
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--num-generations", type=int, default=8,
                    help="G in CoRPO; rollouts per prompt")
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--variance-gate-only", action="store_true",
                    help="Run pre-training variance check on 50 prompts and exit")
    ap.add_argument("--resume", default=None,
                    help="Resume from checkpoint directory")
    ap.add_argument(
        "--copy-to",
        default=None,
        help="After training, copy final adapter to this path (e.g., Drive). Uses shutil.copytree.",
    )
    ap.add_argument(
        "--checkpoint-sync-dir",
        default=None,
        help=(
            "If set, every checkpoint saved during training is also mirrored here "
            "(typically a Drive path). Survives Colab session boundaries — required "
            "for any run that won't finish in one session. Uses shutil.copytree."
        ),
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
    from unsloth import FastLanguageModel
    from trl import GRPOConfig

    from corpo_trainer import CoRPOTrainer

    # 1. Load training prompts and the clean defect-tuple labels (v5: no opponent)
    with open(args.train_prompts) as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    defect_labels = load_defect_labels(args.defect_labels)
    n_labeled = sum(1 for v in defect_labels.values() if v)
    print(
        f"[corpo_train] loaded {len(prompts)} prompts + defect labels for "
        f"{len(defect_labels)} instances ({n_labeled} with >=1 defect, "
        f"{len(defect_labels) - n_labeled} clean)",
        file=sys.stderr,
    )

    # 2. Load model via FastLanguageModel — canonical Unsloth GRPO path.
    # `fast_inference=True` attaches a `vllm_engine` attribute to the model,
    # which Unsloth's patched GRPOTrainer reads (`self.llm = model.vllm_engine`)
    # in place of standard TRL vLLM colocation. Passing the v4 adapter as
    # `model_name` lets Unsloth auto-detect the base model from its
    # adapter_config.json and load both in one shot — already trainable.
    print(
        f"[corpo_train] loading v4 adapter ({args.v4_adapter}) via FastLanguageModel",
        file=sys.stderr,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.v4_adapter,
        max_seq_length=8192,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.9,
    )

    # 3. Format prompts via chat template (matches run_ood_eval format)
    SYSTEM_MSG = (
        "You are a Senior Software Engineer reviewing code changes. "
        "Provide clear, actionable feedback."
    )
    USER_TEMPLATE = (
        "Review the following code diff and provide feedback:\n```diff\n{diff}\n```"
    )

    # Char truncation for the diff. vLLM's max_model_len is 8192 (from FastLanguageModel
    # max_seq_length=8192). Subtract max_completion_length=2048 → 6144 tokens for the
    # prompt. Allow ~200 tokens for system+template overhead → ~5900 tokens for the
    # diff. At a conservative 1.5 chars/token for dense code, that's ~8800 chars.
    # 12000 chars previously crashed at step 32 (a diff tokenized to 8906 tokens —
    # ~1.35 chars/token). 5000 chars is the safe ceiling: even at 1.0 chars/token
    # (worst case) it stays under 5000 tokens, well below the 6144 budget.
    DIFF_CHAR_LIMIT = 5000

    # Additionally, filter out any prompt that still produces >6000 tokens after
    # truncation+templating, so a worst-case diff can't crash the run mid-training.
    def _format(p):
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(diff=p["diff"][:DIFF_CHAR_LIMIT])},
        ]
        p["prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return p

    formatted = [_format(p) for p in prompts]
    pre_filter = len(formatted)
    formatted = [
        p for p in formatted
        if len(tokenizer.encode(p["prompt"], add_special_tokens=False)) <= 6000
    ]
    dropped = pre_filter - len(formatted)
    if dropped:
        print(
            f"[corpo_train] dropped {dropped}/{pre_filter} prompts that exceed 6000 tokens after truncation",
            file=sys.stderr,
        )
    dataset = Dataset.from_list(formatted)

    # 4. Reward function — verifiable reward against clean defect tuples (v5).
    # Default match_fn (None) uses defect_match's DeepSeek constrained yes/no judge.
    reward_fn = build_reward_fn(defect_labels)

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
        # TRL requires per_device_train_batch_size % num_generations == 0 (the device
        # micro-batch holds one prompt's G completions). So pdtbs = num_generations
        # (one prompt's group per micro-step) and gradient_accumulation_steps =
        # prompts_per_step (accumulate that many prompts before an optimizer step).
        # The old pdtbs=prompts_per_step(4) % num_generations(8) != 0 -> ValueError at init.
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=args.prompts_per_step,
        num_train_epochs=args.epochs,
        save_steps=args.checkpoint_every,
        logging_steps=10,
        temperature=1.0,
        use_vllm=True,
        # NOTE: vllm_mode intentionally NOT set. Unsloth's patched GRPOTrainer
        # uses `model.vllm_engine` directly (from FastLanguageModel) rather than
        # TRL's standard server/colocate paths. The default ("server") is a
        # no-op when Unsloth's patches are active.
        bf16=True,
        loss_type="dr_grpo",  # Run #3 fix: length-bias mitigation (arXiv:2503.20783)
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

    # Mirror every saved checkpoint to a persistent location (typically Drive).
    # Colab session disconnects wipe /content/, so without this any in-session
    # checkpoints are lost; resume becomes impossible.
    if args.checkpoint_sync_dir:
        import shutil as _shutil
        from transformers import TrainerCallback
        _sync_dir = Path(args.checkpoint_sync_dir)

        class _DriveCheckpointSync(TrainerCallback):
            def on_save(self, _args, state, control, **kwargs):
                step = state.global_step
                src = Path(_args.output_dir) / f"checkpoint-{step}"
                if not src.exists():
                    return
                _sync_dir.mkdir(parents=True, exist_ok=True)
                dst = _sync_dir / f"checkpoint-{step}"
                if dst.exists():
                    _shutil.rmtree(dst)
                _shutil.copytree(src, dst)
                print(
                    f"[corpo_train] synced checkpoint-{step} → {dst}",
                    file=sys.stderr,
                )

        trainer.add_callback(_DriveCheckpointSync())
        print(
            f"[corpo_train] checkpoint sync enabled → {_sync_dir}",
            file=sys.stderr,
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

    # Pre-flight: assert TRL exposes loss_type so Dr.GRPO can be applied.
    # If silently absent, the run would degenerate to default ("bnpo") which
    # is the source of the length pathology documented in Runs #1 and #2.
    import inspect
    from trl import GRPOConfig as _GRPOC
    _trl_params = inspect.signature(_GRPOC).parameters
    if "loss_type" not in _trl_params:
        raise RuntimeError(
            "TRL version does not expose GRPOConfig.loss_type. Abort: Run #3 "
            "depends on loss_type='dr_grpo' to fix the length-aggregation bias. "
            "Verify TRL is pinned at 0.22.2 (or compatible) and retry."
        )
    print("[corpo_train] pre-flight: TRL exposes loss_type ✓", file=sys.stderr)

    # GRPO requires per_device_train_batch_size % num_generations == 0. run_training sets
    # pdtbs = num_generations, so this holds structurally; gradient_accumulation_steps =
    # prompts_per_step accumulates that many prompts per optimizer step.
    print(
        f"[corpo_train] pre-flight: batch ok (pdtbs=num_generations={args.num_generations}, "
        f"grad_accum=prompts_per_step={args.prompts_per_step}) ✓",
        file=sys.stderr,
    )

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
    """vLLM-generate num_generations RAW rollouts per prompt using base + v4 LoRA.

    Returns list of length len(prompts), each containing num_generations RAW
    completion strings (with <think>/<review>). The variance gate scores these via
    verifiable_reward, which extracts internally — so the gate must NOT pre-extract
    (doing so was a double-extraction skew vs. training). Diff truncation (5000 chars)
    and temperature=1.0 match run_training exactly for a faithful pre-flight.

    Lazy-imports vLLM and transformers — only invoked when running on Colab GPU.
    """
    import gc

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    SYSTEM_MSG = "You are a Senior Software Engineer reviewing code changes. Provide clear, actionable feedback."
    USER_TEMPLATE = "Review the following code diff and provide feedback:\n```diff\n{diff}\n```"

    formatted = []
    for p in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(diff=p["diff"][:5000])},
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
        rollouts = [comp.text for comp in o.outputs]  # RAW; reward extracts internally
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

    Generates `args.num_generations` rollouts for each of up to 50 random training
    prompts using the v4 LoRA via vLLM, scores them with `verifiable_reward`, and
    inspects the within-group reward std as a signal-strength check. Also prints
    a histogram and percentile-based R_min suggestions so the operator can pick
    a calibrated `--r-min-correct` value for the actual training run.

    Returns True if mean within-group std >= 0.10. Side effect: writes a verdict,
    histogram, and percentile candidates to stderr.
    """
    import json
    import random

    # Load 50 random train prompts
    with open(args.train_prompts) as fh:
        all_prompts = [json.loads(line) for line in fh if line.strip()]
    rng = random.Random(42)
    sample = rng.sample(all_prompts, min(50, len(all_prompts)))

    defect_labels = load_defect_labels(args.defect_labels)

    print(f"[variance-gate] generating {len(sample)} x {args.num_generations} v4 rollouts", file=sys.stderr)
    rollouts_by_prompt = _generate_v4_rollouts(
        args.v4_adapter, args.base_model, sample, args.num_generations, args.max_new_tokens
    )

    from concurrent.futures import ThreadPoolExecutor

    # Score the RAW rollout (verifiable_reward extracts once internally) so the gate
    # measures exactly what training measures — no double-extraction skew.
    def _score_one(work_item):
        i, j, prompt, rollout = work_item
        defects = defect_labels.get(prompt["instance_id"], [])
        return i, j, verifiable_reward(prompt["diff"], rollout, defects)

    work = []
    for i, prompt in enumerate(sample):
        for j, rollout in enumerate(rollouts_by_prompt[i]):
            work.append((i, j, prompt, rollout))

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

    # R_min calibration: CoRPO paper sets R_min at the correctness boundary
    # (below median of correct rollouts, above max of incorrect). For continuous
    # composite rewards in [0,1], pick from the rollout distribution itself —
    # NOT an aspirational quality target. Default recommendation: p33.
    p25, p33, p40, p50 = np.percentile(rewards.ravel(), [25, 33, 40, 50])
    if not passed:
        print(
            "[variance-gate] WARNING: gate FAILED — rollout distribution is too narrow. "
            "Percentile values below are unreliable; do NOT use them for R_min. "
            "Investigate the reward function or base-sample cache before training.",
            file=sys.stderr,
        )
    print(f"[variance-gate] R_min candidates (pick from rollout distribution):", file=sys.stderr)
    print(f"  p25={p25:.3f}   p33={p33:.3f}   p40={p40:.3f}   p50={p50:.3f}", file=sys.stderr)
    if passed:
        print(f"  -> recommended default: p33 = {p33:.3f}", file=sys.stderr)
        print(f"  -> pass via: --r-min-correct {p33:.3f}", file=sys.stderr)
    else:
        print("  -> recommendation withheld (gate failed; see WARNING above)", file=sys.stderr)
    print(f"[variance-gate] verdict: {'PASS' if passed else 'FAIL'}", file=sys.stderr)
    return passed


def load_defect_labels(path) -> dict[str, list[dict]]:
    """Load clean defect tuples keyed by instance_id (built by label_defects.py).

    JSONL: one {"instance_id": ..., "defects": [{path,line,issue_type,canonical_desc}, ...]}
    per line. An empty "defects" list marks a clean diff (model should find nothing).
    """
    import json
    labels: dict[str, list[dict]] = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[row["instance_id"]] = row.get("defects", [])
    return labels


def build_reward_fn(defect_labels: dict[str, list[dict]], match_fn=None):
    """Return a TRL-compatible reward closure scoring each completion with verifiable_reward.

    TRL calls reward_fn(prompts, completions, **kwargs); kwargs carries the dataset
    columns (instance_id, diff) replicated per completion. Each completion is scored
    against its instance's clean defect tuples (a missing instance => clean diff, []).
    Semantic-matcher calls are parallelized (16 workers); match_fn is injectable for tests.
    """
    from concurrent.futures import ThreadPoolExecutor

    def reward_fn(prompts, completions, **kwargs):
        instance_ids = kwargs["instance_id"]
        diffs = kwargs["diff"]

        def _score_one(i):
            defects = defect_labels.get(instance_ids[i], [])
            return verifiable_reward(diffs[i], completions[i], defects, match_fn=match_fn)

        with ThreadPoolExecutor(max_workers=16) as ex:
            return list(ex.map(_score_one, range(len(prompts))))

    return reward_fn


if __name__ == "__main__":
    main()
