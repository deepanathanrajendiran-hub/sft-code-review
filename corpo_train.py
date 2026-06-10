"""CoRPO training entry point.

Usage:
    # Pre-training variance-gate sanity check (does NOT train)
    python corpo_train.py --variance-gate-only \\
        --v4-adapter /path/to/v4-lora \\
        --v4-backup  /path/to/v4-lora-backup \\
        --train-prompts ood_train_prompts.jsonl \\
        --defect-labels cache/defect_labels.jsonl \\
        --output-dir /content/corpo-out

    # Full training (~4-6h on Colab A100). --v4-merged is REQUIRED for training:
    # the policy is merged-v4 + a fresh LoRA so the KL reference (adapters disabled)
    # is actually v4, and prompts are never left-truncated (max_prompt_length=6144,
    # not TRL's silent 512 default).
    python corpo_train.py [same args as above without --variance-gate-only] \\
        --v4-merged /content/sft-v4-merged-for-eval

    # Resume from checkpoint. Checkpoints are HF-named `checkpoint-<N>`. After a Colab
    # disconnect the local --output-dir is wiped; resume from the Drive mirror (the path
    # passed to --checkpoint-sync-dir), e.g.:
    #   python corpo_train.py --resume /content/drive/MyDrive/sft/corpo-out-v5/checkpoint-150 [other args]

Refuses to start if --v4-backup is missing adapter_config.json, so we never train
over the only copy of v4.
"""
from __future__ import annotations

# Unsloth must be imported before anything that pulls in trl. vLLM 0.12+ dropped
# GuidedDecodingParams from vllm.sampling_params, but TRL 0.22.2 still imports it at
# module load; Unsloth's init monkey-patches a shim in. try/except keeps local test
# envs (no unsloth) working.
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
                    help="Path to v4 LoRA adapter (used by the variance gate's vLLM LoRA hot-swap)")
    ap.add_argument("--v4-merged", default=None,
                    help="Path to the MERGED v4 model (e.g. sft-v4-merged-for-eval). Required for "
                         "training: the policy is merged-v4 + a FRESH LoRA, so TRL's disable_adapter "
                         "KL reference is actually v4. Loading the adapter directly would make the "
                         "KL reference the BASE model (adapters disabled = base), pulling the policy "
                         "away from v4 — the v5.0/v5.1 bug.")
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
        help="KL coefficient to the v4 reference. Only a true v4 anchor when the model is "
             "loaded via --v4-merged + fresh LoRA (disable_adapter then yields merged v4). "
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
    """Refuse to proceed unless the backup exists and is a distinct path from the adapter.

    adapter_path is optional so tests can check the backup alone; main() always passes it.
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

    rewards is (n_prompts, num_generations) — one row per prompt's G rollouts. threshold
    is the minimum mean within-group std we need before the reward signal is worth training on.
    """
    if rewards.ndim != 2:
        raise ValueError(f"expected 2D rewards (n_prompts, G), got shape {rewards.shape}")
    per_group_std = rewards.std(axis=1)
    mean_std = float(per_group_std.mean())
    return mean_std >= threshold, mean_std

def _make_lora_config_tolerant() -> None:
    """Make peft.LoraConfig ignore kwargs it doesn't know.

    The installed unsloth is newer than the pinned peft 0.17.1 and passes
    LoraConfig kwargs that 0.17.1 rejects (e.g. ensure_weight_tying, added in
    peft 0.18). peft 0.18+ can't be installed against the pinned transformers
    4.56.2 (its tensor-parallel import needs 4.57+), so we tolerate instead of
    upgrade. Dropping these is safe here: they only affect embedding/lm_head
    tying or exotic LoRA variants, and we target attention+MLP modules only.
    Idempotent; patches the class object, so unsloth's module-level
    `from peft import LoraConfig` sees it too.
    """
    import inspect
    import peft

    if getattr(peft.LoraConfig, "_kwarg_shim_installed", False):
        return
    orig_init = peft.LoraConfig.__init__
    valid = set(inspect.signature(orig_init).parameters) - {"self"}

    def tolerant_init(self, *a, **kw):
        dropped = sorted(k for k in kw if k not in valid)
        if dropped:
            print(
                f"[corpo_train] LoraConfig: dropping kwargs unsupported by "
                f"peft {peft.__version__}: {dropped}",
                file=sys.stderr,
            )
        orig_init(self, *a, **{k: v for k, v in kw.items() if k in valid})

    peft.LoraConfig.__init__ = tolerant_init
    peft.LoraConfig._kwarg_shim_installed = True


def run_training(args: argparse.Namespace) -> None:
    """Load merged-v4 + a fresh LoRA, build the CoRPO trainer, and train.

    Heavy deps (torch, transformers, peft, trl, datasets) are lazy-imported here so the
    unit tests for parse_args / verify_v4_backup / _check_variance_gate don't need them.
    """
    import json

    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import GRPOConfig

    from corpo_trainer import CoRPOTrainer

    if not args.v4_merged:
        raise SystemExit(
            "[corpo_train] --v4-merged is required for training. Loading the adapter "
            "directly makes TRL's disable_adapter KL reference the BASE model, not v4 "
            "(the v5.0/v5.1 bug). Pass the merged v4 dir (e.g. sft-v4-merged-for-eval)."
        )

    with open(args.train_prompts) as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    defect_labels = load_defect_labels(args.defect_labels)
    # Records absent from the labels (ambiguous: a comment failed grounding and no
    # grounded defect survived) must NOT train — build_reward_fn would default them
    # to [] and the clean-restraint penalty would punish flagging real defects.
    pre_label_filter = len(prompts)
    prompts = [p for p in prompts if p["instance_id"] in defect_labels]
    if pre_label_filter - len(prompts):
        print(
            f"[corpo_train] dropped {pre_label_filter - len(prompts)}/{pre_label_filter} "
            f"prompts with ambiguous/missing defect labels",
            file=sys.stderr,
        )
    n_labeled = sum(1 for v in defect_labels.values() if v)
    print(
        f"[corpo_train] loaded {len(prompts)} prompts + defect labels for "
        f"{len(defect_labels)} instances ({n_labeled} with >=1 defect, "
        f"{len(defect_labels) - n_labeled} clean)",
        file=sys.stderr,
    )

    # Policy = merged v4 weights + a FRESH LoRA. With a PEFT model TRL computes the KL
    # reference by disabling adapters, so the reference is the underlying weights: here
    # that's merged v4 (correct anchor). Loading base+v4-adapter instead would anchor
    # to BASE — the bug that dragged v5.0/v5.1 toward base behavior.
    # fast_inference=True attaches a vllm_engine that Unsloth's patched GRPOTrainer uses.
    print(
        f"[corpo_train] loading merged v4 ({args.v4_merged}) + fresh LoRA via FastLanguageModel",
        file=sys.stderr,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.v4_merged,
        max_seq_length=8192,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.9,
    )
    _make_lora_config_tolerant()
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,  # TRL disables dropout for GRPO anyway
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Chat template must match run_ood_eval's format.
    SYSTEM_MSG = (
        "You are a Senior Software Engineer reviewing code changes. "
        "Provide clear, actionable feedback."
    )
    USER_TEMPLATE = (
        "Review the following code diff and provide feedback:\n```diff\n{diff}\n```"
    )

    # Diff char cap. max_model_len is 8192; minus max_completion_length=2048 leaves ~6144
    # tokens for the prompt, minus ~200 for system+template overhead. At a conservative
    # ~1.0 chars/token (worst case for dense code) 5000 chars stays under the budget.
    # We've seen a 12000-char diff tokenize to 8906 tokens and crash mid-run, so keep this low.
    DIFF_CHAR_LIMIT = 5000

    # Even after truncation+templating a pathological diff could exceed budget, so we drop
    # any prompt still over 6000 tokens rather than risk a crash mid-training.
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

    # match_fn=None falls back to defect_match's DeepSeek constrained yes/no judge.
    reward_fn = build_reward_fn(defect_labels)

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
        # TRL's DEFAULT max_prompt_length is 512 and it LEFT-truncates every prompt to
        # the last 512 tokens — which silently cut the system message and most of the
        # diff in every pre-v5.2 run. 6144 covers the 6000-token prompt filter above;
        # 6144 + 2048 completion = 8192 = max_seq_length.
        max_prompt_length=6144,
        max_completion_length=args.max_new_tokens,
        # TRL needs per_device_train_batch_size % num_generations == 0 (the device
        # micro-batch holds one prompt's G completions), so pdtbs = num_generations and
        # gradient_accumulation_steps = prompts_per_step (accumulate that many prompts
        # before stepping). pdtbs=prompts_per_step would fail this check at init.
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=args.prompts_per_step,
        num_train_epochs=args.epochs,
        save_steps=args.checkpoint_every,
        logging_steps=10,
        temperature=1.0,
        use_vllm=True,
        # vllm_mode left unset on purpose: Unsloth's patched GRPOTrainer uses
        # model.vllm_engine directly, so TRL's server/colocate default is a no-op here.
        bf16=True,
        loss_type="dr_grpo",  # length-bias mitigation (arXiv:2503.20783)
        scale_rewards="none",  # CoRPOTrainer requires this
    )
    print(
        f"[corpo_train] prompt regime: max_prompt_length={grpo_config.max_prompt_length} "
        f"(prompts pre-filtered to <=6000 tokens; nothing gets left-truncated)",
        file=sys.stderr,
    )

    trainer = CoRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        r_min_correct=args.r_min_correct,
    )

    # Mirror each checkpoint to a persistent path. Colab disconnects wipe /content/, so
    # without this any in-session checkpoints are lost and resume is impossible.
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

    print(
        f"[corpo_train] starting training; r_min_correct={args.r_min_correct}",
        file=sys.stderr,
    )
    trainer.train(resume_from_checkpoint=args.resume)

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

    # Pre-flight: confirm TRL exposes loss_type. If it's silently absent the run falls
    # back to the default ("bnpo"), which is the length pathology we're trying to avoid.
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

    # Pre-flight: TRL's GRPOConfig.max_prompt_length DEFAULTS to 512 and left-truncates
    # every prompt. run_training sets it to 6144 explicitly; this check exists so the
    # invariant is loud if the config construction ever changes.
    if "max_prompt_length" not in _trl_params:
        raise RuntimeError(
            "TRL GRPOConfig does not expose max_prompt_length — cannot guarantee "
            "prompts won't be silently truncated to TRL's internal default. Abort."
        )
    print("[corpo_train] pre-flight: max_prompt_length is configurable (set to 6144 in training) ✓",
          file=sys.stderr)

    # run_training sets pdtbs = num_generations so the divisibility constraint holds
    # structurally; just echo the resulting batch shape.
    print(
        f"[corpo_train] pre-flight: batch ok (pdtbs=num_generations={args.num_generations}, "
        f"grad_accum=prompts_per_step={args.prompts_per_step}) ✓",
        file=sys.stderr,
    )

    if args.variance_gate_only:
        passed = run_variance_gate(args)
        sys.exit(0 if passed else 1)

    if not args.v4_merged:
        raise SystemExit(
            "[corpo_train] --v4-merged is required for training (KL must anchor to "
            "merged v4, not base). Pass e.g. /content/sft-v4-merged-for-eval."
        )
    run_training(args)

def _generate_v4_rollouts(
    adapter_path: str,
    base_model: str,
    prompts: list[dict],
    num_generations: int,
    max_new_tokens: int,
) -> list[list[str]]:
    """vLLM-generate num_generations raw rollouts per prompt using base + v4 LoRA.

    Returns one list per prompt of num_generations raw completion strings (with
    <think>/<review>). The gate scores these via verifiable_reward, which extracts
    internally, so we must NOT pre-extract here (that was a double-extraction skew vs
    training). Diff truncation (5000 chars) and temperature=1.0 match run_training so the
    pre-flight is faithful. vLLM/transformers are lazy-imported (Colab GPU only).
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
        max_lora_rank=64,  # v4 is r=32, alpha=64; max_lora_rank must be >= rank
    )
    sp = SamplingParams(
        temperature=1.0,
        max_tokens=max_new_tokens,
        n=num_generations,
        # repetition_penalty left at default 1.0 to match TRL's training sampling — the
        # gate has to sample exactly like training to be a faithful pre-flight.
    )
    lora_request = LoRARequest("v4", 1, adapter_path)
    outputs = llm.generate(formatted, sp, lora_request=lora_request)
    result: list[list[str]] = []
    for o in outputs:
        rollouts = [comp.text for comp in o.outputs]  # raw; reward extracts internally
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
    """Generate G rollouts × up to 50 prompts with the v4 policy and check reward spread.

    Scores the rollouts with verifiable_reward and inspects the within-group reward std
    as a signal-strength check. Also prints a histogram and percentile-based R_min
    candidates so the operator can pick a calibrated --r-min-correct for the real run.
    Samples only prompts with unambiguous defect labels (same filter as training) so the
    gate measures the same prompt population training will see.
    Returns True if mean within-group std >= 0.10; writes the verdict to stderr either way.
    """
    import json
    import random

    defect_labels = load_defect_labels(args.defect_labels)

    with open(args.train_prompts) as fh:
        all_prompts = [json.loads(line) for line in fh if line.strip()]
    pre_filter = len(all_prompts)
    all_prompts = [p for p in all_prompts if p["instance_id"] in defect_labels]
    if pre_filter - len(all_prompts):
        print(
            f"[variance-gate] dropped {pre_filter - len(all_prompts)}/{pre_filter} prompts "
            f"with ambiguous/missing defect labels (matches training filter)",
            file=sys.stderr,
        )
    rng = random.Random(42)
    sample = rng.sample(all_prompts, min(50, len(all_prompts)))

    print(f"[variance-gate] generating {len(sample)} x {args.num_generations} v4 rollouts", file=sys.stderr)
    rollouts_by_prompt = _generate_v4_rollouts(
        args.v4_adapter, args.base_model, sample, args.num_generations, args.max_new_tokens
    )

    from concurrent.futures import ThreadPoolExecutor

    # Score the raw rollout (verifiable_reward extracts once internally) so the gate
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

    # Diagnose the zero-reward mass: hard 0s are format failures (truncated <think>
    # or empty extraction), not "low quality". They are excluded from the R_min
    # percentiles below — a >33% point mass at 0 made pooled p33 = 0.000, and
    # R_min=0 never clamps a non-negative reward (CoRPO degenerates to plain GRPO).
    flat = rewards.ravel()
    zero_mask = flat < 1e-9
    n_zero = int(zero_mask.sum())
    if n_zero:
        n_trunc = sum(
            1
            for prompt_rollouts in rollouts_by_prompt
            for r in prompt_rollouts
            if "<think>" in r and "</think>" not in r
        )
        print(
            f"[variance-gate] zero-reward rollouts: {n_zero}/{len(flat)} "
            f"(unclosed <think> i.e. truncated at max-new-tokens: {n_trunc}; "
            f"empty/other extraction failures: {max(0, n_zero - n_trunc)})",
            file=sys.stderr,
        )
    nonzero = flat[~zero_mask]

    # The CoRPO paper sets R_min at the correctness boundary (below the median of correct
    # rollouts, above the max of incorrect). For continuous [0,1] rewards we read it off
    # the NONZERO rollout distribution (format failures aren't "incorrect reviews",
    # they're degenerate outputs); p33 by default.
    if not passed:
        print(
            "[variance-gate] WARNING: gate FAILED — rollout distribution is too narrow. "
            "Percentile values below are unreliable; do NOT use them for R_min. "
            "Investigate the reward function or base-sample cache before training.",
            file=sys.stderr,
        )
    if len(nonzero) == 0:
        print(
            "[variance-gate] every rollout scored 0 — no R_min candidates. "
            "Fix the reward/extraction before training.",
            file=sys.stderr,
        )
        print(f"[variance-gate] verdict: FAIL", file=sys.stderr)
        return False
    p25, p33, p40, p50 = np.percentile(nonzero, [25, 33, 40, 50])
    print(f"[variance-gate] R_min candidates (percentiles of NONZERO rewards, n={len(nonzero)}):",
          file=sys.stderr)
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

    One JSON object per line; an empty "defects" list marks a clean diff (the model
    should find nothing). Records with no grounded defects but >=1 comment dropped
    for grounding (n_ungrounded > 0) are AMBIGUOUS — the diff may contain a real
    defect we just couldn't ground — so they're skipped here; callers must drop
    prompts whose instance_id is absent (treating them as clean would punish the
    model for flagging real issues). Older label files without n_ungrounded load
    unchanged.
    """
    import json
    labels: dict[str, list[dict]] = {}
    skipped_ambiguous = 0
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            defects = row.get("defects", [])
            if not defects and row.get("n_ungrounded", 0) > 0:
                skipped_ambiguous += 1
                continue
            labels[row["instance_id"]] = defects
    if skipped_ambiguous:
        print(
            f"[corpo_train] skipped {skipped_ambiguous} ambiguous label records "
            f"(no grounded defects but ungrounded defect comments exist)",
            file=sys.stderr,
        )
    return labels


def build_reward_fn(defect_labels: dict[str, list[dict]], match_fn=None):
    """Return a TRL-compatible reward closure that scores completions with verifiable_reward.

    TRL calls reward_fn(prompts, completions, **kwargs), where kwargs carries the dataset
    columns (instance_id, diff) replicated per completion. Each completion is scored
    against its instance's defect tuples (a missing instance means a clean diff, []).
    Matcher calls run on 16 workers; match_fn is injectable for tests.
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
