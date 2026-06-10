"""Mid-training eval: sweep the v5 checkpoints, score each against the v4 bar.

Must be run as a subprocess (`python mid_eval.py ...`), not inside a notebook
cell: vLLM v1's engine calls sys.stdout.fileno(), which a notebook stdout doesn't
support (io.UnsupportedOperation: fileno). Same reason we shell out for the
variance gate and run_ood_eval.

Builds the prompt set once (same 12000-char budgeted regime as run_ood_eval, so
the comparison vs the v4 preds is apples-to-apples), then hot-swaps each
checkpoint's LoRA over MERGED v4 (v5.2+ checkpoints are deltas on merged-v4
weights, not on the raw base — no per-checkpoint re-merge) and scores with
score_v5, reporting defect_recall / fp_rate / halluc vs v4.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _dr(s):
    return s["defect_recall_labeled"] if s["defect_recall_labeled"] is not None else 0.0


def _fp(s):
    return s["fp_rate_clean"] if s["fp_rate_clean"] is not None else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", required=True, help="dir holding checkpoint-N subdirs (Drive mirror)")
    ap.add_argument("--v4-preds", required=True, help="ood_preds_v4.jsonl (provides the fixed sample + v4_pred bar)")
    ap.add_argument("--labels", required=True, help="defect_labels_eval.jsonl")
    ap.add_argument("--v4-merged", required=True,
                    help="Merged v4 model dir. v5.2+ checkpoints are LoRA deltas ON MERGED V4 "
                         "(training loads merged-v4 + fresh LoRA), so they must be hot-swapped "
                         "over merged v4 — swapping over the raw base evaluates the wrong model.")
    ap.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct",
                    help="Used only for the tokenizer (chat template matches v4's)")
    ap.add_argument("--n-samples", type=int, default=50)
    args = ap.parse_args()

    import gc
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    import score_v5
    from run_ood_eval import _extract_review, format_prompt_with_budget

    labels = {}
    for line in open(args.labels):
        if line.strip():
            r = json.loads(line)
            labels[r["instance_id"]] = r.get("defects", [])
    rows = [json.loads(l) for l in open(args.v4_preds) if l.strip()]
    sample = random.Random(42).sample(rows, min(args.n_samples, len(rows)))

    v4 = score_v5.score(sample, labels, "v4_pred")
    v4_dr, v4_fp, v4_h = _dr(v4), _fp(v4), v4["halluc_mean"]
    print(f"[mid-eval] v4 bar (n={len(sample)}): defect_recall={v4_dr:.3f} fp_rate={v4_fp:.3f} halluc={v4_h:.3f}",
          flush=True)

    ckpts = sorted(
        [d for d in Path(args.checkpoint_root).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    print(f"[mid-eval] checkpoints: {[d.name for d in ckpts]}", flush=True)
    if not ckpts:
        print(f"[mid-eval] no checkpoints under {args.checkpoint_root}", file=sys.stderr)
        return

    tok = AutoTokenizer.from_pretrained(args.base_model)
    # Same prompt regime as run_ood_eval generated the v4 bar with (12000-char budgeted
    # truncation). The old flat diff[:5000] cap gave checkpoints less context than the
    # v4 preds they were compared against.
    prompts = [format_prompt_with_budget(r["diff"], tok) for r in sample]

    llm = LLM(model=args.v4_merged, gpu_memory_utilization=0.85, max_model_len=8192,
              enable_lora=True, max_lora_rank=64)
    sp = SamplingParams(temperature=0, max_tokens=4096, repetition_penalty=1.1)

    best = None
    for ckpt in ckpts:
        step = int(ckpt.name.split("-")[1])
        outs = llm.generate(prompts, sp, lora_request=LoRARequest(f"v5-{step}", step, str(ckpt)))
        preds = [{"instance_id": sample[i]["instance_id"], "diff": sample[i]["diff"],
                  "corpo_pred": _extract_review(o.outputs[0].text)} for i, o in enumerate(outs)]
        s = score_v5.score(preds, labels, "corpo_pred")
        ok = _fp(s) <= v4_fp + 1e-9 and s["halluc_mean"] <= v4_h + 1e-9
        print(f"[mid-eval] step {step}: defect_recall={_dr(s):.3f} (d{_dr(s)-v4_dr:+.3f}) "
              f"fp_rate={_fp(s):.3f} (d{_fp(s)-v4_fp:+.3f}) halluc={s['halluc_mean']:.3f} (d{s['halluc_mean']-v4_h:+.3f})"
              f"{'  [beats v4]' if (ok and _dr(s) > v4_dr) else ''}", flush=True)
        if ok and (best is None or _dr(s) > best[1]):
            best = (step, _dr(s))

    del llm
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    if best:
        print(f"[mid-eval] BEST (beats fp/halluc bar, max recall): checkpoint-{best[0]} "
              f"(defect_recall={best[1]:.3f} vs v4 {v4_dr:.3f})", flush=True)
    else:
        print("[mid-eval] no checkpoint kept fp_rate & halluc <= v4 on this subset", flush=True)


if __name__ == "__main__":
    main()
