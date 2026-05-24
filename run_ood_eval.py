"""Sequential vLLM inference: v4 first, then base, on a JSONL of diffs.

Loads each model in turn, generates with production sampling
(temp=0, max_tokens=4096, repetition_penalty=1.1), applies _extract_review
post-process, and writes per-row predictions.

Must be invoked as `python run_ood_eval.py …` (not imported) — vLLM v1's
multiprocessing requires a main-guard.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any




def _extract_review(raw: str) -> str:
    """Extract the `<review>` block from a generated trace.

    1. Primary: `<review>` after the last `</think>`.
    2. Secondary: last `<review>` block whose content isn't just `...`.
    3. Fallback: strip `<think>` blocks, return remainder.
    """
    last_think = raw.rfind("</think>")
    if last_think != -1:
        after = raw[last_think + len("</think>"):]
        m = re.search(r"<review>(.*?)</review>", after, re.DOTALL)
        if m:
            return m.group(1).strip()

    matches = list(re.finditer(r"<review>(.*?)</review>", raw, re.DOTALL))
    for m in reversed(matches):
        body = m.group(1).strip()
        if body and body != "...":
            return body

    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    stripped = re.sub(r"<review>\s*\.\.\.\s*</review>", "", stripped, flags=re.DOTALL)
    return stripped.strip()


def _generate(model_path: str, diffs: list[str], tokenizer) -> list[str]:
    from vllm import LLM, SamplingParams

    SYSTEM_MSG = "You are a Senior Software Engineer reviewing code changes. Provide clear, actionable feedback."
    USER_TEMPLATE = "Review this code diff:\n\n```diff\n{diff}\n```"

    formatted_prompts = []
    for diff in diffs:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(diff=diff[:3000])},
        ]
        formatted_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )
    sp = SamplingParams(
        temperature=0,
        max_tokens=4096,
        repetition_penalty=1.1,
    )
    outputs = llm.generate(formatted_prompts, sp)
    raw_texts = [o.outputs[0].text for o in outputs]
    # Free GPU memory before next model loads
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return raw_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ood_input.jsonl")
    ap.add_argument("--output", required=True, help="ood_preds.jsonl")
    ap.add_argument("--v4-model", required=True)
    ap.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test on first N rows")
    args = ap.parse_args()

    rows: list[dict] = []
    with Path(args.input).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"[run_ood_eval] {len(rows)} rows; generating v4 then base", flush=True)
    diffs = [row["diff"] for row in rows]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"[run_ood_eval] loaded tokenizer from {args.base_model}", flush=True)

    print(f"[run_ood_eval] loading v4 ({args.v4_model})", flush=True)
    v4_raw = _generate(args.v4_model, diffs, tokenizer)
    v4_extracted = [_extract_review(t) for t in v4_raw]

    print(f"[run_ood_eval] loading base ({args.base_model})", flush=True)
    base_raw = _generate(args.base_model, diffs, tokenizer)
    # base doesn't emit <think>/<review>; pass through stripped
    base_extracted = [_extract_review(t) if "<review>" in t else t.strip() for t in base_raw]

    out_path = Path(args.output)
    with out_path.open("w") as fh:
        for row, v4_pred, base_pred in zip(rows, v4_extracted, base_extracted):
            out_row: dict[str, Any] = {
                **row,
                "v4_pred": v4_pred,
                "base_pred": base_pred,
            }
            fh.write(json.dumps(out_row) + "\n")
    print(f"[run_ood_eval] wrote {len(rows)} predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
