"""OOD evaluation runner for code-review LoRAs.

Primary use: invoke as a CLI script via `python run_ood_eval.py ...` to
generate predictions and compute metrics on the SWE-CARE eval set.

The module is also safe to import — specifically, `_extract_review` is
used by `corpo_reward.py` as a library helper for reward-time review
extraction. Heavy dependencies (vLLM, transformers) are imported inside
function bodies, not at module load, so importing this file has no side
effects beyond stdlib imports.
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
    USER_TEMPLATE = "Review the following code diff and provide feedback:\n```diff\n{diff}\n```"

    # Token budget = max_model_len - max_tokens - safety margin.
    # COUPLING: this constant assumes max_model_len=8192 (LLM() below) and
    # max_tokens=4096 (SamplingParams below). If either changes, update this too.
    INPUT_TOKEN_BUDGET = 8192 - 4096 - 100  # = 3996

    def _format_with_budget(diff: str) -> str:
        """Format a single diff under INPUT_TOKEN_BUDGET, iteratively shrinking on overflow.

        Adversarial diffs (URLs, base64, non-ASCII) can have <2.5 chars/token,
        so 12000 chars may exceed 4096 tokens. Shrink to 70% and retry until it fits.
        """
        truncated_chars = 12000
        formatted = ""
        while truncated_chars >= 200:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_TEMPLATE.format(diff=diff[:truncated_chars])},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            token_count = len(tokenizer.encode(formatted))
            if token_count <= INPUT_TOKEN_BUDGET:
                return formatted
            # Over budget — shrink to 70% and retry
            truncated_chars = int(truncated_chars * 0.7)
        # Hard floor reached — return what we have; vLLM may still fit it
        return formatted

    formatted_prompts = [_format_with_budget(diff) for diff in diffs]

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
    ap.add_argument(
        "--skip-base",
        action="store_true",
        default=False,
        help="Skip base-model generation (e.g. when base preds already cached). "
             "Writes base_pred as empty string for each row to preserve column shape.",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    with Path(args.input).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if args.limit is not None:
        rows = rows[: args.limit]

    if args.skip_base:
        print(f"[run_ood_eval] {len(rows)} rows; generating v4 only (--skip-base)", flush=True)
    else:
        print(f"[run_ood_eval] {len(rows)} rows; generating v4 then base", flush=True)
    diffs = [row["diff"] for row in rows]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"[run_ood_eval] loaded tokenizer from {args.base_model}", flush=True)

    # Sanity: confirm v4 didn't customize chat_template (would silently confound pairwise)
    v4_tok = AutoTokenizer.from_pretrained(args.v4_model)
    assert v4_tok.chat_template == tokenizer.chat_template, (
        f"v4 chat_template differs from base — would silently confound pairwise. "
        f"Either rebuild v4 with matching template or load tokenizer per-model."
    )
    del v4_tok  # release; we only need base
    print(f"[run_ood_eval] chat_template matches v4_model: OK", flush=True)

    print(f"[run_ood_eval] loading v4 ({args.v4_model})", flush=True)
    v4_raw = _generate(args.v4_model, diffs, tokenizer)
    v4_extracted = [_extract_review(t) for t in v4_raw]

    if not args.skip_base:
        print(f"[run_ood_eval] loading base ({args.base_model})", flush=True)
        base_raw = _generate(args.base_model, diffs, tokenizer)
        # base doesn't emit <think>/<review>; pass through stripped
        base_extracted = [_extract_review(t) if "<review>" in t else t.strip() for t in base_raw]
    else:
        print(f"[run_ood_eval] --skip-base set; base_pred will be empty strings", flush=True)
        base_extracted = [""] * len(rows)

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
