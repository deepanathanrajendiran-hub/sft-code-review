"""OOD eval runner for the code-review LoRAs.

Run as a CLI to generate predictions and metrics on the SWE-CARE eval set.
Also importable: corpo_reward.py reuses `_extract_review` at reward time. The
heavy deps (vLLM, transformers) are imported inside functions, so importing
this module pulls in nothing but stdlib.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any


def _extract_review(raw: str) -> str:
    """Pull the `<review>` block out of a generated trace.

    Prefer the review after the last `</think>`; otherwise the last non-`...`
    review block; otherwise strip the think blocks and return the rest.
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

    # max_model_len - max_tokens - margin. Coupled to the LLM() and
    # SamplingParams() values below; bump this if either changes.
    INPUT_TOKEN_BUDGET = 8192 - 4096 - 100  # = 3996

    def _format_with_budget(diff: str) -> str:
        """Format one diff to fit INPUT_TOKEN_BUDGET, shrinking the char cap on overflow.

        Adversarial diffs (URLs, base64, non-ASCII) can run under 2.5 chars/token,
        so even 12000 chars can blow the budget. Retry at 70% until it fits.
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
            truncated_chars = int(truncated_chars * 0.7)
        # hit the floor; hand it to vLLM anyway, it may still fit
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
    # free the GPU before the next model loads
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

    # a customized v4 chat_template would silently confound the pairwise comparison
    v4_tok = AutoTokenizer.from_pretrained(args.v4_model)
    assert v4_tok.chat_template == tokenizer.chat_template, (
        f"v4 chat_template differs from base — would silently confound pairwise. "
        f"Either rebuild v4 with matching template or load tokenizer per-model."
    )
    del v4_tok
    print(f"[run_ood_eval] chat_template matches v4_model: OK", flush=True)

    print(f"[run_ood_eval] loading v4 ({args.v4_model})", flush=True)
    v4_raw = _generate(args.v4_model, diffs, tokenizer)
    v4_extracted = [_extract_review(t) for t in v4_raw]

    if not args.skip_base:
        print(f"[run_ood_eval] loading base ({args.base_model})", flush=True)
        base_raw = _generate(args.base_model, diffs, tokenizer)
        # base has no <think>/<review> scaffolding; pass it through stripped
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
