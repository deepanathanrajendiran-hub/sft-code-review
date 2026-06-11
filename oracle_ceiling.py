"""Diff-only oracle ceiling: can a frontier model find the labeled defects from the diff alone?

Answers the question the recall plateau left open. A frontier model (DeepSeek V4-Pro,
thinking enabled) reviews the SAME diffs with the SAME prompt and diff budget the student
models saw, and is scored with the SAME semantic matcher against the SAME labels. Its
recall is the ceiling of the task-as-posed:

  - oracle recall >> student recall  -> headroom exists at diff-only; the student is
    training- or capability-limited and recall-targeted work is justified.
  - oracle recall ~= student recall  -> the task-as-posed (diff-only context) is the
    ceiling; no training method fixes it — the lever is more context, not more RL.

Also re-scores everything on style-filtered labels (the label census found 25/373 tuples
are style nits that leaked through the classifier), so the ceiling is measured against
labels worth hitting.

Usage (no GPU; needs DEEPSEEK_API_KEY):
    python oracle_ceiling.py \
        --input  ood_preds_v4.jsonl       # provides instance_id, diff, v4_pred \
        --v5-preds ood_preds_v5.jsonl     # optional; v5 output lives in its 'v4_pred' field \
        --labels defect_labels_eval.jsonl \
        --output oracle_preds.jsonl       # resumable: re-running skips generated records

Smoke-test cost first with --limit 20.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from defect_match import _with_retries

# Same task framing + diff budget the student models were evaluated with
# (run_ood_eval.SYSTEM_MSG / USER_TEMPLATE / 12000-char budgeted truncation).
ORACLE_SYSTEM = "You are a Senior Software Engineer reviewing code changes. Provide clear, actionable feedback."
ORACLE_USER = "Review the following code diff and provide feedback:\n```diff\n{diff}\n```"
DIFF_CHAR_BUDGET = 12000


def _deepseek_review(diff: str, max_tokens: int = 6000) -> str:
    """One diff-only review from V4-Pro. Thinking left ENABLED (default) on purpose —
    the oracle should get its best shot; we measure the ceiling, not the latency.

    max_tokens must cover the REASONING tokens too (DeepSeek bills thinking against
    it); a small budget gets exhausted mid-think and content comes back empty — the
    bug that zeroed the first smoke test. If content is still empty, retry once with
    double the budget.
    """
    from ood_metrics import _get_deepseek_client, DEEPSEEK_V4_PRO

    client = _get_deepseek_client()

    def _call(budget: int) -> str:
        resp = _with_retries(lambda: client.chat.completions.create(
            model=DEEPSEEK_V4_PRO,
            max_tokens=budget,
            messages=[
                {"role": "system", "content": ORACLE_SYSTEM},
                {"role": "user", "content": ORACLE_USER.format(diff=diff[:DIFF_CHAR_BUDGET])},
            ],
        ))
        return (resp.choices[0].message.content or "").strip()

    review = _call(max_tokens)
    if not review:
        review = _call(max_tokens * 2)
    return review


# module-level binding so tests can patch the reviewer (project convention)
_review_fn = _deepseek_review


def filter_style(labels: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Drop style-type tuples. A record whose defects all drop becomes clean ([]) —
    the thread only discussed style, so 'find nothing' is the right target there."""
    return {
        iid: [d for d in defects if d.get("issue_type") != "style"]
        for iid, defects in labels.items()
    }


def generate_oracle(rows: list[dict], out_path, workers: int = 8,
                    max_tokens: int = 6000, review_fn=None) -> Path:
    """Generate oracle reviews for rows missing from out_path (resumable append).

    Records whose stored oracle_pred is EMPTY are treated as not-done and
    regenerated — the first smoke test wrote empty reviews (thinking-budget
    exhaustion) and those must not be skipped on resume.
    """
    fn = review_fn or _review_fn
    out = Path(out_path)
    done: set[str] = set()
    kept_lines: list[str] = []
    if out.exists():
        for line in out.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            if (rec.get("oracle_pred") or "").strip():
                done.add(rec["instance_id"])
                kept_lines.append(line if line.endswith("\n") else line + "\n")
        # rewrite without the empty records so the file stays one-row-per-instance
        if len(kept_lines) < sum(1 for l in out.open() if l.strip()):
            out.write_text("".join(kept_lines))
    todo = [r for r in rows if r["instance_id"] not in done]
    print(f"[oracle] {len(done)} already generated, {len(todo)} to go", file=sys.stderr)
    if not todo:
        return out
    n_empty = 0
    with out.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        for row, review in zip(todo, ex.map(lambda r: fn(r["diff"], max_tokens), todo)):
            if not (review or "").strip():
                n_empty += 1
            fh.write(json.dumps({
                "instance_id": row["instance_id"],
                "diff": row["diff"],
                "oracle_pred": review,
            }) + "\n")
            fh.flush()  # line-by-line so an interrupt loses at most one record
    if n_empty:
        print(f"[oracle] WARNING: {n_empty}/{len(todo)} reviews came back EMPTY even after "
              f"the doubled-budget retry — raise --max-tokens and re-run (they'll regenerate)",
              file=sys.stderr)
    return out


def _load(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def main():
    import score_v5

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ood_preds_v4.jsonl (instance_id, diff, v4_pred)")
    ap.add_argument("--v5-preds", default=None, help="ood_preds_v5.jsonl (model output in 'v4_pred')")
    ap.add_argument("--labels", required=True, help="defect_labels_eval.jsonl")
    ap.add_argument("--output", required=True, help="oracle_preds.jsonl (resumable)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=6000,
                    help="MUST cover DeepSeek's reasoning tokens too; small budgets "
                         "exhaust mid-think and return empty reviews")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on first N rows")
    ap.add_argument("--score-only", action="store_true", help="skip generation; score existing --output")
    args = ap.parse_args()

    rows = _load(args.input)
    if args.limit:
        rows = rows[: args.limit]
    keep = {r["instance_id"] for r in rows}

    labels_raw: dict[str, list[dict]] = {}
    for r in _load(args.labels):
        labels_raw[r["instance_id"]] = r.get("defects", [])
    labels_nostyle = filter_style(labels_raw)
    n_style = sum(len(v) for v in labels_raw.values()) - sum(len(v) for v in labels_nostyle.values())
    print(f"[oracle] labels: {sum(len(v) for v in labels_raw.values())} tuples "
          f"({n_style} style tuples dropped in the filtered view)", file=sys.stderr)

    if not args.score_only:
        generate_oracle(rows, args.output, workers=args.workers, max_tokens=args.max_tokens)
    oracle_rows = [r for r in _load(args.output) if r["instance_id"] in keep]

    contenders = [("v4", rows, "v4_pred"), ("oracle", oracle_rows, "oracle_pred")]
    if args.v5_preds:
        # restrict v5 preds to the same subset so smoke-test rows are comparable
        v5_rows = [r for r in _load(args.v5_preds) if r["instance_id"] in keep]
        contenders.insert(1, ("v5.2", v5_rows, "v4_pred"))

    for label_name, labels in (("raw labels", labels_raw), ("style-filtered", labels_nostyle)):
        print(f"\n=== {label_name} ===")
        print(f"{'model':8s} {'recall':>8} {'precision':>10} {'fp_rate':>8} {'halluc':>8} {'n_labeled':>10}")
        for name, preds, field in contenders:
            s = score_v5.score(preds, labels, field)
            def _f(x):
                return f"{x:.3f}" if x is not None else "  n/a"
            print(f"{name:8s} {_f(s['defect_recall_labeled']):>8} {_f(s['precision_labeled']):>10} "
                  f"{_f(s['fp_rate_clean']):>8} {s['halluc_mean']:>8.3f} {s['n_labeled']:>6}/{s['n_total']}")

    print(
        "\nREAD: if oracle recall >> student recall, the task has headroom at diff-only and\n"
        "recall-targeted training/data is justified. If oracle ~= student, diff-only context\n"
        "is the ceiling — the lever is more context (PR description, files), not more RL.\n"
        "Oracle fp_rate is also informative: a high oracle fp on 'clean' records suggests\n"
        "label incompleteness rather than over-flagging.",
    )


if __name__ == "__main__":
    main()
