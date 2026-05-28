"""Extract clean, grounded defect tuples from raw PR review threads (v5 Stage 1).

SWE-CARE `reference_comments` are raw PR discussion threads — `@user:` questions,
`@author:` replies, ```suggestion``` blocks, "good point / done" — NOT clean defect
labels. This module classifies each thread comment as a real defect vs. a
question/style/reply, and emits a clean tuple {path, line, issue_type, canonical_desc}
for the real ones. The result is the judge-independent label set the v5 verifiable
reward (corpo_reward.verifiable_reward) and the eval asset both consume.

Grounding rules (so the reward only credits defects the model can actually see):
  - drop comments whose `path` is absent from the diff (uncatchable from the diff alone)
  - drop comments with no `path`

The classifier (`extract_fn`) is injectable so all assembly/grounding logic is unit
tested with no API. The default `_extract_fn` is the DeepSeek V4-Pro classifier; the
extraction is GROUNDED in the existing human comment text (it classifies/condenses an
existing comment — it does NOT free-generate new issues, which would rebuild a gameable
judge with extra steps).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def diff_paths(diff: str) -> set[str]:
    """File paths touched by a unified diff (from the `+++ b/...` / `--- a/...` headers)."""
    paths: set[str] = set()
    for line in diff.splitlines():
        if line.startswith(("+++ ", "--- ")):
            p = line[4:].strip()
            if p in ("/dev/null", ""):
                continue
            if p.startswith(("a/", "b/")):
                p = p[2:]
            paths.add(p)
    return paths


def parse_extraction(raw: str) -> dict | None:
    """Parse a classifier response into {issue_type, canonical_desc} or None.

    None means 'not a real defect' (question/style/reply) or unparseable. A defect
    must have is_defect truthy AND both issue_type and canonical_desc present.
    """
    if not raw:
        return None
    text = raw.strip()
    # strip ```json ... ``` fences if present
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or not obj.get("is_defect"):
        return None
    issue_type = obj.get("issue_type")
    canonical_desc = obj.get("canonical_desc")
    if not issue_type or not canonical_desc:
        return None
    return {"issue_type": issue_type, "canonical_desc": canonical_desc}


def extract_defects_for_record(record: dict, extract_fn=None) -> dict:
    """Return {instance_id, defects:[{path,line,issue_type,canonical_desc}, ...]} for one record.

    Grounding (path-in-diff) is checked BEFORE the classifier so ungrounded comments
    cost no API call. An empty defects list is a clean diff (the model should find nothing).
    """
    fn = extract_fn or _extract_fn
    paths = diff_paths(record.get("diff", ""))
    defects: list[dict] = []
    for c in record.get("reference_comments", []):
        path = c.get("path")
        if not path or path not in paths:
            continue  # ungrounded — uncatchable from the diff
        res = fn(record["diff"], c)
        if not res:
            continue  # classified non-defect
        defects.append({
            "path": path,
            "line": c.get("line"),
            "issue_type": res["issue_type"],
            "canonical_desc": res["canonical_desc"],
        })
    return {"instance_id": record["instance_id"], "defects": defects}


def build_extraction_prompt(diff: str, comment: dict) -> str:
    return (
        "You are triaging a single PR review-thread comment into a clean defect label.\n"
        "Decide whether the comment reports a REAL code defect the author should fix "
        "(bug, security, resource leak, correctness, performance, api-contract) — as opposed "
        "to a question, a style nit, a praise/ack, or a pure reply.\n\n"
        f"DIFF (truncated):\n{diff[:4000]}\n\n"
        f"COMMENT (file {comment.get('path')}, line {comment.get('line')}):\n{comment.get('text', '')[:1500]}\n\n"
        "Respond with ONLY a JSON object:\n"
        '{\"is_defect\": true|false, \"issue_type\": \"bug|security|resource_leak|correctness|performance|api_contract|style\", '
        '\"canonical_desc\": \"one concise sentence describing the issue, grounded in the comment\"}\n'
        "Set is_defect=false for questions, style nits, acknowledgements, or replies. "
        "Do NOT invent issues not present in the comment."
    )


def _deepseek_extract(diff: str, comment: dict) -> dict | None:
    """Default classifier: DeepSeek V4-Pro, grounded in the existing comment text."""
    from ood_metrics import _get_deepseek_client

    client = _get_deepseek_client()
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": build_extraction_prompt(diff, comment)}],
        max_tokens=200,
        temperature=0.0,
    )
    return parse_extraction(resp.choices[0].message.content)


# Module-level binding so tests can patch the classifier (mirrors corpo_reward._judge_fn).
_extract_fn = _deepseek_extract


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with {instance_id, diff, reference_comments}")
    ap.add_argument("--output", required=True, help="JSONL {instance_id, defects:[...]} per line")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).open() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[label_defects] extracting defects from {len(rows)} records", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        labels = list(ex.map(extract_defects_for_record, rows))

    n_defects = sum(len(r["defects"]) for r in labels)
    n_clean = sum(1 for r in labels if not r["defects"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in labels:
            fh.write(json.dumps(r) + "\n")
    print(
        f"[label_defects] wrote {len(labels)} records, {n_defects} defect tuples, "
        f"{n_clean} clean diffs -> {out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
