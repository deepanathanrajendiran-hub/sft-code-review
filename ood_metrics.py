"""Six metrics for OOD code-review evaluation against SWE-CARE labels.

Pure-Python; no GPU required. Importable from notebooks or callable as CLI.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---- location extraction ----

# Matches `path/to/file.py:42`  or `path/to/file.py`
_FILE_LINE_RE = re.compile(
    r"`([A-Za-z0-9_./\-]+\.(?:py|js|jsx|ts|tsx|java|go|rb|c|cpp|h|hpp|rs))(?::(\d+))?`"
)
# Matches `identifier_or_method` — bare backticked code references
_IDENT_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{1,40})`")


def extract_locations(review: str) -> list[dict[str, Any]]:
    """Parse a free-form review into a list of {file?, line?, identifier?} dicts.

    A review may yield multiple entries. file/line pairs are extracted from
    backticked paths; bare identifiers are emitted as identifier-only entries.
    """
    if not review:
        return []
    out: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str | None, int | None]] = set()
    seen_idents: set[str] = set()

    for m in _FILE_LINE_RE.finditer(review):
        file = m.group(1)
        line = int(m.group(2)) if m.group(2) else None
        key = (file, line)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        out.append({"file": file, "line": line})

    # bare identifiers — only those NOT already captured as a file
    captured_files = {entry["file"] for entry in out}
    for m in _IDENT_RE.finditer(review):
        ident = m.group(1)
        if ident in captured_files:
            continue
        # heuristic: skip pure-numeric and very short tokens
        if ident.isdigit() or len(ident) < 3:
            continue
        if ident in seen_idents:
            continue
        seen_idents.add(ident)
        out.append({"identifier": ident})

    return out


# ---- IoU metrics ----


def _file_line_pairs(locs: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(loc["file"], loc["line"]) for loc in locs if loc.get("file") and loc.get("line")}


def _label_pairs(labels: list[dict]) -> set[tuple[str, int]]:
    return {(lbl["path"], lbl["line"]) for lbl in labels if lbl.get("path") and lbl.get("line")}


def iou_strict(pred: dict, labels: list[dict]) -> float:
    """IoU on exact (file, line) pairs between v4_pred and reference labels."""
    pred_locs = extract_locations(pred.get("v4_pred", ""))
    pred_pairs = _file_line_pairs(pred_locs)
    label_pairs = _label_pairs(labels)
    if not pred_pairs and not label_pairs:
        return 0.0
    inter = len(pred_pairs & label_pairs)
    union = len(pred_pairs | label_pairs)
    return inter / union if union else 0.0


def iou_lenient(pred: dict, labels: list[dict], line_tol: int = 5) -> float:
    """IoU with relaxed matching:
       - Same file, line within ±line_tol → match
       - Same identifier appears anywhere in label text → match
    """
    pred_locs = extract_locations(pred.get("v4_pred", ""))
    if not pred_locs and not labels:
        return 0.0

    matched_pred: set[int] = set()
    matched_label: set[int] = set()

    for i, loc in enumerate(pred_locs):
        for j, lbl in enumerate(labels):
            if j in matched_label:
                continue
            if _lenient_match(loc, lbl, line_tol):
                matched_pred.add(i)
                matched_label.add(j)
                break

    inter = len(matched_pred)
    union = len(pred_locs) + len(labels) - inter
    return inter / union if union else 0.0


def _lenient_match(loc: dict, lbl: dict, line_tol: int) -> bool:
    # file + nearby line
    if loc.get("file") and loc.get("line") and lbl.get("path") and lbl.get("line"):
        if loc["file"] == lbl["path"] and abs(loc["line"] - lbl["line"]) <= line_tol:
            return True
    # identifier appearance in label text — word-boundary match to avoid substring false-positives
    if loc.get("identifier") and lbl.get("text"):
        pattern = r"\b" + re.escape(loc["identifier"]) + r"\b"
        if re.search(pattern, lbl["text"]):
            return True
    return False

# ---- hit-rate ----


def hit_rate(pred: dict, labels: list[dict], line_tol: int = 5) -> float:
    """Fraction of reference labels caught by v4_pred (lenient line match)."""
    if not labels:
        return 0.0
    pred_locs = extract_locations(pred.get("v4_pred", ""))
    caught = 0
    for lbl in labels:
        for loc in pred_locs:
            if _lenient_match(loc, lbl, line_tol):
                caught += 1
                break
    return caught / len(labels)

# ---- hallucination rate ----

# Mirror of the stopword list in generate_traces_gemini.py and eval.ipynb Cell 5.
# Keep these three in sync when adding new stopwords.
STOPWORDS = frozenset({
    # Python keywords
    "True", "False", "None", "if", "else", "elif", "for", "while", "def",
    "class", "return", "yield", "import", "from", "as", "with", "try",
    "except", "finally", "raise", "pass", "break", "continue", "lambda",
    "global", "nonlocal", "assert", "and", "or", "not", "in", "is",
    # Common English (often backticked as code by reviewers)
    "where", "returns", "Returns", "expects", "Expects", "because", "however",
    "therefore", "should", "must", "may", "might", "could", "would",
    "this", "that", "these", "those", "such", "etc",
    # Typing module
    "Optional", "Any", "List", "Dict", "Tuple", "Set", "Union", "Iterator",
    "Iterable", "Callable", "Awaitable", "Generator", "TypeVar", "Generic",
    "Type", "ClassVar", "Final", "Literal", "Protocol", "Annotated",
    # Exceptions
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "NotImplementedError", "FileNotFoundError",
    "ImportError", "AssertionError", "StopIteration", "ZeroDivisionError",
    # Common third-party API surfaces (false-positive prone)
    "JSONResponse", "JWTError", "UploadFile", "Request", "Response",
    "Path", "Query", "Body", "Form", "Field", "BaseModel",
    "HTTPException", "Depends", "Header", "Cookie",
    # ML libs
    "torch", "numpy", "pandas", "sklearn", "Tensor", "DataFrame", "Series",
    "Module", "Linear", "Conv2d", "BatchNorm", "Dropout", "ReLU",
    # Test / framework
    "pytest", "fixture", "mock", "Mock", "patch", "unittest", "TestCase",
    "self", "cls", "args", "kwargs",
    # Generic
    "data", "value", "result", "obj", "item", "key", "name", "type",
    "id", "code", "text", "url", "path", "file", "line", "row", "col",
    # Boolean-y words
    "true", "false", "null", "nil", "yes", "no", "ok",
    # Python builtins
    "str", "int", "float", "bool", "bytes", "list", "dict", "set", "tuple",
    "len", "range", "print", "open", "super", "property", "staticmethod",
    "classmethod", "iter", "next", "map", "filter", "sorted", "enumerate",
    "zip", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "repr", "hash",
})


def _diff_identifiers(diff: str) -> set[str]:
    """Extract identifiers from + and - lines of a diff."""
    idents: set[str] = set()
    for line in diff.splitlines():
        if not line or line[0] not in "+-":
            continue
        if line.startswith(("+++", "---")):
            continue
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,40}", line):
            idents.add(tok)
    return idents


def hallucination_rate(pred: dict) -> float:
    """Fraction of backticked identifiers in v4_pred that don't appear in the diff
    (and aren't in STOPWORDS).

    Returns the per-instance rate: hallucinated / total_backticked.
    """
    review = pred.get("v4_pred", "")
    diff = pred.get("diff", "")
    if not review:
        return 0.0

    diff_idents = _diff_identifiers(diff)
    backticked = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]{1,40})`", review))
    if not backticked:
        return 0.0

    candidates = backticked - STOPWORDS
    if not candidates:
        return 0.0

    hallucinated = {c for c in candidates if c not in diff_idents}
    return len(hallucinated) / len(backticked)

# ---- breakdowns ----


def breakdown_by_difficulty(
    preds: list[dict],
    labels_by_instance: list[list[dict]],
    metric_fn,
) -> dict[str, float]:
    """Run metric_fn per-instance and average per difficulty bucket."""
    return _groupby_avg(preds, labels_by_instance, metric_fn, key_field="difficulty")


def breakdown_by_problem_domain(
    preds: list[dict],
    labels_by_instance: list[list[dict]],
    metric_fn,
) -> dict[str, float]:
    """Run metric_fn per-instance and average per problem_domain bucket."""
    return _groupby_avg(preds, labels_by_instance, metric_fn, key_field="problem_domain")


def _groupby_avg(preds, labels_by_instance, metric_fn, key_field) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for pred, labels in zip(preds, labels_by_instance):
        bucket = pred.get(key_field, "unknown")
        groups.setdefault(bucket, []).append(metric_fn(pred, labels))
    return {k: sum(v) / len(v) for k, v in groups.items()}


# ---- pairwise (Haiku 3-vote majority) ----

PAIRWISE_PROMPT = """You are comparing two code reviews. Pick the better one.

DIFF:
{diff}

REVIEW A:
{a}

REVIEW B:
{b}

Which review is better at identifying real issues? Respond with exactly "A" or "B".
"""


def _haiku_vote(diff: str, a: str, b: str, api_key: str) -> str:
    """Single judge vote. Returns "A" or "B" (or "tie" if unclear)."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": PAIRWISE_PROMPT.format(diff=diff[:8000], a=a, b=b),
        }],
    )
    text = msg.content[0].text.strip().upper()
    if text.startswith("A"):
        return "v4"  # caller will pass v4 as A
    if text.startswith("B"):
        return "base"
    return "tie"


def pairwise_win(preds: list[dict], api_key: str, n_votes: int = 3) -> dict:
    """3-vote majority pairwise: v4_pred (A) vs base_pred (B). Returns aggregate dict."""
    v4_wins = 0
    base_wins = 0
    ties = 0
    for pred in preds:
        votes = [
            _haiku_vote(pred["diff"], pred["v4_pred"], pred["base_pred"], api_key)
            for _ in range(n_votes)
        ]
        v4_count = sum(1 for v in votes if v == "v4")
        base_count = sum(1 for v in votes if v == "base")
        if v4_count > base_count:
            v4_wins += 1
        elif base_count > v4_count:
            base_wins += 1
        else:
            ties += 1
    total = len(preds)
    return {
        "total": total,
        "v4_wins": v4_wins,
        "base_wins": base_wins,
        "ties": ties,
        "win_rate": v4_wins / total if total else 0.0,
    }


# ---- CLI ----


def _load_jsonl(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="ood_preds.jsonl")
    ap.add_argument("--labels", required=True, help="ood_input.jsonl")
    ap.add_argument("--output", default="ood_eval_results.json")
    ap.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    ap.add_argument("--skip-pairwise", action="store_true")
    args = ap.parse_args()

    preds = _load_jsonl(args.preds)
    inputs = _load_jsonl(args.labels)
    # Align by instance_id
    by_id = {row["instance_id"]: row for row in inputs}
    labels_by_instance = [by_id[p["instance_id"]]["reference_comments"] for p in preds]

    results: dict[str, Any] = {
        "n_instances": len(preds),
        "iou_strict_mean": sum(iou_strict(p, l) for p, l in zip(preds, labels_by_instance)) / len(preds),
        "iou_lenient_mean": sum(iou_lenient(p, l) for p, l in zip(preds, labels_by_instance)) / len(preds),
        "hit_rate_mean": sum(hit_rate(p, l) for p, l in zip(preds, labels_by_instance)) / len(preds),
        "hallucination_rate_mean": sum(hallucination_rate(p) for p in preds) / len(preds),
        "iou_lenient_by_difficulty": breakdown_by_difficulty(preds, labels_by_instance, iou_lenient),
        "iou_lenient_by_problem_domain": breakdown_by_problem_domain(preds, labels_by_instance, iou_lenient),
        "hit_rate_by_difficulty": breakdown_by_difficulty(preds, labels_by_instance, hit_rate),
    }

    if not args.skip_pairwise:
        if not args.api_key:
            raise SystemExit("Need --api-key or ANTHROPIC_API_KEY for pairwise")
        results["pairwise"] = pairwise_win(preds, args.api_key)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[ood_metrics] wrote {args.output}")


if __name__ == "__main__":
    main()
