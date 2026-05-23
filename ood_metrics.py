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
