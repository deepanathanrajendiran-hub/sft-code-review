"""Six metrics for OOD code-review evaluation against SWE-CARE labels.

Pure-Python; no GPU required. Importable from notebooks or callable as CLI.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import threading

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
        return 1.0  # vacuous agreement — model correctly said nothing on a clean diff
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
        return 1.0  # vacuous agreement — model correctly said nothing on a clean diff

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
        return 1.0  # vacuous — model correctly didn't flag a clean diff
    pred_locs = extract_locations(pred.get("v4_pred", ""))
    caught = 0
    for lbl in labels:
        for loc in pred_locs:
            if _lenient_match(loc, lbl, line_tol):
                caught += 1
                break
    return caught / len(labels)

def hit_rate_strict(pred: dict, labels: list[dict]) -> float:
    """Fraction of reference labels caught by v4_pred (strict exact file+line match)."""
    if not labels:
        return 1.0  # vacuous — model correctly didn't flag a clean diff
    pred_locs = extract_locations(pred.get("v4_pred", ""))
    pred_pairs = _file_line_pairs(pred_locs)
    caught = sum(
        1 for lbl in labels
        if lbl.get("path") and lbl.get("line") and (lbl["path"], lbl["line"]) in pred_pairs
    )
    return caught / len(labels)

# ---- hallucination rate ----

# Mirror of the stopword list in eval.ipynb Cell 5 (_STOP).
# Keep these in sync when adding new stopwords.
STOPWORDS = frozenset({
    # Python keywords (≥3 chars; shorter ones already filtered by _IDENT_RE min length)
    'self', 'return', 'import', 'class', 'with', 'from', 'def',
    'the', 'and', 'for', 'not', 'this', 'that',
    'try', 'except', 'finally', 'raise', 'async', 'await', 'assert',
    'lambda', 'yield', 'pass', 'elif', 'else', 'while', 'break',
    'continue', 'global', 'nonlocal',
    # Built-in primitives
    'True', 'False', 'None',
    'bool', 'bytes', 'bytearray', 'complex', 'dict', 'float', 'frozenset',
    'int', 'list', 'object', 'set', 'str', 'tuple', 'type',
    # Standard exception types
    'BaseException', 'Exception',
    'ArithmeticError', 'AssertionError', 'AttributeError', 'BufferError', 'EOFError',
    'FloatingPointError', 'GeneratorExit', 'ImportError', 'ModuleNotFoundError',
    'IndexError', 'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError',
    'NameError', 'NotImplementedError', 'OSError', 'OverflowError',
    'RecursionError', 'ReferenceError', 'RuntimeError',
    'StopAsyncIteration', 'StopIteration',
    'SyntaxError', 'IndentationError', 'TabError',
    'SystemError', 'SystemExit',
    'TypeError', 'UnboundLocalError', 'ValueError', 'ZeroDivisionError',
    'UnicodeError', 'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeTranslateError',
    'ConnectionError', 'ConnectionAbortedError', 'ConnectionRefusedError', 'ConnectionResetError',
    'BlockingIOError', 'BrokenPipeError', 'ChildProcessError',
    'FileExistsError', 'FileNotFoundError', 'InterruptedError',
    'IsADirectoryError', 'NotADirectoryError', 'PermissionError',
    'ProcessLookupError', 'TimeoutError',
    # typing-module vocabulary
    'Optional', 'Union', 'Callable', 'Any', 'Annotated', 'Literal',
    'Iterable', 'Iterator', 'AsyncIterable', 'AsyncIterator',
    'Generator', 'AsyncGenerator', 'Coroutine', 'Awaitable',
    'Sequence', 'MutableSequence', 'Mapping', 'MutableMapping',
    'List', 'Dict', 'Tuple', 'Set', 'FrozenSet',
    'Type', 'ClassVar', 'TypeVar', 'Generic', 'Final',
    'Protocol', 'NamedTuple', 'TypedDict', 'NewType', 'cast', 'overload',
    # Common builtins reviewers reference as vocabulary
    'print', 'len', 'range', 'super', 'property',
    'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr',
    'staticmethod', 'classmethod', 'callable',
    # ML library names
    'huggingface_hub', 'transformers', 'accelerate', 'datasets',
    'torch', 'torchvision', 'pytorch', 'tensorflow', 'jax', 'flax',
    'numpy', 'pandas', 'sklearn', 'scipy', 'pytest', 'sentencepiece',
    # Universal ML attribute vocabulary
    'batch_size', 'num_channels', 'attention_mask', 'pixel_values',
    'return_tensors', 'input_ids', 'hidden_states', 'logits', 'embeddings',
    # Common framework method/attribute names
    'outputs', 'headers', 'forward', 'generate', 'encode', 'decode',
    # Dunders (comprehensive — these are Python protocol methods, never hallucinations)
    '__call__', '__init__', '__new__', '__del__', '__post_init__',
    '__repr__', '__str__', '__bytes__', '__format__', '__hash__', '__bool__',
    '__len__', '__length_hint__', '__sizeof__',
    '__iter__', '__next__', '__reversed__',
    '__getitem__', '__setitem__', '__delitem__', '__contains__', '__missing__',
    '__getattr__', '__getattribute__', '__setattr__', '__delattr__', '__dir__',
    '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
    '__add__', '__radd__', '__iadd__', '__sub__', '__rsub__', '__isub__',
    '__mul__', '__rmul__', '__imul__', '__truediv__', '__rtruediv__', '__itruediv__',
    '__floordiv__', '__rfloordiv__', '__mod__', '__rmod__',
    '__pow__', '__rpow__', '__matmul__', '__rmatmul__',
    '__and__', '__or__', '__xor__', '__lshift__', '__rshift__',
    '__neg__', '__pos__', '__abs__', '__invert__', '__round__',
    '__int__', '__float__', '__complex__', '__index__',
    '__enter__', '__exit__', '__aenter__', '__aexit__',
    '__aiter__', '__anext__', '__await__',
    '__copy__', '__deepcopy__',
    '__getstate__', '__setstate__', '__reduce__', '__reduce_ex__', '__getnewargs__',
    '__instancecheck__', '__subclasscheck__', '__init_subclass__', '__class_getitem__',
    '__prepare__', '__set_name__', '__match_args__',
    '__base__', '__bases__', '__class__', '__dict__', '__name__', '__module__',
    '__mro__', '__qualname__', '__subclasses__', '__doc__', '__weakref__',
    '__slots__', '__annotations__', '__defaults__', '__kwdefaults__',
    '__file__', '__path__', '__package__', '__loader__', '__spec__',
    '__all__', '__author__', '__version__',
    # FastAPI / Pydantic / stdlib / pytest vocabulary
    'APIRoute', 'Response', 'Request',
    'validation_alias', 'serialization_alias',
    'PurePath', 'Path', 'PathLike',
    'HTTPConnection', 'HTTPSConnection',
    'pytest_runtest_setup', 'pytest_runtest_call', 'pytest_collection_modifyitems',
    # English words that look like identifiers in backticks
    'where', 'when', 'whenever', 'because', 'then', 'after', 'before',
    'using', 'via', 'uses', 'without', 'inside', 'into', 'contains',
    'becomes', 'has', 'have', 'being', 'causes', 'likely', 'something',
    'expects', 'accepts', 'returns',
    'here', 'there', 'must', 'should', 'would', 'could', 'note', 'also',
    'still', 'under', 'over', 'above', 'below', 'between', 'during',
    'against', 'unlike', 'like',
    # Standard-library + commonly-discussed third-party API names
    'weakref', 'urllib', 'urllib3', 'requests', 'httpx', 'aiohttp',
    'starlette', 'redoc', 'swagger', 'openapi', 'asgi', 'wsgi',
    'redoc_url', 'swagger_url', 'openapi_url', 'docs_url',
    'encode_multipart_formdata', 'multipart_formdata',
    'werkzeug', 'flask', 'django', 'fastapi', 'uvicorn', 'gunicorn',
    'pydantic_core', 'email_validator',
    'JWTError', 'UploadFile', 'get_type_hints',
    'checks', 'sets', 'default', 'redirect_slashes',
    # More stdlib modules
    'argparse', 'logging', 'warnings', 'inspect', 'traceback',
    'importlib', 'pkgutil', 'pkg_resources',
    'subprocess', 'multiprocessing', 'threading', 'asyncio', 'queue',
    'json', 'pickle', 'copy', 'os', 'sys', 're',
    'time', 'datetime', 'calendar', 'random', 'math', 'decimal', 'statistics',
    'enum', 'abc', 'contextlib', 'tempfile', 'shutil', 'glob', 'fnmatch',
    'dataclasses', 'functools', 'itertools', 'collections',
    'pathlib', 'platform', 'socket', 'select', 'selectors',
    'http', 'ftplib', 'smtplib', 'email',
    'xml', 'csv', 'configparser',
    'hashlib', 'hmac', 'secrets', 'ssl',
    'gzip', 'zlib', 'bz2', 'lzma', 'zipfile', 'tarfile',
    'io', 'codecs', 'locale', 'gettext',
    'typing', 'typing_extensions',
    'getpass', 'pwd', 'grp',
    'unittest', 'mock', 'pytest',
    # Common third-party libraries
    'sqlalchemy', 'alembic', 'sqlmodel',
    'celery', 'redis', 'memcached',
    'jinja', 'jinja2', 'mako',
    'click', 'typer', 'rich', 'loguru', 'structlog',
    'attrs', 'attr', 'cattrs',
    'marshmallow', 'wtforms',
    'cryptography', 'bcrypt', 'passlib', 'jwt', 'pyjwt', 'PyJWT',
    'setuptools', 'pip', 'poetry', 'hatch', 'flit',
    'tox', 'nox', 'mypy', 'ruff', 'black', 'isort', 'flake8', 'pylint', 'bandit',
    'coverage', 'hypothesis',
    'xarray', 'polars', 'pyarrow', 'dask',
    'xgboost', 'lightgbm', 'catboost', 'optuna',
    'wandb', 'mlflow', 'tensorboard',
    'vllm', 'unsloth', 'peft', 'trl', 'bitsandbytes',
    'tiktoken', 'tokenizers',
    'langchain', 'llamaindex', 'chromadb', 'pinecone', 'qdrant', 'faiss',
    'openai', 'anthropic', 'cohere',
    'gradio', 'streamlit',
    # Pytest fixtures / common test vars (frequently backticked in reviews)
    'caplog', 'mocker', 'monkeypatch', 'tmpdir', 'tmp_path',
    'capsys', 'capfd', 'recwarn', 'request',
    # Generic placeholder names (clearly not project-specific identifiers)
    'obj', 'val', 'var', 'tmp', 'temp', 'idx', 'src', 'dst',
    'msg', 'err', 'exc', 'res', 'ctx', 'env', 'cfg', 'conf',
    'cls', 'fn', 'func',
})


def _diff_identifiers(diff: str) -> set[str]:
    """Extract identifiers from a diff hunk — INCLUDING context lines and @@ headers.

    A reviewer can validly reference any symbol visible in the diff context
    (surrounding function names, imports, unchanged code referenced in the change).
    Restricting to +/- lines only under-counts the legitimate symbol pool and
    inflates the hallucination rate on PRs where v4 references context symbols.
    """
    idents: set[str] = set()
    in_hunk = False
    for line in diff.splitlines():
        if not line:
            continue
        # @@ headers often include the enclosing function name (e.g. "@@ ... def foo(...)").
        # Capture identifiers from them too.
        if line.startswith("@@"):
            in_hunk = True
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,40}", line):
                idents.add(tok)
            continue
        # Skip file-level diff metadata lines (paths, hashes, mode bits, etc.).
        if line.startswith(("+++", "---", "diff ", "index ", "new file", "deleted file",
                            "similarity index", "rename ", "Binary files", "GIT binary")):
            continue
        if in_hunk:
            # Include +, -, and " " (space-prefixed context) lines. Skip the rare "\" lines
            # (no-newline-at-end-of-file markers) which carry no useful identifiers.
            if line[0] in "+- ":
                for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,40}", line):
                    idents.add(tok)
    return idents


def hallucination_rate(pred: dict) -> float:
    """Fraction of backticked identifiers in v4_pred that don't appear in the diff
    (and aren't in STOPWORDS).

    Returns the per-instance rate: hallucinated / candidates (non-stopword backticks).
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
    return len(hallucinated) / len(candidates)

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


# ---- pairwise (Haiku 3-vote majority — production judge, matches eval.ipynb Cell 6) ----

HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

_judge_client = None
_judge_client_lock = threading.Lock()


def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        with _judge_client_lock:
            # Double-checked locking — recheck inside the lock
            if _judge_client is None:
                import anthropic
                _judge_client = anthropic.AnthropicBedrock(aws_region="us-west-2")
    return _judge_client


def haiku_pairwise_judge(review_a: str, review_b: str, diff: str, reference: str) -> str:
    """One call. Randomized A/B order. Returns 'A', 'B', or 'TIE'."""
    swap = random.random() > 0.5
    ra, rb = (review_b, review_a) if swap else (review_a, review_b)

    resp = _get_judge_client().messages.create(
        model=HAIKU_MODEL,
        max_tokens=4,
        messages=[{"role": "user", "content": (
            "Compare two code reviews for the same diff. Which is better?\n\n"
            f"DIFF:\n{diff[:2000]}\n\n"
            f"REFERENCE (expert review):\n{reference[:500]}\n\n"
            f"REVIEW A:\n{ra[:500]}\n\n"
            f"REVIEW B:\n{rb[:500]}\n\n"
            "Criteria: accuracy, actionability, specificity, relevance.\n"
            "Reply ONLY: A, B, or TIE"
        )}],
    )
    result = resp.content[0].text.strip().upper()
    if "TIE" in result:
        result = "TIE"
    elif "A" in result and "B" not in result:
        result = "A"
    elif "B" in result and "A" not in result:
        result = "B"
    else:
        result = "TIE"
    if swap:
        if result == "A": result = "B"
        elif result == "B": result = "A"
    return result


def haiku_pairwise_judge_3vote(review_a: str, review_b: str, diff: str, reference: str) -> str:
    """3-vote majority. Returns 'A', 'B', or 'TIE'."""
    votes = [haiku_pairwise_judge(review_a, review_b, diff, reference) for _ in range(3)]
    c = Counter(votes)
    top_label, top_count = c.most_common(1)[0]
    return top_label if top_count >= 2 else "TIE"


# ---- pairwise (DeepSeek V4-Flash binary judge — for CoRPO training only) ----

DEEPSEEK_V4_FLASH = "deepseek-v4-flash"

_deepseek_client = None
_deepseek_client_lock = threading.Lock()


def _get_deepseek_client():
    """Return cached OpenAI-compatible client pointed at DeepSeek API."""
    global _deepseek_client
    if _deepseek_client is None:
        with _deepseek_client_lock:
            if _deepseek_client is None:
                import openai
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "DEEPSEEK_API_KEY env var required for DeepSeek judge"
                    )
                _deepseek_client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )
    return _deepseek_client


def deepseek_v4flash_pairwise_judge(
    review_a: str, review_b: str, diff: str, reference: str
) -> str:
    """One call. Randomized A/B order. Returns 'A', 'B', or 'TIE'.

    Mirrors haiku_pairwise_judge() prompt + truncations for protocol parity.
    """
    swap = random.random() > 0.5
    ra, rb = (review_b, review_a) if swap else (review_a, review_b)

    client = _get_deepseek_client()
    resp = client.chat.completions.create(
        model=DEEPSEEK_V4_FLASH,
        max_tokens=4,
        messages=[{"role": "user", "content": (
            "Compare two code reviews for the same diff. Which is better?\n\n"
            f"DIFF:\n{diff[:2000]}\n\n"
            f"REFERENCE (expert review):\n{reference[:500]}\n\n"
            f"REVIEW A:\n{ra[:500]}\n\n"
            f"REVIEW B:\n{rb[:500]}\n\n"
            "Criteria: accuracy, actionability, specificity, relevance.\n"
            "Reply ONLY: A, B, or TIE"
        )}],
    )
    result = resp.choices[0].message.content.strip().upper()
    if "TIE" in result:
        result = "TIE"
    elif "A" in result and "B" not in result:
        result = "A"
    elif "B" in result and "A" not in result:
        result = "B"
    else:
        result = "TIE"
    if swap:
        if result == "A": result = "B"
        elif result == "B": result = "A"
    return result


def bootstrap_winrate_ci(verdicts: list[str], which: str = "A", n_iter: int = 2000, ci: int = 95) -> tuple[float, float, float]:
    """Returns (mean_winrate, ci_lo, ci_hi)."""
    rng = np.random.default_rng(42)
    indicator = np.array([1 if v == which else 0 for v in verdicts])
    N = len(indicator)
    if N == 0:
        return 0.0, 0.0, 0.0
    samples = np.array([indicator[rng.integers(0, N, N)].mean() for _ in range(n_iter)])
    alpha = (100 - ci) / 2
    lo = float(np.percentile(samples, alpha))
    hi = float(np.percentile(samples, 100 - alpha))
    return float(indicator.mean()), lo, hi


def pairwise_win(
    preds: list[dict],
    api_key: str = "",  # ignored; AnthropicBedrock uses AWS credentials
    n_votes: int = 3,  # ignored; haiku_pairwise_judge_3vote hardcodes 3
    max_workers: int = 16,
) -> dict:
    """3-vote majority pairwise: v4_pred (A) vs base_pred (B) with order swap and
    reference review. Returns aggregate dict including bootstrap 95% CI.
    """
    def _vote_one(pred):
        return haiku_pairwise_judge_3vote(
            review_a=pred["v4_pred"],
            review_b=pred["base_pred"],
            diff=pred.get("diff", ""),
            reference=pred.get("reference_text", ""),
        )

    verdicts: list[str] = [None] * len(preds)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(_vote_one, p): i for i, p in enumerate(preds)}
        for fut in as_completed(future_to_idx):
            verdicts[future_to_idx[fut]] = fut.result()

    mean_a, lo_a, hi_a = bootstrap_winrate_ci(verdicts, which="A")
    mean_b, lo_b, hi_b = bootstrap_winrate_ci(verdicts, which="B")
    mean_t, lo_t, hi_t = bootstrap_winrate_ci(verdicts, which="TIE")

    total = len(preds)
    return {
        "total": total,
        "v4_wins": int(mean_a * total),
        "base_wins": int(mean_b * total),
        "ties": int(mean_t * total),
        "win_rate": mean_a,
        "win_rate_ci_lo": lo_a,
        "win_rate_ci_hi": hi_a,
        "tie_rate": mean_t,
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
    ap.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="DEPRECATED: ignored; pairwise uses AnthropicBedrock with AWS credentials",
    )
    ap.add_argument("--skip-pairwise", action="store_true")
    ap.add_argument(
        "--skip-base-calibration",
        action="store_true",
        help="Skip dual-scoring base_pred (loses base calibration but slightly faster)",
    )
    args = ap.parse_args()

    preds = _load_jsonl(args.preds)
    inputs = _load_jsonl(args.labels)
    if not preds:
        raise SystemExit(f"No predictions found in {args.preds}")
    # Align by instance_id
    by_id = {row["instance_id"]: row for row in inputs}
    labels_by_instance = [by_id[p["instance_id"]]["reference_comments"] for p in preds]

    def _score(rows, pred_field):
        """Score all per-row metrics treating `pred_field` as the subject of evaluation.

        Inline-swaps the pred_field into "v4_pred" so the existing metric functions
        (which read pred["v4_pred"]) work uniformly for v4 or base.
        """
        subbed = [{**r, "v4_pred": r.get(pred_field, "")} for r in rows]
        n = len(subbed)
        return {
            "iou_strict_mean": sum(iou_strict(p, l) for p, l in zip(subbed, labels_by_instance)) / n,
            "iou_lenient_mean": sum(iou_lenient(p, l) for p, l in zip(subbed, labels_by_instance)) / n,
            "hit_rate_mean": sum(hit_rate(p, l) for p, l in zip(subbed, labels_by_instance)) / n,
            "hit_rate_strict_mean": sum(hit_rate_strict(p, l) for p, l in zip(subbed, labels_by_instance)) / n,
            "hallucination_rate_mean": sum(hallucination_rate(p) for p in subbed) / n,
            "iou_lenient_by_difficulty": breakdown_by_difficulty(subbed, labels_by_instance, iou_lenient),
            "iou_lenient_by_problem_domain": breakdown_by_problem_domain(subbed, labels_by_instance, iou_lenient),
            "hit_rate_by_difficulty": breakdown_by_difficulty(subbed, labels_by_instance, hit_rate),
        }

    v4_metrics = _score(preds, "v4_pred")
    results: dict[str, Any] = {
        "n_instances": len(preds),
        **v4_metrics,  # v4 metrics at top level for backward compat
    }

    if not args.skip_base_calibration:
        # Dual-score base on the same OOD distribution — needed by the decision gate to
        # distinguish "v4 looks weak" from "the metric undersells everyone equally".
        results["base"] = _score(preds, "base_pred")

    if not args.skip_pairwise:
        results["pairwise"] = pairwise_win(preds, args.api_key)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[ood_metrics] wrote {args.output}")


if __name__ == "__main__":
    main()
