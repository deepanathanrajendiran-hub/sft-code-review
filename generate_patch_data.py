#!/usr/bin/env python3
"""Synthetic training data for the 9 known bug patterns.

Generates 50 records per pattern, then merges them with
train_dataset_rewritten.jsonl to produce train_dataset_v3.jsonl.

Sonnet writes the diffs (better code reasoning), Haiku writes the reviews
(fast and cheap, and it already has the diff to work from).

    pip install anthropic rouge-score
    export ANTHROPIC_API_KEY=...

    python generate_patch_data.py               # generate synthetic + merge to v3
    python generate_patch_data.py --synth-only  # generate synthetic, skip merge
    python generate_patch_data.py --merge-only  # merge existing synthetic with rewritten
    python generate_patch_data.py --dry-run 2   # 2 per pattern (smoke test)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import anthropic
    from rouge_score import rouge_scorer
except ImportError:
    print("Missing deps. Run: pip install anthropic rouge-score", file=sys.stderr)
    raise

REWRITTEN_FILE   = "train_dataset_rewritten.jsonl"
SYNTHETIC_FILE   = "train_dataset_synthetic.jsonl"
V3_FILE          = "train_dataset_v3.jsonl"

SONNET_MODEL     = "claude-sonnet-4-6"
HAIKU_MODEL      = "claude-haiku-4-5-20251001"

N_PER_PATTERN    = 50
MAX_WORKERS      = 8
DEDUP_THRESHOLD  = 0.8
OVERGEN_FACTOR   = 1.4          # produce 1.4× target per pattern to cover dups/errors

DIFF_MAX_TOKENS    = 700
REVIEW_MAX_TOKENS  = 400
DIFF_TEMPERATURE   = 0.9        # high variety across diffs
REVIEW_TEMPERATURE = 0.3        # low variance on review style

PATTERNS: dict[str, str] = {
    "first_removal": (
        "Django queryset where .first() is removed, leaving code that returns a QuerySet "
        "instead of a model instance. The caller then accesses an attribute (e.g., .name, "
        ".id, .email) on the QuerySet object, causing an AttributeError at runtime."
    ),
    "magic_numbers": (
        "Python function that introduces hardcoded numeric literals (e.g., 3, 30, 100, 86400) "
        "instead of named constants (MAX_RETRIES, TIMEOUT_SECONDS, MAX_ITEMS, CACHE_TTL). "
        "The magic numbers appear in conditionals, default arguments, or sleep() calls."
    ),
    "boolean_string": (
        "Python code that passes the string literals \"true\" or \"false\" where Python "
        "booleans True/False are expected — common in config parsing, feature flags, or "
        "API request bodies. The critical bug: the string \"false\" is truthy in Python, "
        "so any boolean check silently passes."
    ),
    "django_patterns": (
        "Django view, manager, or serializer code with an ORM anti-pattern. Examples: "
        "calling .get() without catching DoesNotExist; an N+1 query (database queries "
        "inside a loop); missing select_related()/prefetch_related() on a foreign key; "
        "or unintended lazy-loading on QuerySet iteration in a template or loop."
    ),
    "flask_fastapi_security": (
        "Flask or FastAPI endpoint with a security issue. Examples: a path or query "
        "parameter used directly in a raw SQL query (SQL injection); a sensitive endpoint "
        "with no authentication decorator; user input passed to eval() or exec(); or a "
        "file upload without type/size validation."
    ),
    "js_async_bugs": (
        "JavaScript or TypeScript async/await bug. Examples: unhandled promise rejection "
        "(missing .catch() or try/await); forgetting 'await' so the function returns a "
        "Promise instead of the resolved value; race condition from unawaited concurrent "
        "writes; or forgetting to await inside a for-loop over async calls."
    ),
    "null_handling": (
        "Python function that fails to handle None correctly. Examples: dereferencing a "
        "value that can be None; mutable default argument (def f(items=[])); dict.get() "
        "used without a default where None flows into an arithmetic or comparison; or "
        "checking equality with == None instead of is None."
    ),
    "resource_leaks": (
        "Python code that opens a file, socket, database connection, or subprocess "
        "without a context manager. Examples: open(path) without close(); socket.socket() "
        "that leaks on an exception path; or a database cursor not released in a "
        "finally block."
    ),
    "no_issue_diffs": (
        "A small, clean code change with NO bug or antipattern. Examples: adding a "
        "well-named helper function; renaming a variable for clarity; fixing a docstring "
        "typo; or extracting a repeated literal into a named constant. The diff should "
        "be something a careful reviewer would approve without objections."
    ),
}

_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
_needs_api = "--merge-only" not in sys.argv

if _needs_api:
    if not _ANTHROPIC_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY")
    _anthropic = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
else:
    _anthropic = None

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

_print_lock    = threading.Lock()
_dup_count     = 0
_err_count     = 0
_first_samples_shown = 0

_DIFF_PROMPT = """Generate a realistic code diff in unified diff format for this bug pattern:

{pattern_desc}

Requirements:
- Use real-looking variable, function, and module names drawn from a production codebase (web app, data pipeline, or common library)
- Never use placeholder names like 'foo', 'bar', 'test', 'example', 'my_func', 'mymodule'
- Include an @@ hunk header with plausible line numbers and 3 lines of context
- Keep the diff concise: 15-30 lines total
- Variation seed {seed}: use a DIFFERENT module, function, and class name from any prior attempt; vary the surrounding context

Output ONLY the raw unified diff. No explanation, no markdown fences, no preamble."""


_REVIEW_PROMPT = """You are a senior software engineer writing a code review comment.

Given this diff, write a concise actionable review that:
- Identifies the specific bug or antipattern by name
- References the exact variable, function, or method from the diff
- Explains WHY it is a problem in one sentence
- Gives the specific fix with the correct code
- Is 2-4 sentences total, no bullet points, no headers

IMPORTANT: If the diff contains no bug or antipattern (the code is clean and the change is valid), write a brief review that acknowledges the change is correct and notes one concrete reason it is fine. Do NOT invent issues to fill space.

DIFF:
{diff}

Write the review:"""


def generate_diff(pattern_desc: str, seed: int, retries: int = 3) -> str:
    """Sonnet: produce a realistic diff embodying the pattern."""
    prompt = _DIFF_PROMPT.format(pattern_desc=pattern_desc, seed=seed)
    last = None
    for attempt in range(retries):
        try:
            resp = _anthropic.messages.create(
                model=SONNET_MODEL,
                max_tokens=DIFF_MAX_TOKENS,
                temperature=DIFF_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Sonnet diff-gen failed after {retries} retries: {last}")


def generate_review(diff: str, retries: int = 3) -> str:
    """Haiku: produce a concise reference review for a diff."""
    prompt = _REVIEW_PROMPT.format(diff=diff)
    last = None
    for attempt in range(retries):
        try:
            resp = _anthropic.messages.create(
                model=HAIKU_MODEL,
                max_tokens=REVIEW_MAX_TOKENS,
                temperature=REVIEW_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Haiku review-gen failed after {retries} retries: {last}")


def is_duplicate(diff: str, reference_diffs: list[str],
                 threshold: float = DEDUP_THRESHOLD) -> bool:
    for ref in reference_diffs:
        if _rouge.score(ref, diff)["rougeL"].fmeasure > threshold:
            return True
    return False


def build_one(pattern_name: str, pattern_desc: str, seed: int,
              reference_diffs: list[str]) -> dict | None:
    """Build one synthetic record. Returns None if it dup'd or errored."""
    global _dup_count, _err_count, _first_samples_shown

    try:
        diff = generate_diff(pattern_desc, seed)
    except Exception as e:
        with _print_lock:
            _err_count += 1
            if _err_count <= 3:
                print(f"  [ERR] diff-gen {pattern_name}#{seed}: {e}")
        return None

    if is_duplicate(diff, reference_diffs):
        with _print_lock:
            _dup_count += 1
        return None

    try:
        review = generate_review(diff)
    except Exception as e:
        with _print_lock:
            _err_count += 1
            if _err_count <= 3:
                print(f"  [ERR] review-gen {pattern_name}#{seed}: {e}")
        return None

    record = {
        "instruction": "Review this code change.",
        "input":       diff,
        "output":      review,
        "repo":        f"synthetic/{pattern_name}",
    }

    # dump the first few records so quality is visible early in the run
    with _print_lock:
        if _first_samples_shown < 3:
            _first_samples_shown += 1
            print(f"\n═══ sample #{_first_samples_shown} — {pattern_name} ═══")
            print(f"DIFF ({len(diff)} ch):\n{diff[:400]}")
            print(f"REVIEW ({len(review)} ch):\n{review[:400]}")
            print("═" * 60 + "\n")

    return record


def load_reference_diffs(path: str, max_records: int = 500) -> list[str]:
    """Load up to `max_records` diffs from an existing JSONL for dedup."""
    if not os.path.exists(path):
        return []
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # only the most recent N — a full scan is slow and barely adds hits
    return [r.get("input", "") for r in records[-max_records:] if r.get("input")]


def generate_synthetic(n_per_pattern: int = N_PER_PATTERN) -> list[dict]:
    """Generate `n_per_pattern` records per pattern, in parallel."""
    reference_diffs = load_reference_diffs(REWRITTEN_FILE)
    if reference_diffs:
        print(f"Dedup against {len(reference_diffs)} existing diffs (from {REWRITTEN_FILE})")
    else:
        print(f"No dedup reference (missing {REWRITTEN_FILE} — dedup is only vs each other)")

    # over-generate so dups and errors don't leave a pattern short
    tasks_per_pattern = int(n_per_pattern * OVERGEN_FACTOR) + 2
    tasks = [
        (pname, pdesc, seed)
        for pname, pdesc in PATTERNS.items()
        for seed in range(tasks_per_pattern)
    ]
    target_total = len(PATTERNS) * n_per_pattern
    print(f"Target: {target_total} ({n_per_pattern}/pattern × {len(PATTERNS)} patterns)")
    print(f"Submitting {len(tasks)} tasks with {MAX_WORKERS} workers...\n")

    results_by_pattern: dict[str, list[dict]] = {p: [] for p in PATTERNS}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(build_one, pname, pdesc, seed, reference_diffs): pname
            for pname, pdesc, seed in tasks
        }

        done = 0
        for fut in as_completed(futures):
            pname = futures[fut]
            try:
                rec = fut.result()
            except Exception:
                continue
            if rec is None:
                continue
            if len(results_by_pattern[pname]) >= n_per_pattern:
                continue
            results_by_pattern[pname].append(rec)
            done = sum(len(rs) for rs in results_by_pattern.values())
            if done % 25 == 0 or done == target_total:
                per = {p[:10]: len(rs) for p, rs in results_by_pattern.items()}
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                print(f"  [{done}/{target_total}] {rate:.1f}/s  {per}")
            if all(len(rs) >= n_per_pattern for rs in results_by_pattern.values()):
                # every pattern is full; stop waiting on the rest of the futures
                break

    for pname, rs in results_by_pattern.items():
        if len(rs) < n_per_pattern:
            print(f"  [WARN] {pname}: only {len(rs)} records (target {n_per_pattern})")

    all_records = [r for rs in results_by_pattern.values() for r in rs]
    print(
        f"\nGenerated {len(all_records)} synthetic records in {(time.time()-t0)/60:.1f} min"
        f" — {_dup_count} duplicates rejected, {_err_count} errors"
    )
    return all_records


def save_synthetic(records: list[dict], path: str = SYNTHETIC_FILE) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} synthetic records to {path}")


def merge_to_v3() -> None:
    """Combine rewritten records (minus 'dropped') + synthetic into v3."""
    if not os.path.exists(REWRITTEN_FILE):
        raise SystemExit(f"Missing {REWRITTEN_FILE} — run rewrite_data_gemini.py first")
    if not os.path.exists(SYNTHETIC_FILE):
        raise SystemExit(f"Missing {SYNTHETIC_FILE} — run with --synth-only first")

    kept, dropped = [], 0
    with open(REWRITTEN_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("rewrite_status") == "dropped":
                dropped += 1
                continue
            # training only needs the 4 core fields; drop the rest
            kept.append({
                "instruction": r.get("instruction", "Review this code change."),
                "input":       r["input"],
                "output":      r["output"],
                "repo":        r.get("repo", ""),
            })

    synthetic: list[dict] = []
    with open(SYNTHETIC_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                synthetic.append(json.loads(line))

    v3 = kept + synthetic
    with open(V3_FILE, "w") as f:
        for r in v3:
            f.write(json.dumps(r) + "\n")

    synth_by_pattern = Counter(
        r["repo"].split("/", 1)[-1] for r in synthetic
    )
    print(f"\nMerge → {V3_FILE}")
    print(f"  rewritten kept:    {len(kept)}")
    print(f"  rewritten dropped: {dropped}")
    print(f"  synthetic:         {len(synthetic)}")
    for p, n in sorted(synth_by_pattern.items()):
        print(f"    {p:<26} {n}")
    print(f"  TOTAL:             {len(v3)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    synth_only = "--synth-only" in args
    merge_only = "--merge-only" in args

    n_per_pattern = N_PER_PATTERN
    if "--dry-run" in args:
        i = args.index("--dry-run")
        if i + 1 < len(args):
            n_per_pattern = int(args[i + 1])
            print(f"[DRY RUN] generating only {n_per_pattern} per pattern")

    if not merge_only:
        records = generate_synthetic(n_per_pattern=n_per_pattern)
        save_synthetic(records)

    if not synth_only and not ("--dry-run" in args):
        merge_to_v3()
