#!/usr/bin/env python3
"""
Rewrite the clean SFT reviews into more bug-focused ones via Gemini.

Reads train_dataset_clean.jsonl and writes train_dataset_rewritten.jsonl.
Each record's `output` is either kept as-is (when already specific) or
rewritten to name a concrete bug; the human original is always kept in
`original_output`.

The model is told to return weak reviews unchanged, so only the vague ones
actually get rewritten. Three filters catch model mistakes (identifier-match,
length sanity, soft never-invent). Runs are resumable — each record is written
as it finishes, keyed by `_idx`, so re-running picks up where it stopped.

Usage:
    python rewrite_data_gemini.py                # full run (resumable)
    python rewrite_data_gemini.py --sanity       # sanity checks, then exit
    python rewrite_data_gemini.py --dry-run 20   # process first 20 only
    python rewrite_data_gemini.py --samples 10   # print 10 rewrites from existing output

Requirements:
    pip install google-genai

Auth (Vertex AI via GCP project):
    Either set a service account:
        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    Or use local gcloud ADC:
        gcloud auth application-default login

Environment:
    GOOGLE_CLOUD_PROJECT   (required) — your GCP project ID
    GOOGLE_CLOUD_LOCATION  (optional) — defaults to us-central1
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# sanity checks and --samples don't touch the SDK, so import is optional
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

INPUT_FILE        = "train_dataset_clean.jsonl"
OUTPUT_FILE       = "train_dataset_rewritten.jsonl"
MODEL_NAME        = "gemini-2.5-pro"
MAX_WORKERS       = 8              # keep low; higher counts hit rate limits on retry
TEMPERATURE       = 0.3
MAX_OUTPUT_TOKENS = 1500           # 2.5 Pro reserves tokens for hidden thinking
THINKING_BUDGET   = 128            # 2.5-pro minimum; task doesn't need deep reasoning
DIFF_TRUNC        = 3000           # matches SFT Phase 2 truncation (CLAUDE.md §2.2)
PROGRESS_EVERY    = 100

# must match eval.ipynb Cell 5 or the hallucination metric drifts
_STOP = frozenset({
    'self', 'return', 'import', 'class', 'True', 'False', 'None', 'with',
    'from', 'def', 'the', 'and', 'for', 'not', 'this', 'that', 'list',
    'dict', 'str', 'int',
})

_IDENT_RE      = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b')
_BACKTICKED_RE = re.compile(r'`([^`]+)`')
_PURE_IDENT_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{2,}$')

_FALSE_REF_PATTERNS = [
    re.compile(r'the `([a-zA-Z_][a-zA-Z0-9_]{2,})` (?:is|was|should|has|does|needs)', re.IGNORECASE),
    re.compile(r'`([a-zA-Z_][a-zA-Z0-9_]{2,})` (?:function|method|variable|class|field|attribute)', re.IGNORECASE),
]

GENERIC_PHRASES = (
    'please confirm', 'please provide more context', 'no issues found',
    'lgtm', 'looks good to me',
)

# prefer an API key (AI Studio, simpler) and fall back to Vertex via project id
_API_KEY    = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
_LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
_needs_gemini = "--sanity" not in sys.argv and "--samples" not in sys.argv

if _needs_gemini:
    if genai is None:
        raise SystemExit(
            "google-genai not installed. Run: pip install google-genai"
        )

    if _API_KEY:
        _client = genai.Client(api_key=_API_KEY)
        print(f"AI Studio client: model={MODEL_NAME}")
    elif _PROJECT_ID:
        _client = genai.Client(vertexai=True, project=_PROJECT_ID, location=_LOCATION)
        print(f"Vertex AI client: project={_PROJECT_ID} location={_LOCATION} model={MODEL_NAME}")
    else:
        raise SystemExit(
            "Set one of:\n"
            "  GEMINI_API_KEY   (for AI Studio — simpler, broader model access)\n"
            "  GOOGLE_CLOUD_PROJECT  (for Vertex AI — also needs ADC auth)\n"
        )
else:
    _client = None

_error_count = 0
_error_lock = threading.Lock()
_output_lock = threading.Lock()
_verbose_count = 0
_verbose_lock = threading.Lock()


def extract_diff_identifiers(diff: str) -> set[str]:
    """Identifiers (3+ chars) on +/- lines of a unified diff, minus stopwords.
    The +++/--- file headers are skipped."""
    changed = []
    for line in diff.split('\n'):
        if not line or line.startswith(('+++', '---')):
            continue
        if line.startswith(('+', '-')):
            changed.append(line[1:])
    return set(_IDENT_RE.findall(' '.join(changed))) - _STOP


def identifier_match_count(review: str, diff_ids: set[str]) -> int:
    return sum(1 for ident in diff_ids if ident in review)


def classify_tier(record: dict, diff_ids: set[str] | None = None) -> str:
    """Rough pre-call tier, for reporting only — the filters make the real call.
        'drop' : short + generic, probably not worth training on
        'weak' : medium length, few identifier matches — likely rewritten
        'keep' : long or specific — likely returned unchanged
    """
    out = record.get('output', '')
    length = len(out)
    if diff_ids is None:
        diff_ids = extract_diff_identifiers(record.get('input', ''))
    matches = identifier_match_count(out, diff_ids)
    lower = out.lower()
    is_generic = any(g in lower for g in GENERIC_PHRASES)

    if length < 40 or (is_generic and length < 120):
        return 'drop'
    if length < 300 and matches <= 1:
        return 'weak'
    return 'keep'


def passes_safety_filters(
    rewrite: str,
    original: str,
    diff_full: str,
    diff_ids: set[str],
) -> tuple[bool, str]:
    """The three filters from CLAUDE.md §1.2."""
    if identifier_match_count(rewrite, diff_ids) == 0:
        return False, "no_diff_identifier"

    # Asymmetric length window. Weak originals benefit from expansion (terse
    # nitpick → specific bug), so the ceiling is generous; the floor catches
    # truncation/gibberish. Hard cap at 1500 to still catch runaway responses.
    lo = max(20, int(len(original) * 0.3))
    hi = min(1500, max(600, int(len(original) * 3.0)))
    if not (lo <= len(rewrite) <= hi):
        return False, f"length_{len(rewrite)}vs{len(original)}"

    # Soft never-invent: flag text that references a symbol as if it already
    # exists in the diff when it doesn't. Only specific phrasings are checked,
    # so suggested new code like `os.environ.get('DB_PASS')` is allowed through.
    for pat in _FALSE_REF_PATTERNS:
        for m in pat.finditer(rewrite):
            ident = m.group(1)
            if ident in _STOP:
                continue
            if ident not in diff_full:
                return False, f"false_reference_{ident}"

    return True, ""


_PROMPT_TEMPLATE = """You are improving a code review to be more specific and bug-focused.

DIFF:
{diff}

ORIGINAL REVIEW:
{review}

Rules:
- If the ORIGINAL is already specific (references variables/lines from the diff, identifies a concrete issue), return it UNCHANGED.
- Otherwise, rewrite to: (a) name the specific bug or antipattern, (b) reference the exact variable/function/line from the diff, (c) explain WHY it is a problem, (d) give the specific fix.
- Match the ORIGINAL's length and tone. Never add bullets, headers, or multiple paragraphs.
- Never invent bugs that are not in the diff. If the diff looks fine, say so briefly.

Output ONLY the (possibly unchanged) review text — no preamble, no explanation, no "Here is the improved review:" etc."""


def call_gemini(diff: str, original: str, max_retries: int = 3) -> tuple[str, str]:
    """Returns (text, finish_reason). finish_reason is the enum name, e.g. 'STOP'
    or 'MAX_TOKENS'; the caller should reject anything other than 'STOP'."""
    prompt = _PROMPT_TEMPLATE.format(diff=diff, review=original)
    cfg = genai_types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=cfg,
            )

            # normalize finish_reason (enum or dotted string) down to its name
            finish_reason = 'UNKNOWN'
            if getattr(resp, 'candidates', None):
                fr = resp.candidates[0].finish_reason
                finish_reason = getattr(fr, 'name', None) or str(fr)
                if '.' in finish_reason:
                    finish_reason = finish_reason.rsplit('.', 1)[-1]

            text = getattr(resp, 'text', None) or ''
            return text.strip(), finish_reason
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"Gemini failed after {max_retries} retries: {last_err}")


def process_one(idx: int, record: dict) -> dict:
    global _error_count, _verbose_count

    diff_full = record.get('input', '')
    diff_trunc = diff_full[:DIFF_TRUNC]
    original = record.get('output', '')
    diff_ids = extract_diff_identifiers(diff_full)

    base = {
        'instruction': record.get('instruction', 'Review this code change.'),
        'input': diff_full,
        'original_output': original,
        'repo': record.get('repo', ''),
        '_idx': idx,
    }

    tier = classify_tier(record, diff_ids)

    # broken records: keep them but mark dropped, skip the API call
    if tier == 'drop':
        return {**base, 'output': original, 'rewrite_status': 'dropped'}

    # long + already very specific: almost never worth a rewrite
    if tier == 'keep' and identifier_match_count(original, diff_ids) >= 3:
        return {**base, 'output': original, 'rewrite_status': 'kept_short_circuit'}

    try:
        rewrite, finish_reason = call_gemini(diff_trunc, original)
    except Exception as e:
        with _error_lock:
            _error_count += 1
            if _error_count <= 5:
                print(f"  [WARN] Gemini error #{_error_count} on idx={idx}: {e}")
        return {
            **base,
            'output': original,
            'rewrite_status': 'gemini_error',
            '_error': str(e)[:200],
        }

    # anything but STOP means a truncated/blocked response — don't trust it
    if finish_reason != 'STOP':
        with _verbose_lock:
            if _verbose_count < 3:
                _verbose_count += 1
                print(f"\n⚠ gemini_incomplete #{_verbose_count} (idx={idx}) finish_reason={finish_reason}")
                print(f"  orig ({len(original)} ch): {original[:150]}")
                print(f"  got  ({len(rewrite)} ch): {rewrite[:150]}")
        return {
            **base,
            'output': original,
            'rewrite_status': f'gemini_incomplete:{finish_reason}',
        }

    if rewrite.strip() == original.strip():
        return {**base, 'output': original, 'rewrite_status': 'kept_by_gemini'}

    passed, reason = passes_safety_filters(rewrite, original, diff_full, diff_ids)
    if not passed:
        return {
            **base,
            'output': original,
            'rewrite_status': f'filtered:{reason}',
        }

    result = {**base, 'output': rewrite, 'rewrite_status': 'rewritten'}

    # dump the first few rewrites so quality is visible early in a run
    with _verbose_lock:
        if _verbose_count < 3:
            _verbose_count += 1
            print(f"\n═══ sample rewrite #{_verbose_count} (idx={idx}, repo={result['repo']}) ═══")
            print(f"ORIGINAL  ({len(original)} chars):\n  {original[:220]}")
            print(f"REWRITTEN ({len(rewrite)} chars):\n  {rewrite[:220]}")
            print("═" * 60 + "\n")

    return result


def load_and_clean_output(path: str) -> set[int]:
    """
    Scan the output file for already-processed indices.
    Transient failures (gemini_error / gemini_incomplete) are dropped and the
    file rewritten without them, so those indices get retried next run.
    Returns the set of indices considered done.
    """
    if not os.path.exists(path):
        return set()

    kept = []
    dropped_errors = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = record.get('rewrite_status', '')
            if status == 'gemini_error' or status.startswith('gemini_incomplete'):
                dropped_errors += 1
                continue
            kept.append(record)

    if dropped_errors:
        with open(path, 'w') as f:
            for rec in kept:
                f.write(json.dumps(rec) + '\n')
        print(f"Resume cleanup: dropped {dropped_errors} prior error/incomplete rows (will retry)")

    processed = {r['_idx'] for r in kept if r.get('_idx') is not None}
    return processed


def print_existing_samples(path: str, n: int = 10) -> None:
    """Print n random rewrites from an existing output file (for manual QC)."""
    if not os.path.exists(path):
        print(f"No output file at {path}")
        return
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    rewritten = [r for r in records if r.get('rewrite_status') == 'rewritten']
    if not rewritten:
        print("No records with rewrite_status='rewritten' yet.")
        return
    random.seed(42)
    random.shuffle(rewritten)
    for i, r in enumerate(rewritten[:n], 1):
        print(f"\n═══ sample {i}/{n} (idx={r.get('_idx')}, repo={r.get('repo')}) ═══")
        print(f"ORIGINAL:\n  {r['original_output'][:300]}")
        print(f"REWRITTEN:\n  {r['output'][:300]}")
    print()


def run_sanity_checks() -> None:
    print("Running sanity checks...")

    d = "@@ -1,3 +1,3 @@\n-def foo():\n+def bar_fn():\n     return 1"
    ids = extract_diff_identifiers(d)
    assert 'bar_fn' in ids and 'foo' in ids, f"bar_fn/foo missing: {ids}"
    assert 'return' not in ids, "return should be stopword-filtered"
    print("  ok extract_diff_identifiers")

    d2 = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old_value\n+new_value"
    ids2 = extract_diff_identifiers(d2)
    assert 'old_value' in ids2 and 'new_value' in ids2, f"got {ids2}"
    assert 'foo' not in ids2, f"foo.py headers should be skipped: {ids2}"
    print("  ok extract_diff_identifiers (headers ignored)")

    assert identifier_match_count("the `bar_fn` is wrong", {'bar_fn'}) == 1
    assert identifier_match_count("nothing here", {'foo'}) == 0
    assert identifier_match_count("use `bar_fn` and `foo`", {'bar_fn', 'foo'}) == 2
    print("  ok identifier_match_count")

    assert classify_tier({'input': '', 'output': 'lgtm'}) == 'drop'
    assert classify_tier({'input': '', 'output': 'Please confirm the merge.'}) == 'drop'
    long_specific = {
        'input': '@@\n+def bar_fn():\n-def foo():',
        'output': 'Rename `foo` to `bar_fn` throughout the module; this will affect callers in ' + ('x ' * 150),
    }
    tier = classify_tier(long_specific)
    assert tier == 'keep', f"long+specific should be keep, got {tier}"
    print("  ok classify_tier")

    diff = "-def foo():\n+def bar_fn():"
    ids = {'foo', 'bar_fn'}
    ok, r = passes_safety_filters(
        "Rename `bar_fn` back to `foo` to preserve API compatibility.",
        "Rename the function back.",
        diff, ids,
    )
    assert ok, f"good review should pass: {r}"

    ok, r = passes_safety_filters("This review is generic.", "Original.", diff, ids)
    assert not ok and r == "no_diff_identifier", f"got {ok=} {r=}"

    # 5000 > the 1500 hard cap
    ok, r = passes_safety_filters(
        "x" * 5000 + " `foo`",
        "Original.",
        diff, ids,
    )
    assert not ok and r.startswith("length_"), f"got {ok=} {r=}"

    # short original (80 ch) expanded to ~300 ch, still has a diff identifier
    short_orig = "x" * 80
    expanded = "The `foo` function has an issue here: " + ("y " * 120) + "`foo`"
    ok, r = passes_safety_filters(expanded, short_orig, diff, ids)
    assert ok, f"short→expanded rewrite should pass: {r}"

    # "the `X` is" where X isn't in the diff
    ok, r = passes_safety_filters(
        "The `nonexistent_fn` is missing here, though `foo` is also affected.",
        "Rename the function.",
        diff, ids,
    )
    assert not ok and "false_reference_nonexistent_fn" in r, f"got {ok=} {r=}"

    # suggested new code shouldn't count as a false reference
    ok, r = passes_safety_filters(
        "Replace the call with `os.environ.get('DB_PASS')` to avoid hardcoding `foo`.",
        "Don't hardcode.",
        diff, ids,
    )
    assert ok, f"suggested new code should pass: {r}"

    print("  ok passes_safety_filters")
    print("All sanity checks passed.\n")


def main(dry_run_n: int | None = None) -> None:
    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {INPUT_FILE}")

    # cheap distribution preview before we spend API $
    tier_preview = Counter()
    for r in records[:2000]:
        tier_preview[classify_tier(r)] += 1
    print(f"Tier preview (first 2000 records): {dict(tier_preview)}")
    print("  (only 'drop' is skipped unconditionally; 'weak' and 'keep' both go to Gemini.)")

    done = load_and_clean_output(OUTPUT_FILE)
    if done:
        print(f"Resume: {len(done)} records already in {OUTPUT_FILE}")

    remaining = [(i, r) for i, r in enumerate(records) if i not in done]
    if dry_run_n is not None:
        remaining = remaining[:dry_run_n]
        print(f"[DRY RUN] processing only {len(remaining)} records")

    if not remaining:
        print("Nothing to do — all records already processed.")
        return

    print(f"Processing {len(remaining)} records with {MAX_WORKERS} workers...\n")

    stats: Counter[str] = Counter()
    completed = 0
    t0 = time.time()

    with open(OUTPUT_FILE, 'a') as fout, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:

        futures = {pool.submit(process_one, idx, rec): idx for idx, rec in remaining}

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"  [ERROR] idx={idx} crashed: {e}")
                continue

            with _output_lock:
                fout.write(json.dumps(result) + '\n')
                fout.flush()

            # collapse 'filtered:reason' into a single 'filtered' bucket for the line
            status = result['rewrite_status'].split(':', 1)[0]
            stats[status] += 1
            completed += 1
            if completed % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta_min = (len(remaining) - completed) / rate / 60 if rate > 0 else 0
                summary = " ".join(f"{k}={v}" for k, v in stats.most_common())
                print(f"  [{completed}/{len(remaining)}] {rate:.1f}/s  ETA {eta_min:.0f}min  {summary}")

    total = sum(stats.values())
    print(f"\n{'─' * 60}")
    print(f"Done. Processed {completed} records in {(time.time()-t0)/60:.1f} min.")
    print("Status breakdown:")
    for status, count in stats.most_common():
        pct = 100 * count / total if total else 0
        print(f"  {status:<30} {count:>6}  ({pct:5.1f}%)")
    print(f"\nGemini API errors encountered: {_error_count}")
    print(f"Output: {OUTPUT_FILE}")
    print(
        "\nNext steps per CLAUDE.md:\n"
        "  1. Run `python rewrite_data_gemini.py --samples 20` to eyeball quality\n"
        "  2. If OK, expand generate_patch_data.py to 450 records (Phase 1.4)\n"
        "  3. Merge rewritten + synthetic -> train_dataset_v3.jsonl\n"
        "  4. Proceed to Phase 2 (sft.ipynb edits)"
    )


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--sanity" in args:
        run_sanity_checks()
        sys.exit(0)

    if "--samples" in args:
        i = args.index("--samples")
        n = int(args[i + 1]) if i + 1 < len(args) else 10
        print_existing_samples(OUTPUT_FILE, n=n)
        sys.exit(0)

    # always sanity-check before the real job
    run_sanity_checks()

    dry_run_n = None
    if "--dry-run" in args:
        i = args.index("--dry-run")
        if i + 1 < len(args):
            dry_run_n = int(args[i + 1])

    main(dry_run_n=dry_run_n)
