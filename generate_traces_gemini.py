#!/usr/bin/env python3
"""Generate reasoning-trace SFT targets for the v4 code-reviewer.

Reads train_dataset_v3.jsonl (the same diffs v3 was trained on) and produces a
structured <think>+<review> trace per record via a two-call reconciliation flow:

    <think>   independent step-by-step reasoning about the diff
    </think>
    <review>  concise final review
    </review>

Call 1 sees only the diff and reasons forward; Call 2 sees the diff plus Call 1's
draft plus the v3 reference and produces the final review. The SFT target is Call
1's <think> + Call 2's <review>, so the trace the student learns from is never
anchored on a reference it won't have at inference. The full blob is the v4 SFT
target; <think> can be stripped at inference for latency.

Four filters reject bad traces; failures fall back to the v3 review:
  1. review identifier-match — <review> mentions a diff identifier
  2. trace identifier subset — every backticked id in <review> is in diff or ref
  3. length sanity          — <review> within bounds, <think> under the cap
  4. schema compliance      — both blocks present and parseable

Records v3 already flagged junk (rewrite_status='dropped') pass through untouched.

Usage:
    python generate_traces_gemini.py --sanity        # offline checks then exit
    python generate_traces_gemini.py --dry-run 20    # process first 20 only
    python generate_traces_gemini.py                 # full run, resumable
    python generate_traces_gemini.py --samples 10    # print 10 traces from existing output

Auth: export DEEPSEEK_API_KEY=sk-... (get one at
https://platform.deepseek.com/api_keys).
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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

INPUT_FILE        = "train_dataset_v3.jsonl"
# two-call reconciliation output; the older single-call dataset is kept on disk for audit only
OUTPUT_FILE       = "train_dataset_v4_traces_o2.jsonl"
# DeepSeek V4-Pro outputs are MIT-licensed, so safe to use as SFT data
MODEL_NAME        = os.environ.get("MODEL_ID", "deepseek-v4-pro")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MAX_WORKERS       = int(os.environ.get("MAX_WORKERS", "16"))  # V4-Pro is ~30s/call; bump via MAX_WORKERS for more concurrency
TEMPERATURE       = 0.3
MAX_OUTPUT_TOKENS = 6000            # plenty for our trace shape
DIFF_TRUNC        = 3000            # matches SFT Cell 3 truncation
PROGRESS_EVERY    = 100

# length budgets for filter 3
THINK_MAX_CHARS   = 10000           # Call-1 independent reasoning runs long; 10k still catches actual runaway rambles
REVIEW_LEN_LOW    = 0.5             # × v3 review length
REVIEW_LEN_HIGH   = 2.5             # × v3 review length
REVIEW_HARD_FLOOR = 20              # don't accept near-empty <review> even for short anchors
REVIEW_HARD_CAP   = 1500            # runaway-completion safety

# Stopwords for the never-invent filter — superset of eval.ipynb Cell 5 /
# rewrite_data_gemini.py. Includes real Python/library vocabulary (exception
# types, typing names, framework APIs) so the filter doesn't reject legitimate
# reasoning like "raises TypeError when X is None". Keep in sync with eval.ipynb.
_STOP = frozenset({
    # Python keywords (≥3 chars; shorter ones are below _IDENT_RE's minimum length)
    'self', 'return', 'import', 'class', 'with', 'from', 'def',
    'the', 'and', 'for', 'not', 'this', 'that',
    'try', 'except', 'finally', 'raise', 'async', 'await', 'assert',
    'lambda', 'yield', 'pass', 'elif', 'else', 'while', 'break',
    'continue', 'global', 'nonlocal',
    # built-in primitives
    'True', 'False', 'None',
    'bool', 'bytes', 'bytearray', 'complex', 'dict', 'float', 'frozenset',
    'int', 'list', 'object', 'set', 'str', 'tuple', 'type',
    # standard exception types
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
    # typing-module vocabulary — annotations, not symbols that live in a given diff
    'Optional', 'Union', 'Callable', 'Any', 'Annotated', 'Literal',
    'Iterable', 'Iterator', 'AsyncIterable', 'AsyncIterator',
    'Generator', 'AsyncGenerator', 'Coroutine', 'Awaitable',
    'Sequence', 'MutableSequence', 'Mapping', 'MutableMapping',
    'List', 'Dict', 'Tuple', 'Set', 'FrozenSet',
    'Type', 'ClassVar', 'TypeVar', 'Generic', 'Final',
    'Protocol', 'NamedTuple', 'TypedDict', 'NewType', 'cast', 'overload',
    # common builtins reviewers reference as vocabulary
    'print', 'len', 'range', 'super', 'property',
    'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr',
    'staticmethod', 'classmethod', 'callable',
    # ML library names — vocabulary, not project-specific symbols
    'huggingface_hub', 'transformers', 'accelerate', 'datasets',
    'torch', 'torchvision', 'pytorch', 'tensorflow', 'jax', 'flax',
    'numpy', 'pandas', 'sklearn', 'scipy', 'pytest', 'sentencepiece',
    # universal ML attribute vocabulary common to nearly every model PR review
    'batch_size', 'num_channels', 'attention_mask', 'pixel_values',
    'return_tensors', 'input_ids', 'hidden_states', 'logits', 'embeddings',
    # common framework method/attribute names that come up as vocabulary
    'outputs', 'headers', 'forward', 'generate', 'encode', 'decode',
    # dunders (instance protocol)
    '__call__', '__init__', '__repr__', '__str__', '__len__', '__iter__',
    # class introspection dunders — reasoning vocabulary about hierarchies
    '__base__', '__bases__', '__class__', '__dict__', '__name__', '__module__',
    '__mro__', '__qualname__', '__subclasses__',
    # FastAPI / Pydantic / stdlib / pytest vocabulary
    'APIRoute', 'Response', 'Request',
    'validation_alias', 'serialization_alias',
    'PurePath', 'Path', 'PathLike',
    'HTTPConnection', 'HTTPSConnection',
    'pytest_runtest_setup', 'pytest_runtest_call', 'pytest_collection_modifyitems',
    # English words V4-Pro routinely backticks during Call-1 reasoning
    'where', 'when', 'whenever', 'because', 'then', 'after', 'before',
    'using', 'via', 'uses', 'without', 'inside', 'into', 'contains',
    'becomes', 'has', 'have', 'being', 'causes', 'likely', 'something',
    'expects', 'accepts', 'returns',
    'here', 'there', 'must', 'should', 'would', 'could', 'note', 'also',
    'still', 'under', 'over', 'above', 'below', 'between', 'during',
    'against', 'unlike', 'like',
    # stdlib module names + third-party API surfaces referenced as comparison vocabulary
    'weakref', 'urllib', 'urllib3', 'requests', 'httpx',
    'starlette', 'redoc', 'swagger', 'openapi', 'asgi', 'wsgi',
    'redoc_url', 'swagger_url', 'openapi_url', 'docs_url',
    'encode_multipart_formdata', 'multipart_formdata',
    'werkzeug', 'flask', 'django',
    'pydantic_core', 'email_validator',
    'JWTError', 'UploadFile', 'get_type_hints',
    'checks', 'sets', 'default', 'redirect_slashes',
})

_IDENT_RE      = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b')
_BACKTICKED_RE = re.compile(r'`([^`]+)`')

# non-greedy + DOTALL so <think> can span newlines
_THINK_RE  = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
_REVIEW_RE = re.compile(r'<review>(.*?)</review>', re.DOTALL | re.IGNORECASE)
# _DRAFT_RE is unused at runtime (the draft comes straight from Call 1's content)
# but kept for parser symmetry / sanity tests.
_DRAFT_RE  = re.compile(r'<draft_review>(.*?)</draft_review>', re.DOTALL | re.IGNORECASE)
_RECON_RE  = re.compile(r'<reconciliation>(.*?)</reconciliation>', re.DOTALL | re.IGNORECASE)
_LABEL_RE  = re.compile(r'LABEL:\s*(AGREE|REFERENCE_BETTER|DRAFT_BETTER|BOTH_VALID)', re.IGNORECASE)

_needs_client = "--sanity" not in sys.argv and "--samples" not in sys.argv

# single shared client; the OpenAI SDK pools connections and is thread-safe for create()
_client_obj = None

def _get_client():
    return _client_obj

if _needs_client:
    if OpenAI is None:
        raise SystemExit("openai SDK not installed. Run: pip install openai")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "DEEPSEEK_API_KEY not set. Get one at https://platform.deepseek.com/api_keys "
            "and export it: export DEEPSEEK_API_KEY=sk-..."
        )
    _client_obj = OpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        timeout=300.0,
        max_retries=0,  # our own retry loop handles backoff; let the SDK fail fast
    )
    print(f"DeepSeek client: base_url={DEEPSEEK_BASE_URL} model={MODEL_NAME}")

_error_count = 0
_error_lock = threading.Lock()
_output_lock = threading.Lock()
_verbose_count = 0
_verbose_lock = threading.Lock()


def extract_diff_identifiers(diff: str) -> set[str]:
    """Identifiers (3+ chars) on +/- lines of a unified diff, minus stopwords.

    Ignores the +++/--- file header lines. Mirrors rewrite_data_gemini.py.
    """
    changed = []
    for line in diff.split('\n'):
        if not line or line.startswith(('+++', '---')):
            continue
        if line.startswith(('+', '-')):
            changed.append(line[1:])
    return set(_IDENT_RE.findall(' '.join(changed))) - _STOP


def parse_trace(text: str) -> tuple[str | None, str | None]:
    """Return whitespace-trimmed (think, review), or (None, None) if either block
    is missing or empty."""
    t = _THINK_RE.search(text)
    r = _REVIEW_RE.search(text)
    if not t or not r:
        return None, None
    think  = t.group(1).strip()
    review = r.group(1).strip()
    if not think or not review:
        return None, None
    return think, review


def passes_filters(
    think: str,
    review: str,
    v3_review: str,
    diff_full: str,
    diff_ids: set[str],
) -> tuple[bool, str]:
    """The four trace filters.

    Filter 2 (identifier subset) only fires on <review>, not <think>. Call 1
    sees only the diff and reasons forward, so it routinely mentions library API
    names (`JSONResponse` etc.) as exploratory vocabulary — that's fine. The
    final <review> is what the student learns to emit, so we still hard-reject
    fabricated identifiers there.
    """

    # 1. review must mention a diff identifier — unless the v3 review was itself a
    #    legitimate identifier-free "no issue" review, in which case so can ours.
    review_has_id = any(ident in review for ident in diff_ids)
    v3_has_id     = any(ident in v3_review for ident in diff_ids)
    if not review_has_id and v3_has_id:
        return False, "review_no_diff_identifier"

    # 2. every backticked identifier in <review> must appear in the diff or the v3
    #    reference. The reference is allowed because it's the conclusion we anchor
    #    on; any name it proposes (e.g. a new constant) is legitimate.
    diff_and_ref = diff_full + "\n" + v3_review
    for ident in _BACKTICKED_RE.findall(review):
        ident = ident.strip()
        if not _IDENT_RE.fullmatch(ident):
            continue
        if ident in _STOP:
            continue
        if ident not in diff_and_ref:
            return False, f"false_ref_review_{ident}"

    # 3. length sanity (asymmetric)
    if len(think) > THINK_MAX_CHARS:
        return False, f"think_too_long_{len(think)}"

    v3_len = max(1, len(v3_review))
    # floor allows valid contractions ("...no issues spotted." -> "No issue found.")
    # without accepting truly empty completions; ceil caps runaway growth.
    lo = max(5, min(REVIEW_HARD_FLOOR, int(v3_len * 0.3)))
    hi = min(REVIEW_HARD_CAP, max(400, int(v3_len * REVIEW_LEN_HIGH)))
    if not (lo <= len(review) <= hi):
        return False, f"review_length_{len(review)}vs{v3_len}"

    # 4. schema compliance is enforced upstream — both blocks are non-empty here.

    return True, ""


# Call 1 sees ONLY the diff, so its reasoning_content is pure independent review —
# that's what we use as the SFT <think> target (the student never sees a reference
# at inference). Call 2 sees diff + draft + reference and produces the final
# review, letting the human signal improve the conclusion without polluting the trace.

_CALL1_PROMPT = """You are an expert code reviewer. Review the following code diff.

DIFF:
{diff}

Reason carefully about what the diff actually shows. In your thinking, work through:
- Diff summary (≤30 words)
- Intent of the change
- Step-by-step inspection: bugs, security, performance/resource, style/API contract
- Identifier-grounded conclusion (which line/identifier, if any, is the problem)

Then produce a concise review in <review>...</review>:
- Only mention identifiers (variables, functions, classes) that appear in the diff
- Length budget: 60–300 chars; match a typical PR comment
- If the diff has no real issue, say so plainly
- No bullets, no headers"""


_CALL2_PROMPT = """You previously reviewed a code diff independently and produced a draft review. Another reviewer has independently reviewed the same diff. Reconcile your draft with theirs.

DIFF:
{diff}

YOUR DRAFT REVIEW:
{draft}

REFERENCE REVIEW (a second opinion — may be wrong, shallow, or a nitpick):
{review}

MANDATORY OUTPUT FORMAT — emit BOTH blocks in this exact order. Do not skip the reconciliation block, even when the answer is AGREE. Skipping it makes your output unusable for downstream auditing.

<reconciliation>
LABEL: <one of: AGREE | REFERENCE_BETTER | DRAFT_BETTER | BOTH_VALID>
<one to three sentences explaining your choice>
</reconciliation>
<review>
<the final review text>
</review>

Label meanings:
- AGREE             — your draft and the reference reach the same conclusion (still emit this block)
- REFERENCE_BETTER  — the reference caught something you missed; you will incorporate it
- DRAFT_BETTER      — the reference is wrong, a nitpick, or off-topic; you will stick with your draft
- BOTH_VALID        — both reach valid but different conclusions about different aspects

Final review rules:
- If AGREE or DRAFT_BETTER, use your draft text essentially as-is
- If REFERENCE_BETTER, update your draft to incorporate the reference's insight
- If BOTH_VALID, pick the more substantive angle
- Only mention identifiers (variables, functions, classes) that appear in the DIFF
- Length budget: 60–300 chars; no bullets, no headers
- Be honest about your label — do not pick AGREE just to avoid effort, and do not pick DRAFT_BETTER just to defend your draft"""


def call_model(diff: str, reference_review: str, max_retries: int = 3) -> tuple[str, str, str, str, str, str]:
    """Run the two-call reconciliation flow.

    Returns (think, draft_review, recon_label, recon_text, final_review, stop_reason).

    Call 1 sees only the diff; its reasoning_content is the SFT <think> target and
    its draft review feeds Call 2. Call 2 sees diff + draft + reference and emits a
    reconciliation block (LABEL on the first line) plus the final review. Call 2's
    reasoning_content is discarded — it has seen the reference, so it's anchored.

    stop_reason reflects whichever call ended non-STOP so upstream
    incomplete-handling still applies.
    """
    prompt1 = _CALL1_PROMPT.format(diff=diff)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp1 = _get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt1}],
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "high",
                },
            )
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Call 1 failed after {max_retries} retries: {last_err}")

    msg1 = resp1.choices[0].message
    think = (getattr(msg1, "reasoning_content", None) or "").strip()
    text1 = (getattr(msg1, "content", None) or "").strip()

    # some responses inline <think> in content instead of reasoning_content
    if not think and "<think>" in text1:
        m = _THINK_RE.search(text1)
        if m:
            think = m.group(1).strip()
            text1 = _THINK_RE.sub("", text1).strip()

    m_r1 = _REVIEW_RE.search(text1)
    draft = m_r1.group(1).strip() if m_r1 else text1.strip()

    stop1 = (resp1.choices[0].finish_reason or "unknown").upper()

    # bail before spending Call 2 if Call 1 truncated or produced no draft
    if stop1 != "STOP" or not draft:
        return think, draft, "", "", "", stop1

    prompt2 = _CALL2_PROMPT.format(diff=diff, draft=draft, review=reference_review)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp2 = _get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt2}],
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "high",
                },
            )
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Call 2 failed after {max_retries} retries: {last_err}")

    msg2 = resp2.choices[0].message
    text2 = (getattr(msg2, "content", None) or "").strip()

    recon_text = ""
    recon_label = ""
    m_rc = _RECON_RE.search(text2)
    if m_rc:
        recon_text = m_rc.group(1).strip()
        m_l = _LABEL_RE.search(recon_text)
        if m_l:
            recon_label = m_l.group(1).upper()

    final = ""
    m_r2 = _REVIEW_RE.search(text2)
    if m_r2:
        final = m_r2.group(1).strip()

    stop2 = (resp2.choices[0].finish_reason or "unknown").upper()
    stop = stop2 if stop2 != "STOP" else stop1

    return think, draft, recon_label, recon_text, final, stop


def process_one(idx: int, record: dict) -> dict:
    global _error_count, _verbose_count

    diff_full  = record.get('input', '')
    diff_trunc = diff_full[:DIFF_TRUNC]
    v3_review  = record.get('output', '')
    v3_status  = record.get('rewrite_status', '')
    diff_ids   = extract_diff_identifiers(diff_full)

    base = {
        'instruction': record.get('instruction', 'Review this code change.'),
        'input': diff_full,
        'repo': record.get('repo', ''),
        'v3_output': v3_review,
        'original_output': record.get('original_output', v3_review),
        'v3_status': v3_status,
        '_idx': idx,
    }

    # junk v3 already flagged — keep v3 output, skip the model call
    if v3_status == 'dropped' or len(v3_review.strip()) < REVIEW_HARD_FLOOR:
        return {
            **base,
            'output': v3_review,
            'review_only': v3_review,
            'think_only': '',
            'draft_review': '',
            'recon_label': '',
            'recon_text': '',
            'trace_status': 'skipped_junk',
        }

    try:
        think, draft, recon_label, recon_text, final, finish_reason = call_model(diff_trunc, v3_review)
    except Exception as e:
        with _error_lock:
            _error_count += 1
            if _error_count <= 5:
                print(f"  [WARN] Model error #{_error_count} on idx={idx}: {e}")
        return {
            **base,
            'output': v3_review,
            'review_only': v3_review,
            'think_only': '',
            'draft_review': '',
            'recon_label': '',
            'recon_text': '',
            'trace_status': 'gemini_error',  # status name kept for resume-compat
            '_error': str(e)[:200],
        }

    if finish_reason != 'STOP':
        with _verbose_lock:
            if _verbose_count < 3:
                _verbose_count += 1
                print(f"\n⚠ incomplete #{_verbose_count} (idx={idx}) finish_reason={finish_reason}")
        return {
            **base,
            'output': v3_review,
            'review_only': v3_review,
            'think_only': '',
            'draft_review': draft,
            'recon_label': recon_label,
            'recon_text': recon_text,
            'trace_status': f'gemini_incomplete:{finish_reason}',
        }

    # Only the three load-bearing fields are required: think (SFT trace target),
    # draft (proves Call 1 reached a conclusion), final (SFT review target). The
    # reconciliation block is optional — V4-Pro routinely skips emitting it on
    # AGREE even though the silent reconciliation still happened (Call 2 saw the
    # reference). Treat a missing label as unlabeled, not corrupt.
    schema_problems = []
    if not think:
        schema_problems.append('no_think')
    if not draft:
        schema_problems.append('no_draft')
    if not final:
        schema_problems.append('no_final_review')
    if schema_problems:
        return {
            **base,
            'output': v3_review,
            'review_only': v3_review,
            'think_only': '',
            'draft_review': draft,
            'recon_label': recon_label,
            'recon_text': recon_text,
            'trace_status': 'schema_invalid:' + ','.join(schema_problems),
            '_filtered_think': think,
            '_filtered_review': final,
        }

    passed, reason = passes_filters(think, final, v3_review, diff_full, diff_ids)
    if not passed:
        return {
            **base,
            'output': v3_review,
            'review_only': v3_review,
            'think_only': '',
            'draft_review': draft,
            'recon_label': recon_label,
            'recon_text': recon_text,
            'trace_status': f'filtered:{reason}',
            '_filtered_think': think,
            '_filtered_review': final,
        }

    # SFT target = Call 1's <think> + Call 2's final <review>. The reconciliation
    # block is preserved for audit but excluded from the target — the student has
    # no reference at inference, so training on reconciliation teaches an unusable skill.
    full_trace = f"<think>\n{think}\n</think>\n<review>\n{final}\n</review>"

    result = {
        **base,
        'output': full_trace,
        'review_only': final,
        'think_only': think,
        'draft_review': draft,
        'recon_label': recon_label,
        'recon_text': recon_text,
        'trace_status': 'trace_generated',
    }

    # eyeball the first 3 traces so you can ctrl-c early if something looks off
    with _verbose_lock:
        if _verbose_count < 3:
            _verbose_count += 1
            print(f"\n═══ sample trace #{_verbose_count} (idx={idx}, repo={result['repo']}) label={recon_label or '(none)'} ═══")
            print(f"V3 REVIEW ({len(v3_review)} ch): {v3_review[:200]}")
            print(f"<think> ({len(think)} ch):\n  {think[:300]}")
            print(f"DRAFT ({len(draft)} ch): {draft[:200]}")
            print(f"FINAL <review> ({len(final)} ch):\n  {final[:200]}")
            print("═" * 60 + "\n")

    return result


def load_and_clean_output(path: str) -> set[int]:
    """Read prior output and return the set of already-done indices, dropping
    transient error/incomplete rows so they get retried. Same logic as
    rewrite_data_gemini.py."""
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
            status = record.get('trace_status', '')
            if status == 'gemini_error' or status.startswith('gemini_incomplete'):
                dropped_errors += 1
                continue
            kept.append(record)

    if dropped_errors:
        with open(path, 'w') as f:
            for rec in kept:
                f.write(json.dumps(rec) + '\n')
        print(f"Resume cleanup: dropped {dropped_errors} prior error/incomplete rows (will retry)")

    return {r['_idx'] for r in kept if r.get('_idx') is not None}


def print_existing_samples(path: str, n: int = 10) -> None:
    if not os.path.exists(path):
        print(f"No output file at {path}")
        return
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    good = [r for r in records if r.get('trace_status') == 'trace_generated']
    if not good:
        print("No records with trace_status='trace_generated' yet.")
        return
    random.seed(42)
    random.shuffle(good)
    for i, r in enumerate(good[:n], 1):
        print(f"\n═══ sample {i}/{n} (idx={r.get('_idx')}, repo={r.get('repo')}) ═══")
        print(f"V3 REVIEW: {r['v3_output'][:250]}")
        print(f"<think>:\n  {r.get('think_only', '')[:400]}")
        print(f"<review>:\n  {r.get('review_only', '')[:250]}")
    print()


def run_sanity_checks() -> None:
    print("Running sanity checks...")

    d = "@@ -1,3 +1,3 @@\n-def foo():\n+def bar_fn():\n     return 1"
    ids = extract_diff_identifiers(d)
    assert 'bar_fn' in ids and 'foo' in ids, f"got {ids}"
    assert 'return' not in ids
    print("  ok extract_diff_identifiers")

    raw = "<think>\nstep 1\n</think>\n<review>\nfix it\n</review>"
    t, r = parse_trace(raw)
    assert t == "step 1" and r == "fix it", f"got {t=} {r=}"
    print("  ok parse_trace (happy path)")

    t, r = parse_trace("<review>only this</review>")
    assert t is None and r is None
    t, r = parse_trace("plain text with no tags")
    assert t is None and r is None
    t, r = parse_trace("<think></think><review>x</review>")
    assert t is None and r is None
    print("  ok parse_trace (rejection cases)")

    diff = "-def foo():\n+def bar_fn():"
    ids = {'foo', 'bar_fn'}
    ok, reason = passes_filters(
        think="Diff renames foo to bar_fn. The `bar_fn` rename breaks callers.",
        review="Rename `bar_fn` back to `foo` to preserve API compatibility.",
        v3_review="Rename the function back to foo.",
        diff_full=diff,
        diff_ids=ids,
    )
    assert ok, f"good trace should pass: {reason}"
    print("  ok passes_filters (happy path)")

    # non-diff identifier in <think> is allowed under two-call (exploratory reasoning)
    ok, reason = passes_filters(
        think="The `imaginary_fn` is the actual issue here.",
        review="Rename `bar_fn` back to `foo`.",
        v3_review="Rename the function back to foo.",
        diff_full=diff,
        diff_ids=ids,
    )
    assert ok, f"think-only non-diff identifier should pass under option-2: {reason}"
    print("  ok passes_filters (allows non-diff identifier in <think> — exploratory reasoning)")

    # review mentions a real diff id (filter 1 passes) so filter 2 must catch the fabricated `nope`
    ok, reason = passes_filters(
        think="Rename `bar_fn` to `foo`.",
        review="Fix `bar_fn` because `nope` is also affected here.",
        v3_review="Rename back to foo.",
        diff_full=diff,
        diff_ids=ids,
    )
    assert not ok and "false_ref_review_nope" in reason, f"got {ok=} {reason=}"
    print("  ok passes_filters (rejects fabricated <review> identifier)")

    ok, reason = passes_filters(
        think="`bar_fn` " + "x" * (THINK_MAX_CHARS + 100),
        review="Rename `bar_fn` back to `foo`.",
        v3_review="Rename the function back.",
        diff_full=diff,
        diff_ids=ids,
    )
    assert not ok and reason.startswith("think_too_long"), f"got {ok=} {reason=}"
    print("  ok passes_filters (rejects oversized <think>)")

    # real diff id keeps filter 1 happy; 12 chars vs 54-char v3 -> 0.3×54=16 floor -> fail
    ok, reason = passes_filters(
        think="Rename `bar_fn` to `foo`.",
        review="fix `bar_fn`",
        v3_review="Rename the function back to foo for API compatibility.",
        diff_full=diff,
        diff_ids=ids,
    )
    assert not ok and reason.startswith("review_length"), f"got {ok=} {reason=}"
    print("  ok passes_filters (rejects too-short <review>)")

    # no-issue case where v3 also has no identifiers; v3 must be ≥20 chars to clear
    # the upstream skipped_junk gate before the filters even run.
    ok, reason = passes_filters(
        think="Diff renames a variable. No correctness impact.",
        review="No issue found.",
        v3_review="The change looks correct, no issues spotted.",
        diff_full=diff,
        diff_ids=ids,
    )
    assert ok, f"no-issue/no-issue should pass: {reason}"
    print("  ok passes_filters (no-issue case)")

    # suggested new multi-token code shouldn't trip false_ref
    ok, reason = passes_filters(
        think="Replace the call with `os.environ.get('DB_PASS')`. `bar_fn` is affected.",
        review="Use `os.environ.get('DB_PASS')` instead of hardcoding `bar_fn`.",
        v3_review="Don't hardcode.",
        diff_full=diff,
        diff_ids=ids,
    )
    assert ok, f"suggested new code should pass: {reason}"
    print("  ok passes_filters (suggested new code allowed)")

    sample = """<reconciliation>
LABEL: REFERENCE_BETTER
The reference caught a missing comment that I overlooked.
</reconciliation>
<review>
Final identifier-grounded review here.
</review>"""
    m_rc = _RECON_RE.search(sample)
    assert m_rc, "reconciliation block not parsed"
    m_l = _LABEL_RE.search(m_rc.group(1))
    assert m_l and m_l.group(1).upper() == "REFERENCE_BETTER", f"label parse failed: {m_l}"
    m_r = _REVIEW_RE.search(sample)
    assert m_r and "identifier-grounded" in m_r.group(1)
    print("  ok option-2 parsers (REFERENCE_BETTER)")

    for lbl in ("AGREE", "DRAFT_BETTER", "BOTH_VALID"):
        s = f"<reconciliation>\nLABEL: {lbl}\nbecause.\n</reconciliation>\n<review>\nx mentioning `bar_fn`.\n</review>"
        m_rc = _RECON_RE.search(s)
        assert m_rc
        m_l = _LABEL_RE.search(m_rc.group(1))
        assert m_l and m_l.group(1).upper() == lbl
    print("  ok option-2 parsers (all 4 labels)")

    # missing LABEL line -> recon present but label empty
    s_bad = "<reconciliation>\njust some text, no label\n</reconciliation>\n<review>\nx</review>"
    m_rc = _RECON_RE.search(s_bad)
    assert m_rc
    assert not _LABEL_RE.search(m_rc.group(1))
    print("  ok option-2 parsers (missing label flagged)")

    print("All sanity checks passed.\n")


def main(dry_run_n: int | None = None) -> None:
    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {INPUT_FILE}")

    status_preview = Counter(r.get('rewrite_status', 'unknown') for r in records)
    print(f"v3 status distribution: {dict(status_preview.most_common())}")

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

            status = result['trace_status'].split(':', 1)[0]
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
    print(f"\nModel API errors encountered: {_error_count}")
    print(f"Output: {OUTPUT_FILE}")
    print(
        "\nNext steps (Phase A.2 of the plan):\n"
        "  1. python generate_traces_gemini.py --samples 20  # eyeball trace quality\n"
        "  2. Edit sft.ipynb Cell 3 to load train_dataset_v4_traces.jsonl\n"
        "     and use the 'output' field (full <think>+<review>) as the SFT target\n"
        "  3. Bump max_seq_length 2048 → 3072 and lora_alpha 32 → 64 in Cell 4\n"
        "  4. Train on Colab/RunPod; eval against v3 with the patched eval.ipynb"
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

    run_sanity_checks()

    dry_run_n = None
    if "--dry-run" in args:
        i = args.index("--dry-run")
        if i + 1 < len(args):
            dry_run_n = int(args[i + 1])

    main(dry_run_n=dry_run_n)
