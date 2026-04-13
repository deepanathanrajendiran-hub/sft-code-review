import os
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
from tqdm import tqdm
import json
import time

# --- CONFIGURATION ---
PROJECT_ID = "halogen-sol-348516"
LOCATION = "us-central1"
INPUT_FILE = "train_dataset_clean.jsonl"
OUTPUT_FILE = "train_dataset_rewritten.jsonl"

# --- INIT ---
print(f"Connecting to Vertex AI in {LOCATION}...")
vertexai.init(project=PROJECT_ID, location=LOCATION)

active_model = None
for name in ["gemini-2.0-flash-001", "gemini-1.5-flash"]:
    try:
        model = GenerativeModel(name)
        model.generate_content("test")
        active_model = model
        print(f"Connected to: {name}")
        break
    except:
        continue

if not active_model:
    raise RuntimeError("Could not connect to any Gemini model.")

# --- LOAD DATA ---
data = []
with open(INPUT_FILE, 'r') as f:
    for line in f:
        data.append(json.loads(line))
print(f"Loaded {len(data)} samples")

# --- RESUME LOGIC ---
processed_hashes = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                processed_hashes.add(hash(entry.get('input', '')[:500]))
            except:
                pass
    print(f"Found {len(processed_hashes)} already processed. Resuming...")

# --- COUNTERS ---
stats = {
    "total": 0,
    "skipped_duplicate": 0,
    "rewritten": 0,
    "filtered": 0,
    "api_errors": 0,
    "json_parse_errors": 0,
    "rate_limited": 0,
}

# --- PROMPT ---
safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

PROMPT = """You are a Senior Software Engineer writing a code review for a pull request.

Given a code diff and an existing review comment, rewrite the review to be:
1. SPECIFIC - Reference actual code, variable names, or line changes from the diff
2. ACTIONABLE - Tell the developer exactly what to change and why
3. CRITICAL - If there's a bug, security issue, or bad practice, call it out clearly
4. CONCISE - 1-3 sentences max

If the diff has a security issue (hardcoded secrets, SQL injection, etc.), start with "SECURITY:" and explain the risk.
If the diff has a bug (missing null checks, wrong logic, etc.), start with "BUG:" and explain what will break.
If the diff is just a style/naming issue, explain why the naming matters.

If the diff is trivial (whitespace, formatting only) and the review adds no value, output: {{"keep": false}}

Diff:
{diff}

Existing review:
{comment}

Output JSON only: {{"keep": true/false, "rewritten": "your improved review"}}"""

# --- MAIN LOOP ---
print("Starting rewrite... (Ctrl+C to stop safely)")

with open(OUTPUT_FILE, "a") as f:
    for sample in tqdm(data, unit="sample"):
        stats["total"] += 1

        diff_hash = hash(sample['input'][:500])
        if diff_hash in processed_hashes:
            stats["skipped_duplicate"] += 1
            continue

        for attempt in range(5):
            try:
                response = active_model.generate_content(
                    PROMPT.format(
                        diff=sample['input'][:5000],
                        comment=sample['output'],
                    ),
                    generation_config={"response_mime_type": "application/json"},
                    safety_settings=safety,
                )

                try:
                    raw = response.text.replace("```json", "").replace("```", "").strip()
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    stats["json_parse_errors"] += 1
                    break

                keep_val = result.get("keep")
                if keep_val in [True, "true", "True"]:
                    rewritten = result.get("rewritten", "").strip()
                    if rewritten and len(rewritten) >= 20:
                        entry = {
                            "instruction": "Review this code change.",
                            "input": sample['input'],
                            "output": rewritten,
                            "repo": sample.get('repo', ''),
                        }
                        f.write(json.dumps(entry) + "\n")
                        f.flush()
                        processed_hashes.add(diff_hash)
                        stats["rewritten"] += 1
                else:
                    stats["filtered"] += 1

                break

            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    stats["rate_limited"] += 1
                    time.sleep(20 * (attempt + 1))
                    continue
                else:
                    stats["api_errors"] += 1
                    if stats["api_errors"] <= 5:
                        print(f"\nAPI Error: {str(e)[:100]}")
                    break

        time.sleep(0.3)

        if stats["total"] % 1000 == 0:
            print(f"\nProgress: {stats}")

# --- FINAL REPORT ---
print("\n" + "=" * 50)
print("REWRITE COMPLETE")
print("=" * 50)
print(f"Total processed: {stats['total']}")
print(f"Rewritten: {stats['rewritten']}")
print(f"Filtered out: {stats['filtered']}")
print(f"Skipped (duplicate/resume): {stats['skipped_duplicate']}")
print(f"API errors: {stats['api_errors']}")
print(f"JSON parse errors: {stats['json_parse_errors']}")
print(f"Rate limited retries: {stats['rate_limited']}")
print(f"\nOutput saved to: {OUTPUT_FILE}")
