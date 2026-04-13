import os
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
from tqdm import tqdm
import json
import time

# --- CONFIGURATION ---
PROJECT_ID = "halogen-sol-348516"  # Your Project ID
LOCATION = "us-central1"
INPUT_FILE = "results.jsonl"  # Changed to JSONL
OUTPUT_FILE = "train_dataset_final_1.jsonl"
LOG_FILE = "distillation_log.json"

# --- INIT ---
print(f"🚀 Connecting to Vertex AI in {LOCATION}...")
vertexai.init(project=PROJECT_ID, location=LOCATION)

active_model = None
for name in ["gemini-2.0-flash-001", "gemini-1.5-flash"]:
    try:
        model = GenerativeModel(name)
        model.generate_content("test")
        active_model = model
        print(f"✅ Connected to: {name}")
        break
    except:
        continue

if not active_model:
    raise RuntimeError("❌ Could not connect to any Gemini model.")

# --- LOAD DATA ---
df = pd.read_json(INPUT_FILE, lines=True)
print(f"📂 Loaded {len(df)} rows.")

# --- RESUME LOGIC ---
processed_indices = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
                # Track by diff hash to detect duplicates
                processed_indices.add(hash(entry.get('input', '')[:500]))
            except:
                pass
    print(f"ℹ️  Found {len(processed_indices)} already processed. Resuming...")

# --- COUNTERS ---
stats = {
    "total": 0,
    "skipped_null": 0,
    "skipped_duplicate": 0,
    "kept": 0,
    "filtered": 0,
    "api_errors": 0,
    "json_parse_errors": 0,
    "rate_limited": 0
}

# --- MAIN LOOP ---
safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

PROMPT = """
Rewrite this code review comment as a clear, actionable instruction.
If it's social pleasantry, vague, or just a question without substance, output: {{ "keep": false }}

Diff:
{diff}

Comment:
{comment}

Output JSON only: {{ "keep": true/false, "rewritten": "..." }}
"""

print("🚀 Starting Distillation... (Ctrl+C to stop safely)")

with open(OUTPUT_FILE, "a") as f:
    for index, row in tqdm(df.iterrows(), total=len(df), unit="row"):
        stats["total"] += 1
        
        # Skip nulls
        if pd.isna(row.get('diff_hunk')) or pd.isna(row.get('review_comment')):
            stats["skipped_null"] += 1
            continue
        
        # Skip already processed (for resume)
        diff_hash = hash(str(row['diff_hunk'])[:500])
        if diff_hash in processed_indices:
            stats["skipped_duplicate"] += 1
            continue

        for attempt in range(5):
            try:
                response = active_model.generate_content(
                    PROMPT.format(
                        diff=str(row['diff_hunk'])[:5000], 
                        comment=row['review_comment']
                    ),
                    generation_config={"response_mime_type": "application/json"},
                    safety_settings=safety
                )
                
                # Parse response
                try:
                    raw = response.text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    stats["json_parse_errors"] += 1
                    break
                
                # Check keep (handle both bool and string)
                keep_val = data.get("keep")
                if keep_val in [True, "true", "True"]:
                    if data.get("rewritten"):
                        entry = {
                            "instruction": "Review this code change.",
                            "input": row['diff_hunk'],
                            "output": data['rewritten'],
                            "repo": row['repository']
                        }
                        f.write(json.dumps(entry) + "\n")
                        f.flush()
                        processed_indices.add(diff_hash)
                        stats["kept"] += 1
                else:
                    stats["filtered"] += 1
                
                break  # Success, exit retry loop
                
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    stats["rate_limited"] += 1
                    time.sleep(20 * (attempt + 1))
                    continue
                else:
                    stats["api_errors"] += 1
                    if stats["api_errors"] <= 5:
                        print(f"\n⚠️ API Error: {str(e)[:100]}")
                    break

        time.sleep(0.3)  # Rate limit buffer
        
        # Periodic stats (every 1000 rows)
        if stats["total"] % 1000 == 0:
            print(f"\n📊 Progress: {stats}")

# --- FINAL REPORT ---
print("\n" + "="*50)
print("✅ DISTILLATION COMPLETE")
print("="*50)
print(f"Total processed: {stats['total']}")
print(f"Kept: {stats['kept']}")
print(f"Filtered out: {stats['filtered']}")
print(f"Skipped (null): {stats['skipped_null']}")
print(f"Skipped (duplicate): {stats['skipped_duplicate']}")
print(f"API errors: {stats['api_errors']}")
print(f"JSON parse errors: {stats['json_parse_errors']}")
print(f"Rate limited retries: {stats['rate_limited']}")
print(f"\n📁 Output saved to: {OUTPUT_FILE}")