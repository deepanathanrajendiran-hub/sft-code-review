"""
Generate DPO preference pairs.
- Chosen: Gemini-rewritten reviews (from train_dataset_rewritten.jsonl)
- Rejected: SFT model's own outputs (on-policy)

NOTE: This script needs GPU for SFT model inference.
      Run in Colab or on a machine with the SFT model available.
"""

import json
from tqdm import tqdm
import torch
from unsloth import FastLanguageModel

# --- CONFIGURATION ---
REWRITTEN_DATA = "train_dataset_rewritten.jsonl"
OUTPUT_FILE = "dpo_pairs.jsonl"
SFT_MODEL_PATH = "code-reviewer-merged"  # or "code-reviewer-lora"
MAX_SAMPLES = 5000
SYSTEM_MSG = "You are a Senior Software Engineer reviewing code changes. Provide clear, actionable feedback."

# --- LOAD SFT MODEL ---
print("Loading SFT model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_MODEL_PATH,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)
print("Model loaded")

# --- LOAD REWRITTEN DATA ---
data = []
with open(REWRITTEN_DATA, 'r') as f:
    for line in f:
        data.append(json.loads(line))
print(f"Loaded {len(data)} rewritten samples")

# Limit to MAX_SAMPLES
data = data[:MAX_SAMPLES]
print(f"Using {len(data)} samples for DPO pair generation")

# --- GENERATE PAIRS ---
dpo_pairs = []
skipped = 0

for sample in tqdm(data, desc="Generating DPO pairs"):
    diff = sample['input'][:4000]
    chosen_review = sample['output']

    # Generate rejected response from SFT model
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Review this code diff:\n\n```diff\n{diff}\n```"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    rejected_review = response.split("assistant")[-1].strip()

    # Quality gate: skip if too similar or rejected is empty
    if len(rejected_review) < 20:
        skipped += 1
        continue

    if rejected_review.lower()[:80] == chosen_review.lower()[:80]:
        skipped += 1
        continue

    # Save in CONVERSATIONAL format (message dicts) for TRL DPOTrainer
    pair = {
        "prompt": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"Review this code diff:\n\n```diff\n{diff}\n```"},
        ],
        "chosen": [
            {"role": "assistant", "content": chosen_review},
        ],
        "rejected": [
            {"role": "assistant", "content": rejected_review},
        ],
    }
    dpo_pairs.append(pair)

print(f"\nGenerated {len(dpo_pairs)} DPO pairs (skipped {skipped})")

# --- SAVE ---
with open(OUTPUT_FILE, 'w') as f:
    for pair in dpo_pairs:
        f.write(json.dumps(pair) + '\n')

print(f"Saved to {OUTPUT_FILE}")
