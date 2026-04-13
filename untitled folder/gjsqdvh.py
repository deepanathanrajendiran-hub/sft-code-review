"""
DPO Pair Quality Diagnostic
Run in Colab after uploading dpo_pairs.jsonl
"""
import json, random, numpy as np
from rouge_score.rouge_scorer import RougeScorer

with open("dpo_pairs.jsonl") as f:
    pairs = [json.loads(l) for l in f]

print(f"Total pairs: {len(pairs)}\n")

# --- 1. Sample 5 pairs side-by-side ---
scorer = RougeScorer(["rougeL"], use_stemmer=True)
random.seed(42)
samples = random.sample(pairs, min(5, len(pairs)))

for i, p in enumerate(samples):
    chosen_text = p["chosen"][-1]["content"]    # last assistant turn
    rejected_text = p["rejected"][-1]["content"]
    overlap = scorer.score(chosen_text, rejected_text)["rougeL"].fmeasure

    print(f"{'='*60}")
    print(f"PAIR {i+1} | ROUGE-L overlap: {overlap:.2f}")
    print(f"{'='*60}")
    print(f"CHOSEN  ({len(chosen_text)} chars):\n{chosen_text[:300]}")
    print(f"\nREJECTED ({len(rejected_text)} chars):\n{rejected_text[:300]}")
    print()

# --- 2. Dataset-wide stats ---
overlaps, c_lens, r_lens = [], [], []
for p in pairs:
    c = p["chosen"][-1]["content"]
    r = p["rejected"][-1]["content"]
    c_lens.append(len(c))
    r_lens.append(len(r))
    overlaps.append(scorer.score(c, r)["rougeL"].fmeasure)

print(f"\n{'='*60}")
print("DATASET-WIDE STATS")
print(f"{'='*60}")
print(f"Chosen  len:  mean={np.mean(c_lens):.0f}  std={np.std(c_lens):.0f}")
print(f"Rejected len: mean={np.mean(r_lens):.0f}  std={np.std(r_lens):.0f}")
print(f"ROUGE-L overlap (chosen vs rejected):")
print(f"  mean={np.mean(overlaps):.3f}  median={np.median(overlaps):.3f}")
print(f"  min={np.min(overlaps):.3f}  max={np.max(overlaps):.3f}")
print(f"\nPairs with overlap > 0.8 (too similar): {sum(1 for o in overlaps if o > 0.8)}/{len(pairs)}")
print(f"Pairs with overlap < 0.3 (good contrast): {sum(1 for o in overlaps if o < 0.3)}/{len(pairs)}")