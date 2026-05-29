# Code-Reviewer LoRA — Qwen2.5-Coder-7B

Fine-tuning `unsloth/Qwen2.5-Coder-7B-Instruct` into a specialized **code-review** model that reads a diff and writes a reviewer-style critique. The shipped model is a LoRA adapter trained with **reasoning-trace SFT** (per-record `<think>` + `<review>` targets) distilled from DeepSeek V4-Pro.

This repo is also an **honest experiment log**: it documents both what worked (trace-distillation SFT, which decisively beats the base model) and what didn't (three rounds of RL — GRPO and CoRPO — which never beat the SFT model on a capability-capped 7B). The negative RL result, and the judge-independent evaluation harness built to prove it, are first-class deliverables here.

- **Production model:** `code-reviewer-lora-v4-traces` (LoRA adapter, ~300 MB)
- **Detailed metrics:** [`docs/RESULTS.md`](docs/RESULTS.md)
- **Full experiment log + methodology + lessons:** [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- **Decision log (chronological):** [`journal.md`](journal.md)

---

## Headline result

**SFT v4 (trace distillation) vs. the base model**, 200-sample in-distribution eval (3-vote Haiku judge):

| Metric | Base | **SFT v4** |
|---|---|---|
| Pairwise win vs base | — | **86.0%**  CI [81.5%, 91%] |
| Haiku absolute mean (0–10) | 4.33 | **5.86** |
| Pass@1 ≥ 7 | 7% | **46.0%** |
| Hallucination rate | 14.5% | **7.0%** |
| ROUGE-L vs expert reference | 0.099 | **0.180** |

A few-shot CoT prompt-engineering baseline on the same base model wins only **13.5%** pairwise vs v4 — fine-tuning is carrying real weight at this scale.

**RL did not improve on this.** GRPO (judge reward) and CoRPO (verifiable reward) were tried across three rounds; none beat v4 on a held-out, judge-independent eval. See [The RL negative result](#the-rl-negative-result).

---

## What this model does

Input: a unified `diff`. Output: a `<think>` reasoning trace followed by a `<review>` block — a senior-engineer-style review of the change.

```
diff ──► tokenizer ──► model.generate(
                          temperature=0,
                          max_tokens=4096,
                          repetition_penalty=1.1,   # required — greedy w/o this loops on ~17% of inputs
                      ) ──► _extract_review ──► final review
```

### Required inference settings (production)

```python
from vllm import SamplingParams
SamplingParams(temperature=0, max_tokens=4096, repetition_penalty=1.1)
```

Plus the `_extract_review()` post-processor (`run_ood_eval.py`): the trained model sometimes writes a placeholder `<review>...</review>` *inside* its `<think>` block before emitting the real review after `</think>`. The extractor takes the `<review>` after the last `</think>`, falling back to the last non-placeholder block. Without it, ~17% of outputs extract as the placeholder dots and look like failures.

---

## How it was built (replication)

```
train_dataset_clean.jsonl   ──►  rewrite (v3)  ──►  generate_traces_gemini.py  ──►  train_dataset_v4_traces_o2.jsonl
  12,488 human PR reviews          12,876            two-call reconciliation,           11,309 trace records
  (IMMUTABLE ground truth)         records           DeepSeek V4-Pro, ~$21, ~5h          (+ audit metadata)
                                                            │
                                                            ▼
                                                       sft.ipynb  (LoRA r=32, α=64, 2 epochs, A100)
                                                            │
                                                            ▼
                                              code-reviewer-lora-v4-traces  (production)
```

### 1. Generate reasoning traces (one-shot; v4 is complete)

The load-bearing design choice is **two-call reconciliation** — single-call generation with the reference review in the prompt made the model reverse-engineer reasoning to fit the given conclusion ("We need to provide a final review matching the reference…"). Instead:

- **Call 1** sees only the diff → independent `<think>` + draft review.
- **Call 2** sees diff + draft + reference → reconciliation label (AGREE / REFERENCE_BETTER / DRAFT_BETTER / BOTH_VALID) + final review.
- **SFT target** = Call 1's `<think>` + Call 2's `<review>`. The reconciliation block is kept per-record for audit but is **not** in the training target (the student has no reference at inference).

```bash
export DEEPSEEK_API_KEY=sk-...
python generate_traces_gemini.py --sanity          # offline checks
python generate_traces_gemini.py --dry-run 50      # 50-record dry run
MAX_WORKERS=24 nohup caffeinate -i -s .venv/bin/python generate_traces_gemini.py \
  > trace_gen_run.log 2>&1 &                        # ~5h, ~$21 at 24 workers
```

Four safety filters (`passes_filters`) reject ungrounded traces: review must mention ≥1 diff identifier (F1); every backticked identifier in the review must appear in the diff or the reference (F2); length sanity (F3); schema compliance (F4). A ~150-entry stopword list (kept in sync with the eval) prevents false hallucination flags on language keywords / well-known library symbols.

### 2. Train (Colab / RunPod A100 80 GB)

Open `sft.ipynb`, run cells in order. LoRA r=32, `lora_alpha=64`, lr=1e-4, 2 epochs, `max_seq_length=8192`, packing on.

> **Critical:** save the merged model to `/content/` (local SSD) first, then `cp -r` to Drive. Direct Drive saves of 14 GB models get truncated by async sync (we lost 3 of 4 shards to this once).

### 3. Evaluate

- **In-distribution (200 samples, judge-based):** `untitled folder/eval.ipynb` — 3-vote Haiku pairwise + bootstrap CI, no-issue probe, 9×3 failure-pattern suite, prompt-engineering gauntlet, per-repo breakdown.
- **Out-of-distribution (632 records, judge-independent):** the v5 harness below — recall against clean defect tuples + hallucination, **no LLM quality judge in the loop.**

```bash
# 1) build clean defect labels from the eval set (needs DEEPSEEK_API_KEY)
python label_defects.py --input ood_preds_v4.jsonl --output cache/defect_labels.jsonl
# 2) score any model's predictions, judge-independent
python score_v5.py --preds ood_preds_v4.jsonl --labels cache/defect_labels.jsonl --pred-fields v4_pred
# 3) compare two models with a PAIRED bootstrap (not two noisy means — see docs/EXPERIMENTS.md)
python compare_recall.py --preds-a ood_preds_v4.jsonl --field-a v4_pred \
    --preds-b v5_preds.jsonl --field-b corpo_pred --labels cache/defect_labels.jsonl
```

### Tests

```bash
python -m pytest -q          # 191 passing, 1 skipped
```

---

## The RL negative result

After v4 shipped, we tried to push recall up / hallucination down with RL. **It didn't work**, and the way it failed is the interesting part:

| Attempt | Reward | Outcome vs v4 (632-record judge-independent OOD) |
|---|---|---|
| GRPO Runs #1–#3 | pairwise LLM judge | regressed (−22pp, −52pp, −6pp); judge was gameable |
| v5.0 (CoRPO) | `0.6·recall + 0.3·grounding + 0.1·length`, F1 | recall **collapsed** 0.094 → 0.070 (model went quiet) |
| v5.1 ckpt-75 | same, F-beta=2 (recall-favoring) | **statistical tie** on both axes; over-flagging on disagreements |
| v5.1 ckpt-150 | same, more training | **worse**: recall ↓, over-flagging ↑, repetition-loop pathology emerged |

**Conclusion:** on a capability-capped 7B reviewing diff-only context, RL redistributes *assertiveness* along the precision/recall frontier — it doesn't push the frontier outward, so it can't beat a model (v4) already sitting near that frontier — "RL restores, rarely exceeds" (literature in [`docs/EXPERIMENTS.md#references`](docs/EXPERIMENTS.md#references)).

**Root cause of the over-flagging**, found in the training data: only **8.3%** of v4's 12,876 training targets are clean "no-issue" verdicts vs **51%** that assert a defect. Human PR reviews are *about* problems, so the distilled traces inherited the skew, and the model learned to almost always find something — flagging a defect on **77%** of genuinely clean diffs. **This is a data problem RL cannot fix.** The planned remedy (v4.1) is to rebalance the SFT data toward restraint. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md#v41-the-data-fix).

---

## Repository map

| Path | What |
|---|---|
| `sft.ipynb` | v4 SFT training notebook (production) |
| `generate_traces_gemini.py` | Two-call reconciliation trace generator (DeepSeek V4-Pro) |
| `train_dataset_clean.jsonl` | 12,488 human PR reviews — **IMMUTABLE ground truth** |
| `train_dataset_v4_traces_o2.jsonl` | v4 SFT input (11,309 trace records + audit metadata) |
| `run_ood_eval.py` | OOD prediction generator + `_extract_review` post-processor |
| `label_defects.py` | PR thread → clean grounded defect tuples (DeepSeek; injectable classifier) |
| `defect_match.py` | Semantic "was this defect caught?" matcher + recall |
| `score_v5.py` | Judge-independent eval: recall / precision / fp-rate / hallucination |
| `compare_recall.py` | **Paired bootstrap** significance for recall/fp/halluc deltas (v5 vs v4) |
| `corpo_reward.py`, `corpo_train.py`, `corpo_train.ipynb` | CoRPO RL pipeline (the negative-result experiments) |
| `untitled folder/eval.ipynb` | 14-cell in-distribution eval pipeline |
| `journal.md` | Full chronological decision log |

GB-scale model weights are **not** committed (see `.gitignore`).

---

## Status & future work

- ✅ **v4 SFT is production** — beats base 86% pairwise, halved hallucination, 27/27 failure-pattern tests.
- ❌ **RL (GRPO/CoRPO) dropped** — three rounds, no improvement over v4; documented as a negative result.
- 🔜 **v4.1 data rebalance** — add clean-diff / no-issue trace examples (target ~30% clean, up from 8%) to cut the 77% false-positive rate. Single-stage SFT, no RL. High-confidence on the hallucination axis; recall is capability-limited.
- 🔜 **OOD breadth** — the headline 86% is in-distribution (transformers / sklearn / pydantic / fastapi). The 632-record OOD set is judge-independent but recall there is low (~0.09) for all models, reflecting the 7B's ceiling on diff-only review.
