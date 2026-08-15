# Code-Reviewer LoRA — Qwen2.5-Coder-7B

A fine-tune of `Qwen2.5-Coder-7B-Instruct` that reads a diff and writes a senior-engineer-style review. The shipped model is a LoRA adapter trained with **reasoning-trace SFT** — each training target is a `<think>` block followed by a `<review>` block, distilled from DeepSeek V4-Pro.

It's also an honest experiment log. The SFT model works: it beats the base model 86% head-to-head and halves the hallucination rate. Three rounds of RL on top of it (GRPO, then CoRPO with a verifiable reward) did **not** beat it — and the judge-independent harness built to prove that is, in the end, the more reusable artifact. Both the win and the negative result are written up here.

- **Production model:** `code-reviewer-lora-v4-traces` (LoRA adapter, ~300 MB — weights aren't in git)
- **Metrics:** [`docs/RESULTS.md`](docs/RESULTS.md)
- **Methodology, the RL saga, lessons, references:** [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)

---

## Headline result

SFT v4 vs. the base model, 200-sample in-distribution eval, 3-vote Claude Haiku judge:

| Metric | Base | **SFT v4** |
|---|---|---|
| Pairwise win vs base | — | **86.0%**  (95% CI [81.5%, 91%]) |
| Absolute quality, 0–10 | 4.33 | **5.86** |
| Pass@1 ≥ 7/10 | 7% | **46.0%** |
| Hallucination rate | 14.5% | **7.0%** |
| ROUGE-L vs expert reference | 0.099 | **0.180** |

A few-shot chain-of-thought prompt on the same base model — the "do you even need to fine-tune?" challenger — wins only **13.5%** head-to-head against v4. At this scale the fine-tuning is doing real work.

RL did not improve on this; see [The RL negative result](#the-rl-negative-result).

---

## What it does

Input: a unified diff. Output: a `<think>` reasoning trace, then a `<review>`.

```
diff ──► model.generate(temperature=0, max_tokens=4096, repetition_penalty=1.1) ──► _extract_review ──► review
```

Production inference needs all three sampling settings and the extractor:

```python
from vllm import SamplingParams
SamplingParams(temperature=0, max_tokens=4096, repetition_penalty=1.1)
```

- `repetition_penalty=1.1` is required — greedy decoding without it loops on ~17% of inputs.
- `max_tokens=4096` because v4 traces run ~3500 chars; 1024 truncates 17% of outputs.
- `_extract_review()` (in `run_ood_eval.py`) pulls the real review out: the model sometimes writes a placeholder `<review>...</review>` *inside* its `<think>` block before emitting the actual one after `</think>`. The extractor takes the block after the last `</think>`, falling back to the last non-placeholder block. Without it ~17% of outputs look like failures.

---

## How it was built

```
train_dataset_clean.jsonl  ──►  rewrite (v3)  ──►  generate_traces_gemini.py  ──►  train_dataset_v4_traces_o2.jsonl
 12,488 human PR reviews         12,876            two-call reconciliation              11,309 trace records
 (immutable ground truth)        records           DeepSeek V4-Pro, ~$21, ~5h           (+ audit metadata)
                                                          │
                                                          ▼
                                                    sft.ipynb  (LoRA r=32, α=64, 2 epochs, A100)
                                                          │
                                                          ▼
                                            code-reviewer-lora-v4-traces
```

### 1. Trace generation — two-call reconciliation

The load-bearing design choice. Asking the teacher for `<think>`+`<review>` with the reference review *in the prompt* fails: the traces open with "We need to provide a final review matching the reference," i.e. the model reverse-engineers reasoning to fit the answer, which teaches the student nothing. So generation is split:

- **Call 1** sees only the diff → independent `<think>` + draft review.
- **Call 2** sees diff + draft + reference → a reconciliation label (AGREE / REFERENCE_BETTER / DRAFT_BETTER / BOTH_VALID) + final review.
- **The SFT target** is Call 1's `<think>` plus Call 2's `<review>`. The reconciliation block is kept per-record for audit but never trained on — the student has no reference at inference.

Four grounding filters (`passes_filters`) reject ungrounded traces: the review must name a diff identifier (F1); every backticked identifier in the review must appear in the diff or reference (F2); length sanity (F3); schema (F4). A ~150-entry stopword list — shared with the eval — keeps language keywords and well-known library symbols from counting as hallucinations.

```bash
export DEEPSEEK_API_KEY=sk-...
python generate_traces_gemini.py --sanity          # offline checks
python generate_traces_gemini.py --dry-run 50      # 50-record dry run
MAX_WORKERS=24 python generate_traces_gemini.py    # full run, ~5h, ~$21
```

### 2. Train

Open `sft.ipynb` and run it top to bottom: LoRA r=32, α=64, lr=1e-4, 2 epochs, `max_seq_length=8192`, packing on. One A100, clean curves (train loss 1.6 → 0.84, val 1.03 → 0.88).

> Save the merged model to local SSD (`/content/`) first, then `cp -r` to Drive. Direct Drive saves of 14 GB models get truncated by async sync — we lost 3 of 4 shards to this once.

### 3. Evaluate

- **In-distribution, judge-based:** `eval/eval.ipynb` — 3-vote Haiku pairwise + bootstrap CI, a no-issue probe, a 9×3 failure-pattern suite, a prompt-engineering gauntlet, and a per-repo breakdown.
- **Out-of-distribution, judge-independent:** the harness below — recall against clean defect tuples + a hallucination metric, with no LLM quality judge in the loop.

```bash
# 1) clean defect labels from the eval set (needs DEEPSEEK_API_KEY)
python label_defects.py --input ood_preds_v4.jsonl --output cache/defect_labels.jsonl
# 2) score predictions, judge-independent
python score_v5.py --preds ood_preds_v4.jsonl --labels cache/defect_labels.jsonl --pred-fields v4_pred
# 3) compare two models with a PAIRED bootstrap, not two noisy means (see docs/EXPERIMENTS.md)
python compare_recall.py --preds-a ood_preds_v4.jsonl --field-a v4_pred \
    --preds-b ood_preds_v5.jsonl --field-b v4_pred --labels cache/defect_labels.jsonl
```

### Tests

```bash
python -m pytest -q          # 191 passed, 1 skipped
```

---

## The RL negative result

After v4 shipped, the obvious next move was to push recall up and hallucination down with RL. It didn't work, and *how* it failed is the interesting part:

| Attempt | Reward | Outcome vs v4 (632-record judge-independent OOD) |
|---|---|---|
| GRPO runs #1–#3 | pairwise LLM judge | regressed (−22pp, −52pp, −6pp); the judge was gameable |
| v5.0 (CoRPO) | `0.6·recall + 0.3·grounding + 0.1·length`, F1 | recall **collapsed** 0.094 → 0.070 — the model went quiet |
| v5.1 ckpt-75 | same, F-beta=2 (recall-favoring) | **statistical tie** on both axes; over-flagging on disagreements |
| v5.1 ckpt-150 | same, more training | **worse**: recall down, over-flagging up, a repetition-loop pathology emerged |

**Why.** v4 already sits near its precision/recall frontier. An RL reward that trades recall against precision slides the model *along* that frontier — v5.0 toward quiet, v5.1 toward loud — without pushing it outward. Beating v4 on **both** axes needs a frontier shift, i.e. more capability, which RL on a frozen 7B with diff-only context doesn't supply. This matches the literature: small-model RL restores latent capability but rarely exceeds the SFT teacher (citations in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md#references)).

**Root cause of the over-flagging — it's in the data, not the RL.** Only **8.3%** of v4's 12,876 SFT targets are clean "no-issue" verdicts; **51%** assert a defect. Human PR reviews are *about* problems, so the distilled traces inherited the skew and the model learned to almost always find something — flagging a defect on **77%** of genuinely clean diffs. RL can't fix that. The planned remedy is v4.1: rebalance the SFT data toward restraint, single-stage SFT, no RL. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md#v41-the-data-fix).

---

## Repository map

```
README.md
docs/
  RESULTS.md          metrics: in-distribution + judge-independent OOD
  EXPERIMENTS.md      methodology, the RL saga, lessons, references

# data pipeline
generate_traces_gemini.py   two-call reconciliation trace generator (DeepSeek V4-Pro)
rewrite_data_gemini.py      v3 instruction-format rewrite
generate_patch_data.py      synthetic failure-pattern records (9 patterns)

# training
sft.ipynb                   v4 SFT (LoRA, Unsloth)

# evaluation — judge-based
eval/eval.ipynb             in-distribution: 3-vote pairwise, failure patterns, prompt-eng gauntlet
eval/eval_results.json

# evaluation — judge-independent (the reusable harness)
run_ood_eval.py             OOD prediction generator + _extract_review
ood_metrics.py              recall / hit-rate / hallucination metrics
label_defects.py            PR thread → clean grounded defect tuples
defect_match.py             semantic "was this defect caught?" matcher
score_v5.py                 recall / precision / fp-rate / hallucination
compare_recall.py           paired-bootstrap significance (v5 vs v4)
mid_eval.py                 checkpoint eval during training
eval/ood_eval.ipynb
eval/ood_eval_results_v4.json

# RL (the negative-result experiments)
corpo_reward.py             verifiable reward (recall + grounding + length)
corpo_train.py              CoRPO driver + pre-flight variance gate
corpo_trainer.py
corpo_decision_gate.py
corpo_train.ipynb           Colab driver

# SWE-CARE loading / splits
swecare_loader.py
swecare_split.py

tests/                      191 passing
```

Datasets and prediction dumps (`*.jsonl`) and model weights (`*.zip`, merged checkpoints) are **not** committed — they're large and regenerable. They're gitignored and live locally. `train_dataset_clean.jsonl` (the 12,488 human PR reviews) is the immutable ground truth the pipeline starts from.

---

## Status & next

- ✅ **v4 SFT is production** — beats base 86% pairwise in-distribution and 79% out-of-distribution, hallucination halved. Every in-distribution number is reproduced in [`eval/eval.ipynb`](eval/eval.ipynb)'s stored outputs.
- ⚠️ **v4's known failure mode: it over-flags clean code.** It invents a bug on 3/20 hand-built clean diffs where base invents none, and flags something on ~75% of label-clean OOD diffs (base: ~69%). This is the v4.1 target. (An earlier revision of this README claimed base scored 9/27 on the failure-pattern suite — that was wrong; base scores 27/27 and the suite does not discriminate between models.)
- ⚠️ **RL: v4 still ships, but not because RL failed.** Three rounds were invalidated by pipeline bugs (see below); the one clean run (v5.2) reached recall parity with a *statistically significant* cut in clean-diff false flagging (0.749 → 0.658, paired CI [−0.165, −0.043]). `checkpoint-150` is kept as a low-false-alarm variant — it needs an unclosed-`<think>` guard in the extractor before deployment (4/632 outputs loop).
- 🔜 **v4.1 data rebalance** — add clean-diff / no-issue traces (target ~30% clean, up from 8%) to cut the 77% false-positive rate. Single-stage SFT, no RL. High-confidence on hallucination; recall is capability-limited.
- 🔜 **OOD breadth** — the 86% headline is in-distribution (transformers / sklearn / pydantic / fastapi). The 632-record OOD set is judge-independent, but recall there is ~0.09 for every model, reflecting the 7B's ceiling on diff-only review.
