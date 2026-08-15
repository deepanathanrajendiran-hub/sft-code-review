# Code-Reviewer LoRA — Qwen2.5-Coder-7B

A fine-tune of `Qwen2.5-Coder-7B-Instruct` that reads a diff and writes a senior-engineer-style review. The shipped model is a LoRA adapter trained with **reasoning-trace SFT** — each training target is a `<think>` block followed by a `<review>` block, distilled from DeepSeek V4-Pro.

It's also an honest experiment log. The SFT model works: it beats the base model 86% head-to-head in-distribution, 79% out-of-distribution, and halves the hallucination rate. Four rounds of RL followed. A later pipeline audit invalidated the first three — they had been training on left-truncated prompts. The one clean run reached recall parity with a *statistically significant* cut in false flagging on clean diffs: not enough to displace v4 under the pre-registered ship rule, but the project's first real RL gain. The judge-free harness that settled it is the more reusable artifact. The wins, the invalidated runs, and the corrections are all written up here.

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

Every figure above is reproduced in [`eval/eval.ipynb`](eval/eval.ipynb)'s stored outputs. And it holds up off-distribution — on 632 records from 88 unseen repos, v4 wins **79.0%** of pairwise comparisons (499–121–12, 95% CI [75.8%, 82.1%]).

A few-shot chain-of-thought prompt on the same base model — the "do you even need to fine-tune?" challenger — wins only **13.5%** head-to-head against v4. At this scale the fine-tuning is doing real work.

RL reached recall parity with a significant false-alarm reduction; v4 still ships. See [The RL arc](#the-rl-arc-three-invalidated-rounds-then-one-clean-run).

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

Four grounding filters (`passes_filters`) reject ungrounded traces: the review must name a diff identifier (F1); every backticked identifier in the review must appear in the diff or reference (F2); length sanity (F3); schema (F4). A 278-term stopword list — byte-identical between the generator's F2 filter and the eval's hallucination metric — keeps language keywords and well-known library symbols from counting as hallucinations.

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
- **Out-of-distribution, no quality judge:** the harness below — recall against grounded defect tuples + a hallucination metric, with no LLM *preference* judge anywhere in the metric. (A constrained LLM still does the defect labelling and the "was this caught?" matching; what's removed is the "which review is better" judge.)

```bash
# 1) clean defect labels from the eval set (needs DEEPSEEK_API_KEY)
python label_defects.py --input ood_preds_v4.jsonl --output cache/defect_labels.jsonl
# 2) score predictions (no quality judge)
python score_v5.py --preds ood_preds_v4.jsonl --labels cache/defect_labels.jsonl --pred-fields v4_pred
# 3) compare two models with a PAIRED bootstrap, not two noisy means (see docs/EXPERIMENTS.md)
python compare_recall.py --preds-a ood_preds_v4.jsonl --field-a v4_pred \
    --preds-b ood_preds_v5.jsonl --field-b v4_pred --labels cache/defect_labels.jsonl
```

### Tests

```bash
pip install pytest numpy openai
python -m pytest -q     # 193 passed, 2 skipped   (no GPU deps)
                        # 201 passed, 1 skipped   (with torch + trl installed)
```

---

## The RL arc: three invalidated rounds, then one clean run

After v4 shipped, the obvious next move was to push recall up and false flagging down with RL. Three rounds said no — and then a code review of the *pipeline* (not the results) found five defects meaning those three rounds had measured nothing at all.

| Attempt | Reward | Outcome vs v4 (632-record OOD) | status |
|---|---|---|---|
| GRPO runs #1–#3 | pairwise LLM judge | regressed (−22pp, −52pp, −6pp); the judge was gameable | **invalidated** |
| v5.0 | verifiable reward, F1 | recall **collapsed** 0.094 → 0.070 — the model went quiet | **invalidated** |
| v5.1 ckpt-75 | same, F-beta=2 (recall-favoring) | statistical tie on both axes; over-flagging on disagreements | **invalidated** |
| v5.1 ckpt-150 | same, more training | worse: recall down, over-flagging up, repetition-loop pathology | **invalidated** |
| **v5.2 ckpt-150** | same, `RECALL_BETA=1.5`, true v4 KL anchor, full prompts | recall **parity** (paired Δ −0.018, CI [−0.065, +0.026]); **clean-diff flag rate 0.749 → 0.658, CI [−0.165, −0.043] — significant** | **valid** |

**The five defects.** TRL's `GRPOConfig` defaults `max_prompt_length` to 512 and *left*-truncates: ~87% of training prompts lost the system message and most of the diff, so the policy was scored on defects it could not see. The KL term meant to anchor the policy to v4 was computed by PEFT with adapters disabled — i.e. it anchored to *base*, actively pulling the policy away from v4. Truncated `<think>` blocks fell through the extractor and were scored as reviews. Records whose defect comments failed diff-grounding were trained as "find nothing" on diffs containing real defects. And mid-training eval gave checkpoints 5000-char diffs against the v4 baseline's 12000-char budget. Two of the five are now structurally unrepeatable: `corpo_train.py` hard-exits without an explicitly merged reference model, and asserts at pre-flight that TRL still exposes the truncation knob.

**What the one clean run shows.** v5.2 trained healthily — reward 0.33 → 0.64, completion truncations 46% → 5%, KL ≤ 0.0034, no length collapse, 372 steps. On the single pre-registered confirmatory test it cut clean-diff flagging significantly at recall parity, and the restraint is real rather than an artifact: review-length tail compression (mean 898 → 516 chars, median unchanged), identifiers per review 6.0 → 3.2, and correct "no issues" verdicts on diffs where v4 fabricates. **v4 still ships**, because the pre-registered ship rule required a significant *recall* gain and the n=50 mid-eval's promising +0.073 regressed to exact parity at n=632 — textbook winner's curse, which is why the protocol existed.

**Why recall didn't move.** Not because the model had no room — the [oracle ceiling](docs/RESULTS.md) measures ~4× headroom on the same diff-only prompts. The reward's quality term is `F₁.₅(recall, precision)`, which is *identically zero* whenever a rollout catches nothing. So on the ~90% of labeled diffs the student cannot yet solve, all 8 rollouts in a group score the same on the axis that mattered, and the only live contrast came from the grounding and length terms — while clean-record restraint (53% of training prompts) was densely and cheaply rewarded. The gradient wasn't absent; it was dense on one axis and degenerate on the other, which predicts the observed result exactly. (Leading hypothesis, not a measurement — per-component reward variance was never logged. That's the first instrument to add.)

**The over-flagging is mostly a data problem.** Only **8.3%** of the 12,876-record v3 corpus the v4 traces were distilled from carries a clean "no-issue" verdict; **51%** assert a defect. Human PR reviews are *about* problems, so the traces inherited the skew and the model learned to almost always find something. But note the earlier claim that "RL can't fix that" was wrong — v5.2 cut clean-diff flagging significantly. Data is the bigger lever, not the only one. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md#v41-the-data-fix).

> **Note on the algorithm actually run.** The trainer implements the CoRPO clipped baseline `max(R_min, group_mean)`, but the executed v5.2 run auto-selected `R_MIN = 0.0` from the variance gate's p33. Because the reward is non-negative by construction, that clamp is an identity operation — so the shipped run is **Dr. GRPO with a group-mean baseline**, in a CoRPO-capable trainer. Recorded here rather than quietly relabelled.

---

## Repository map

```
README.md
docs/
  RESULTS.md          metrics: in-distribution + OOD (no quality judge)
  EXPERIMENTS.md      methodology, the RL saga, lessons, references

# data pipeline
generate_traces_gemini.py   two-call reconciliation trace generator (DeepSeek V4-Pro)
rewrite_data_gemini.py      v3 instruction-format rewrite
generate_patch_data.py      synthetic failure-pattern records (9 patterns)

# training
sft.ipynb                   v4 SFT (LoRA, Unsloth)

# evaluation — judge-based
eval/eval.ipynb             in-distribution: 3-vote pairwise, failure patterns, prompt-eng gauntlet
eval/eval_results.json      NOTE: v3-era dump; the v4 numbers live in eval.ipynb's stored outputs

# evaluation — no quality judge (the reusable harness)
run_ood_eval.py             OOD prediction generator + _extract_review
ood_metrics.py              hit-rate / hallucination metrics + the OOD pairwise judge (judged, not judge-free)
label_defects.py            PR thread → clean grounded defect tuples
defect_match.py             semantic "was this defect caught?" matcher
score_v5.py                 recall / precision / fp-rate / hallucination
compare_recall.py           paired-bootstrap significance (v5 vs v4)
oracle_ceiling.py           frontier-teacher recall ceiling on the same prompts (the 0.388 result)
mid_eval.py                 checkpoint eval during training
eval/ood_eval.ipynb
eval/ood_eval_results_v4.json

# RL (the negative-result experiments)
corpo_reward.py             verifiable reward (recall + grounding + length)
corpo_train.py              CoRPO driver + pre-flight variance gate
corpo_trainer.py
corpo_decision_gate.py      superseded Run-#3 judge gate; v5.2 shipped on compare_recall.py
corpo_train.ipynb           Colab driver

# SWE-CARE loading / splits
swecare_loader.py
swecare_split.py

tests/                      201 passing (193 without torch/trl)
```

Datasets and prediction dumps (`*.jsonl`) and model weights (`*.zip`, merged checkpoints) are **not** committed — they're large and regenerable. They're gitignored and live locally. `train_dataset_clean.jsonl` (the 12,488 human PR reviews) is the immutable ground truth the pipeline starts from.

---

## Status & next

- ✅ **v4 SFT is production** — beats base 86% pairwise in-distribution and 79% out-of-distribution, hallucination halved. Every in-distribution number is reproduced in [`eval/eval.ipynb`](eval/eval.ipynb)'s stored outputs.
- ⚠️ **v4's known failure mode: it over-flags clean code.** It invents a bug on 3/20 hand-built clean diffs where base invents none, and flags something on ~75% of label-clean OOD diffs (base: ~69%). This is the v4.1 target. (An earlier revision of this README claimed base scored 9/27 on the failure-pattern suite — that was wrong; base scores 27/27 and the suite does not discriminate between models.)
- ⚠️ **RL: v4 still ships, but not because RL failed.** Three rounds were invalidated by pipeline bugs (see below); the one clean run (v5.2) reached recall parity with a *statistically significant* cut in clean-diff false flagging (0.749 → 0.658, paired CI [−0.165, −0.043]). `checkpoint-150` is kept as a low-false-alarm variant — it needs an unclosed-`<think>` guard in the extractor before deployment (4/632 outputs loop).
- 🔜 **v4.1 data rebalance** — add clean-diff / no-issue traces (target ~30% clean, up from 8%) to cut the 77% false-positive rate. Single-stage SFT, no RL. High-confidence on hallucination; recall is capability-limited.
- 🔜 **Recall is the open problem** — ~0.09 for every *student* model on the OOD set, but a frontier teacher scores **0.388** on the same diff-only prompts (`oracle_ceiling.py`). The gap is training, not context, so oracle trace distillation is the next lever — and the labels need enriching first (even the teacher "false-alarms" on 97% of label-clean records, so that metric measures thread-agreement, not correctness).
