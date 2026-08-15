# Experiments & Methodology

This is the full technical log: the data pipeline, the evaluation harness (the reusable part), and the RL experiments — three rounds invalidated by pipeline defects, then one clean run that produced a significant gain on one axis and parity on the other. For the metrics tables see [`RESULTS.md`](RESULTS.md).

## Contents
- [Data pipeline: two-call reconciliation](#data-pipeline-two-call-reconciliation)
- [Training (v4)](#training-v4)
- [Evaluation methodology](#evaluation-methodology)
- [The RL saga: three invalidated rounds, then one clean run](#the-rl-saga-three-invalidated-rounds-then-one-clean-run)
- [Why RL moved restraint but not recall](#why-rl-moved-restraint-but-not-recall)
- [v4.1: the data fix](#v41-the-data-fix)
- [Lessons](#lessons)
- [Replicating the RL experiments](#replicating-the-rl-experiments)

---

## Data pipeline: two-call reconciliation

Start: `train_dataset_clean.jsonl` — 12,488 real human PR-review comments (immutable). These were rewritten into a cleaner instruction format (v3), then distilled into reasoning traces for v4 — reasoning-trace distillation into a smaller student, in the spirit of DeepSeek-R1's distillation result [9].

The naive approach — ask the teacher (DeepSeek V4-Pro) for a `<think>`+`<review>` with the reference review *in the prompt* — fails: sampling 20 such traces, they opened with *"We need to provide a final review matching the reference review's conclusion."* The model reverse-engineers reasoning to land on the given answer, which teaches the student nothing about *how* to reason.

**Fix — split into two calls:**

| | Sees | Produces |
|---|---|---|
| **Call 1** | diff only | independent `reasoning_content` + draft review |
| **Call 2** | diff + Call-1 draft + reference | reconciliation label + final review |

- **SFT target** = Call 1's `<think>` ⊕ Call 2's `<review>`. The student gets genuine from-scratch reasoning paired with a reference-reconciled conclusion.
- The **reconciliation block** (AGREE / REFERENCE_BETTER / DRAFT_BETTER / BOTH_VALID + rationale) is preserved per record for audit but is **not** a training target — the student has no reference at inference time.

Reconciliation-label distribution on the final 11,309 traces: AGREE 5.3%, REFERENCE_BETTER 13.8%, DRAFT_BETTER 31.4%, BOTH_VALID 15.9%, none 33.6%. The 31% DRAFT_BETTER rate (teacher overrode the human reference) matches the independently-measured 31% nitpick rate in the original human reviews — the teacher is correctly filtering nitpicks, not fabricating. A manual audit of 50 stratified DRAFT_BETTER samples found 0/50 wrong overrides.

**Four grounding filters** (`generate_traces_gemini.py::passes_filters`):

1. **F1** — the `<review>` must mention ≥1 identifier from the diff's +/− lines (unless the reference was also identifier-free).
2. **F2** — every backticked identifier in the `<review>` must appear in the diff or the reference. *(F2 on `<think>` was disabled — Call-1's exploratory reasoning legitimately names library APIs like `JSONResponse`/`JWTError` that aren't in the diff.)*
3. **F3** — length sanity: `<think>` ≤ 10k chars; `<review>` within 0.5×–2.5× the reference length.
4. **F4** — schema: both blocks present and non-empty.

A 278-term stopword list (Python keywords, exception/typing vocab, common backticked English, well-known third-party APIs) is **shared** between the generator's F2 filter and the eval's hallucination metric, so "hallucination" means the same thing on both sides.

Yield: 11,309 trace records from 12,876 inputs (87.8%).

---

## Training (v4)

| param | value |
|---|---|
| base | `unsloth/Qwen2.5-Coder-7B-Instruct` |
| LoRA r | 32 |
| `lora_alpha` | 64 *(2× v3 — more headroom for trace targets)* |
| `lora_dropout` | 0.05 |
| learning rate | 1e-4 |
| epochs | 2 |
| `max_seq_length` | 8192 |
| packing | True |

Single-stage SFT, no preference optimization. (DPO was tried twice — v4, v5-DPO — and dropped; both collapsed back to SFT outputs.)

> **Save-order gotcha:** write the merged 16-bit model to local SSD (`/content/`) first, *then* `cp -r` to Drive. Direct Drive saves of 14 GB models are truncated by async sync — we once got 3 of 4 shards saved as 0 bytes.

---

## Evaluation methodology

> Metric definitions, with the code that computes each one: [`METRICS.md`](METRICS.md).

### In-distribution (judge-based)

`eval/eval.ipynb` — vLLM generation for base + SFT, then: 3-vote Haiku pairwise with bootstrap CI, absolute Haiku scoring, ROUGE/BLEU vs reference, a no-issue probe, a 9-pattern × 3 failure-mode suite, a prompt-engineering gauntlet (v4 vs few-shot CoT on the same base), and a per-repo breakdown. Empty-review-bug and single-vote issues from the v3 era were patched here.

### Out-of-distribution (judge-independent) — the reusable harness

The judge-based eval has a single point of failure: the Haiku judge. For the RL work we built a harness with **no LLM quality judge in the reward or the metric**:

| file | role |
|---|---|
| `label_defects.py` | turns a PR review thread into clean, grounded **defect tuples** `{path, line, issue_type, canonical_desc}` (DeepSeek; classifier injectable for tests) |
| `defect_match.py` | semantic "was this defect caught?" matcher → recall; claim counter → precision |
| `score_v5.py` | aggregates recall / precision / fp-rate / hallucination over the 632 OOD records (16-way parallel) |
| `compare_recall.py` | **paired bootstrap** significance for v5-vs-v4 deltas |

**The paired-bootstrap insight (`compare_recall.py`).** Comparing two models by their mean recall is the wrong test. Per-record recall is ≈ {0, 0.5, 1.0} (std ≈ 0.4); over ~150 labeled records the standard error on the mean is ≈ 0.033 — *larger* than the v4-vs-v5 differences we were trying to detect. A Monte Carlo on this data:

| n_labeled | true Δrecall | **paired** detects | unpaired (two-means) detects |
|---|---|---|---|
| 150 | +0.038 | **100%** | 4% |
| 150 | +0.043 (noisy) | **97%** | 12% |
| 300 | +0.038 | **100%** | 31% |

A two-means comparison **misses a real +0.04 recall gain ~90% of the time** at this sample size; the paired test catches it ~100% while keeping the false-positive rate at the nominal floor. Every v5-vs-v4 recall claim in this repo uses the paired bootstrap test [10].

**The backtick-metric blind spot.** The hallucination metric counts backticked identifiers absent from the diff — i.e. *lexical* fabrication (inventing identifier names). It is **blind to semantic over-flagging**: asserting a bug that doesn't exist *using identifiers that are in the diff* (e.g. "`Tuple` is undefined → NameError" when `Tuple` is imported in the diff). v5's real failure mode was semantic, so the aggregate hallucination metric *understated* it. We caught this only with a **disagreement analysis** — sampling records where v4 and v5 differ and reading the diff + ground truth to classify each as a real catch vs a fabrication.

---

## The RL saga: three invalidated rounds, then one clean run

**Motivation.** v4 ships, but recall is ~0.09 and it flags something on ~75% of clean diffs — worse than its own base model (0.689). Could RL push recall up and flagging down? Three rounds said no; a pipeline audit then found those three rounds had measured nothing, and the one clean rerun split the difference.

### Round 1 — GRPO with a pairwise LLM judge (Runs #1–#3)

Reward = does this rollout beat an opponent, per an LLM judge. Every run regressed (−22pp, −52pp, −6pp vs v4). Root causes, found via a literature sweep + ablation: (a) wrong opponent (compared vs *base*, not vs v4); (b) `kl_beta = 0` removed the only anchor to v4 → capability/format drift; (c) the judge saw only `review[:500]` and was gameable on length/structure — LLM judges are known to carry verbosity/position biases [7]. GRPO [1] with an ordinal/gameable reward was dropped. **CoRPO** [2] — which clips the group baseline at a correctness threshold so failed rollouts are never positively reinforced — was adopted to fix the ordinal-reward failure mode, with the **Dr. GRPO** loss [3] for length-bias.

### Round 2 — v5 verifiable reward

Replace the gameable judge with a **verifiable** reward:

```
R = 0.6 · quality + 0.3 · grounding(1 − halluc) + 0.1 · length_sanity
    quality(labeled) = F_beta(recall, precision)    # recall = caught/known, precision = caught/claims
    quality(clean)   = max(0, 1 − penalty · n_claims)
```

No opponent, no quality judge → the proxy-reward Goodhart overoptimization [8] that the gameable pairwise judge invited can't apply the same way; over-claiming is penalized by the precision term. `kl_beta = 0.02` restores the v4 anchor.

- **v5.0 (F1, i.e. beta=1):** recall **collapsed** 0.094 → 0.070. The clean-record penalty dominates (~53% of training data is clean-ish), so the model learned to go quiet — fewer claims, lower recall.
- **v5.1 (F-beta = 2):** rebalanced to favor recall 4× precision, to undo the collapse. At ckpt-75 it produced a **statistical tie** with v4 on every aggregate axis — but the disagreement analysis showed it had slid the *other* way: on 275 records where the models disagreed, v5.1 fabricated a concrete bug v4 correctly waved off **11× in 45 samples** vs 3 real catches. The aggregate hallucination metric (backtick) couldn't see this because the fabrications used real diff identifiers.
- **v5.1 ckpt-150 (more training):** got **worse** — louder (review length 1027 → 1310, assertions 8.2 → 11.4 per review), recall down (0.094 → 0.080), over-flagging up (138 → 162 solo-asserts), and a new **repetition-loop pathology** (a review emitting "Bug:" hundreds of times) that worsened with training (22 → 29 → 35 records).

### Round 2.5 — the pipeline audit that changed the story

Before running another reward variant, a code review of the pipeline (not the results) found **five
defects that had invalidated every run above**:

1. **`max_prompt_length` was never set** — TRL's GRPOConfig defaults to 512 and *left-truncates*; with
   SWE-CARE diffs (median 6.8k chars), ~87% of training prompts lost the system message and most of the
   diff. The policy was scored on recall against defects it literally could not see — in Runs #1–#3
   *and* v5.0/v5.1. The variance gate generated full prompts, so it never measured the training regime.
2. **The KL anchor pointed at base, not v4.** The policy was loaded as base + v4-adapter (a PEFT model),
   and TRL computes the PEFT reference with adapters disabled — i.e. raw base. `kl_beta=0.02` was
   actively pulling the policy *away* from v4. Fixed by loading merged-v4 weights + a fresh LoRA.
3. **Truncated rollouts leaked reasoning into the reward** — an unclosed `<think>` fell through the
   extractor as the "review" and got partial credit. Now scores 0.
4. **Ambiguous "clean" labels** — records whose defect comments failed diff-grounding were trained as
   "find nothing" on diffs with real defects. Now excluded.
5. **mid-eval prompt asymmetry** — checkpoints got 5000-char diffs vs the v4 baseline's 12000-char
   budget. Now identical.

The earlier "RL is dead" conclusion was therefore drawn from broken experiments. The elaborate
root-cause analyses (literature sweeps, adversarial agent passes) had validated *conceptual* causes
while missing config-default plumbing — the lesson is to inspect the actual tensors entering the
trainer before reaching for papers.

### Round 3 — v5.2: the fixed-pipeline run (one pre-registered test)

v5.2 = the verifiable reward with `RECALL_BETA=1.5`, `CLEAN_CLAIM_PENALTY=0.35`, `R_MIN=0.45`,
`kl_beta=0.02` to a true merged-v4 reference, full prompts. **Note on `R_MIN`:** this write-up long reported 0.45, but the executed notebook auto-selected **`R_MIN = 0.0`** from the gate's p33 (the truncation-zero point mass had collapsed the percentile). With a non-negative reward `clamp(min=0.0)` is an identity op, so the shipped run is **Dr. GRPO with a group-mean baseline**, in a CoRPO-capable trainer — the CoRPO clip never fired. The run itself was the first healthy RL
trajectory in the project: reward 0.33 → 0.64, completion truncations 46% → 5%, KL ≤ 0.0034, no
collapse. Mid-eval recall peaked at checkpoint-150 and declined after (continued training bought
restraint at the cost of catches); checkpoint-150 went to the full 632-record paired bootstrap as the
single confirmatory test:

- **recall:** parity with v4 (paired Δ −0.018, CI [−0.065, +0.026]) — the mid-eval's +0.073 regressed
  exactly as a winner's-curse selection effect predicts.
- **fp_rate on clean diffs: 0.749 → 0.658 (Δ −0.103, CI [−0.165, −0.043]) — the project's first
  statistically significant RL improvement**, consistent across all five checkpoints and verified
  qualitatively (tail compression, correct "no issues" verdicts where v4 fabricates).
- **hallucination:** not significantly different. **Verdict: keep v4** per the pre-registered ship rule;
  checkpoint-150 is preserved as a low-false-alarm variant. Footnotes: 4/632 outputs regress into
  `<think>` repetition loops (production needs an unclosed-think extractor guard), and a label census
  (25/373 tuples are style nits; the dominant "correctness" bucket includes reviewer-preference
  comments) shows the recall metric is partly bounded by label quality, not capability.

---

## Why RL moved restraint but not recall

*(Revised after v5.2: the analysis below was originally written from the invalidated v5.0/v5.1 runs.
The fixed-pipeline v5.2 run supports a sharper version of the same conclusion — RL moved the model to
a strictly better point on the fp axis at recall parity, but did not move recall. And the recall
ceiling is now known to have two components: model capability AND label quality — both models catch
real defects that score zero because the reference thread discussed something else.)*

v4 sits near its own **precision/recall frontier**. RL with a recall/precision-traded reward slides the model *along* that frontier — v5.0 toward quiet (recall ↓, halluc ↓), v5.1 toward loud (halluc ↑, recall flat) — but never pushes the frontier *outward*. To beat v4 on **both** axes you need a frontier shift, i.e. more capability, which RL on a frozen 7B with diff-only context doesn't provide. This matches the literature on small-model RL: it **restores latent capability but rarely exceeds the SFT teacher** [4], and RLVR's reasoning gains are bounded by the base model at large pass@k [5] — RL redistributes rather than expands capability. (High SFT scores are also a poor predictor of RL outcomes [6].)

**Superseded by the oracle experiment.** An earlier revision of this section called recall ≈ 0.09 an intrinsic ceiling of diff-only context. `oracle_ceiling.py` refutes that: a frontier teacher on the *same* 632 diffs, same prompt, same 12000-char budget, same matcher and labels scores **0.388**. The plateau is the student's capability plus incomplete labels — not the task. What survives is narrower and better supported: a recall/precision-traded reward slides the model along its own assertiveness curve, and v5.2 moved it strictly inward on the false-positive axis at recall parity. See [RESULTS.md](RESULTS.md) for the full oracle table.

---

## v4.1: the data fix

The over-flagging is not an RL artifact — it's baked into the SFT data. Classification of v4's 12,876 review targets: **8.3% clean** "no-issue" verdicts vs **51% assert a defect** (40% descriptive). The model was trained to almost always find something, so it flags a defect on 77% of clean diffs.

**Plan (single-stage SFT, no RL):**
1. **Source clean diffs** (~4–5k, to reach ~30% clean targets): (a) apply the suggested fix to defect examples → the post-fix diff is provably clean; (b) mine low-defect-risk diff types (comment/docstring-only, import-only, test-only) via a planned `classify_diff_risk` classifier.
2. **Generate restrained targets** where the `<think>` *verifies before clearing* ("checked bounds, None-handling, the assertion — all correct") so the model learns to reason to a clean verdict, not rubber-stamp.
3. **Retrain** with `sft.ipynb` as-is; rebalanced data.
4. **Evaluate** with the paired harness. Success = fp-rate down (paired-significant) **and** recall ≥ v4 (paired CI not significantly down).

**Honest framing:** the hallucination axis is the high-confidence win (data directly teaches restraint). Recall is capability-limited; the realistic target is *recall parity with materially fewer false positives*. The risk to manage is over-correcting into under-flagging — controlled by the ~30% ratio and the paired recall CI as a guardrail.

---

## Lessons

1. **Pick the eval before the method.** A judge-independent, paired metric exposed that the apparent v5 "wins" were noise; a single-run two-means comparison would have shipped a non-improvement.
2. **Metrics have blind spots — read the outputs.** The backtick hallucination metric missed semantic over-flagging entirely. The disagreement read (humans/agents reading actual diffs vs ground truth) caught what the aggregate hid.
3. **The teacher's noise can exceed the signal.** The DeepSeek matcher's run-to-run variance (recall ±0.005, precision ±0.018) was comparable to the effect sizes — pairing and bootstrapping were not optional.
4. **RL redistributes; data is still the bigger lever — but "RL can't teach it" was wrong.** v5.2 *did* teach restraint on clean diffs (flag rate −10pp, paired CI excluding zero) without losing recall. What RL could not do was raise detection, and the reward shows why: quality is `F₁.₅(recall, precision)`, which is identically zero when a rollout catches nothing, so on the ~90% of labeled diffs the student can't solve, the recall term contributes no within-group contrast and the live gradient sits entirely on grounding/length/restraint. Capability still has to come from data.
5. **Negative results are results — but audit the pipeline before you trust one.** Three of the four RL rounds turned out to be measuring left-truncated prompts, and the "RL is dead" conclusion drawn from them had to be publicly retracted. The supported conclusion, from one clean run, is narrower: verifiable-reward RL buys restraint, not detection, on this setup.

---

## Replicating the RL experiments

```bash
# Build clean defect labels for train + eval splits (DeepSeek)
python label_defects.py --input <split>.jsonl --output cache/defect_labels_<split>.jsonl

# Pre-flight variance gate (does NOT train): is there reward spread to learn from?
python corpo_train.py --variance-gate-only \
    --v4-adapter <v4-lora> --v4-backup <v4-lora-backup> \
    --train-prompts ood_train_prompts.jsonl \
    --defect-labels cache/defect_labels_train.jsonl --output-dir /content/corpo-out

# Train (CoRPO; ~4–6h A100). Checkpoints mirror to --checkpoint-sync-dir (survives Colab disconnects).
python corpo_train.py [same args, no --variance-gate-only] --v4-merged <merged-v4-dir> \
    --kl-beta 0.02 --num-generations 8 --prompts-per-step 4 --checkpoint-every 75 \
    --checkpoint-sync-dir /content/drive/MyDrive/sft/corpo-out-v5
# resume after a disconnect:  add  --resume /content/drive/MyDrive/sft/corpo-out-v5/checkpoint-<N>

# Evaluate a checkpoint, judge-independent + paired vs v4
python run_ood_eval.py --input ood_input.jsonl --output ood_preds_v5.jsonl \
    --v4-model <merged-checkpoint-dir> --skip-base
python score_v5.py --preds ood_preds_v5.jsonl --labels cache/defect_labels_eval.jsonl --pred-fields v4_pred
python compare_recall.py --preds-a ood_preds_v4.jsonl --field-a v4_pred \
    --preds-b ood_preds_v5.jsonl --field-b v4_pred --labels cache/defect_labels_eval.jsonl
```

The full Colab driver is `corpo_train.ipynb` (install → labels+gate → variance gate → train → mid-eval → merge → paired verdict).

---

## References

All arXiv IDs verified against their abstract pages.

1. **Shao et al. (2024).** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:[2402.03300](https://arxiv.org/abs/2402.03300). — Introduces **GRPO**, the critic-free, group-relative RL algorithm used here.
2. **Garg & Venkatesh (2025), Cerebras.** *The Peril of Preference: Why GRPO Fails on Ordinal Rewards.* arXiv:[2511.04439](https://arxiv.org/abs/2511.04439). — Introduces **CoRPO**, which clips the group baseline at a correctness threshold so failed rollouts are never positively reinforced — the structural fix for GRPO's ordinal/continuous-reward failure mode.
3. **Liu et al. (2025).** *Understanding R1-Zero-Like Training: A Critical Perspective.* arXiv:[2503.20783](https://arxiv.org/abs/2503.20783). — Identifies GRPO's length-aggregation bias and introduces **Dr. GRPO** (drops length/std normalization), the loss used in our runs.
4. **Jin et al. (2025).** *RL Fine-Tuning Heals OOD Forgetting in SFT.* arXiv:[2509.12235](https://arxiv.org/abs/2509.12235). — RL restores capability lost in late SFT but does not surpass the SFT peak ("restores, rarely exceeds").
5. **Yue et al. (2025, NeurIPS).** *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* arXiv:[2504.13837](https://arxiv.org/abs/2504.13837). — RLVR gains are bounded by the base model at large pass@k; RL redistributes rather than expands capability.
6. **Kang et al. (2025).** *Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead.* arXiv:[2510.01624](https://arxiv.org/abs/2510.01624). — High SFT scores poorly predict downstream RL outcomes.
7. **Zheng et al. (2023, NeurIPS D&B).** *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* arXiv:[2306.05685](https://arxiv.org/abs/2306.05685). — LLM judges exhibit position / verbosity / self-enhancement biases; motivates abandoning the gameable pairwise-judge reward.
8. **Gao, Schulman & Hilton (2022).** *Scaling Laws for Reward Model Overoptimization.* arXiv:[2210.10760](https://arxiv.org/abs/2210.10760). — Optimizing an imperfect proxy reward degrades the true objective (Goodhart) — the over-claiming/length-gaming we observed.
9. **DeepSeek-AI; Guo et al. (2025).** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:[2501.12948](https://arxiv.org/abs/2501.12948) (also *Nature* 645:633–638). — Precedent for distilling teacher reasoning traces into a smaller student.
10. **Dror et al. (2018, ACL).** *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing.* ACL 2018, [P18-1128](https://aclanthology.org/P18-1128/) (companion appendix arXiv:[1809.01448](https://arxiv.org/abs/1809.01448)). — Endorses the paired bootstrap for comparing two systems on a shared test set.
