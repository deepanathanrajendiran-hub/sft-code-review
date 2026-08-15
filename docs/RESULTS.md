# Results

All numbers are from the evaluation pipelines in this repo. **Every metric below is defined against its implementing code in [`METRICS.md`](METRICS.md)** — read that before quoting a number, because three metrics do not mean what their names suggest and "hallucination" refers to two different measures.

Two distinct eval regimes:

1. **In-distribution, judge-based** — 200 samples drawn from the training distribution (transformers / sklearn / pydantic / fastapi), scored by a 3-vote Claude Haiku judge. This is the headline benchmark v4 was shipped on.
2. **Out-of-distribution, judge-independent** — 632 records from ~90 unseen repos, scored by recall against clean defect tuples + a backtick-grounding hallucination metric, with **no LLM quality judge**. Built for the v5 RL experiments to remove the judge as a single point of failure.

---

## 1. In-distribution, 200-sample (3-vote Haiku judge)

| Metric | Base | SFT v3 (prior) | **SFT v4** |
|---|---|---|---|
| Pairwise win vs base | — | 71.5% | **86.0%**  CI [81.5%, 91%] |
| Haiku absolute mean (0–10) | 4.33 | 4.25 *(regressed)* | **5.86** |
| Pass@1 ≥ 7 | 7% | 14.5% | **46.0%** |
| Pass@1 ≥ 8 (strict) | 2% | — | **24.5%** |
| Hallucination rate | 14.5% | 34.5% *(3× worse)* | **7.0%** *(halved vs base)* |
| Diff-identifier mention rate *(labelled "issue detection" in the notebook — **not** a defect-detection metric; [why](METRICS.md#diff-identifier-mention-rate--965--920))* | 96.5% | 94.5% | 92.0% |
| ROUGE-L vs expert reference | 0.099 | — | **0.180** |
| Failure-pattern tests (9 patterns × 3) | 27/27 | 27/27 | 27/27 *(non-discriminating — see below)* |
| **No-issue probe** (20 clean diffs, invents a bug) | **0/20** | — | **3/20** *(v4 is worse)* |
| Prompt-eng gauntlet (v4 vs few-shot CoT) | — | — | **v4 wins 82.5%** |
| Mean review length (chars) | 2,321 | — | 246 |

All in-distribution numbers above are reproduced in the stored outputs of [`eval/eval.ipynb`](../eval/eval.ipynb) (200 samples, 3-vote Haiku judge, W&B run linked in Cell 10). Judge reliability: 100% agreement between single-vote and 3-vote on 50 random pairs.

**Reading it:** v4 reverses every v3 regression. v3 had pushed hallucination *up* to 34.5% and absolute quality *down* to 4.25; v4 brings hallucination to 7.0% and quality to 5.86 while winning 86% of head-to-head comparisons. The few-shot CoT prompt-engineering baseline — the obvious "do you even need fine-tuning?" challenger — wins only 13.5% against v4.

**Two honest corrections to earlier versions of this table.**

1. **The failure-pattern suite does not discriminate.** An earlier revision reported base at 9/27. The executed notebook shows the base model passing **100% on all nine patterns**, identical to v4 (`SFT Δ +0.0%` on every row). The suite is a regression guard, not evidence of an improvement, and it should not be quoted as one.
2. **v4 over-flags on clean code, and it is worse than base at it.** The no-issue probe has the base model inventing a bug on 0/20 hand-built clean diffs and v4 on 3/20 (`list → tuple`, a `TYPE_CHECKING` import, a docstring edit). The notebook labels this a gate v4 fails. This is the same regression the out-of-distribution harness measures at scale (flag rate 0.755 vs base 0.689, §2) and the axis the v5.2 RL run later improved with significance (0.749 → 0.658). Three independent instruments, one failure mode.

Also note v4's mean review is **246 chars against base's 2,321** — a ~90% length reduction. Because the hallucination metric counts ungrounded backticked identifiers, part of the 14.5% → 7.0% drop reflects v4 simply saying less. The no-issue probe is the counterweight: it measures *semantic* over-claiming, which the lexical metric cannot see, and on that axis v4 is worse.

> The `eval/eval_results.json` checked into the repo is the **v3-era** dump (its `SFT` block shows the 34.5% hallucination; its `GRPO` block is a broken empty-extraction run). The v4 numbers above come from the patched 3-vote pipeline in `eval/eval.ipynb`.

### Training curves (v4)

Clean, no overfitting: train loss 1.6 → 0.84, val loss 1.03 → 0.88, train–val gap ~0.05 at the end of 2 epochs (A100 80 GB).

---

## 2. Out-of-distribution, 632-record, judge-independent

Metrics (all from `score_v5.py`; full definitions in [`METRICS.md`](METRICS.md)):

- **`defect_recall`** — fraction of known defects caught, on the 299 records that have a labeled defect (semantic matcher).
- **`precision`** — caught / claims on those records.
- **`fp_rate`** — fraction of *clean* records where the model asserted a defect (false-positive / over-flagging rate).
- **`halluc`** — backtick-grounding hallucination (fraction of backticked identifiers absent from the diff and not in the stopword allow-list).

### v4 vs. the base model — how much did fine-tuning actually improve *defect detection*?

This is the only table in this repo that answers that question. Both models scored in the **same run**, same matcher, same labels (632 records, 281 labeled):

| | base | **SFT v4** | change |
|---|---|---|---|
| **`defect_recall`** — caught a defect a human reviewer flagged | 0.072 | **0.090** | **+25% relative** (+1.8 pp) |
| **`precision`** — of the defects it claims, share that are real | 0.032 | **0.074** | **+131% relative (2.3×)** |
| `fp_rate` (clean) — flags something on a clean diff | 0.689 | 0.755 | **worse** (+6.6 pp) |
| `halluc` — ungrounded backticked identifiers | 0.062 | 0.036 | −42% |

A second, independent scoring pass put the recall gain higher (0.064 → 0.093, **+45%**); the matcher's run-to-run noise is the reason for the range. **State the detection gain as ~+25–45% relative on recall and ~2–3× on precision** — never as a single decimal.

Three things must travel with those numbers:

1. **The absolute level is low.** 0.09 recall means v4 catches roughly one in eleven human-flagged defects. The gain is real and it is on a small base.
2. **Precision is the bigger and more meaningful win.** Fine-tuning roughly doubled-to-tripled the chance that a claim v4 makes corresponds to a real reviewed defect — that's what makes the reviews usable, more than the recall delta.
3. **The "Issue detection 96.5% → 92.0%" row in §1 is not a detection metric** and must not be read as one. `compute_issue_detection` returns the fraction of reviews that mention *any identifier appearing in the diff* — a groundedness/on-topic proxy. Base scores higher on it because it writes 2,321-char reviews that name everything; v4's are 246 chars. It went down because the model got terser, not because it got worse at finding bugs.

Recall ≈ 0.09 is **not** a ceiling imposed by diff-only context — the oracle experiment below measures ~4× headroom on the identical prompts. It is a student-capability limit, compounded by incomplete labels.

> **Measurement caveat — the matcher is non-deterministic.** The same v4 predictions scored across two runs gave recall 0.094 vs 0.099, precision 0.070 vs 0.088, fp 0.754 vs 0.772. The DeepSeek semantic matcher's run-to-run noise is as large as the v4-vs-v5 gaps we were chasing. This is *why* `compare_recall.py` (paired bootstrap) exists, and why single-run point comparisons are not trusted here.

### v5 RL checkpoints (CoRPO, vs v4, same OOD set)

| Model | recall | fp_rate | halluc | verdict |
|---|---|---|---|---|
| **v4** | ~0.094 | ~0.77 | 0.036 | baseline |
| v5.0 ckpt-75 (F1 reward) | **0.070** | 0.73 | 0.032 | recall **collapsed** — model went quiet |
| v5.1 ckpt-75 (F-beta=2) | 0.080–0.103 *(noisy)* | 0.74–0.77 | 0.031 | **statistical tie** both axes |
| v5.1 ckpt-150 (more training) | **0.080** | — | 0.031 | **worse** — see trajectory below |
| **v5.2 ckpt-150 (fixed pipeline)** | 0.087–0.093 *(= v4, parity)* | **0.658** | 0.029 | **fp −10pp SIGNIFICANT; recall parity → keep v4 per ship rule** |

> **Important caveat on v5.0/v5.1 (and Runs #1–#3):** a later code review found those runs were
> invalidated by pipeline defects — TRL's default `max_prompt_length=512` silently left-truncated
> ~87% of training prompts (the policy couldn't see the diffs it was scored on), and the KL term
> anchored to the *base* model instead of v4 (PEFT `disable_adapter` reference). Their rows are kept
> as a record, but they say nothing about RL-on-v4. v5.2 is the first run with a working pipeline.

**No RL checkpoint beats v4 on both axes with statistical significance.** Paired-bootstrap CIs:

- v5.1 ckpt-75 hallucination delta vs v4: **[−0.020, +0.009]** — crosses zero (not significant).
- v5.1 ckpt-75 bug-finding (lexical proxy) delta: **+0.6pp, CI [−2.6pp, +3.9pp]** — not significant.

### Trajectory: what more β=2 training did (v4 → ckpt75 → ckpt150)

Deterministic local analysis (no API), confirming the model got **louder, not better**:

| signal | v4 | ckpt-75 | ckpt-150 | trend |
|---|---|---|---|---|
| mean review length (chars) | 898 | 1027 | 1310 | ↑ louder |
| backticked idents / review | 6.0 | 8.2 | 11.4 | ↑ more assertions |
| clean-verdict rate | 0.149 | 0.136 | 0.116 | ↓ fewer "no issue" |
| true-fabrication rate (denoised) | 0.034 | 0.025 | 0.030 | flat (CI crosses 0) |
| over-flagging (solo-asserts, pop.) | 312 | 138 | 162 | ↑ at ckpt-150 |
| real catches v4-missed (45-sample) | — | — | 5 vs v4's 9 | favors v4 |
| **repetition-loop pathology (records)** | 22 | 29 | 35 | ↑ **new regression** |

ckpt-150 asserts *more* but its extra assertions skew toward fabrication/off-target nitpicks, not real catches — and a repetition-loop degeneration (a review emitting "Bug:" hundreds of times) appeared and worsened with training.

### v5.2 — the fixed-pipeline run (2026-06-11)

After repairing the five pipeline defects (prompt truncation, KL anchor, think-leak scoring,
ambiguous-clean labels, mid-eval asymmetry), one pre-registered CoRPO run (RECALL_BETA=1.5,
CLEAN_CLAIM_PENALTY=0.35, R_MIN=0.45, kl_beta=0.02 to a true merged-v4 reference) trained cleanly:
reward 0.33 → 0.64, completion truncations 46% → 5%, KL ≤ 0.0034, no length collapse. Mid-eval recall
peaked at checkpoint-150 and declined after (later training bought restraint at the cost of catches);
checkpoint-150 was the single confirmatory test on the full 632-record paired bootstrap:

| axis | v4 → v5.2 | paired delta (95% CI) | call |
|---|---|---|---|
| defect_recall | 0.087 → 0.093 (point) | −0.018 [−0.065, +0.026] | **parity** (not significant) |
| fp_rate (clean) | 0.749 → 0.658 | **−0.103 [−0.165, −0.043]** | **significant improvement** |
| hallucination | 0.036 → 0.029 | −0.007 [−0.023, +0.008] | not significant |

**Verdict: keep v4 as production** (the pre-registered ship rule requires a significant recall gain) —
but the false-positive reduction is real, consistent across all five checkpoints, and qualitatively
verified (v5.2's reviews show tail compression — mean 898→516 chars with the median unchanged — and
its "no issues" verdicts on clean diffs read as correct judgment where v4 fabricates). Checkpoint-150
is preserved as a low-false-alarm variant (same recall, ~12% relatively fewer clean-diff false alarms).

Two honest footnotes: (1) a mild repetition-loop regression — 4/632 outputs loop inside `<think>` at
deployment settings (v4: 0/632), so production use needs an unclosed-think guard in the extractor;
(2) a label census (373 tuples: 195 correctness, 114 bug, 29 api_contract, **25 style**, 9 perf,
1 security) plus a disagreement read showed the recall metric is partly bounded by label quality —
both models catch real defects that score zero because the human thread discussed something else, and
some "clean" records carry plausible real issues both models independently flag. Recall ≈ 0.09
measures thread-matching, not bug-finding; label-quality work is the cheapest path to a sharper metric.

### Diff-only oracle ceiling (2026-06-11)

To decide whether the recall plateau is the *task's* ceiling (diff-only context) or the *student's*,
`oracle_ceiling.py` had DeepSeek V4-Pro (thinking enabled) review the same 632 diffs with the same
prompt and 12000-char budget, scored by the same matcher on the same labels:

| model | recall | precision | fp_rate (clean) | halluc |
|---|---|---|---|---|
| v4 | 0.095 | 0.093 | 0.744 | 0.036 |
| v5.2 ckpt-150 | 0.115 | 0.118 | 0.650 | 0.029 |
| **oracle (V4-Pro)** | **0.388** | 0.186 | 0.966 | 0.127 |

*(style-filtered labels shift every number by ≤0.01 — style leakage was not the binding label problem)*

Two conclusions. **First, the task has ~4× recall headroom at diff-only context** — the students are
training/capability-limited, not context-limited, which makes recall-targeted distillation (the
oracle is the teacher) the highest-leverage next step, with RL after. **Second, the clean labels are
substantially incomplete:** even the frontier model "false-alarms" on 97% of label-clean records and
only 19% of its claims match labels — so fp_rate/precision here measure agreement with one reviewer's
thread, not error rates. v5.2's fp improvement stands as a behavioral restraint difference, but no
absolute fp number from this harness should be quoted as a false-alarm rate. (Caveat: the oracle, the
labeler, and the matcher are all V4-Pro — some same-family phrasing leniency may inflate the ceiling,
but not plausibly by 4×.)

---

## 3. Root cause: the training-data balance

Classification of v4's 12,876 SFT review targets (`train_dataset_v4_traces_o2.jsonl`):

| target type | count | share |
|---|---|---|
| CLEAN-only ("no issues" verdict) | 1,069 | **8.3%** |
| FLAG-only (asserts a defect) | 6,278 | 48.8% |
| BOTH (caveat + clean-ish) | 346 | 2.7% |
| NEITHER (descriptive only) | 5,183 | 40.3% |

Only 8.3% of targets teach restraint; 51% teach flagging. This directly explains the 77% false-positive rate — and it's the lever v4.1 targets. See [`docs/EXPERIMENTS.md`](EXPERIMENTS.md).
