# Metrics: what each number actually measures

Every metric in [`RESULTS.md`](RESULTS.md) is defined here against the code that computes it. Written because three of them do not mean what their names suggest, and one name refers to two different metrics.

**Read this first if you are about to quote a number from this repo.**

---

## The two eval regimes answer different questions

| | in-distribution | out-of-distribution |
|---|---|---|
| where | [`eval/eval.ipynb`](../eval/eval.ipynb) | `score_v5.py`, `defect_match.py` |
| sample | 200 records, 4 training repos | 632 records, 88 unseen repos |
| question | **"is this a better review?"** | **"did it find the actual bug?"** |
| ground truth | an LLM judge's preference | defect tuples extracted from the real PR thread |
| failure mode | the judge is a single point of failure | the labels are one reviewer's comments, not a complete bug list |

They can disagree without either being wrong. The first measures review *quality*; the second measures *detection*.

---

## 1. Judge-based metrics (in-distribution)

### Pairwise win rate — 86.0%, CI [81.5%, 91%]
`ood_metrics.py:440` `haiku_pairwise_judge_3vote`. The judge sees two reviews of the same diff, blind, in randomized order, and picks the better one. Three independent votes, majority wins; no majority is a tie.

**It is a preference rate and carries no magnitude.** 86% means "preferred in 86% of comparisons," not "86% better" and not "3× better." ~50% would mean indistinguishable.

### Haiku absolute mean — 4.33 → 5.86
The judge scores each review 0–10 on its own, with no comparison. Useful as a sanity check, but it compresses: most reviews land between 4 and 6 regardless of quality, which is why the pairwise comparison exists.

### Pass@1 ≥7 — 7% → 46%  (strict ≥8 — 2% → 24.5%)
Fraction of reviews scoring at least 7/10 from the absolute judge. **The most decision-relevant of the three:** it answers "how often is the output actually good enough to use," rather than "is it better on average."

### ROUGE-L — 0.099 → 0.180
Longest-common-subsequence word overlap with the human reference review. A correct review phrased differently from the human scores low, and a wrong review that echoes the diff's vocabulary scores high. Report it for completeness; do not lean on it.

### Diff-identifier mention rate — 96.5% → 92.0%
**Labelled `issue_detection` in the notebook. It does not measure issue detection.**

```python
def compute_issue_detection(predictions, diffs):
    detected = 0
    for pred, diff in zip(predictions, diffs):
        if any(i in pred for i in diff_identifiers(diff)):
            detected += 1
    return detected / len(predictions)
```

It returns the fraction of reviews that mention **any** identifier appearing in the diff — a "did you stay on topic" check, not "did you find a bug." The base model scores higher because it writes 2,321-character reviews that name everything; v4's average 246 characters. **The number went down because the model got terser, not worse at finding bugs.** For actual detection, see §2.

`specificity_rate` is the same measure restricted to identifiers of 4+ characters.

---

## 2. Defect-based metrics (out-of-distribution) — the detection numbers

Two LLM calls per record, both constrained:

1. `label_defects.py` turns a PR review thread into grounded defect tuples `{path, line, issue_type, canonical_desc}`.
2. `defect_match.py:14` `defect_caught` asks, per defect, a yes/no question: *does this review catch this specific defect?*

No preference judge is involved — nothing asks "which review is better."

### `defect_recall` — caught ÷ known defects
`defect_match.py:43`. Of the bugs a human reviewer actually flagged on this diff, what share did the model find. Averaged over the records that have at least one labeled defect (`score_v5.py:69`).

**This is the detection metric.** v4 scores ~0.090 — roughly one defect in eleven.

### `precision` — caught ÷ claims asserted
`corpo_reward.py:196`, `min(1.0, caught / n_claims)`. The denominator comes from `defect_match.py:31` `count_claims`, an LLM counting distinct defect assertions in the review. Of everything the model claimed, what share landed on a labeled defect.

### `fp_rate` (clean) — flagged something on a diff with no labeled defect
`corpo_reward.py:204`, `1.0 if n_claims > 0 else 0.0`, averaged over records with no labeled defect. v4 asserts a defect on ~75% of them.

> ### The caveat that governs all three
> These are scored against **one reviewer's comment thread**, not a complete list of the diff's bugs. The oracle experiment measured how much that matters: a frontier teacher trips `fp_rate` on **96.6%** of "clean" records and only **19%** of its claims match a label. It is finding real problems nobody happened to comment on.
>
> So `fp_rate` and `precision` measure **agreement with one human**, not correctness. Never quote an absolute `fp_rate` as a false-alarm rate. A *difference* between two models on the same labels is meaningful; the absolute level is not.

---

## 3. "Hallucination" is two different metrics

Same name, two implementations, **different denominators. They are not comparable to each other.**

| | in-distribution (14.5% → 7.0%) | out-of-distribution (0.062 → 0.036) |
|---|---|---|
| where | `eval/eval.ipynb` Cell 5 | `ood_metrics.py:324` |
| unit | **per review, binary** | **per review, ratio** |
| means | share of reviews containing **≥1** fabricated identifier | of the backticked identifiers used, share absent from the diff — then averaged |

Both work the same way underneath: extract every `` `backticked` `` identifier, drop anything on the shared 278-term stopword list (language keywords, exception types, well-known library symbols), and check whether the rest appear in the diff. The stopword list is byte-identical between the trace generator's F2 filter and the eval metric, so "hallucination" means the same thing in training and evaluation.

### What both are blind to
They catch invented **names** only. Asserting a bug that does not exist, using identifiers that really are in the diff — *"`Tuple` is undefined → NameError"* when `Tuple` is imported two lines up — scores as perfectly grounded.

That is **semantic over-flagging**, and it is this model's actual failure mode. Two instruments catch what the lexical metric cannot:

- **the no-issue probe** (20 hand-built clean diffs, blind binary classifier): base invents a bug on 0/20, v4 on 3/20;
- **disagreement analysis** (read model outputs against ground truth where two models differ).

---

## 4. The RL reward

`corpo_reward.py:167-210`:

```
reward = 0.6·quality + 0.3·grounding + 0.1·length
```

| term | on a diff with labeled defects | on a clean diff |
|---|---|---|
| **quality** (0.6) | `F₁.₅(recall, precision)` — an F-score weighting recall ~2.25× precision | `max(0, 1 − 0.35 × claims)` — full marks for finding nothing, zero at 3 claims |
| **grounding** (0.3) | `1 − hallucination_rate` | same |
| **length** (0.1) | full credit 150–1000 chars, tapering to 0 at 0 and at 2,500 | same |

A rollout that never closes `<think>` (i.e. was truncated) scores a hard **0.0** (`corpo_reward.py:218`), as does an empty or `...` placeholder extraction.

**The load-bearing property:** `F_beta(0, precision) = 0` exactly (`corpo_reward.py:178`). Catch nothing and 60% of the reward is zero no matter how well-written the review is. On the ~90% of labeled diffs this student cannot yet solve, every rollout in a group therefore scores identically on the quality term, and the only surviving within-group contrast comes from grounding and length. That is the mechanism behind the RL result in [`EXPERIMENTS.md`](EXPERIMENTS.md): restraint improved significantly, detection did not move.

---

## 5. Training telemetry

| signal | meaning | observed |
|---|---|---|
| `reward` | mean of the above | 0.33 → 0.64 |
| `kl` | per-token divergence from the SFT reference policy | ≤ 0.0034 — the policy barely moved; it learned format and restraint (cheap in KL), not detection |
| `completions/clipped_ratio` | share of rollouts hitting the token cap unfinished | 46% → 5% — the model learned to finish |
| `frac_reward_zero_std` | share of 8-rollout groups where every rollout scored **identically** — those contribute no gradient at all | ≤ 0.10 |
| variance gate | mean within-group reward spread, measured **before** training (`corpo_train.py:124`). Below the 0.10 threshold means there is nothing to learn from | 0.2508 |

---

## 6. The statistics

### Bootstrap CI
Resample the results with replacement 2,000 times; report the middle 95% of outcomes. `[81.5%, 91%]` means the win rate is stable under resampling.

### Paired bootstrap — `compare_recall.py:39`
Resamples the **per-record differences** (model B − model A on the *same* diff), not two independent averages. This cancels "that diff was just hard," which otherwise swamps the effect: per-record recall is roughly {0, 0.5, 1.0}, so at n≈150 the standard error on the mean is *larger* than the differences being measured. A Monte Carlo on this data put the naive two-means test at 4–12% detection of a real +0.04 recall gain, against ~100% for the paired test.

Only instance IDs present in both prediction sets are compared, and the test is one-sided by design: an improvement requires the recall CI's **lower** bound > 0, or the fp/hallucination CI's **upper** bound < 0.

### Reading a CI
- **excludes zero** → significant.
- **crosses zero** → indistinguishable from noise at this sample size. This is what **parity** means — *no gain detectable*, **not** *no gain exists, and not identical*.

### The matcher is non-deterministic
Re-scoring identical predictions gives recall 0.087–0.099 and can flip the sign of a small delta. Run-to-run noise is comparable to the effects being chased. **This is why every model comparison in this repo uses a paired CI and never a single point estimate**, and why detection gains are quoted as ranges.

---

## Quick reference: do not say this

| ❌ | ✅ |
|---|---|
| "86% better" / "3× better" | "preferred in 86% of blind comparisons" |
| "issue detection dropped, so it detects fewer bugs" | "reviews got 90% shorter, so they mention fewer diff identifiers" |
| "hallucination 7.0% vs OOD 0.036 — it improved" | two different metrics; not comparable |
| "false-alarm rate is 75%" | "flags something on 75% of *label-clean* diffs — even a frontier model trips this on 97%" |
| "RL made recall worse" | "recall parity — no difference detectable at n=632" |
| a single decimal for the detection gain | "+25–45% relative on recall; matcher noise sets the range" |
