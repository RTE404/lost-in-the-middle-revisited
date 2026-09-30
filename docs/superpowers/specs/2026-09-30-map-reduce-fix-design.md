# Map-Reduce Fix for "Lost in the Middle": Design (Phase A)

**Status:** draft for review, revised after the research audit (`reports/Lost in middle map reduce audit.md`)
**Date:** 2026-09-30
**Depends on:** the baseline in `lost-in-the-middle-plan.md` (same models, data, sample, scoring and inference stack)
**Goal:** Phase A is a portfolio piece: "I reproduced it, then tested a fix." Phase C, later, turns the same functions into a drop-in RAG library.

---

## 1. What we are testing, and what changed after the audit

**Question:** if the 20 documents are answered in small groups and the answers are then combined, does the position effect disappear **without losing accuracy**?

**The method is not new.** LangChain's map_rerank, LLM×MapReduce and Chain-of-Agents all chunk a context and combine the answers. Our contribution is a careful test on small 2024–26 open models using Liu et al.'s position protocol, with controls. Chain-of-Agents is the closest published result: on NQ it narrowed the position spread only from 6.1 to 4.9 points, and it found that voting across independent chunks scored 8–11 points below smarter ways of combining answers. So expect a modest result.

**What the audit changed, in plain terms:**

| Before | Problem found | Now |
|---|---|---|
| Success = the best-minus-worst "gap" shrinks | That gap is above 0 even when there's no effect (about 2–4 points from noise), so it can "shrink" by chance | Fixed, pre-registered contrasts (§6) |
| "Accuracy not worse" = the CI isn't entirely below 0 | Noisier runs pass more easily, which is backwards | A real non-inferiority test with a 2-point margin (§6) |
| One group holds the answer; the others say NOT FOUND | False: distractors are same-topic hard negatives, and 104–682 of 2,655 questions have a distractor containing a gold answer string. Small models often answer instead of abstaining | Expect several verified candidates per question; choose between them with a scoring call that treats every candidate the same way (§4) |
| Ties broken by mean logprob | Ties become the *usual* case, and logprobs from different prompts aren't comparable (and are noisy in fp16) | Logprob voting kept only as a sensitivity check |
| Groups kept in original order | The gold's slot inside its group (1/1/2/3/4) and its neighbours both follow its global position, which is a confound | Seeded, balanced gold slot; log its neighbours (§3) |
| Score the final text | Some gold answers match almost anything (idx 1451 `"*"`, idx 1840 `"S"` matches "NOT FOUND") | Score only the parsed answer; an abstention scores 0 before the metric (§5) |
| No controls | Map-reduce changes the prompt, the number of distractors and the length all at once | One extra control run, plus two controls that come free from the map outputs (§7) |

---

## 2. Gate: only fix what is broken

Run map-reduce only on a setting where the baseline shows a **significant position effect**, meaning a first−middle or last−middle contrast from the baseline plan's §5. Check in this order and stop at the first that qualifies:

1. **20 documents** (the baseline's main curve).
2. **30 documents.** This is Liu's own setting and uses the authors' data. Recent evidence says 3–4B models may show only about 5 points at 20 documents, or nothing detectable.
3. **Key-value, 300 pairs.** This is probably at ceiling for Qwen, so it's the last resort. Use it only if the baseline's key-value stage shows real position flips.

**If none qualifies,** phase A reports: "these models are no longer measurably lost in the middle at these lengths, so there is nothing for map-reduce to fix." That is a valid, publishable result. We decide this outcome before running anything.

---

## 3. Split (map input)

- **Same 20 documents** as the baseline for that question and position. Groups are 5 contiguous blocks of 4 (docs 1–4, 5–8, …).
- **Gold slot balanced.** Inside the gold's group, place the gold document at a slot 1–4 assigned from a seeded schedule, so that at every global position each slot is used by about a quarter of the questions. The other 3 documents keep their original order. Gold-free groups are not changed.
  - Why: without this, a leftover within-group position effect would look like a position effect across the whole curve.
- **Deduplication.** Gold-free groups are often identical across the 5 positions; for example, docs d8–d11 form the same group whether the gold is at position 1 or 5. Each unique (question, group contents) prompt runs **once** and is reused, which is about 13 unique map calls per question instead of 25.
- **Logged for every item:** gold group, gold slot, the retrieval ranks of the gold's 3 group-mates, and how many answer-bearing distractors are in each group.
- **Known limitation, reported rather than fixed:** a gold at position 1 shares its group with the highest-ranked (hardest) distractors, and a gold at position 20 with the lowest-ranked. We report accuracy by group-mate rank so readers can see whether this matters.
- **Identify the gold with the `isgold` field, never with `hasanswer`.**

---

## 4. Map, verify, choose (reduce)

### 4.1 Map prompt

A new template, `qa_map.prompt`. It reuses the authors' document line format exactly (`Document [i](Title: …) text`, renumbered 1–4) and adds the new instructions **before** the final answer cue:

```
Write a high-quality answer for the given question using only the provided
search results (some of which might be irrelevant). If none of the search
results contain the answer, write NOT FOUND.

Document [1](Title: ...) ...
...
Document [4](Title: ...) ...

Question: {question}
Reply in exactly this format:
Answer: <short answer, or NOT FOUND>
Evidence: <copy the sentence from the documents that contains the answer>
```

- The shared instruction comes first, so vLLM's prefix cache can reuse it across calls.
- Settings: `SamplingParams(temperature=0, max_tokens=150, logprobs=1)`, always passed explicitly. Qwen models' `generation_config.json` otherwise makes vLLM **sample** instead of decoding greedily. `max_model_len=2048`.
- Outputs that hit the 150-token limit are logged as their own failure type. Raise the limit to 200 if more than 1% of pilot outputs hit it.
- Read template files in text mode (the checkout has CRLF line endings).

### 4.2 Parse

- **Answer** is the text after `Answer:` on its line. **Evidence** is everything after `Evidence:` to the end, and it may span lines.
- Treat any `<think>…</think>` block, a missing `Answer:`, a NaN or -inf logprob, or output that is only `!` characters as **invalid**. Re-run invalid prompts once, then count them as NOT FOUND and report how many there were.
- Hedged or list answers, such as "1994 or 1995", are counted per method and treated as invalid in the primary analysis.

### 4.3 Verify: keep only answers grounded in the documents

Normalize text as follows: lowercase, strip punctuation, collapse whitespace. A candidate is **verified** if all of these hold:
1. The normalized answer is **not empty** and appears inside the normalized evidence.
2. The evidence has **at least 5 normalized words**, because short "evidence" like "1938" would match almost anything.
3. The evidence appears word for word inside that group's documents, **titles included**.
   - Graded fallback, fixed before the full run: rapidfuzz `partial_ratio ≥ 90` against one document, **and** the answer appears word for word in that document span. This forgives small copying slips without accepting invented quotes.

We report the verifier's **false-rejection rate**: how often a correct answer from the gold group is thrown out. Thirty-six gold passages have no single sentence containing the answer, so a small non-zero rate is expected.

### 4.4 Choose one answer (primary rule, fixed in advance)

- **Score every candidate with one common yes/no prompt.** For each distinct normalized answer (verified ones first), run one tiny call:

  ```
  Question: {question}
  Sentence: {evidence}
  Proposed answer: {answer}
  Does the sentence show that the proposed answer is correct? Answer Yes or No.
  ```

  From the first output token, take P(Yes) = softmax over the logprobs of "Yes" and "No" (`logprobs=20`, `max_tokens=1`). An answer backed by several groups keeps its highest P(Yes).
- **Why:** every candidate is judged by the same short prompt, so the scores are comparable. The prompt is too short for position to matter, and it adds only about 80 tokens per candidate.
- **Pick** the verified answer with the highest P(Yes). If two are exactly tied, pick by normalized answer string, which is arbitrary but independent of order. **Never break ties by group order**, because that would build a position bias back in.
- **Fallback:** if nothing is verified, pick the unverified candidate with the highest P(Yes). If there are no candidates, **abstain**. The primary result includes the fallback, since the baseline always answers too. We also report results **without** it.

### 4.5 Sensitivity checks (run on the same saved map outputs, no extra map calls)

- Majority vote with a mean-logprob tie-break, which was the original design.
- An LLM judge that sees the shuffled verified candidates and their evidence.
- The AUROC of each confidence signal for separating right from wrong candidates. If P(Yes) is close to 0.5, we say so.

---

## 5. Scoring

- Score **only the final chosen answer string**, with any `Answer:` prefix removed, using the authors' `best_subspan_em`. Never score the Evidence text, the raw map outputs, or several outputs joined together.
- **An abstention scores 0 before the metric runs.** Otherwise answers like idx 1840's `"S"` would match the literal text "NOT FOUND".
- **idx 1451** has gold answer `"*"`, which counts every prediction as correct. Keep it in both methods (it cancels out in paired comparisons) and document it. Add unit tests for idx 1451 and 1840.
- **Secondary metrics:** strict normalized exact match, and mean output length per method. Baseline answers are longer and can get free substring credit.

---

## 6. Statistics (fixed in advance)

All tests are paired over **question IDs** (all 2,655 questions), using 10,000 bootstrap resamples with a fixed seed. Each resampled question brings all of its positions and both methods with it.

| Test | Definition | Pass rule |
|---|---|---|
| **1. Non-inferiority (primary)** | Per question, average the 5 positions; D = MR − baseline | Lower end of the 95% CI of D > **−2 points** |
| **2. Position effect removed (primary)** | U = mean(pos 1, pos 20) − mean(pos 5, 10, 15) for each method; ΔU = U_base − U_MR | 95% CI of ΔU excludes 0, in the direction of MR being flatter |
| 3. Middle recovered (secondary) | MR − baseline at position 10; share of the gap recovered = (MR₁₀ − base₁₀)/(base₁ − base₁₀) | Reported with CI; marked unstable if the denominator's CI touches 0 |
| 4. Residual position effect (secondary) | Accuracy by gold slot (1–4) and by group-mate rank | Descriptive |

**Outcomes:**

| Outcome | Rule |
|---|---|
| **Works** | Tests 1 and 2 both pass (Holm correction across the two models) |
| **Flatter but worse** | Test 2 passes, test 1 fails |
| **No effect** | Test 2 fails |

- Note that test 2 is **close to guaranteed** whenever the baseline has an effect, because map-reduce removes long prompts by design. **The real question is test 1**: does it keep the accuracy? The write-up says this plainly.
- The best-minus-worst range is reported only as a description, next to its "no effect" value from a within-question permutation (shuffle the position labels 10,000 times).
- Write the predicted outcome for each model in the notebook before running.

---

## 7. Controls and reference points

| Control | Cost | What it isolates |
|---|---|---|
| **20-doc map-prompt control:** all 20 documents in one prompt, using the map prompt format | One extra long-prompt run, the same size as the baseline's main curve | Whether the gain comes from the new prompt wording rather than from splitting |
| **Gold-group accuracy:** the gold group's own answer | Free (already in the map outputs) | The effect of fewer distractors (a 4-document baseline) |
| **Oracle reduce:** correct if *any* group's candidate is correct | Free | Separates map failures from reduce failures |
| **Closed-book and oracle** | Free (already in the baseline) | The floor and the ceiling |

**Map-stage diagnostics** (free), for each model: the NOT FOUND rate on gold-free groups, the false-NOT-FOUND rate on gold groups, the verifier's false-rejection rate, how often each decision path was used (verified, fallback, abstain), and the share of outputs that hit the length limit.

If the Kaggle quota is tight, run the 20-doc control on a seeded 1,000-question subset. Its CIs will be wider, and we say so.

---

## 8. Pilot first, then freeze

Pilot: 200 seeded questions × 5 positions, per model. It must show:
- Outputs follow the `Answer:`/`Evidence:` format at least 95% of the time. Otherwise fix the prompt and repeat the pilot.
- Truncation rate (decides 150 vs 200 tokens), the verifier's false-rejection rate (decides whether the fuzzy fallback threshold stays at 90), and the AUROC of each signal.
- Read 20 raw outputs by hand.

After the pilot, **freeze** the thresholds and the primary rule, and record them in the notebook's first cell. Pilot questions are included in the full run; the pilot only sets parameters, it isn't a separate result.

---

## 9. Key-value version (only if §2 sends us there)

- 15 groups of 20 pairs, using a local formatter. The authors' `get_kv_retrieval_prompt` raises `ValueError` on any group that doesn't contain the key, which is 14 of the 15.
- The gold slot is balanced as in §3. The baseline's positions 0/74/149/224/299 otherwise fall at slots 0/14/9/4/19.
- Output: `Value: <uuid>` and `Evidence: "<key>": "<value>"`. Verify that it's a UUID, that the key equals the query key, that the value equals the answer, and that the pair really is in that group. Keys are unique, so any verified answer is correct: no yes/no scoring is needed.
- Never score the group outputs joined together.

---

## 10. Code (fits the existing `litm` package)

| Unit | Job |
|---|---|
| `litm/mapreduce.py` | `split()`, `build_map_prompt()`, `parse()`, `verify()`, `choose()`: plain functions, no GPU, fully unit-tested |
| `litm/generate.py` | `Generator.generate(prompts, params) -> [text, token_ids, logprobs, finish_reason]`. The vLLM version now; an API version in phase C |
| `litm/prompts/qa_map.prompt`, `qa_check.prompt` | The two new templates |
| Records | Map outputs go in their own `.jsonl`, keyed by (method, task, position, idx, group, content hash); final answers go in another. The analysis reads **only** final answers. The existing `(task, position, idx)` key would collide |
| Sample | Same question list and model revisions as the baseline |

Inference settings are the baseline plan's: vLLM 0.18.1, fp16, TRITON_ATTN, one model per T4. Add: assert compute capability (7,5) at startup; `max_model_len=2048` for map and check calls; never request `prompt_logprobs` (it disables the prefix cache); fixed batch size and input order.

**Tests before any GPU run:** splitting and slot balancing, parse edge cases, every verify rule (empty answer, short evidence, titles, multi-line text), deterministic tie handling, scoring of idx 1451 and 1840, and dedupe keys.

---

## 11. Budget (rough, to be confirmed in the pilot)

These are estimates, not measurements, for all 2,655 questions per model:

| Run | Estimate |
|---|---|
| Map calls: about 34,500 unique, about 670 tokens each | about 3–4 h |
| Yes/no scoring calls: tiny | about 0.5 h |
| 20-doc map-prompt control | about 3.5–4.5 h (like the baseline main curve) |
| **Total per model** | **about 7–9 h**; both models run in parallel on the two T4s |

This probably doesn't fit in the same Kaggle week as the baseline (30 h/week, possibly counted at 2× for T4×2), so **plan it for week 2**. The pilot measures real speed first.

---

## 12. Out of scope for phase A

- Reordering (LangChain-style); maybe a later bonus.
- Tuning the group size (fixed at 4 documents / 20 pairs).
- NoLiMa-style harder retrieval data, more models, training-based or attention-based fixes.
- Questions that need two documents at once.
- The library itself (phase C).

---

## 13. Deliverable

The phase-A section of the public notebook and README:
- one chart per model: the baseline and map-reduce curves, the 20-doc map-prompt control, and closed-book and oracle as dashed lines;
- the table of tests 1–4 with CIs;
- the map-stage diagnostics;
- an honest paragraph on what the fix does and doesn't do.

Every number traces back to saved outputs.
