# Lost in the Middle, Revisited: Project Plan (Minimum Version)

**Paper:** *Lost in the Middle: How Language Models Use Long Contexts*, Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni and Liang, TACL 2024 ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172))
**Authors' code and data:** [github.com/nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle) (MIT licence; last updated 2024-01-04)
**Owner:** Rohan Tiwari
**Budget:** Zero. Free Kaggle GPUs (T4 x2). Colab only for small jobs.
**Last updated:** 2026-09-30, after a fact-check against current sources (`reports/Lost in the middle plan check.md`)

---

## 1. Goal

Answer one question with the least work that still gives a trustworthy result:

> **Do current small open models still lose information placed in the middle of a long prompt?**

We don't need an exact copy of the paper, just a **verifiable, good-enough result**: a public notebook anyone can rerun on a free GPU, with every number traceable to a saved model output.

---

## 2. The paper in one paragraph

The authors gave a model a question and several documents, only one of which held the answer, and moved that document from first to last. Accuracy was highest when the answer was at the **start or end** and lowest in the **middle** (a U shape). For GPT-3.5 with 20 documents, a middle answer scored **below** having no documents at all. A pure lookup task (find one key in a long list of random pairs) showed the same effect for many models.

### Numbers to compare against

| Item | Value (source) |
|---|---|
| Questions | 2,655 NaturalQuestions-Open queries (§2.1) |
| 20-doc answer positions | 1, 5, 10, 15, 20 (Table 6); 0-based indices 0, 4, 9, 14, 19 in the data files |
| GPT-3.5-Turbo, 20 docs | 75.8 / 57.2 / 53.8 / 55.4 / 63.2% by position (Table 6, checked against the PDF) |
| GPT-3.5-Turbo closed-book / oracle | 56.1% / 88.3% (Table 1) |
| Llama-2-70b-chat, 20 docs (open-model reference) | 56.8 / 53.3 / 54.1 / 59.7 / 69.5%; oracle 84.7%, closed-book 35.3% (repo `EXPERIMENTS.md`) |
| Prompt length in the paper | 20 docs about 4K tokens; 300 key-value pairs about 16K (GPT tokenizer). **Measure with our tokenizers.** |
| Key-value data | 75, 140, 300 pairs; 500 examples each (§3.2) |
| Decoding | Greedy, up to 100 new tokens (repo scripts) |

### What to expect (all three shapes are possible)
- **Recency only:** Llama-2-7B favoured only the end (paper, Appendix E).
- **U shape:** newer work finds the U in modern models when the prompt fills roughly 25–50% of the context window, and says size is not the main driver (Veseli et al. 2025).
- **Primacy dominant:** 2026 tests on 7–8B Qwen-family models with this same NQ setup found the first position best, with first-vs-last gaps of 5–7 points.
- **Expect small gaps**, around 5 points, not the 20+ points of 2023 models. That is why section 3 uses all 2,655 questions.
- **The prompt fills different shares of each model's context window**, which may change the shape. Measured with our tokenizers: 20-doc prompts average about 3.1K tokens (about 10% of Qwen2.5-3B's 32K window, about 1% of Qwen3-4B-Instruct-2507's 262K); key-value prompts are about **20.4K tokens** (Qwen's tokenizer splits UUIDs more finely than GPT's), which is **about 62% of Qwen2.5-3B's window**, past the point where recency tends to win, and about 8% of 2507's. Report this share next to every result.
- **"Worse than no documents" probably won't repeat.** Small models know far fewer NQ answers without documents (one unverified estimate puts Qwen2.5-3B at about 14%) than GPT-3.5 did (56%).
- **Small models sometimes fail to use even the single correct document**, which makes the oracle check in step 3 essential.

---

## 3. What we run

### Models
| Model | Role | Notes |
|---|---|---|
| Qwen2.5-3B-Instruct | 2024 reference | 32K context. Qwen Research licence (not Apache). Its template adds a default system prompt |
| Qwen3-4B-Instruct-2507 | Current model | Apache 2.0, 262K context, **never produces thinking text**, same layout as Qwen3-4B |

Both fit on one 16 GB T4 in float16 (about 6 GB and 8 GB of weights). **Pin each model's Hugging Face revision hash.**

Why not newer: Qwen3.5-4B (March 2026) needs vLLM nightly and linear-attention kernels that have failed on older GPUs. Gemma 4 E2B uses 512-token sliding windows (which confound a position study) and the Gemma family has float16 overflow problems. Both are optional add-ons (section 10) behind a smoke test.

### Tests
| Test | Setup | Prompts per model |
|---|---|---|
| **Baselines** | Closed-book and oracle, all 2,655 questions | 5,310 (short) |
| **Main curve** | 20 documents, answer at positions 1, 5, 10, 15, 20, **all 2,655 questions** | 13,275 (about 3.1K tokens each, max 3.4K) |
| **Long lookup** | Key-value, 300 pairs, positions 1, 75, 150, 225, 300 (indices 0/74/149/224/299), authors' 500 examples, **staged** (below) | 1,000–2,500 (about 20.4K tokens each) |

**Why all 2,655 questions:** at 500 questions a paired test can only detect gaps of about 5–6 points, which is right where modern models are expected to sit. At 2,655 it detects about 2.7 points. The 3–4K-token prompts are cheap, so this costs only about 3–4 extra GPU-hours per model.

**Key-value staging:** run the first 200 examples at all 5 positions. If fewer than 5% of examples flip between positions (models are near 100% on this task), stop: 200 is enough. Otherwise run the `kv` stage again with `KV_LIMIT = 500`; it resumes and adds the remaining 300. The 20K-token prompts cost about 7x more each than QA prompts, so this is where compute is saved.

### GPU time (estimate; replace with pilot measurements)
Measured prompt tokens per model: QA + baselines about 41.5M; key-value about 20.4M (200 examples) or 50.9M (500).

| | Qwen2.5-3B | Qwen3-4B-2507 |
|---|---|---|
| QA + baselines | about 3.5 h | about 4.6 h |
| Key-value, 200 examples / +300 more | about 2.5 h / +3.7 h | about 3.8 h / +5.6 h |
| Rerun check (section 8) | about 1 h | about 1.5 h |
| **Total** | **about 7 h (11 h with 500 KV)** | **about 10 h (15.5 h with 500 KV)** |

With one model per T4 running at the same time, that is **about 10 hours wall-clock (15.5 with 500 key-value examples)**, split into separate background runs (smoke, pilot, qa, kv 200, kv +300, rerun), each under the 12-hour session limit. These figures come from FLOP arithmetic at about 20 TFLOPS effective, because nobody has published T4 speeds for these models. Actual speed could be 1.5x slower.

**Kaggle quota:** about 30 hours a week. Sources disagree on whether T4 x2 uses quota at 1x or 2x the clock time. **Check the quota meter before and after the pilot.** At 1x the plan fits in one week; at 2x, the 500-example key-value extension moves to a second week.

---

## 4. Rules we don't compromise on

1. **Use the authors' data** from the repo's `main` branch (re-uploaded 2024-01-04; older copies have easier distractors). Pin the repo commit hash.
   - QA: `qa_data/20_total_documents/…gold_at_{0,4,9,14,19}.jsonl.gz`.
   - Baselines: `nq-open-oracle.jsonl.gz`. Oracle uses `qa.prompt` with the one document; closed-book uses `closedbook_qa.prompt` on the same file.
   - Key-value: `kv-retrieval-300_keys.jsonl.gz`. Move the gold pair to each position at run time with the authors' code.
2. **Copy the authors' code instead of installing it.** `pip install -e .` fails on outdated GPU dependencies. Copy `prompting.py`, `metrics.py` and `prompts/` (MIT, with credit). Install `pydantic`, `regex` and `xopen`, and run their `tests/test_prompting.py` to check it works with the current pydantic.
3. **Score like the authors:**
   - QA: strip leading whitespace, keep only the **first line** of the output, then apply `best_subspan_em` (normalise the text, then check whether any gold answer appears in it).
   - Key-value: case-insensitive check that the value appears in the output.
4. **Question IDs are line numbers.** The data files have no ID field. Check that the question text is identical at each line across all position files.
5. **Greedy decoding, 100-token output limit.**
6. **Length check before running:** count tokens with each model's tokenizer and stop if any prompt exceeds `max_model_len`. Never truncate silently (the authors skipped over-long prompts rather than cutting them).
7. **float16** (the T4 has no bfloat16). Before any full run, a **smoke test** (the notebook's `smoke` stage: 19 prompts per model, including five of about 20K tokens). Stop if any output is empty, garbled or NaN.
8. **Chat templates as shipped:** put the paper's prompt in each model's own chat template as one user message. Keep any system prompt the template adds (Qwen2.5 adds one; the authors also kept Llama-2's default). **Save the fully rendered prompt** with each output.
9. **Save every raw output** with its line number, position and rendered prompt.

---

## 5. How we read the result

Decide this before running, and write it in the notebook's first cell with a guess per model.

**Pre-registered positions:** first = position 1, middle = **position 10** (index 9) for QA and **position 150** (index 149) for key-value, last = final position. The middle is fixed in advance; picking the worst middle position after seeing results would bias the test towards finding a U.

**Two contrasts per model and test:** first − middle, and last − middle. Use a paired bootstrap over questions (10,000 resamples, fixed seed) for 95% CIs, McNemar's exact test for p-values, and Holm correction over the two contrasts.

| Verdict | Rule |
|---|---|
| **U shape** | Both contrasts significantly above 0 |
| **Primacy only** | First − middle significant; last − middle not |
| **Recency only** | Last − middle significant; first − middle not |
| **Flat** | Neither significant, **and** both 95% CIs lie within **±3 points** (a fixed equivalence margin) |
| **Inconclusive** | Neither significant, but the CIs are too wide to call it flat |
| **Other** | The middle is significantly *better* than an end |

(The Flat margin is fixed rather than tied to the MDE: the MDE widens as fast as the CI, which would let a tiny sample be called Flat.)

Also report:
- **MDE achieved**, computed from the observed share of questions that flip between positions.
- **Gap:** best minus worst position (descriptive only).
- **Oracle and closed-book** next to the curve, and whether any position falls below closed-book.
- **Share of the context window** the prompt fills.
- Optional, no extra GPU time: the curve using only questions the model gets wrong closed-book.

---

## 6. The notebook

One public Kaggle notebook, five sections:

1. **Setup:** install pinned packages; download the authors' data at a pinned commit; copy their prompting and scoring code; pin model revisions; record versions of everything.
2. **Build prompts:** paper prompt in the chat template; token-count check; save rendered prompts.
3. **Run:** vLLM, greedy decoding. Write outputs to `.jsonl` after every batch; on restart, skip finished (line, position) pairs.
4. **Score:** authors' scorers; accuracy by position; paired bootstrap, McNemar, Holm, MDE; verdict per section 5.
5. **Plot:** the 20-doc curves for both models, with closed-book and oracle as dashed lines and the paper's GPT-3.5 and Llama-2-70b-chat curves for reference; a key-value chart.

### vLLM setup on a Kaggle T4
- Install **`vllm==0.18.1`** (validated on Kaggle T4s in Aug–Sep 2026) and let it bring its own torch; restart the kernel afterwards. The kaggle-vllm project's install method avoids clashes with Kaggle's preinstalled packages.
- Set `VLLM_ATTENTION_BACKEND=TRITON_ATTN`. FlashAttention needs a newer GPU, and vLLM 0.22+ can crash on T4 by picking FlashInfer.
- `dtype="float16"`, `gpu_memory_utilization=0.9`, `max_model_len=20608` (longest measured prompt, 20,458 tokens, plus 100 new tokens). Setting it explicitly matters: 2507's 262K default fails vLLM's startup memory check.
- If CUDA-graph capture fails, add `enforce_eager=True`.
- **Fallback:** Hugging Face transformers with SDPA attention and prompts sorted by length. Plain attention runs out of memory at 20K tokens.

### Free-tier rules
- Choose **T4 x2**. The P100 option no longer works at all: Kaggle's current PyTorch has no kernels for it.
- Run full jobs as **"Save Version"** background runs. Interactive sessions die after 20 idle minutes; background runs last up to about 12 hours.
- **One model per GPU** (`CUDA_VISIBLE_DEVICES=0` and `1`). Splitting a small model over both GPUs was slower in Kaggle tests.
- Keep each run under 12 hours: separate runs for QA, key-value and the rerun check.
- Store the Hugging Face token in **Kaggle Secrets** (only needed for gated models).
- Outputs are tens of MB, well within `/kaggle/working`'s roughly 20 GB.
- If Kaggle quota runs out: wait for the Saturday reset. Paid credits on Lightning AI or Modal are an option, but not needed for this plan.

---

## 7. Steps

| Step | Work | Done when |
|---|---|---|
| 1. Set up | Get the data at a pinned commit; copy and test the authors' code; install vLLM 0.18.1 on a Kaggle T4; pin versions and revisions | The authors' tests pass and one prompt runs and is scored |
| 2. Smoke test and pilot | For each model: `smoke` stage (19 prompts incl. five 20K-token ones), then `pilot` stage (140 QA and 20 key-value prompts). Record tokens/second and quota used | Outputs look sane; budget in section 3 updated with measured speed and quota burn |
| 3. Check the pipeline | Baselines for Qwen2.5-3B-Instruct | Oracle clearly above closed-book. If oracle is poor, inspect outputs before going further |
| 4. Main curves | 20-doc QA, all 2,655 questions, both models in parallel | All QA outputs saved |
| 5. Key-value | First 200 examples at 5 positions; continue to 500 if more than 5% flip | All key-value outputs saved |
| 6. Analyse | Score, bootstrap, McNemar, MDE, plots, verdicts | Every number traces to a saved output |
| 7. Verify | Rescore all saved outputs. Rerun 300 questions x 5 positions and 50 key-value x 5 in a fresh session | Rescoring matches exactly; rerun within ±2 points per position and at least 97% of answers identical |
| 8. Publish | Make the notebook and outputs public; write the README | A stranger can reproduce the charts from the README |

---

## 8. What "reproducible" means

| Level | Check | Pass |
|---|---|---|
| **Exact** | Rescore the published raw outputs | Identical numbers |
| **Rerun** | Rerun a subset (300 questions x 5 positions + 50 key-value x 5) on a free T4 | Each position within ±2 points; at least 97% of individual answers identical; same verdict |

Greedy decoding on GPUs isn't bit-identical across batch sizes and library versions, hence the tolerance. A subset rerun costs about 2 GPU-hours instead of the full 15–20.

---

## 9. Risks

| Risk | Plan |
|---|---|
| vLLM fails on the T4 | Check the TRITON_ATTN and eager settings; otherwise use the transformers fallback and run the 200-example key-value stop |
| float16 gives NaN or garbage | Caught by the smoke test. Try `enforce_eager=True`; if it persists, swap that model |
| GPU time is 1.5x the estimate | Stop key-value at 200 examples; spread runs over two quota weeks |
| T4 x2 uses quota at 2x | Same as above |
| Oracle accuracy is low | Read 20 outputs. If answers are right but mis-formatted, note it; don't change the scorer after seeing the results |
| Flat or inconclusive result | Both are valid outcomes. Report the MDE so readers can tell "no effect" from "not enough data" |
| Key-value near 100% everywhere | Expected for current models on exact lookup; report it and cite it as a ceiling |
| Model already knows many answers | Closed-book baseline plus the optional closed-book-wrong curve |

---

## 10. Add later, if time allows (most valuable first)

1. **Question before and after the documents** (the paper's mitigation). Already supported by the authors' prompt files (`qa_with_query_aware_contextualization.prompt`, `kv_retrieval_with_query_aware_contextualization.prompt`).
2. **30 documents**, to see if the effect grows with input length.
3. **A newer model** (Qwen3.5-4B, or Llama-3.2-3B-Instruct with Meta's licence approved), after the same smoke test.
4. **Base vs instruct** (Qwen2.5-3B).

---

## 11. Deliverables

1. **A public Kaggle notebook** with outputs attached.
2. **A GitHub repo** with the notebook, raw outputs, rendered prompts and a README containing:
   - the charts, and the verdict with its MDE for each model;
   - a comparison with the paper;
   - a list of differences: models, float16, chat templates with default system prompts, 5 key-value positions, key-value example count.
3. **Optional blog post:** "Is 'Lost in the Middle' still true in 2026?"

**Resume line (fill in):**
> Reproduced *Lost in the Middle* (TACL 2024) on current open models using only free GPUs, with a public one-click rerun. Found [a U-shaped / primacy / recency / no detectable] position effect on 2,655 questions, with a first-vs-middle gap of X points (paired bootstrap, 95% CI; minimum detectable effect Y points).
