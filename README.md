# Lost in the Middle, revisited (free-tier reproduction)

Do current small open models still lose information placed in the middle of a long prompt?
This repo re-runs the position experiment of *Lost in the Middle* (Liu et al., TACL 2024,
[arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) on **Qwen2.5-3B-Instruct** and
**Qwen3-4B-Instruct-2507**, using only free Kaggle GPUs.

- Plan and pre-registered analysis: [`lost-in-the-middle-plan.md`](lost-in-the-middle-plan.md)
- Fact-check of the plan against current sources: [`reports/Lost in the middle plan check.md`](reports/Lost%20in%20the%20middle%20plan%20check.md)
- Map-reduce fix (phase A) design: [`docs/superpowers/specs/2026-09-30-map-reduce-fix-design.md`](docs/superpowers/specs/2026-09-30-map-reduce-fix-design.md); audit behind it: [`reports/Lost in middle map reduce audit.md`](reports/Lost%20in%20middle%20map%20reduce%20audit.md)
- Results: *not yet run*

## What runs

| Test | Setup |
|---|---|
| Baselines | Closed-book and oracle, all 2,655 NaturalQuestions-Open questions |
| Main curve | 20 documents, answer at positions 1, 5, 10, 15, 20, all 2,655 questions |
| Long lookup | 300 key-value pairs (~16K tokens), key at positions 1, 75, 150, 225, 300 |

Data, prompts and scoring come from the authors' repo
([nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle), MIT),
pinned at commit `29b8a6d`. Their prompting and metric code is vendored in
`litm/vendor/lost_in_the_middle/` with its licence.

## Layout

```
litm/
  data.py        # authors' files -> prompt items; alignment check
  scoring.py     # authors' QA (first line, best_subspan_em) and KV scorers
  stats.py       # paired bootstrap, exact McNemar, Holm, MDE, shape verdict
  run.py         # vLLM runner, one model per GPU, resumable
  analyze.py     # tables, verdicts, figures, rerun check (CPU)
  generate.py    # generate(prompts, params) interface; greedy fp16 vLLM backend with logprobs
  mapreduce/     # split -> map prompt -> parse -> verify -> choose; run.py (stages), analyze.py
  vendor/        # authors' prompting.py, metrics.py, prompts/ (MIT)
notebooks/litm_kaggle.ipynb   # self-contained Kaggle notebook (built by tools/build_notebook.py)
tests/                        # scoring vs authors', statistics, data alignment
```

## Reproduce

**On Kaggle (GPU):** upload `notebooks/litm_kaggle.ipynb`, set Accelerator to *GPU T4 x2*
and Internet on. Run the stages in order by setting `STAGE` and using *Save & Run All
(Commit)*: `smoke` → `pilot` → `qa` → `kv` → `rerun`. To continue from an earlier version,
attach its output as an input; finished prompts are skipped.

**Locally (CPU), from saved outputs:**

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # or .venv/bin/pip
git clone https://github.com/nelson-liu/lost-in-the-middle third_party/lost-in-the-middle
python -m pytest
python -m litm.analyze results/ --out analysis/
python -m litm.analyze results/ --out analysis/ --tag rerun   # rerun check
```

Rescoring the published outputs must reproduce every number exactly. A GPU rerun of the
subset must land within ±2 points per position with ≥97% identical answers (greedy decoding
is not bit-identical across GPUs and library versions).

## Differences from the paper

- Models: small 2024–2025 open models instead of GPT-3.5, Claude-1.3, MPT-30B and LongChat-13B.
- float16 on T4 (no bfloat16); vLLM 0.18.1 with Triton attention.
- Prompts go into each model's chat template as shipped, including Qwen2.5's default system
  prompt (the authors likewise kept Llama-2's).
- QA scoring strips leading whitespace before taking the first line.
- Key-value: 300 pairs only, 5 positions, 200 or 500 examples.

## Map-reduce fix (phase A)

Answers the same questions from 4-document groups instead of one long prompt. Each group
answers with a quoted evidence sentence; answers whose quote is not in that group's documents
are dropped; the remaining candidates are scored by one shared yes/no prompt and the best is
kept. Run it only on a setting where the baseline shows a position effect (20 documents, then
30, then key-value).

Kaggle: set `MR_TASK` and `MR_SAMPLE`, then run `mr_map` → `mr_check` → `mr_judge` →
`mr_control` → `mr_reduce` → `mr_analyze`. Pilot (200 questions) first; freeze the thresholds
in `litm/mapreduce/verify.py` and the 150-token limit in `litm/mapreduce/run.py`; then run
with `MR_SAMPLE = "all"`. Map outputs are keyed by prompt, so the full run reuses the pilot's.

Locally, from saved outputs:

    python -m litm.mapreduce.run --model qwen2.5-3b --task qa20 --stage reduce --data third_party/lost-in-the-middle --out results --questions all
    python -m litm.mapreduce.analyze results --task qa20 --sample all --out analysis

Pass rules (pre-registered): accuracy averaged over positions no more than 2 points below the
baseline (lower 95% CI bound), **and** a flatter curve (the U contrast, first and last versus
the middle positions, drops with a CI above 0), Holm-corrected across the two models.
