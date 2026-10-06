# Map-Reduce Fix (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a map-reduce mitigation for the "lost in the middle" position effect to the existing `litm` reproduction. The mitigation splits the documents into small groups, answers each group, verifies the answers against the documents and chooses one. Also add the analysis that tests it against the baseline, and the Kaggle notebook stages that run it.

**Architecture:** A new package `litm/mapreduce/` holds small, pure units: `split`, `prompts` (+ `templates/`), `parse`, `verify`, `choose`. Around them sit a resumable GPU/CPU runner (`run.py`) and an analysis module (`analyze.py`). A new `litm/generate.py` puts vLLM behind a single `generate(prompts, params)` interface. Existing baseline modules only get additions: `qa30` data support, final-answer scoring and map-reduce statistics. Every generated output is stored in a JSONL file keyed by a hash of the exact prompt and settings. Identical groups are therefore generated once, reruns resume, and changing a prompt automatically invalidates old outputs.

**Tech Stack:** Python 3.10+, numpy, pydantic 2, regex, xopen, rapidfuzz (new), pytest. On Kaggle: vLLM 0.18.1 (torch 2.10), Qwen2.5-3B-Instruct and Qwen3-4B-Instruct-2507, fp16 on T4.

**Spec:** `docs/superpowers/specs/2026-09-30-map-reduce-fix-design.md`. Background: `reports/Lost in middle map reduce audit.md`. Baseline plan: `lost-in-the-middle-plan.md`.

## Global Constraints

- Same models and revisions as the baseline: `litm.run.MODELS` (`qwen2.5-3b`, `qwen3-4b-2507`). Same authors' data commit `29b8a6d042ce29abccee3db1a73171a107d7e6af`.
- Inference: vLLM 0.18.1, `dtype="float16"`, `attention_backend="TRITON_ATTN"`, one model per T4. Refuse to start unless the compute capability is `(7, 5)`.
- Always pass `SamplingParams(temperature=0.0, max_tokens=…, logprobs=…)` explicitly. Never request `prompt_logprobs`.
- Map calls: `max_tokens=150`, `logprobs=1`, `max_model_len=2048`. Yes/no check: `max_tokens=1`, `logprobs=20`. Judge: `max_tokens=10`. 20/30-doc control: `max_model_len=8192`.
- Groups: 4 documents (20 docs → 5×4; 30 docs → 6×4 + 2×3) and 20 key-value pairs (300 → 15×20). The gold's slot inside its group comes from a seeded (`SEED = 0`), balanced schedule. Nothing else moves.
- Verification: `MIN_EVIDENCE_WORDS = 5`, `FUZZY_THRESHOLD = 90`, `MIN_COVERAGE = 0.9`. Freeze these after the pilot.
- Never break ties by group order. Exact ties go to the alphabetically first normalized answer.
- Score only the final chosen answer string. An abstention scores 0 **before** the metric runs.
- Statistics: paired over question IDs; 10,000 bootstrap resamples; seed 0; non-inferiority margin 0.02; Holm across models at one-sided α = 0.025.
- Read templates in text mode (the checkout has CRLF line endings).
- Do **not** modify `litm/run.py` (the baseline runner, which is being worked on elsewhere). Changes to `litm/data.py`, `litm/analyze.py`, `litm/scoring.py` and `litm/stats.py` are additive only, and existing tests must keep passing.
- Local commands run from the project root with `.venv/Scripts/python` (Windows; Git Bash).
- **The git repo has no commits yet** and its files are uncommitted work from another session. Before the first commit in Task 1, ask the user whether to make an initial commit of the existing files. Stage only the files each task names.

## Review Focus

Each of these is pinned by a test in the task named.
- **The Kaggle session dies mid-stage:** rerunning the same command finishes the job with no duplicate or lost outputs (Task 10, `test_resume_after_an_interrupted_run`).
- **fp16 overflow gives NaN logprobs or "!!!!" output:** the prompt is re-run once. If it still fails it counts as invalid, never as an answer (Task 6 `test_invalid_outputs`, Task 10 `test_broken_outputs_are_retried_once`).
- **Every group says NOT FOUND:** the final answer is an abstention scored 0, with no crash and no "NOT FOUND" string reaching the metric (Task 2 `test_abstention_scores_zero_before_the_metric`, Task 10 `test_every_group_saying_not_found_means_abstain`).
- **The model wraps its reply in Markdown or changes case** (`**Answer:** Paris`): it's parsed like the plain form (Task 6 `test_markdown_and_case_variants`).
- **A prompt template is edited after the pilot:** old outputs are not reused for the new prompt (Task 10 `test_job_key_changes_with_prompt_and_settings`).

---

## File Structure

| File | Responsibility |
|---|---|
| `litm/data.py` (modify) | Add `qa30` (the authors' 30-document files) plus `POSITIONS` / `MIDDLES` lookups |
| `litm/analyze.py` (modify) | Baseline curves and verdicts for `qa30` |
| `litm/scoring.py` (modify) | `score_final`, `strict_em`, `score_final_kv`, `strip_answer_prefix` |
| `litm/stats.py` (modify) | `bootstrap_mean`, `one_sided_p`, `u_contrast`, `non_inferiority`, `delta_u`, `recovered_share`, `range_minus_null`, `auroc` |
| `litm/generate.py` (create) | `GenParams`, `Generation`, `to_generation`, `VLLMGenerator` |
| `litm/mapreduce/__init__.py` (create) | Empty package marker |
| `litm/mapreduce/split.py` (create) | Group sizes, gold-slot schedule, map items for QA and key-value, sampling |
| `litm/mapreduce/templates/*.prompt` (create) | `qa_map`, `kv_map`, `qa_check`, `qa_judge` |
| `litm/mapreduce/prompts.py` (create) | Prompt builders (the authors' document and key-value formatting) |
| `litm/mapreduce/parse.py` (create) | Parse map, yes/no and judge outputs; detect broken outputs; answer logprob |
| `litm/mapreduce/verify.py` (create) | Evidence checks for QA (exact, then fuzzy) and key-value |
| `litm/mapreduce/choose.py` (create) | Primary rule, vote, judge and key-value choice rules |
| `litm/mapreduce/run.py` (create) | CLI: resumable `map` / `check` / `judge` / `control` GPU stages, then CPU `reduce` |
| `litm/mapreduce/analyze.py` (create) | Tests 1–4, diagnostics, report, plot |
| `tools/build_notebook.py` (modify) | `qa30` and `mr_*` stages, rapidfuzz install, `build()` for testing |
| `requirements.txt`, `README.md` (modify) | rapidfuzz; the map-reduce section |
| `tests/test_*.py` (create) | One new test file per unit (existing test files are not edited) |

---

### Task 1: Baseline support for 30 documents

This is the spec's §2 gate, step 2: map-reduce may have to run on 30 documents, which needs a 30-document baseline.

**Files:**
- Modify: `litm/data.py` (constants block at lines 24-27; `qa_path`, `qa_items`, `all_items`)
- Modify: `litm/analyze.py:10` (import), `:18` (`CURVES`), `plot()`
- Test: `tests/test_qa30.py`

**Interfaces:**
- Produces:
  - `litm.data.QA30_POSITIONS = [0, 4, 9, 14, 19, 24, 29]`, `QA30_MIDDLE = 14`
  - `POSITIONS: dict[str, list[int]]` and `MIDDLES: dict[str, int]`, keyed by `"qa20" | "qa30" | "kv300"`
  - `qa_path(root, gold_index, n_docs=20) -> Path`
  - `qa_items` and `all_items` accept task `"qa30"`
  - `litm.analyze.CURVES["qa30"]`

- [ ] **Step 1: Ask the user about an initial commit.** The repo has no commits. Ask: "Commit the existing files as an initial commit before this plan's changes?" If yes, run:

```bash
git add .gitignore README.md lost-in-the-middle-plan.md pyproject.toml requirements.txt litm tests tools notebooks docs reports
git commit -m "chore: initial commit of the baseline reproduction

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Write the failing test** at `tests/test_qa30.py`:

```python
from pathlib import Path

import pytest

from litm.analyze import CURVES
from litm.data import QA30_POSITIONS, all_items, qa_items

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "lost-in-the-middle"
needs_data = pytest.mark.skipif(not ROOT.exists(), reason="authors' repo not checked out in third_party/")


@needs_data
def test_qa30_items():
    item = next(qa_items(ROOT, "qa30", 29))
    assert item["task"] == "qa30" and item["position"] == 29
    assert item["prompt"].count("Document [") == 30
    counts = {}
    for x in all_items(ROOT, ["qa30"]):
        counts[x["position"]] = counts.get(x["position"], 0) + 1
    assert counts == {p: 2655 for p in QA30_POSITIONS}


def test_qa30_curve_uses_position_15_as_middle():
    assert CURVES["qa30"] == (QA30_POSITIONS, 14)
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_qa30.py -v`
Expected: FAIL with `ImportError: cannot import name 'QA30_POSITIONS'`

- [ ] **Step 4: Implement.** In `litm/data.py`, replace the constants block

```python
QA_POSITIONS = [0, 4, 9, 14, 19]  # 0-based; paper positions 1, 5, 10, 15, 20
KV_POSITIONS = [0, 74, 149, 224, 299]  # 0-based; positions 1, 75, 150, 225, 300
QA_MIDDLE = 9
KV_MIDDLE = 149
```

with

```python
QA_POSITIONS = [0, 4, 9, 14, 19]  # 0-based; paper positions 1, 5, 10, 15, 20
KV_POSITIONS = [0, 74, 149, 224, 299]  # 0-based; positions 1, 75, 150, 225, 300
QA30_POSITIONS = [0, 4, 9, 14, 19, 24, 29]  # 0-based; the authors' 30-document files
QA_MIDDLE = 9
KV_MIDDLE = 149
QA30_MIDDLE = 14
POSITIONS = {"qa20": QA_POSITIONS, "qa30": QA30_POSITIONS, "kv300": KV_POSITIONS}
MIDDLES = {"qa20": QA_MIDDLE, "qa30": QA30_MIDDLE, "kv300": KV_MIDDLE}
```

Replace `qa_path` with:

```python
def qa_path(root: Path, gold_index: int, n_docs: int = 20) -> Path:
    return root / "qa_data" / f"{n_docs}_total_documents" / f"nq-open-{n_docs}_total_documents_gold_at_{gold_index}.jsonl.gz"
```

In `qa_items`, replace the docstring and first line with:

```python
    """task is 'closedbook', 'oracle', 'qa20' or 'qa30'; position is the 0-based gold index for qa20/qa30."""
    path = qa_path(root, position, int(task[2:])) if task in ("qa20", "qa30") else oracle_path(root)
```

In `all_items`, change the docstring to `"""Build every item for the named tasks: closedbook, oracle, qa20, qa30, kv300."""` and replace

```python
        elif task == "qa20":
            for position in QA_POSITIONS:
```

with

```python
        elif task in ("qa20", "qa30"):
            for position in POSITIONS[task]:
```

In `litm/analyze.py`, change the import to

```python
from litm.data import KV_MIDDLE, KV_POSITIONS, QA30_MIDDLE, QA30_POSITIONS, QA_MIDDLE, QA_POSITIONS
```

and replace the `CURVES` line with

```python
CURVES = {"qa20": (QA_POSITIONS, QA_MIDDLE), "qa30": (QA30_POSITIONS, QA30_MIDDLE), "kv300": (KV_POSITIONS, KV_MIDDLE)}
TITLES = {"qa20": "20 documents", "qa30": "30 documents", "kv300": "300 key-value pairs"}
```

In `plot()`, change `if task == "qa20":` (the closed-book/oracle block inside the model loop) to `if task.startswith("qa"):`. Replace

```python
        ax.set_xlabel("Position of the answer" if task == "qa20" else "Position of the key")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("20 documents" if task == "qa20" else "300 key-value pairs")
```

with

```python
        ax.set_xlabel("Position of the answer" if task.startswith("qa") else "Position of the key")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(TITLES[task])
```

Leave the paper reference lines (`if task == "qa20":` before `PAPER_QA20`) unchanged.

- [ ] **Step 5: Run the new and existing tests**

Run: `.venv/Scripts/python -m pytest tests/test_qa30.py tests/test_data.py tests/test_run.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add litm/data.py litm/analyze.py tests/test_qa30.py
git commit -m "feat: baseline support for the authors' 30-document files

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Final-answer scoring

**Files:**
- Modify: `litm/scoring.py`
- Test: `tests/test_scoring_final.py`

**Interfaces:**
- Produces:
  - `strip_answer_prefix(answer: str) -> str`
  - `score_final(answer: str, abstained: bool, answers: list[str]) -> int`
  - `strict_em(answer: str, abstained: bool, answers: list[str]) -> int`
  - `score_final_kv(answer: str, abstained: bool, value: str) -> int`

- [ ] **Step 1: Write the failing test** at `tests/test_scoring_final.py`:

```python
from litm.scoring import score_final, score_final_kv, strict_em, strip_answer_prefix

IDX_1451 = ["a rotationally symmetric saltire", "the symbol ⊕", "*"]  # "*" normalizes to ""
IDX_1840 = ["S"]


def test_abstention_scores_zero_before_the_metric():
    assert score_final("", True, ["Paris"]) == 0
    assert score_final("NOT FOUND", True, IDX_1840) == 0  # "S" would match "not found"
    assert score_final("", True, IDX_1451) == 0


def test_idx_1451_matches_any_real_answer_in_both_methods():
    # Documented quirk: kept in both methods, so it cancels in paired comparisons.
    assert score_final("Paris", False, IDX_1451) == 1


def test_only_the_answer_string_is_scored():
    assert score_final("Answer: Wilhelm Röntgen", False, ["Röntgen"]) == 1
    assert score_final("Einstein", False, ["Röntgen"]) == 0
    assert strip_answer_prefix("  answer: Paris ") == "Paris"


def test_strict_em():
    assert strict_em("The Beatles", False, ["Beatles"]) == 1
    assert strict_em("The Beatles and Wings", False, ["Beatles"]) == 0
    assert strict_em("anything", False, ["*"]) == 0
    assert strict_em("Beatles", True, ["Beatles"]) == 0


def test_kv_final():
    value = "703a7ce5-f17f-4e6d-b895-5836ba5ec71c"
    assert score_final_kv(value, False, value) == 1
    assert score_final_kv("", True, value) == 0
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_scoring_final.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_final'`

- [ ] **Step 3: Implement.** In `litm/scoring.py`, replace the imports

```python
from typing import List

from litm.vendor.lost_in_the_middle.metrics import best_subspan_em
```

with

```python
import re
from typing import List

from litm.vendor.lost_in_the_middle.metrics import best_subspan_em, normalize_answer
```

and append at the end of the file:

```python


def strip_answer_prefix(answer: str) -> str:
    return re.sub(r"^\s*answer\s*:\s*", "", answer, flags=re.I).strip()


def score_final(answer: str, abstained: bool, answers: List[str]) -> int:
    """Map-reduce final answers (spec §5): an abstention scores 0 before the metric runs, so a
    gold answer like "S" can never match the words NOT FOUND."""
    if abstained or not answer.strip():
        return 0
    return int(best_subspan_em(prediction=strip_answer_prefix(answer), ground_truths=answers))


def strict_em(answer: str, abstained: bool, answers: List[str]) -> int:
    """Secondary metric: the normalized answer equals a normalized gold answer."""
    prediction = normalize_answer(strip_answer_prefix(answer))
    if abstained or not prediction:
        return 0
    return int(any(prediction == normalize_answer(gold) for gold in answers))


def score_final_kv(answer: str, abstained: bool, value: str) -> int:
    return 0 if abstained else score_kv(answer, value)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_scoring_final.py tests/test_scoring.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add litm/scoring.py tests/test_scoring_final.py
git commit -m "feat: score map-reduce final answers with abstention as 0

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Map-reduce statistics

**Files:**
- Modify: `litm/stats.py` (append)
- Test: `tests/test_stats_mapreduce.py`

**Interfaces:**
- Consumes: `N_BOOTSTRAP`, `SEED` (already in `litm/stats.py`)
- Produces (all score inputs are `[questions, positions]` arrays of 0/1, with rows aligned on question IDs):
  - `NI_MARGIN = 0.02`
  - `bootstrap_mean(x) -> {"mean", "low", "high", "boot"}`
  - `one_sided_p(boot, threshold) -> float` (share of draws ≤ threshold, +1 smoothed)
  - `u_contrast(scores) -> np.ndarray` (per question)
  - `non_inferiority(new, ref, margin=NI_MARGIN) -> {"diff", "low", "high", "margin", "p", "pass"}`
  - `delta_u(base, new) -> {"u_base", "u_new", "diff", "low", "high", "p", "pass"}`
  - `recovered_share(base, new, middle, first=0) -> {"share", "low", "high", "unstable"}`
  - `range_minus_null(scores, n_perm=N_BOOTSTRAP, seed=SEED) -> {"range", "null_mean", "excess"}`
  - `auroc(scores, labels) -> float`

- [ ] **Step 1: Write the failing test** at `tests/test_stats_mapreduce.py`:

```python
import numpy as np
import pytest

from litm.stats import auroc, delta_u, non_inferiority, one_sided_p, range_minus_null, recovered_share, u_contrast


def test_u_contrast():
    s = np.array([[1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])
    assert u_contrast(s).tolist() == pytest.approx([1.0, 0.0, -1.0])


def test_non_inferiority_needs_the_lower_bound_above_minus_margin():
    rng = np.random.default_rng(0)
    ref = (rng.random((2000, 5)) < 0.6).astype(int)
    worse = ref.copy()
    worse[(rng.random(ref.shape) < 0.05) & (ref == 1)] = 0  # about 3 points worse
    result = non_inferiority(worse, ref)
    assert result["diff"] < -0.02 and not result["pass"]
    assert non_inferiority(ref, ref)["pass"]


def test_non_inferiority_does_not_reward_noise():
    rng = np.random.default_rng(1)
    ref = (rng.random((30, 5)) < 0.6).astype(int)
    new = (rng.random((30, 5)) < 0.6).astype(int)
    result = non_inferiority(new, ref)
    # The old rule ("CI not entirely below 0") would pass this tiny, noisy comparison.
    assert result["high"] > 0 and not result["pass"]


def test_delta_u_detects_a_flattened_curve():
    rng = np.random.default_rng(2)
    p_base = np.array([0.75, 0.55, 0.55, 0.55, 0.70])
    base = (rng.random((2000, 5)) < p_base).astype(int)
    flat = (rng.random((2000, 5)) < 0.65).astype(int)
    result = delta_u(base, flat)
    assert result["diff"] == pytest.approx(0.175, abs=0.05)
    assert result["pass"] and result["p"] < 0.001
    assert not delta_u(base, base)["pass"]


def test_one_sided_p():
    assert one_sided_p(np.array([1.0, 2.0, 3.0]), 0.0) == pytest.approx(0.25)
    assert one_sided_p(np.array([-1.0, -2.0, 3.0]), 0.0) == pytest.approx(0.75)


def test_recovered_share():
    base = np.array([[1, 0]] * 50 + [[1, 1]] * 50)  # first 100%, middle 50%
    new = np.array([[1, 1]] * 75 + [[1, 0]] * 25)   # middle 75%
    r = recovered_share(base, new, middle=1)
    assert r["share"] == pytest.approx(0.5) and not r["unstable"]
    flat = np.array([[1, 1]] * 50 + [[0, 0]] * 50)
    assert recovered_share(flat, new, middle=1)["unstable"]


def test_range_minus_null_is_near_zero_without_a_position_effect():
    rng = np.random.default_rng(3)
    scores = (rng.random((500, 5)) < 0.6).astype(int)
    r = range_minus_null(scores, n_perm=2000)
    assert r["null_mean"] > 0.02            # the raw range is biased upwards
    assert abs(r["excess"]) < 0.04
    assert range_minus_null(scores, n_perm=2000) == r


def test_auroc():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.5, 0.5], [0, 1]) == 0.5
    assert auroc([0.9, 0.1], [0, 1]) == 0.0
    assert np.isnan(auroc([0.3], [1]))
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_stats_mapreduce.py -v`
Expected: FAIL with `ImportError: cannot import name 'auroc'`

- [ ] **Step 3: Implement.** Append to `litm/stats.py`:

```python


# ---------------------------------------------------------------------------
# Map-reduce comparison (spec §6). Inputs are [questions, positions] arrays of 0/1 scores,
# rows aligned on the same question IDs, so every resample carries both methods together.

NI_MARGIN = 0.02  # pre-registered non-inferiority margin: 2 points


def bootstrap_mean(x: Sequence[float], n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> Dict:
    """Mean of a per-question vector, with its percentile 95% CI and the bootstrap draws."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(x.mean()), "low": float(low), "high": float(high), "boot": boot}


def one_sided_p(boot: np.ndarray, threshold: float) -> float:
    """Bootstrap p-value for H0: mean <= threshold."""
    return float((np.sum(boot <= threshold) + 1) / (len(boot) + 1))


def u_contrast(scores) -> np.ndarray:
    """Per question: mean(first, last position) - mean(the positions in between)."""
    s = np.asarray(scores, dtype=float)
    return s[:, [0, -1]].mean(axis=1) - s[:, 1:-1].mean(axis=1)


def non_inferiority(new, ref, margin: float = NI_MARGIN) -> Dict:
    """Test 1: average over positions per question, then require CI low > -margin."""
    diff = np.asarray(new, dtype=float).mean(axis=1) - np.asarray(ref, dtype=float).mean(axis=1)
    r = bootstrap_mean(diff)
    return {"diff": r["mean"], "low": r["low"], "high": r["high"], "margin": margin,
            "p": one_sided_p(r["boot"], -margin), "pass": r["low"] > -margin}


def delta_u(base, new) -> Dict:
    """Test 2: U(baseline) - U(new) per question; pass if its CI is above 0 (new is flatter)."""
    u_base, u_new = u_contrast(base), u_contrast(new)
    r = bootstrap_mean(u_base - u_new)
    return {"u_base": float(u_base.mean()), "u_new": float(u_new.mean()), "diff": r["mean"],
            "low": r["low"], "high": r["high"], "p": one_sided_p(r["boot"], 0.0), "pass": r["low"] > 0}


def recovered_share(base, new, middle: int, first: int = 0, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> Dict:
    """Test 3: (new_mid - base_mid) / (base_first - base_mid), unstable if the denominator's CI touches 0."""
    b, n = np.asarray(base, dtype=float), np.asarray(new, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(b), size=(n_boot, len(b)))
    num = n[:, middle][idx].mean(axis=1) - b[:, middle][idx].mean(axis=1)
    den = b[:, first][idx].mean(axis=1) - b[:, middle][idx].mean(axis=1)
    point_den = b[:, first].mean() - b[:, middle].mean()
    point = (n[:, middle].mean() - b[:, middle].mean()) / point_den if point_den else float("nan")
    ok = den != 0
    low, high = np.percentile(num[ok] / den[ok], [2.5, 97.5]) if ok.any() else (float("nan"), float("nan"))
    den_low, den_high = np.percentile(den, [2.5, 97.5])
    return {"share": float(point), "low": float(low), "high": float(high), "unstable": bool(den_low <= 0 <= den_high)}


def range_minus_null(scores, n_perm: int = N_BOOTSTRAP, seed: int = SEED, chunk: int = 250) -> Dict:
    """Best-minus-worst position accuracy, and its mean when position labels are shuffled
    within each question (the "no position effect" value). Descriptive only."""
    s = np.asarray(scores, dtype=float)
    acc = s.mean(axis=0)
    rng = np.random.default_rng(seed)
    null = []
    for start in range(0, n_perm, chunk):
        k = min(chunk, n_perm - start)
        order = np.argsort(rng.random((k, *s.shape)), axis=2)
        means = np.take_along_axis(np.broadcast_to(s, (k, *s.shape)), order, axis=2).mean(axis=1)
        null.append(means.max(axis=1) - means.min(axis=1))
    null_mean = float(np.concatenate(null).mean())
    observed = float(acc.max() - acc.min())
    return {"range": observed, "null_mean": null_mean, "excess": observed - null_mean}


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve (ties count half); nan if either class is missing."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    ranks = (np.bincount(inverse, weights=ranks) / counts)[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_stats_mapreduce.py tests/test_stats.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add litm/stats.py tests/test_stats_mapreduce.py
git commit -m "feat: non-inferiority, difference-in-contrasts and diagnostic statistics

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Split documents into groups

**Files:**
- Create: `litm/mapreduce/__init__.py` (empty), `litm/mapreduce/split.py`
- Test: `tests/test_mr_split.py`

**Interfaces:**
- Consumes: `litm.data.qa_path(root, gold_index, n_docs)`, `kv_path`, `read_jsonl` (Task 1)
- Produces:
  - `SEED = 0`, `QA_GROUP_SIZE = 4`, `KV_GROUP_SIZE = 20`, `PILOT_SIZE = 200`
  - `group_sizes(n_items, group_size) -> list[int]`
  - `group_of(index, sizes) -> int`
  - `gold_slots(n_questions, position, group_size, seed=SEED) -> np.ndarray`
  - `split_with_gold(items, gold_index, sizes, slot) -> (list[list], int)`
  - `content_key(*parts) -> str` (24 hex chars)
  - `contains_answer(text, answers) -> bool`
  - `sample_ids(n_total, sample: "pilot" | "all", limit=None, seed=SEED) -> list[int]`
  - `qa_map_items(root, n_docs, positions, question_ids) -> list[dict]`
  - `kv_map_items(root, positions, question_ids, n_keys=300) -> list[dict]`
  - An **item** dict has these keys:
    - `task`, `position`, `idx`, `group`, `question`, `answers`
    - `docs`: QA `[{"title", "text"}]`; key-value `[[key, value]]`
    - `is_gold_group`, `gold_slot` (int or None), `mate_ranks` (list or None), `n_answer_bearing`, `key`
    - key-value items also have `query_key`

- [ ] **Step 1: Write the failing test** at `tests/test_mr_split.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from litm.data import qa_path, read_jsonl
from litm.mapreduce.split import (
    contains_answer,
    content_key,
    gold_slots,
    group_of,
    group_sizes,
    kv_map_items,
    qa_map_items,
    sample_ids,
    split_with_gold,
)

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "lost-in-the-middle"
needs_data = pytest.mark.skipif(not ROOT.exists(), reason="authors' repo not checked out in third_party/")


def test_group_sizes():
    assert group_sizes(20, 4) == [4] * 5
    assert group_sizes(30, 4) == [4] * 6 + [3] * 2
    assert group_sizes(300, 20) == [20] * 15


def test_group_of():
    assert [group_of(i, [4] * 5) for i in (0, 3, 4, 9, 14, 19)] == [0, 0, 1, 2, 3, 4]
    sizes = [4] * 6 + [3] * 2
    assert group_of(24, sizes) == 6 and group_of(26, sizes) == 6 and group_of(27, sizes) == 7 and group_of(29, sizes) == 7


def test_gold_slots_are_balanced_seeded_and_differ_by_position():
    slots = gold_slots(2655, 9, 4)
    counts = np.bincount(slots, minlength=4)
    assert counts.max() - counts.min() <= 1
    assert (gold_slots(2655, 9, 4) == slots).all()
    assert not (gold_slots(2655, 4, 4) == slots).all()


def test_split_with_gold_moves_only_the_gold():
    items = list("abcdefghijklmnopqrst")  # gold is "j", index 9
    groups, gold_group = split_with_gold(items, 9, [4] * 5, slot=3)
    assert gold_group == 2
    assert groups[2] == ["i", "k", "l", "j"]
    assert groups[0] == list("abcd") and groups[1] == list("efgh") and groups[4] == list("qrst")


def test_content_key_depends_on_order():
    assert content_key("qa20", 1, [["t", "a"], ["t", "b"]]) != content_key("qa20", 1, [["t", "b"], ["t", "a"]])
    assert content_key("qa20", 1, [["t", "a"]]) == content_key("qa20", 1, [["t", "a"]])


def test_contains_answer_uses_the_metric_rule_and_skips_empty_answers():
    assert contains_answer("The Beatles were a band", ["beatles"])
    assert not contains_answer("anything at all", ["*"])  # "*" normalizes to ""


def test_sample_ids():
    pilot = sample_ids(2655, "pilot")
    assert len(pilot) == 200 and pilot == sorted(pilot) and pilot == sample_ids(2655, "pilot")
    assert sample_ids(10, "all") == list(range(10))
    assert sample_ids(2655, "pilot", limit=3) == pilot[:3]
    assert sample_ids(20, "pilot") == list(range(20))


@needs_data
def test_qa_map_items_structure_and_dedupe():
    positions = [0, 4, 9, 14, 19]
    items = qa_map_items(ROOT, 20, positions, [0, 1])
    assert len(items) == 5 * 2 * 5
    for position in positions:
        source = read_jsonl(qa_path(ROOT, position))
        for idx in (0, 1):
            groups = [i for i in items if i["position"] == position and i["idx"] == idx]
            assert [len(g["docs"]) for g in groups] == [4] * 5
            gold = [g for g in groups if g["is_gold_group"]]
            assert len(gold) == 1 and gold[0]["group"] == position // 4
            assert len(gold[0]["mate_ranks"]) == 3
            gold_ctx = source[idx]["ctxs"][position]
            assert gold[0]["docs"][gold[0]["gold_slot"]] == {"title": gold_ctx["title"], "text": gold_ctx["text"]}
    # Gold-free groups repeat across positions: 8 of them + 5 gold groups per question.
    assert len({i["key"] for i in items if i["idx"] == 0}) == 13


@needs_data
def test_qa30_items_use_uneven_groups():
    items = qa_map_items(ROOT, 30, [29], [0])
    assert [len(i["docs"]) for i in items] == [4] * 6 + [3] * 2
    assert [i["group"] for i in items if i["is_gold_group"]] == [7]


@needs_data
def test_kv_map_items_structure():
    items = kv_map_items(ROOT, [0, 149], [0])
    assert len(items) == 2 * 15
    for position in (0, 149):
        groups = [i for i in items if i["position"] == position]
        assert [len(g["docs"]) for g in groups] == [20] * 15
        gold = [g for g in groups if g["is_gold_group"]]
        assert len(gold) == 1 and gold[0]["group"] == position // 20
        key, value = gold[0]["docs"][gold[0]["gold_slot"]]
        assert key == gold[0]["query_key"] and [value] == gold[0]["answers"]
        assert sum(g["query_key"] in [k for k, _ in g["docs"]] for g in groups) == 1
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_mr_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litm.mapreduce'`

- [ ] **Step 3: Implement.** Create an empty `litm/mapreduce/__init__.py`, and `litm/mapreduce/split.py`:

```python
"""Cut each prompt's documents (or key-value pairs) into small groups for map calls (spec §3, §9).

Groups are contiguous blocks in the original order. Inside the gold's group, the gold is moved to
a slot assigned by a seeded, balanced schedule, so each slot is used equally often at every global
position. Nothing else moves.
"""
import hashlib
import json
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from litm.data import kv_path, qa_path, read_jsonl
from litm.vendor.lost_in_the_middle.metrics import normalize_answer

SEED = 0
QA_GROUP_SIZE = 4
KV_GROUP_SIZE = 20
PILOT_SIZE = 200


def group_sizes(n_items: int, group_size: int) -> List[int]:
    """As even as possible, none above group_size: 20 -> 5x4, 30 -> 6x4 + 2x3, 300 -> 15x20."""
    n_groups = ceil(n_items / group_size)
    base, extra = divmod(n_items, n_groups)
    return [base + 1] * extra + [base] * (n_groups - extra)


def group_of(index: int, sizes: Sequence[int]) -> int:
    """Which group the item at `index` falls in."""
    return int(np.searchsorted(np.cumsum(sizes), index, side="right"))


def gold_slots(n_questions: int, position: int, group_size: int, seed: int = SEED) -> np.ndarray:
    """Slot of the gold inside its group, per question, for one global position.
    Every slot gets n_questions / group_size questions (+-1)."""
    order = np.random.default_rng([seed, position]).permutation(n_questions)
    slots = np.empty(n_questions, dtype=int)
    slots[order] = np.arange(n_questions) % group_size
    return slots


def split_with_gold(items: Sequence, gold_index: int, sizes: Sequence[int], slot: int) -> Tuple[List[list], int]:
    """Contiguous groups of `items`; the gold moves to `slot` inside its own group."""
    groups, start = [], 0
    for size in sizes:
        groups.append(list(items[start:start + size]))
        start += size
    gold_group = group_of(gold_index, sizes)
    group = groups[gold_group]
    gold = group.pop(gold_index - sum(sizes[:gold_group]))
    group.insert(slot, gold)
    return groups, gold_group


def content_key(*parts) -> str:
    """Stable short hash of JSON-serialisable parts."""
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def contains_answer(text: str, answers: Sequence[str]) -> bool:
    """The metric's own rule (normalized substring); answers that normalize to "" never count."""
    norm = normalize_answer(text)
    return any(normalize_answer(a) and normalize_answer(a) in norm for a in answers)


def sample_ids(n_total: int, sample: str, limit=None, seed: int = SEED) -> List[int]:
    """'pilot': a seeded 200-question sample; 'all': every question. `limit` keeps the first N."""
    if sample == "pilot":
        rng = np.random.default_rng(seed)
        ids = sorted(int(i) for i in rng.choice(n_total, size=min(PILOT_SIZE, n_total), replace=False))
    elif sample == "all":
        ids = list(range(n_total))
    else:
        raise ValueError(f"Unknown sample: {sample}")
    return ids[:limit] if limit is not None else ids


def qa_map_items(root: Path, n_docs: int, positions: Sequence[int], question_ids: Sequence[int]) -> List[Dict]:
    """One item per (position, question, group), read from the authors' files."""
    task = f"qa{n_docs}"
    sizes = group_sizes(n_docs, QA_GROUP_SIZE)
    items = []
    for position in positions:
        examples = read_jsonl(qa_path(root, position, n_docs))
        gold_group = group_of(position, sizes)
        slots = gold_slots(len(examples), position, sizes[gold_group])
        for idx in sorted(set(question_ids)):
            example = examples[idx]
            docs, rank = [], 0
            for ctx in example["ctxs"]:
                is_gold = bool(ctx.get("isgold"))
                docs.append({"title": ctx["title"], "text": ctx["text"], "isgold": is_gold, "rank": None if is_gold else rank})
                rank += 0 if is_gold else 1
            if not docs[position]["isgold"]:
                raise ValueError(f"{task} line {idx}: the gold document is not at index {position}")
            groups, _ = split_with_gold(docs, position, sizes, int(slots[idx]))
            for g, group in enumerate(groups):
                is_gold_group = g == gold_group
                items.append({
                    "task": task, "position": position, "idx": idx, "group": g,
                    "question": example["question"], "answers": example["answers"],
                    "docs": [{"title": d["title"], "text": d["text"]} for d in group],
                    "is_gold_group": is_gold_group,
                    "gold_slot": int(slots[idx]) if is_gold_group else None,
                    "mate_ranks": [d["rank"] for d in group if not d["isgold"]] if is_gold_group else None,
                    "n_answer_bearing": sum(
                        contains_answer(f"{d['title']} {d['text']}", example["answers"]) for d in group if not d["isgold"]
                    ),
                    "key": content_key(task, idx, [[d["title"], d["text"]] for d in group]),
                })
    return items


def kv_map_items(root: Path, positions: Sequence[int], question_ids: Sequence[int], n_keys: int = 300) -> List[Dict]:
    """One item per (position, example, group of 20 pairs). The gold pair is first moved to
    `position`, exactly as litm.data.kv_items does."""
    task = f"kv{n_keys}"
    examples = read_jsonl(kv_path(root, n_keys))
    sizes = group_sizes(n_keys, KV_GROUP_SIZE)
    items = []
    for position in positions:
        gold_group = group_of(position, sizes)
        slots = gold_slots(len(examples), position, sizes[gold_group])
        for idx in sorted(set(question_ids)):
            example = examples[idx]
            records = deepcopy(example["ordered_kv_records"])
            gold = records.pop(records.index([example["key"], example["value"]]))
            records.insert(position, gold)
            groups, _ = split_with_gold(records, position, sizes, int(slots[idx]))
            for g, group in enumerate(groups):
                is_gold_group = g == gold_group
                items.append({
                    "task": task, "position": position, "idx": idx, "group": g,
                    "question": example["key"], "query_key": example["key"], "answers": [example["value"]],
                    "docs": [list(pair) for pair in group],
                    "is_gold_group": is_gold_group,
                    "gold_slot": int(slots[idx]) if is_gold_group else None,
                    "mate_ranks": None,
                    "n_answer_bearing": 0,
                    "key": content_key(task, idx, group),
                })
    return items
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_mr_split.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add litm/mapreduce/__init__.py litm/mapreduce/split.py tests/test_mr_split.py
git commit -m "feat: split documents into groups with a balanced gold slot

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Prompt templates and builders

**Files:**
- Create: `litm/mapreduce/templates/qa_map.prompt`, `kv_map.prompt`, `qa_check.prompt`, `qa_judge.prompt`
- Create: `litm/mapreduce/prompts.py`
- Test: `tests/test_mr_prompts.py`

**Interfaces:**
- Consumes: item dicts from Task 4 (`task`, `question`, `docs`, `query_key`)
- Produces:
  - `format_documents(docs) -> str`, `format_kv_records(pairs) -> str`
  - `qa_map_prompt(question, docs) -> str`, `kv_map_prompt(pairs, key) -> str`, `map_prompt(item) -> str`
  - `check_prompt(question, evidence, answer) -> str`
  - `judge_prompt(question, candidates: list[tuple[answer, evidence]]) -> str`

- [ ] **Step 1: Write the failing test** at `tests/test_mr_prompts.py`:

```python
from litm.mapreduce.prompts import (
    check_prompt,
    format_documents,
    format_kv_records,
    judge_prompt,
    kv_map_prompt,
    map_prompt,
    qa_map_prompt,
)
from litm.vendor.lost_in_the_middle.prompting import Document, get_kv_retrieval_prompt, get_qa_prompt

DOCS = [{"title": "T1", "text": "Alpha text."}, {"title": "T2", "text": "Beta text."}]


def test_document_lines_match_the_authors_byte_for_byte():
    authors = get_qa_prompt("q?", [Document(title=d["title"], text=d["text"]) for d in DOCS], False, False)
    assert format_documents(DOCS) in authors


def test_qa_map_prompt():
    prompt = qa_map_prompt("who?", DOCS)
    assert prompt.startswith("Write a high-quality answer for the given question using only the provided search results")
    assert "If none of the search results contain the answer, write NOT FOUND." in prompt
    assert "Document [1](Title: T1) Alpha text.\nDocument [2](Title: T2) Beta text." in prompt
    assert prompt.index("Question: who?") < prompt.index("Reply in exactly this format:")
    assert prompt.endswith("Evidence: <copy the sentence from the documents that contains the answer>")
    assert "\r" not in prompt


def test_kv_formatter_matches_the_authors_and_allows_a_missing_key():
    pairs = [["k1", "v1"], ["k2", "v2"]]
    assert format_kv_records(pairs) in get_kv_retrieval_prompt(pairs, "k1")
    prompt = kv_map_prompt(pairs, "absent-key")  # the authors' builder raises ValueError here
    assert 'Key: "absent-key"' in prompt and "NOT FOUND" in prompt
    assert prompt.endswith('Evidence: <copy the "key": "value" pair from the JSON object>')


def test_map_prompt_dispatches_on_task():
    assert map_prompt({"task": "qa20", "question": "who?", "docs": DOCS}) == qa_map_prompt("who?", DOCS)
    kv = {"task": "kv300", "query_key": "k1", "docs": [["k1", "v1"], ["k2", "v2"]]}
    assert map_prompt(kv) == kv_map_prompt(kv["docs"], "k1")


def test_check_and_judge_prompts():
    check = check_prompt("who won?", "Röntgen won in 1901.", "Röntgen")
    assert check == (
        "Question: who won?\nSentence: Röntgen won in 1901.\nProposed answer: Röntgen\n"
        "Does the sentence show that the proposed answer is correct? Answer Yes or No."
    )
    judge = judge_prompt("who won?", [("Röntgen", "Röntgen won."), ("Curie", "Curie won.")])
    assert "[1] Answer: Röntgen\n    Sentence: Röntgen won.\n[2] Answer: Curie\n    Sentence: Curie won." in judge
    assert judge.endswith("Choice: <candidate number>")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_mr_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litm.mapreduce.prompts'`

- [ ] **Step 3: Create the templates.** Each file ends with a single newline.

`litm/mapreduce/templates/qa_map.prompt`:

```
Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). If none of the search results contain the answer, write NOT FOUND.

{search_results}

Question: {question}
Reply in exactly this format:
Answer: <short answer, or NOT FOUND>
Evidence: <copy the sentence from the documents that contains the answer>
```

`litm/mapreduce/templates/kv_map.prompt`:

```
Extract the value corresponding to the specified key in the JSON object below. If the key is not in the JSON object, write NOT FOUND.

JSON data:
{formatted_kv_records}

Key: "{key}"
Reply in exactly this format:
Value: <the value, or NOT FOUND>
Evidence: <copy the "key": "value" pair from the JSON object>
```

`litm/mapreduce/templates/qa_check.prompt`:

```
Question: {question}
Sentence: {evidence}
Proposed answer: {answer}
Does the sentence show that the proposed answer is correct? Answer Yes or No.
```

`litm/mapreduce/templates/qa_judge.prompt`:

```
Question: {question}

Candidate answers, each with the sentence it came from:
{candidates}

Which candidate answers the question correctly? Reply in exactly this format:
Choice: <candidate number>
```

- [ ] **Step 4: Implement** `litm/mapreduce/prompts.py`:

```python
"""Prompt builders for map, yes/no check and judge calls (spec §4).

Document and key-value lines are formatted exactly as the authors' prompting.py does. Templates
are read in text mode, so CRLF checkouts produce the same prompts.
"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

TEMPLATES = Path(__file__).parent / "templates"


def _template(name: str) -> str:
    with open(TEMPLATES / name, encoding="utf-8") as f:
        return f.read().rstrip("\n")


def format_documents(docs: Sequence[Dict]) -> str:
    return "\n".join(f"Document [{i + 1}](Title: {d['title']}) {d['text']}" for i, d in enumerate(docs))


def format_kv_records(pairs: Sequence[Sequence[str]]) -> str:
    """The authors' JSON-ish serialisation, without their check that the query key is present."""
    out = ""
    for index, (key, value) in enumerate(pairs):
        start = "{" if index == 0 else " "
        end = ",\n" if index != len(pairs) - 1 else "}"
        out += f'{start}"{key}": "{value}"{end}'
    return out


def qa_map_prompt(question: str, docs: Sequence[Dict]) -> str:
    return _template("qa_map.prompt").format(question=question, search_results=format_documents(docs))


def kv_map_prompt(pairs: Sequence[Sequence[str]], key: str) -> str:
    return _template("kv_map.prompt").format(formatted_kv_records=format_kv_records(pairs), key=key)


def map_prompt(item: Dict) -> str:
    if item["task"].startswith("kv"):
        return kv_map_prompt(item["docs"], item["query_key"])
    return qa_map_prompt(item["question"], item["docs"])


def check_prompt(question: str, evidence: str, answer: str) -> str:
    return _template("qa_check.prompt").format(question=question, evidence=evidence, answer=answer)


def judge_prompt(question: str, candidates: List[Tuple[str, str]]) -> str:
    lines = "\n".join(f"[{i + 1}] Answer: {a}\n    Sentence: {e}" for i, (a, e) in enumerate(candidates))
    return _template("qa_judge.prompt").format(question=question, candidates=lines)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_mr_prompts.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add litm/mapreduce/templates litm/mapreduce/prompts.py tests/test_mr_prompts.py
git commit -m "feat: map, yes/no check and judge prompt templates

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Parse model outputs

**Files:**
- Create: `litm/mapreduce/parse.py`
- Test: `tests/test_mr_parse.py`

**Interfaces:**
- Produces:
  - `Parsed(status: "answer" | "not_found" | "invalid", answer="", evidence="", hedged=False, reason="")`, a frozen dataclass
  - `broken_reason(text, token_logprobs=()) -> str` (`""` if fine, else `"bad logprob"` / `"only !"`)
  - `parse_map_output(text, token_logprobs=(), label="answer") -> Parsed` (use `label="value"` for key-value)
  - `answer_logprob(token_texts, token_logprobs, label="answer") -> float | None`
  - `p_yes(top: iterable of (token, logprob)) -> (float, bool)`
  - `parse_judge(text, n_candidates) -> int | None` (0-based)

- [ ] **Step 1: Write the failing test** at `tests/test_mr_parse.py`:

```python
import math

import pytest

from litm.mapreduce.parse import Parsed, answer_logprob, broken_reason, p_yes, parse_judge, parse_map_output


def test_plain_output():
    parsed = parse_map_output("Answer: Wilhelm Röntgen\nEvidence: Röntgen received the first prize.")
    assert parsed == Parsed("answer", "Wilhelm Röntgen", "Röntgen received the first prize.", False, "")


def test_markdown_and_case_variants():
    for text in [
        "**Answer:** Paris\n**Evidence:** Paris is the capital of France.",
        "answer : Paris\nevidence: Paris is the capital of France.",
        "**Answer**: Paris\nEvidence: Paris is the capital of France.",
    ]:
        parsed = parse_map_output(text)
        assert (parsed.status, parsed.answer) == ("answer", "Paris")
        assert parsed.evidence.endswith("capital of France.")


def test_evidence_may_span_lines():
    parsed = parse_map_output("Answer: 1901\nEvidence: It was awarded\nin 1901 to Röntgen.")
    assert parsed.evidence == "It was awarded\nin 1901 to Röntgen."


def test_not_found_variants():
    for text in ["Answer: NOT FOUND", "Answer: Not found.\nEvidence: none", "Answer: **NOT FOUND**"]:
        assert parse_map_output(text).status == "not_found"


def test_think_block_is_skipped():
    assert parse_map_output("<think>\n\n</think>\n\nAnswer: Paris\nEvidence: x").answer == "Paris"


def test_invalid_outputs():
    assert parse_map_output("Paris is the answer.") == Parsed("invalid", reason="no answer line")
    assert parse_map_output("Answer:\nParis").reason == "empty answer"
    assert parse_map_output("!!!!!!!!").reason == "only !"
    assert parse_map_output("Answer: Paris", [-0.1, float("nan")]).reason == "bad logprob"
    assert broken_reason("Answer: Paris", [-0.1, -2.0]) == ""


def test_hedged_answers_are_flagged():
    assert parse_map_output("Answer: 1994 or 1995\nEvidence: e").hedged
    assert parse_map_output("Answer: 1994; 1995\nEvidence: e").hedged
    assert not parse_map_output("Answer: Simon and Garfunkel\nEvidence: e").hedged


def test_value_label_for_key_value_outputs():
    parsed = parse_map_output('Value: 703a7ce5-f17f-4e6d-b895-5836ba5ec71c\nEvidence: "k": "v"', label="value")
    assert parsed.answer == "703a7ce5-f17f-4e6d-b895-5836ba5ec71c" and parsed.evidence == '"k": "v"'


def test_answer_logprob_covers_only_the_answer_tokens():
    texts = ["Answer", ":", " Wil", "helm", "\n", "Evidence", ":", " x"]
    logprobs = [-9.0, -9.0, -1.0, -3.0, -9.0, -9.0, -9.0, -9.0]
    assert answer_logprob(texts, logprobs) == pytest.approx(-2.0)
    assert answer_logprob(["Nothing"], [-1.0]) is None


def test_p_yes():
    value, found = p_yes([(" Yes", math.log(0.6)), ("No", math.log(0.2)), ("yes", math.log(0.1))])
    assert found and value == pytest.approx(0.7 / 0.9)
    assert p_yes([["No", 0.0]]) == (0.0, True)  # lists, as read back from JSON
    assert p_yes([("Maybe", -0.1)]) == (0.5, False)


def test_parse_judge():
    assert parse_judge("Choice: 2", 3) == 1
    assert parse_judge("**Choice:** [1]", 3) == 0
    assert parse_judge("Choice: 7", 3) is None
    assert parse_judge("I think both", 3) is None
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_mr_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litm.mapreduce.parse'`

- [ ] **Step 3: Implement** `litm/mapreduce/parse.py`:

```python
"""Read map, yes/no and judge outputs (spec §4.2)."""
import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from litm.vendor.lost_in_the_middle.metrics import normalize_answer

HEDGE_RE = re.compile(r"\bor\b|;", re.I)
EVIDENCE_RE = re.compile(r"^[ \t*_#>-]*evidence[ \t]*[*_]*[ \t]*:[ \t]*[*_]*[ \t]*(.*)", re.I | re.M | re.S)
JUDGE_RE = re.compile(r"choice[ \t]*[*_]*[ \t]*:[ \t]*[*_\[( \t]*(\d+)", re.I)


def _field(label: str) -> "re.Pattern":
    """A line like 'Answer: X', '**Answer:** X' or 'answer : X' (any case); group 1 is X."""
    return re.compile(r"^[ \t*_#>-]*" + label + r"[ \t]*[*_]*[ \t]*:[ \t]*[*_]*[ \t]*(.*?)[ \t*_]*$", re.I | re.M)


@dataclass(frozen=True)
class Parsed:
    status: str  # "answer" | "not_found" | "invalid"
    answer: str = ""
    evidence: str = ""
    hedged: bool = False
    reason: str = ""


def broken_reason(text: str, token_logprobs: Sequence[float] = ()) -> str:
    """Signs of fp16 overflow: a NaN/inf logprob, or output that is only '!' characters."""
    if any(math.isnan(x) or math.isinf(x) for x in token_logprobs):
        return "bad logprob"
    if re.fullmatch(r"\s*!+\s*", text):
        return "only !"
    return ""


def _body_start(text: str) -> int:
    """Skip any <think>...</think> block."""
    end = text.rfind("</think>")
    return 0 if end < 0 else end + len("</think>")


def parse_map_output(text: str, token_logprobs: Sequence[float] = (), label: str = "answer") -> Parsed:
    reason = broken_reason(text, token_logprobs)
    if reason:
        return Parsed("invalid", reason=reason)
    body = text[_body_start(text):]
    match = _field(label).search(body)
    if not match:
        return Parsed("invalid", reason=f"no {label} line")
    answer = match.group(1).strip()
    if not answer:
        return Parsed("invalid", reason="empty answer")
    if normalize_answer(answer).startswith("not found"):
        return Parsed("not_found")
    evidence = EVIDENCE_RE.search(body, match.end())
    return Parsed("answer", answer, evidence.group(1).strip() if evidence else "", bool(HEDGE_RE.search(answer)))


def answer_logprob(token_texts: Sequence[str], token_logprobs: Sequence[float], label: str = "answer") -> Optional[float]:
    """Mean logprob of the tokens that overlap the answer text (sensitivity tie-break only)."""
    text = "".join(token_texts)
    match = _field(label).search(text, _body_start(text))
    if not match or match.start(1) == match.end(1):
        return None
    start, end, pos, picked = match.start(1), match.end(1), 0, []
    for piece, logprob in zip(token_texts, token_logprobs):
        if pos < end and pos + len(piece) > start:
            picked.append(logprob)
        pos += len(piece)
    return sum(picked) / len(picked) if picked else None


def p_yes(top: Iterable[Tuple[str, float]]) -> Tuple[float, bool]:
    """P(Yes) / (P(Yes) + P(No)) from the first token's top logprobs; (0.5, False) if neither shows up."""
    yes = no = 0.0
    for token, logprob in top:
        word = token.strip().lower()
        if word == "yes":
            yes += math.exp(logprob)
        elif word == "no":
            no += math.exp(logprob)
    if yes + no == 0:
        return 0.5, False
    return yes / (yes + no), True


def parse_judge(text: str, n_candidates: int) -> Optional[int]:
    """0-based index of the judge's choice, or None if missing or out of range."""
    match = JUDGE_RE.search(text)
    if not match:
        return None
    choice = int(match.group(1)) - 1
    return choice if 0 <= choice < n_candidates else None
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_mr_parse.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add litm/mapreduce/parse.py tests/test_mr_parse.py
git commit -m "feat: parse map, yes/no and judge outputs; flag broken fp16 outputs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Verify evidence

**Files:**
- Modify: `requirements.txt` (add `rapidfuzz` after `xopen`)
- Create: `litm/mapreduce/verify.py`
- Test: `tests/test_mr_verify.py`

**Interfaces:**
- Produces:
  - `MIN_EVIDENCE_WORDS = 5`, `FUZZY_THRESHOLD = 90`, `MIN_COVERAGE = 0.9`
  - `Verdict(verified: bool, how: str)`, a frozen dataclass. `how` is `"exact"`, `"fuzzy"`, or the rejection reason: `"empty answer"`, `"short evidence"`, `"answer not in evidence"`, `"evidence not in documents"`, `"value not a UUID"`, `"no key-value pair in evidence"`, `"evidence key is not the query key"`, `"value differs from evidence"`, `"pair not in group"`
  - `verify_qa(answer, evidence, docs, fuzzy_threshold=FUZZY_THRESHOLD) -> Verdict`
  - `verify_kv(value, evidence, pairs, query_key) -> Verdict`

- [ ] **Step 1: Install the dependency.** Add the line `rapidfuzz` after `xopen` in `requirements.txt`, then run:

Run: `.venv/Scripts/python -m pip install rapidfuzz`
Expected: `Successfully installed rapidfuzz-3.x` (tested with 3.14.6)

- [ ] **Step 2: Write the failing test** at `tests/test_mr_verify.py`:

```python
from litm.mapreduce.verify import Verdict, verify_kv, verify_qa

DOC = {
    "title": "Nobel Prize in Physics",
    "text": "The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen, of Germany, "
            "who received 150,782 SEK. John Bardeen is the only laureate to win the prize twice.",
}
QUOTE = "The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen, of Germany"


def test_exact_quote_is_verified():
    assert verify_qa("Wilhelm Conrad Röntgen", QUOTE, [DOC]) == Verdict(True, "exact")


def test_the_title_counts_as_document_text():
    assert verify_qa("Physics", "Nobel Prize in Physics The first Nobel Prize", [DOC]) == Verdict(True, "exact")


def test_evidence_across_a_newline_in_the_document():
    doc = {"title": "T", "text": "He was born in Ulm.\nHe died in Princeton in 1955 at the age of 76."}
    assert verify_qa("1955", "born in Ulm. He died in Princeton in 1955", [doc]).verified


def test_rejections():
    assert verify_qa("", QUOTE, [DOC]).how == "empty answer"
    assert verify_qa("*", QUOTE, [DOC]).how == "empty answer"
    assert verify_qa("1901", "awarded in 1901", [DOC]).how == "short evidence"
    assert verify_qa("Einstein", QUOTE, [DOC]).how == "answer not in evidence"
    assert verify_qa("one", "none of the first Nobel Prize winners", [DOC]).how == "answer not in evidence"
    invented = "The first Nobel Prize in Physics was awarded in 1901 to Albert Einstein of Switzerland"
    assert verify_qa("Einstein", invented, [DOC]).how == "evidence not in documents"


def test_fuzzy_match_forgives_small_copy_slips():
    slip = "The first Nobel prize in physics was given in 1901 to Wilhelm Conrad Röntgen, of Germany"
    assert verify_qa("1901", slip, [DOC]) == Verdict(True, "fuzzy")


def test_fuzzy_match_needs_the_answer_in_the_document_not_just_the_quote():
    respelled = "The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Rontgen, of Germany"
    assert verify_qa("Rontgen", respelled, [DOC]).how == "evidence not in documents"


def test_fuzzy_match_rejects_real_text_padded_with_invented_text():
    padded = DOC["text"] + " He also invented the telephone and the radio in 1950."
    assert verify_qa("1950", padded, [DOC]).how == "evidence not in documents"


K1, V1 = "2a8d601d-1d69-4e64-9f90-8ad825a74195", "bb3ba2a5-7de8-434b-a86e-a88bb9fa7289"
K2, V2 = "a54e2eed-e625-4570-9f74-3624e77d6684", "d1ff29be-4e2a-4208-a182-0cea716be3d4"
PAIRS = [[K1, V1], [K2, V2]]


def test_kv_verified():
    assert verify_kv(V1, f'"{K1}": "{V1}"', PAIRS, K1) == Verdict(True, "exact")
    assert verify_kv(f'"{V1.upper()}"', f'{K1}: {V1}', PAIRS, K1).verified


def test_kv_rejections():
    assert verify_kv("NOT A UUID", f'"{K1}": "{V1}"', PAIRS, K1).how == "value not a UUID"
    assert verify_kv(V1, "the value is above", PAIRS, K1).how == "no key-value pair in evidence"
    assert verify_kv(V2, f'"{K2}": "{V2}"', PAIRS, K1).how == "evidence key is not the query key"
    assert verify_kv(V2, f'"{K1}": "{V1}"', PAIRS, K1).how == "value differs from evidence"
    assert verify_kv(V1, f'"{K1}": "{V1}"', [[K2, V2]], K1).how == "pair not in group"
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_mr_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litm.mapreduce.verify'`

- [ ] **Step 4: Implement** `litm/mapreduce/verify.py`:

```python
"""Keep only answers grounded in the group's own documents (spec §4.3, §9)."""
import re
from dataclasses import dataclass
from typing import Dict, Sequence

from rapidfuzz import fuzz

from litm.vendor.lost_in_the_middle.metrics import normalize_answer

MIN_EVIDENCE_WORDS = 5
FUZZY_THRESHOLD = 90  # rapidfuzz partial_ratio; frozen after the pilot (spec §8)
MIN_COVERAGE = 0.9    # the fuzzy match must cover 90% of the evidence, so real text + invented text fails

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
UUID_RE = re.compile(UUID, re.I)
PAIR_RE = re.compile(r'"?\s*(' + UUID + r')\s*"?\s*:\s*"?\s*(' + UUID + r')', re.I)


@dataclass(frozen=True)
class Verdict:
    verified: bool
    how: str  # "exact" | "fuzzy" | why it was rejected


def _has_words(needle: str, haystack: str) -> bool:
    """Whole-word containment on normalized text ("one" is not inside "none")."""
    return f" {needle} " in f" {haystack} "


def _widen(text: str, start: int, end: int) -> str:
    """Grow [start, end) to whole words."""
    start = text.rfind(" ", 0, start) + 1
    stop = text.find(" ", end)
    return text[start:] if stop < 0 else text[start:stop]


def verify_qa(answer: str, evidence: str, docs: Sequence[Dict], fuzzy_threshold: float = FUZZY_THRESHOLD) -> Verdict:
    a, e = normalize_answer(answer), normalize_answer(evidence)
    if not a:
        return Verdict(False, "empty answer")
    if len(e.split()) < MIN_EVIDENCE_WORDS:
        return Verdict(False, "short evidence")
    if not _has_words(a, e):
        return Verdict(False, "answer not in evidence")
    texts = [normalize_answer(f"{d['title']} {d['text']}") for d in docs]
    if any(_has_words(e, t) for t in texts):
        return Verdict(True, "exact")
    for t in texts:
        match = fuzz.partial_ratio_alignment(e, t)
        if match is None:
            continue
        coverage = (match.src_end - match.src_start) / len(e)
        if (match.score >= fuzzy_threshold and coverage >= MIN_COVERAGE
                and _has_words(a, _widen(t, match.dest_start, match.dest_end))):
            return Verdict(True, "fuzzy")
    return Verdict(False, "evidence not in documents")


def verify_kv(value: str, evidence: str, pairs: Sequence[Sequence[str]], query_key: str) -> Verdict:
    value = value.strip().strip('"').lower()
    if not UUID_RE.fullmatch(value):
        return Verdict(False, "value not a UUID")
    match = PAIR_RE.search(evidence)
    if not match:
        return Verdict(False, "no key-value pair in evidence")
    key, evidence_value = match.group(1).lower(), match.group(2).lower()
    if key != query_key.lower():
        return Verdict(False, "evidence key is not the query key")
    if evidence_value != value:
        return Verdict(False, "value differs from evidence")
    if (key, value) not in {(k.lower(), v.lower()) for k, v in pairs}:
        return Verdict(False, "pair not in group")
    return Verdict(True, "exact")
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_mr_verify.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt litm/mapreduce/verify.py tests/test_mr_verify.py
git commit -m "feat: verify map answers against the group's documents

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Choose one answer

**Files:**
- Create: `litm/mapreduce/choose.py`
- Test: `tests/test_mr_choose.py`

**Interfaces:**
- Produces:
  - `Candidate(group, answer, evidence, verified, hedged, p_yes=None, logprob=None)`, a frozen dataclass
  - `Decision(answer, path, abstained)`, a frozen dataclass; `ABSTAIN = Decision("", "abstain", True)`
  - `choose_primary(cands, use_fallback=True) -> Decision`; raises `ValueError` if a usable candidate has no `p_yes`
  - `choose_vote(cands) -> Decision`
  - `distinct_verified(cands) -> list[str]` (sorted, normalized)
  - `choose_judge(cands, judged: str | None) -> Decision`
  - `choose_kv(cands) -> Decision`

- [ ] **Step 1: Write the failing test** at `tests/test_mr_choose.py`:

```python
import pytest

from litm.mapreduce.choose import (
    ABSTAIN,
    Candidate,
    Decision,
    choose_judge,
    choose_kv,
    choose_primary,
    choose_vote,
    distinct_verified,
)


def cand(group, answer, verified=True, p=0.5, logprob=-1.0, hedged=False):
    return Candidate(group, answer, f"evidence for {answer}", verified, hedged, p, logprob)


def test_primary_prefers_verified_answers_by_p_yes():
    cands = [cand(0, "Curie", verified=False, p=0.9), cand(1, "Röntgen", p=0.6), cand(2, "Einstein", p=0.4)]
    assert choose_primary(cands) == Decision("Röntgen", "verified", False)


def test_primary_scores_an_answer_by_its_best_supporting_candidate():
    cands = [cand(0, "Einstein", p=0.7), cand(1, "the Röntgen", p=0.2), cand(3, "Röntgen", p=0.8)]
    assert choose_primary(cands).answer == "Röntgen"


def test_exact_ties_do_not_depend_on_group_order():
    a, b = cand(0, "beta", p=0.5), cand(4, "alpha", p=0.5)
    assert choose_primary([a, b]).answer == choose_primary([b, a]).answer == "alpha"


def test_fallback_and_abstain():
    cands = [cand(0, "Curie", verified=False, p=0.3), cand(1, "Bohr", verified=False, p=0.6)]
    assert choose_primary(cands) == Decision("Bohr", "fallback", False)
    assert choose_primary(cands, use_fallback=False) == ABSTAIN
    assert choose_primary([]) == ABSTAIN


def test_hedged_candidates_are_ignored():
    cands = [cand(0, "1994 or 1995", p=0.99, hedged=True), cand(1, "1994", p=0.1)]
    assert choose_primary(cands).answer == "1994"


def test_missing_yes_no_score_is_an_error():
    with pytest.raises(ValueError, match="yes/no"):
        choose_primary([cand(0, "Curie", p=None)])


def test_vote_counts_first_then_logprob():
    cands = [cand(0, "Curie", logprob=-0.1), cand(1, "Röntgen", logprob=-2.0), cand(2, "Röntgen", logprob=-3.0)]
    assert choose_vote(cands).answer == "Röntgen"
    assert choose_vote([cand(0, "Curie", logprob=-0.1), cand(1, "Röntgen", logprob=-2.0)]).answer == "Curie"
    assert choose_vote([cand(0, "Curie", verified=False, logprob=-0.1)]) == Decision("Curie", "fallback", False)


def test_judge_is_used_only_with_two_or_more_verified_answers():
    cands = [cand(0, "Curie", p=0.9), cand(1, "Röntgen", p=0.1)]
    assert distinct_verified(cands) == ["curie", "röntgen"]
    assert choose_judge(cands, "röntgen") == Decision("Röntgen", "judge", False)
    assert choose_judge(cands, None) == choose_primary(cands)
    one = [cand(0, "Curie", p=0.9)]
    assert choose_judge(one, "curie") == choose_primary(one)


def test_kv_choose():
    assert choose_kv([cand(2, "v-other", verified=False), cand(3, "v1")]) == Decision("v1", "verified", False)
    assert choose_kv([cand(3, "v1", verified=False)]) == ABSTAIN
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_mr_choose.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litm.mapreduce.choose'`

- [ ] **Step 3: Implement** `litm/mapreduce/choose.py`:

```python
"""Choose one final answer from the groups' candidates (spec §4.4, §4.5, §9).

No rule ever breaks a tie by group order: that would rebuild a position bias into the reduce step.
Exact ties go to the alphabetically first normalized answer.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from litm.vendor.lost_in_the_middle.metrics import normalize_answer


@dataclass(frozen=True)
class Candidate:
    group: int
    answer: str
    evidence: str
    verified: bool
    hedged: bool
    p_yes: Optional[float] = None    # from the shared yes/no prompt (primary rule)
    logprob: Optional[float] = None  # mean answer-token logprob (sensitivity rule)


@dataclass(frozen=True)
class Decision:
    answer: str      # "" when abstained
    path: str        # "verified" | "fallback" | "judge" | "abstain" | ...
    abstained: bool


ABSTAIN = Decision("", "abstain", True)


def _usable(cands: Sequence[Candidate]) -> List[Candidate]:
    return [c for c in cands if not c.hedged]


def _pick(cands: Sequence[Candidate], score: Callable) -> str:
    """The normalized answer whose best supporting candidate scores highest; return that
    candidate's surface text."""
    by_answer = defaultdict(list)
    for c in cands:
        by_answer[normalize_answer(c.answer)].append(c)
    best = max(sorted(by_answer), key=lambda norm: max(score(c) for c in by_answer[norm]))
    return max(by_answer[best], key=score).answer


def _p(c: Candidate) -> float:
    if c.p_yes is None:
        raise ValueError(f"candidate {c.answer!r} from group {c.group} has no yes/no score")
    return c.p_yes


def _logprob(c: Candidate) -> float:
    return c.logprob if c.logprob is not None else float("-inf")


def choose_primary(cands: Sequence[Candidate], use_fallback: bool = True) -> Decision:
    """Highest P(Yes) among verified answers; else (fallback) among unverified ones; else abstain."""
    usable = _usable(cands)
    verified = [c for c in usable if c.verified]
    if verified:
        return Decision(_pick(verified, _p), "verified", False)
    if use_fallback and usable:
        return Decision(_pick(usable, _p), "fallback", False)
    return ABSTAIN


def choose_vote(cands: Sequence[Candidate]) -> Decision:
    """The original design: most votes among verified answers, ties by mean answer logprob."""
    usable = _usable(cands)
    verified = [c for c in usable if c.verified]
    if verified:
        votes = Counter(normalize_answer(c.answer) for c in verified)
        return Decision(_pick(verified, lambda c: (votes[normalize_answer(c.answer)], _logprob(c))), "verified", False)
    if usable:
        return Decision(_pick(usable, _logprob), "fallback", False)
    return ABSTAIN


def distinct_verified(cands: Sequence[Candidate]) -> List[str]:
    return sorted({normalize_answer(c.answer) for c in _usable(cands) if c.verified})


def choose_judge(cands: Sequence[Candidate], judged: Optional[str]) -> Decision:
    """The judge's pick (a normalized answer) when there were 2+ verified answers; else the primary rule."""
    if judged is not None and len(distinct_verified(cands)) >= 2:
        for c in _usable(cands):
            if c.verified and normalize_answer(c.answer) == judged:
                return Decision(c.answer, "judge", False)
    return choose_primary(cands)


def choose_kv(cands: Sequence[Candidate]) -> Decision:
    """Keys are unique, so a verified value can only come from the gold group."""
    verified = [c for c in cands if c.verified]
    return Decision(verified[0].answer, "verified", False) if verified else ABSTAIN
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_mr_choose.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add litm/mapreduce/choose.py tests/test_mr_choose.py
git commit -m "feat: order-independent rules for choosing the final answer

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Generator interface

**Files:**
- Create: `litm/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `litm.run.MODELS`, `litm.run.render(tokenizer, prompt)` (existing; importing does not modify `run.py`)
- Produces:
  - `GenParams(max_tokens: int, logprobs: int = 1)`, a frozen dataclass
  - `Generation(text, token_logprobs: list[float], token_texts: list[str], first_top: list[(str, float)], finish_reason)`, a frozen dataclass
  - `to_generation(completion) -> Generation`
  - `VLLMGenerator(model_key, max_model_len, enforce_eager=False, require_t4=True)` with `.generate(prompts: list[str], params: GenParams) -> list[Generation]`; raises `SystemExit` if a prompt is too long or the GPU isn't sm_75

- [ ] **Step 1: Write the failing test** at `tests/test_generate.py`:

```python
"""litm.generate with stand-ins for vLLM, torch and the tokenizer."""
import sys
import types

import pytest

from litm.generate import GenParams, Generation, to_generation


def logprob(value, text):
    return types.SimpleNamespace(logprob=value, decoded_token=text)


def test_to_generation():
    completion = types.SimpleNamespace(
        text="Yes.", token_ids=[7, 9], finish_reason="length",
        logprobs=[{7: logprob(-0.1, "Yes"), 8: logprob(-2.5, "No")}, {9: logprob(-0.5, ".")}],
    )
    assert to_generation(completion) == Generation("Yes.", [-0.1, -0.5], ["Yes", "."], [("Yes", -0.1), ("No", -2.5)], "length")


def test_to_generation_without_logprobs():
    completion = types.SimpleNamespace(text="x", token_ids=[1], finish_reason="stop", logprobs=None)
    assert to_generation(completion) == Generation("x", [], [], [], "stop")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return f"<user>{messages[0]['content']}<assistant>"

    def __call__(self, texts, add_special_tokens):
        return {"input_ids": [list(range(len(t) // 4)) for t in texts]}


class FakeLLM:
    def __init__(self, **kwargs):
        FakeLLM.kwargs = kwargs

    def generate(self, prompts, sampling, use_tqdm):
        FakeLLM.sampling = sampling
        out = types.SimpleNamespace(text="ok", token_ids=[1], finish_reason="stop", logprobs=[{1: logprob(-0.2, "ok")}])
        return [types.SimpleNamespace(outputs=[out]) for _ in prompts]


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = FakeLLM, lambda **kw: kw
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: prompt_token_ids
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda: (7, 5))
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())
    for name, module in {"vllm": vllm, "vllm.inputs": inputs, "torch": torch, "transformers": transformers}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return torch


def test_vllm_generator_is_greedy_fp16_triton(fake_vllm):
    from litm.generate import VLLMGenerator

    gen = VLLMGenerator("qwen2.5-3b", max_model_len=2048)
    out = gen.generate(["hello", "world"], GenParams(max_tokens=150))
    assert [g.text for g in out] == ["ok", "ok"] and out[0].token_logprobs == [-0.2]
    assert FakeLLM.kwargs["dtype"] == "float16" and FakeLLM.kwargs["attention_backend"] == "TRITON_ATTN"
    assert FakeLLM.kwargs["max_model_len"] == 2048
    assert FakeLLM.sampling == {"temperature": 0.0, "max_tokens": 150, "logprobs": 1}


def test_vllm_generator_refuses_long_prompts_and_other_gpus(fake_vllm, monkeypatch):
    from litm.generate import VLLMGenerator

    gen = VLLMGenerator("qwen2.5-3b", max_model_len=40)
    with pytest.raises(SystemExit, match="exceed"):
        gen.generate(["x" * 400], GenParams(max_tokens=10))
    monkeypatch.setattr(fake_vllm.cuda, "get_device_capability", lambda: (8, 0))
    with pytest.raises(SystemExit, match="sm_75"):
        VLLMGenerator("qwen2.5-3b", max_model_len=2048)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litm.generate'`

- [ ] **Step 3: Implement** `litm/generate.py`:

```python
"""One interface for text generation: generate(prompts, params) -> [Generation].

VLLMGenerator runs on a Kaggle T4 now; phase C can add an API-backed class with the same method.
Decoding is always greedy with explicit SamplingParams: Qwen's generation_config.json would
otherwise make vLLM sample.
"""
from dataclasses import dataclass
from typing import List, Tuple

from litm.run import MODELS, render


@dataclass(frozen=True)
class GenParams:
    max_tokens: int
    logprobs: int = 1  # top-k logprobs per generated token; the chosen token is always included


@dataclass(frozen=True)
class Generation:
    text: str
    token_logprobs: List[float]           # logprob of each generated token
    token_texts: List[str]                # decoded text of each generated token
    first_top: List[Tuple[str, float]]    # top-k (token text, logprob) at the first position
    finish_reason: str                    # "stop" or "length"


def to_generation(completion) -> Generation:
    """Plain lists from a vLLM CompletionOutput (logprobs: one {token_id: Logprob} dict per token)."""
    positions = completion.logprobs or []
    chosen = [position[token_id] for token_id, position in zip(completion.token_ids, positions)]
    first_top = [(lp.decoded_token or "", lp.logprob) for lp in positions[0].values()] if positions else []
    return Generation(
        text=completion.text,
        token_logprobs=[lp.logprob for lp in chosen],
        token_texts=[lp.decoded_token or "" for lp in chosen],
        first_top=first_top,
        finish_reason=completion.finish_reason,
    )


class VLLMGenerator:
    def __init__(self, model_key: str, max_model_len: int, enforce_eager: bool = False, require_t4: bool = True):
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        capability = torch.cuda.get_device_capability()
        if require_t4 and tuple(capability) != (7, 5):
            raise SystemExit(f"Expected a T4 (sm_75), got compute capability {capability}")
        model = MODELS[model_key]
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(model["hf_id"], revision=model["revision"])
        self.llm = LLM(
            model=model["hf_id"],
            revision=model["revision"],
            dtype="float16",  # T4 has no bfloat16
            attention_backend="TRITON_ATTN",
            max_model_len=max_model_len,
            gpu_memory_utilization=0.9,
            enforce_eager=enforce_eager,
            seed=0,
        )

    def generate(self, prompts: List[str], params: GenParams) -> List[Generation]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        rendered = [render(self.tokenizer, p) for p in prompts]
        ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        too_long = [len(x) for x in ids if len(x) + params.max_tokens > self.max_model_len]
        if too_long:
            raise SystemExit(f"{len(too_long)} prompts exceed max_model_len {self.max_model_len} "
                             f"(longest {max(too_long)} tokens)")
        # Never pass prompt_logprobs: it disables vLLM's prefix cache.
        sampling = SamplingParams(temperature=0.0, max_tokens=params.max_tokens, logprobs=params.logprobs)
        results = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sampling, use_tqdm=False)
        return [to_generation(r.outputs[0]) for r in results]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_generate.py tests/test_run.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add litm/generate.py tests/test_generate.py
git commit -m "feat: generator interface with greedy fp16 vLLM backend and logprobs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Map-reduce runner (GPU stages and CPU reduce)

**Files:**
- Create: `litm/mapreduce/run.py`
- Test: `tests/test_mr_run.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 2 and 4–9
- Produces:
  - CLI: `python -m litm.mapreduce.run --model M --task qa20|qa30|kv300 --stage map|check|judge|control|reduce --data AUTHORS --out results [--questions pilot|all] [--limit N] [--kv-limit N] [--batch-size 256] [--enforce-eager]`
  - Stage output files `results/<model>__mr-<stage>.jsonl`. Each row: `{key, attempt, text, token_logprobs, token_texts, first_top, finish_reason}`. They are shared across tasks and samples, because the key is `job_key(prompt, params)`.
  - `results/<model>__mr-<task>-final-<sample>.jsonl`, one row per (position, idx, method), with fields `model, task, position, idx, method, answer, abstained, path, correct, strict, answer_words, gold_group, gold_slot, mate_ranks, n_candidates, n_verified`. Methods: `mr`, `mr_nofallback`, `mr_vote`, `mr_judge`, `control`, `gold_group`, `oracle_reduce` (key-value: `mr`, `gold_group`, `oracle_reduce`).
  - `results/<model>__mr-<task>-diag-<sample>.jsonl`, one row per group, with fields `model, task, position, idx, group, is_gold_group, n_answer_bearing, status, reason, hedged, verified, verify_how, finish_reason, attempt, cand_correct, p_yes, logprob`
  - Functions used by tests: `main(argv)`, `load_records(path) -> dict`, `job_key(prompt, params) -> str`, `make_generator(...)` (the monkeypatch point)

- [ ] **Step 1: Write the failing test** at `tests/test_mr_run.py`:

```python
"""litm.mapreduce.run end to end on the authors' data, with a scripted stand-in for the model."""
import json
import math
import re
from pathlib import Path

import pytest

from litm.data import oracle_path, read_jsonl
from litm.generate import GenParams, Generation
from litm.mapreduce import run as mr_run

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "lost-in-the-middle"
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="authors' repo not checked out in third_party/")

DOC_LINE = re.compile(r"^Document \[\d+\]\(Title: (.*?)\) (.*)$", re.M)
KV_LINE = re.compile(r'"([0-9a-f-]{36})": "([0-9a-f-]{36})"')


class FakeGenerator:
    """Behaves like a tidy model: quotes the sentence holding a gold answer, else says NOT FOUND."""

    def __init__(self, answers_by_question):
        self.answers = answers_by_question
        self.calls = 0

    @staticmethod
    def output(text, first_top=()):
        pieces = re.findall(r"\S+|\s+", text)
        return Generation(text, [-0.5] * len(pieces), pieces, list(first_top), "stop")

    def respond(self, prompt):
        if "Proposed answer:" in prompt:
            return self.output("Yes", [("Yes", math.log(0.8)), ("No", math.log(0.2))])
        if "Which candidate answers the question correctly?" in prompt:
            return self.output("Choice: 1")
        key = re.search(r'^Key: "(.*)"$', prompt, re.M)
        if key:
            for k, v in KV_LINE.findall(prompt):
                if k == key.group(1):
                    return self.output(f'Value: {v}\nEvidence: "{k}": "{v}"')
            return self.output("Value: NOT FOUND")
        question = re.search(r"^Question: (.*)$", prompt, re.M).group(1)
        for _, text in DOC_LINE.findall(prompt):
            for answer in self.answers.get(question, []):
                if answer and answer in text:
                    sentence = next((s for s in re.split(r"(?<=[.!?])\s+", text) if answer in s), text)
                    return self.output(f"Answer: {answer}\nEvidence: {sentence}")
        return self.output("Answer: NOT FOUND")

    def generate(self, prompts, params):
        self.calls += 1
        return [self.respond(p) for p in prompts]


@pytest.fixture(scope="module")
def answers():
    return {ex["question"]: ex["answers"] for ex in read_jsonl(oracle_path(ROOT))}


@pytest.fixture
def fake(monkeypatch, answers):
    gen = FakeGenerator(answers)
    monkeypatch.setattr(mr_run, "make_generator", lambda *args, **kwargs: gen)
    return gen


def cli(tmp_path, stage, task="qa20", *extra):
    return ["--model", "qwen2.5-3b", "--task", task, "--stage", stage, "--data", str(ROOT),
            "--out", str(tmp_path), "--questions", "pilot", "--limit", "3", *extra]


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_all(tmp_path, task="qa20", *extra):
    stages = ("map", "reduce") if task.startswith("kv") else ("map", "check", "judge", "control", "reduce")
    for stage in stages:
        mr_run.main(cli(tmp_path, stage, task, *extra))


def test_job_key_changes_with_prompt_and_settings():
    params = GenParams(max_tokens=150)
    assert mr_run.job_key("a", params) == mr_run.job_key("a", GenParams(max_tokens=150))
    assert mr_run.job_key("a", params) != mr_run.job_key("b", params)
    assert mr_run.job_key("a", params) != mr_run.job_key("a", GenParams(max_tokens=200))


def test_map_stage_generates_each_unique_prompt_once_and_resumes(tmp_path, fake):
    mr_run.main(cli(tmp_path, "map"))
    path = tmp_path / "qwen2.5-3b__mr-map.jsonl"
    records = read(path)
    assert len(records) == 3 * 13  # 13 distinct groups per question across the 5 positions
    assert len({r["key"] for r in records}) == len(records)
    assert all(r["attempt"] == 1 and r["finish_reason"] == "stop" for r in records)
    calls = fake.calls
    mr_run.main(cli(tmp_path, "map"))
    assert fake.calls == calls and len(read(path)) == len(records)


def test_resume_after_an_interrupted_run(tmp_path, fake):
    mr_run.main(cli(tmp_path, "map"))
    path = tmp_path / "qwen2.5-3b__mr-map.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:10]), encoding="utf-8")
    mr_run.main(cli(tmp_path, "map"))
    records = read(path)
    assert len(records) == 39 and len({r["key"] for r in records}) == 39


def test_broken_outputs_are_retried_once(tmp_path, fake, monkeypatch):
    real = fake.generate

    def first_call_broken(prompts, params):
        out = real(prompts, params)
        if fake.calls == 1:
            return [Generation("!!!!", [float("nan")], ["!!!!"], [], "length") for _ in out]
        return out

    monkeypatch.setattr(fake, "generate", first_call_broken)
    mr_run.main(cli(tmp_path, "map", "qa20", "--batch-size", "1000"))
    path = tmp_path / "qwen2.5-3b__mr-map.jsonl"
    assert len(read(path)) == 2 * 39
    latest = mr_run.load_records(path)
    assert all(r["attempt"] == 2 and r["text"] != "!!!!" for r in latest.values())


def test_full_qa_pipeline(tmp_path, fake):
    run_all(tmp_path)
    finals = read(tmp_path / "qwen2.5-3b__mr-qa20-final-pilot.jsonl")
    methods = {r["method"] for r in finals}
    assert methods == {"mr", "mr_nofallback", "mr_vote", "mr_judge", "control", "gold_group", "oracle_reduce"}
    assert len(finals) == 3 * 5 * len(methods)
    for r in finals:
        if r["abstained"]:
            assert r["correct"] == 0 and r["answer"] == ""
    by_key = {(r["method"], r["position"], r["idx"]): r for r in finals}
    for (method, position, idx), r in by_key.items():
        if method == "gold_group":
            assert by_key[("oracle_reduce", position, idx)]["correct"] >= r["correct"]
    assert {r["gold_slot"] for r in finals} <= {0, 1, 2, 3}
    diags = read(tmp_path / "qwen2.5-3b__mr-qa20-diag-pilot.jsonl")
    assert len(diags) == 3 * 5 * 5
    assert sum(d["is_gold_group"] for d in diags) == 3 * 5


def test_every_group_saying_not_found_means_abstain(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(fake, "respond", lambda prompt: FakeGenerator.output("Answer: NOT FOUND"))
    run_all(tmp_path)
    finals = [r for r in read(tmp_path / "qwen2.5-3b__mr-qa20-final-pilot.jsonl") if r["method"] == "mr"]
    assert len(finals) == 15
    assert all(r["abstained"] and r["path"] == "abstain" and r["correct"] == 0 for r in finals)


def test_reduce_needs_the_check_stage(tmp_path, fake):
    mr_run.main(cli(tmp_path, "map"))
    with pytest.raises(SystemExit, match="check"):
        mr_run.main(cli(tmp_path, "reduce"))


def test_full_kv_pipeline(tmp_path, fake):
    run_all(tmp_path, "kv300", "--kv-limit", "20")
    finals = read(tmp_path / "qwen2.5-3b__mr-kv300-final-pilot.jsonl")
    assert {r["method"] for r in finals} == {"mr", "gold_group", "oracle_reduce"}
    assert len(finals) == 3 * 5 * 3
    assert all(r["correct"] == 1 for r in finals if r["method"] == "mr")


def test_kv_rejects_qa_only_stages(tmp_path, fake):
    with pytest.raises(SystemExit, match="not used"):
        mr_run.main(cli(tmp_path, "check", "kv300"))
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_mr_run.py -v`
Expected: FAIL with `ImportError: cannot import name 'run' from 'litm.mapreduce'`

- [ ] **Step 3: Implement** `litm/mapreduce/run.py`:

```python
"""Map-reduce runs (spec: docs/superpowers/specs/2026-09-30-map-reduce-fix-design.md).

GPU stages, one model per GPU, in this order (the last three are QA only):
    python -m litm.mapreduce.run --model qwen2.5-3b --task qa20 --stage map --data <authors repo> --out results --questions pilot
    ... --stage check / judge / control
then on CPU:
    ... --stage reduce

Every generated output is saved in <model>__mr-<stage>.jsonl, keyed by a hash of the exact prompt and
generation settings, so identical groups are generated once, reruns resume, and any prompt change
invalidates old outputs automatically. `reduce` writes <model>__mr-<task>-final-<sample>.jsonl (one
row per question, position and method) and <model>__mr-<task>-diag-<sample>.jsonl (one row per group).
"""
import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from litm.data import POSITIONS, kv_path, oracle_path, qa_path, read_jsonl
from litm.generate import GenParams
from litm.mapreduce.choose import (
    ABSTAIN,
    Candidate,
    Decision,
    choose_judge,
    choose_kv,
    choose_primary,
    choose_vote,
    distinct_verified,
)
from litm.mapreduce.parse import answer_logprob, broken_reason, p_yes, parse_judge, parse_map_output
from litm.mapreduce.prompts import check_prompt, judge_prompt, map_prompt, qa_map_prompt
from litm.mapreduce.split import SEED, content_key, kv_map_items, qa_map_items, sample_ids
from litm.mapreduce.verify import verify_kv, verify_qa
from litm.run import MODELS, environment
from litm.scoring import score_final, score_final_kv, strict_em
from litm.vendor.lost_in_the_middle.metrics import normalize_answer

MAP_PARAMS = GenParams(max_tokens=150, logprobs=1)
CHECK_PARAMS = GenParams(max_tokens=1, logprobs=20)
JUDGE_PARAMS = GenParams(max_tokens=10, logprobs=1)
CONTROL_PARAMS = MAP_PARAMS
SHORT_MAX_LEN = 2048    # map, check and judge prompts are under 1.5K tokens
CONTROL_MAX_LEN = 8192  # 20 or 30 documents in one prompt (about 3.1K / 4.7K tokens)
GPU_STAGES = ("map", "check", "judge", "control")
QA_ONLY_STAGES = ("check", "judge", "control")


# ---------------------------------------------------------------- files and resumable generation

def stage_path(args, stage: str) -> Path:
    return args.out / f"{args.model}__mr-{stage}.jsonl"


def result_path(args, kind: str) -> Path:
    return args.out / f"{args.model}__mr-{args.task}-{kind}-{args.questions}.jsonl"


def job_key(prompt: str, params: GenParams) -> str:
    return content_key(prompt, params.max_tokens, params.logprobs)


def load_records(path: Path) -> dict:
    """{key: record}; a later attempt for the same key replaces an earlier one."""
    records = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    old = records.get(record["key"])
                    if old is None or record["attempt"] >= old["attempt"]:
                        records[record["key"]] = record
    return records


def run_prompts(gen, path: Path, prompts, params: GenParams, batch_size: int) -> None:
    """Generate every prompt not yet saved in `path`, appending after each batch.
    A broken output (NaN logprob, only '!') is re-run once, in a later batch."""
    prompts = list(dict.fromkeys(prompts))
    path.touch()  # an empty file still records that the stage ran (reduce checks for it)
    done = load_records(path)
    todo = []
    for prompt in prompts:
        previous = done.get(job_key(prompt, params))
        if previous is None:
            todo.append((prompt, 1))
        elif previous["attempt"] == 1 and broken_reason(previous["text"], previous["token_logprobs"]):
            todo.append((prompt, 2))
    print(f"{path.name}: {len(prompts)} prompts, {len(prompts) - len(todo)} done, {len(todo)} to run", flush=True)
    while todo:
        retry = []
        with open(path, "a", encoding="utf-8") as f:
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                outputs = gen.generate([prompt for prompt, _ in batch], params)
                for (prompt, attempt), g in zip(batch, outputs):
                    record = {"key": job_key(prompt, params), "attempt": attempt, "text": g.text,
                              "token_logprobs": g.token_logprobs, "token_texts": g.token_texts,
                              "first_top": g.first_top, "finish_reason": g.finish_reason}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if attempt == 1 and broken_reason(g.text, g.token_logprobs):
                        retry.append((prompt, 2))
                f.flush()
                print(f"{path.name}: {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
        todo = retry


def make_generator(model: str, max_model_len: int, enforce_eager: bool, meta_path: Path):
    from litm.generate import VLLMGenerator

    gen = VLLMGenerator(model, max_model_len, enforce_eager)
    meta = {"model": model, **MODELS[model], "max_model_len": max_model_len, "environment": environment()}
    meta_path.write_text(json.dumps(meta, indent=2))
    return gen


# ---------------------------------------------------------------- items

def question_ids(args) -> list:
    if args.task.startswith("kv"):
        n_total = len(read_jsonl(kv_path(args.data)))
        if args.kv_limit is not None:
            n_total = min(n_total, args.kv_limit)
    else:
        n_total = len(read_jsonl(oracle_path(args.data)))
    return sample_ids(n_total, args.questions, args.limit)


def map_items(args) -> list:
    ids = question_ids(args)
    if args.task.startswith("kv"):
        return kv_map_items(args.data, POSITIONS[args.task], ids)
    return qa_map_items(args.data, int(args.task[2:]), POSITIONS[args.task], ids)


def question_groups(items) -> dict:
    """{(position, idx): [its group items]}"""
    groups = defaultdict(list)
    for item in items:
        groups[(item["position"], item["idx"])].append(item)
    return dict(sorted(groups.items()))


def control_prompts(args) -> dict:
    """{(position, idx): all documents in one prompt, original order, map prompt wording} (spec §7)."""
    ids, n_docs, prompts = set(question_ids(args)), int(args.task[2:]), {}
    for position in POSITIONS[args.task]:
        for idx, example in enumerate(read_jsonl(qa_path(args.data, position, n_docs))):
            if idx in ids:
                docs = [{"title": c["title"], "text": c["text"]} for c in example["ctxs"]]
                prompts[(position, idx)] = qa_map_prompt(example["question"], docs)
    return prompts


# ---------------------------------------------------------------- parse + verify one question's groups

def collect(groups, maps: dict):
    """-> (candidates, rows) where rows are (item, map record, parsed, verdict) for every group."""
    cands, rows = [], []
    for item in groups:
        kv = item["task"].startswith("kv")
        label = "value" if kv else "answer"
        record = maps.get(job_key(map_prompt(item), MAP_PARAMS))
        if record is None:
            raise SystemExit(f"Missing map output for {item['task']} line {item['idx']} group {item['group']}: "
                             "run --stage map first")
        parsed = parse_map_output(record["text"], record["token_logprobs"], label)
        verdict = None
        if parsed.status == "answer":
            if kv:
                verdict = verify_kv(parsed.answer, parsed.evidence, item["docs"], item["query_key"])
            else:
                verdict = verify_qa(parsed.answer, parsed.evidence, item["docs"])
            cands.append(Candidate(item["group"], parsed.answer, parsed.evidence, verdict.verified, parsed.hedged,
                                   None, answer_logprob(record["token_texts"], record["token_logprobs"], label)))
        rows.append((item, record, parsed, verdict))
    return cands, rows


def lookup_p_yes(checks: dict, question: str, cand: Candidate) -> float:
    record = checks.get(job_key(check_prompt(question, cand.evidence, cand.answer), CHECK_PARAMS))
    if record is None:
        raise SystemExit(f"Missing yes/no output for {cand.answer!r}: run --stage check first")
    return p_yes(record["first_top"])[0]


def judge_prompt_for(question: str, cands, position: int, idx: int):
    """(prompt, shown order of normalized answers), or None with fewer than two verified answers."""
    distinct = distinct_verified(cands)
    if len(distinct) < 2:
        return None
    order = [distinct[i] for i in np.random.default_rng([SEED, position, idx]).permutation(len(distinct))]
    shown = {}
    for c in cands:
        norm = normalize_answer(c.answer)
        if c.verified and not c.hedged and norm not in shown:
            shown[norm] = c
    return judge_prompt(question, [(shown[n].answer, shown[n].evidence) for n in order]), order


# ---------------------------------------------------------------- GPU stages

def stage_map(args, gen) -> None:
    prompts = [map_prompt(item) for item in map_items(args)]
    run_prompts(gen, stage_path(args, "map"), prompts, MAP_PARAMS, args.batch_size)


def stage_check(args, gen) -> None:
    maps, prompts = load_records(stage_path(args, "map")), []
    for groups in question_groups(map_items(args)).values():
        cands, _ = collect(groups, maps)
        prompts += [check_prompt(groups[0]["question"], c.evidence, c.answer) for c in cands if not c.hedged]
    run_prompts(gen, stage_path(args, "check"), prompts, CHECK_PARAMS, args.batch_size)


def stage_judge(args, gen) -> None:
    maps, prompts = load_records(stage_path(args, "map")), []
    for (position, idx), groups in question_groups(map_items(args)).items():
        cands, _ = collect(groups, maps)
        built = judge_prompt_for(groups[0]["question"], cands, position, idx)
        if built is not None:
            prompts.append(built[0])
    run_prompts(gen, stage_path(args, "judge"), prompts, JUDGE_PARAMS, args.batch_size)


def stage_control(args, gen) -> None:
    run_prompts(gen, stage_path(args, "control"), list(control_prompts(args).values()), CONTROL_PARAMS, args.batch_size)


# ---------------------------------------------------------------- reduce (CPU)

def reduce_task(args) -> None:
    kv = args.task.startswith("kv")
    maps = load_records(stage_path(args, "map"))
    checks = {} if kv else load_records(stage_path(args, "check"))
    judge_file = stage_path(args, "judge")
    judges = load_records(judge_file) if not kv and judge_file.exists() else None
    controls = {} if kv else load_records(stage_path(args, "control"))
    control = {} if kv else control_prompts(args)
    finals, diags = [], []

    for (position, idx), groups in question_groups(map_items(args)).items():
        question, answers = groups[0]["question"], groups[0]["answers"]

        def score(decision: Decision) -> int:
            if kv:
                return score_final_kv(decision.answer, decision.abstained, answers[0])
            return score_final(decision.answer, decision.abstained, answers)

        cands, rows = collect(groups, maps)
        if not kv:
            cands = [c if c.hedged else dataclasses.replace(c, p_yes=lookup_p_yes(checks, question, c)) for c in cands]
        gold_item, _, gold_parsed, _ = next(row for row in rows if row[0]["is_gold_group"])
        decisions = {"gold_group": Decision(gold_parsed.answer, "gold_group", False)
                     if gold_parsed.status == "answer" else ABSTAIN}
        if kv:
            decisions["mr"] = choose_kv(cands)
        else:
            decisions["mr"] = choose_primary(cands)
            decisions["mr_nofallback"] = choose_primary(cands, use_fallback=False)
            decisions["mr_vote"] = choose_vote(cands)
            if judges is not None:
                judged, built = None, judge_prompt_for(question, cands, position, idx)
                if built is not None:
                    record = judges.get(job_key(built[0], JUDGE_PARAMS))
                    if record is None:
                        raise SystemExit(f"Missing judge output for {args.task} line {idx}: run --stage judge first")
                    choice = parse_judge(record["text"], len(built[1]))
                    judged = None if choice is None else built[1][choice]
                decisions["mr_judge"] = choose_judge(cands, judged)
            record = controls.get(job_key(control[(position, idx)], CONTROL_PARAMS))
            if record is not None:
                parsed = parse_map_output(record["text"], record["token_logprobs"])
                decisions["control"] = Decision(parsed.answer, "control", False) if parsed.status == "answer" else ABSTAIN

        common = {"model": args.model, "task": args.task, "position": position, "idx": idx,
                  "gold_group": gold_item["group"], "gold_slot": gold_item["gold_slot"],
                  "mate_ranks": gold_item["mate_ranks"], "n_candidates": len(cands),
                  "n_verified": len(distinct_verified(cands))}
        for method, d in decisions.items():
            finals.append({**common, "method": method, "answer": d.answer, "abstained": d.abstained, "path": d.path,
                           "correct": score(d), "strict": 0 if kv else strict_em(d.answer, d.abstained, answers),
                           "answer_words": len(d.answer.split())})
        oracle = int(any(score(Decision(c.answer, "", False)) for c in cands))
        finals.append({**common, "method": "oracle_reduce", "answer": "", "abstained": False, "path": "oracle",
                       "correct": oracle, "strict": 0, "answer_words": 0})

        by_group = {c.group: c for c in cands}
        for item, record, parsed, verdict in rows:
            cand = by_group.get(item["group"])
            diags.append({
                "model": args.model, "task": args.task, "position": position, "idx": idx, "group": item["group"],
                "is_gold_group": item["is_gold_group"], "n_answer_bearing": item["n_answer_bearing"],
                "status": parsed.status, "reason": parsed.reason, "hedged": parsed.hedged,
                "verified": bool(verdict and verdict.verified), "verify_how": verdict.how if verdict else "",
                "finish_reason": record["finish_reason"], "attempt": record["attempt"],
                "cand_correct": score(Decision(parsed.answer, "", False)) if parsed.status == "answer" else 0,
                "p_yes": cand.p_yes if cand else None, "logprob": cand.logprob if cand else None,
            })

    for kind, rows in (("final", finals), ("diag", diags)):
        with open(result_path(args, kind), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"reduce: {len(finals)} final rows, {len(diags)} group rows -> {result_path(args, 'final').name}")


# ---------------------------------------------------------------- CLI

STAGES = {"map": stage_map, "check": stage_check, "judge": stage_judge, "control": stage_control}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--task", choices=sorted(POSITIONS), required=True)
    parser.add_argument("--stage", choices=[*GPU_STAGES, "reduce"], required=True)
    parser.add_argument("--data", type=Path, required=True, help="Path to the authors' repo checkout")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--limit", type=int, default=None, help="Keep only the first N questions of the sample")
    parser.add_argument("--kv-limit", type=int, default=None, help="Key-value: use the first N examples, as the baseline did")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(argv)

    if args.task.startswith("kv") and args.stage in QA_ONLY_STAGES:
        raise SystemExit(f"Stage {args.stage} is not used for {args.task} (spec §9)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "reduce":
        reduce_task(args)
        return
    max_model_len = CONTROL_MAX_LEN if args.stage == "control" else SHORT_MAX_LEN
    gen = make_generator(args.model, max_model_len, args.enforce_eager,
                         args.out / f"{args.model}__mr-{args.stage}.meta.json")
    STAGES[args.stage](args, gen)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests** (about 1 minute, because they read the authors' files)

Run: `.venv/Scripts/python -m pytest tests/test_mr_run.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all PASS (skips only where matplotlib or the data is missing)

- [ ] **Step 6: Commit**

```bash
git add litm/mapreduce/run.py tests/test_mr_run.py
git commit -m "feat: resumable map-reduce runner with map, check, judge, control and reduce stages

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Map-reduce analysis

**Files:**
- Create: `litm/mapreduce/analyze.py`
- Test: `tests/test_mr_analyze.py`

**Interfaces:**
- Consumes:
  - `litm.analyze.load_scores(results)` (baseline)
  - `litm.data.POSITIONS`, `MIDDLES` (Task 1)
  - stats functions (Task 3)
  - the final and diag files (Task 10)
- Produces:
  - CLI: `python -m litm.mapreduce.analyze results --task qa20 --sample pilot --out analysis [--no-plot]`
  - `analyze(results, task, sample) -> {"task", "sample", "positions", "models": {model: {...}}}`
  - `to_markdown(summary) -> str`
  - Per model: `n`, `baseline_verdict`, `gate_open`, `non_inferiority`, `delta_u`, `recovered`, `range`, `curves`, `by_slot`, `by_mate_rank`, `paths`, `diagnostics`, `secondary`, `reference`, `outcome` (`"Works" | "Flatter but worse" | "No effect"`)
  - Files: `analysis/mr_<task>_<sample>.{md,json}` and `analysis/mr_<task>_<model>.png`

- [ ] **Step 1: Write the failing test** at `tests/test_mr_analyze.py`:

```python
import json

import numpy as np
import pytest

from litm.mapreduce.analyze import analyze, main, to_markdown

POSITIONS = [0, 4, 9, 14, 19]
MR_METHODS = {"mr": 0.68, "mr_nofallback": 0.60, "mr_vote": 0.66, "mr_judge": 0.67,
              "control": 0.60, "gold_group": 0.70, "oracle_reduce": 0.80}


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def synthetic_results(tmp_path, base_probs, n=600):
    """Baseline with the given per-position accuracy; map-reduce flat at MR_METHODS rates."""
    rng = np.random.default_rng(0)
    base, finals, diags = [], [], []
    for idx in range(n):
        base.append({"model": "toy", "task": "closedbook", "position": None, "idx": idx,
                     "answers": ["yes"], "output": "yes" if rng.random() < 0.3 else "no"})
        for p, prob in zip(POSITIONS, base_probs):
            ok = rng.random() < prob
            base.append({"model": "toy", "task": "qa20", "position": p, "idx": idx,
                         "answers": ["yes"], "output": "yes" if ok else "no"})
            for method, mprob in MR_METHODS.items():
                correct = int(rng.random() < mprob)
                finals.append({"model": "toy", "task": "qa20", "position": p, "idx": idx, "method": method,
                               "answer": "yes", "abstained": False, "path": "verified", "correct": correct,
                               "strict": correct, "answer_words": 1, "gold_group": p // 4, "gold_slot": idx % 4,
                               "mate_ranks": [p // 4 * 4, p // 4 * 4 + 1, p // 4 * 4 + 2],
                               "n_candidates": 2, "n_verified": 1})
            for g in range(5):
                gold = g == p // 4
                diags.append({"model": "toy", "task": "qa20", "position": p, "idx": idx, "group": g,
                              "is_gold_group": gold, "n_answer_bearing": 0,
                              "status": "answer" if gold else "not_found", "reason": "", "hedged": False,
                              "verified": gold, "verify_how": "exact" if gold else "", "finish_reason": "stop",
                              "attempt": 1, "cand_correct": int(gold and idx % 3 > 0),
                              "p_yes": (0.9 if idx % 3 else 0.2) if gold else None,
                              "logprob": -0.5 if gold else None})
    write(tmp_path / "toy.jsonl", base)
    write(tmp_path / "toy__mr-qa20-final-pilot.jsonl", finals)
    write(tmp_path / "toy__mr-qa20-diag-pilot.jsonl", diags)


def test_a_flat_non_inferior_map_reduce_works(tmp_path):
    synthetic_results(tmp_path, [0.75, 0.55, 0.55, 0.55, 0.70])
    s = analyze(tmp_path, "qa20", "pilot")["models"]["toy"]
    assert s["baseline_verdict"] == "U shape" and s["gate_open"]
    assert s["delta_u"]["diff"] > 0.1 and s["delta_u"]["holm_pass"]
    assert s["non_inferiority"]["diff"] > 0 and s["non_inferiority"]["holm_pass"]
    assert s["outcome"] == "Works"
    assert set(s["curves"]) == {"baseline", *MR_METHODS}
    assert s["diagnostics"]["not_found_gold_free"] == 1.0
    assert s["diagnostics"]["auroc_p_yes"] == 1.0
    assert set(s["by_slot"]) == {"0", "1", "2", "3"}
    assert "closedbook" in s["reference"]
    assert s["secondary"]["baseline"]["answer_words"] == 1.0
    assert set(s["secondary"]) == {"baseline", *MR_METHODS} - {"oracle_reduce"}


def test_a_worse_map_reduce_is_flatter_but_worse(tmp_path):
    synthetic_results(tmp_path, [0.90, 0.72, 0.72, 0.72, 0.85])
    s = analyze(tmp_path, "qa20", "pilot")["models"]["toy"]
    assert s["outcome"] == "Flatter but worse"


def test_a_flat_baseline_closes_the_gate(tmp_path):
    synthetic_results(tmp_path, [0.68] * 5)
    summary = analyze(tmp_path, "qa20", "pilot")
    s = summary["models"]["toy"]
    assert not s["gate_open"]
    assert "Gate closed" in to_markdown(summary)


def test_main_writes_the_report(tmp_path):
    synthetic_results(tmp_path, [0.75, 0.55, 0.55, 0.55, 0.70], n=100)
    out = tmp_path / "analysis"
    main([str(tmp_path), "--task", "qa20", "--sample", "pilot", "--out", str(out), "--no-plot"])
    assert (out / "mr_qa20_pilot.md").read_text(encoding="utf-8").startswith("# Map-reduce vs baseline")
    assert json.loads((out / "mr_qa20_pilot.json").read_text())["models"]["toy"]["outcome"] == "Works"


def test_main_writes_a_plot(tmp_path):
    pytest.importorskip("matplotlib")
    synthetic_results(tmp_path, [0.75, 0.55, 0.55, 0.55, 0.70], n=100)
    out = tmp_path / "analysis"
    main([str(tmp_path), "--task", "qa20", "--sample", "pilot", "--out", str(out)])
    assert (out / "mr_qa20_toy.png").exists()


def test_missing_reduce_output_is_explained(tmp_path):
    with pytest.raises(SystemExit, match="reduce"):
        analyze(tmp_path, "qa20", "pilot")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_mr_analyze.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litm.mapreduce.analyze'`

- [ ] **Step 3: Implement** `litm/mapreduce/analyze.py`. Report text is ASCII only, so printing never fails on a Windows console.

```python
"""Score map-reduce against the baseline (spec §6, §7). Runs on CPU.

    python -m litm.mapreduce.analyze results --task qa20 --sample pilot --out analysis
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from litm.analyze import load_scores
from litm.data import MIDDLES, POSITIONS
from litm.scoring import strict_em
from litm.stats import accuracy_ci, auroc, delta_u, holm, non_inferiority, range_minus_null, recovered_share, verdict

PRIMARY_ALPHA = 0.025  # one-sided, the same as a two-sided 95% CI
METHODS = ["mr", "mr_nofallback", "mr_vote", "mr_judge", "control", "gold_group", "oracle_reduce"]
GATE_OPEN = ("U shape", "Primacy only", "Recency only")


def read_rows(results: Path, pattern: str) -> list:
    rows = []
    for path in sorted(results.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def matrix(by_key: dict, positions, ids) -> np.ndarray:
    return np.array([[by_key[(p, i)]["correct"] for p in positions] for i in ids], dtype=float)


def by_field(rows, key) -> dict:
    groups = defaultdict(list)
    for r in rows:
        k = key(r)
        if k is not None:
            groups[k].append(r["correct"])
    return {str(k): {"acc": float(np.mean(v)), "n": len(v)} for k, v in sorted(groups.items())}


def mate_bin(row):
    """Mean retrieval rank of the gold's group-mates, in bins of 4 (0 = hardest distractors)."""
    return int(np.mean(row["mate_ranks"])) // 4 * 4 if row.get("mate_ranks") else None


def diagnostics(rows) -> dict:
    if not rows:
        return {}
    gold = [r for r in rows if r["is_gold_group"]]
    free = [r for r in rows if not r["is_gold_group"]]

    def share(subset, condition):
        return float(np.mean([bool(condition(r)) for r in subset])) if subset else float("nan")

    def signal(field):
        scored = [r for r in rows if r["status"] == "answer" and not r["hedged"] and r[field] is not None]
        return auroc([r[field] for r in scored], [r["cand_correct"] for r in scored])

    return {
        "not_found_gold_free": share(free, lambda r: r["status"] == "not_found"),
        "not_found_gold": share(gold, lambda r: r["status"] == "not_found"),
        "verified_gold_free": share(free, lambda r: r["verified"]),
        "verifier_false_rejection": share([r for r in gold if r["status"] == "answer" and r["cand_correct"]],
                                          lambda r: not r["verified"]),
        "invalid": share(rows, lambda r: r["status"] == "invalid"),
        "hedged": share(rows, lambda r: r["hedged"]),
        "truncated": share(rows, lambda r: r["finish_reason"] == "length"),
        "retried": share(rows, lambda r: r["attempt"] > 1),
        "auroc_p_yes": signal("p_yes"),
        "auroc_logprob": signal("logprob"),
    }


def secondary(results: Path, model: str, task: str, methods: dict, positions, ids) -> dict:
    """Spec §5: strict exact match and answer length per method. Baseline answers are the
    first output line, as the authors score them; longer answers get free substring credit."""
    wanted = {(p, i) for p in positions for i in ids}
    out = {}
    base_rows = [r for r in read_rows(results, f"{model}.jsonl")
                 if r["task"] == task and (r["position"], r["idx"]) in wanted]
    if base_rows:
        first_lines = [r["output"].strip().split("\n")[0] for r in base_rows]
        out["baseline"] = {"strict": float(np.mean([strict_em(t, False, r["answers"]) for t, r in zip(first_lines, base_rows)])),
                           "answer_words": float(np.mean([len(t.split()) for t in first_lines]))}
    for method in METHODS:
        rows = [methods[method][k] for k in wanted if k in methods.get(method, {})]
        if rows and method != "oracle_reduce":
            out[method] = {"strict": float(np.mean([r["strict"] for r in rows])),
                           "answer_words": float(np.mean([r["answer_words"] for r in rows]))}
    return out


def analyze(results: Path, task: str, sample: str) -> dict:
    positions = POSITIONS[task]
    mid = positions.index(MIDDLES[task])
    base_scores = load_scores(results)
    finals = defaultdict(lambda: defaultdict(dict))
    for r in read_rows(results, f"*__mr-{task}-final-{sample}.jsonl"):
        finals[r["model"]][r["method"]][(r["position"], r["idx"])] = r
    diags = defaultdict(list)
    for r in read_rows(results, f"*__mr-{task}-diag-{sample}.jsonl"):
        diags[r["model"]].append(r)
    if not finals:
        raise SystemExit(f"No *__mr-{task}-final-{sample}.jsonl files in {results}: run --stage reduce first")

    models = {}
    for model, methods in sorted(finals.items()):
        base = base_scores.get(model, {})
        ids = sorted({i for _, i in methods["mr"]})
        ids = [i for i in ids if all((p, i) in methods["mr"] and i in base.get((task, p), {}) for p in positions)]
        if not ids:
            raise SystemExit(f"{model}: no questions have both baseline and map-reduce results for {task}")
        B = np.array([[base[(task, p)][i] for p in positions] for i in ids], dtype=float)
        M = matrix(methods["mr"], positions, ids)
        curves = {"baseline": [accuracy_ci(B[:, j]) for j in range(len(positions))]}
        for method in METHODS:
            if all((p, i) in methods.get(method, {}) for p in positions for i in ids):
                X = matrix(methods[method], positions, ids)
                curves[method] = [accuracy_ci(X[:, j]) for j in range(len(positions))]
        shape = verdict(B[:, 0], B[:, mid], B[:, -1])["shape"]
        mr_rows = [methods["mr"][(p, i)] for p in positions for i in ids]
        models[model] = {
            "n": len(ids),
            "baseline_verdict": shape,
            "gate_open": shape in GATE_OPEN,
            "non_inferiority": non_inferiority(M, B),
            "delta_u": delta_u(B, M),
            "recovered": recovered_share(B, M, mid),
            "range": {"baseline": range_minus_null(B), "mr": range_minus_null(M)},
            "curves": curves,
            "by_slot": by_field(mr_rows, lambda r: r["gold_slot"]),
            "by_mate_rank": by_field(mr_rows, mate_bin),
            "paths": dict(Counter(r["path"] for r in mr_rows)),
            "diagnostics": diagnostics(diags[model]),
            "secondary": secondary(results, model, task, methods, positions, ids),
            "reference": {name: accuracy_ci(list(base[(name, None)].values()))
                          for name in ("closedbook", "oracle") if (name, None) in base},
        }

    names = list(models)
    for test in ("non_inferiority", "delta_u"):
        for name, rejected in zip(names, holm([models[n][test]["p"] for n in names], alpha=PRIMARY_ALPHA)):
            models[name][test]["holm_pass"] = bool(rejected)
    for name in names:
        ni, du = models[name]["non_inferiority"]["holm_pass"], models[name]["delta_u"]["holm_pass"]
        models[name]["outcome"] = "Works" if ni and du else "Flatter but worse" if du else "No effect"
    return {"task": task, "sample": sample, "positions": [p + 1 for p in positions], "models": models}


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def to_markdown(summary: dict) -> str:
    lines = [f"# Map-reduce vs baseline: {summary['task']} ({summary['sample']})", ""]
    for model, s in summary["models"].items():
        ni, du, rec = s["non_inferiority"], s["delta_u"], s["recovered"]
        lines += [f"## {model}: **{s['outcome']}** (n = {s['n']} questions)", ""]
        if not s["gate_open"]:
            lines += [f"> Gate closed: the baseline verdict is *{s['baseline_verdict']}*, so there is no "
                      "position effect for map-reduce to fix (spec section 2). Report the outcome as moot.", ""]
        lines += [
            f"- Baseline verdict: {s['baseline_verdict']}",
            f"- Test 1, non-inferiority (margin {pct(ni['margin'])} pts): MR - baseline = {pct(ni['diff'])} pts "
            f"(95% CI {pct(ni['low'])} to {pct(ni['high'])}), pass after Holm: {ni['holm_pass']}",
            f"- Test 2, flatter: U baseline {pct(du['u_base'])} vs MR {pct(du['u_new'])}; dU = {pct(du['diff'])} pts "
            f"(95% CI {pct(du['low'])} to {pct(du['high'])}), pass after Holm: {du['holm_pass']}",
            f"- Test 3, middle gap recovered: {pct(rec['share'])}% (95% CI {pct(rec['low'])} to {pct(rec['high'])})"
            + (" (unstable: the baseline gap's CI touches 0)" if rec["unstable"] else ""),
            f"- Best-worst range (descriptive): baseline {pct(s['range']['baseline']['range'])} "
            f"(no-effect value {pct(s['range']['baseline']['null_mean'])}), MR {pct(s['range']['mr']['range'])} "
            f"(no-effect value {pct(s['range']['mr']['null_mean'])})",
            "", "| Method | " + " | ".join(f"pos {p}" for p in summary["positions"]) + " |",
            "|---|" + "---|" * len(summary["positions"]),
        ]
        for method, points in s["curves"].items():
            lines.append(f"| {method} | " + " | ".join(pct(a["acc"]) for a in points) + " |")
        for name, a in s["reference"].items():
            lines.append(f"\n{name}: {pct(a['acc'])}")
        lines += ["", "**Accuracy by gold slot inside its group (MR):** "
                  + ", ".join(f"slot {int(k) + 1}: {pct(v['acc'])} (n={v['n']})" for k, v in s["by_slot"].items())]
        if s["by_mate_rank"]:
            lines.append("**By mean rank of the gold's group-mates (0 = hardest):** "
                         + ", ".join(f"{k}+: {pct(v['acc'])} (n={v['n']})" for k, v in s["by_mate_rank"].items()))
        lines.append("**Strict exact match / mean answer words:** " + ", ".join(
            f"{m} {pct(v['strict'])}% / {v['answer_words']:.1f}" for m, v in s["secondary"].items()))
        lines.append("**Decision paths (MR):** " + ", ".join(f"{k} {v}" for k, v in sorted(s["paths"].items())))
        lines += ["", "**Map-stage diagnostics:**"]
        lines += [f"- {k.replace('_', ' ')}: {v:.3f}" if k.startswith("auroc") else f"- {k.replace('_', ' ')}: {pct(v)}%"
                  for k, v in s["diagnostics"].items()]
        lines.append("")
    return "\n".join(lines)


def plot(summary: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = summary["positions"]
    for model, s in summary["models"].items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in ("baseline", "mr", "control"):
            if method in s["curves"]:
                points = s["curves"][method]
                y = [100 * a["acc"] for a in points]
                err = [[100 * (a["acc"] - a["low"]) for a in points], [100 * (a["high"] - a["acc"]) for a in points]]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=method)
        for name, style in (("closedbook", ":"), ("oracle", "--")):
            if name in s["reference"]:
                ax.axhline(100 * s["reference"][name]["acc"], ls=style, color="grey", label=name)
        ax.set_xticks(x)
        ax.set_xlabel("Position of the answer")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model}: {summary['task']}, map-reduce vs one prompt")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"mr_{summary['task']}_{model}.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--task", choices=sorted(POSITIONS), default="qa20")
    parser.add_argument("--sample", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = analyze(args.results, args.task, args.sample)
    stem = f"mr_{args.task}_{args.sample}"
    (args.out / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=float))
    report = to_markdown(summary)
    (args.out / f"{stem}.md").write_text(report, encoding="utf-8")
    print(report)
    if not args.no_plot:
        plot(summary, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests** (about 30 seconds; the plot test skips if matplotlib isn't installed locally)

Run: `.venv/Scripts/python -m pytest tests/test_mr_analyze.py -v`
Expected: 5 PASS, 1 PASS or SKIP (`test_main_writes_a_plot`)

- [ ] **Step 5: Commit**

```bash
git add litm/mapreduce/analyze.py tests/test_mr_analyze.py
git commit -m "feat: map-reduce analysis with non-inferiority, flatness and diagnostics

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Kaggle notebook stages and README

**Files:**
- Modify: `tools/build_notebook.py` (INTRO, CONFIG, GUESSES, SETUP, RUN, INSPECT, ANALYZE, `main` → `build` + `main`)
- Regenerate: `notebooks/litm_kaggle.ipynb`
- Modify: `README.md`
- Test: `tests/test_build_notebook.py`

**Interfaces:**
- Consumes: the CLIs from Tasks 10 and 11
- Produces:
  - `tools/build_notebook.build() -> nbformat.NotebookNode`
  - Notebook stages `qa30`, `mr_map`, `mr_check`, `mr_judge`, `mr_control`, `mr_reduce`, `mr_analyze`
  - Config `MR_TASK` and `MR_SAMPLE`

- [ ] **Step 1: Write the failing test** at `tests/test_build_notebook.py`:

```python
"""The Kaggle notebook embeds every litm file and every code cell is valid Python."""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "tools" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_embeds_the_mapreduce_package_and_compiles():
    nb = load_builder().build()
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    embedded = {c.splitlines()[0].split(" ", 1)[1] for c in code if c.startswith("%%writefile")}
    assert {"litm/generate.py", "litm/mapreduce/run.py", "litm/mapreduce/analyze.py",
            "litm/mapreduce/templates/qa_map.prompt", "litm/mapreduce/templates/qa_check.prompt"} <= embedded
    for source in code:
        if not source.startswith("%%writefile"):
            ast.parse(source)
    everything = "\n".join(code)
    for needle in ('"mr_map"', '"mr_reduce"', '"mr_analyze"', '"qa30"', "rapidfuzz", "MR_GUESSES"):
        assert needle in everything
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_build_notebook.py -v`
Expected: FAIL with `AttributeError: module 'build_notebook' has no attribute 'build'`

- [ ] **Step 3: Edit `tools/build_notebook.py`.** Make each replacement exactly as shown. The cell strings are triple-quoted, so the new code inside them must not contain `"""`.

1. In `INTRO`, after the paragraph ending ``finished prompts are skipped.``, insert a blank line and:

```
**Map-reduce fix (phase A):** only after the baseline shows a position effect (spec §2; `qa30` is
the second setting to try). Set `MR_TASK` and `MR_SAMPLE`, then run `mr_map` → `mr_check` →
`mr_judge` → `mr_control` → `mr_reduce` → `mr_analyze`. Pilot first (`MR_SAMPLE = "pilot"`),
freeze the thresholds, then repeat with `"all"`.
```

2. In `CONFIG`, replace the `STAGE = ...` line with:

```python
STAGE = "smoke"   # smoke | pilot | qa | qa30 | kv | rerun | analyze | mr_map | mr_check | mr_judge | mr_control | mr_reduce | mr_analyze
```

and after the `ENFORCE_EAGER` line add:

```python
MR_TASK = "qa20"       # map-reduce task: qa20 | qa30 | kv300, the first with a baseline position effect (spec §2)
MR_SAMPLE = "pilot"    # "pilot" (200 seeded questions) first; "all" once thresholds are frozen (spec §8)
```

3. In `GUESSES`, after the closing `}` of `GUESSES = {...}`, add:

```python
# Map-reduce (spec §6), also before its first run: Works | Flatter but worse | No effect
MR_GUESSES = {"qwen2.5-3b": "?", "qwen3-4b-2507": "?"}
```

4. In `SETUP`, change `vllm==0.18.1 pydantic regex xopen numpy matplotlib")` to `vllm==0.18.1 pydantic regex xopen numpy matplotlib rapidfuzz")`.

5. In `RUN`, add `"qa30":  ([], "", ["qa30"]),` after the `"qa":` line of `STAGES`. Then replace everything from `MODELS = ["qwen2.5-3b", "qwen3-4b-2507"]  # model i runs on GPU i` to the end of the `RUN` string with:

```python
MR_STAGES = ["mr_map", "mr_check", "mr_judge", "mr_control"]  # GPU; mr_reduce and mr_analyze run on CPU
MODELS = ["qwen2.5-3b", "qwen3-4b-2507"]  # model i runs on GPU i
MR_ARGS = ["--task", MR_TASK, "--questions", MR_SAMPLE, "--data", AUTHORS, "--out", "results"]
if MR_TASK.startswith("kv"):
    MR_ARGS += ["--kv-limit", str(KV_LIMIT)]  # the same examples as the baseline

def launch(commands):
    # Run one (model, command, log file) per GPU in parallel; report progress every 5 minutes.
    procs = []
    for gpu, (model, cmd, log_name) in enumerate(commands):
        if ENFORCE_EAGER:
            cmd = [*cmd, "--enforce-eager"]
        log = open(log_name, "a")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": WORK}
        procs.append((model, log_name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
        print(f"GPU {gpu}: {model} -> {log_name}")
    started = time.time()
    while any(p.poll() is None for _, _, p in procs):
        time.sleep(300)
        print(f"--- {(time.time() - started) / 60:.0f} min")
        for model, log_name, _ in procs:
            lines = open(log_name).read().splitlines()
            print(model, "|", lines[-1] if lines else "")
    for model, log_name, p in procs:
        print(f"{model}: exit code {p.returncode}")
        if p.returncode != 0:
            print("".join(open(log_name).readlines()[-40:]))

if STAGE in STAGES:
    extra, tag, tasks = STAGES[STAGE]
    commands = []
    for model in MODELS:
        cmd = [PY, "-m", "litm.run", "--model", model, "--tasks", *tasks, "--data", AUTHORS,
               "--out", "results", "--max-model-len", str(MAX_MODEL_LEN), *extra]
        if tag:
            cmd += ["--tag", tag]
        commands.append((model, cmd, f"results/{model}{'__' + tag if tag else ''}.{STAGE}.log"))
    launch(commands)
elif STAGE in MR_STAGES:
    stage = STAGE[len("mr_"):]
    launch([(model, [PY, "-m", "litm.mapreduce.run", "--model", model, "--stage", stage, *MR_ARGS],
             f"results/{model}__mr-{stage}.{MR_TASK}-{MR_SAMPLE}.log") for model in MODELS])
elif STAGE == "mr_reduce":
    for model in MODELS:
        sh(" ".join([f"PYTHONPATH={WORK}", PY, "-m", "litm.mapreduce.run", "--model", model, "--stage", "reduce", *MR_ARGS]))
```

6. In `INSPECT`, right after `import json`, add:

```python
if STAGE == "mr_map":  # spec §8: read 20 raw map outputs per model before the full run
    for model in MODELS:
        print("=" * 30, model)
        with open(f"results/{model}__mr-map.jsonl") as f:
            for line, _ in zip(f, range(20)):
                print(repr(json.loads(line)["text"][:300]))
```

7. Replace the body of `ANALYZE` (the two lines after its opening `"""\`) with:

```python
if STAGE == "mr_analyze":
    sh(" ".join([f"PYTHONPATH={WORK}", PY, "-m", "litm.mapreduce.analyze", "results",
                 "--task", MR_TASK, "--sample", MR_SAMPLE, "--out", "analysis"]))
elif not STAGE.startswith("mr_"):
    tag_args = ["--tag", STAGES[STAGE][1]] if STAGE in STAGES and STAGES[STAGE][1] else []
    sh(" ".join([f"PYTHONPATH={WORK}", PY, "-m", "litm.analyze", "results", "--out", "analysis", *tag_args]))
```

8. Rename `def main():` to `def build():`. Replace its tail, from the `nb.metadata["kernelspec"] = …` line through `print(f"wrote {out} …")`, with:

```python
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main():
    nb = build()
    out = ROOT / "notebooks" / "litm_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    print(f"wrote {out} ({len(nb.cells)} cells)")
```

- [ ] **Step 4: Run the test, then rebuild the notebook**

Run: `.venv/Scripts/python -m pytest tests/test_build_notebook.py -v`
Expected: PASS

Run: `.venv/Scripts/python tools/build_notebook.py`
Expected: `wrote …notebooks\litm_kaggle.ipynb (43 cells)` (the count depends on how many files are in `litm/`)

- [ ] **Step 5: Update `README.md`.** Replace the line `- Results: *not yet run*` with:

```markdown
- Map-reduce fix (phase A) design: [`docs/superpowers/specs/2026-09-30-map-reduce-fix-design.md`](docs/superpowers/specs/2026-09-30-map-reduce-fix-design.md); audit behind it: [`reports/Lost in middle map reduce audit.md`](reports/Lost%20in%20middle%20map%20reduce%20audit.md)
- Results: *not yet run*
```

In the Layout block, after the `analyze.py` line, add:

```
  generate.py    # generate(prompts, params) interface; greedy fp16 vLLM backend with logprobs
  mapreduce/     # split -> map prompt -> parse -> verify -> choose; run.py (stages), analyze.py
```

and append this section at the end of the file:

```markdown
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
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all PASS (skips only where matplotlib or the data is missing)

- [ ] **Step 7: Commit**

```bash
git add tools/build_notebook.py notebooks/litm_kaggle.ipynb README.md tests/test_build_notebook.py
git commit -m "feat: Kaggle notebook stages for qa30 and map-reduce; README section

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## After the code: the Kaggle runbook (not agent tasks; the user runs these)

1. **Gate (spec §2).** Run the baseline `qa` stage. If its 20-document verdict isn't U, Primacy or Recency, run `qa30`. If that is also flat, check the key-value verdict. Record which setting qualifies as `MR_TASK`.
2. **Smoke.** Set `MR_SAMPLE = "pilot"` and run `mr_map` once with `MR_LIMIT = 5` (set it back to `None` afterwards). Confirm that outputs start with `Answer:` and that logprobs are present (`token_logprobs` is non-empty). This also confirms that `decoded_token` is populated in vLLM 0.18.1.
3. **Pilot.** Run `mr_map` → `mr_check` → `mr_judge` → `mr_control` → `mr_reduce` → `mr_analyze` on the 200-question pilot. Read the 20 raw outputs that `INSPECT` prints. Check the diagnostics:
   - format compliance of at least 95% (invalid under 5%);
   - truncation: if over 1%, raise `MAP_PARAMS` to 200 tokens;
   - verifier false-rejection;
   - AUROC.
4. **Freeze.** Write the final thresholds and `MR_GUESSES` in the notebook's first cells and don't change them afterwards.
5. **Full run.** Set `MR_SAMPLE = "all"` and run the same six stages. The pilot's map outputs are reused automatically. Plan for about 7–9 GPU-hours per model (spec §11), probably in week 2.
