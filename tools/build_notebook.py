"""Build notebooks/litm_kaggle.ipynb: a self-contained Kaggle notebook.

The litm package (including the vendored authors' code) is embedded as %%writefile cells,
so the notebook needs no GitHub access to our repo. Rebuild after any change to litm/:

    python tools/build_notebook.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]

INTRO = """\
# Lost in the Middle, revisited: free-tier reproduction

Reproduces the position experiment of *Lost in the Middle* (Liu et al., TACL 2024) on
Qwen2.5-3B-Instruct and Qwen3-4B-Instruct-2507, one model per T4. Plan and pre-registered
analysis: see the repository README.

**Kaggle settings:** Accelerator **GPU T4 x2** (not P100), Internet **on**.

**How to run:** set `STAGE` below, then *Save Version → Save & Run All (Commit)*.
Stages, in order: `smoke` → `pilot` → `qa` → `kv` → `rerun`. To resume or continue from an
earlier version, attach that version's output as an input (*Add Input → Your Work*); its
`results/` files are copied in and finished prompts are skipped.
"""

CONFIG = """\
STAGE = "smoke"   # smoke | pilot | qa | kv | rerun | analyze
KV_LIMIT = 200    # key-value examples per position: 200 first; 500 if >5% of answers flip (plan §3)
MAX_MODEL_LEN = 20608  # longest measured prompt 20,458 tokens + 100 new tokens
ENFORCE_EAGER = False  # set True if CUDA-graph capture fails on the T4
"""

GUESSES = """\
# Pre-registered guesses (plan §5): write these BEFORE the first qa/kv run and do not edit afterwards.
GUESSES = {
    "qwen2.5-3b":    {"qa20": "?", "kv300": "?"},   # U shape | Primacy only | Recency only | Flat
    "qwen3-4b-2507": {"qa20": "?", "kv300": "?"},
}
"""

SETUP = """\
import glob, os, shutil, subprocess, sys, time
WORK = "/kaggle/working"
VENV = "/tmp/vllm-env"
AUTHORS = "/tmp/lost-in-the-middle"
AUTHORS_COMMIT = "29b8a6d042ce29abccee3db1a73171a107d7e6af"
os.chdir(WORK)

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

# Isolated environment for vLLM 0.18.1 (it pins its own torch 2.10.0); this kernel never imports torch.
if not os.path.exists(f"{VENV}/bin/python"):
    sh("pip install -q uv")
    sh(f"uv venv -q {VENV} --python 3.12")
    sh(f"uv pip install -q --python {VENV}/bin/python vllm==0.18.1 pydantic regex xopen numpy matplotlib")
PY = f"{VENV}/bin/python"

# The authors' data and code at a pinned commit.
if not os.path.exists(AUTHORS):
    sh(f"git init -q {AUTHORS} && cd {AUTHORS} && git fetch -q --depth 1 "
       f"https://github.com/nelson-liu/lost-in-the-middle.git {AUTHORS_COMMIT} && git checkout -q FETCH_HEAD")

# Resume: copy results from any attached earlier version of this notebook.
os.makedirs("results", exist_ok=True)
for path in glob.glob("/kaggle/input/**/results/*.jsonl", recursive=True):
    target = os.path.join("results", os.path.basename(path))
    if not os.path.exists(target):
        shutil.copy(path, target)
        print("resumed", target)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
"""

RUN = """\
STAGES = {
    # stage: (extra args, output tag, tasks)
    "smoke": (["--qa-limit", "2", "--kv-limit", "1"], "smoke", ["closedbook", "oracle", "qa20", "kv300"]),
    "pilot": (["--qa-limit", "20", "--kv-limit", "4"], "pilot", ["closedbook", "oracle", "qa20", "kv300"]),
    "qa":    ([], "", ["closedbook", "oracle", "qa20"]),
    "kv":    (["--kv-limit", str(KV_LIMIT)], "", ["kv300"]),
    "rerun": (["--qa-limit", "300", "--kv-limit", "50"], "rerun", ["qa20", "kv300"]),
}
MODELS = ["qwen2.5-3b", "qwen3-4b-2507"]  # model i runs on GPU i

if STAGE in STAGES:
    extra, tag, tasks = STAGES[STAGE]
    procs = []
    for gpu, model in enumerate(MODELS):
        cmd = [PY, "-m", "litm.run", "--model", model, "--tasks", *tasks, "--data", AUTHORS,
               "--out", "results", "--max-model-len", str(MAX_MODEL_LEN), *extra]
        if tag:
            cmd += ["--tag", tag]
        if ENFORCE_EAGER:
            cmd.append("--enforce-eager")
        log = open(f"results/{model}{'__' + tag if tag else ''}.{STAGE}.log", "a")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": WORK}
        procs.append((model, log.name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
        print(f"GPU {gpu}: {model} -> {log.name}")
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
"""

INSPECT = """\
# Smoke/pilot check: read the outputs by hand before any full run.
import json
if STAGE in ("smoke", "pilot"):
    for model in MODELS:
        path = f"results/{model}__{STAGE}.jsonl"
        if not os.path.exists(path):
            continue
        print("=" * 30, model)
        for line in open(path):
            r = json.loads(line)
            if STAGE == "smoke" or r["idx"] < 2:
                print(f"{r['task']}@{r['position']} #{r['idx']} ({r['n_prompt_tokens']} tok, {r['finish_reason']}): "
                      f"{r['output'][:120]!r}  gold={r['answers'][:2]}")
"""

ANALYZE = """\
tag_args = ["--tag", STAGES[STAGE][1]] if STAGE in STAGES and STAGES[STAGE][1] else []
sh(" ".join([f"PYTHONPATH={WORK}", PY, "-m", "litm.analyze", "results", "--out", "analysis", *tag_args]))
"""


def writefile_cells():
    files = [p for p in sorted((ROOT / "litm").rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    directories = sorted({p.parent.relative_to(ROOT).as_posix() for p in files})
    cells = [new_code_cell("import os\nfor d in " + repr(directories) + ":\n    os.makedirs(d, exist_ok=True)")]
    for path in files:
        cells.append(new_code_cell(f"%%writefile {path.relative_to(ROOT).as_posix()}\n{path.read_text(encoding='utf-8')}"))
    return cells


def main():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(INTRO),
        new_code_cell(CONFIG),
        new_code_cell(GUESSES),
        new_markdown_cell("## Setup: environment, authors' data, resume"),
        new_code_cell("import os; os.chdir('/kaggle/working')"),
        new_markdown_cell("## Pipeline code (generated from the repo's `litm/` package; edit there, not here)"),
        *writefile_cells(),
        new_code_cell(SETUP),
        new_markdown_cell("## Run the stage (one model per GPU)"),
        new_code_cell(RUN),
        new_code_cell(INSPECT),
        new_markdown_cell("## Score and analyse"),
        new_code_cell(ANALYZE),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks" / "litm_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    print(f"wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
