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


def test_empty_files_are_created_without_an_empty_writefile_cell():
    # Jupyter rejects a %%writefile cell with no body ("the cell body is empty"), which stopped the first Kaggle run
    nb = load_builder().build()
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    for source in code:
        if source.startswith("%%writefile"):
            assert source.split("\n", 1)[1:] and source.split("\n", 1)[1].strip(), source.splitlines()[0]
    setup = "\n".join(code)
    for path in ("litm/__init__.py", "litm/mapreduce/__init__.py", "litm/vendor/__init__.py",
                 "litm/vendor/lost_in_the_middle/__init__.py"):
        assert repr(path) in setup


def test_mapreduce_config_needs_no_cell_editing():
    # The runbook's smoke (--limit) and spec §7's quota fallback (--control-subset) are config values,
    # and qa30 has a pre-registered guess slot like qa20 and kv300.
    nb = load_builder().build()
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    namespace = {}
    exec(next(c for c in code if c.startswith("STAGE = ")), namespace)
    exec(next(c for c in code if "GUESSES = {" in c), namespace)
    assert namespace["MR_LIMIT"] is None and namespace["MR_CONTROL_SUBSET"] is None
    assert all("qa30" in g for g in namespace["GUESSES"].values())
    run_cell = next(c for c in code if c.startswith("STAGES = {"))
    for limit, subset, expected in ((None, None, []), (5, None, ["--limit", "5"]),
                                    (None, 500, ["--control-subset", "500"])):
        ns = {**namespace, "MR_LIMIT": limit, "MR_CONTROL_SUBSET": subset, "STAGE": "none",
              "PY": "python", "AUTHORS": "/a", "KV_LIMIT": 200, "MR_TASK": "qa20"}
        exec(run_cell, ns)
        assert ns["MR_ARGS"][-len(expected):] == expected if expected else "--limit" not in ns["MR_ARGS"]
