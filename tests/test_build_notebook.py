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
