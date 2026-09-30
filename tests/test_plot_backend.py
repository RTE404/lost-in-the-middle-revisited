"""Plots work under Kaggle's notebook kernel, which exports MPLBACKEND=module://matplotlib_inline.backend_inline
to subprocesses; the vLLM environment's matplotlib has no such backend, so importing it used to fail."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_BACKEND = "module://matplotlib_inline.backend_inline"


@pytest.mark.parametrize("module", ["litm.analyze", "litm.mapreduce.analyze"])
def test_plot_ignores_an_inline_backend_from_the_notebook(module, tmp_path):
    pytest.importorskip("matplotlib")
    code = (f"import {module} as m, pathlib\n"
            "summary = {'task': 'qa20', 'positions': [1, 5, 10, 15, 20], 'models': {}}\n"
            "m.plot(summary if m.__name__.endswith('mapreduce.analyze') else {}, pathlib.Path(r'%s'))\n" % tmp_path)
    env = {"MPLBACKEND": KAGGLE_BACKEND, "PYTHONPATH": str(ROOT), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
