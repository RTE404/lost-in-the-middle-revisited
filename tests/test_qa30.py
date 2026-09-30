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
