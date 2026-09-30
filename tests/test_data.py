import json
from pathlib import Path

import pytest

from litm.analyze import compare_rerun, load_scores, summarize
from litm.data import KV_POSITIONS, QA_POSITIONS, all_items, check_qa_alignment, kv_items, qa_items

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "lost-in-the-middle"
needs_data = pytest.mark.skipif(not ROOT.exists(), reason="authors' repo not checked out in third_party/")


@needs_data
def test_qa_files_align_with_oracle():
    assert check_qa_alignment(ROOT) == 2655


@needs_data
def test_qa_prompt_matches_paper_format():
    item = next(qa_items(ROOT, "qa20", 4))
    prompt = item["prompt"]
    assert prompt.startswith("Write a high-quality answer for the given question using only the provided search results")
    assert prompt.count("Document [") == 20
    assert prompt.rstrip().endswith("Answer:")
    closed = next(qa_items(ROOT, "closedbook"))
    assert closed["prompt"].startswith("Question: ") and "Document [" not in closed["prompt"]
    oracle = next(qa_items(ROOT, "oracle"))
    assert oracle["prompt"].count("Document [") == 1


@needs_data
@pytest.mark.parametrize("position", KV_POSITIONS)
def test_kv_gold_moved_to_position(position):
    item = next(kv_items(ROOT, position, limit=1))
    value = item["answers"][0]
    lines = [line for line in item["prompt"].split("\n") if line.startswith(("{", " "))]
    assert len(lines) == 300
    assert value in lines[position]


@needs_data
def test_item_counts():
    items = all_items(ROOT, ["closedbook", "oracle", "qa20", "kv300"], kv_limit=200)
    counts = {}
    for item in items:
        counts[item["task"]] = counts.get(item["task"], 0) + 1
    assert counts == {"closedbook": 2655, "oracle": 2655, "qa20": 2655 * len(QA_POSITIONS), "kv300": 200 * 5}


def test_analysis_end_to_end_on_synthetic_results(tmp_path):
    records = []
    for idx in range(400):
        for position in QA_POSITIONS:
            correct = position in (0, 19) or idx % 3 == 0  # ends always right, middle mostly wrong
            records.append({"model": "toy", "task": "qa20", "position": position, "idx": idx,
                            "answers": ["yes"], "output": "yes" if correct else "no"})
    with open(tmp_path / "toy.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    with open(tmp_path / "toy__rerun.jsonl", "w") as f:  # tagged files are ignored by default
        f.write(json.dumps({**records[0], "output": "no"}) + "\n")

    summary = summarize(load_scores(tmp_path))
    assert summary["toy"]["verdicts"]["qa20"]["shape"] == "U shape"
    assert summary["toy"]["accuracy"]["qa20@0"]["acc"] == 1.0
    rerun = load_scores(tmp_path, "rerun")
    assert set(rerun["toy"]) == {("qa20", 0)}
    # The one rerun answer flipped from right to wrong: 0% agreement -> fail
    assert compare_rerun(load_scores(tmp_path), rerun).endswith("FAIL")
    assert compare_rerun(load_scores(tmp_path), load_scores(tmp_path)).endswith("PASS")
