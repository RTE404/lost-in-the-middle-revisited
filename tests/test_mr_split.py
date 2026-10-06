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
    assert group_sizes(30, 6) == [6] * 5
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
def test_qa30_items_use_five_groups_of_six():
    # Uneven groups (6x4 + 2x3) would give the last positions an easier gold group (review issue 2)
    items = qa_map_items(ROOT, 30, [0, 29], [0, 1, 2, 3, 4, 5])
    assert [len(i["docs"]) for i in items if i["position"] == 29 and i["idx"] == 0] == [6] * 5
    assert {i["group"] for i in items if i["is_gold_group"] and i["position"] == 29} == {4}
    assert {i["gold_slot"] for i in items if i["is_gold_group"]} <= set(range(6))


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
