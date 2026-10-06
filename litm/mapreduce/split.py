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
QA_GROUP_SIZES = {20: 4, 30: 6}  # even groups only: 30 as 6x4 + 2x3 would give the last positions an easier gold group
KV_GROUP_SIZE = 20
PILOT_SIZE = 200


def group_sizes(n_items: int, group_size: int) -> List[int]:
    """As even as possible, none above group_size: 20 -> 5x4, 30 -> 5x6, 300 -> 15x20."""
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
    sizes = group_sizes(n_docs, QA_GROUP_SIZES[n_docs])
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
