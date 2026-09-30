"""Load the authors' data files and turn them into prompt items.

Every item is identified by (task, position, idx), where idx is the line number in the
authors' file. The files have no ID field, so `check_qa_alignment` confirms that line i
is the same question in every file.
"""
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List

from xopen import xopen

from litm.vendor.lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_kv_retrieval_prompt,
    get_qa_prompt,
)

# Pinned commit of github.com/nelson-liu/lost-in-the-middle (data re-uploaded 2024-01-04).
AUTHORS_COMMIT = "29b8a6d042ce29abccee3db1a73171a107d7e6af"

QA_POSITIONS = [0, 4, 9, 14, 19]  # 0-based; paper positions 1, 5, 10, 15, 20
KV_POSITIONS = [0, 74, 149, 224, 299]  # 0-based; positions 1, 75, 150, 225, 300
QA_MIDDLE = 9
KV_MIDDLE = 149


def read_jsonl(path) -> List[dict]:
    with xopen(str(path)) as f:
        return [json.loads(line) for line in f]


def qa_path(root: Path, gold_index: int) -> Path:
    return root / "qa_data" / "20_total_documents" / f"nq-open-20_total_documents_gold_at_{gold_index}.jsonl.gz"


def oracle_path(root: Path) -> Path:
    return root / "qa_data" / "nq-open-oracle.jsonl.gz"


def kv_path(root: Path, n_keys: int = 300) -> Path:
    return root / "kv_retrieval_data" / f"kv-retrieval-{n_keys}_keys.jsonl.gz"


def check_qa_alignment(root: Path) -> int:
    """Raise if any 20-doc file disagrees with the oracle file on question, answers or gold slot."""
    oracle = read_jsonl(oracle_path(root))
    for gold_index in QA_POSITIONS:
        examples = read_jsonl(qa_path(root, gold_index))
        if len(examples) != len(oracle):
            raise ValueError(f"gold_at_{gold_index}: {len(examples)} examples, oracle has {len(oracle)}")
        for idx, (example, reference) in enumerate(zip(examples, oracle)):
            if example["question"] != reference["question"] or example["answers"] != reference["answers"]:
                raise ValueError(f"gold_at_{gold_index} line {idx} does not match the oracle file")
            gold_slots = [i for i, ctx in enumerate(example["ctxs"]) if ctx.get("isgold")]
            if gold_slots != [gold_index]:
                raise ValueError(f"gold_at_{gold_index} line {idx}: gold document at {gold_slots}")
    return len(oracle)


def qa_items(root: Path, task: str, position=None, query_aware: bool = False) -> Iterator[Dict]:
    """task is 'closedbook', 'oracle' or 'qa20'; position is the 0-based gold index for 'qa20'."""
    path = qa_path(root, position) if task == "qa20" else oracle_path(root)
    for idx, example in enumerate(read_jsonl(path)):
        if task == "closedbook":
            prompt = get_closedbook_qa_prompt(example["question"])
        else:
            documents = [Document.from_dict(ctx) for ctx in example["ctxs"]]
            prompt = get_qa_prompt(
                example["question"],
                documents,
                mention_random_ordering=False,
                query_aware_contextualization=query_aware,
            )
        yield {
            "task": task,
            "position": position,
            "idx": idx,
            "prompt": prompt,
            "answers": example["answers"],
        }


def kv_items(root: Path, position: int, n_keys: int = 300, limit=None, query_aware: bool = False) -> Iterator[Dict]:
    """Move the gold pair to `position`, as the authors' get_kv_responses_* scripts do."""
    for idx, example in enumerate(read_jsonl(kv_path(root, n_keys))):
        if limit is not None and idx >= limit:
            break
        records = deepcopy(example["ordered_kv_records"])
        gold = records.pop(records.index([example["key"], example["value"]]))
        records.insert(position, gold)
        yield {
            "task": f"kv{n_keys}",
            "position": position,
            "idx": idx,
            "prompt": get_kv_retrieval_prompt(records, example["key"], query_aware_contextualization=query_aware),
            "answers": [example["value"]],
        }


def all_items(root: Path, tasks: List[str], kv_limit=None) -> List[Dict]:
    """Build every item for the named tasks: closedbook, oracle, qa20, kv300."""
    items = []
    for task in tasks:
        if task in ("closedbook", "oracle"):
            items.extend(qa_items(root, task))
        elif task == "qa20":
            for position in QA_POSITIONS:
                items.extend(qa_items(root, task, position))
        elif task == "kv300":
            for position in KV_POSITIONS:
                items.extend(kv_items(root, position, 300, limit=kv_limit))
        else:
            raise ValueError(f"Unknown task: {task}")
    return items
