"""Scoring rules copied from the authors' evaluate_*_responses.py scripts.

One deliberate difference: we strip leading whitespace before taking the first line.
The authors' vLLM script strips generations before scoring, but their evaluate script
splits first, so an answer starting with a newline would score 0.
"""
from typing import List

from litm.vendor.lost_in_the_middle.metrics import best_subspan_em


def score_qa(output: str, answers: List[str]) -> int:
    first_line = output.strip().split("\n")[0].strip()
    return int(best_subspan_em(prediction=first_line, ground_truths=answers))


def score_kv(output: str, value: str) -> int:
    return int(value.lower() in output.lower())


def score(record: dict) -> int:
    if record["task"].startswith("kv"):
        return score_kv(record["output"], record["answers"][0])
    return score_qa(record["output"], record["answers"])
