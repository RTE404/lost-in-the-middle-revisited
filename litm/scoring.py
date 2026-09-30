"""Scoring rules copied from the authors' evaluate_*_responses.py scripts.

One deliberate difference: we strip leading whitespace before taking the first line.
The authors' vLLM script strips generations before scoring, but their evaluate script
splits first, so an answer starting with a newline would score 0.
"""
import re
from typing import List

from litm.vendor.lost_in_the_middle.metrics import best_subspan_em, normalize_answer


def score_qa(output: str, answers: List[str]) -> int:
    first_line = output.strip().split("\n")[0].strip()
    return int(best_subspan_em(prediction=first_line, ground_truths=answers))


def score_kv(output: str, value: str) -> int:
    return int(value.lower() in output.lower())


def score(record: dict) -> int:
    if record["task"].startswith("kv"):
        return score_kv(record["output"], record["answers"][0])
    return score_qa(record["output"], record["answers"])


def strip_answer_prefix(answer: str) -> str:
    return re.sub(r"^\s*answer\s*:\s*", "", answer, flags=re.I).strip()


def score_final(answer: str, abstained: bool, answers: List[str]) -> int:
    """Map-reduce final answers (spec §5): an abstention scores 0 before the metric runs, so a
    gold answer like "S" can never match the words NOT FOUND."""
    if abstained or not answer.strip():
        return 0
    return int(best_subspan_em(prediction=strip_answer_prefix(answer), ground_truths=answers))


def strict_em(answer: str, abstained: bool, answers: List[str]) -> int:
    """Secondary metric: the normalized answer equals a normalized gold answer."""
    prediction = normalize_answer(strip_answer_prefix(answer))
    if abstained or not prediction:
        return 0
    return int(any(prediction == normalize_answer(gold) for gold in answers))


def score_final_kv(answer: str, abstained: bool, value: str) -> int:
    return 0 if abstained else score_kv(answer, value)
