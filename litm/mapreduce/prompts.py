"""Prompt builders for map, yes/no check and judge calls (spec §4).

Document and key-value lines are formatted exactly as the authors' prompting.py does. Templates
are read in text mode, so CRLF checkouts produce the same prompts.
"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

TEMPLATES = Path(__file__).parent / "templates"


def _template(name: str) -> str:
    with open(TEMPLATES / name, encoding="utf-8") as f:
        return f.read().rstrip("\n")


def format_documents(docs: Sequence[Dict]) -> str:
    return "\n".join(f"Document [{i + 1}](Title: {d['title']}) {d['text']}" for i, d in enumerate(docs))


def format_kv_records(pairs: Sequence[Sequence[str]]) -> str:
    """The authors' JSON-ish serialisation, without their check that the query key is present."""
    out = ""
    for index, (key, value) in enumerate(pairs):
        start = "{" if index == 0 else " "
        end = ",\n" if index != len(pairs) - 1 else "}"
        out += f'{start}"{key}": "{value}"{end}'
    return out


def qa_map_prompt(question: str, docs: Sequence[Dict]) -> str:
    return _template("qa_map.prompt").format(question=question, search_results=format_documents(docs))


def kv_map_prompt(pairs: Sequence[Sequence[str]], key: str) -> str:
    return _template("kv_map.prompt").format(formatted_kv_records=format_kv_records(pairs), key=key)


def map_prompt(item: Dict) -> str:
    if item["task"].startswith("kv"):
        return kv_map_prompt(item["docs"], item["query_key"])
    return qa_map_prompt(item["question"], item["docs"])


def check_prompt(question: str, evidence: str, answer: str) -> str:
    return _template("qa_check.prompt").format(question=question, evidence=evidence, answer=answer)


def judge_prompt(question: str, candidates: List[Tuple[str, str]]) -> str:
    lines = "\n".join(f"[{i + 1}] Answer: {a}\n    Sentence: {e}" for i, (a, e) in enumerate(candidates))
    return _template("qa_judge.prompt").format(question=question, candidates=lines)
