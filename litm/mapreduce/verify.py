"""Keep only answers grounded in the group's own documents (spec §4.3, §9)."""
import re
from dataclasses import dataclass
from typing import Dict, Sequence

from rapidfuzz import fuzz

from litm.vendor.lost_in_the_middle.metrics import normalize_answer

MIN_EVIDENCE_WORDS = 5
FUZZY_THRESHOLD = 90  # rapidfuzz partial_ratio; frozen after the pilot (spec §8)
MIN_COVERAGE = 0.9    # the fuzzy match must cover 90% of the evidence, so real text + invented text fails

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
UUID_RE = re.compile(UUID, re.I)
PAIR_RE = re.compile(r'"?\s*(' + UUID + r')\s*"?\s*:\s*"?\s*(' + UUID + r')', re.I)


@dataclass(frozen=True)
class Verdict:
    verified: bool
    how: str  # "exact" | "fuzzy" | why it was rejected


def _has_words(needle: str, haystack: str) -> bool:
    """Whole-word containment on normalized text ("one" is not inside "none")."""
    return f" {needle} " in f" {haystack} "


def _widen(text: str, start: int, end: int) -> str:
    """Grow [start, end) to whole words."""
    start = text.rfind(" ", 0, start) + 1
    stop = text.find(" ", end)
    return text[start:] if stop < 0 else text[start:stop]


def verify_qa(answer: str, evidence: str, docs: Sequence[Dict], fuzzy_threshold: float = FUZZY_THRESHOLD) -> Verdict:
    a, e = normalize_answer(answer), normalize_answer(evidence)
    if not a:
        return Verdict(False, "empty answer")
    if len(e.split()) < MIN_EVIDENCE_WORDS:
        return Verdict(False, "short evidence")
    if not _has_words(a, e):
        return Verdict(False, "answer not in evidence")
    texts = [normalize_answer(f"{d['title']} {d['text']}") for d in docs]
    if any(_has_words(e, t) for t in texts):
        return Verdict(True, "exact")
    for t in texts:
        match = fuzz.partial_ratio_alignment(e, t)
        if match is None:
            continue
        coverage = (match.src_end - match.src_start) / len(e)
        if (match.score >= fuzzy_threshold and coverage >= MIN_COVERAGE
                and _has_words(a, _widen(t, match.dest_start, match.dest_end))):
            return Verdict(True, "fuzzy")
    return Verdict(False, "evidence not in documents")


def verify_kv(value: str, evidence: str, pairs: Sequence[Sequence[str]], query_key: str) -> Verdict:
    value = value.strip().strip('"').lower()
    if not UUID_RE.fullmatch(value):
        return Verdict(False, "value not a UUID")
    match = PAIR_RE.search(evidence)
    if not match:
        return Verdict(False, "no key-value pair in evidence")
    key, evidence_value = match.group(1).lower(), match.group(2).lower()
    if key != query_key.lower():
        return Verdict(False, "evidence key is not the query key")
    if evidence_value != value:
        return Verdict(False, "value differs from evidence")
    if (key, value) not in {(k.lower(), v.lower()) for k, v in pairs}:
        return Verdict(False, "pair not in group")
    return Verdict(True, "exact")
