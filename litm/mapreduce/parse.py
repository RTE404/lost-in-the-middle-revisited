"""Read map, yes/no and judge outputs (spec §4.2)."""
import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from litm.vendor.lost_in_the_middle.metrics import normalize_answer

HEDGE_RE = re.compile(r"\bor\b|;", re.I)
EVIDENCE_RE = re.compile(r"^[ \t*_#>-]*evidence[ \t]*[*_]*[ \t]*:[ \t]*[*_]*[ \t]*(.*)", re.I | re.M | re.S)
JUDGE_RE = re.compile(r"choice[ \t]*[*_]*[ \t]*:[ \t]*[*_\[( \t]*(\d+)", re.I)


def _field(label: str) -> "re.Pattern":
    """A line like 'Answer: X', '**Answer:** X' or 'answer : X' (any case); group 1 is X."""
    return re.compile(r"^[ \t*_#>-]*" + label + r"[ \t]*[*_]*[ \t]*:[ \t]*[*_]*[ \t]*(.*?)[ \t*_]*$", re.I | re.M)


@dataclass(frozen=True)
class Parsed:
    status: str  # "answer" | "not_found" | "invalid"
    answer: str = ""
    evidence: str = ""
    hedged: bool = False
    reason: str = ""


def broken_reason(text: str, token_logprobs: Sequence[float] = ()) -> str:
    """Signs of fp16 overflow: a NaN/inf logprob, or output that is only '!' characters."""
    if any(math.isnan(x) or math.isinf(x) for x in token_logprobs):
        return "bad logprob"
    if re.fullmatch(r"\s*!+\s*", text):
        return "only !"
    return ""


def _body_start(text: str) -> int:
    """Skip any <think>...</think> block."""
    end = text.rfind("</think>")
    return 0 if end < 0 else end + len("</think>")


def parse_map_output(text: str, token_logprobs: Sequence[float] = (), label: str = "answer") -> Parsed:
    reason = broken_reason(text, token_logprobs)
    if reason:
        return Parsed("invalid", reason=reason)
    body = text[_body_start(text):]
    match = _field(label).search(body)
    if not match:
        return Parsed("invalid", reason=f"no {label} line")
    answer = match.group(1).strip()
    if not answer:
        return Parsed("invalid", reason="empty answer")
    if normalize_answer(answer).startswith("not found"):
        return Parsed("not_found")
    evidence = EVIDENCE_RE.search(body, match.end())
    return Parsed("answer", answer, evidence.group(1).strip() if evidence else "", bool(HEDGE_RE.search(answer)))


def answer_logprob(token_texts: Sequence[str], token_logprobs: Sequence[float], label: str = "answer") -> Optional[float]:
    """Mean logprob of the tokens that overlap the answer text (sensitivity tie-break only)."""
    text = "".join(token_texts)
    match = _field(label).search(text, _body_start(text))
    if not match or match.start(1) == match.end(1):
        return None
    start, end, pos, picked = match.start(1), match.end(1), 0, []
    for piece, logprob in zip(token_texts, token_logprobs):
        if pos < end and pos + len(piece) > start:
            picked.append(logprob)
        pos += len(piece)
    return sum(picked) / len(picked) if picked else None


def p_yes(top: Iterable[Tuple[str, float]]) -> Tuple[float, bool]:
    """P(Yes) / (P(Yes) + P(No)) from the first token's top logprobs; (0.5, False) if neither shows up."""
    yes = no = 0.0
    for token, logprob in top:
        if not math.isfinite(logprob):  # a broken fp16 output; NaN would win or lose every comparison
            continue
        word = token.strip().lower()
        if word == "yes":
            yes += math.exp(logprob)
        elif word == "no":
            no += math.exp(logprob)
    if yes + no == 0:
        return 0.5, False
    return yes / (yes + no), True


def parse_judge(text: str, n_candidates: int) -> Optional[int]:
    """0-based index of the judge's choice, or None if missing or out of range."""
    match = JUDGE_RE.search(text)
    if not match:
        return None
    choice = int(match.group(1)) - 1
    return choice if 0 <= choice < n_candidates else None
