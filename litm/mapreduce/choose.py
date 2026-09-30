"""Choose one final answer from the groups' candidates (spec §4.4, §4.5, §9).

No rule ever breaks a tie by group order: that would rebuild a position bias into the reduce step.
Exact ties go to the alphabetically first normalized answer.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from litm.vendor.lost_in_the_middle.metrics import normalize_answer


@dataclass(frozen=True)
class Candidate:
    group: int
    answer: str
    evidence: str
    verified: bool
    hedged: bool
    p_yes: Optional[float] = None    # from the shared yes/no prompt (primary rule)
    logprob: Optional[float] = None  # mean answer-token logprob (sensitivity rule)


@dataclass(frozen=True)
class Decision:
    answer: str      # "" when abstained
    path: str        # "verified" | "fallback" | "judge" | "abstain" | ...
    abstained: bool


ABSTAIN = Decision("", "abstain", True)


def _usable(cands: Sequence[Candidate]) -> List[Candidate]:
    return [c for c in cands if not c.hedged]


def _pick(cands: Sequence[Candidate], score: Callable) -> str:
    """The normalized answer whose best supporting candidate scores highest; return that
    candidate's surface text."""
    by_answer = defaultdict(list)
    for c in cands:
        by_answer[normalize_answer(c.answer)].append(c)
    best = max(sorted(by_answer), key=lambda norm: max(score(c) for c in by_answer[norm]))
    return max(by_answer[best], key=score).answer


def _p(c: Candidate) -> float:
    if c.p_yes is None:
        raise ValueError(f"candidate {c.answer!r} from group {c.group} has no yes/no score")
    return c.p_yes


def _logprob(c: Candidate) -> float:
    return c.logprob if c.logprob is not None else float("-inf")


def choose_primary(cands: Sequence[Candidate], use_fallback: bool = True) -> Decision:
    """Highest P(Yes) among verified answers; else (fallback) among unverified ones; else abstain."""
    usable = _usable(cands)
    verified = [c for c in usable if c.verified]
    if verified:
        return Decision(_pick(verified, _p), "verified", False)
    if use_fallback and usable:
        return Decision(_pick(usable, _p), "fallback", False)
    return ABSTAIN


def choose_vote(cands: Sequence[Candidate]) -> Decision:
    """The original design: most votes among verified answers, ties by mean answer logprob."""
    usable = _usable(cands)
    verified = [c for c in usable if c.verified]
    if verified:
        votes = Counter(normalize_answer(c.answer) for c in verified)
        return Decision(_pick(verified, lambda c: (votes[normalize_answer(c.answer)], _logprob(c))), "verified", False)
    if usable:
        return Decision(_pick(usable, _logprob), "fallback", False)
    return ABSTAIN


def distinct_verified(cands: Sequence[Candidate]) -> List[str]:
    return sorted({normalize_answer(c.answer) for c in _usable(cands) if c.verified})


def choose_judge(cands: Sequence[Candidate], judged: Optional[str]) -> Decision:
    """The judge's pick (a normalized answer) when there were 2+ verified answers; else the primary rule."""
    if judged is not None and len(distinct_verified(cands)) >= 2:
        for c in _usable(cands):
            if c.verified and normalize_answer(c.answer) == judged:
                return Decision(c.answer, "judge", False)
    return choose_primary(cands)


def choose_kv(cands: Sequence[Candidate]) -> Decision:
    """Keys are unique, so a verified value can only come from the gold group."""
    verified = [c for c in cands if c.verified]
    return Decision(verified[0].answer, "verified", False) if verified else ABSTAIN
