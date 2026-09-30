import pytest

from litm.mapreduce.choose import (
    ABSTAIN,
    Candidate,
    Decision,
    choose_judge,
    choose_kv,
    choose_primary,
    choose_vote,
    distinct_verified,
)


def cand(group, answer, verified=True, p=0.5, logprob=-1.0, hedged=False):
    return Candidate(group, answer, f"evidence for {answer}", verified, hedged, p, logprob)


def test_primary_prefers_verified_answers_by_p_yes():
    cands = [cand(0, "Curie", verified=False, p=0.9), cand(1, "Röntgen", p=0.6), cand(2, "Einstein", p=0.4)]
    assert choose_primary(cands) == Decision("Röntgen", "verified", False)


def test_primary_scores_an_answer_by_its_best_supporting_candidate():
    cands = [cand(0, "Einstein", p=0.7), cand(1, "the Röntgen", p=0.2), cand(3, "Röntgen", p=0.8)]
    assert choose_primary(cands).answer == "Röntgen"


def test_exact_ties_do_not_depend_on_group_order():
    a, b = cand(0, "beta", p=0.5), cand(4, "alpha", p=0.5)
    assert choose_primary([a, b]).answer == choose_primary([b, a]).answer == "alpha"


def test_fallback_and_abstain():
    cands = [cand(0, "Curie", verified=False, p=0.3), cand(1, "Bohr", verified=False, p=0.6)]
    assert choose_primary(cands) == Decision("Bohr", "fallback", False)
    assert choose_primary(cands, use_fallback=False) == ABSTAIN
    assert choose_primary([]) == ABSTAIN


def test_hedged_candidates_are_ignored():
    cands = [cand(0, "1994 or 1995", p=0.99, hedged=True), cand(1, "1994", p=0.1)]
    assert choose_primary(cands).answer == "1994"


def test_missing_yes_no_score_is_an_error():
    with pytest.raises(ValueError, match="yes/no"):
        choose_primary([cand(0, "Curie", p=None)])


def test_vote_counts_first_then_logprob():
    cands = [cand(0, "Curie", logprob=-0.1), cand(1, "Röntgen", logprob=-2.0), cand(2, "Röntgen", logprob=-3.0)]
    assert choose_vote(cands).answer == "Röntgen"
    assert choose_vote([cand(0, "Curie", logprob=-0.1), cand(1, "Röntgen", logprob=-2.0)]).answer == "Curie"
    assert choose_vote([cand(0, "Curie", verified=False, logprob=-0.1)]) == Decision("Curie", "fallback", False)


def test_judge_is_used_only_with_two_or_more_verified_answers():
    cands = [cand(0, "Curie", p=0.9), cand(1, "Röntgen", p=0.1)]
    assert distinct_verified(cands) == ["curie", "röntgen"]
    assert choose_judge(cands, "röntgen") == Decision("Röntgen", "judge", False)
    assert choose_judge(cands, None) == choose_primary(cands)
    one = [cand(0, "Curie", p=0.9)]
    assert choose_judge(one, "curie") == choose_primary(one)


def test_kv_choose():
    assert choose_kv([cand(2, "v-other", verified=False), cand(3, "v1")]) == Decision("v1", "verified", False)
    assert choose_kv([cand(3, "v1", verified=False)]) == ABSTAIN
