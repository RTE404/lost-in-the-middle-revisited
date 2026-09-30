from litm.scoring import score_final, score_final_kv, strict_em, strip_answer_prefix

IDX_1451 = ["a rotationally symmetric saltire", "the symbol ⊕", "*"]  # "*" normalizes to ""
IDX_1840 = ["S"]


def test_abstention_scores_zero_before_the_metric():
    assert score_final("", True, ["Paris"]) == 0
    assert score_final("NOT FOUND", True, IDX_1840) == 0  # "S" would match "not found"
    assert score_final("", True, IDX_1451) == 0


def test_idx_1451_matches_any_real_answer_in_both_methods():
    # Documented quirk: kept in both methods, so it cancels in paired comparisons.
    assert score_final("Paris", False, IDX_1451) == 1


def test_only_the_answer_string_is_scored():
    assert score_final("Answer: Wilhelm Röntgen", False, ["Röntgen"]) == 1
    assert score_final("Einstein", False, ["Röntgen"]) == 0
    assert strip_answer_prefix("  answer: Paris ") == "Paris"


def test_strict_em():
    assert strict_em("The Beatles", False, ["Beatles"]) == 1
    assert strict_em("The Beatles and Wings", False, ["Beatles"]) == 0
    assert strict_em("anything", False, ["*"]) == 0
    assert strict_em("Beatles", True, ["Beatles"]) == 0


def test_kv_final():
    value = "703a7ce5-f17f-4e6d-b895-5836ba5ec71c"
    assert score_final_kv(value, False, value) == 1
    assert score_final_kv("", True, value) == 0
