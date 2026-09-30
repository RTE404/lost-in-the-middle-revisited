import math

import pytest

from litm.mapreduce.parse import Parsed, answer_logprob, broken_reason, p_yes, parse_judge, parse_map_output


def test_plain_output():
    parsed = parse_map_output("Answer: Wilhelm Röntgen\nEvidence: Röntgen received the first prize.")
    assert parsed == Parsed("answer", "Wilhelm Röntgen", "Röntgen received the first prize.", False, "")


def test_markdown_and_case_variants():
    for text in [
        "**Answer:** Paris\n**Evidence:** Paris is the capital of France.",
        "answer : Paris\nevidence: Paris is the capital of France.",
        "**Answer**: Paris\nEvidence: Paris is the capital of France.",
    ]:
        parsed = parse_map_output(text)
        assert (parsed.status, parsed.answer) == ("answer", "Paris")
        assert parsed.evidence.endswith("capital of France.")


def test_evidence_may_span_lines():
    parsed = parse_map_output("Answer: 1901\nEvidence: It was awarded\nin 1901 to Röntgen.")
    assert parsed.evidence == "It was awarded\nin 1901 to Röntgen."


def test_not_found_variants():
    for text in ["Answer: NOT FOUND", "Answer: Not found.\nEvidence: none", "Answer: **NOT FOUND**"]:
        assert parse_map_output(text).status == "not_found"


def test_think_block_is_skipped():
    assert parse_map_output("<think>\n\n</think>\n\nAnswer: Paris\nEvidence: x").answer == "Paris"


def test_invalid_outputs():
    assert parse_map_output("Paris is the answer.") == Parsed("invalid", reason="no answer line")
    assert parse_map_output("Answer:\nParis").reason == "empty answer"
    assert parse_map_output("!!!!!!!!").reason == "only !"
    assert parse_map_output("Answer: Paris", [-0.1, float("nan")]).reason == "bad logprob"
    assert broken_reason("Answer: Paris", [-0.1, -2.0]) == ""


def test_hedged_answers_are_flagged():
    assert parse_map_output("Answer: 1994 or 1995\nEvidence: e").hedged
    assert parse_map_output("Answer: 1994; 1995\nEvidence: e").hedged
    assert not parse_map_output("Answer: Simon and Garfunkel\nEvidence: e").hedged


def test_value_label_for_key_value_outputs():
    parsed = parse_map_output('Value: 703a7ce5-f17f-4e6d-b895-5836ba5ec71c\nEvidence: "k": "v"', label="value")
    assert parsed.answer == "703a7ce5-f17f-4e6d-b895-5836ba5ec71c" and parsed.evidence == '"k": "v"'


def test_answer_logprob_covers_only_the_answer_tokens():
    texts = ["Answer", ":", " Wil", "helm", "\n", "Evidence", ":", " x"]
    logprobs = [-9.0, -9.0, -1.0, -3.0, -9.0, -9.0, -9.0, -9.0]
    assert answer_logprob(texts, logprobs) == pytest.approx(-2.0)
    assert answer_logprob(["Nothing"], [-1.0]) is None


def test_p_yes():
    value, found = p_yes([(" Yes", math.log(0.6)), ("No", math.log(0.2)), ("yes", math.log(0.1))])
    assert found and value == pytest.approx(0.7 / 0.9)
    assert p_yes([["No", 0.0]]) == (0.0, True)  # lists, as read back from JSON
    assert p_yes([("Maybe", -0.1)]) == (0.5, False)


def test_parse_judge():
    assert parse_judge("Choice: 2", 3) == 1
    assert parse_judge("**Choice:** [1]", 3) == 0
    assert parse_judge("Choice: 7", 3) is None
    assert parse_judge("I think both", 3) is None
