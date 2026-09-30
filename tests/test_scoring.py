from litm.scoring import score, score_kv, score_qa
from litm.vendor.lost_in_the_middle.metrics import best_subspan_em


def authors_qa(output, answers):
    """evaluate_qa_responses.py, verbatim logic."""
    return int(best_subspan_em(prediction=output.split("\n")[0].strip(), ground_truths=answers))


def test_qa_matches_authors_on_ordinary_outputs():
    answers = ["Wilhelm Conrad Röntgen", "Röntgen"]
    for output in [
        "Wilhelm Conrad Röntgen",
        "The first Nobel Prize in Physics went to Röntgen.",
        "Albert Einstein",
        "Einstein.\nRöntgen was second",  # only the first line counts
        "",
    ]:
        assert score_qa(output, answers) == authors_qa(output, answers)


def test_qa_strips_leading_newline_unlike_authors_evaluate_script():
    assert score_qa("\nRöntgen", ["Röntgen"]) == 1
    assert authors_qa("\nRöntgen", ["Röntgen"]) == 0


def test_qa_normalises_articles_and_punctuation():
    assert score_qa("It is the Beatles!", ["Beatles"]) == 1


def test_kv_is_case_insensitive_substring_without_first_line_cut():
    value = "703a7ce5-f17f-4e6d-b895-5836ba5ec71c"
    assert score_kv('"703A7CE5-F17F-4E6D-B895-5836BA5EC71C"', value) == 1
    assert score_kv("Sure.\n" + value, value) == 1
    assert score_kv("703a7ce5-f17f", value) == 0


def test_score_dispatches_on_task():
    assert score({"task": "kv300", "output": "abc", "answers": ["ABC"]}) == 1
    assert score({"task": "qa20", "output": "the abc", "answers": ["abc"]}) == 1
