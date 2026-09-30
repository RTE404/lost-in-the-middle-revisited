from litm.mapreduce.prompts import (
    check_prompt,
    format_documents,
    format_kv_records,
    judge_prompt,
    kv_map_prompt,
    map_prompt,
    qa_map_prompt,
)
from litm.vendor.lost_in_the_middle.prompting import Document, get_kv_retrieval_prompt, get_qa_prompt

DOCS = [{"title": "T1", "text": "Alpha text."}, {"title": "T2", "text": "Beta text."}]


def test_document_lines_match_the_authors_byte_for_byte():
    authors = get_qa_prompt("q?", [Document(title=d["title"], text=d["text"]) for d in DOCS], False, False)
    assert format_documents(DOCS) in authors


def test_qa_map_prompt():
    prompt = qa_map_prompt("who?", DOCS)
    assert prompt.startswith("Write a high-quality answer for the given question using only the provided search results")
    assert "If none of the search results contain the answer, write NOT FOUND." in prompt
    assert "Document [1](Title: T1) Alpha text.\nDocument [2](Title: T2) Beta text." in prompt
    assert prompt.index("Question: who?") < prompt.index("Reply in exactly this format:")
    assert prompt.endswith("Evidence: <copy the sentence from the documents that contains the answer>")
    assert "\r" not in prompt


def test_kv_formatter_matches_the_authors_and_allows_a_missing_key():
    pairs = [["k1", "v1"], ["k2", "v2"]]
    assert format_kv_records(pairs) in get_kv_retrieval_prompt(pairs, "k1")
    prompt = kv_map_prompt(pairs, "absent-key")  # the authors' builder raises ValueError here
    assert 'Key: "absent-key"' in prompt and "NOT FOUND" in prompt
    assert prompt.endswith('Evidence: <copy the "key": "value" pair from the JSON object>')


def test_map_prompt_dispatches_on_task():
    assert map_prompt({"task": "qa20", "question": "who?", "docs": DOCS}) == qa_map_prompt("who?", DOCS)
    kv = {"task": "kv300", "query_key": "k1", "docs": [["k1", "v1"], ["k2", "v2"]]}
    assert map_prompt(kv) == kv_map_prompt(kv["docs"], "k1")


def test_check_and_judge_prompts():
    check = check_prompt("who won?", "Röntgen won in 1901.", "Röntgen")
    assert check == (
        "Question: who won?\nSentence: Röntgen won in 1901.\nProposed answer: Röntgen\n"
        "Does the sentence show that the proposed answer is correct? Answer Yes or No."
    )
    judge = judge_prompt("who won?", [("Röntgen", "Röntgen won."), ("Curie", "Curie won.")])
    assert "[1] Answer: Röntgen\n    Sentence: Röntgen won.\n[2] Answer: Curie\n    Sentence: Curie won." in judge
    assert judge.endswith("Choice: <candidate number>")
