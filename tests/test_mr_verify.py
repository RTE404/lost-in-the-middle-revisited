from litm.mapreduce.verify import Verdict, verify_kv, verify_qa

DOC = {
    "title": "Nobel Prize in Physics",
    "text": "The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen, of Germany, "
            "who received 150,782 SEK. John Bardeen is the only laureate to win the prize twice.",
}
QUOTE = "The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen, of Germany"


def test_exact_quote_is_verified():
    assert verify_qa("Wilhelm Conrad Röntgen", QUOTE, [DOC]) == Verdict(True, "exact")


def test_the_title_counts_as_document_text():
    assert verify_qa("Physics", "Nobel Prize in Physics The first Nobel Prize", [DOC]) == Verdict(True, "exact")


def test_evidence_across_a_newline_in_the_document():
    doc = {"title": "T", "text": "He was born in Ulm.\nHe died in Princeton in 1955 at the age of 76."}
    assert verify_qa("1955", "born in Ulm. He died in Princeton in 1955", [doc]).verified


def test_rejections():
    assert verify_qa("", QUOTE, [DOC]).how == "empty answer"
    assert verify_qa("*", QUOTE, [DOC]).how == "empty answer"
    assert verify_qa("1901", "awarded in 1901", [DOC]).how == "short evidence"
    assert verify_qa("Einstein", QUOTE, [DOC]).how == "answer not in evidence"
    assert verify_qa("one", "none of the first Nobel Prize winners", [DOC]).how == "answer not in evidence"
    invented = "The first Nobel Prize in Physics was awarded in 1901 to Albert Einstein of Switzerland"
    assert verify_qa("Einstein", invented, [DOC]).how == "evidence not in documents"


def test_fuzzy_match_forgives_small_copy_slips():
    slip = "The first Nobel prize in physics was given in 1901 to Wilhelm Conrad Röntgen, of Germany"
    assert verify_qa("1901", slip, [DOC]) == Verdict(True, "fuzzy")


def test_fuzzy_match_needs_the_answer_in_the_document_not_just_the_quote():
    respelled = "The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Rontgen, of Germany"
    assert verify_qa("Rontgen", respelled, [DOC]).how == "evidence not in documents"


def test_fuzzy_match_rejects_real_text_padded_with_invented_text():
    padded = DOC["text"] + " He also invented the telephone and the radio in 1950."
    assert verify_qa("1950", padded, [DOC]).how == "evidence not in documents"


K1, V1 = "2a8d601d-1d69-4e64-9f90-8ad825a74195", "bb3ba2a5-7de8-434b-a86e-a88bb9fa7289"
K2, V2 = "a54e2eed-e625-4570-9f74-3624e77d6684", "d1ff29be-4e2a-4208-a182-0cea716be3d4"
PAIRS = [[K1, V1], [K2, V2]]


def test_kv_verified():
    assert verify_kv(V1, f'"{K1}": "{V1}"', PAIRS, K1) == Verdict(True, "exact")
    assert verify_kv(f'"{V1.upper()}"', f'{K1}: {V1}', PAIRS, K1).verified


def test_kv_rejections():
    assert verify_kv("NOT A UUID", f'"{K1}": "{V1}"', PAIRS, K1).how == "value not a UUID"
    assert verify_kv(V1, "the value is above", PAIRS, K1).how == "no key-value pair in evidence"
    assert verify_kv(V2, f'"{K2}": "{V2}"', PAIRS, K1).how == "evidence key is not the query key"
    assert verify_kv(V2, f'"{K1}": "{V1}"', PAIRS, K1).how == "value differs from evidence"
    assert verify_kv(V1, f'"{K1}": "{V1}"', [[K2, V2]], K1).how == "pair not in group"
