"""litm.mapreduce.run end to end on the authors' data, with a scripted stand-in for the model."""
import json
import math
import re
from pathlib import Path

import pytest

from litm.data import oracle_path, read_jsonl
from litm.generate import GenParams, Generation
from litm.mapreduce import run as mr_run

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "lost-in-the-middle"
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="authors' repo not checked out in third_party/")

DOC_LINE = re.compile(r"^Document \[\d+\]\(Title: (.*?)\) (.*)$", re.M)
KV_LINE = re.compile(r'"([0-9a-f-]{36})": "([0-9a-f-]{36})"')


class FakeGenerator:
    """Behaves like a tidy model: quotes the sentence holding a gold answer, else says NOT FOUND."""

    def __init__(self, answers_by_question):
        self.answers = answers_by_question
        self.calls = 0

    @staticmethod
    def output(text, first_top=()):
        pieces = re.findall(r"\S+|\s+", text)
        return Generation(text, [-0.5] * len(pieces), pieces, list(first_top), "stop")

    def respond(self, prompt):
        if "Proposed answer:" in prompt:
            return self.output("Yes", [("Yes", math.log(0.8)), ("No", math.log(0.2))])
        if "Which candidate answers the question correctly?" in prompt:
            return self.output("Choice: 1")
        key = re.search(r'^Key: "(.*)"$', prompt, re.M)
        if key:
            for k, v in KV_LINE.findall(prompt):
                if k == key.group(1):
                    return self.output(f'Value: {v}\nEvidence: "{k}": "{v}"')
            return self.output("Value: NOT FOUND")
        question = re.search(r"^Question: (.*)$", prompt, re.M).group(1)
        for _, text in DOC_LINE.findall(prompt):
            for answer in self.answers.get(question, []):
                if answer and answer in text:
                    sentence = next((s for s in re.split(r"(?<=[.!?])\s+", text) if answer in s), text)
                    return self.output(f"Answer: {answer}\nEvidence: {sentence}")
        return self.output("Answer: NOT FOUND")

    def generate(self, prompts, params):
        self.calls += 1
        return [self.respond(p) for p in prompts]


@pytest.fixture(scope="module")
def answers():
    return {ex["question"]: ex["answers"] for ex in read_jsonl(oracle_path(ROOT))}


@pytest.fixture
def fake(monkeypatch, answers):
    gen = FakeGenerator(answers)
    monkeypatch.setattr(mr_run, "make_generator", lambda *args, **kwargs: gen)
    return gen


def cli(tmp_path, stage, task="qa20", *extra):
    return ["--model", "qwen2.5-3b", "--task", task, "--stage", stage, "--data", str(ROOT),
            "--out", str(tmp_path), "--questions", "pilot", "--limit", "3", *extra]


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_all(tmp_path, task="qa20", *extra):
    stages = ("map", "reduce") if task.startswith("kv") else ("map", "check", "judge", "control", "reduce")
    for stage in stages:
        mr_run.main(cli(tmp_path, stage, task, *extra))


def test_job_key_changes_with_prompt_and_settings():
    params = GenParams(max_tokens=150)
    assert mr_run.job_key("a", params) == mr_run.job_key("a", GenParams(max_tokens=150))
    assert mr_run.job_key("a", params) != mr_run.job_key("b", params)
    assert mr_run.job_key("a", params) != mr_run.job_key("a", GenParams(max_tokens=200))


def test_map_stage_generates_each_unique_prompt_once_and_resumes(tmp_path, fake):
    mr_run.main(cli(tmp_path, "map"))
    path = tmp_path / "qwen2.5-3b__mr-map.jsonl"
    records = read(path)
    assert len(records) == 3 * 13  # 13 distinct groups per question across the 5 positions
    assert len({r["key"] for r in records}) == len(records)
    assert all(r["attempt"] == 1 and r["finish_reason"] == "stop" for r in records)
    calls = fake.calls
    mr_run.main(cli(tmp_path, "map"))
    assert fake.calls == calls and len(read(path)) == len(records)


def test_resume_after_an_interrupted_run(tmp_path, fake):
    mr_run.main(cli(tmp_path, "map"))
    path = tmp_path / "qwen2.5-3b__mr-map.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:10]), encoding="utf-8")
    mr_run.main(cli(tmp_path, "map"))
    records = read(path)
    assert len(records) == 39 and len({r["key"] for r in records}) == 39


def test_broken_outputs_are_retried_once(tmp_path, fake, monkeypatch):
    real = fake.generate

    def first_call_broken(prompts, params):
        out = real(prompts, params)
        if fake.calls == 1:
            return [Generation("!!!!", [float("nan")], ["!!!!"], [], "length") for _ in out]
        return out

    monkeypatch.setattr(fake, "generate", first_call_broken)
    mr_run.main(cli(tmp_path, "map", "qa20", "--batch-size", "1000"))
    path = tmp_path / "qwen2.5-3b__mr-map.jsonl"
    assert len(read(path)) == 2 * 39
    latest = mr_run.load_records(path)
    assert all(r["attempt"] == 2 and r["text"] != "!!!!" for r in latest.values())


def test_full_qa_pipeline(tmp_path, fake):
    run_all(tmp_path)
    finals = read(tmp_path / "qwen2.5-3b__mr-qa20-final-pilot.jsonl")
    methods = {r["method"] for r in finals}
    assert methods == {"mr", "mr_nofallback", "mr_vote", "mr_judge", "control", "gold_group", "oracle_reduce"}
    assert len(finals) == 3 * 5 * len(methods)
    for r in finals:
        if r["abstained"]:
            assert r["correct"] == 0 and r["answer"] == ""
    by_key = {(r["method"], r["position"], r["idx"]): r for r in finals}
    for (method, position, idx), r in by_key.items():
        if method == "gold_group":
            assert by_key[("oracle_reduce", position, idx)]["correct"] >= r["correct"]
    assert {r["gold_slot"] for r in finals} <= {0, 1, 2, 3}
    diags = read(tmp_path / "qwen2.5-3b__mr-qa20-diag-pilot.jsonl")
    assert len(diags) == 3 * 5 * 5
    assert sum(d["is_gold_group"] for d in diags) == 3 * 5


def test_every_group_saying_not_found_means_abstain(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(fake, "respond", lambda prompt: FakeGenerator.output("Answer: NOT FOUND"))
    run_all(tmp_path)
    finals = [r for r in read(tmp_path / "qwen2.5-3b__mr-qa20-final-pilot.jsonl") if r["method"] == "mr"]
    assert len(finals) == 15
    assert all(r["abstained"] and r["path"] == "abstain" and r["correct"] == 0 for r in finals)


def test_reduce_needs_the_check_stage(tmp_path, fake):
    mr_run.main(cli(tmp_path, "map"))
    with pytest.raises(SystemExit, match="check"):
        mr_run.main(cli(tmp_path, "reduce"))


def test_full_kv_pipeline(tmp_path, fake):
    run_all(tmp_path, "kv300", "--kv-limit", "20")
    finals = read(tmp_path / "qwen2.5-3b__mr-kv300-final-pilot.jsonl")
    assert {r["method"] for r in finals} == {"mr", "gold_group", "oracle_reduce"}
    assert len(finals) == 3 * 5 * 3
    assert all(r["correct"] == 1 for r in finals if r["method"] == "mr")


def test_kv_rejects_qa_only_stages(tmp_path, fake):
    with pytest.raises(SystemExit, match="not used"):
        mr_run.main(cli(tmp_path, "check", "kv300"))


def test_resume_after_a_half_written_last_line(tmp_path, fake):
    mr_run.main(cli(tmp_path, "map"))
    path = tmp_path / "qwen2.5-3b__mr-map.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:10]) + lines[10][:40], encoding="utf-8")  # killed mid-write
    assert len(mr_run.load_records(path)) == 10
    mr_run.main(cli(tmp_path, "map"))
    assert len(mr_run.load_records(path)) == 39
    records = read(path)  # the torn line was cut, not glued to the next record
    assert len(records) == 39 and len({r["key"] for r in records}) == 39


def test_a_bad_line_before_the_end_still_raises(tmp_path):
    path = tmp_path / "x.jsonl"
    good = json.dumps({"key": "a", "attempt": 1}) + "\n"
    path.write_text(good + '{"key": "b", "att\n' + good, encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        mr_run.load_records(path)


def test_control_on_a_seeded_subset(tmp_path, fake):
    run_all(tmp_path, "qa20", "--control-subset", "2")
    finals = read(tmp_path / "qwen2.5-3b__mr-qa20-final-pilot.jsonl")
    control_ids = {r["idx"] for r in finals if r["method"] == "control"}
    all_ids = {r["idx"] for r in finals}
    assert len(control_ids) == 2 and control_ids < all_ids
    assert control_ids == set(mr_run.subset_ids(sorted(all_ids), 2))


def test_subset_ids_is_seeded_not_the_first_n():
    ids = list(range(2655))
    subset = mr_run.subset_ids(ids, 1000)
    assert len(subset) == 1000 and subset == sorted(subset) and subset != ids[:1000]
    assert subset == mr_run.subset_ids(ids, 1000)
    assert mr_run.subset_ids(ids, None) == ids and mr_run.subset_ids(ids[:5], 10) == ids[:5]
