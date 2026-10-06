"""Map-reduce runs (spec: docs/superpowers/specs/2026-09-30-map-reduce-fix-design.md).

GPU stages, one model per GPU, in this order (the last three are QA only):
    python -m litm.mapreduce.run --model qwen2.5-3b --task qa20 --stage map --data <authors repo> --out results --questions pilot
    ... --stage check / judge / control
then on CPU:
    ... --stage reduce

Every generated output is saved in <model>__mr-<stage>.jsonl, keyed by a hash of the exact prompt and
generation settings, so identical groups are generated once, reruns resume, and any prompt change
invalidates old outputs automatically. `reduce` writes <model>__mr-<task>-final-<sample>.jsonl (one
row per question, position and method) and <model>__mr-<task>-diag-<sample>.jsonl (one row per group).
"""
import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from litm.data import POSITIONS, kv_path, oracle_path, qa_path, read_jsonl
from litm.generate import GenParams
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
from litm.mapreduce.parse import answer_logprob, broken_reason, p_yes, parse_judge, parse_map_output
from litm.mapreduce.prompts import check_prompt, judge_prompt, map_prompt, qa_map_prompt
from litm.mapreduce.split import SEED, content_key, kv_map_items, qa_map_items, sample_ids
from litm.mapreduce.verify import verify_kv, verify_qa
from litm.run import MODELS, environment
from litm.scoring import score_final, score_final_kv, strict_em
from litm.vendor.lost_in_the_middle.metrics import normalize_answer

MAP_PARAMS = GenParams(max_tokens=150, logprobs=1)
CHECK_PARAMS = GenParams(max_tokens=1, logprobs=20)
JUDGE_PARAMS = GenParams(max_tokens=10, logprobs=1)
CONTROL_PARAMS = MAP_PARAMS
SHORT_MAX_LEN = 2048    # map, check and judge prompts are at most 1.5K tokens (6-document qa30 groups)
CONTROL_MAX_LEN = 8192  # 20 or 30 documents in one prompt (about 3.1K / 4.7K tokens)
GPU_STAGES = ("map", "check", "judge", "control")
QA_ONLY_STAGES = ("check", "judge", "control")


# ---------------------------------------------------------------- files and resumable generation

def stage_path(args, stage: str) -> Path:
    return args.out / f"{args.model}__mr-{stage}.jsonl"


def result_path(args, kind: str) -> Path:
    return args.out / f"{args.model}__mr-{args.task}-{kind}-{args.questions}.jsonl"


def job_key(prompt: str, params: GenParams) -> str:
    return content_key(prompt, params.max_tokens, params.logprobs)


def load_records(path: Path) -> dict:
    """{key: record}; a later attempt for the same key replaces an earlier one. A last line cut off
    by a killed session (no final newline, possibly mid-character) is skipped, so its prompt is
    simply generated again; a bad line elsewhere raises."""
    records = {}
    if path.exists():
        data = path.read_bytes()
        text = data[:data.rfind(b"\n") + 1].decode("utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            old = records.get(record["key"])
            if old is None or record["attempt"] >= old["attempt"]:
                records[record["key"]] = record
    return records


def cut_torn_tail(path: Path) -> None:
    """Drop a last line left half-written by a killed session, so the next append starts on a fresh line."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[:data.rfind(b"\n") + 1])


def subset_ids(ids, n):
    """A seeded n-question subset of `ids`, sorted (spec §7: the control may run on a seeded subset)."""
    if n is None or n >= len(ids):
        return list(ids)
    rng = np.random.default_rng([SEED, n])
    return sorted(ids[i] for i in rng.choice(len(ids), size=n, replace=False))


def run_prompts(gen, path: Path, prompts, params: GenParams, batch_size: int) -> None:
    """Generate every prompt not yet saved in `path`, appending after each batch.
    A broken output (NaN logprob, only '!') is re-run once, in a later batch."""
    prompts = list(dict.fromkeys(prompts))
    path.touch()  # an empty file still records that the stage ran (reduce checks for it)
    cut_torn_tail(path)
    done = load_records(path)
    todo = []
    for prompt in prompts:
        previous = done.get(job_key(prompt, params))
        if previous is None:
            todo.append((prompt, 1))
        elif previous["attempt"] == 1 and broken_reason(previous["text"], previous["token_logprobs"]):
            todo.append((prompt, 2))
    print(f"{path.name}: {len(prompts)} prompts, {len(prompts) - len(todo)} done, {len(todo)} to run", flush=True)
    while todo:
        retry = []
        with open(path, "a", encoding="utf-8") as f:
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                outputs = gen.generate([prompt for prompt, _ in batch], params)
                for (prompt, attempt), g in zip(batch, outputs):
                    record = {"key": job_key(prompt, params), "attempt": attempt, "text": g.text,
                              "token_logprobs": g.token_logprobs, "token_texts": g.token_texts,
                              "first_top": g.first_top, "finish_reason": g.finish_reason}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if attempt == 1 and broken_reason(g.text, g.token_logprobs):
                        retry.append((prompt, 2))
                f.flush()
                print(f"{path.name}: {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
        todo = retry


def make_generator(model: str, max_model_len: int, enforce_eager: bool, meta_path: Path):
    from litm.generate import VLLMGenerator

    gen = VLLMGenerator(model, max_model_len, enforce_eager)
    meta = {"model": model, **MODELS[model], "max_model_len": max_model_len, "environment": environment()}
    meta_path.write_text(json.dumps(meta, indent=2))
    return gen


# ---------------------------------------------------------------- items

def question_ids(args) -> list:
    if args.task.startswith("kv"):
        n_total = len(read_jsonl(kv_path(args.data)))
        if args.kv_limit is not None:
            n_total = min(n_total, args.kv_limit)
    else:
        n_total = len(read_jsonl(oracle_path(args.data)))
    return sample_ids(n_total, args.questions, args.limit)


def map_items(args) -> list:
    ids = question_ids(args)
    if args.task.startswith("kv"):
        return kv_map_items(args.data, POSITIONS[args.task], ids)
    return qa_map_items(args.data, int(args.task[2:]), POSITIONS[args.task], ids)


def question_groups(items) -> dict:
    """{(position, idx): [its group items]}"""
    groups = defaultdict(list)
    for item in items:
        groups[(item["position"], item["idx"])].append(item)
    return dict(sorted(groups.items()))


def control_prompts(args) -> dict:
    """{(position, idx): all documents in one prompt, original order, map prompt wording} (spec §7)."""
    ids, n_docs, prompts = set(subset_ids(question_ids(args), args.control_subset)), int(args.task[2:]), {}
    for position in POSITIONS[args.task]:
        for idx, example in enumerate(read_jsonl(qa_path(args.data, position, n_docs))):
            if idx in ids:
                docs = [{"title": c["title"], "text": c["text"]} for c in example["ctxs"]]
                prompts[(position, idx)] = qa_map_prompt(example["question"], docs)
    return prompts


# ---------------------------------------------------------------- parse + verify one question's groups

def collect(groups, maps: dict):
    """-> (candidates, rows) where rows are (item, map record, parsed, verdict) for every group."""
    cands, rows = [], []
    for item in groups:
        kv = item["task"].startswith("kv")
        label = "value" if kv else "answer"
        record = maps.get(job_key(map_prompt(item), MAP_PARAMS))
        if record is None:
            raise SystemExit(f"Missing map output for {item['task']} line {item['idx']} group {item['group']}: "
                             "run --stage map first")
        parsed = parse_map_output(record["text"], record["token_logprobs"], label)
        verdict = None
        if parsed.status == "answer":
            if kv:
                verdict = verify_kv(parsed.answer, parsed.evidence, item["docs"], item["query_key"])
            else:
                verdict = verify_qa(parsed.answer, parsed.evidence, item["docs"])
            cands.append(Candidate(item["group"], parsed.answer, parsed.evidence, verdict.verified, parsed.hedged,
                                   None, answer_logprob(record["token_texts"], record["token_logprobs"], label)))
        rows.append((item, record, parsed, verdict))
    return cands, rows


def lookup_p_yes(checks: dict, question: str, cand: Candidate) -> Candidate:
    """`cand` with P(Yes) filled in; p_yes_found is False when neither Yes nor No was in the top tokens (0.5 then)."""
    record = checks.get(job_key(check_prompt(question, cand.evidence, cand.answer), CHECK_PARAMS))
    if record is None:
        raise SystemExit(f"Missing yes/no output for {cand.answer!r}: run --stage check first")
    value, found = p_yes(record["first_top"])
    return dataclasses.replace(cand, p_yes=value, p_yes_found=found)


def judge_prompt_for(question: str, cands, position: int, idx: int):
    """(prompt, shown order of normalized answers), or None with fewer than two verified answers."""
    distinct = distinct_verified(cands)
    if len(distinct) < 2:
        return None
    order = [distinct[i] for i in np.random.default_rng([SEED, position, idx]).permutation(len(distinct))]
    shown = {}
    for c in cands:
        norm = normalize_answer(c.answer)
        if c.verified and not c.hedged and norm not in shown:
            shown[norm] = c
    return judge_prompt(question, [(shown[n].answer, shown[n].evidence) for n in order]), order


# ---------------------------------------------------------------- GPU stages

def stage_map(args, gen) -> None:
    prompts = [map_prompt(item) for item in map_items(args)]
    run_prompts(gen, stage_path(args, "map"), prompts, MAP_PARAMS, args.batch_size)


def stage_check(args, gen) -> None:
    maps, prompts = load_records(stage_path(args, "map")), []
    for groups in question_groups(map_items(args)).values():
        cands, _ = collect(groups, maps)
        prompts += [check_prompt(groups[0]["question"], c.evidence, c.answer) for c in cands if not c.hedged]
    run_prompts(gen, stage_path(args, "check"), prompts, CHECK_PARAMS, args.batch_size)


def stage_judge(args, gen) -> None:
    maps, prompts = load_records(stage_path(args, "map")), []
    for (position, idx), groups in question_groups(map_items(args)).items():
        cands, _ = collect(groups, maps)
        built = judge_prompt_for(groups[0]["question"], cands, position, idx)
        if built is not None:
            prompts.append(built[0])
    run_prompts(gen, stage_path(args, "judge"), prompts, JUDGE_PARAMS, args.batch_size)


def stage_control(args, gen) -> None:
    run_prompts(gen, stage_path(args, "control"), list(control_prompts(args).values()), CONTROL_PARAMS, args.batch_size)


# ---------------------------------------------------------------- reduce (CPU)

def reduce_task(args) -> None:
    kv = args.task.startswith("kv")
    maps = load_records(stage_path(args, "map"))
    checks = {} if kv else load_records(stage_path(args, "check"))
    judge_file = stage_path(args, "judge")
    judges = load_records(judge_file) if not kv and judge_file.exists() else None
    controls = {} if kv else load_records(stage_path(args, "control"))
    control = {} if kv else control_prompts(args)
    finals, diags = [], []

    for (position, idx), groups in question_groups(map_items(args)).items():
        question, answers = groups[0]["question"], groups[0]["answers"]

        def score(decision: Decision) -> int:
            if kv:
                return score_final_kv(decision.answer, decision.abstained, answers[0])
            return score_final(decision.answer, decision.abstained, answers)

        cands, rows = collect(groups, maps)
        if not kv:
            cands = [c if c.hedged else lookup_p_yes(checks, question, c) for c in cands]
        gold_item, _, gold_parsed, _ = next(row for row in rows if row[0]["is_gold_group"])
        decisions = {"gold_group": Decision(gold_parsed.answer, "gold_group", False)
                     if gold_parsed.status == "answer" else ABSTAIN}
        if kv:
            decisions["mr"] = choose_kv(cands)
        else:
            decisions["mr"] = choose_primary(cands)
            decisions["mr_nofallback"] = choose_primary(cands, use_fallback=False)
            decisions["mr_vote"] = choose_vote(cands)
            if judges is not None:
                judged, built = None, judge_prompt_for(question, cands, position, idx)
                if built is not None:
                    record = judges.get(job_key(built[0], JUDGE_PARAMS))
                    if record is None:
                        raise SystemExit(f"Missing judge output for {args.task} line {idx}: run --stage judge first")
                    choice = parse_judge(record["text"], len(built[1]))
                    judged = None if choice is None else built[1][choice]
                decisions["mr_judge"] = choose_judge(cands, judged)
            prompt = control.get((position, idx))  # absent outside a --control-subset
            record = controls.get(job_key(prompt, CONTROL_PARAMS)) if prompt else None
            if record is not None:
                parsed = parse_map_output(record["text"], record["token_logprobs"])
                decisions["control"] = Decision(parsed.answer, "control", False) if parsed.status == "answer" else ABSTAIN

        common = {"model": args.model, "task": args.task, "position": position, "idx": idx,
                  "gold_group": gold_item["group"], "gold_slot": gold_item["gold_slot"],
                  "mate_ranks": gold_item["mate_ranks"], "n_candidates": len(cands),
                  "n_verified": len(distinct_verified(cands))}
        for method, d in decisions.items():
            finals.append({**common, "method": method, "answer": d.answer, "abstained": d.abstained, "path": d.path,
                           "correct": score(d), "strict": 0 if kv else strict_em(d.answer, d.abstained, answers),
                           "answer_words": len(d.answer.split())})
        oracle = int(any(score(Decision(c.answer, "", False)) for c in cands))
        finals.append({**common, "method": "oracle_reduce", "answer": "", "abstained": False, "path": "oracle",
                       "correct": oracle, "strict": 0, "answer_words": 0})

        by_group = {c.group: c for c in cands}
        for item, record, parsed, verdict in rows:
            cand = by_group.get(item["group"])
            diags.append({
                "model": args.model, "task": args.task, "position": position, "idx": idx, "group": item["group"],
                "is_gold_group": item["is_gold_group"], "n_answer_bearing": item["n_answer_bearing"],
                "status": parsed.status, "reason": parsed.reason, "hedged": parsed.hedged,
                "verified": bool(verdict and verdict.verified), "verify_how": verdict.how if verdict else "",
                "finish_reason": record["finish_reason"], "attempt": record["attempt"],
                "cand_correct": score(Decision(parsed.answer, "", False)) if parsed.status == "answer" else 0,
                "p_yes": cand.p_yes if cand else None, "p_yes_found": cand.p_yes_found if cand else None,
                "logprob": cand.logprob if cand else None,
            })

    for kind, rows in (("final", finals), ("diag", diags)):
        with open(result_path(args, kind), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"reduce: {len(finals)} final rows, {len(diags)} group rows -> {result_path(args, 'final').name}")


# ---------------------------------------------------------------- CLI

STAGES = {"map": stage_map, "check": stage_check, "judge": stage_judge, "control": stage_control}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--task", choices=sorted(POSITIONS), required=True)
    parser.add_argument("--stage", choices=[*GPU_STAGES, "reduce"], required=True)
    parser.add_argument("--data", type=Path, required=True, help="Path to the authors' repo checkout")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--limit", type=int, default=None, help="Keep only the first N questions of the sample")
    parser.add_argument("--kv-limit", type=int, default=None, help="Key-value: use the first N examples, as the baseline did")
    parser.add_argument("--control-subset", type=int, default=None,
                        help="Control stage: run on a seeded subset of N questions (spec §7); reduce uses whatever control outputs exist")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(argv)

    if args.task.startswith("kv") and args.stage in QA_ONLY_STAGES:
        raise SystemExit(f"Stage {args.stage} is not used for {args.task} (spec §9)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "reduce":
        reduce_task(args)
        return
    max_model_len = CONTROL_MAX_LEN if args.stage == "control" else SHORT_MAX_LEN
    gen = make_generator(args.model, max_model_len, args.enforce_eager,
                         args.out / f"{args.model}__mr-{args.stage}.meta.json")
    STAGES[args.stage](args, gen)


if __name__ == "__main__":
    main()
