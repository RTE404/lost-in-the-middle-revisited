import json

import numpy as np
import pytest

from litm.mapreduce.analyze import analyze, main, to_markdown

POSITIONS = [0, 4, 9, 14, 19]
MR_METHODS = {"mr": 0.68, "mr_nofallback": 0.60, "mr_vote": 0.66, "mr_judge": 0.67,
              "control": 0.60, "gold_group": 0.70, "oracle_reduce": 0.80}


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def synthetic_results(tmp_path, base_probs, n=600):
    """Baseline with the given per-position accuracy; map-reduce flat at MR_METHODS rates."""
    rng = np.random.default_rng(0)
    base, finals, diags = [], [], []
    for idx in range(n):
        base.append({"model": "toy", "task": "closedbook", "position": None, "idx": idx,
                     "answers": ["yes"], "output": "yes" if rng.random() < 0.3 else "no"})
        for p, prob in zip(POSITIONS, base_probs):
            ok = rng.random() < prob
            base.append({"model": "toy", "task": "qa20", "position": p, "idx": idx,
                         "answers": ["yes"], "output": "yes" if ok else "no"})
            for method, mprob in MR_METHODS.items():
                correct = int(rng.random() < mprob)
                finals.append({"model": "toy", "task": "qa20", "position": p, "idx": idx, "method": method,
                               "answer": "yes", "abstained": False, "path": "verified", "correct": correct,
                               "strict": correct, "answer_words": 1, "gold_group": p // 4, "gold_slot": idx % 4,
                               "mate_ranks": [p // 4 * 4, p // 4 * 4 + 1, p // 4 * 4 + 2],
                               "n_candidates": 2, "n_verified": 1})
            for g in range(5):
                gold = g == p // 4
                diags.append({"model": "toy", "task": "qa20", "position": p, "idx": idx, "group": g,
                              "is_gold_group": gold, "n_answer_bearing": 0,
                              "status": "answer" if gold else "not_found", "reason": "", "hedged": False,
                              "verified": gold, "verify_how": "exact" if gold else "", "finish_reason": "stop",
                              "attempt": 1, "cand_correct": int(gold and idx % 3 > 0),
                              "p_yes": (0.9 if idx % 3 else 0.2) if gold else None,
                              "p_yes_found": (idx % 5 > 0) if gold else None,
                              "logprob": -0.5 if gold else None})
    write(tmp_path / "toy.jsonl", base)
    write(tmp_path / "toy__mr-qa20-final-pilot.jsonl", finals)
    write(tmp_path / "toy__mr-qa20-diag-pilot.jsonl", diags)


def test_a_flat_non_inferior_map_reduce_works(tmp_path):
    synthetic_results(tmp_path, [0.75, 0.55, 0.55, 0.55, 0.70])
    s = analyze(tmp_path, "qa20", "pilot")["models"]["toy"]
    assert s["baseline_verdict"] == "U shape" and s["gate_open"]
    assert s["delta_u"]["diff"] > 0.1 and s["delta_u"]["holm_pass"]
    assert s["non_inferiority"]["diff"] > 0 and s["non_inferiority"]["holm_pass"]
    assert s["outcome"] == "Works"
    assert set(s["curves"]) == {"baseline", *MR_METHODS}
    assert s["diagnostics"]["not_found_gold_free"] == 1.0
    assert s["diagnostics"]["auroc_p_yes"] == 1.0
    assert s["diagnostics"]["p_yes_not_found"] == pytest.approx(0.2)  # idx % 5 == 0 of the scored rows
    assert set(s["by_slot"]) == {"0", "1", "2", "3"}
    assert "closedbook" in s["reference"]
    assert s["secondary"]["baseline"]["answer_words"] == 1.0
    assert set(s["secondary"]) == {"baseline", *MR_METHODS} - {"oracle_reduce"}


def test_a_worse_map_reduce_is_flatter_but_worse(tmp_path):
    synthetic_results(tmp_path, [0.90, 0.72, 0.72, 0.72, 0.85])
    s = analyze(tmp_path, "qa20", "pilot")["models"]["toy"]
    assert s["outcome"] == "Flatter but worse"


def test_a_flat_baseline_closes_the_gate(tmp_path):
    synthetic_results(tmp_path, [0.68] * 5)
    summary = analyze(tmp_path, "qa20", "pilot")
    s = summary["models"]["toy"]
    assert not s["gate_open"]
    assert "Gate closed" in to_markdown(summary)


def test_main_writes_the_report(tmp_path):
    synthetic_results(tmp_path, [0.75, 0.55, 0.55, 0.55, 0.70], n=100)
    out = tmp_path / "analysis"
    main([str(tmp_path), "--task", "qa20", "--sample", "pilot", "--out", str(out), "--no-plot"])
    assert (out / "mr_qa20_pilot.md").read_text(encoding="utf-8").startswith("# Map-reduce vs baseline")
    assert json.loads((out / "mr_qa20_pilot.json").read_text())["models"]["toy"]["outcome"] == "Works"


def test_main_writes_a_plot(tmp_path):
    pytest.importorskip("matplotlib")
    synthetic_results(tmp_path, [0.75, 0.55, 0.55, 0.55, 0.70], n=100)
    out = tmp_path / "analysis"
    main([str(tmp_path), "--task", "qa20", "--sample", "pilot", "--out", str(out)])
    assert (out / "mr_qa20_toy.png").exists()


def test_missing_reduce_output_is_explained(tmp_path):
    with pytest.raises(SystemExit, match="reduce"):
        analyze(tmp_path, "qa20", "pilot")


def drop_rows(path, keep):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    write(path, [r for r in rows if keep(r)])


def test_a_control_run_on_a_subset_still_gets_a_curve(tmp_path):
    synthetic_results(tmp_path, [0.75, 0.55, 0.55, 0.55, 0.70])
    drop_rows(tmp_path / "toy__mr-qa20-final-pilot.jsonl", lambda r: r["method"] != "control" or r["idx"] % 3 == 0)
    summary = analyze(tmp_path, "qa20", "pilot")
    s = summary["models"]["toy"]
    assert "control" in s["curves"]
    assert s["curve_n"]["control"] == 200 and s["curve_n"]["mr"] == 600
    assert "control (n = 200)" in to_markdown(summary)


def test_the_gate_uses_the_whole_baseline_not_the_analysed_sample(tmp_path):
    synthetic_results(tmp_path, [0.75, 0.55, 0.55, 0.55, 0.70])
    for kind in ("final", "diag"):
        drop_rows(tmp_path / f"toy__mr-qa20-{kind}-pilot.jsonl", lambda r: r["idx"] < 20)
    s = analyze(tmp_path, "qa20", "pilot")["models"]["toy"]
    assert s["n"] == 20 and s["gate_n"] == 600
    assert s["baseline_verdict"] == "U shape" and s["gate_open"]
