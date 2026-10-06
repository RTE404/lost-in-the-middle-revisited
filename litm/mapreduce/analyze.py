"""Score map-reduce against the baseline (spec §6, §7). Runs on CPU.

    python -m litm.mapreduce.analyze results --task qa20 --sample pilot --out analysis
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from litm.analyze import load_scores
from litm.data import MIDDLES, POSITIONS
from litm.scoring import strict_em
from litm.stats import accuracy_ci, auroc, delta_u, holm, non_inferiority, range_minus_null, recovered_share, verdict

PRIMARY_ALPHA = 0.025  # one-sided, the same as a two-sided 95% CI
METHODS = ["mr", "mr_nofallback", "mr_vote", "mr_judge", "control", "gold_group", "oracle_reduce"]
GATE_OPEN = ("U shape", "Primacy only", "Recency only")


def read_rows(results: Path, pattern: str) -> list:
    rows = []
    for path in sorted(results.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def matrix(by_key: dict, positions, ids) -> np.ndarray:
    return np.array([[by_key[(p, i)]["correct"] for p in positions] for i in ids], dtype=float)


def by_field(rows, key) -> dict:
    groups = defaultdict(list)
    for r in rows:
        k = key(r)
        if k is not None:
            groups[k].append(r["correct"])
    return {str(k): {"acc": float(np.mean(v)), "n": len(v)} for k, v in sorted(groups.items())}


def mate_bin(row):
    """Mean retrieval rank of the gold's group-mates, in bins of 4 (0 = hardest distractors)."""
    return int(np.mean(row["mate_ranks"])) // 4 * 4 if row.get("mate_ranks") else None


def diagnostics(rows) -> dict:
    if not rows:
        return {}
    gold = [r for r in rows if r["is_gold_group"]]
    free = [r for r in rows if not r["is_gold_group"]]

    def share(subset, condition):
        return float(np.mean([bool(condition(r)) for r in subset])) if subset else float("nan")

    def signal(field):
        scored = [r for r in rows if r["status"] == "answer" and not r["hedged"] and r[field] is not None]
        return auroc([r[field] for r in scored], [r["cand_correct"] for r in scored])

    return {
        "not_found_gold_free": share(free, lambda r: r["status"] == "not_found"),
        "not_found_gold": share(gold, lambda r: r["status"] == "not_found"),
        "verified_gold_free": share(free, lambda r: r["verified"]),
        "verifier_false_rejection": share([r for r in gold if r["status"] == "answer" and r["cand_correct"]],
                                          lambda r: not r["verified"]),
        "invalid": share(rows, lambda r: r["status"] == "invalid"),
        "hedged": share(rows, lambda r: r["hedged"]),
        "truncated": share(rows, lambda r: r["finish_reason"] == "length"),
        "retried": share(rows, lambda r: r["attempt"] > 1),
        "p_yes_not_found": share([r for r in rows if r.get("p_yes_found") is not None],
                                 lambda r: not r["p_yes_found"]),
        "auroc_p_yes": signal("p_yes"),
        "auroc_logprob": signal("logprob"),
    }


def secondary(results: Path, model: str, task: str, methods: dict, positions, ids) -> dict:
    """Spec §5: strict exact match and answer length per method. Baseline answers are the
    first output line, as the authors score them; longer answers get free substring credit."""
    wanted = {(p, i) for p in positions for i in ids}
    out = {}
    base_rows = [r for r in read_rows(results, f"{model}.jsonl")
                 if r["task"] == task and (r["position"], r["idx"]) in wanted]
    if base_rows:
        first_lines = [r["output"].strip().split("\n")[0] for r in base_rows]
        out["baseline"] = {"strict": float(np.mean([strict_em(t, False, r["answers"]) for t, r in zip(first_lines, base_rows)])),
                           "answer_words": float(np.mean([len(t.split()) for t in first_lines]))}
    for method in METHODS:
        rows = [methods[method][k] for k in wanted if k in methods.get(method, {})]
        if rows and method != "oracle_reduce":
            out[method] = {"strict": float(np.mean([r["strict"] for r in rows])),
                           "answer_words": float(np.mean([r["answer_words"] for r in rows]))}
    return out


def analyze(results: Path, task: str, sample: str) -> dict:
    positions = POSITIONS[task]
    mid = positions.index(MIDDLES[task])
    base_scores = load_scores(results)
    finals = defaultdict(lambda: defaultdict(dict))
    for r in read_rows(results, f"*__mr-{task}-final-{sample}.jsonl"):
        finals[r["model"]][r["method"]][(r["position"], r["idx"])] = r
    diags = defaultdict(list)
    for r in read_rows(results, f"*__mr-{task}-diag-{sample}.jsonl"):
        diags[r["model"]].append(r)
    if not finals:
        raise SystemExit(f"No *__mr-{task}-final-{sample}.jsonl files in {results}: run --stage reduce first")

    models = {}
    for model, methods in sorted(finals.items()):
        base = base_scores.get(model, {})
        ids = sorted({i for _, i in methods["mr"]})
        ids = [i for i in ids if all((p, i) in methods["mr"] and i in base.get((task, p), {}) for p in positions)]
        if not ids:
            raise SystemExit(f"{model}: no questions have both baseline and map-reduce results for {task}")
        B = np.array([[base[(task, p)][i] for p in positions] for i in ids], dtype=float)
        M = matrix(methods["mr"], positions, ids)
        curves = {"baseline": [accuracy_ci(B[:, j]) for j in range(len(positions))]}
        curve_n = {"baseline": len(ids)}
        for method in METHODS:
            # a method run on fewer questions (the control on a seeded subset, spec §7) gets its curve on those
            method_ids = [i for i in ids if all((p, i) in methods.get(method, {}) for p in positions)]
            if method_ids:
                X = matrix(methods[method], positions, method_ids)
                curves[method] = [accuracy_ci(X[:, j]) for j in range(len(positions))]
                curve_n[method] = len(method_ids)
        # spec §2: the gate is decided on the whole baseline, not on the questions map-reduce ran on
        gate_ids = sorted(set.intersection(*(set(base.get((task, p), {})) for p in positions)))
        G = np.array([[base[(task, p)][i] for p in positions] for i in gate_ids], dtype=float)
        shape = verdict(G[:, 0], G[:, mid], G[:, -1])["shape"]
        mr_rows = [methods["mr"][(p, i)] for p in positions for i in ids]
        models[model] = {
            "n": len(ids),
            "gate_n": len(gate_ids),
            "baseline_verdict": shape,
            "gate_open": shape in GATE_OPEN,
            "non_inferiority": non_inferiority(M, B),
            "delta_u": delta_u(B, M),
            "recovered": recovered_share(B, M, mid),
            "range": {"baseline": range_minus_null(B), "mr": range_minus_null(M)},
            "curves": curves,
            "curve_n": curve_n,
            "by_slot": by_field(mr_rows, lambda r: r["gold_slot"]),
            "by_mate_rank": by_field(mr_rows, mate_bin),
            "paths": dict(Counter(r["path"] for r in mr_rows)),
            "diagnostics": diagnostics(diags[model]),
            "secondary": secondary(results, model, task, methods, positions, ids),
            "reference": {name: accuracy_ci(list(base[(name, None)].values()))
                          for name in ("closedbook", "oracle") if (name, None) in base},
        }

    names = list(models)
    for test in ("non_inferiority", "delta_u"):
        for name, rejected in zip(names, holm([models[n][test]["p"] for n in names], alpha=PRIMARY_ALPHA)):
            models[name][test]["holm_pass"] = bool(rejected)
    for name in names:
        ni, du = models[name]["non_inferiority"]["holm_pass"], models[name]["delta_u"]["holm_pass"]
        models[name]["outcome"] = "Works" if ni and du else "Flatter but worse" if du else "No effect"
    return {"task": task, "sample": sample, "positions": [p + 1 for p in positions], "models": models}


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def to_markdown(summary: dict) -> str:
    lines = [f"# Map-reduce vs baseline: {summary['task']} ({summary['sample']})", ""]
    for model, s in summary["models"].items():
        ni, du, rec = s["non_inferiority"], s["delta_u"], s["recovered"]
        lines += [f"## {model}: **{s['outcome']}** (n = {s['n']} questions)", ""]
        if not s["gate_open"]:
            lines += [f"> Gate closed: the baseline verdict (all {s['gate_n']} baseline questions) is *{s['baseline_verdict']}*, so there is no "
                      "position effect for map-reduce to fix (spec section 2). Report the outcome as moot.", ""]
        lines += [
            f"- Baseline verdict (gate, all {s['gate_n']} baseline questions): {s['baseline_verdict']}",
            f"- Test 1, non-inferiority (margin {pct(ni['margin'])} pts): MR - baseline = {pct(ni['diff'])} pts "
            f"(95% CI {pct(ni['low'])} to {pct(ni['high'])}), pass after Holm: {ni['holm_pass']}",
            f"- Test 2, flatter: U baseline {pct(du['u_base'])} vs MR {pct(du['u_new'])}; dU = {pct(du['diff'])} pts "
            f"(95% CI {pct(du['low'])} to {pct(du['high'])}), pass after Holm: {du['holm_pass']}",
            f"- Test 3, middle gap recovered: {pct(rec['share'])}% (95% CI {pct(rec['low'])} to {pct(rec['high'])})"
            + (" (unstable: the baseline gap's CI touches 0)" if rec["unstable"] else ""),
            f"- Best-worst range (descriptive): baseline {pct(s['range']['baseline']['range'])} "
            f"(no-effect value {pct(s['range']['baseline']['null_mean'])}), MR {pct(s['range']['mr']['range'])} "
            f"(no-effect value {pct(s['range']['mr']['null_mean'])})",
            "", "| Method | " + " | ".join(f"pos {p}" for p in summary["positions"]) + " |",
            "|---|" + "---|" * len(summary["positions"]),
        ]
        for method, points in s["curves"].items():
            n = s["curve_n"][method]
            label = method if n == s["n"] else f"{method} (n = {n})"
            lines.append(f"| {label} | " + " | ".join(pct(a["acc"]) for a in points) + " |")
        for name, a in s["reference"].items():
            lines.append(f"\n{name}: {pct(a['acc'])}")
        lines += ["", "**Accuracy by gold slot inside its group (MR):** "
                  + ", ".join(f"slot {int(k) + 1}: {pct(v['acc'])} (n={v['n']})" for k, v in s["by_slot"].items())]
        if s["by_mate_rank"]:
            lines.append("**By mean rank of the gold's group-mates (0 = hardest):** "
                         + ", ".join(f"{k}+: {pct(v['acc'])} (n={v['n']})" for k, v in s["by_mate_rank"].items()))
        lines.append("**Strict exact match / mean answer words:** " + ", ".join(
            f"{m} {pct(v['strict'])}% / {v['answer_words']:.1f}" for m, v in s["secondary"].items()))
        lines.append("**Decision paths (MR):** " + ", ".join(f"{k} {v}" for k, v in sorted(s["paths"].items())))
        lines += ["", "**Map-stage diagnostics:**"]
        lines += [f"- {k.replace('_', ' ')}: {v:.3f}" if k.startswith("auroc") else f"- {k.replace('_', ' ')}: {pct(v)}%"
                  for k, v in s["diagnostics"].items()]
        lines.append("")
    return "\n".join(lines)


def plot(summary: dict, out_dir: Path) -> None:
    # Kaggle's notebook kernel exports MPLBACKEND=module://matplotlib_inline.backend_inline, which this
    # environment's matplotlib rejects at import time; plots are only ever written to files.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = summary["positions"]
    for model, s in summary["models"].items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in ("baseline", "mr", "control"):
            if method in s["curves"]:
                points = s["curves"][method]
                y = [100 * a["acc"] for a in points]
                err = [[100 * (a["acc"] - a["low"]) for a in points], [100 * (a["high"] - a["acc"]) for a in points]]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=method)
        for name, style in (("closedbook", ":"), ("oracle", "--")):
            if name in s["reference"]:
                ax.axhline(100 * s["reference"][name]["acc"], ls=style, color="grey", label=name)
        ax.set_xticks(x)
        ax.set_xlabel("Position of the answer")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model}: {summary['task']}, map-reduce vs one prompt")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"mr_{summary['task']}_{model}.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--task", choices=sorted(POSITIONS), default="qa20")
    parser.add_argument("--sample", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = analyze(args.results, args.task, args.sample)
    stem = f"mr_{args.task}_{args.sample}"
    (args.out / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=float))
    report = to_markdown(summary)
    (args.out / f"{stem}.md").write_text(report, encoding="utf-8")
    print(report)
    if not args.no_plot:
        plot(summary, args.out)


if __name__ == "__main__":
    main()
