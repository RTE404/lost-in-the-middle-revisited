"""Score saved outputs and produce the tables, verdicts and figures. Runs on CPU.

    python -m litm.analyze results/ --out analysis/
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from litm.data import KV_MIDDLE, KV_POSITIONS, QA30_MIDDLE, QA30_POSITIONS, QA_MIDDLE, QA_POSITIONS
from litm.scoring import score
from litm.stats import accuracy_ci, verdict

# Paper reference curves, 20 documents, positions 1/5/10/15/20.
PAPER_QA20 = {
    "gpt-3.5-turbo-0613 (paper)": [75.8, 57.2, 53.8, 55.4, 63.2],
    "llama-2-70b-chat (authors' repo)": [56.8, 53.3, 54.1, 59.7, 69.5],
}
CURVES = {"qa20": (QA_POSITIONS, QA_MIDDLE), "qa30": (QA30_POSITIONS, QA30_MIDDLE), "kv300": (KV_POSITIONS, KV_MIDDLE)}
TITLES = {"qa20": "20 documents", "qa30": "30 documents", "kv300": "300 key-value pairs"}


def load_scores(results_dir: Path, tag: str = "") -> dict:
    """{model: {(task, position): {idx: 0/1}}} from every <model>[__tag].jsonl file with this tag."""
    scores = defaultdict(lambda: defaultdict(dict))
    for path in sorted(results_dir.glob("*.jsonl")):
        if path.name.endswith(".prompt_samples.jsonl"):
            continue
        file_tag = path.stem.split("__")[1] if "__" in path.stem else ""
        if file_tag != tag:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                scores[record["model"]][(record["task"], record["position"])][record["idx"]] = score(record)
    return scores


def summarize(scores: dict) -> dict:
    summary = {}
    for model, groups in scores.items():
        model_summary = {"accuracy": {}, "verdicts": {}}
        for (task, position), by_idx in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            name = task if position is None else f"{task}@{position}"
            model_summary["accuracy"][name] = accuracy_ci(list(by_idx.values()))
        for task, (positions, middle) in CURVES.items():
            if not all((task, p) in groups for p in positions):
                continue
            # Only questions present at every position, so contrasts stay paired.
            common = sorted(set.intersection(*(set(groups[(task, p)]) for p in positions)))
            first, mid, last = ([groups[(task, p)][i] for i in common] for p in (positions[0], middle, positions[-1]))
            result = verdict(first, mid, last)
            result["n"] = len(common)
            model_summary["verdicts"][task] = result
        summary[model] = model_summary
    return summary


def compare_rerun(main: dict, rerun: dict, tolerance: float = 0.02, min_agreement: float = 0.97) -> str:
    """Plan section 8: per position, rerun accuracy within +/-2 points and >=97% identical scores."""
    lines = ["| Model | Setting | Main | Rerun | Agreement | Pass |", "|---|---|---|---|---|---|"]
    all_pass = True
    for model, groups in rerun.items():
        for (task, position), by_idx in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            reference = main.get(model, {}).get((task, position), {})
            common = sorted(set(by_idx) & set(reference))
            if not common:
                continue
            a = sum(reference[i] for i in common) / len(common)
            b = sum(by_idx[i] for i in common) / len(common)
            agreement = sum(reference[i] == by_idx[i] for i in common) / len(common)
            ok = abs(a - b) <= tolerance and agreement >= min_agreement
            all_pass &= ok
            lines.append(f"| {model} | {task}@{position} (n={len(common)}) | {100 * a:.1f} | {100 * b:.1f} | "
                         f"{100 * agreement:.1f}% | {'yes' if ok else 'NO'} |")
    lines.append(f"\nRerun check: {'PASS' if all_pass else 'FAIL'}")
    return "\n".join(lines)


def to_markdown(summary: dict) -> str:
    lines = []
    for model, s in summary.items():
        lines += [f"## {model}", "", "| Setting | Accuracy | 95% CI | n |", "|---|---|---|---|"]
        for name, a in s["accuracy"].items():
            lines.append(f"| {name} | {100 * a['acc']:.1f} | {100 * a['low']:.1f}–{100 * a['high']:.1f} | {a['n']} |")
        for task, v in s["verdicts"].items():
            lines += ["", f"**{task} verdict: {v['shape']}** (n = {v['n']})", ""]
            for label in ("first_vs_middle", "last_vs_middle"):
                c = v[label]
                lines.append(
                    f"- {label.replace('_', ' ')}: {100 * c['diff']:+.1f} pts "
                    f"(95% CI {100 * c['low']:+.1f} to {100 * c['high']:+.1f}), McNemar p = {c['p']:.3g}, "
                    f"discordance {100 * c['discordance']:.1f}%, MDE {100 * c['mde']:.1f} pts"
                )
        lines.append("")
    return "\n".join(lines)


def plot(summary: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for task, (positions, _) in CURVES.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [p + 1 for p in positions]
        for model, s in summary.items():
            points = [s["accuracy"].get(f"{task}@{p}") for p in positions]
            if not all(points):
                continue
            y = [100 * a["acc"] for a in points]
            err = [[100 * (a["acc"] - a["low"]) for a in points], [100 * (a["high"] - a["acc"]) for a in points]]
            line = ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=model)
            if task.startswith("qa"):
                for baseline, style in (("closedbook", ":"), ("oracle", "--")):
                    if baseline in s["accuracy"]:
                        ax.axhline(100 * s["accuracy"][baseline]["acc"], ls=style, color=line[0].get_color(), alpha=0.6,
                                   label=f"{model} {baseline}")
        if task == "qa20":
            for name, y in PAPER_QA20.items():
                ax.plot(x, y, marker=".", color="grey", alpha=0.5, ls="-.", label=name)
        ax.set_xticks(x)
        ax.set_xlabel("Position of the answer" if task.startswith("qa") else "Position of the key")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(TITLES[task])
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{task}.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument("--tag", default="", help="Analyse <model>__<tag>.jsonl files instead of the main runs")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.tag == "rerun":
        report = compare_rerun(load_scores(args.results), load_scores(args.results, "rerun"))
        (args.out / "rerun_check.md").write_text(report, encoding="utf-8")
        print(report)
        return
    summary = summarize(load_scores(args.results, args.tag))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    report = to_markdown(summary)
    (args.out / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    if not args.no_plot:
        plot(summary, args.out)


if __name__ == "__main__":
    main()
