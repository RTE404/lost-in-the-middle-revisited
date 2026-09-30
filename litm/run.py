"""Run one model on a set of tasks with vLLM, saving every output as it goes.

Usage (one model per GPU):
    CUDA_VISIBLE_DEVICES=0 python -m litm.run --model qwen2.5-3b --tasks closedbook oracle qa20 --data <authors repo> --out results
    CUDA_VISIBLE_DEVICES=1 python -m litm.run --model qwen3-4b-2507 --tasks kv300 --kv-limit 200 ...

Rerunning the same command resumes: finished (task, position, idx) triples are skipped.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from litm.data import all_items

MODELS = {
    "qwen2.5-3b": {"hf_id": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1"},
    "qwen3-4b-2507": {"hf_id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
}
MAX_NEW_TOKENS = 100


def key(record) -> tuple:
    return (record["task"], record["position"], record["idx"])


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {key(json.loads(line)) for line in f if line.strip()}


def render(tokenizer, prompt: str) -> str:
    """The paper prompt as one user message in the model's own chat template, as shipped."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def environment() -> dict:
    import torch
    import transformers
    import vllm

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--tasks", nargs="+", required=True, help="closedbook oracle qa20 kv300")
    parser.add_argument("--data", type=Path, required=True, help="Path to the authors' repo checkout")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kv-limit", type=int, default=None, help="Use only the first N key-value examples")
    parser.add_argument("--qa-limit", type=int, default=None, help="Use only the first N questions (pilot/rerun)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=20608)  # longest prompt measured: 20,458 tokens (kv300) + 100 new tokens
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tag", default="", help="Suffix for the output file, e.g. 'pilot' or 'rerun'")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = MODELS[args.model]
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}{'__' + args.tag if args.tag else ''}"
    out_path = args.out / f"{stem}.jsonl"
    meta_path = args.out / f"{stem}.meta.json"
    samples_path = args.out / f"{stem}.prompt_samples.jsonl"

    items = all_items(args.data, args.tasks, kv_limit=args.kv_limit)
    if args.qa_limit is not None:
        items = [x for x in items if x["task"].startswith("kv") or x["idx"] < args.qa_limit]
    done = load_done(out_path)
    pending = [x for x in items if key(x) not in done]
    print(f"{len(items)} items, {len(done)} already done, {len(pending)} to run", flush=True)
    if not pending:
        return

    # Render and tokenize everything first so over-long prompts fail before any GPU time is spent.
    tokenizer = AutoTokenizer.from_pretrained(model["hf_id"], revision=model["revision"])
    too_long = []
    seen_groups = set()
    rendered_all = [render(tokenizer, item["prompt"]) for item in pending]
    token_ids_all = tokenizer(rendered_all, add_special_tokens=False)["input_ids"]  # batched = parallel
    with open(samples_path, "a", encoding="utf-8") as samples:
        for item, rendered, token_ids in zip(pending, rendered_all, token_ids_all):
            item["token_ids"] = token_ids
            item["prompt_sha256"] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            if len(item["token_ids"]) + MAX_NEW_TOKENS > args.max_model_len:
                too_long.append((key(item), len(item["token_ids"])))
            group = (item["task"], item["position"])
            if group not in seen_groups:  # one full rendered prompt per (task, position)
                seen_groups.add(group)
                samples.write(json.dumps({**{k: item[k] for k in ("task", "position", "idx")}, "rendered": rendered}) + "\n")
    if too_long:
        raise SystemExit(f"{len(too_long)} prompts exceed --max-model-len {args.max_model_len}, e.g. {too_long[:3]}")
    lengths = [len(x["token_ids"]) for x in pending]
    print(f"prompt tokens: max {max(lengths)}, mean {sum(lengths) / len(lengths):.0f}, total {sum(lengths)}", flush=True)

    llm = LLM(
        model=model["hf_id"],
        revision=model["revision"],
        dtype="float16",  # T4 has no bfloat16
        attention_backend="TRITON_ATTN",  # FlashAttention needs sm80+; FlashInfer has crashed on T4
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    meta = {
        "model": args.model,
        **model,
        "tasks": args.tasks,
        "kv_limit": args.kv_limit,
        "qa_limit": args.qa_limit,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "max_new_tokens": MAX_NEW_TOKENS,
        "environment": environment(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    started, tokens_done = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as out:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            results = llm.generate([TokensPrompt(prompt_token_ids=x["token_ids"]) for x in batch], sampling, use_tqdm=False)
            for item, result in zip(batch, results):
                completion = result.outputs[0]
                record = {k: item[k] for k in ("task", "position", "idx", "answers", "prompt_sha256")}
                record.update(
                    model=args.model,
                    n_prompt_tokens=len(item["token_ids"]),
                    output=completion.text,
                    n_output_tokens=len(completion.token_ids),
                    finish_reason=completion.finish_reason,
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            tokens_done += sum(len(x["token_ids"]) for x in batch)
            elapsed = time.time() - started
            print(
                f"{start + len(batch)}/{len(pending)} done, {elapsed / 60:.1f} min, {tokens_done / elapsed:.0f} prompt tok/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
