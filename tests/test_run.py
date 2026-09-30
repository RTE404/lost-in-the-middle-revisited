"""Exercise litm.run end to end with stand-ins for vLLM, torch and the tokenizer."""
import json
import sys
import types
from pathlib import Path

import pytest

from litm import run

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "lost-in-the-middle"
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="authors' repo not checked out in third_party/")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return f"<user>{messages[0]['content']}<assistant>"

    def __call__(self, texts, add_special_tokens):
        return {"input_ids": [list(range(len(t) // 4)) for t in texts]}


class FakeLLM:
    calls = 0

    def __init__(self, **kwargs):
        assert kwargs["dtype"] == "float16" and kwargs["attention_backend"] == "TRITON_ATTN"

    def generate(self, prompts, sampling, use_tqdm):
        FakeLLM.calls += 1
        completion = types.SimpleNamespace(text=" Paris\nmore", token_ids=[1, 2, 3], finish_reason="stop")
        return [types.SimpleNamespace(outputs=[completion]) for _ in prompts]


@pytest.fixture
def fakes(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams, vllm.__version__ = FakeLLM, lambda **kw: kw, "fake"
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: prompt_token_ids
    torch = types.ModuleType("torch")
    torch.__version__ = "fake"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "fake"
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())
    modules = {"vllm": vllm, "vllm.inputs": inputs, "torch": torch, "transformers": transformers}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    FakeLLM.calls = 0


def test_run_writes_resumes_and_guards_length(tmp_path, fakes):
    args = ["--model", "qwen2.5-3b", "--tasks", "closedbook", "qa20", "--data", str(ROOT),
            "--out", str(tmp_path), "--qa-limit", "3", "--batch-size", "4", "--tag", "pilot"]
    run.main(args)
    out = tmp_path / "qwen2.5-3b__pilot.jsonl"
    records = [json.loads(line) for line in out.open()]
    assert len(records) == 3 + 3 * 5
    assert {(r["task"], r["position"]) for r in records} == {("closedbook", None)} | {("qa20", p) for p in (0, 4, 9, 14, 19)}
    assert all(r["output"] == " Paris\nmore" and r["finish_reason"] == "stop" for r in records)
    assert json.loads((tmp_path / "qwen2.5-3b__pilot.meta.json").read_text())["revision"]
    samples = [json.loads(line) for line in (tmp_path / "qwen2.5-3b__pilot.prompt_samples.jsonl").open()]
    assert len(samples) == 6 and samples[0]["rendered"].startswith("<user>")

    # Second run: nothing left to do, no generation calls, no duplicate lines.
    calls = FakeLLM.calls
    run.main(args)
    assert FakeLLM.calls == calls
    assert len(out.read_text().splitlines()) == len(records)

    # Over-long prompts stop the run before the model loads.
    with pytest.raises(SystemExit, match="exceed"):
        run.main([*args[:-2], "--tag", "short", "--max-model-len", "200"])
