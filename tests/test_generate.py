"""litm.generate with stand-ins for vLLM, torch and the tokenizer."""
import sys
import types

import pytest

from litm.generate import GenParams, Generation, to_generation


def logprob(value, text):
    return types.SimpleNamespace(logprob=value, decoded_token=text)


def test_to_generation():
    completion = types.SimpleNamespace(
        text="Yes.", token_ids=[7, 9], finish_reason="length",
        logprobs=[{7: logprob(-0.1, "Yes"), 8: logprob(-2.5, "No")}, {9: logprob(-0.5, ".")}],
    )
    assert to_generation(completion) == Generation("Yes.", [-0.1, -0.5], ["Yes", "."], [("Yes", -0.1), ("No", -2.5)], "length")


def test_to_generation_without_logprobs():
    completion = types.SimpleNamespace(text="x", token_ids=[1], finish_reason="stop", logprobs=None)
    assert to_generation(completion) == Generation("x", [], [], [], "stop")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return f"<user>{messages[0]['content']}<assistant>"

    def __call__(self, texts, add_special_tokens):
        return {"input_ids": [list(range(len(t) // 4)) for t in texts]}


class FakeLLM:
    def __init__(self, **kwargs):
        FakeLLM.kwargs = kwargs

    def generate(self, prompts, sampling, use_tqdm):
        FakeLLM.sampling = sampling
        out = types.SimpleNamespace(text="ok", token_ids=[1], finish_reason="stop", logprobs=[{1: logprob(-0.2, "ok")}])
        return [types.SimpleNamespace(outputs=[out]) for _ in prompts]


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = FakeLLM, lambda **kw: kw
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: prompt_token_ids
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda: (7, 5))
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())
    for name, module in {"vllm": vllm, "vllm.inputs": inputs, "torch": torch, "transformers": transformers}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return torch


def test_vllm_generator_is_greedy_fp16_triton(fake_vllm):
    from litm.generate import VLLMGenerator

    gen = VLLMGenerator("qwen2.5-3b", max_model_len=2048)
    out = gen.generate(["hello", "world"], GenParams(max_tokens=150))
    assert [g.text for g in out] == ["ok", "ok"] and out[0].token_logprobs == [-0.2]
    assert FakeLLM.kwargs["dtype"] == "float16" and FakeLLM.kwargs["attention_backend"] == "TRITON_ATTN"
    assert FakeLLM.kwargs["max_model_len"] == 2048
    assert FakeLLM.sampling == {"temperature": 0.0, "max_tokens": 150, "logprobs": 1}


def test_vllm_generator_refuses_long_prompts_and_other_gpus(fake_vllm, monkeypatch):
    from litm.generate import VLLMGenerator

    gen = VLLMGenerator("qwen2.5-3b", max_model_len=40)
    with pytest.raises(SystemExit, match="exceed"):
        gen.generate(["x" * 400], GenParams(max_tokens=10))
    monkeypatch.setattr(fake_vllm.cuda, "get_device_capability", lambda: (8, 0))
    with pytest.raises(SystemExit, match="sm_75"):
        VLLMGenerator("qwen2.5-3b", max_model_len=2048)
