"""One interface for text generation: generate(prompts, params) -> [Generation].

VLLMGenerator runs on a Kaggle T4 now; phase C can add an API-backed class with the same method.
Decoding is always greedy with explicit SamplingParams: Qwen's generation_config.json would
otherwise make vLLM sample.
"""
from dataclasses import dataclass
from typing import List, Tuple

from litm.run import MODELS, render


@dataclass(frozen=True)
class GenParams:
    max_tokens: int
    logprobs: int = 1  # top-k logprobs per generated token; the chosen token is always included


@dataclass(frozen=True)
class Generation:
    text: str
    token_logprobs: List[float]           # logprob of each generated token
    token_texts: List[str]                # decoded text of each generated token
    first_top: List[Tuple[str, float]]    # top-k (token text, logprob) at the first position
    finish_reason: str                    # "stop" or "length"


def to_generation(completion) -> Generation:
    """Plain lists from a vLLM CompletionOutput (logprobs: one {token_id: Logprob} dict per token)."""
    positions = completion.logprobs or []
    chosen = [position[token_id] for token_id, position in zip(completion.token_ids, positions)]
    first_top = [(lp.decoded_token or "", lp.logprob) for lp in positions[0].values()] if positions else []
    return Generation(
        text=completion.text,
        token_logprobs=[lp.logprob for lp in chosen],
        token_texts=[lp.decoded_token or "" for lp in chosen],
        first_top=first_top,
        finish_reason=completion.finish_reason,
    )


class VLLMGenerator:
    def __init__(self, model_key: str, max_model_len: int, enforce_eager: bool = False, require_t4: bool = True):
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        capability = torch.cuda.get_device_capability()
        if require_t4 and tuple(capability) != (7, 5):
            raise SystemExit(f"Expected a T4 (sm_75), got compute capability {capability}")
        model = MODELS[model_key]
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(model["hf_id"], revision=model["revision"])
        self.llm = LLM(
            model=model["hf_id"],
            revision=model["revision"],
            dtype="float16",  # T4 has no bfloat16
            attention_backend="TRITON_ATTN",
            max_model_len=max_model_len,
            gpu_memory_utilization=0.9,
            enforce_eager=enforce_eager,
            seed=0,
        )

    def generate(self, prompts: List[str], params: GenParams) -> List[Generation]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        rendered = [render(self.tokenizer, p) for p in prompts]
        ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        too_long = [len(x) for x in ids if len(x) + params.max_tokens > self.max_model_len]
        if too_long:
            raise SystemExit(f"{len(too_long)} prompts exceed max_model_len {self.max_model_len} "
                             f"(longest {max(too_long)} tokens)")
        # Never pass prompt_logprobs: it disables vLLM's prefix cache.
        sampling = SamplingParams(temperature=0.0, max_tokens=params.max_tokens, logprobs=params.logprobs)
        results = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sampling, use_tqdm=False)
        return [to_generation(r.outputs[0]) for r in results]
