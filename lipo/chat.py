"""Thin wrapper over a Tinker sampling client that renders chat messages, samples, and tracks spend."""

from dataclasses import dataclass, field

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Dollars per million tokens, from tinker-docs models_and_pricing (checked 2026-11-16).
PRICES_PER_MILLION = {
    "Qwen/Qwen3.5-4B": {"prefill": 0.33, "sample": 1.005},
    "openai/gpt-oss-120b": {"prefill": 0.33, "sample": 0.84},
}


@dataclass
class SpendMeter:
    """Running count of prompt and sampled tokens per model, convertible to an estimated dollar cost."""

    prefill_tokens: dict[str, int] = field(default_factory=dict)
    sampled_tokens: dict[str, int] = field(default_factory=dict)

    def add(self, model_name: str, prefill: int, sampled: int) -> None:
        """Record one request's token counts for a model."""
        self.prefill_tokens[model_name] = self.prefill_tokens.get(model_name, 0) + prefill
        self.sampled_tokens[model_name] = self.sampled_tokens.get(model_name, 0) + sampled

    def dollars(self) -> float:
        """Estimate total spend from recorded tokens and the price table."""
        total = 0.0
        for model_name, price in PRICES_PER_MILLION.items():
            total += self.prefill_tokens.get(model_name, 0) * price["prefill"] / 1e6
            total += self.sampled_tokens.get(model_name, 0) * price["sample"] / 1e6
        return total


@dataclass
class ChatModel:
    """A base model on Tinker plus the renderer that formats its chat prompts."""

    model_name: str
    sampler: tinker.SamplingClient
    renderer: renderers.Renderer
    meter: SpendMeter

    @classmethod
    async def create(
        cls, service: tinker.ServiceClient, model_name: str, renderer_name: str, meter: SpendMeter
    ) -> "ChatModel":
        """Open a sampling client for a base model and build its renderer."""
        sampler = await service.create_sampling_client(base_model=model_name)
        renderer = renderers.get_renderer(renderer_name, get_tokenizer(model_name), model_name=model_name)
        return cls(model_name, sampler, renderer, meter)

    async def reply(
        self, messages: list[renderers.Message], samples: int, max_tokens: int, temperature: float
    ) -> list[str]:
        """Sample `samples` assistant replies to a conversation and return their final text (thinking removed)."""
        prompt = self.renderer.build_generation_prompt(messages)
        response = await self.sampler.sample(
            prompt=prompt,
            num_samples=samples,
            sampling_params=types.SamplingParams(
                max_tokens=max_tokens, temperature=temperature, stop=self.renderer.get_stop_sequences()
            ),
        )
        sampled = sum(len(sequence.tokens) for sequence in response.sequences)
        # GOTCHA: num_samples shares one prefill, so prompt tokens are billed once per request, not per sample.
        self.meter.add(self.model_name, prompt.length, sampled)
        texts = []
        for sequence in response.sequences:
            message, _termination = self.renderer.parse_response(sequence.tokens)
            texts.append(renderers.get_text_content(message).strip())
        return texts
