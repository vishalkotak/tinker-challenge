"""Score a policy on Kevin Wu's protocol: his 50 eval prompts x 4 samples, 300 tokens, temp 0.7, his judge rubric."""

import argparse
import asyncio
import json
import random
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from lipo.chat import ChatModel, SpendMeter
from lipo.checks import check_answer, is_ascii_only
from lipo.judge import judge_many
from lipo.pilot import NO_E_INSTRUCTION

DATA_DIR = Path(__file__).parent / "data"
RUNS_DIR = Path(__file__).parent / "runs"
POLICY_BASE = "Qwen/Qwen3.5-4B"
POLICY_RENDERER = "qwen3_5_disable_thinking"
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_RENDERER = "gpt_oss_high_reasoning"

# GOTCHA: these four settings are Wu's eval protocol; changing any of them breaks the head-to-head.
SAMPLES_PER_PROMPT = 4
MAX_TOKENS = 300
TEMPERATURE = 0.7
BOOTSTRAP_RESAMPLES = 1000


@dataclass
class ScoredAnswer:
    """One eval answer with Wu's e-check, our strict checks, and the gold judge's scores."""

    prompt: str
    answer: str
    cut_off: bool
    wu_e_pass: bool
    strict_pass: bool
    coherence: int | None
    responds: int | None
    judge_usable: bool

    @property
    def wu_usable(self) -> bool:
        """Wu's definition: no e of any kind (accent-normalized) and both judge scores >= 4."""
        return self.wu_e_pass and self.judge_usable

    @property
    def strict_usable(self) -> bool:
        """Wu's definition plus our gates: ASCII only, >= 12 real words, no dropped-e spellings, no loops."""
        return self.strict_pass and self.judge_usable


def wu_e_pass(text: str) -> bool:
    """Wu's core/checks.py e_pass: non-empty and no e after NFKD normalization and casefolding."""
    return bool(text.strip()) and "e" not in unicodedata.normalize("NFKD", text).casefold()


def bootstrap_interval(flags: list[bool], seed: int = 0) -> tuple[float, float]:
    """Return the 5th and 95th percentile of the mean over bootstrap resamples (Wu reports a 90% interval)."""
    generator = random.Random(seed)
    means = sorted(sum(generator.choices(flags, k=len(flags))) / len(flags) for _ in range(BOOTSTRAP_RESAMPLES))
    return means[int(0.05 * BOOTSTRAP_RESAMPLES)], means[int(0.95 * BOOTSTRAP_RESAMPLES) - 1]


async def generate(service: tinker.ServiceClient, model_path: str | None, prompts: list[str], meter: SpendMeter):
    """Sample SAMPLES_PER_PROMPT answers per prompt from a checkpoint (or the base model) without any mask."""
    sampler = await (
        service.create_sampling_client(model_path=model_path)
        if model_path
        else service.create_sampling_client(base_model=POLICY_BASE)
    )
    renderer = renderers.get_renderer(POLICY_RENDERER, get_tokenizer(POLICY_BASE), model_name=POLICY_BASE)

    async def one(prompt: str) -> list[tuple[str, str, bool]]:
        """Sample one prompt's answers and flag which hit the token limit."""
        messages = [{"role": "system", "content": NO_E_INSTRUCTION}, {"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(messages)
        response = await sampler.sample(
            prompt=model_input,
            num_samples=SAMPLES_PER_PROMPT,
            sampling_params=types.SamplingParams(
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=renderer.get_stop_sequences()
            ),
        )
        meter.add(POLICY_BASE, model_input.length, sum(len(sequence.tokens) for sequence in response.sequences))
        rows = []
        for sequence in response.sequences:
            message, _ = renderer.parse_response(sequence.tokens)
            # GOTCHA: a reply that used every token was cut off; Wu's judge is told so and doesn't penalize it.
            rows.append((prompt, renderers.get_text_content(message).strip(), len(sequence.tokens) >= MAX_TOKENS))
        return rows

    groups = await asyncio.gather(*(one(prompt) for prompt in prompts))
    return [row for group in groups for row in group]


def summarize(scored: list[ScoredAnswer]) -> dict:
    """Compute usable rates with 90% bootstrap intervals plus length and e-pass diagnostics."""
    summary = {}
    for name in ("wu_usable", "strict_usable", "wu_e_pass", "strict_pass", "judge_usable"):
        flags = [bool(getattr(answer, name)) for answer in scored]
        low, high = bootstrap_interval(flags)
        summary[name] = dict(rate=sum(flags) / len(flags), low=low, high=high)
    word_counts = sorted(check_answer(answer.answer).word_count for answer in scored)
    summary["median_words"] = word_counts[len(word_counts) // 2]
    summary["share_under_12_words"] = sum(count < 12 for count in word_counts) / len(word_counts)
    summary["non_ascii_share"] = sum(not is_ascii_only(answer.answer) for answer in scored) / len(scored)
    summary["count"] = len(scored)
    return summary


async def main() -> None:
    """Generate eval answers for one policy, judge them, and save answers plus a summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="tinker:// sampler checkpoint; omit for the base model")
    parser.add_argument("--name", required=True)
    parser.add_argument("--answers-file", default=None, help="score pre-generated answers (jsonl with prompt, answer, cut_off)")
    parser.add_argument("--prompts-file", default=str(DATA_DIR / "short_eval_50.jsonl"),
                        help="defaults to Wu's eval set; use data/dev_50.jsonl for checkpoint selection")
    arguments = parser.parse_args()

    prompts = [json.loads(line)["prompt"] for line in Path(arguments.prompts_file).open()]
    meter = SpendMeter()
    service = tinker.ServiceClient(user_metadata={"description": f"lipogram eval: {arguments.name}"})
    if arguments.answers_file:
        rows = [(r["prompt"], r["answer"], r["cut_off"]) for r in map(json.loads, open(arguments.answers_file))]
    else:
        rows = await generate(service, arguments.model_path, prompts, meter)
    judge_model = await ChatModel.create(service, JUDGE_MODEL, JUDGE_RENDERER, meter)
    verdicts = await judge_many(judge_model, rows)

    scored = [
        ScoredAnswer(prompt, answer, cut_off, wu_e_pass(answer), check_answer(answer, prompt).passes_gates,
                     verdict.coherence, verdict.responds, verdict.usable)
        for (prompt, answer, cut_off), verdict in zip(rows, verdicts)
    ]
    summary = summarize(scored)
    summary["spend_dollars"] = meter.dollars()
    out_dir = RUNS_DIR / "eval" / arguments.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "answers.jsonl").open("w") as handle:
        for answer in scored:
            handle.write(json.dumps(asdict(answer)) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for key in ("wu_usable", "strict_usable", "wu_e_pass"):
        value = summary[key]
        print(f"{key:14s} {100 * value['rate']:5.1f}%  [{100 * value['low']:.1f}, {100 * value['high']:.1f}]")
    print(f"median words {summary['median_words']}, under-12 share {summary['share_under_12_words']:.2f}, "
          f"spend ${summary['spend_dollars']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
