"""Mask-free RL from the SFT warm start: gated judge reward, per-token e penalty, and decaying prefix hints for all-fail groups."""

import argparse
import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tinker
from tinker import TensorData, types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from lipo.chat import ChatModel, SpendMeter
from lipo.checks import check_answer
from lipo.judge import judge_many
from lipo.pilot import NO_E_INSTRUCTION

DATA_DIR = Path(__file__).parent / "data"
RUNS_DIR = Path(__file__).parent / "runs"
BASE_MODEL = "Qwen/Qwen3.5-4B"
RENDERER = "qwen3_5_disable_thinking"
JUDGE_MODEL = "openai/gpt-oss-120b"
CHECKPOINT_TTL_SECONDS = 14 * 24 * 3600

MAX_TOKENS = 300
# GOTCHA: maps the judge's min(coherence, responds) to reward. Eval counts only >= 4 as usable, so a 3
# must earn little; run A gave a 3 half credit and the policy settled for e-free but rambling answers.
QUALITY_REWARD = {5: 1.0, 4: 0.8, 3: 0.15, 2: 0.0, 1: 0.0}
LENGTH_SOFT_LIMIT_WORDS = 110
# GOTCHA: answers with more e's than this get zero reward without a judge call; it saves money and
# is safe because max(0, 1 - 0.25 * e_count) is already 0 at 4 e's.
MAX_E_FOR_JUDGING = 3


@dataclass
class Rollout:
    """One sampled answer: the tokens we train on, their sampler logprobs, any hint prefix, and its reward."""

    prompt: str
    prompt_tokens: list[int]
    hint_tokens: list[int]
    sampled_tokens: list[int]
    sampled_logprobs: list[float]
    answer: str
    reward: float = 0.0


def load_prompts_with_references() -> list[tuple[str, str]]:
    """Return (prompt, reference lipogram) pairs from the teacher set; references feed the prefix hints."""
    return [(row["prompt"], row["answer"]) for row in map(json.loads, (DATA_DIR / "sft_teacher.jsonl").open())]


def hint_fraction(step: int, total_steps: int) -> float:
    """Share of the reference answer given as a hint: 50% for the first third, 25% for the second, then 0."""
    # GOTCHA: decays to exactly 0 so the final policy is graded with no help; QuestA found the 0% stage
    # adds no gain by itself, it just proves the model stands alone.
    if step < total_steps / 3:
        return 0.5
    if step < 2 * total_steps / 3:
        return 0.25
    return 0.0


def token_contains_e(tokenizer, token_id: int) -> bool:
    """Return True if a single token's text contains e or E."""
    return "e" in tokenizer.decode([token_id]).lower()


async def score(rollouts: list[Rollout], judge_model: ChatModel, e_penalty_per_letter: float) -> None:
    """Fill in each rollout's reward: 0 if a hack gate fails, else judge quality times the e penalty."""
    to_judge = []
    for rollout in rollouts:
        result = check_answer(rollout.answer, rollout.prompt)
        # GOTCHA: hack gates are checked with e's ignored, so a fluent answer with 1-3 e's still earns
        # partial credit; requiring e-free here would make almost every early group all-zero.
        hack_free = (result.ascii_only and not result.dropped_e_words and result.word_count >= 12
                     and result.unknown_word_share <= 0.05 and result.repeated_ngram_share <= 0.30)
        # GOTCHA: skip the judge when the e factor is already 0 (e.g. 2+ e's at penalty 0.5); the reward
        # would be 0 regardless, so the call is pure cost.
        if hack_free and result.e_count <= MAX_E_FOR_JUDGING and 1 - e_penalty_per_letter * result.e_count > 0:
            to_judge.append((rollout, max(0.0, 1 - e_penalty_per_letter * result.e_count)))
    verdicts = await judge_many(judge_model, [(r.prompt, r.answer, len(r.sampled_tokens) >= MAX_TOKENS) for r, _ in to_judge])
    for (rollout, e_factor), verdict in zip(to_judge, verdicts):
        if verdict.coherence is None:
            continue
        quality = QUALITY_REWARD[min(verdict.coherence, verdict.responds)]
        words = check_answer(rollout.answer).word_count
        # GOTCHA: soft, not hard: answers past ~110 words drifted off topic in run A, but a hard cap
        # would teach truncation, which the judge forgives and the length gate can't see.
        length_factor = 1.0 if words <= LENGTH_SOFT_LIMIT_WORDS else max(0.3, 1 - (words - LENGTH_SOFT_LIMIT_WORDS) / 100)
        rollout.reward = quality * e_factor * length_factor


def to_datum(rollout: Rollout, advantage: float, tokenizer, token_penalty: float) -> types.Datum:
    """Build an importance-sampling Datum; prompt and hint tokens get zero advantage, e-tokens get an extra penalty."""
    context = rollout.prompt_tokens + rollout.hint_tokens
    observed = len(context) - 1
    token_advantages = [advantage - token_penalty * token_contains_e(tokenizer, token) for token in rollout.sampled_tokens]
    return types.Datum(
        model_input=types.ModelInput.from_ints(context + rollout.sampled_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(np.array([0] * observed + rollout.sampled_tokens)),
            "logprobs": TensorData.from_numpy(np.array([0.0] * observed + rollout.sampled_logprobs)),
            # GOTCHA: hint tokens are part of the context, not trained on; giving them advantage would
            # train the policy on text it never sampled, which breaks the importance-sampling ratio.
            "advantages": TensorData.from_numpy(np.array([0.0] * observed + token_advantages)),
        },
    )


async def main() -> None:
    """Run RL steps, logging reward and e-free rate, and save a sampler checkpoint every `save_every` steps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="rl")
    parser.add_argument("--init-state", required=True, help="tinker:// state path from sft.py")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--token-penalty", type=float, default=0.1)
    parser.add_argument("--hints", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-dollars", type=float, default=6.0)
    parser.add_argument("--judge-renderer", default="gpt_oss_medium_reasoning")
    parser.add_argument("--e-penalty", type=float, default=0.25, help="reward lost per e in the answer")
    arguments = parser.parse_args()

    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER, tokenizer, model_name=BASE_MODEL)
    meter = SpendMeter()
    service = tinker.ServiceClient(user_metadata={"description": f"lipogram RL {arguments.name}"})
    training = await service.create_training_client_from_state(arguments.init_state)
    judge_model = await ChatModel.create(service, JUDGE_MODEL, arguments.judge_renderer, meter)
    pairs = load_prompts_with_references()
    generator = random.Random(0)
    out_dir = RUNS_DIR / arguments.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "log.jsonl").open("a")

    for step in range(arguments.steps):
        sampler = await training.save_weights_and_get_sampling_client()
        batch_pairs = generator.sample(pairs, arguments.prompts_per_step)
        fraction = hint_fraction(step, arguments.steps) if arguments.hints else 0.0

        async def rollout_group(prompt: str, reference: str, with_hint: bool) -> list[Rollout]:
            """Sample one group for a prompt, optionally continuing from a prefix of the reference answer."""
            messages = [{"role": "system", "content": NO_E_INSTRUCTION}, {"role": "user", "content": prompt}]
            prompt_tokens = renderer.build_generation_prompt(messages).to_ints()
            hint_text = " ".join(reference.split()[: int(len(reference.split()) * fraction)]) + " " if with_hint else ""
            hint_tokens = tokenizer.encode(hint_text, add_special_tokens=False) if hint_text.strip() else []
            response = await sampler.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens + hint_tokens),
                num_samples=arguments.group_size,
                sampling_params=types.SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0, stop=renderer.get_stop_sequences()),
            )
            meter.add(BASE_MODEL, len(prompt_tokens) + len(hint_tokens),
                      sum(len(sequence.tokens) for sequence in response.sequences))
            rollouts = []
            for sequence in response.sequences:
                message, _ = renderer.parse_response(sequence.tokens)
                answer = hint_text + renderers.get_text_content(message).strip()
                rollouts.append(Rollout(prompt, prompt_tokens, hint_tokens, list(sequence.tokens),
                                        list(sequence.logprobs), answer.strip()))
            return rollouts

        groups = await asyncio.gather(*(rollout_group(prompt, reference, False) for prompt, reference in batch_pairs))
        await score([rollout for group in groups for rollout in group], judge_model, arguments.e_penalty)
        hinted = 0
        if fraction > 0:
            # GOTCHA: hints only replace groups with zero total reward (BREAD); groups with any success
            # already carry signal, and hinting them would hide what the policy can do alone.
            failing = [index for index, group in enumerate(groups) if max(r.reward for r in group) == 0]
            retries = await asyncio.gather(*(rollout_group(*batch_pairs[index], True) for index in failing))
            await score([rollout for group in retries for rollout in group], judge_model, arguments.e_penalty)
            for index, group in zip(failing, retries):
                groups[index] = group
            hinted = len(failing)

        data = []
        for group in groups:
            rewards = [rollout.reward for rollout in group]
            mean = sum(rewards) / len(rewards)
            # GOTCHA: constant-reward groups give zero advantage everywhere except the token penalty,
            # which alone would push toward short answers; skip them entirely.
            if max(rewards) == min(rewards):
                continue
            for rollout in group:
                data.append(to_datum(rollout, rollout.reward - mean, tokenizer, arguments.token_penalty))
        if data:
            forward_backward = training.forward_backward(data, "importance_sampling")
            optimizer_step = training.optim_step(types.AdamParams(learning_rate=arguments.learning_rate))
            await forward_backward
            await optimizer_step

        unhinted = [rollout for group in groups for rollout in group if not rollout.hint_tokens]
        record = dict(
            step=step,
            hint_fraction=fraction,
            hinted_groups=hinted,
            trained_groups=len(data) // arguments.group_size,
            mean_reward=float(np.mean([r.reward for g in groups for r in g])),
            unhinted_e_free=float(np.mean([check_answer(r.answer).e_free for r in unhinted])) if unhinted else None,
            unhinted_gate_pass=float(np.mean([check_answer(r.answer, r.prompt).passes_gates for r in unhinted])) if unhinted else None,
            median_words=int(np.median([check_answer(r.answer).word_count for g in groups for r in g])),
            sample_spend=round(meter.dollars(), 3),
            example=groups[0][0].answer[:160],
        )
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        print(json.dumps({key: value for key, value in record.items() if key != "example"}), flush=True)

        if (step + 1) % arguments.save_every == 0 or step + 1 == arguments.steps:
            saved = await training.save_weights_for_sampler(name=f"{arguments.name}-step{step + 1}", ttl_seconds=CHECKPOINT_TTL_SECONDS)
            # GOTCHA: also save full state (weights + optimizer); sampler weights alone can't be resumed,
            # which is why run B could not be extended past step 60.
            state = await training.save_state(name=f"{arguments.name}-state-step{step + 1}", ttl_seconds=CHECKPOINT_TTL_SECONDS)
            print(f"checkpoint step {step + 1}: {saved.path} state {state.path}", flush=True)
            with (out_dir / "checkpoints.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(step=step + 1, sampler_path=saved.path, state_path=state.path)) + "\n")
        # GOTCHA: the meter only counts sampling and judging, not training; the cap is a guard, not a bill.
        if meter.dollars() > arguments.max_dollars:
            print(f"stopping: sampling+judging spend ${meter.dollars():.2f} over cap", flush=True)
            break


if __name__ == "__main__":
    asyncio.run(main())
