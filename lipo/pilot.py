"""$1 pilot: measure how often Qwen3.5-4B writes a gate-passing lipogram unmasked, and how cheaply gpt-oss-120b can repair drafts into clean ones."""

import argparse
import asyncio
import datetime
import json
import random
from pathlib import Path

import tinker

from lipo.chat import ChatModel, SpendMeter
from lipo.checks import check_answer

DATA_DIR = Path(__file__).parent / "data"
RUNS_DIR = Path(__file__).parent / "runs"

POLICY_MODEL = "Qwen/Qwen3.5-4B"
TEACHER_MODEL = "openai/gpt-oss-120b"

# GOTCHA: copied verbatim from Wu's scripts/evaluate.py so the base rate is comparable to his runs.
NO_E_INSTRUCTION = (
    "Answer in natural English without the letters 'e' or 'E'. Preserve correct "
    "spelling and answer the user's request. This restriction applies only to "
    "your final answer, not your thinking."
)

TEACHER_SYSTEM = (
    "You write answers that never contain the letter e, uppercase or lowercase. "
    "Use plain ASCII only, with straight quotes. Write 40 to 150 words of natural, correctly spelled English "
    "that directly answers the request. To avoid e, swap whole words for synonyms or rephrase. "
    "Never drop or misspell letters, never invent words, never switch languages. Reply with the answer only."
)

MAX_SPEND_DOLLARS = 1.50

# GOTCHA: only applied to teacher drafts (training data cleanup); policy answers are judged raw,
# or the eval would forgive the very homoglyph tricks the ASCII gate exists to catch.
TYPOGRAPHIC_TO_ASCII = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                                      "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": " - ",
                                      "\u2026": "...", "\u00a0": " "})


def load_prompts(count: int, seed: int) -> list[str]:
    """Pick `count` prompts from Wu's RL pool, which excludes his 50 eval prompts."""
    rows = [json.loads(line) for line in (DATA_DIR / "rl_pool.jsonl").open()]
    random.Random(seed).shuffle(rows)
    return [row["prompt"] for row in rows[:count]]


def repair_feedback(answer: str, prompt: str) -> str:
    """Build a correction message naming the exact words that broke the rules."""
    result = check_answer(answer, prompt)
    e_words = sorted({word for word in answer.split() if "e" in word.lower()})
    problems = []
    if e_words:
        problems.append("These words contain the letter e: " + ", ".join(e_words[:40]) + ".")
    if result.unknown_words:
        problems.append("These are not real English words: " + ", ".join(sorted(set(result.unknown_words))[:20]) + ".")
    if not result.ascii_only:
        odd_characters = sorted({character for character in answer if ord(character) >= 128})
        problems.append("Use only plain ASCII characters; remove: " + " ".join(odd_characters[:20]) + ".")
    if result.word_count < 40:
        problems.append("Write at least 40 words.")
    problems.append("Rewrite the whole answer with no letter e at all, swapping whole words or rephrasing.")
    return " ".join(problems)


async def measure_policy(policy: ChatModel, prompts: list[str], samples: int, temperature: float) -> list[dict]:
    """Sample the unmasked policy on each prompt and record which answers pass every gate."""

    async def one(prompt: str) -> list[dict]:
        """Sample one prompt's group and check each answer."""
        messages = [{"role": "system", "content": NO_E_INSTRUCTION}, {"role": "user", "content": prompt}]
        answers = await policy.reply(messages, samples=samples, max_tokens=300, temperature=temperature)
        return [dict(prompt=prompt, answer=answer, **check_summary(answer, prompt)) for answer in answers]

    groups = await asyncio.gather(*(one(prompt) for prompt in prompts))
    return [row for group in groups for row in group]


async def measure_teacher(
    teacher: ChatModel, prompts: list[str], meter: SpendMeter, drafts: int, repair_rounds: int
) -> list[dict]:
    """Sample several drafts, repair the one with fewest e's, and record rounds used and acceptance."""
    limiter = asyncio.Semaphore(16)

    async def one(prompt: str) -> dict:
        """Draft answers, then repair the best one up to `repair_rounds` times until it passes the gates."""
        async with limiter:
            messages = [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": prompt}]
            for round_index in range(repair_rounds + 1):
                # GOTCHA: check spend before every call, not just per prompt, so a runaway loop can't overshoot.
                if meter.dollars() > MAX_SPEND_DOLLARS:
                    return dict(prompt=prompt, answer="", rounds=round_index, accepted=False, aborted=True)
                samples = drafts if round_index == 0 else 1
                raw_answers = await teacher.reply(messages, samples=samples, max_tokens=3000, temperature=0.7)
                candidates = [raw.translate(TYPOGRAPHIC_TO_ASCII) for raw in raw_answers]
                # GOTCHA: prefer a draft that already passes; otherwise repair the one with the fewest e's.
                answer = min(candidates, key=lambda text: (not check_answer(text, prompt).passes_gates,
                                                           check_answer(text, prompt).e_count))
                summary = check_summary(answer, prompt)
                if summary["passes_gates"]:
                    return dict(prompt=prompt, answer=answer, rounds=round_index, accepted=True, **summary)
                messages = messages + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": repair_feedback(answer, prompt)},
                ]
            return dict(prompt=prompt, answer=answer, rounds=repair_rounds, accepted=False, **summary)

    return list(await asyncio.gather(*(one(prompt) for prompt in prompts)))


def check_summary(answer: str, prompt: str) -> dict:
    """Flatten a CheckResult into JSON-friendly fields for logging."""
    result = check_answer(answer, prompt)
    return dict(
        passes_gates=result.passes_gates,
        e_free=result.e_free,
        e_count=result.e_count,
        word_count=result.word_count,
        unknown_words=list(result.unknown_words),
        dropped_e_words=list(result.dropped_e_words),
    )


async def main() -> None:
    """Run both pilot measurements, save every answer, and print rates and spend."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-prompts", type=int, default=40)
    parser.add_argument("--policy-samples", type=int, default=16)
    parser.add_argument("--teacher-prompts", type=int, default=30)
    parser.add_argument("--teacher-reasoning", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--drafts", type=int, default=4)
    parser.add_argument("--repair-rounds", type=int, default=6)
    arguments = parser.parse_args()

    run_dir = RUNS_DIR / f"pilot_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    meter = SpendMeter()
    service = tinker.ServiceClient(user_metadata={"description": "lipogram pilot: base rate + teacher repair"})
    policy = await ChatModel.create(service, POLICY_MODEL, "qwen3_5_disable_thinking", meter)
    teacher = await ChatModel.create(service, TEACHER_MODEL, f"gpt_oss_{arguments.teacher_reasoning}_reasoning", meter)

    # GOTCHA: teacher prompts always come from the same slice (after the first 40), so reruns with
    # --policy-prompts 0 retest the exact prompts the previous teacher run failed on.
    prompts = load_prompts(40 + arguments.teacher_prompts, seed=0)
    policy_prompts, teacher_prompts = prompts[: arguments.policy_prompts], prompts[40:]

    policy_rows = (
        await measure_policy(policy, policy_prompts, arguments.policy_samples, temperature=1.0) if policy_prompts else []
    )
    print(f"policy spend so far ${meter.dollars():.3f}")
    teacher_rows = await measure_teacher(teacher, teacher_prompts, meter, arguments.drafts, arguments.repair_rounds)

    for name, rows in (("policy", policy_rows), ("teacher", teacher_rows)):
        with (run_dir / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    passing = sum(row["passes_gates"] for row in policy_rows)
    e_free = sum(row["e_free"] for row in policy_rows)
    groups_with_success = sum(
        any(row["passes_gates"] for row in policy_rows if row["prompt"] == prompt) for prompt in policy_prompts
    )
    accepted = [row for row in teacher_rows if row["accepted"]]
    print(f"\nPOLICY (unmasked, temp 1.0): {passing}/{len(policy_rows)} pass all gates, {e_free} e-free")
    print(f"  groups with >=1 passing answer: {groups_with_success}/{len(policy_prompts)}")
    print(f"TEACHER repair: {len(accepted)}/{len(teacher_rows)} accepted")
    print(f"  rounds used by accepted: {sorted(row['rounds'] for row in accepted)}")
    print(f"TOTAL estimated spend ${meter.dollars():.3f}")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
