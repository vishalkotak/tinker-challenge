"""Generate the SFT warm-start set: teacher lipograms that pass every gate and Wu's usable bar under the gold judge."""

import argparse
import asyncio
import json
import random
from pathlib import Path

import tinker

from lipo.chat import ChatModel, SpendMeter
from lipo.checks import check_answer
from lipo.judge import judge_many
from lipo.pilot import TEACHER_MODEL, measure_teacher
import lipo.pilot as pilot

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "sft_teacher.jsonl"
JUDGE_RENDERER = "gpt_oss_high_reasoning"


def training_prompts(count: int, seed: int) -> list[str]:
    """Sample prompts from Wu's RL pool; the pool has zero overlap with his 50 eval prompts (checked)."""
    eval_prompts = {json.loads(line)["prompt"].strip() for line in (DATA_DIR / "short_eval_50.jsonl").open()}
    pool = [json.loads(line)["prompt"] for line in (DATA_DIR / "rl_pool.jsonl").open()]
    # GOTCHA: re-filter against the eval set anyway, so a future pool edit can't leak eval prompts into SFT.
    pool = [prompt for prompt in pool if prompt.strip() not in eval_prompts]
    random.Random(seed).shuffle(pool)
    return pool[:count]


async def main() -> None:
    """Generate, gate, judge, and save SFT examples, stopping before the spend cap."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=700)
    parser.add_argument("--max-dollars", type=float, default=9.0)
    arguments = parser.parse_args()

    # GOTCHA: pilot's measure_teacher reads this module-level cap, so it must be raised before generation.
    pilot.MAX_SPEND_DOLLARS = arguments.max_dollars
    meter = SpendMeter()
    service = tinker.ServiceClient(user_metadata={"description": "lipogram SFT data: teacher repair + gold judge"})
    teacher = await ChatModel.create(service, TEACHER_MODEL, "gpt_oss_medium_reasoning", meter)
    judge_model = await ChatModel.create(service, TEACHER_MODEL, JUDGE_RENDERER, meter)

    prompts = training_prompts(arguments.prompts, seed=1)
    teacher_rows = await measure_teacher(teacher, prompts, meter, drafts=4, repair_rounds=6)
    accepted = [row for row in teacher_rows if row["accepted"]]
    print(f"teacher accepted {len(accepted)}/{len(prompts)}, spend ${meter.dollars():.2f}")

    judgments = await judge_many(judge_model, [(row["prompt"], row["answer"], False) for row in accepted])
    kept = []
    with OUTPUT_PATH.open("w") as handle:
        for row, verdict in zip(accepted, judgments):
            if not verdict.usable:
                continue
            record = dict(prompt=row["prompt"], answer=row["answer"], coherence=verdict.coherence,
                          responds=verdict.responds, source="gpt-oss-120b-repair")
            handle.write(json.dumps(record) + "\n")
            kept.append(record)
    print(f"judge kept {len(kept)}/{len(accepted)} as usable; total spend ${meter.dollars():.2f}")
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
