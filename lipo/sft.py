"""SFT warm start: fine-tune Qwen3.5-4B (LoRA) on gate-passing, judge-approved lipograms, saving a checkpoint per epoch."""

import argparse
import asyncio
import json
import random
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from lipo.checks import check_answer
from lipo.pilot import NO_E_INSTRUCTION

DATA_DIR = Path(__file__).parent / "data"
RUNS_DIR = Path(__file__).parent / "runs"
BASE_MODEL = "Qwen/Qwen3.5-4B"
RENDERER = "qwen3_5_disable_thinking"
CHECKPOINT_TTL_SECONDS = 14 * 24 * 3600


def load_examples() -> list[tuple[str, str]]:
    """Combine our teacher lipograms with Wu's 200 verified answers, keeping only ones that pass every gate."""
    eval_prompts = {json.loads(line)["prompt"].strip() for line in (DATA_DIR / "short_eval_50.jsonl").open()}
    pairs = [(row["prompt"], row["answer"]) for row in map(json.loads, (DATA_DIR / "sft_teacher.jsonl").open())]
    pairs += [(row["prompt"], row["response"]) for row in map(json.loads, (DATA_DIR / "verified_200_20261015.jsonl").open())]
    # GOTCHA: gate every example (including Wu's) with the same checks RL uses, so SFT never teaches a
    # pattern the RL reward would later punish, and drop any eval prompt as a last leak guard.
    return [(prompt, answer) for prompt, answer in pairs
            if prompt.strip() not in eval_prompts and check_answer(answer, prompt).passes_gates]


def to_datum(prompt: str, answer: str, renderer: renderers.Renderer) -> tinker.Datum:
    """Render one (prompt, answer) conversation, training only on the assistant reply."""
    # GOTCHA: train with the same no-e system prompt the eval uses, so the warm start conditions on it.
    messages = [
        {"role": "system", "content": NO_E_INSTRUCTION},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    return conversation_to_datum(messages, renderer, max_length=1024, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)


async def main() -> None:
    """Train for a few epochs and print each epoch's sampler checkpoint path and mean loss."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="sft")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4.9e-4, help="cookbook get_lr() for Qwen3.5-4B LoRA")
    arguments = parser.parse_args()

    renderer = renderers.get_renderer(RENDERER, get_tokenizer(BASE_MODEL), model_name=BASE_MODEL)
    examples = load_examples()
    print(f"{len(examples)} training examples")
    service = tinker.ServiceClient(user_metadata={"description": f"lipogram SFT {arguments.name}"})
    training = await service.create_lora_training_client(base_model=BASE_MODEL, rank=arguments.rank, seed=0)

    run_log = []
    step = 0
    for epoch in range(arguments.epochs):
        random.Random(epoch).shuffle(examples)
        losses = []
        for start in range(0, len(examples), arguments.batch_size):
            batch = [to_datum(prompt, answer, renderer) for prompt, answer in examples[start : start + arguments.batch_size]]
            # GOTCHA: submit forward_backward and optim_step before awaiting either, so Tinker can pipeline
            # them in the same clock cycle; awaiting in between roughly doubles wall time.
            forward_backward = training.forward_backward(batch, "cross_entropy")
            optimizer_step = training.optim_step(types.AdamParams(learning_rate=arguments.learning_rate))
            output = await forward_backward
            await optimizer_step
            # GOTCHA: divide by the summed weights (trained answer tokens), not sequence length, or prompt
            # tokens dilute the loss and epochs with longer prompts look artificially better.
            trained_tokens = sum(sum(datum.loss_fn_inputs["weights"].data) for datum in batch)
            losses.append(output.metrics.get("loss:sum", 0.0) / max(trained_tokens, 1))
            step += 1
        saved = await training.save_weights_for_sampler(
            name=f"{arguments.name}-epoch{epoch + 1}", ttl_seconds=CHECKPOINT_TTL_SECONDS
        )
        state = await training.save_state(name=f"{arguments.name}-state-epoch{epoch + 1}", ttl_seconds=CHECKPOINT_TTL_SECONDS)
        record = dict(epoch=epoch + 1, steps=step, mean_loss=sum(losses) / len(losses),
                      sampler_path=saved.path, state_path=state.path)
        run_log.append(record)
        print(json.dumps(record))

    out_dir = RUNS_DIR / arguments.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints.json").write_text(json.dumps(run_log, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
