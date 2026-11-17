# Lipograms without a mask

Teaching Qwen3.5-4B to answer questions without the letter "e", with no decoding mask, on Tinker, for about $32.

Every published lipogram model I could find blocks e-tokens while decoding. Tinker's sampler can't do that, so this model has to avoid "e" on its own, from its weights.

## Results

Kevin Wu's eval ([Qwen, Minus the E](https://kevin-wu.ca/blog/qwn), [repo](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj)): his 50 prompts x 4 samples, 300 tokens, temp 0.7, his judge prompt word for word. Judge is gpt-oss-120b (high reasoning) for every row, including Wu's own answers, which I re-judged from his repo.

| Model | Usable (Wu's rule) | Strict usable | No "e" | Median words |
|---|---|---|---|---|
| Qwen3.5-4B, instruction only | 3.0% | 0.0% | 3.5% | 27 |
| + SFT on 693 lipograms | 27.0% | 27.0% | 39% | 76 |
| + RL run A (step 60) | 48.0% | 46.0% | 81% | 100 |
| + RL run C (step 100) | 58.0% | 57.5% | 84% | 76 |
| **+ RL run D (step 25), mean of 3 evals** | **63.7%** | **63.5%** | 86% | ~68 |
| Wu, Qwen3-1.7B, SFT + GRPO, **with mask** | **68.0%** | 40.5% | 100% | 19 |

- **Usable (Wu's rule):** no e of any kind, and the judge gives 4+ for both coherence and "responds to the request".
- **Strict usable:** the same, plus ASCII only, 12+ real words, no dropped-e spellings ("th" for "the"), and no loops.

So:

- **On Wu's own rule I lose, 63.7% vs 68.0%.** That's inside the noise (his 90% interval is 62.5 to 73.5), but I'm not calling it a win.
- **On strict usable I win by 23 points, 63.5% vs 40.5%,** with intervals that don't overlap. Wu's rubric scores "Both." or "Football." as a perfect answer, and 32% of his answers are under 12 words. Mine never are.
- The final checkpoint was picked on 200 dev prompts the model never saw, not on the eval set. All three eval runs are reported (66.5, 62.5, 62.0), not the best one.

## How it works

1. **Checks first** (`checks.py`): strict e-check, ASCII only, a dictionary gate, dropped-e detection, length and loop checks. They pass 196 of Wu's 200 hand-verified answers and reject every hack I tried.
2. **Teacher data** (`make_sft_data.py`): gpt-oss-120b writes 4 drafts, then gets told the exact words that contain "e" and rewrites, for up to 6 rounds. 697 of 700 prompts came out clean, about $0.012 each. I kept the 497 the judge rated usable, plus 196 of Wu's.
3. **SFT** (`sft.py`): LoRA rank 32, 3 epochs. 27% usable.
4. **RL** (`rl.py`): importance-sampling policy gradient, 8 prompts x 8 samples per step. The reward is:
   - zero if a hack gate fails;
   - otherwise judge quality, mapped 5 → 1.0, 4 → 0.8, 3 → 0.15;
   - times max(0, 1 - 0.5 per e), times a soft length penalty past 110 words;
   - plus a -0.2 advantage on every sampled token that contains "e".

## What didn't work, in order

- **Plain RL from the base model.** 0 of 640 base samples passed the gates. The only e-free ones were 1 to 3 words, so RL had nothing to learn from.
- **The first teacher setup.** Low reasoning and 3 repair rounds got only 7 of 30 clean. Medium reasoning, 4 drafts and 6 rounds got 30 of 30.
- **Run A: half credit for a judge score of 3.** The model learned e-free rambling ("Thus... no doubt..."), and answers grew from 80 to 150 words. 48%.
- **Run B: quality fixed, too little e pressure.** 48.5%, with only 67.5% e-free.
- **Run C: both fixes combined.** Length blew up to 226 words mid-run, then the length penalty pulled it back. 58%.
- **Run D: train against the same judge I evaluate with.** Wu did this too. Jumped to 66.5% in 25 steps. The biggest single fix.
- **Run D, segment 2.** Another 50 steps collapsed to 41.5% on dev. Kept the earlier checkpoint.
- **Ideas killed before running.** The `docs/research/` folder has four rounds of reading and attempts to kill ideas, including a KL-to-optimum benchmark that turned out to be done already (Zhao et al. 2024) and impossible to estimate at this length.

The leftover failures are prompts built around e-heavy names, like Verona, Newton, Athens, Forrest Gump and "Saint-Germain-les-Belles". The model writes around them and the judge docks it for relevance.

## Try it

The best checkpoint is `tinker://d1353f6d-9f46-531a-8ae3-a94af21000e8:train:0/sampler_weights/rl_d1-step25`. Paste it into the [Tinker Playground](https://tinker.thinkingmachines.ai/playground) with the system prompt from `pilot.py`.

## Reproduce

```bash
uv sync
uv run tinker auth login
uv run python -m lipo.test_checks                 # free
uv run python -m lipo.pilot                       # ~$0.45: base rate + teacher loop
uv run python -m lipo.make_sft_data               # ~$8.6
uv run python -m lipo.sft --name sft_r32          # ~$0.4
uv run python -m lipo.rl --name rl_c --init-state <sft state> --steps 100 \
  --learning-rate 3e-5 --token-penalty 0.2 --e-penalty 0.5 --save-every 25
uv run python -m lipo.rl --name rl_d1 --init-state <rl_c step-100 state> --steps 25 \
  --learning-rate 2e-5 --token-penalty 0.2 --e-penalty 0.5 --judge-renderer gpt_oss_high_reasoning
uv run python -m lipo.evaluate --name mine --model-path <sampler path>   # ~$0.18 per eval
```

Every run's logs, rollouts and judged eval answers are in `runs/`.

## Cost

About $32, from my own token log. The Tinker console matched it within about a dollar when I checked.

| Item | Cost |
|---|---|
| Pilot | $0.45 |
| Teacher data | $8.60 |
| SFT | $0.40 |
| RL runs A to D | ~$17 |
| ~24 evals | ~$5.30 |

## Credits

- Kevin Wu for the protocol, judge prompt, verified answers and prompt pool (MIT).
- The prompts come from [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC BY-SA 3.0) and [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (CC BY-NC 4.0), so `data/` is for non-commercial use.
- The word list is [google-10000-english](https://github.com/first20hours/google-10000-english).
