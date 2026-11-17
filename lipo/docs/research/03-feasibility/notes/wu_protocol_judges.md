# Kevin Wu's "Qwen, Minus the E" protocol, related artifacts, and gold judge options

## 1. Wu's eval protocol, training details, failure analysis, released artifacts, HN critiques

### Takeaway
Wu's headline number (60.5% usable, SFT then 300 GRPO steps, Qwen3-1.7B, always masked) comes from 50 validation prompts x 4 samples = 200 answers, a 300-token cap, and a gpt-6-luna judge that scores coherence and responsiveness from 1 to 5 (usable = both >= 4 plus a mechanical no-e check). Code, SFT data, run records and both LoRA adapters are public, so a mask-free 4B run can reuse his exact prompts and judge. There is one big caveat: every result Wu reports, including the 4B row, uses the decode-time mask. I found no HN thread.

### Cited Findings
**Eval protocol**
- Eval set: 50 prompts from a 200-prompt validation split, 4 sampled answers each, so 200 answers per condition. Prompts come from Dolly and No Robots, minus prompts that can't be answered without "e" (e.g. "Who painted the Mona Lisa?"). Persona-style No Robots chats and code requests were removed. — [Wu blog](https://kevin-wu.ca/blog/qwn)
- The 50 are "the first 50 of 75 validation prompts that can be answered without e". The 25 dropped prompts are logged with reasons. — [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- Length cap is 300 tokens (about 200 words). The system message tells the model to avoid "e" and "E" and is used in both training and eval. — [Wu blog](https://kevin-wu.ca/blog/qwn)
- Sampling: the blog gives temperature 0.7 for RL but no eval temperature or top_p. The experiment notes use temperature 0.7, thinking off and seed 0 for chat comparisons, and 4 seeds per prompt for the short-answer eval. Top-k 20 / top-p 0.8 were tested but are not stated as the eval default. The SFT adapter degrades at temperature 1.0. — [Wu blog](https://kevin-wu.ca/blog/qwn); [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- Mask: logits are set to -inf for every token containing "e". The mask also blocks "E" and non-ASCII characters, so look-alikes such as "é" cannot appear. — [Wu blog](https://kevin-wu.ca/blog/qwn)
- Some earlier runs used a looser mask. The stricter ASCII-only mask removed non-ASCII typography, which 44 of 95 rewrites contained. — [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- "Usable" requires all three of: (1) no e/E, including accented forms, checked in code; (2) answers the prompt; (3) reads as normal English with correct spelling and sensible ideas. — [Wu blog](https://kevin-wu.ca/blog/qwn)
- Judge: `gpt-6-luna` (OpenAI API) with reasoning effort `none`, max_tokens 1000, 2 retries, 8 workers, and no temperature set. It scores Coherence and Responds as integers 1 to 5, with `PASS_SCORE = 4` on both. The prompt never mentions the letter e. Factual accuracy is excluded. Repetition caps coherence (three or more repeats caps it at 3). Truncated answers are judged on the text present and carry a "cut off at the length limit" label. The judge replies in JSON, and request and answer are wrapped in `<<<` / `>>>`. — [core/judge.py](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/core/judge.py)
- Judge calibration: unmasked Qwen3-1.7B with no e rule scores 80.5% on coherence and responsiveness alone (Wu's target was over 80%). There was no human agreement check; earlier reviews were subagent reviews. Against reference sets the judge gives 98-99% usable on verified answers, 89-91% coherence agreement on 64 answers, and 0/48 on failed RL outputs. Two borderline answers flipped verdict between judge v1 and v2. — [Wu blog](https://kevin-wu.ca/blog/qwn); [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- Confidence intervals: paired bootstrap, 1,000 resamples over the 50 prompts, reporting the middle 90%. — [Wu blog](https://kevin-wu.ca/blog/qwn)

**Training**
- Qwen3-1.7B, LoRA rank 16, on a local 16 GB GPU. SFT uses 180 e-free answers to training-split prompts, written by frontier models and then corrected (`data/rewrites/verified_200_20261015.jsonl`). Each SFT run stops early when validation loss starts rising. — [Wu blog](https://kevin-wu.ca/blog/qwn)
- GRPO-style RL: 4 prompts x 8 answers per step, 300 steps (about one pass over a 1,235-prompt pool), thinking off, temperature 0.7, max 300 tokens, LR 2e-5, KL weight 0.02, one LoRA update per step, eval every 25 steps. Reward = (coherence-1 + responds-1)/8, in [0,1]. Advantage = (reward - group mean)/group std. — [Wu blog](https://kevin-wu.ca/blog/qwn)

**Results (usable, n=200)**
- Unmasked + instruction: 0%. Masked, no training: 26%. Masked + Gadsby SFT (about 50k words): about 26%. SFT on 15/30/45/90/135/180 examples: 10/20.5/22.5/31.5/34/35%. Masked + 300 RL steps: 43% (+17, 90% CI +8 to +26). SFT + 300 RL steps: **60.5%** (121/200; +25.5, CI +17.5 to +33.5). Most of the gain arrives by about step 125. The SFT run peaks at 61.5% at steps 225 and 250. — [Wu blog](https://kevin-wu.ca/blog/qwn)
- 4B comparison (Qwen3-4B, not 3.5): 87.5% on coherence and responsiveness alone, 0% with instruction only, and 12.5% with instruction plus mask (lower than 1.7B's 26%). No trained 4B result is reported. — [Wu blog](https://kevin-wu.ca/blog/qwn)
- Coverage: on 50 new prompts with 16 tries each, at least one usable answer appeared for 78% of prompts (masked) and 64% (SFT adapter). — [Wu blog](https://kevin-wu.ca/blog/qwn)

**Failure analysis**
- Instruction alone: the model used "e" in every answer. Mask alone produces nonsense, e.g. repeated "riddiculous". — [Wu blog](https://kevin-wu.ca/blog/qwn)
- RL on the e rule alone (no judge, no mask) produced reward hacks: streams of letters that aren't words, and lines like "i want to know what you want to know." Stronger KL penalties and different discount factors gave nonsense or a single repeated sentence. — [Wu blog](https://kevin-wu.ca/blog/qwn)
- The notes report no controlled mask-free RL run with the judge as reward. The early prototypes collapsed, but their configs weren't preserved. — [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- Answer shrinkage: mean tokens fell from 92 to 42 (base to base+RL) and from 70 to 40 (SFT to SFT+RL). Answers under 25 tokens rose from 51 to 124 (base) and from 17 to 102 (SFT). For the masked base, all of the gain came from coherence, and clean half-answers rose from 24 to 66. Two fixes, a min-score reward and dividing loss by the 300-token cap, did not raise the usable rate. Median length at step 150 was 30-38 tokens, versus 53 before RL. The notes blame the summed reward (5+3 equals 4+4) and GRPO's per-length loss normalization. — [Wu blog](https://kevin-wu.ca/blog/qwn); [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- Judge leniency: one-word answers ("SNCF", "football") got coherence 5. A non-answer about "cliché" also got coherence 5. The SFT adapter sometimes swaps in a different e-free word and answers about that ("clik" for "cliché"). — [Wu blog](https://kevin-wu.ca/blog/qwn)

**Released artifacts**
- Code: https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj (judge `core/judge.py`, SFT data `data/rewrites/verified_200_20261015.jsonl`, `runs/`, `docs/experiment_history.md`). Adapters, both of which need the mask: https://huggingface.co/k33wu/qwen3-1.7b-no-e (SFT+RL) and https://huggingface.co/k33wu/qwen3-1.7b-no-e-sft180. — [Wu blog](https://kevin-wu.ca/blog/qwn)

**HN**
- A search for an HN thread on "Qwen, Minus the E" found none. The candidate item 48966120 turned out to be a "Qwen 3.8" release thread with no relevant methodology discussion. — [HN item 48966120](https://news.ycombinator.com/item?id=48966120)

### Inferences
- For apples-to-apples: use the same 50 prompts x 4 samples, the 300-token cap, temperature 0.7 with thinking off, Wu's judge prompt verbatim, and his usable rule. Report the paired-bootstrap 90% CI. Use gpt-6-luna with reasoning off if possible. Otherwise, re-judge Wu's released adapters' outputs with whatever judge you use, so both sides share a judge.
- A mask-free model must pass a check at least as strict as Wu's mask: no e/E, no accented e, and an ASCII-only output. Otherwise it gains an advantage the masked model can't have.
- Report median length and the under-25-token count next to the usable rate. Wu's own judge rewards short clean answers, so a new model could "win" through shrinkage alone.
- With n=200 and CI widths of about ±8 points, differences under about 10 points against 60.5% are not significant.
- Wu's 4B row is Qwen3-4B, not Qwen3.5-4B. No 4B trained baseline exists, so a Qwen3.5-4B comparison is cross-size and cross-generation unless you also run the 1.7B.

### Gaps
- The exact eval-time temperature/top_p and the judge temperature are not explicitly stated (likely 0.7 and the API default).
- The verbatim judge prompt was not extracted; it is available in `core/judge.py`.
- The exact code for the e check (does it reject digits? which accented characters?) was not located. The judge file takes a precomputed `e_ok` flag.
- No HN thread or external critiques were found.

## 2. prateekfr/qwen3-1.7b-no-e-lora

### Takeaway
This is an apparent replication or derivative of Wu's recipe (mask, then SFT on "180 rewrites", then GRPO on Qwen3-1.7B) that claims 88% usable. The card doesn't document the eval set, the judge, or the computation, so the claim is not comparable to Wu's 60.5%.

### Cited Findings
- Method: a pre-computed boolean mask blocks 103,639 tokens (68.33% of the Qwen3 vocab; any token whose lowercased text contains "e"). SFT uses 4-bit QLoRA on "180 curated, human-verified e-free instruction pairs". GRPO optimizes "fluency, length consistency, and lexical diversity". — [HF card](https://huggingface.co/prateekfr/qwen3-1.7b-no-e-lora)
- Results on the "Gadsby Benchmark Suite" (strict pass / usable / n-gram diversity): Base + mask 88/28/91.5%, SFT + mask 96/80/91.5%, RL + mask 100/88/92.5%. — [HF card](https://huggingface.co/prateekfr/qwen3-1.7b-no-e-lora)
- The card says unaligned base models "attempt foreign-language token substitutions to evade the constraint". The usable rate "reflects pure English ASCII responses". — [HF card](https://huggingface.co/prateekfr/qwen3-1.7b-no-e-lora)
- The card gives no eval set size, judge, reward weights, step counts, training code, date, or mention of Wu. — [HF card](https://huggingface.co/prateekfr/qwen3-1.7b-no-e-lora)

### Inferences
- 88% strict pass under a mask suggests the mask leaks, plausibly through non-ASCII or e-variant tokens, since it lowercases but doesn't block accented forms. That matches the "foreign-language token substitutions" note and shows why an ASCII whitelist matters.
- The "180 rewrites" count matches Wu's SFT set, so it likely reuses Wu's data. Treat the result as an unverified, informal data point, not a baseline.

### Gaps
- Benchmark size and composition, judge model, and definition of "usable" are unknown.

## 3. Gold judge options on Tinker and Liu et al. 2603.12246

### Takeaway
Tinker serves many different-family models for sampling: gpt-oss-120b/20b, DeepSeek-V4.1-Flash, Kimi K2.6/K3.1-Pro, GLM-5.3, Nemotron 3/3.5, and Inkling, plus Qwen3.5-397B/Qwen3.6-35B/Qwen3.8-27B. Llama is retired. Liu et al. used gpt-oss-120b (high reasoning) as the gold judge, so it is the natural, cheap ($0.84/M sampled tokens), non-Qwen gold choice. A frontier API judge (gpt-6-luna, as Wu used) can serve as a second check.

### Cited Findings
**Tinker lineup (Models & Pricing page, prices per M tokens: prefill / sample)**
- Qwen3.5-4B (dense, Hybrid+Vision, `Qwen/Qwen3.5-4B`): $0.33 / $1.005; train $0.737. Qwen3.5-9B, Qwen3.5-9B-Base, and Qwen3.5-35B-A3B-Base are also available. — [Tinker Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- Qwen3.5-397B-A17B: $1.65 / $7.50. Qwen3.6-35B-A3B: $0.45 / $1.335. Qwen3.8-27B: $0.6975 / $5.595. Qwen3-8B: $0.195 / $0.60. The Qwen3.5-35B-A3B instruct model was retired Jun 12, 2026 and replaced by Qwen3.6-35B-A3B. — [Tinker Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/); [Tinker deprecations](https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/)
- Different families: GPT-OSS-120B (reasoning, MoE 117B/5B active, 32K): $0.33 / $0.84. GPT-OSS-20B: $0.18 / $0.45. DeepSeek-V4.1-Flash (552B/16B, hybrid, 256K): $0.60 / $3.60. DeepSeek-V3.1 retired Oct 23. Kimi-K2.6 (1T/32B): $2.205 / $5.49. Kimi-K3.1-Pro (2.8T, reasoning): $6.00 / $25.00. GLM-5.3: $3.78 / $12.15. GLM-5.3-Flash: $0.84 / $2.11. Nemotron-3-Super-120B-A12B: $0.57 / $1.44 (50% discount, hybrid). Nemotron-3-Ultra-550B: $1.80 / $6.225 (discounted). Nemotron-3.5-Lightning-30B-A3B: $0.184 / $0.495 (discounted). Inkling 2 Small (594B/17B): $0.30 / $1.00. Inkling-Small: $0.58 / $1.44. Inkling: $1.87 / $4.68. — [Tinker Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- Llama: all Llama models (3.1-8B/70B, 3.2-1B/3B, 3.3-70B-Instruct) retired Jun 12, 2026. — [Tinker deprecations](https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/)
- Serverless inference (beta) is available only for the Inkling models. Others are sampled through Tinker sampling clients. — [Tinker Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)

**Liu et al., arXiv 2603.12246 ("Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training", Mar 12, 2026)**
- Gold judge: gpt-oss-120b at high reasoning effort, temperature 0, up to 8,192 tokens. It was chosen for open weights, visible reasoning, and strength, and Qwen3 was avoided as gold because the trained judges were Qwen3 (self-bias). — [arXiv HTML](https://arxiv.org/html/2603.12246)
- Non-reasoning judges get hacked at every size (1.7B-14B). The policy hits near-max training reward (9) while gold score falls later. Larger judges delay the collapse. KL 0.001-0.1 did not prevent it, and rubrics improved agreement but didn't stop hacking. — [arXiv HTML](https://arxiv.org/html/2603.12246)
- Reasoning judges distilled from gold reasoning keep improving gold scores, with the gains emerging late (about 700-1,000 steps). RL-only reasoning judges get hacked like non-reasoning ones. An untuned Qwen3-4B judge gives limited gold gains. — [arXiv HTML](https://arxiv.org/html/2603.12246)
- Static agreement with gold doesn't predict policy-training usefulness. Reasoning effort raises agreement: alpha 79.88 at low, 85.99 at medium, 89.34 at high. — [arXiv HTML](https://arxiv.org/html/2603.12246)
- Hacking transfers. The learned adversarial pattern (fake "platform policy" refusal, invented policy, fake end marker, self-assessment) fooled the gold judge despite prompt changes. On Arena-Hard-V2 creative writing it scored 89.6 under a GPT-4.1 judge (o3 scored 92.4) but only 39.1 on hard prompts. GPT-4.1 treats injected "requirements" as real. — [arXiv HTML](https://arxiv.org/html/2603.12246)
- Caveats: "gold" is itself an LLM; the main policy is Llama-3.1-8B; the test set has 1K examples; and the curves appear only in figures. — [arXiv HTML](https://arxiv.org/html/2603.12246)

### Inferences
- Recommended setup: proxy/training judge = whatever is cheap (e.g. a Qwen model or a Wu-style non-reasoning judge). Gold = gpt-oss-120b at high reasoning on Tinker ($0.84/M sampled; judging 200 answers with about 1K reasoning tokens each costs roughly $0.20 per checkpoint). Its family differs from the Qwen3.5-4B policy, which avoids self-preference. As a second gold check, use a different family again (DeepSeek-V4.1-Flash, Kimi-K2.6, or Nemotron-3-Super) to test whether hacks transfer.
- Avoid Qwen3.5-397B/Qwen3.6/Qwen3.8 as gold for a Qwen policy, following Liu et al.'s self-bias reasoning.
- For overoptimization curves, plot proxy reward against gold usable rate per checkpoint (every 25 steps, as Wu did). Liu et al. warn that divergence can appear late, so run past Wu's 300 steps if budget allows.
- Use mechanical checks, not the judge, for the letter constraint. Liu et al.'s hack fooled even the gold judge, so treat rule-based detectors as the real gold for the lipogram part.

### Gaps
- No Tinker-specific documentation on judge latency or rate limits was found.
- Liu et al. don't test lipogram or constraint tasks, and their curve values are only in figures.
- gpt-oss-120b's context on Tinker is 32K (128K variant exists). That is fine for judging.

## 4. Lipogram-specific reward hacks and simple detectors

### Takeaway
Documented hacks include Cyrillic look-alikes (reported in Approximately Aligned Decoding), foreign-language token substitution (Prateek card), letter streams and degenerate repetition (Wu, no mask), one-word or half answers (Wu, with mask), and word-substitution drift ("clik" for "cliché"). An ASCII whitelist, a dictionary OOV rate, and length and repetition stats catch most of these cheaply.

### Cited Findings
- Approximately Aligned Decoding (Melcer et al.) checked lipogram outputs for violations of the constraint's intent; its flagged examples reportedly included Cyrillic lookalike swaps. This came from a search summary only, and I could not read the PDF to confirm the rubric or rates. — [arXiv 2410.01103](https://arxiv.org/abs/2410.01103)
- Homoglyphs are visually similar characters with different encodings, such as Latin vs Cyrillic "a". They are used to evade detectors. — [SilverSpeak, arXiv 2406.11239](https://arxiv.org/abs/2406.11239)
- The "Gatsby without the E" lipogram paper measures only an E-Score (count of the forbidden letter), which would miss look-alike substitutions. — [arXiv 2505.20501](https://arxiv.org/abs/2505.20501)
- Unaligned base models "attempt foreign-language token substitutions to evade the constraint". — [prateekfr HF card](https://huggingface.co/prateekfr/qwen3-1.7b-no-e-lora)
- Wu observed letter streams that aren't words, a repeated generic sentence, the misspelling "riddiculous" (mask only), the substituted word "clik", one-word answers, and truncation-like shrinkage. — [Wu blog](https://kevin-wu.ca/blog/qwn)
- Wu blocks non-ASCII and accented e at decode time, and his judge caps coherence for repetition. — [Wu blog](https://kevin-wu.ca/blog/qwn); [core/judge.py](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/core/judge.py)

### Inferences
- Detectors, all in a few lines of Python:
  1. ASCII whitelist (reject any codepoint > 127, or allow only printable ASCII). This catches Cyrillic е (U+0435), é, full-width Ｅ, and language switching.
  2. Run NFKD normalization plus a confusables map before the e check, as a belt-and-braces step.
  3. Case-insensitive `[eE]` check.
  4. Dictionary OOV rate: tokenize words, look them up in an English wordlist (e.g. SCOWL or a `wordfreq` threshold), and flag answers where more than about 5% of alphabetic tokens are out of vocabulary. This catches dropped-letter misspellings ("Gatbsy", "riddiculous", "clik") and letter streams. Whitelist proper nouns from the prompt.
  5. Length floor and ratio to the reference answer (catches shrinkage and truncation), plus the fraction of answers under 25 tokens, matching Wu's stat.
  6. Distinct-n / repeated n-gram rate (catches repetition loops).
  7. Digit ratio. Digits can stand in for e-words ("3" for "three"); flag a high digit share or digits the prompt doesn't call for.
- Give the gold judge the OOV and ASCII flags, or AND them into "usable", so misspellings can't pass on judge leniency. Wu's judge already requires correct spelling but is an LLM and can miss it.

### Gaps
- I found no published lipogram RL paper that quantifies hack rates for homoglyphs, digits, or misspellings. The "Gatbsy" example and the digit hack are hypotheses in the prompt, not documented in the sources found.
