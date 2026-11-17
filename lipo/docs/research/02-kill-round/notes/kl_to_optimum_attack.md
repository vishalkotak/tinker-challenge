# Adversarial check: Candidate A (rank RL recipes by KL to the KL-regularized optimum) and A2 (answer shrinkage is partly optimal)

Research date: 2026-11-16. Read-only. About 15 tool calls. Where a source is from my training knowledge and not re-fetched this session, I say so.

## Q0. What the challenge actually is (this undercuts the framing)

### Takeaway
The official challenge reward is a capped, length-normalized e-RATE penalty, not an e-COUNT. It has no KL penalty and trains for 10 steps. A target built on e-count is the candidate's own invention, and it hard-codes length shrinkage into the target.

### Cited Findings
- `lipogram.py` in the official repo: `BASE_MODEL = "Qwen/Qwen3.5-4B"`, `LORA_RANK = 8`, `GROUP_SIZE = 8`, `TRAINING_STEPS = 10`, `MAX_TOKENS = 200`, `LEARNING_RATE = 8e-4`, sampling at temperature 1.0, thinking disabled — [tinker-challenge/lipogram.py](https://github.com/thinking-machines-lab/tinker-challenge) (raw file fetched 2026-11-16)
- Reward: `e_penalty = min(1.5, e_rate(text) / 0.10)`, where `e_rate` = count of "e" / number of alphabetic characters. The judge term `JUDGE_WEIGHT * grade/5` is present, but the grade is hard-coded to 0 ("commented out"). So the reward is a capped penalty on the e fraction, independent of length — [lipogram.py](https://github.com/thinking-machines-lab/tinker-challenge)
- Advantage: group mean baseline divided by group std (GRPO-style). It is broadcast to every token. The loss is `loss_fn="importance_sampling"`. Groups with identical rewards are skipped — [lipogram.py](https://github.com/thinking-machines-lab/tinker-challenge)
- The README lists a KL penalty only as a "Hard" extension: "try adding a KL penalty to the advantage", with no coefficient — [tinker-challenge README](https://github.com/thinking-machines-lab/tinker-challenge)
- Thinking Machines also used the "never uses the letter e" lipogram as the demo target in the Inkling launch — [Introducing Inkling](https://thinkingmachines.ai/news/introducing-inkling/)
- A community Qwen3-1.7B no-e model combines a logit mask, SFT and GRPO, and self-reports 100% strict pass with the mask on. So mask+RL pipelines are already public — [prateekfr/qwen3-1.7b-no-e](https://huggingface.co/prateekfr/qwen3-1.7b-no-e)

### Inferences
- Default runs have β = 0, so there is no π* to compare against. The analyst must pick β after the fact, and each recipe must either be retrained with a KL term or be scored against a β it never optimized. The ranking will depend on that choice.
- If the target uses e-count, then π*(length) ∝ p_base(length)·E[exp(-c·count) | length]. That decays roughly like exp(-c·0.12·letters). A2's headline ("shrinkage is partly optimal") then holds by construction and teaches nothing. Under the official rate reward, length is not penalized directly, so A2 has to be re-derived (see Q2).

### Gaps
- I did not check whether the repo history or leaderboard ever used an e-count reward.

## Q1. Prior art: evaluating LM fine-tunes by (bounded) KL to the KL-regularized / target optimum

### Takeaway
This has been done. Zhao et al. (ICML 2024) already report bidirectional KL to the target for PPO, DPG and several twisted-SMC methods, using BDMC/SMC log Z sandwiches. The DPG/GDC line (2021-2023) reports KL(p||π) to an explicit EBM target for REINFORCE, Ziegler-style KL-penalized RL and DPG on word-constraint tasks. What remains new is only the lipogram task and the modern recipe zoo (CISPO, GSPO, OPD).

### Cited Findings
- Zhao, Brekelmans, Makhzani, Grosse, "Probabilistic Inference in Language Models via Twisted SMC" (arXiv 2404.17546, ICML 2024): bidirectional SMC bounds on log Z. The bound gap bounds the symmetrized KL. Sec. 5.1 is about "evaluating fine-tuned models", and Table 1 casts PPO as minimizing KL(q||σ) and DPG as minimizing KL(σ||q) — [arXiv 2404.17546](https://arxiv.org/html/2404.17546)
- Same paper, results reported as KL(q||σ) / KL(σ||q):
  - Toxicity (TinyStories-33M): PPO 0.98/1.32, DPG 1.09/1.12, twisted CTL 1.11/1.07.
  - Sentiment (GPT2-Medium): PPO 1.04±0.31 / 0.87±0.20, CTL 0.55/0.47.
  - Infilling: DPG 13.27/4.90, PPO 19.37/14.07.
  - Upper bounds come from exact target samples via the BDMC trick (sample (s, o) jointly).
  - The toxic-threshold task "needed hundreds of thousands of rejection samples."
  - Source: [arXiv 2404.17546](https://arxiv.org/html/2404.17546)
- Korbak, Elsahar, Kruszewski, Dymetman, "On Reinforcement Learning and Distribution Matching for Fine-Tuning LMs with no Catastrophic Forgetting" (arXiv 2206.00761, NeurIPS 2022): uses D_KL(p, π_θ) to the target EBM as a validation metric for GDC++, GDC, Ziegler and REINFORCE on GPT-2 small. The pointwise tasks include single-word constraints ("amazing", "WikiLeaks"). It uses self-normalized importance sampling, and the baseline B = Z = E_π[P/π] — [arXiv 2206.00761](https://arxiv.org/html/2206.00761v2)
- Korbak, Perez, Buckley, "RL with KL penalties is better viewed as Bayesian inference" (EMNLP Findings 2022): KL-regularized RL is variational inference toward the posterior π* ∝ p_base·exp(r/β) — [arXiv 2205.11275](https://arxiv.org/abs/2205.11275)
- Go et al., "Aligning LMs with Preferences through f-divergence Minimization" (ICML 2023): RLHF is reverse KL to an implicit target, GDC is forward KL. They compare divergences across targets and find that "no single divergence is best" — [arXiv 2302.08215](https://arxiv.org/pdf/2302.08215)
- Hu et al., "Amortizing intractable inference in LLMs" (GFlowNet fine-tuning, ICLR 2024): distribution matching to an intractable posterior, compared against max-likelihood and reward-maximizing RL — [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bc667ac84ef58f2b5022da97a465cbab-Abstract-Conference.html)
- Shah et al., "A Comedy of Estimators: On KL Regularization in RL Training of LLMs" (arXiv 2512.21852, v4 dated 25 Aug 2026):
  - K1-in-reward has an unbiased gradient and performed best.
  - K3-in-loss (the GRPO setting) is biased but stable, and underperformed.
  - It notes high variance of sequence-level importance ratios for long sequences.
  - It does NOT measure KL to π* or log Z.
  - Source: [arXiv 2512.21852](https://arxiv.org/html/2512.21852)
- The rlhfbook regularization chapter states the closed form of π* and its partition function as textbook material — [rlhfbook ch. 15](https://rlhfbook.com/c/15-regularization)

### Inferences
- The core method (sandwich log Z, report KL to π* per fine-tuning method) is a 2024 ICML paper, co-authored by Grosse, who created BDMC. It is not new.
- A lipogram or letter-constraint version: I found none in the KL-to-target literature. The 2022 GDC word-presence constraints are the nearest cousin. This is a task delta, not a method delta.
- The token-mask decoding arm is also prior art. That locally constrained decoding distorts the global constrained distribution is known from grammar-aligned decoding (Park et al., NeurIPS 2024) and the SMC steering work (Lew et al. 2023; Loula et al., ICLR 2025). I cite these from training knowledge and did not re-fetch them.

### Gaps
- I did not find any 2025-2026 paper that ranks CISPO, GSPO or on-policy distillation by KL to π*. Not finding one does not prove none exists.
- I did not check AGRO or COPR this session.

## Q1b. Fatal technical point: ranking needs no log Z at all

### Takeaway
KL(π||π*) = log Z − [E_π r/β − KL(π||p_base)]. log Z is the same constant for every recipe, so the ranking by KL(π||π*) is exactly the ranking by the KL-regularized objective. That objective is the reward-vs-KL frontier everyone already plots. The bounds machinery only adds the absolute offset, and the offset is the infeasible part.

### Cited Findings
- The identity follows from π* ∝ p_base·exp(r/β), as in [Korbak et al. 2022](https://arxiv.org/abs/2205.11275) and [rlhfbook ch. 15](https://rlhfbook.com/c/15-regularization).

### Inferences
- Under this lens, "rank recipes by KL to optimum" is "rank recipes by E[r] − β·KL(π||base) at a chosen β". That is a single point on the reward/KL Pareto frontier (standard since Gao et al. 2022 and the DPO frontier plots; training knowledge). A reviewer will say this immediately.
- What would be non-trivial: absolute KL(π||π*) (how far from optimal in nats), plus forward KL(π*||π) for mode dropping. Both need log Z and/or target samples. See Q3 for why that fails in the interesting regime.

## Q2. Length bias: is KL-regularized-optimum shrinkage known? Objective vs optimizer decomposition?

### Takeaway
The optimizer side is well covered (Dr. GRPO and later papers on per-response normalization and length invariance). The objective side is trivial under an e-count reward. Under the challenge's e-rate reward it is subtle and possibly interesting, but small.

### Cited Findings
- Dr. GRPO (Liu et al., "Understanding R1-Zero-Like Training", arXiv 2503.20783, 2025) attributes length inflation to GRPO's per-response 1/|y| normalization and std normalization, and removes both. I cite this from training knowledge; the summary also appears in [arXiv 2509.09177](https://arxiv.org/pdf/2509.09177v1) and [arXiv 2510.04474](https://arxiv.org/pdf/2510.04474v2).
- "On the Impossibility of Unbiased and Length-Invariant Policy Optimization with Outcome Rewards" (arXiv 2607.23364, 2026): GRPO is approximately length-invariant but biased, and Dr. GRPO is unbiased but not length-invariant — [takara summary](https://tldr.takara.ai/p/2607.23364) (secondary summary, not the primary paper)
- DRPO-style work derives a closed-form KL-regularized target for length-based rewards in efficient reasoning — [arXiv 2510.04474](https://arxiv.org/pdf/2510.04474v2) (I did not verify the derivation)
- The challenge uses GRPO-style std normalization and the token-summed `importance_sampling` loss with a sequence-level advantage — [lipogram.py](https://github.com/thinking-machines-lab/tinker-challenge)

### Inferences
- Under an e-count target, length shrinkage being optimal is a one-line derivation: the penalty is roughly linear in letter count. Not publishable as a finding.
- Under the official e-rate target, my own derivation (not sourced): E[exp(−a·rate) | L] ≈ exp(−aμ + a²σ²_L/2), with σ²_L ∝ 1/L. So the optimum mildly favors SHORT answers, because short answers have more variance in their rate and the exponential tilt rewards variance. That is a real objective-induced shrinkage effect, but probably tiny at L ≈ 800 letters, and it can be computed analytically. A2 survives only in this reframed form.
- Confound: at MAX_TOKENS=200 many base answers to "Why is the sky blue?" probably hit the cap. "Shrinkage" may then partly be learned early EOS replacing truncation. Truncated samples are not natural samples from π*, so the cap has to be part of the target definition.

### Gaps
- No source quantifies how often base Qwen3.5-4B hits the 200-token cap on these prompts. It needs a measurement.

## Q3. Failure modes: importance-weight variance, IWAE samples, rejection-sampling acceptance, Tinker API

### Takeaway
In the regime that matters (e-rate pushed from about 12% to a few %), exact target sampling is astronomically rare. Log-weight variance under the base proposal is huge. The upper bound, and therefore the absolute KL and the forward KL, is infeasible there. The soft-target fix makes rejection sampling feasible only when the tilt is so weak the target is barely a lipogram. Tinker exposes enough for the ELBO, but has no logit mask for token-mask decoding.

### Cited Findings
- Tinker `sample()` returns per-token logprobs of sampled tokens. The challenge code reads `seq.logprobs` directly — [lipogram.py](https://github.com/thinking-machines-lab/tinker-challenge)
- `compute_logprobs(prompt, prompt_logprobs_last_n=None)` returns "a list of log probabilities for each token in the prompt". Entries can be None. Top-k logprobs are available via `topk_sample_logprobs` and `topk_prompt_logprobs`, and specific token ids via `target_prompt_logprobs`. Full-vocab logits are not documented — [Tinker SamplingClient docs](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md)
- `SamplingParams` has only max_tokens, seed, stop, temperature, top_k and top_p. There is no logit_bias or token mask — [Tinker SamplingParams](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/samplingparams/index.md)
- Tinker ships losses for importance_sampling, PPO, CISPO, DRO, GSPO and StableReinforce, plus on- and off-policy distillation recipes — [Tinker llms.txt](https://tinker-docs.thinkingmachines.ai/llms.txt)
- Zhao et al.: sequential importance sampling from the base proposal "often fails" without resampling or twists, and a thresholded target needed hundreds of thousands of rejection samples even for short TinyStories outputs — [arXiv 2404.17546](https://arxiv.org/html/2404.17546)
- Comedy of Estimators: sequence-level importance ratios have high variance for long sequences (qualitative only, no numbers) — [arXiv 2512.21852](https://arxiv.org/html/2512.21852)

### Inferences (my back-of-envelope, not sourced)
- 200 tokens is about 150 words, about 750 letters, about 90 "e" at a 12% rate.
  - Hard zero-e acceptance under the base is roughly 0.88^750, about e^−96. Infeasible.
- Soft count target with c = λ/β: acceptance = Z = E_base[exp(−c·count)], and the target mean shifts by about −c·Var(count).
  - Acceptance of 1e-3 or better needs c·90 ≲ 7, so c ≲ 0.08.
  - Var(count) is about 90 (binomial) plus topic and length variance, so the target only moves from about 90 e's to about 80. That is not a lipogram.
- Soft rate target: the base std of the rate per answer is about 1-2 points (binomial √(.12·.88/750) ≈ 1.2 points, plus topic spread).
  - Matching what RL reaches (rate about 2-4%) needs a shift of 5-8 std.
  - Rejection log-acceptance is then about −(shift²/2) ≈ −12 to −32 nats, so 1e5 to 1e14 base samples per exact target sample. Infeasible at the interesting β.
- BDMC's "exact posterior sample" trick (sample y from the base, then o ~ Bernoulli(exp(−c·e))) is exactly rejection sampling when there is one observation. It gives no escape.
- IWAE lower bound with the trained policy as proposal: the bound is tight only if K ≳ exp(Var[log w]).
  - Var[log w] equals Var over π of (log p_base − log π − c·e). For a LoRA policy that moved the e-rate 10 points over 750 letters, I expect it to be several nats², possibly tens.
  - So the IWAE gain over the ELBO will be visible but not certifiably tight. Without an upper bound you cannot tell how loose it is.
- What is cheap and sound: the per-recipe ELBO, E_π[−c·e + log p_base − log π], from sampled logprobs plus base `compute_logprobs` on (prompt + response).
  - Each sample costs one prefill. A few hundred samples per prompt give a small SE.
  - Caveat: sampler-vs-trainer logprob numerical mismatch. Score both π and the base with `compute_logprobs` for consistency.
- Token-mask decoding cannot be done through Tinker sampling (there is no mask). You would need token-by-token calls with top-k logprobs, or a local HF/vLLM copy of Qwen3.5-4B.
  - Its π_mask(y) is computable in principle, but the allowed-token mass at every step needs the full-vocab sum. top-k gives only an approximation.
- Weight variance also depends on prompt. Report it per prompt: the challenge uses a handful of prompts, and averaging over them hides the variance.

### Gaps
- No measured log-weight variance for Qwen-scale LoRA policies at 200 tokens. It must be measured.
- I did not confirm whether Tinker documents sampler/trainer logprob mismatch.
- No measured base e-rate distribution for Qwen3.5-4B on the challenge prompts.

## Q4. Would the result be interesting?

### Takeaway
Probably not, in its current form. The ranking reduces to the reward-KL frontier at a chosen β, which is known methodology and depends on β. The absolute numbers that would be novel are infeasible to certify in the regime where recipes differ. Expected differences across importance sampling, PPO, CISPO and GSPO at 10 steps with LoRA rank 8 are mostly about optimization speed and stability, not distributional fidelity.

### Cited Findings
- Prior comparisons do find non-trivial reorderings between reverse and forward KL (e.g. PPO best on forward toxicity KL but worst on infilling). Interesting differences tend to show up in the mode-dropping direction KL(π*||π) — [arXiv 2404.17546](https://arxiv.org/html/2404.17546)
- K1-in-reward vs K3-in-loss changes outcomes measurably — [arXiv 2512.21852](https://arxiv.org/html/2512.21852)

### Inferences
- The interesting direction (forward KL, mode dropping) is exactly the one that needs target samples. At strong tilt those are infeasible.
- On-policy distillation toward which teacher? There is no π* sampler, so OPD needs a stronger teacher, and that is not π*. SFT on which data? Rejection-filtered base samples are a sample from π* only at weak tilt. Both arms are ill-defined at the interesting β.

## Verdict: KILL as stated. A narrow reframing could survive.

Reasons:
1. The method is prior art: Zhao et al. ICML 2024 (BDMC/SMC bounds, PPO vs DPG KL to target) and Korbak et al. 2022 (KL to EBM for REINFORCE, Ziegler and DPG on word constraints).
2. Ranking by KL(π||π*) is identical to ranking by the regularized objective, so log Z bounds add nothing to the ranking.
3. The absolute or forward KL needs exact target samples. Acceptance is about e^−12 to e^−96 at lipogram-strength tilt, and the soft-target fix works only where the target is barely different from the base.
4. A2 is tautological under an e-count target. The real challenge reward is e-rate (capped at 1.5, no KL, β = 0), so the e-count target is not even the challenge's objective.
5. Token-mask decoding is not possible through Tinker sampling (no logit mask), and its distortion is known prior art.

Fix list (if pursued anyway):
- Use the challenge's actual e-rate reward, with the cap and the 200-token truncation included in π*.
- Retrain every recipe with an explicit KL-to-base term at fixed β (K1-in-reward, per Comedy of Estimators), so π* is the objective each recipe actually optimizes.
- Report the ELBO-based ranking honestly as "regularized objective". Sweep β and show where rankings cross; that crossing is the possibly novel plot.
- Restrict claims of absolute KL to weak-tilt β, where rejection sampling gives exact target samples (budget 1e3-1e5 base samples per prompt). Measure the base e-rate distribution first to pick β.
- At strong tilt, report only the ELBO/IWAE lower bound plus ESS and Var[log w]. Optionally run twisted SMC, using the best policy as proposal, for a better log Z estimate, but call it an estimate, not a bound.
- Reframe A2 as variance-driven shrinkage under a rate reward (exp tilt favors high-variance short answers), measured against GRPO-std and token-sum optimizer effects. Run Dr. GRPO-style ablations to split objective-induced from algorithm-induced length change.
- Run token-mask decoding locally (HF or vLLM), not through Tinker, and compute its exact local-normalization distortion.
