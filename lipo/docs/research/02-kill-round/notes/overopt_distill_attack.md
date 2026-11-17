# Adversarial check of candidates B (two-axis overoptimization curves) and D (masked-teacher on-policy distillation)

Context: Tinker Lipogram Challenge. Qwen3.5-4B on Tinker, no letter "e", Inkling-Small judge, about $25 budget. Research date 2026-11-16.

Verdicts up front:
- **B: SURVIVES, but narrowly and only with changes.** Nobody has published the exact design: Gao-form fits against KL, a judge proxy, and one exact reward component. The neighboring work is recent and close (Mar and Oct 2026). Kill risks: the "gold" judge is also gameable, and with a single proxy there is no scaling-law axis. Fix: do best-of-n first, use one RL run with checkpoints, and swap the proxy-size axis for something you can vary (judge reasoning effort, or the weight on e-rate).
- **D: SURVIVES only after it is reframed. As pitched, it is technically weak.** Tinker has no logit mask or logit processor. The per-token renormalizer does not matter for zero-discount on-policy distillation, so the teacher reduces to "raw base logprob, with -inf on e-tokens". The fixed point is then the locally constrained decoding (LCD) distribution, which is known to be myopic and biased. At full discount it collapses into KL-regularized RL with a hard constraint. Fix: present it as "LCD-base as a dense KL anchor plus a judge reward", compared against plain-base-KL RL.

## B prior art: Gao-style fits with LLM-judge proxies or multi-component rewards with one exact component

### Takeaway
No source found fits Gao et al. functional forms (gold vs sqrt(KL) for best-of-n and RL) using an LLM-judge proxy against a stronger LLM gold judge, with one exact reward component. The closest 2026 papers (Liu et al. 2603.12246; Wang 2610.15778) run the proxy-judge-vs-gold-judge setup but plot against steps or report correlations, and fit no Gao forms. The "one exact component" axis appears to be open. Moskovitz-style composite rewards are the nearest multi-component prior art.

### Cited Findings
- Gao, Schulman, Hilton (2022): a gold RM stands in for humans and proxy RMs are trained on its labels. Gold score rises and then falls with optimization (RL and best-of-n). The fitted coefficients vary smoothly, roughly log-linearly, with proxy RM size. — [arXiv 2210.10760](https://ar5iv.labs.arxiv.org/html/2210.10760)
- Rafailov et al. 2024 (direct alignment algorithms) use GPT-4 win rates as the gold proxy and report that the classical scaling-law form still fits. This is the earliest LLM-judge-as-gold use of the Gao forms found. — [arXiv 2406.02900](https://arxiv.org/pdf/2406.02900)
- Liu et al., "Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training" (submitted 2026-03-12):
  - Setup: gold judge gpt-oss-120b; proxy judges Qwen3 1.7B-14B; policies Llama-3.1-8B, Qwen2.5-7B, Qwen3-4B trained with GRPO for up to 1,200 steps on about 117K prompts, 8 samples per step, no KL by default; 1K held-out evaluation prompts.
  - Results are plotted against training steps, not KL. Gao is cited for the setup, but no Gao functional forms are fit.
  - A KL sweep from 0.001 to 0.1 did not prevent the gold-score decline.
  - Judges score pointwise 0-9, and the reward is the expected score over the judge's normalized token probabilities.
  - Policies trained against reasoning judges produce adversarial outputs that also fool other LLM judges on Arena-Hard.
  - The number of independent runs is not stated.
  - Sources: [arXiv abs](https://arxiv.org/abs/2603.12246), [HTML](https://arxiv.org/html/2603.12246v1)
- Wang, "Who Sets the Pace of Judge Hacking? Judge Capability versus the Policy-Judge Gap" (v1 2026-10-11, v2 2026-10-14):
  - Setup: Qwen2.5 0.5B and 1.5B policies trained with GRPO on GSM8K against frozen reference-free LLM judges.
  - Findings: exploitation depth correlates with judge AUC (Spearman -0.69 over 12 cells); no configuration gained more than 0.04 held-out accuracy. No KL x-axis and no Gao forms.
  - The abstract counts 12 cells, but the described setup totals 9. Seed counts are not stated.
  - Source: [arXiv 2610.15778](https://arxiv.org/abs/2610.15778)
- "The Verification Horizon" (Wang et al., 2026-06-24, coding agents) lists "Goodhart curves under RL pressure" only as future work. — [emergentmind 2606.26300](https://www.emergentmind.com/papers/2606.26300)
- Background from my own knowledge (not re-fetched this session):
  - Moskovitz et al. 2023, "Confronting Reward Model Overoptimization with Constrained RLHF" (arXiv 2310.04373): composite rewards built from several component RMs, each with its own "proxy point" where it stops correlating with gold. This is the nearest multi-component prior art, but all components are learned RMs.
  - Coste et al. 2023, "Reward Model Ensembles Help Mitigate Overoptimization" (arXiv 2310.02743): a Gao-style synthetic gold setup with ensembles, for both best-of-n and PPO.
  - Neither has an exact (programmatic) component.
- No Thinking Machines post on Inkling judge gaming was found. A search for one returned only third-party work. — [search results incl. LessWrong judge-hack post](https://www.lesswrong.com/posts/Kc7Tc6XbbNeRK3Wg9/when-a-claude-judge-recognizes-the-hack-but-still-says)

### Inferences
- The novel element of B is narrow: decomposing overoptimization when one axis (e-rate) is exact and cannot be overoptimized. This lets you test whether quality-judge hacking shows up as a trade against the exact axis, for example by exploiting judge leniency on e-free but stilted text. Given Liu and Wang, a reviewer would call the judge-proxy-vs-judge-gold part "done in 2026". Lead with the exact-axis decomposition instead.
- Gao's "scaling law" is about how the coefficients scale with proxy size. With one proxy (Inkling-Small) you get one curve, not a law. Fix: vary proxy strength along an axis you control:
  - Inkling-Small reasoning effort, which is adjustable according to aggregator sources.
  - The e-rate weight in the combined reward.
  - n for best-of-n, which is already the x-axis.
- Cost (inference):
  - Best-of-n is cheap because the KL is analytic: log n - (n-1)/n.
  - Example: 200 prompts × n=64 × about 200 tokens is about 2.6M sampled tokens, and 12.8K judge calls at about 500 input tokens is about 6.4M input tokens. At aggregator-listed Inkling-Small prices ($0.30/M input, $1.20/M output; [opper.ai](https://opper.ai/thinking-machines/inkling-small), unverified), that is about $2-4 for the proxy before reasoning tokens.
  - The gold judge (Inkling or Inkling 2 Small) costs more, and its price was not found.
  - RL curves do not need many runs. Gao's RL curves come from checkpoints along a single run per proxy, so one or two RL runs with about 8 checkpoints each, gold-scored on 200 prompts, should fit in $25 if rollouts are short.
- Sample counts for clean curves: Liu used 1K evaluation prompts. Wang's 12 configurations were not enough to separate the hypotheses. Neither reports Gao-quality fits, so the noise floor needed for a clean sqrt(KL) fit at this scale is unknown.

### Gaps
- Tinker per-token prices for Qwen3.5-4B sampling and training, and Inkling or Inkling 2 Small judge prices, were not verified on official pages.
- No 2025-2026 "reward hacking scaling laws" paper fitting Gao forms with LLM judges was found. The emergentmind aggregator mentions "Kim et al. 2025, γ metric", which I did not verify.
- Inkling model facts conflict:
  - Inkling-Small: a July 2026 release according to aggregators.
  - Inkling 2 Small: a 594B MoE with 17B active parameters, released 2026-11-05 with closed weights, according to [letsdatascience](https://letsdatascience.com/news/thinking-machines-lab-releases-inkling-2-small-a-594b-moe-mo-2b2a96ce) and [the-decoder](https://the-decoder.com/thinking-machines-bets-on-efficiency-over-size-with-its-second-model-inkling-small/).
  - Check which models the challenge actually exposes.

## B failure modes: gameable gold, integer judge noise, KL estimation

### Takeaway
All three are real, and all three have cheap fixes. The fatal one is treating an LLM gold judge as ground truth: Liu et al. show hacked outputs transfer across judges.

### Cited Findings
- Policies optimized against reasoning judges learned adversarial outputs that also fooled other LLM judges on Arena-Hard. Gold judges in the same family are therefore not independent. — [arXiv 2603.12246](https://arxiv.org/abs/2603.12246)
- Liu et al. use the expected score over the judge's score-token probabilities rather than the argmax integer. — [HTML](https://arxiv.org/html/2603.12246v1)
- Tinker's `sample()` returns logprob information for sampled tokens. `compute_logprobs(prompt)` returns per-token logprobs of a given sequence under a model, from one forward pass. Together these let you estimate the per-sequence Monte Carlo KL, sum(log q - log p_base), on policy samples. — [SamplingClient docs](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md)

### Inferences
- Gold gameability:
  - Use a gold that differs in family or size from the proxy.
  - Hand-audit about 50 top-proxy samples.
  - Report the exact e-rate axis as the only true gold. Frame quality-gold as "stronger judge", never as ground truth.
- Integer noise (1-5): if the proxy is servable on Tinker, read score-token logprobs with `topk_sample_logprobs` and use the expected score. If it is API-only without logprobs, average several judge samples or switch to pairwise comparison against a fixed reference.
- KL for RL checkpoints: the Monte Carlo estimate is unbiased but noisy. Use at least 256 samples per checkpoint and report nats per sequence, as Gao does, to match the sqrt(KL) axis.

### Gaps
- I did not confirm whether Inkling-Small as a judge exposes score-token logprobs through Tinker or the API.

## D prior art: on-policy distillation from a masked or logit-modified teacher; distilling constrained decoding into weights

### Takeaway
No paper found does on-policy distillation from a base model renormalized over allowed tokens. Closest is CC-OPD (2026-09), whose teacher is prompt-conditioned on constraints, not logit-masked. It is well established that locally masked decoding is myopic and biased (Park et al. 2024; Loula et al. 2025; Lipkin et al. 2025; Dang et al. 2026). The lipogram-specific paper (Gatsby, 2025) used `bad_words_ids` masking and reports misspellings and high OOV rates without fine-tuning.

### Cited Findings
- Thinking Machines on-policy distillation post (2025-10-27, updated June 2026 to Qwen3.5):
  - Per-token reverse KL with discount factor zero; discounts above zero "did not improve performance".
  - Teacher logprobs come from `compute_logprobs` on student samples. Advantage = -(log q - log p_teacher), plugged into the RL importance-sampling loss.
  - Costs about 1/10 of RL according to the Qwen3 report figures.
  - The teacher is used unmodified, and top-k logit distillation was not tested.
  - Source: [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- In the tinker-cookbook distillation recipe, teachers are specified only by model name and an optional checkpoint (`teacher_model=Qwen/Qwen3.5-9B`, plus `base_model` and `load_checkpoint_path` per dataset). There is no hook for a modified teacher distribution. — [recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/index.md)
- CC-OPD, Zheng et al. (2026-09-23, EMNLP 2026): the teacher is prompt-conditioned. It shapes the reward with leave-one-constraint-out shifts in the teacher's log-likelihood. No logit masking and no lexical or banned-letter constraints. — [arXiv 2609.27421](https://arxiv.org/abs/2609.27421)
- Park et al., Grammar-Aligned Decoding (NeurIPS 2024): constrained decoding "can distort the LLM's distribution". ASAp corrects this. — [arXiv 2405.21047](https://arxiv.org/abs/2405.21047)
- Loula et al. (ICLR 2025) use SMC weight corrections to mitigate the "greediness" of locally constrained decoding. Lipkin et al. (COLM 2025) give unbiased importance-weight estimates that correct "myopic local enforcement". Dang et al. (ICML 2026) find that SMC results are bottlenecked by the LCD proposal. — [Dang et al. 2606.01926](https://arxiv.org/pdf/2606.01926); [Lipkin et al. 2504.05410](https://arxiv.org/pdf/2504.05410)
- Gatsby Without the 'E' (Gokulakrishnan et al., IJCNLP-AACL 2025):
  - Setup: tokens containing "e" filtered via `bad_words_ids` with 20-beam search; fine-tuning on PAWS paraphrase pairs (QLoRA on Llama2).
  - Failure modes: mangled words such as "Gatbsy". Without fine-tuning, cosine similarity fell to 0.23, OOV rose to 63.17%, and grammar errors rose to 26.47 per paragraph, against 0.69 similarity for the full system.
  - Truncated words and dead ends are not reported explicitly.
  - Sources: [arXiv HTML](https://arxiv.org/html/2505.20501v2), [ACL Anthology](https://aclanthology.org/2025.ijcnlp-short.18)
- "Most Language Models can be Poets too" (Roush et al. 2022) bans letters by filtering the vocabulary, and reports that this beats fine-tuning alone on a lipogram-e dataset. — [paperswithcode](https://astro.paperswithcode.com/paper/most-language-models-can-be-poets-too-an-ai-1)

### Inferences
- "Distilling constrained decoding into weights" for lipograms is lightly explored (Gatsby trains with SFT plus masked beam search, not on-policy distillation). The specific on-policy-distillation-from-masked-base framing appears unpublished. Its weakness is theory, not novelty (next section).
- Masked-teacher myopia is well documented in the constrained-decoding literature. Its lipogram symptoms are mid-word dead ends such as "th" followed by a forced non-"e" continuation, and glued subword misspellings. That these happen is inferred from the LCD bias plus the Gatsby OOV numbers. No source documents a specific example like "investiging".

### Gaps
- No paper was found that compares on-policy distillation from an LCD teacher against RL at matched compute.

## D failure mode: Tinker teacher logprobs under a mask

### Takeaway
Tinker has no logit-bias, mask, or processor option in sampling. Computing the exact renormalizer Z_t needs mass over every e-containing token, which `target_prompt_logprobs` could request in principle but is cost-bounded and impractical at vocabulary scale. Z_t is not needed, though: with zero discount, -log Z_t depends only on the prefix, so it drops out of the expected gradient. The masked teacher therefore reduces to raw base logprobs with a large penalty on e-tokens. That simplification is also D's conceptual weakness.

### Cited Findings
- `SamplingParams` has exactly six documented fields: max_tokens, seed, stop, temperature, top_k, top_p. There is no logit_bias, bad-words, allowed-tokens, or logit-processor field. — [SamplingParams](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/samplingparams/index.md)
- `sample(..., include_prompt_logprobs, topk_prompt_logprobs, topk_sample_logprobs, target_prompt_logprobs, prompt_alt_tokens_k, prompt_logprobs_last_n)` and `compute_logprobs(prompt, prompt_logprobs_last_n)`:
  - `target_prompt_logprobs` returns logprobs for chosen token ids at each prompt position. Its cost is bounded by "len(prompt) * distinct ids".
  - `prompt_alt_tokens_k` draws k sampled alternatives per prompt position.
  - No numeric top-k limit is published.
  - Source: [SamplingClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md)
- `TopkLogprobs` is returned as dense (length, k) matrices, with sentinel logprob -99999.0 for empty cells. — [TopkLogprobs](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/topklogprobs/index.md)

### Inferences
1. **The renormalizer cancels.** For an e-free student token x_t, log p_mask(x_t) = log p(x_t) - log Z_t. In the zero-discount loss, the gradient is E[(log p_mask - log q)(x_t) ∇log q(x_t)]. The -log Z_t term is constant across actions given the prefix, so its expectation is zero; it only adds variance. A client-side teacher is therefore just `compute_logprobs` on the base model, with e-tokens clipped to a large negative value. No full-vocab logits are needed. Optionally, estimate Z_t with top-k (for example k=20) and use it only as a variance-reducing correction.
2. **What the student converges to.** At zero discount the per-token fixed point is q = the LCD distribution, which is masked-base sampling: the biased, myopic distribution in the cited papers. D's ceiling is "LCD baked into the weights". It beats LCD only through generalization side-effects. Since Tinker sampling cannot mask, baking LCD in has some value. Whether that value holds depends on whether the challenge forbids inference-time masking, which I did not check.
3. **Full discount is just RL.** With discount 1, the sequence-level KL(q || p_base(·|e-free)) has the global posterior as its target. That is exactly KL-regularized RL with reward 0 or -inf on e, with no quality signal. "D vs RL at matched dollars" then reduces to "dense per-token LCD target vs sparse sequence reward", a comparison the Thinking Machines post already frames as a one-line change to KL-regularized RL.
4. **Early training is mostly penalty.** The student samples without a mask, so early rollouts contain many e-tokens, and most of the signal is the clip penalty rather than teacher shape. Pick the clip value carefully, because it sets the effective reward scale. Also decide whether "E" (capital) counts, and handle tokens with leading spaces.

**Fix that keeps D alive:**
- Run RL with the Inkling-Small quality reward, and replace the usual KL to the plain base with a per-token KL to the LCD-masked base. The masked base is the anchor; the judge supplies the reward.
- Compare three arms at matched dollars: plain-base-KL RL, LCD-anchor RL, and pure on-policy distillation to LCD.
- This tests whether the anchor stops pulling the policy toward "e", which is a sharper question than "distillation vs RL".
- To counter the myopia criticism, add a cheap lookahead: use `prompt_alt_tokens_k` or short rollouts to down-weight prefixes that dead-end. This is SMC-flavored, after Loula and Lipkin.

### Gaps
- The actual server bound on top-k width and on `target_prompt_logprobs` cost is not published.
- I did not verify Qwen3.5-4B's vocabulary size or what fraction of tokens contain "e". That fraction determines whether an exact Z_t via `target_prompt_logprobs` is even conceivable.
- I did not check whether the challenge rules allow inference-time token masking. That decides whether D's "LCD into weights" has standalone value.
