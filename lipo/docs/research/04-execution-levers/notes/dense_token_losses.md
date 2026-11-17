# Dense token-level objectives for forbidding the "e" token set, and Tinker feasibility

Feasibility in brief: **yes, with one approximation.** Tinker custom losses only see logprobs of the tokens you list in `target_tokens`. They never see full logits or the logsumexp over the vocabulary. However, `target_tokens` can be 2D `(N, K)`, so you get logprobs for K candidate tokens per position. That lets you compute an exact unlikelihood term `-log(1 - sum_{c in S_t} p(c))` over a chosen subset S_t of e-tokens. The full set of about 69k e-tokens per position is too large to ship over the wire. The practical subset is the e-tokens in the policy's top-K at each position, taken from `topk_prompt_logprobs`. That subset is where nearly all of the e-mass sits.

## Q1. Unlikelihood training and variants: effectiveness and fluency cost

### Takeaway
Token-level unlikelihood (UL) is the standard dense objective for pushing a token set down. The original paper reports that it keeps perplexity roughly intact when the negative set is small and context-dependent. No paper found fine-tunes a model to avoid a letter. Lipogram work uses decode-time masking and reports a steep fluency cost as the banned set grows, so a 28% vocab ban sits in the high-cost regime whatever the method.

### Cited Findings
- Welleck et al. (ICLR 2020) introduce token-level and sequence-level unlikelihood. Token-level UL lowers the probability of a candidate negative set C_t at each step via `-sum_{c in C_t} log(1 - p(c|x_<t))`, added to MLE. They report less repetitive, less dull text "while maintaining perplexity", and human evaluators preferred it (one-sided binomial test, p < .05). — [Welleck et al. 2019, arXiv 1908.04319](https://arxiv.org/pdf/1908.04319); [ICLR text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2020/neural_text_generation_with_unlikelihood_training__c24bf007.txt)
- Lagutin et al. combine UL with an RL method and report further reduced repetition without hurting LM quality. This is a precedent for "UL plus RL". — [arXiv 2101.04229](https://arxiv.org/pdf/2101.04229.pdf)
- Critique: UL ignores the relationship between label tokens and negative candidates, which limits its gains (motivating ScaleGrad and contrastive variants). — [Jiang et al., arXiv 2205.02517](https://ar5iv.labs.arxiv.org/html/2205.02517)
- Targeted Negative Training (TNT; Zhang, Ranganath, Tafvizi, TMLR) frames "minimal targeted updates": suppress unwanted outputs while keeping the rest of the distribution close to the original model, using negatives from the model's own generations. It reports a better trade-off than baselines that just push probability down. The abstract gives no figures. — [arXiv 2406.13660](https://ar5iv.labs.arxiv.org/html/2406.13660)
- In detox work, UL-style penalties over-penalize acceptable content (COUNT, EMNLP Findings 2023). Fine-tuning on filtered/detoxified data hurts perplexity disproportionately, and this grows with fine-tuning length. — [COUNT](https://aclanthology.org/2023.findings-emnlp.579.pdf)
- Lipograms specifically: "Gatsby Without the 'E'" (IJCNLP-AACL 2025) bans e-tokens at decode time via `bad_words_ids` plus beam search and no-repeat n-gram. Excluding up to 3.6% of the most common letters had minimal impact on meaning, but "translation fidelity rapidly and predictably decays with stronger lipogram constraints". Prompting alone led models to delete e's character by character. — [arXiv 2505.20501](https://arxiv.org/html/2505.20501v2)
- Constrained Text Generation Studio: filtering tokens before sampling beats asking the model to obey the constraint, but token filters "cripple a language model's ability to generate rare words" split into subwords. — [arXiv 2306.15926](https://arxiv.org/pdf/2306.15926)

### Inferences
- A UL term over e-tokens acts as a soft, learned version of `bad_words_ids`. At convergence it can make the model put roughly zero mass on e-tokens. The fluency cost comes from the constraint itself (28% of the vocab, and "e" is in about 57% of common English words), not from UL. Expect the same degradation the Gatsby paper sees for strong constraints, unless RL teaches the model to pick e-free paraphrases early.
- UL uses `log(1-p)`, which is bounded: its gradient vanishes as p goes to 0. Plain negative log-likelihood ("gradient ascent") on e-tokens is unbounded and tends to collapse the model. Use the bounded form.
- To limit collateral drift, pair UL with either an MLE/KL anchor on e-free text, or a TNT-style KL to the reference model on the non-forbidden mass.

### Gaps
- No paper found that fine-tunes an LLM to avoid a letter, or that applies UL to a forbidden set as large as about 28% of the vocab. Fluency cost at that scale is unmeasured.
- I did not get exact UL perplexity tables or TNT numbers. Only abstracts and summaries were read.

## Q2. Tinker loss API: can a custom loss see full-vocab logits, logsumexp, or set-probability sums?

### Takeaway
No full logits or logsumexp. Every custom loss receives only `logprobs` gathered at `target_tokens`. Since `target_tokens` may be `(N, K)`, you can sum `exp(logprobs)` over K chosen token ids per position, which gives exact set probabilities for any explicit subset. Full-vocab e-sets (K of about 69k) are impractical in payload size, so use top-K e-tokens per position.

### Cited Findings
- Built-in `LossFnType` in installed tinker 0.29.1: `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro`. GSPO and StableReinforce appear only in the newer docs index. — local `.venv/lib/python3.13/site-packages/tinker/types/loss_fn_type.py`; [llms.txt](https://tinker-docs.thinkingmachines.ai/llms.txt)
- `cross_entropy` accepts `target_tokens` and `weights` of shape `(N,)` or `(N, K)`. With `(N, K)` it computes a weighted sum over the K candidates per position (the docs show K=20 top-K distillation). Output logprobs have the targets' shape. No max K is stated. — [Cross-entropy docs](https://tinker-docs.thinkingmachines.ai/tinker/losses/cross-entropy/index.md)
- "The Tinker API does not expose full-vocabulary logits", so the SDFT recipe uses a top-K (K=20) approximation to full-vocab KL. — [SDFT recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/sdft/)
- `forward_backward_custom` (0.29.1): (1) runs a server forward with the `cross_entropy` surrogate and zero weights to get logprobs; (2) calls your `loss_fn(data, logprobs_list)` locally in torch; (3) runs `loss.backward()`; (4) sends `weights = -grad` back for a linear-surrogate forward_backward. It accepts only `loss_fn_inputs` keys `{"target_tokens", "weights"}`, and `loss_type_input` only `"logprobs"`. — local `tinker/lib/public_interfaces/training_client.py` lines 388-552
- Because step (4) sends arbitrary signed `-grad` as `weights`, the server's `cross_entropy` must accept signed (negative) and per-slot `(N, K)` weights. — same file, lines 476-497
- The docs' custom-loss page has a "multi-target masked cross-entropy" example: `target_tokens` of shape `[N, 4]`, logprobs returned `[N, K=4]`, renormalized over 4 answer letters. Cost is about 1.5x FLOPs of one forward_backward and up to 3x wall time. A failed custom op poisons the client, so you must resume from checkpoint with a new client. — [Custom loss docs](https://tinker-docs.thinkingmachines.ai/tinker/losses/custom/index.md)
- `TensorData` supports sparse CSR for `target_tokens` and `weights` (`from_torch_sparse`, used only when it saves space). — local `tinker/types/datum.py` (`_SPARSE_ELIGIBLE_KEYS`), `tinker/types/tensor_data.py` lines 36-155
- Newer SDK (docs at 0.39.0): `ServiceClient.create_custom_loss(fn, input_ranks)` (alpha). It traces `fn` with `torch.fx`, which must use torch and math only, with no data-dependent Python branching, no `nn` modules and no captured tensors. It runs per datum and sums. Inputs are `logprobs` (rank matches `target_tokens`: 0, 1 `[T]`, or 2 `[T,K]`) plus `loss_fn_inputs` keys. If supported, it runs server-side in a single forward/backward (`forward_backward(data, "custom", {"loss_id": ...})`), with no extra client forward. `CustomLossGraph`/`CustomLossNode` are the serialized fx graph. The server rebuilds it from its own allowlist of ops. — [service_client.py](https://raw.githubusercontent.com/thinking-machines-lab/tinker/main/src/tinker/lib/public_interfaces/service_client.py); [training_client.py](https://raw.githubusercontent.com/thinking-machines-lab/tinker/main/src/tinker/lib/public_interfaces/training_client.py); [CustomLossGraph](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/customlossgraph/index.md); [CustomLossInput](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/customlossinput/index.md)
- `create_custom_loss`, `CustomLossGraph` and `LossHandle` are absent from installed 0.29.1 (grep finds nothing). Only the two-pass callable path exists there. — local grep of `.venv/.../tinker`
- Sampler exposes `topk_prompt_logprobs` and `topk_sample_logprobs` (int, top-k per position). No max k is documented in the SDK. — local `tinker/types/sample_request.py` lines 54-59
- The cookbook's SDFT uses exactly this pattern: teacher-force through a SamplingClient with `topk_prompt_logprobs=topk`, then build `(N, K)` `target_tokens`/`weights` and call `forward_backward_custom_async` with a reverse-KL custom loss. — local `tinker_cookbook/distillation/sdft.py` lines 470-560, 693-712

### Inferences
- **Exact full-set UL is technically expressible but impractical.** With K = about 69k e-token ids per position and N = 512, `target_tokens` alone is 35M int64 values (about 280 MB) per datum, plus weights. The forward-pass weights are all zero, so CSR helps there, but `target_tokens` is dense, and the backward weights are dense too. Server memory and limits for large K are unknown. Do not rely on it.
- **Recommended exact-on-subset UL.** For each position t, take S_t = {e-containing ids in the policy's top-K} (K of about 20 to 64), plus the sampled token if it contains "e". Then `p_e(t) = sum exp(logprobs[t, k])` over valid slots, and `L_UL = -sum_t mask_t * log(1 - p_e(t) + eps)`. Tail e-mass outside top-K is ignored. On-policy it is usually small, and you can log `1 - sum_topK p` as a bound.
- Signed `(N, K)` weights with built-in `cross_entropy` give a cheap linear penalty (`+lambda * logprob` of e-tokens) with no custom pass, but it is unbounded (gradient ascent). Prefer the custom `log(1-p)` form, or clip by zeroing weights once p_e < threshold.
- Combine UL and RL in one custom loss: PPO/IS surrogate on column 0 (sampled token) plus UL on columns 1..K. Alternatively, alternate steps: built-in `importance_sampling`/`ppo` step, then a custom UL step, both before `optim_step`, so gradients accumulate.

### Gaps
- Server-side op allowlist for `create_custom_loss` is not documented (`exp`, `log`, `log1p`, `sum`, `clamp` presumably fine, but unconfirmed).
- No documented max K for `topk_prompt_logprobs` or for `(N, K)` targets. Test K=64 and K=256 when credentials exist.
- Whether rank-2 `target_tokens` works with built-in `ppo`/`importance_sampling` (not only `cross_entropy`) is undocumented.

## Q3. Token-level reward/advantage shaping for forbidden-token penalties

### Takeaway
Tinker RL losses already take per-token advantages, so a dense penalty on sampled e-tokens is trivial with built-ins. That is "unlikelihood on sampled tokens only". It is weaker than top-K UL because it only touches tokens that were actually emitted. The literature supports dense token rewards (RTO) over sequence-end-only rewards for credit assignment.

### Cited Findings
- RTO (Zhong et al., "DPO Meets PPO") models RLHF as a token MDP with per-token rewards, instead of one reward on the final token. It reports +7.5 AlpacaEval 2 and +4.1 Arena-Hard over PPO, reaching PPO performance with 1/8 the data (self-reported). — [arXiv 2404.18922](https://arxiv.org/abs/2404.18922v4)
- Tinker `loss_fn_inputs` keys include per-token `advantages`, `logprobs`, `clip_low_threshold` and `clip_high_threshold` (float32, length N). — local `tinker/types/datum.py` `_KEY_TO_TYPE`
- Final Token Preference Optimization (2025 preprint) adjusts logits only where a banned pattern appears. This is another localized negative signal. — via search summary, [arXiv 2505.16722](https://arxiv.org/pdf/2505.16722v3) (not read in full)

### Inferences
- Built-in dense penalty: `advantage_t = A_seq - lambda * 1[token_t contains "e"]` (optionally also penalizing the 1-3 preceding tokens that committed to an e-word). Use it with `importance_sampling` or `ppo`. No custom loss, no extra pass.
- Hierarchy of options, cheapest to strongest:
  (a) per-token advantage penalty on sampled e-tokens;
  (b) signed `(N, K)` cross_entropy on top-K e-tokens;
  (c) custom `log(1 - sum p)` UL on top-K e-tokens plus the RL surrogate;
  (d) TNT-style KL-to-reference on the e-free renormalized top-K, to preserve fluency.
- A sampler-side logit mask is still needed at eval if the hard constraint must be exact. UL reduces e-mass but cannot guarantee zero. If the eval permits a hard mask, UL mostly helps by making the model plan e-free paths, so masked decoding hits fewer dead ends.

### Gaps
- No published result on per-token penalties for character-level lexical constraints at this scale.
- Qwen3.5-4B vocab size (248k) and the 28% e-token share come from the task brief and were not re-verified here.
