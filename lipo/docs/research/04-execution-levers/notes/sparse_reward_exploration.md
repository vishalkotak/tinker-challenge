# Rare-success exploration for mask-free lipogram RL (Qwen3.5-4B, Tinker LoRA)

Scope note: no paper found trains an LLM with RL to avoid a letter. Every method below comes from math reasoning RLVR (2025), and its transfer to a lipogram is my inference. Search budget was about 15 tool calls; SRFT, ReLIFT and UFT details come from abstracts and snippets, not full papers.

## Baseline facts about the target problem

### Takeaway
Wu's unmasked RL attempts were small early prototypes with per-token e penalties. They collapsed to one repeated e-free phrase or to letter noise. His working result always used the decode mask, so a mask-free run has no prior success to copy.

### Cited Findings
- Early prototype, per-token e-based reward, 8 copies of one prompt, 64 tokens, 40 batches: every final sample was "i want to know what you want" ([experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)).
- Same setup plus a KL penalty to the base model, 60 batches: 7/8 samples were e-free but turned into broken strings such as "Tgth rthrt". KL did not prevent the collapse ([experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)).
- Revised prototype, 30 batches: all 8 samples were "i want to know what you want to know". RL loss reached 0.00845 while diversity collapsed. Exact configs were not saved and none of these runs had a held-out eval ([experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)).
- Wu's best masked recipe (Qwen3-1.7B, LoRA rank 16): SFT on 180 e-free answers, then GRPO with 4 prompts x 8 answers, 300 steps, temperature 0.7, 300 max tokens, LR 2e-5, KL 0.02, and judge reward (coherence-1 + responds-1)/8. It reached 60.5% usable. Masked RL without SFT reached 43%, and the masked base reached 26% ([Wu blog](https://kevin-wu.ca/blog/qwn/)).
- Qwen3-4B (not 3.5) scored 0% usable with the instruction alone and 12.5% with instruction plus mask. No trained 4B result was reported ([Wu blog](https://kevin-wu.ca/blog/qwn/)).
- The closest published decoding-side work uses constrained beam search that filters e tokens ([Gatsby without the 'E', 2505.20501, May 2025](https://arxiv.org/html/2505.20501v2)). A 2026 benchmark finds most models genuinely violate lipogram constraints ([2608.12426, Aug 2026](https://arxiv.org/pdf/2608.12426)).

### Inferences
- Wu's collapse matches the usual failure of a constraint-only reward: the cheapest way to score is a short, fixed e-free string. The reward must couple "no e" with quality from step 1, or the policy finds that shortcut.
- At 4B with the mask, 12.5% usable means the unmasked base success rate for "e-free and good" is close to zero. This is exactly the regime where LUFFY, BREAD and QuestA report that on-policy GRPO gets no signal.

### Gaps
- No unmasked run in Wu's logs used a judge-coupled reward, so whether the "judge x e-free" product alone collapses is untested.

## Off-policy and demonstration-guided RL (LUFFY, SRFT, ReLIFT, UFT, prefix/hint methods)

### Takeaway
The best-documented options are LUFFY (one expert trace per group, with policy shaping) and prefix-hint methods (QuestA, BREAD). Each beat SFT-then-RL or plain GRPO on math with 1.5B to 8B models, and each is reported to train in regimes where on-policy RL gets zero reward. Both map onto Tinker's `importance_sampling` Datum format. LUFFY's shaping term needs a custom loss.

### Cited Findings
**LUFFY** (arXiv 2504.14945, Apr 2025, revised Jun 2025)
- Setup: 8 rollouts per prompt, with 1 off-policy DeepSeek-R1 trace and 7 on-policy rollouts. Rollout batch 128, update batch 64, temperature 1.0, KL 0, entropy coefficient 0.01, Dr. GRPO normalization (no length or std normalization), constant LR 1e-6, 500 steps ([LUFFY html](https://arxiv.org/html/2504.14945v5)).
- Policy shaping: off-policy token probabilities pass through f(x) = x/(x+γ) with γ = 0.1. An extra variant removes the on-policy clip, because clipping suppressed learning from off-policy traces ([LUFFY html](https://arxiv.org/html/2504.14945v5)).
- Qwen2.5-Math-7B in-distribution average: LUFFY 50.1, SFT+RL 48.2, on-policy RL 45.5, SFT 44.1, RL with an SFT loss 40.1. Out-of-distribution: LUFFY 57.8, SFT 47.5, SFT+RL 44.8. Compute: LUFFY 77x8 GPU-hours vs SFT+RL 130x8 ([LUFFY html](https://arxiv.org/html/2504.14945v5)).
- Gains over on-policy RL on other models: Qwen2.5-Math-1.5B +8.0 (38.0 vs 30.0), Qwen2.5-Instruct-7B +5.5, LLaMA-3.1-8B +3.6. On the hard subset, LLaMA's on-policy reward collapsed to zero while LUFFY improved steadily ([LUFFY html](https://arxiv.org/html/2504.14945v5)).
- Ablation: mixing in off-policy traces without shaping gains fast early, then entropy collapses faster than in on-policy RL and performance converges back to on-policy RL. Shaping keeps entropy up ([LUFFY html](https://arxiv.org/html/2504.14945v5)).

**QuestA** (arXiv 2507.13266, Jul 2025; v4 Mar 2026)
- Hint = the first p% of the final solution, prepended to the question. Curriculum: 100 steps at p=50%, then 1,900 steps at p=25%. A 0% stage gave no gains. No hints at eval ([QuestA html](https://arxiv.org/html/2507.13266v3)).
- Problem selection: keep problems where a weak model solved 0 or 1 of 8. Then keep hinted prompts where the training model gets 0 to 4 of 8 correct ([QuestA html](https://arxiv.org/html/2507.13266v3)).
- Training: GRPO with DAPO dynamic filtering, no KL, group 16, batch 128, temperature 1.0, clip 0.2, constant LR 2e-5 ([QuestA html](https://arxiv.org/html/2507.13266v3)).
- Results: hints kept or improved pass@k at large k, unlike standard RL. The hinted model matched the 2K-step baseline in about 1.1K steps. Reported 1.5B SOTA pass@1 (v1): AIME24 67.1, AIME25 59.5, HMMT25 35.5. No comparison with SFT on full solutions ([QuestA html](https://arxiv.org/html/2507.13266v3); [arXiv](https://arxiv.org/pdf/2507.13266)).

**BREAD** (arXiv 2506.17211, Jun 2025, NeurIPS 2025)
- Theory: SFT+RL can fail completely when expert traces are too hard for a small model to express, or when the initial model's reward is exponentially sparse ([arXiv](https://arxiv.org/html/2506.17211v1); [NeurIPS](https://neurips.cc/virtual/2025/poster/118358)).
- Method: a GRPO variant that, when every self-generated trace in a group fails, inserts a short expert prefix and lets the model finish. This guarantees at least one success per update. Claimed results: under 40% of ground-truth traces needed, about 3x faster training than GRPO, and it solves problems SFT+RL cannot ([arXiv](https://arxiv.org/html/2506.17211v1)).

**SRFT, ReLIFT, UFT** (abstract or snippet level only)
- SRFT (2506.19767, Jun 2025): single-stage SFT plus RL with entropy-aware weighting. 59.1% math average (Qwen2.5-Math-7B per reviews), +9.0 over zero-RL and +4.8 over SFT ([arXiv](https://arxiv.org/html/2506.19767v1)).
- ReLIFT (2506.07527, Jun 2025): mostly RL, with SFT interleaved on hard questions whose solutions are collected online. Beats RL and SFT using 13% of the demonstration data ([arXiv](https://arxiv.org/html/2506.07527v1)).
- UFT (2505.16984, May 2025): unifies SFT and RFT in one process ([listing](https://fugumt.com/fugumt/paper_check/2505.16984v1_enmode)). The Salesforce paper of the same name (2410.21438) is about SFT plus alignment, not RL with demonstrations ([arXiv](https://www.arxiv.org/pdf/2410.21438)).

**Tinker implementability**
- `importance_sampling`, `ppo`, `cispo` and `dro` each take exactly `target_tokens`, `logprobs` (sampler logprobs, used as q in the p/q ratio) and `advantages`. Extra keys such as `weights` are rejected. An advantage of 0 masks a token out. Arbitrary losses go through `forward_backward_custom` ([Tinker losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/index.md)).

### Inferences
- **Prefix-guided (QuestA/BREAD style) is the easiest to build on Tinker.** Put the first 25 to 50% of a verified e-free answer into the assistant turn as prompt (advantage 0), sample the rest unmasked, score the whole answer, and use only the model's own tokens. The datums stay fully on-policy, so the `importance_sampling` loss is used unchanged. BREAD's rule fits sparse lipogram reward well: add the prefix only for prompts whose group had no success. Anneal p down to 0.
- **LUFFY style on Tinker:** append one verified e-free answer to each group of 7 or 8 rollouts, with its reward computed like any other and group-centered advantages. You need a value for `logprobs` (q). Setting q to 0 (prob 1) gives a gradient of roughly p·∇log p·A. That is plain mixed-policy GRPO without shaping, which LUFFY found plateaus with entropy collapse. The f(x)=x/(x+0.1) shaping needs `forward_backward_custom`. I did not verify how LUFFY's code sets q, so check the repo before building.
- For the lipogram, "too hard to express" (BREAD's failure case 1) is real. Frontier-written e-free answers may use vocabulary the 4B rarely produces. Prefix hints sidestep this better than full-trace imitation.
- Wu's 180 answers (public) can supply the demonstrations or prefixes.

### Gaps
- No method above has been tested on lexical or format constraints or on open-ended chat with judge rewards. Every result is math with verifiable answers.
- I did not read the SRFT, ReLIFT or UFT full texts, so their hyperparameters are missing.
- How LUFFY sets the off-policy logprob is unverified.

## SFT warm-start size

### Takeaway
No source gives a demonstration count at which RL "starts working". The evidence points to a small-to-moderate SFT set with early stopping, since heavy SFT can cut RL exploration. Wu's own sweep plateaued around 90 to 180 examples.

### Cited Findings
- Wu sweep (masked eval, base 26%): 15/30/45/90/135/180 examples gave 10/20.5/22.5/31.5/34/35% usable. Doubling 90 to 180 added 3.5 points, and 15 examples made answers longer and worse ([experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)).
- SFT 180 plus RL beat RL alone by 17.5 points (60.5 vs 43) under the mask ([Wu blog](https://kevin-wu.ca/blog/qwn/)).
- SPEAR (Qwen3-4B, MATH, 2026): performance peaked when about 30% of the data went to SFT and fell as the SFT share rose further, which the authors attribute to lost exploration diversity ([arXiv 2608.26550](https://arxiv.org/pdf/2608.26550)).
- Short-CoT SFT warm-up gives a moderate boost over cold-start RL, and the boost shrinks as tasks get harder ([arXiv 2505.24273, May 2025](https://arxiv.org/html/2505.24273v1)).
- BREAD argues that SFT+RL can fail outright when traces exceed what the small model can express ([arXiv](https://arxiv.org/html/2506.17211v1)).

### Inferences
- For mask-free training, SFT's main job is to lift the unmasked e-free success rate above roughly 1/G per group so GRPO gets signal. Measure the unmasked pass rate (e-free and judge ≥ 4) after SFT on 90 vs 180 vs about 500 examples before choosing. Prefer early stopping on validation loss, as Wu did.
- Prefix hints or LUFFY mixing can substitute for part of the SFT data, since BREAD needed under 40% of the traces.

### Gaps
- No study measures SFT size against unmasked lexical-constraint success.

## Curriculum on constraint strength and reward shaping for lexical constraints

### Takeaway
I found no paper that tightens an e-rate threshold over time. The closest evidence is QuestA's hint annealing (50% to 25%), which shows curricula that reduce help work, while a 0% stage added nothing. Lexical-constraint RL (Liu et al. 2021 for CommonGen) is reported to struggle with strict constraints.

### Cited Findings
- QuestA's two-stage hint curriculum (50% then 25%) and its no-gain 0% stage ([QuestA html](https://arxiv.org/html/2507.13266v3)).
- The Gatsby paper notes that Liu et al. (2021) used RL to improve concept coverage and that it struggles with stricter constraints ([2505.20501](https://arxiv.org/html/2505.20501v2)).
- Decode-time alternatives exist: constrained beam search that filters e tokens ([2505.20501](https://arxiv.org/html/2505.20501v2)) and frequency steering to zero for a forbidden letter ([2506.09846](https://arxiv.org/pdf/2506.09846)). Both can generate e-free training data.

### Inferences
- A threshold curriculum (reward 1 if e-rate < τ, τ from about 5% down to 0) risks rewarding the "near-miss with a few e's" mode and then hitting a cliff at τ=0. A graded reward of judge_score × max(0, 1 − k·e_count) gives variance inside groups without a threshold schedule. Cap the e term so it cannot outweigh quality, otherwise the short e-free string wins again, as in Wu's prototype.
- An annealed prefix (hint curriculum) is better supported by evidence than an annealed constraint.
- Constrained decoding (the mask or beam search) can work as a teacher. Use masked samples from the current policy as the "off-policy" members of the group (LUFFY-style), then drop them over time. This is an untested idea.

### Gaps
- I did not locate the challenge's exact capped e-rate reward definition in this pass; see `lipogram.py` in [tinker-challenge](https://github.com/thinking-machines-lab/tinker-challenge).
- No source covers NeuroLogic decoding as an RL data source for lexical constraints. I did not search it specifically.

## Zero-variance groups when all samples fail

### Takeaway
All-fail groups give zero advantage under group-mean centering. DAPO handles them by oversampling and filtering. Tinker has a `remove_constant_reward_groups` flag. Filtering alone cannot create signal when nearly every group fails, which is why hints or demonstrations are needed.

### Cited Findings
- DAPO (2503.14476, Mar 2025): oversample and drop prompts with accuracy 0 or 1, keeping 0 < correct < G per group, and refill the batch ([DAPO](https://arxiv.org/pdf/2503.14476); [project](https://dapo-sia.github.io)).
- P(group is unanimous) = p^G + (1−p)^G ([2607.00152, 2026](https://arxiv.org/pdf/2607.00152)). At G=4 on GSM8K, 0.69 of logged groups were degenerate. That paper proposes a fixed-reference advantage A = 2r − 1 as a complement ([2605.07689, 2026](https://www.alphaxiv.org/abs/2605.07689)).
- Tinker defaults: group 4 (range 2 to 16), mean-centered advantages without std normalization (Dr. GRPO), `remove_constant_reward_groups` default False, LR 1e-5 start (range 1e-6 to 1e-4), KL 0.05 suggested start, temperature 0.7 to 1.0 ([Tinker RL hyperparams](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/index.md)).

### Inferences
- With p near 0, p^G stays near 1 even at G=16, so DAPO filtering just discards most of the batch. Use a graded reward (judge score × e penalty) so groups have variance, larger G (16, as in QuestA), and BREAD-style prefix injection for groups with no success.
- A fixed-reference advantage (2r−1) would push down every all-fail sample, including fluent near-misses. That can speed the collapse into letter streams, so avoid it here.

### Gaps
- No evidence was found for graded-reward vs binary-reward GRPO on a constraint task.

## Avoiding length collapse and degenerate outputs under LLM-judge rewards

### Takeaway
LLM judges are easy to hack. Length collapsed to under 30 tokens in one study, and KL did not prevent judge overfitting in another. Use hard mechanical gates, a minimum length, a held-out second judge, and early stopping on that judge.

### Cited Findings
- With Qwen2.5-72B-Instruct as the generative reward model, policy response length fell to under 30 tokens early in training. Superficial "master keys" (reasoning openers, non-word symbols) fooled judges ([One Token to Fool LLM-as-a-Judge, 2507.08794, Jul 2025](https://arxiv.org/html/2507.08794v1)).
- GRPO on Llama-3.1-8B with judge rewards overfit the training judge quickly while a gold judge saw degradation. KL to the original policy did not help, and rubric-conditioned judges also got hacked ([2603.12246, 2026](https://arxiv.org/pdf/2603.12246)).
- Rubric RL failures in practice include length bias and self-praise ([CHERRL 2606.04923](https://arxiv.org/html/2606.04923)). Multi-role rubrics gave +1.7 BiGGen and +3.4 HealthBench-Hard over single-voice rubrics ([2607.01830](https://www.alphaxiv.org/abs/2607.01830)).
- LUFFY's mixed-policy ablation showed entropy collapse without shaping ([LUFFY](https://arxiv.org/html/2504.14945v5)).
- Wu's judge flagged 0/48 failed RL outputs as usable. He validated the judge against reference sets but not against humans ([Wu blog](https://kevin-wu.ca/blog/qwn/)).

### Inferences
- Reward = 0 unless the answer is e-free and above a minimum word count. Also zero it for repeated n-grams or non-dictionary token rates above a threshold (this catches "Tgth rthrt"). Otherwise use the judge score. Track mean length, distinct-n, and a second held-out judge every 25 steps, and stop when they diverge.
- Most of Wu's gain came by about step 125 of 300, so short runs with frequent eval are cheap insurance.

### Gaps
- No direct study was found on length collapse specifically under coherence or responsiveness judges for short chat answers.
