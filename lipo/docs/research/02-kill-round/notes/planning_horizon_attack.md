# Candidate C attack: "RL extends a model's planning horizon" (Tinker Lipogram, Qwen3.5-4B)

Research date: 2026-11-16. Scope: 2023 to Nov 2026. Builds on round 1 notes (`Lipogram paper scan round 1/planning_tokenization.md`), which already summarized Future Lens, Wu/Morris/Levine 2024, Anthropic biology 2025, Nainani 2025, Ma & Rui 2026, Dong 2025. This file adds the adversarial checks. Facts are cited; inferences are labeled.

**Verdict (inference): SURVIVES, but only in a modified form.** Nobody has done it yet, and it is technically feasible. Run as specified, though (the default challenge run, "max decodable k" vs step, probes on the policy's own outputs), it will most likely give a confounded or null result. Below are three kill-level risks, each with a fix.

## 1. Prior art: is it already done?

### Takeaway
I found no paper that probes RL or fine-tuned checkpoints for a change in future-token or planning horizon. The closest work compares base vs instruct (Nainani, Dong) or tracks generic feature change under RL. The slot is open, but the most cited recent paper (Ma & Rui 2026) already shows that probe-decodable plans are often causally unused, so a probe-only result will be discounted.

### Cited Findings
- Ma & Rui, "Where's the Plan?" (arXiv May 2026, ICML 2026 MechInterp workshop): Qwen3 at 0.6B, 1.7B, 4B, 8B, 14B and 32B, plus Gemma-3 and Llama-3. The upcoming rhyme is linearly decodable at the line boundary, but only Gemma-3-27B causally relies on it, and patching other models at the boundary has near-zero effect. The paper does not discuss RL or instruction tuning, and does not cover Qwen3.5. — [arXiv 2605.07984](https://arxiv.org/abs/2605.07984), [HTML](https://arxiv.org/html/2605.07984)
- Same paper, generic text: linear probes on 1,000 Pile sequences predict the token k steps ahead (top-5). Accuracy falls with k, and "the k=8 confidence band overlaps the unigram baseline across all layers in all three models." Controls: a unigram baseline, a general-text negative control, zero-vector and donor-prompt patching baselines, random-head and comma-head controls, and Wilson plus cluster-bootstrap CIs. Hewitt & Liang is cited, but no control-task or selectivity experiment is run. Main patching figures use N=100 (5 prompt pairs x 20 samples). — [HTML](https://arxiv.org/html/2605.07984)
- Nainani et al. 2025 (Gemma-2-2B): instruction tuning "refines existing planning behaviors in the base model rather than creating them"; RL is hypothesized but not tested. — [arXiv 2508.18098](https://arxiv.org/html/2508.18098v1)
- Dong et al. 2025: instruct models beat base models on response-planning probes, and RL models are not evaluated. — [arXiv 2502.06258](https://arxiv.org/html/2502.06258v2)
- ParaScopes (Pochinkov et al., arXiv Oct 31 2025): residual-stream decoders for paragraph-scale plans. In small models, information corresponding to 5+ tokens of future context is decodable. It is not about RL checkpoints (abstract and summary level only). — [arXiv 2511.00180](https://arxiv.org/abs/2511.00180v1), [LessWrong](https://www.lesswrong.com/posts/9NqgYesCutErskdmu/parascopes-do-language-models-plan-the-upcoming-paragraph)
- A broad search for "probing RL checkpoints planning/lookahead" returned only adjacent 2026 work. Examples: "The Depth Ceiling" (from-scratch transformers learn latent planning up to about 3 lookahead steps), ICLR 2026 token-level RLVR distribution-shift analyses (RL changes concentrate in a small fraction of high-divergence token distributions), and Next-ToBE (training with soft future-token targets). None probes planning horizon before vs after RL. — [arXiv 2604.06427](https://arxiv.org/html/2604.06427), [ICLR 2026 RLVR shifts](https://proceedings.iclr.cc/paper_files/paper/2026/file/c9b7e6175f2bd3c2824f24aa7ac313d6-Paper-Conference.pdf), [Next-ToBE](https://proceedings.iclr.cc/paper_files/paper/2026/file/e4a0d8aef3567f742b0794844d9b5847-Paper-Conference.pdf)
- The closest RL-and-planning result is behavioral only. R^3 (robotics, 2026) counts first-person planning phrases in reasoning traces: 11 in the base model vs 23 after RL. No internal probes. — [arXiv 2608.26053](https://arxiv.org/pdf/2608.26053)
- Round 1 (snippet level, unverified): 2026 preprints report that GRPO reorganizes features rather than adding concept information ([alphaXiv 2610.39182](https://www.alphaxiv.org/abs/2610.39182)), and that steering RL-associated SAE features into the base recovers about 80% of the RL gain ([arXiv 2609.15064](https://arxiv.org/pdf/2609.15064v1)). RLVR updates avoid the principal subspaces, with minimal spectral drift ([arXiv 2511.08567](https://arxiv.org/abs/2511.08567)).

### Inferences
- The project is not already done. The risk is not being scooped; it is that reviewers who know Ma & Rui will reject probe-only horizon claims, so the causal arm is mandatory.
- The prior work predicts that RL reweights existing lookahead features rather than creating new ones. The likely honest headline is "decodability barely moves; causal reliance or readout changes," or simply a null.

### Gaps
- I could not access OpenAI or DeepMind internal work on planning in reasoning models. I found no public paper from either on probing planning horizon across RL checkpoints.
- I did not read the full text of the 2026 RL-feature preprints.

## 2. Confounds and standard controls

### Takeaway
The confound you named is real and severe, and the default reward makes it worse. The RL policy's own text will contain almost no e-words, so the label "word k ahead contains e" becomes near-constant on RL samples. Probes must be trained and tested on the same fixed, teacher-forced corpus for every checkpoint, with lexical baselines and selectivity reported.

### Cited Findings
- The challenge reward is a pure sequence-level e-rate penalty, `-min(1.5, e_rate/0.10)`, plus an optional judge term that is commented out by default (`grade = 0`). The README says the starter outputs "degrade into gibberish". — [lipogram.py](https://raw.githubusercontent.com/thinking-machines-lab/tinker-challenge/main/lipogram.py), [repo](https://github.com/thinking-machines-lab/tinker-challenge)
- Even in base models, generic k-ahead probes fall to the unigram baseline by k=8. So "max decodable k" sits on a decaying curve, and its value depends on the baseline and the threshold. — [Ma & Rui HTML](https://arxiv.org/html/2605.07984)
- Hewitt & Liang (2019) control tasks and selectivity are the standard check that a probe is not memorizing. Ma & Rui cite the warning but do not run it. — [Ma & Rui HTML](https://arxiv.org/html/2605.07984)

### Inferences
- **Base-rate shift.** On RL-sampled text, positives (e-words ahead) vanish, so AUC and accuracy become uninterpretable or meaningless. Fix: build one corpus (base-model generations, or human text), teacher-force it through every checkpoint, and keep the labels fixed.
- **Lexical shortcut.** "Word k ahead contains e" correlates with the current tokens: function-word patterns ("the", "he", "be", "are" recur), topic, and position in the sentence. Baselines to beat: (a) a logistic model on bag-of-current-and-previous token ids, (b) the probe applied to layer-0 embeddings, (c) a control task with random labels per word type (selectivity).
- **The real comparison is a delta.** The clean quantity is AUC(RL checkpoint, layer, k) minus AUC(base, layer, k) on identical inputs, with bootstrap CIs over documents. A thresholded "max k" amplifies noise.
- **Tokenization.** Labels must be at word level, with "word k ahead" fixed relative to word boundaries. Qwen3.5's roughly 248K vocabulary means many words are single tokens, which shrinks the "within-word planning" part.

### Gaps
- I found no standard protocol specific to probing across fine-tuning checkpoints. The approach above is my synthesis.

## 3. Feasibility on an M4 Pro 24 GB Mac with Tinker checkpoints

### Takeaway
Feasible, with three traps. (1) The default script saves non-downloadable `sampler_weights`. (2) Issue #1014 silently drops the unembedding LoRA when loading into PEFT for Qwen3.5-4B. (3) Qwen3.5's Gated DeltaNet layers fall back to slow pure-PyTorch on MPS. Memory is not the problem.

### Cited Findings
- Challenge defaults: `BASE_MODEL = "Qwen/Qwen3.5-4B"`, `LORA_RANK = 8`, `TRAINING_STEPS = 10`, `GROUP_SIZE = 8`, 8 training prompts, `MAX_TOKENS = 200`, LR 8e-4. Each step calls `save_weights_for_sampler_async`, so the run produces 11 sampler checkpoints. The script also has a `LEADERBOARD_NAME` env var, so some leaderboard exists, which contradicts the round 1 note of "no leaderboard". — [lipogram.py](https://raw.githubusercontent.com/thinking-machines-lab/tinker-challenge/main/lipogram.py)
- Tinker docs say only `external_weights` checkpoints are downloadable, and `get_external_weights_urls` rejects `weights` and `sampler_weights` paths. Use `save_weights_external`, then fetch the signed URLs, which expire. With no TTL, checkpoints persist until deleted. Other doc pages and examples show `sampler_weights` downloads, so the docs conflict. — [Tinker weights docs](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/), [CLI reference](https://tinker-docs.thinkingmachines.ai/tinker/cli/checkpoint/), [download API](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/weights/download/)
- tinker-cookbook #1014 (opened 2026-11-15, open, PR #1015): on tied-embedding models (Qwen/Qwen3.5-4B, Qwen3-4B-Instruct-2507), `build_lora_adapter` maps the unembedding LoRA onto `embed_tokens`. PEFT only warns about the missing keys, and on a tiny tied Qwen3.5 "PEFT logits match the base model exactly". The fix is to map `unembed_tokens` to `lm_head` (error about 1.8e-07 after the fix). A separate problem: `build_hf_model` on tied models adds the delta to the shared embedding weight, which also changes input embeddings, and whether that matches Tinker's sampler is unverified. — [issue #1014](https://github.com/thinking-machines-lab/tinker-cookbook/issues/1014)
- Official Qwen3.5-4B config: `Qwen3_5ForConditionalGeneration` with a vision tower (depth 24), 32 text layers in a repeating 3x linear_attention + 1x full_attention pattern (full attention at indices 3, 7, ..., 31), hidden size 2560, `tie_word_embeddings: true`, a 1-layer MTP head, bf16. — [config.json](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json)
- The Transformers fast path needs flash-linear-attention and causal-conv1d, both CUDA-only. Without them, the gated delta rule uses a pure-PyTorch fallback, measured at about 2.35x slower on CUDA. A third-party Metal shim exists (mps-linear-attention), unverified. — [Unsloth PR #5434](https://github.com/unslothai/unsloth/pull/5434), [mps-linear-attention](https://github.yezi957.workers.dev/mpsops/mps-linear-attention)
- TransformerLens has a Qwen3.5 bridge adapter with GatedDeltaNet hooks (`hook_linear_attn_in/out`). In the multimodal adapter, only full-attention layers get `attn` modules. I found no explicit nnsight support for Qwen3.5 (nnterp exists over nnsight). — [TL qwen3_5](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.model_bridge.supported_architectures.qwen3_5.html), [TL GatedDeltaNet](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.model_bridge.generalized_components.gated_delta_net.html), [nnterp arXiv 2511.14465](https://arxiv.org/pdf/2511.14465v2)

### Inferences
- **Memory:** about 4-5B params in bf16 is about 9-10 GB, which fits in 24 GB with room for activation caches. Residual-stream patching only needs forward passes. Plain PyTorch hooks on `model.model.language_model.layers[i]` outputs are enough, and no TransformerLens is needed.
- **Speed:** the slow DeltaNet fallback matters little for 200-token teacher-forced passes. Expect seconds per sequence. Thousands of sequences times 11+ checkpoints is hours, not days. Generation-based causal tests (steer, then sample) cost more but are bounded.
- **The #1014 trap is a near-kill if missed.** Under the default reward, much of the RL effect plausibly sits in the unembedding LoRA, which is a myopic logit-level "don't emit e-tokens" bias. Load with stock PEFT and you probe a model that behaves like the base, so you get a null by construction. Load with `build_hf_model` and you also perturb the input embeddings. Always check local logprobs against Tinker sampler logprobs on the same tokens before probing.
- **Patching on the residual stream is unaffected by the linear-attention recurrence**, but patching a position does not alter the DeltaNet state of earlier positions. That is the same causal semantics as standard residual patching, so it is fine. Patching inside the recurrent state is harder and not needed.
- Native MTP pretraining (the 1-layer MTP head) means the base model was explicitly trained to encode t+1 and t+2 information. Base lookahead at small k is likely already high, which leaves less headroom.

### Gaps
- I did not confirm whether `save_weights_external` works for LoRA runs on Qwen3.5-4B, or what storage costs.
- Actual MPS throughput for Qwen3.5-4B forward passes was not measured.
- It is unverified whether HF Transformers loads the MTP layer (irrelevant to probing, but it matters if you want to compare against MTP logits).

## 4. Will the effect be clear or tiny?

### Takeaway (inference)
With the default run (10 steps, rank 8, 8 prompts, e-only reward), expect a tiny or uninterpretable effect. The policy can collapse to degenerate e-free text without any longer-range planning, and logit-level suppression solves most of the task myopically. A clear result needs a run in which lookahead is actually rewarded and the RL weight change is large enough to matter.

### Cited Findings
- The default reward leads to gibberish. The judge reward is off by default. — [repo](https://github.com/thinking-machines-lab/tinker-challenge), [lipogram.py](https://raw.githubusercontent.com/thinking-machines-lab/tinker-challenge/main/lipogram.py)
- RLVR updates are sparse-looking and avoid principal directions ([arXiv 2511.08567](https://arxiv.org/abs/2511.08567)). Base vs tuned planning differences in prior work are refinements, not new capacity ([Nainani 2025](https://arxiv.org/html/2508.18098v1)).
- Hard token masking for lipograms produces misspellings and lookalike characters, which shows myopic avoidance is a cheap alternative to planning. — [arXiv 2410.01103](https://arxiv.org/html/2410.01103v2)

### Inferences
- Planning only pays off when fluency is also rewarded (judge on). Without it, the cheapest policy is "pick e-free tokens greedily", which needs no horizon. Prediction: with the e-only reward, the horizon delta is about 0. With e plus judge reward, there is a possible small gain at k=1-2, mostly at mid-to-late layers.
- That "e-only RL does not plan; e-plus-fluency RL does" contrast is itself a cleaner, more publishable result than "RL extends horizon".
- Ten steps on 8 prompts is too little RL signal to measure trends across checkpoints. You need about 50-200 steps and roughly 100+ diverse prompts.

### Gaps
- No pilot data. Effect size is unknown until a base-vs-final probe delta is measured once.

## Fix list (inference, ordered)
1. Run your own training (fork of lipogram.py): turn the judge on, use about 100+ prompts, about 100 steps, and call `save_weights_external` every about 10 steps with no TTL. Keep the default e-only run as a contrast arm.
2. Patch #1014 locally (map unembed to lm_head; do not merge into tied embeddings). Gate everything on local logprobs matching Tinker sampler logprobs.
3. Use a fixed teacher-forced probe corpus (base-model and human text) across all checkpoints, with word-level labels.
4. Report the AUC delta vs base per (layer, k) with bootstrap CIs, against baselines: token-id bag, layer-0 probe, Hewitt-Liang control task. Drop the "max k" threshold, or report it only alongside the deltas.
5. Decompose the RL effect: unembed-LoRA-only vs body-LoRA-only. If unembed-only reproduces most of the e-rate drop, the policy is myopic and the horizon claim dies.
6. Causal arm: steer along the probe direction (and patch residuals from e-free-continuation donors) at position t. Measure the e-rate of words t+k in sampled continuations, base vs RL, N >= 100 per condition as in Ma & Rui.
7. Pre-register the prediction (null for e-only, small k=1-2 gain for e plus judge) so a null still counts as a finding.
