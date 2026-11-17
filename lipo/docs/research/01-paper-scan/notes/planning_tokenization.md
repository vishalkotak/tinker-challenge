# Planning/lookahead inside LLMs and tokenization effects, applied to an e-free (lipogram) RL task

Scope: 2023 to Nov 2026 papers relevant to RL fine-tuning Qwen3.5-4B (Tinker) to answer without "e", with probe analysis on an M4 Pro 24 GB Mac. Facts are cited; inferences are marked as such. Source line separators use "(Source)" instead of dashes.

## 1. Planning evidence: methods and findings

### Takeaway
There is solid, method-diverse evidence that LMs encode future tokens before emitting them (linear probes, myopic training, attribution graphs, activation patching), but causal use of that plan is model- and task-dependent: probes often decode a "plan" the model does not actually rely on. Rhyming couplets are the standard testbed, and they map naturally onto "choose an e-free word before its tokens."

### Cited Findings
- Future Lens (Pal, Sun, Yuan, Wallace, Bau; arXiv Nov 8 2023; CoNLL 2023). GPT-J-6B. Linear models plus causal "transplant" of a single hidden state predict tokens 2+ positions ahead; at some layers >48% agreement with the model's own subsequent-token predictions. ([arXiv 2311.04897](https://arxiv.org/abs/2311.04897), [project](https://future.baulab.info), [ACL Anthology](https://preview.aclanthology.org/setup/2023.conll-1.37))
- "Do Language Models Plan Ahead for Future Tokens?" (Wu, Morris, Levine; arXiv Apr 2024, v2 Aug 2024). Separates "pre-caching" (deliberately computing features useful only later) from "breadcrumbs" (features useful now happen to help later). Method: myopic training that blocks gradients from the current loss into earlier positions; the "myopia gap" measures pre-caching. GPT-2 small: CE 3.28 vanilla vs 3.40 myopic (gap ~0.12), read as mostly breadcrumbs. Pythia 14M to 2.8B: the gap grows with scale from 70M and is non-negligible at 2.8B. Myopic models do not benefit from filler tokens on multiplication. ([arXiv 2404.00859v2](https://arxiv.org/html/2404.00859v2))
- Anthropic, "On the Biology of a Large Language Model" (Lindsey et al.; Mar 27 2025). Claude 3.5 Haiku, cross-layer transcoder (~30M features) + attribution graphs + interventions on the real model. Candidate rhyme words (e.g. "rabbit", "habit") are active at the newline token before line 2; steering only works at that position; planned words also shape intermediate words (working backward, e.g. promoting "like a"). Injecting a planned word made the line end with it in 70% of 25 poems; planned-word features found in about half of poems examined. ([Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/biology.html))
- "Detecting and Characterizing Planning in Language Models" (Nainani, Vaidyanathan, Watts, Assis, Rigg; arXiv Aug 25 2025). Gemma-2-2B base/instruct with GemmaScope SAEs. Planning = Future-Token Encoding (latent's logit-lens ranks a future token in top-K) + Precursor Influence (ablating the latent earlier removes the future token). Criteria work for any distance m-n >= 1. Gemma-2-2B showed no planning on the Anthropic rhyme task (it improvises); on MBPP code it switches between planning and improvising, sometimes plans toward wrong targets. ([arXiv 2508.18098](https://arxiv.org/html/2508.18098v1))
- "Where's the Plan?" (Ma, Rui; arXiv May 8 2026, v2 Jun 11 2026; ICML 2026 MechInterp workshop). Qwen3, Gemma-3, Llama-3, >10 scales. Upcoming rhyme is linearly decodable at the line boundary, stronger with scale in all three families. But activation patching shows only Gemma-3-27B causally relies on the line-boundary encoding (handoff around layer 30, five heads recover ~90% of rhyme routing); all other models condition on the rhyme word itself and patching the boundary has near-zero effect despite strong probe signal. ([arXiv 2605.07984](https://arxiv.org/abs/2605.07984))
- "Emergent Response Planning in LLMs" (Dong, Zhou, Liu, Yang, Lu; arXiv Feb 2025, v2 Jun 2025). Llama-2/3, Mistral, Qwen2 base+instruct. Small MLP probes on prompt representations predict response length, reasoning-step count, story character choice, MC answer, confidence. Structure attributes peak in middle layers, content in later layers. Probe accuracy high early, dips mid-response, recovers near end. Probes beat the model's own verbalized self-estimates. No causal claims. ([arXiv 2502.06258v2](https://arxiv.org/html/2502.06258v2))
- Jenner et al., "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network" (arXiv Jun 2 2024; NeurIPS 2024). Leela Chess Zero policy net represents future moves of the best line and these are causally important (activation patching); a simple probe predicts the optimal move 2 turns ahead with 92% accuracy on positions with a single best line. Existence proof, not universal. Follow-up by Cruz (arXiv 2505.21552, 2025) extends to 5th and 7th moves; look-ahead is highly context-dependent. ([NeurIPS PDF](https://proceedings.nips.cc/paper_files/paper/2024/file/37d9f19150fce07bced2a81fc87d47a6-Paper-Conference.pdf), [arXiv 2406.00877](https://arxiv.org/abs/2406.00877v1), [Cruz 2505.21552](https://arxiv.org/html/2505.21552v1))
- Multi-token prediction (Gloeckle et al.; arXiv Apr 30 2024; ICML 2024). n independent output heads on a shared trunk predict the next n tokens; 13B models solve 12% more HumanEval and 17% more MBPP than next-token baselines; up to 3x faster inference with 4-token prediction; helps induction heads/algorithmic reasoning in small experiments. ([PMLR](https://proceedings.mlr.press/v235/gloeckle24a.html), [arXiv 2404.19737](https://arxiv.org/pdf/2404.19737))
- An EMNLP 2024 paper on look-ahead planning mechanistic interpretability reportedly found information about Step 6 while generating Step 1 (seen only via an aggregator; not verified against the original). ([Lacuna summary](https://lacuna.tiptreesystems.com/work/unlocking-the-future-exploring-look-ahead-planning-mechanistic-interpretability/wrk_de983be0541ecb4a6ee4235789c74d38))
- ParaScopes (arXiv 2511.00180, Nov 2025) studies what activations encode about future text (title only seen; contents not read). ([arXiv 2511.00180](https://arxiv.org/pdf/2511.00180))

### Inferences
- The key methodological lesson from Ma & Rui 2026 and Nainani et al. 2025: a decodable future-word probe is not proof of planning. A credible lipogram study needs a causal test (patching/steering) alongside probes.
- Lipogram is a cleaner planning testbed than rhyme: the constraint applies at every word, not only line-end, so planning signals can be measured at many positions per sample, giving far more data per generation.
- Gemma-2-2B did not plan rhymes, while Claude 3.5 Haiku and Gemma-3-27B did. A 4B model sits plausibly near the threshold, which makes "RL pushes a 4B model across the planning threshold" a testable and interesting claim.

### Gaps
- No published result specifically on Qwen3.5 (only Qwen3 in Ma & Rui 2026). Whether Qwen3.5-4B plans rhymes causally is unknown.
- I did not verify the EMNLP 2024 look-ahead paper or ParaScopes contents from primary text.

## 2. Has anyone measured whether RL/fine-tuning changes planning horizon or future-token representations?

### Takeaway
Not directly, as far as I found. Instruction tuning has been compared to base models (more/better planning in instruct), and a few 2026 preprints track general representation change under RL, but no paper measures a change in planning horizon (how many tokens ahead the plan is decodable/causal) caused by RL. This is the open, novel slot.

### Cited Findings
- Nainani et al. 2025: on MBPP, base Gemma-2-2B solves all tasks where instruct improvises but only 54% (7/13) of tasks where instruct plans; "instruction tuning refines existing planning behaviors in the base model rather than creating them from scratch." They hypothesize RL strengthens planning but do not test it. ([arXiv 2508.18098](https://arxiv.org/html/2508.18098v1))
- Dong et al. 2025: instruct models beat base models on response-planning probes, especially structure attributes; base models still show weaker positive signals, suggesting the representations arise in pretraining. RL models not evaluated. ([arXiv 2502.06258v2](https://arxiv.org/html/2502.06258v2))
- 2026 preprints found via search snippets only (not read; unreviewed): GRPO on Qwen2.5-1.5B for Tic-Tac-Toe reportedly reorganizes which features dominate rather than adding concept information, with probe AUC shifts between checkpoints ([alphaXiv 2610.39182](https://www.alphaxiv.org/abs/2610.39182)); a fixed SAE dictionary across RL checkpoints reportedly shows gradual feature change, and steering RL-associated features into the base model recovers ~80% of the RL gain ([arXiv 2609.15064](https://arxiv.org/pdf/2609.15064v1)); an IBM summary reports post-RL representations closely resemble the mid-trained checkpoint by CKA ([IBM Research](https://research.ibm.com/blog/mid-training-for-better-ai-reasoning)).
- A theory paper analyzes benefits/pitfalls of RL vs SFT for LM path planning (conclusions not checked). ([arXiv 2509.22613](https://arxiv.org/pdf/2509.22613))

### Inferences
- Novel experiment candidate: "RL extends the causal planning horizon." Train with e-free reward on Tinker; save base and several RL checkpoints; for each, measure (a) probe accuracy for "will the word k tokens/words ahead contain e" or the identity of the next content word, as a function of k, and (b) causal effect of patching/steering at earlier positions. The headline plot: decodable and causal horizon k* versus RL step.
- The general-RL findings (reorganization rather than new information; RL directions steerable into base) predict that base Qwen already holds some lookahead and RL amplifies its use. That predicts probe accuracy may change little while causal reliance grows, which echoes the Ma & Rui probe-vs-patching dissociation and would itself be a striking result.
- Wu et al.'s myopia framing gives a second angle: RL reward on whole completions is non-myopic by construction, so it plausibly increases pre-caching. This is my inference, not a tested claim.

### Gaps
- No paper found measuring planning horizon before vs after RL. The 2026 RL-interpretability preprints were not read in full; their claims are snippet-level.

## 3. Tokenization and letter constraints

### Takeaway
Models mostly know the spelling of their tokens but use it poorly; character knowledge is reconstructed in middle/upper layers rather than present in embeddings. Hard token masking for letter constraints is known to distort outputs into misspellings, fragments, and lookalike characters, which is the "dead end / degenerate spelling" phenomenon.

### Cited Findings
- Hiraoka & Inui, "Spelling-out is not Straightforward" (arXiv 2506.10641, Jun 2025; Findings of EMNLP 2025, pp. 13340-13353). Embedding layer does not fully encode characters beyond the first; models reconstruct character knowledge in intermediate and higher layers, with a sharp "breakthrough" layer for spelling accuracy. Methods: probing classifiers, knowledge neurons, attention inspection. Single-token words only. ([ACL Anthology](https://aclanthology.org/2025.findings-emnlp.719/), [arXiv](https://arxiv.org/html/2506.10641v1))
- CUTE (Edman, Schmid, Fraser; arXiv Sep 2024; EMNLP 2024). Spelling, inverse spelling, containment, insertion/deletion/substitution/swap. Most models seem to know the spelling of their tokens yet fail to use it to manipulate text; similar or better performance on random strings than on real words. ([arXiv 2409.15452](https://arxiv.org/html/2409.15452v2))
- "Tokenization Falling Short" (arXiv 2406.11687, Jun 2024; Findings of EMNLP 2024). Tokenization is sensitive to typos and length variation and ignores token-internal structure; scale mitigates but does not remove it; BPE-dropout (subword regularization) helps. ([arXiv 2406.11687](https://arxiv.org/abs/2406.11687), [ACL Anthology](https://preview.aclanthology.org/dashboard/2024.findings-emnlp.86))
- "Gatsby without the 'E'" (Gokulakrishnan, Balasubramanian, Saba, Skiena; arXiv 2505.20501; IJCNLP-AACL 2025 short). Rewrites The Great Gatsby without "e"; constrained beam search with every e-containing token on a bad-words list removes all "e"; excluding letters up to 3.6% frequency (up to "u") barely changes meaning, but quality declines quickly and predictably as constraints tighten. PDF figure names suggest analyses of OOV words and grammar mistakes (not read). ([ACL Anthology](https://aclanthology.org/2025.ijcnlp-short.18/), [arXiv](https://arxiv.org/pdf/2505.20501v1))
- "Approximately Aligned Decoding" (Melcer et al., Amazon; arXiv 2410.01103, Oct 2024; venue not confirmed, NeurIPS checklist present). Lipogram experiment on Mistral-7B-Instruct-v0.2: avoid one vowel, 25 prompts, 200 tokens. Token-masking "constrained" decoding amplifies low-probability continuations after committing to a prefix and produces misspellings ("investiging", "computionl") and Cyrillic lookalike characters. Human ratings (quality 1-5 / constraint-intent 1-3): unconstrained 4.68/1.00, AprAD 4.52/2.84, constrained 3.56/2.32, ASAp 1.72/2.36 (ASAp hit the invocation cap). ([arXiv 2410.01103v2](https://arxiv.org/html/2410.01103v2))
- Locally constrained resampling notes greedy per-token masking is myopic. ([arXiv 2410.13111](https://arxiv.org/pdf/2410.13111))
- Grammar-constrained decoding literature names "token misalignment" between token boundaries and constraint units as the central difficulty; naive masking yields suboptimal token choices (DOMINO, arXiv 2403.06988, 2024; Park, Zhou, D'Antoni, arXiv 2502.05111, 2025). ([DOMINO](https://arxiv.org/pdf/2403.06988), [Park et al.](https://arxiv.org/abs/2502.05111))

### Inferences
- The AprAD lipogram result is direct prior evidence for "dead ends": masking tokens forces the model mid-word into e-free fragments (misspellings, homoglyphs). This is the decoding analog of what RL must avoid by planning at the word level, and gives a ready baseline: masked-decoding base model vs RL model, scored for e-rate, misspelling rate, and quality.
- A reward that only checks for "e" invites the same hacks AprAD saw: misspelled stems, Unicode lookalikes, truncation. Reward should whitelist ASCII and penalize non-dictionary words.
- Hiraoka & Inui imply "contains e" for a token is not readable from embeddings; it should become linearly decodable only after a mid-layer breakthrough. That gives a layer-resolved probe and a question: does RL move the "contains e" breakthrough earlier or make it sharper?
- Multi-token words matter: an e-free first token ("th") can commit to an e-containing word ("the"). Measuring "commitment points" where all likely completions of a partial word contain e is a concrete, novel dead-end metric.

### Gaps
- No work found on BPE-dropout or tokenizer-free (byte-level) models specifically for lipograms. No published analysis of Qwen3.5's tokenizer for e-containing token fraction (easy to compute locally).

## 4. Concrete probe designs feasible for a 4B model on a laptop

### Takeaway
All needed probes are linear/logit-lens style on cached residual activations, cheap on an M4 Pro once activations are dumped. Inference-only Qwen3.5-4B in bf16 needs ~8 GB, leaving room for activation caching. The critical addition is a causal test, per the probe-vs-patching gap.

### Cited Findings
- Linear probes on hidden states for tokens k ahead (Future Lens; Ma & Rui) and MLP probes on prompt states (Dong et al.) are the established designs. ([Future Lens](https://arxiv.org/abs/2311.04897), [Ma & Rui](https://arxiv.org/abs/2605.07984), [Dong et al.](https://arxiv.org/html/2502.06258v2))
- Future-Token Encoding via logit lens of SAE latents plus Precursor Influence via ablation is a published, horizon-agnostic criterion. ([Nainani et al.](https://arxiv.org/html/2508.18098v1))
- Activation patching at line boundary vs target word separates decodable-but-unused from causal plans. ([Ma & Rui](https://arxiv.org/abs/2605.07984))
- Unigram/frequency baselines are used to show probes learn more than token frequency. (search snippet of [arXiv 2605.07984](https://arxiv.org/pdf/2605.07984); not verified in full text)

### Inferences (proposed designs, not from papers)
- P1 "Contains-e" spelling probe: per layer, logistic probe on the residual at a token predicting whether that token's string contains "e". Expect a Hiraoka-style breakthrough layer. Compare base vs RL checkpoints.
- P2 Lookahead e-probe: at position t, predict whether the word starting k words ahead will contain "e" (in the model's own unconstrained continuation). Plot AUC vs k and layer; horizon k* = largest k with AUC above a frequency baseline. Headline: k* vs RL step.
- P3 Next-content-word identity probe (Future Lens style) restricted to e-free vs e-containing candidate sets.
- P4 Causal test: patch the residual at t from a prompt where the natural next word is e-free into one where it is not (or steer along the P2 direction), and measure change in e-rate of words k ahead. Compare base vs RL. This guards against the probe-only failure mode.
- P5 Dead-end rate: under token masking on the base model, count positions where the emitted prefix forces an e-word or misspelling (AprAD-style failures). RL success predicts fewer dead ends even without masking.
- P6 Myopia-style ablation (Wu et al.): restricted to the RL stage, e.g. compare RL with per-token reward vs sequence reward and see if only the latter extends k*. Feasibility depends on Tinker's reward granularity (not checked).
- Controls: frequency baseline, random-label probes, matched prompts, and base model given an explicit "avoid e" system prompt so improvement is not just instruction following.

### Gaps
- Whether Tinker exposes checkpoint weights for local download (needed for probes on base vs RL checkpoints) was not checked here.
- No published SAE/transcoder for Qwen3.5-4B found; P4 should use plain activation patching or probe-direction steering instead.
