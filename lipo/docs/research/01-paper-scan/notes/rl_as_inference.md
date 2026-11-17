# RL-as-inference and distribution matching for LLMs, applied to a lipogram (no "e") task

Scan date: 2026-11-16. Scope: 2021-2026 papers. Target under study: pi*(y|x) ∝ p_base(y|x)·exp(-λ·e_count(y)/β), with Qwen3.5-4B trained on Tinker.

Labels used below: **[fetched]** means I read the arXiv abstract or HTML page this session. **[memory, unverified]** means it comes from my prior knowledge and was not checked this session. The report writer should treat that material as a lead, not as a fact.

## Q1. Korbak/Perez/Buckley 2022, DPG/GDC, f-DPG, Korbak 2022 "RL and distribution matching": what they measured, at what scale, and whether they used lexical constraints

### Takeaway
The distribution-matching line (Naver Labs Europe, 2021-2023) is the closest prior work. It defines an explicit EBM target, trains a policy toward it, and reports KL to the target as a metric. Its experiments are at GPT-2 scale, and its "lexical" constraints were *include word X* (pointwise), not letter avoidance. Nothing in this line works at the 4B scale or with character-level constraints.

### Cited Findings
- f-DPG ("Aligning Language Models with Preferences through f-divergence Minimization", Go, Korbak, Kruszewski, Rozen, Ryu, Dymetman; arXiv 16 Feb 2023, rev. Jun 2023) generalizes DPG to any f-divergence toward any evaluable target, which unifies RLHF (reverse KL) and GDC (forward KL). It finds "no universally optimal objective" and reports that Jensen-Shannon often beats forward KL on the alignment/diversity trade-off. The abstract says results hold "as model size increases" but names no models. [fetched] — [arXiv 2302.08215](https://arxiv.org/abs/2302.08215)
- Korbak, Elsahar, Kruszewski, Dymetman, "On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting" (arXiv 1 Jun 2022, rev. 14 Nov 2022). It argues that KL-control RL can be viewed as distribution matching, and it imports baselines into DPG to cut gradient variance. It reports better "constraint satisfaction, stability and sample efficiency" on controllable generation that constrains topic, sentiment and gender distributions. The abstract names no models and gives no word-level constraints. [fetched] — [arXiv 2206.00761](https://arxiv.org/abs/2206.00761). (My memory, unverified: published at NeurIPS 2022, on GPT-2.)
- GDC (Khalifa, Elsahar, Dymetman, "A Distributional Approach to Controlled Text Generation", ICLR 2021). I fetched the PDF but could not extract its text this session. — [arXiv 2012.11635](https://arxiv.org/abs/2012.11635)
  - [memory, unverified] Base model GPT-2 small. Pointwise constraints were "text contains word w" for single words (examples I recall are "amazing", "WikiLeaks", "Vampire"), plus word lists and classifiers. There were also distributional constraints (e.g. 50% female biographies). Metrics: KL(p‖πθ) to the EBM target, KL(πθ‖a) to the base, constraint satisfaction rate, Self-BLEU and distinct-n. Z was estimated by importance sampling. The constraints were *inclusion* constraints. I recall none for avoiding a word or a character.
- Korbak, Perez, Buckley, "RL with KL penalties is better viewed as Bayesian inference" (2022). [memory, unverified] arXiv 2205.11275, EMNLP 2022 Findings. It is a short conceptual paper: KL-regularized RL equals variational inference toward the posterior p_base·exp(r/β)/Z, and it argues that this explains why RL without KL degenerates. I believe it has no new large experiments. Not fetched this session.
- The twisted-SMC paper (Q2) lists DPG as the proposal-learning method that minimizes KL(σ‖q) and PPO as the one that minimizes KL(q‖σ). [fetched] — [arXiv html 2404.17546](https://arxiv.org/html/2404.17546)

### Inferences
- The lipogram target is a GDC-style pointwise EBM: hard version p_base·1[no e], soft version p_base·exp(-λc/β). The DPG family (forward KL, with exact target samples available by rejection) is a natural baseline to run next to on-policy RL. f-DPG's finding that the choice of divergence trades fidelity against diversity predicts that reverse-KL RL will be more mode-seeking than DPG on the same target.
- Lexical avoidance at the *character* level is untested in this line. It is harder than word inclusion because the constraint touches nearly every token: about 1 in 8 English letters is "e", and most BPE tokens either contain "e" or are next to one that does.

### Gaps
- I could not confirm the exact GDC and DPG constraint words, model sizes and KL numbers, because PDF extraction failed (no pdftotext available).
- I did not confirm the f-DPG base models or tasks. My memory, unverified: GPT-2 with scalar-attribute and lexical tasks, plus a summarization task.

## Q2. GFlowNet fine-tuning, twisted SMC, LLaMPPL/SMC steering, Self-Steering LMs: tasks, metrics, sizes, letter constraints

### Takeaway
Twisted SMC (Zhao et al. 2024) is the only work found that turns log-Z bounds into a **head-to-head KL(q‖σ) / KL(σ‖q) table that includes PPO and DPG proposals**, and it does this only at 33M-355M scale. The SMC-steering line (LLaMPPL, Loula et al. 2025, DisCIPL 2025) scales to programmatic constraints on 1B-8B models. It reports task accuracy, not KL to the target. No paper found trains a policy toward a letter-avoidance posterior.

### Cited Findings
- **Twisted SMC** (Zhao, Brekelmans, Makhzani, Grosse; arXiv 26 Apr 2024; ICML 2024). It casts RLHF, red-teaming, prompting and infilling as sampling from an unnormalized target. Twists are learned with a contrastive loss (CTL) that connects to soft RL. The paper introduces **bidirectional SMC bounds on log Z**: the upper bound needs exact target samples, and the gap between the bounds bounds the symmetrized KL. [fetched] — [arXiv 2404.17546](https://arxiv.org/abs/2404.17546)
  - Models: TinyStories-33M for toxic-story generation and infilling, GPT2-Medium for sentiment. [fetched] — [html](https://arxiv.org/html/2404.17546)
  - Exact target samples come from ancestral sampling on simulated data (the BDMC trick), or from rejection sampling when the likelihood ratio is bounded, as it is for indicator-style targets with the base model as proposal. [fetched] — [html](https://arxiv.org/html/2404.17546)
  - Results, forward KL(q‖σ) / reverse KL(σ‖q):
    - Toxicity: PPO is best on forward at 0.98±0.01, but its reverse KL is 1.32±0.04. Contrastive is best on reverse at 1.07, with DPG at 1.12.
    - Sentiment: Contrastive is best on both, at 0.55 and 0.47. PPO is high-variance, at 1.04±0.31 and 0.87±0.20.
    - Infilling: DPG is best at 13.27 and 4.90, while PPO's reverse KL is 14.07.
    - The authors argue that the reverse-KL estimate diagnoses mode-dropping in mode-seeking PPO.
    - [fetched] — [html](https://arxiv.org/html/2404.17546)
- **GFlowNet fine-tuning**, "Amortizing intractable inference in large language models" (Hu, Jain, Elmoznino, Kaddar, Lajoie, Bengio, Malkin; arXiv 6 Oct 2023; ICLR 2024). It fine-tunes LLMs with GFlowNets, described as "diversity-seeking RL", to sample intractable posteriors. Tasks: sequence continuation, infilling, constrained generation, and chain-of-thought or tool use treated as latent variables. [fetched] — [arXiv 2310.04363](https://arxiv.org/abs/2310.04363)
  - [memory, unverified] The models were GPT-2 class and GPT-J 6B. Next-sentence continuation was scored by log-likelihood under the base and by diversity. The paper reports that reward-maximizing RL (PPO) collapses diversity compared with the GFlowNet.
  - Not confirmed from the abstract: model sizes and metrics. The abstract does not discuss length bias.
- **Loula et al.**, "Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo" (arXiv 17 Apr 2025; 15 authors including Lew, Vieira, Cotterell, Mansinghka, O'Donnell). It builds SMC on LLaMPPL with domain constraints and reallocates compute during generation. Tasks: Python for data science, text-to-SQL, goal inference, molecule synthesis. The authors report that small open models beat models 8x larger and closed fine-tuned ones, and they attribute the gains to a better posterior approximation. The abstract names no lexical constraints. [fetched] — [arXiv 2504.13139](https://arxiv.org/abs/2504.13139). The user's brief lists this as ICLR 2025, but the arXiv page shows no venue. I did not verify ICLR 2025.
- **Self-Steering LMs / DisCIPL** (Grand, Tenenbaum, Mansinghka, Lew, Andreas; arXiv 2504.07081, Apr 2025, rev. Aug 2025; COLM 2025, earlier at an ICLR 2025 workshop). A Planner LM writes a LLaMPPL inference program, and Follower LMs run it with SMC or IS. On the COLLIE and PUZZLES benchmarks, a Llama-3.2-1B Follower matches or beats GPT-4o and o1. Planner-written programs fall short of expert-written ones. — [arXiv 2504.07081](https://arxiv.org/abs/2504.07081); [ICLR listing](https://iclr.cc/virtual/2025/36211)
- **COLLIE** (Yao et al., ICLR 2024) has 2080 instances across 13 constraint structures, with constraints at the character, word, sentence and paragraph levels. GPT-4 reaches about 50.9% satisfaction and struggles with counting and positional constraints. I did not confirm whether it has a letter-*exclusion* constraint. — [arXiv 2307.08689](https://arxiv.org/abs/2307.08689v1)
- [memory, unverified] Lew et al. 2023, "Sequential Monte Carlo Steering of LLMs using Probabilistic Programs" (arXiv 2306.03081), introduced LLaMPPL. Its demos included hard lexical constraints, such as words under 5 letters and infilling, on LLaMA-7B-class models. It used qualitative examples and had no KL benchmark. Not fetched.

### Inferences
- "No e" is ideal for token-level SMC or locally constrained decoding. Masking every token that contains "e" gives a proposal whose importance weight is the product of the per-step allowed mass. That yields an unbiased estimator of Z for the hard constraint, along with SMC lower bounds. Rejection sampling from p_base gives exact target samples for short answers, which enables the Zhao-style upper bound. The lipogram therefore supports **bidirectional log-Z bounds at 4B scale**, which the twisted-SMC paper only did at 33M-355M.
- The SMC-steering papers already show that inference-time methods hit hard character-level constraints with small models. A trained-weights result is interesting only if it is framed as *amortization*: how close the trained πθ gets to π* compared with SMC at equal compute.

### Gaps
- Exact model sizes and metrics for Hu et al. and Loula et al. were not confirmed from full text.
- I found no paper that uses a lipogram or single-letter avoidance as a posterior-inference or RL target. That rests on searches covering abstracts only. It is absence of evidence, not proof.

## Q3. Is there work benchmarking PPO/GRPO/IS etc. by exact distance to a known KL-regularized optimum on a real LLM? What do COPR and arXiv 2601.16403 measure?

### Takeaway
I found none at real-LLM scale. Distance to π* is measured in three places:
- toy tabular settings (AGRO)
- tiny models with learned proposals (twisted SMC: TinyStories-33M, GPT2-Medium)
- theory (COPR's lower bound, and 2601.16403, which is a generalization bound that does not measure KL)

A 4B-scale benchmark of standard RL algorithms by KL(π‖π*) and KL(π*‖π) on a tractable target looks open.

### Cited Findings
- **arXiv 2601.16403** is "Towards a Theoretical Understanding to the Generalization of RLHF" (Li, Yi, Wang, Cui, Liu; 23 Jan 2026). It is a purely theoretical paper: with a linear reward model, algorithmic stability and a feature-coverage condition, the empirical-optimum policy has an O(n^-1/2) generalization bound, extended to GA and SGA. No KL-to-π* or log-Z measurement is described, and no specific models are named. [fetched] — [arXiv 2601.16403](https://arxiv.org/abs/2601.16403)
- **COPR** (Zhang et al., "COPR: Continual Human Preference Learning via Optimal Policy Regularization", ACL 2025 Findings, pp. 5377-5398; earlier arXiv 2310.15694 and 2402.14228). It is a non-RL offline continual-learning method built on the closed form π* ∝ π_ref·exp(r/β). It fits the policy to the optimal distribution *renormalized over sampled responses*, and regularizes toward stored historical optimal policies to reduce forgetting. Its theory gives a lower bound on KL(policy‖optimal) under insufficient sampling. I did not confirm whether it measures KL to π* empirically. — [ACL Anthology](https://aclanthology.org/2025.findings-acl.281); [arXiv 2402.14228](https://arxiv.org/html/2402.14228v2)
- **AGRO** (Tang, Cohen, Zhang, Valko, Munos; arXiv 2503.19612, Mar 2025) measures normalized KL to π* during training **only in a tabular toy**. It shows that KL-regularized PG with off-policy data does not converge to π*, while AGRO does. Its LLM experiments (Llama 3 8B on MATH) report only reward and accuracy. [fetched] — [arXiv 2503.19612](https://arxiv.org/html/2503.19612v2)
- **"A Comedy of Estimators: On KL Regularization in RL Training of LLMs"** (Shah, ..., Bengio, Jain, Venkatraman, Courville; arXiv 2512.21852, Dec 2025, v4 Aug 2026) analyzes the gradient bias of KL-estimator configurations (k1/k2/k3 and their placement in reward versus loss). It finds that biased configurations destabilize on-policy training and that KL helps asynchronous off-policy training. Models: Qwen2.5-7B, Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507. The abstract does not say whether it measures distance to π*. [fetched] — [arXiv 2512.21852](https://arxiv.org/abs/2512.21852)
- Other related work surfaced by search but not read:
  - A*-PO (arXiv 2505.20686) estimates V* offline from π_ref samples and reports near-optimal performance relative to π*.
  - Q♯ (arXiv 2502.20548) has PAC bounds to π* in deterministic MDPs.
  - "A Theoretical Lens for RL-Tuned LMs via Energy-Based Models" (arXiv 2512.18730) frames π* as an EBM.
  - "On KL-Regularized Policy Optimization" (arXiv 2610.08963) discusses how length-normalizing log-ratios changes the objective.
  - [2505.20686](https://arxiv.org/html/2505.20686); [2502.20548](https://arxiv.org/pdf/2502.20548); [2512.18730](https://arxiv.org/html/2512.18730); [2610.08963](https://arxiv.org/abs/2610.08963)

### Inferences
- For any trained πθ we can compute the exact ELBO, E_πθ[log p_base + r/β − log πθ], from its own samples, because p_base and πθ log-probs are both available on Tinker. Then KL(πθ‖π*) = log Z − ELBO. Given a tight log-Z estimate from IS, SMC or the masked-proposal estimator, every algorithm gets an exact-up-to-log-Z-error score. Rejection-sampled target samples give the forward KL(π*‖πθ). This is precisely the measurement missing at real-LLM scale.
- GRPO's group-mean or std normalization and its token-level KL-in-loss (k3) do not optimize the sequence-level objective whose optimum is π*. On a target with known π*, one can measure how far each variant's fixed point drifts. The "Comedy of Estimators" results suggest this drift depends on the estimator choice. The magnitude is untested here.

### Gaps
- Whether COPR reports an empirical KL-to-π* number is unconfirmed, because I did not read the full text.
- The full texts of A*-PO, 2512.18730 and 2610.08963 were not read for whether they measure KL to π* on real models.

## Q4. Length bias: does p(y | constraint) favor shorter sequences, and has RL "answer shrinkage" been shown to be a correct consequence of the objective?

### Takeaway
I found no paper that states and tests "KL-regularized optimum under a per-occurrence penalty or hard avoidance constraint shortens outputs, so RL shrinkage is correct rather than reward hacking." The math makes it nearly certain for the lipogram (see Inferences), so this looks like an open, cleanly testable point.

### Cited Findings
- DRPO (arXiv 2510.04474) derives, in an appendix, the Gibbs optimum π_ref·exp(length reward/λ) for an explicit length reward under KL. This is length as a reward, not length as a side effect of a constraint. — [arXiv 2510.04474](https://arxiv.org/pdf/2510.04474)
- KLPO (arXiv 2610.08963) notes that length-averaging log-ratios changes the residual in KL-regularized optimization. This is a training-objective detail, not a claim about the length of π*. — [arXiv 2610.08963](https://arxiv.org/abs/2610.08963)
- A targeted search for "the KL-constrained optimum biases toward shorter responses" found no direct source. — (search log, no URL)

### Inferences (my derivation, not from a source)
- Hard constraint: p(y|no e) ∝ p_base(y)·1[e_count(y)=0]. If each token independently has probability q_t of containing "e", the constraint factor is about ∏(1−q_t), which decays roughly geometrically in length. The posterior therefore reweights the base length distribution by about (1−q̄)^L, so **shorter answers are favored by the exact target**. Early EOS is "cheap" evidence.
- Soft penalty: exp(−λ·c/β) with E[c] ≈ ρ·L gives roughly exp(−λρL/β), again geometric in L. So π* shortens answers by an amount predictable from base statistics.
- Prediction: some RL "answer shrinkage" is the correct optimum. One can split the observed length drop into (a) the part π* predicts, from the length distribution of rejection-sampled or SMC-weighted base samples, and (b) any excess due to mode-seeking reverse KL or GRPO normalization. Part (b) is the real algorithmic artifact.
- If the desired behavior is "same answer, no e", the target itself must change: condition on length, add a length or answer-quality potential, or use a per-token rate penalty. That is a target-design choice, not an RL fix.

### Gaps
- No empirical paper found that measures the length of π* compared with the RL-trained policy on any constraint task.
- [memory, unverified] Length-bias discussions in RLHF (e.g. Singhal et al. 2023, "A Long Way to Go") concern reward models favoring *longer* answers. That is the opposite direction and a different mechanism. Not fetched.

---

### Summary of what is done and what is open (synthesis, inference)

**Done**
- The theory that KL-regularized RL equals variational inference toward p_base·exp(r/β).
- DPG/GDC/f-DPG distribution matching with KL-to-target metrics, at GPT-2 scale with word-inclusion constraints.
- Bidirectional log-Z / KL tables that include PPO and DPG proposals, at 33M-355M.
- SMC steering of 1B-8B models for hard programmatic constraints, measured by task accuracy.
- Tabular demonstrations that PG misses π* off-policy.

**Open**
1. Exact or bounded KL(πθ‖π*) and KL(π*‖πθ) for PPO, GRPO, IS-PG, DPG and on-policy distillation on a ≥4B model with a tractable target.
2. Character-level avoidance as a posterior target.
3. Decomposing length shrinkage into the share π* predicts and the excess that is an artifact.
4. Amortization gap: trained weights compared with SMC at matched compute, scored on the same log-Z scale.

### Candidate ideas likely to interest Thinking Machines / Schulman (inference)
- **"Lipogram-KL" benchmark.** Use log Z from a masked-token IS/SMC estimator plus rejection-sampled exact targets to score each RL recipe on Tinker by both KL directions against π*. Cheap reward, exact base logprobs, 4B scale, and the first real-LLM table of its kind.
- **"Shrinkage is correct" result.** Predict the length distribution of π* before training, then show which part of the RL length drop is optimal and which part is algorithm-induced, for GRPO normalization and for k1/k3 KL placement.
- **Estimator ablation against ground truth.** Rerun the "Comedy of Estimators" configurations, but score them by KL to π* instead of downstream accuracy.
- **Amortized SMC.** Distill SMC-weighted samples into LoRA weights (in the spirit of Hu et al.) and plot KL to π* against training compute, compared with on-policy RL.
- **Schulman tie-in.** Schulman's "Approximating KL divergence" note (k1/k2/k3) is the origin of the estimators. [memory, unverified: joschu.net blog, 2020]
