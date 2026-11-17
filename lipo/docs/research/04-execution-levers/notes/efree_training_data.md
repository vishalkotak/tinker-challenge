# Sources and generation methods for e-free SFT data (lipogram warm start for Qwen3.5-4B)

Scope: the cheapest way to get thousands of good English Q&A pairs whose answers have no ASCII "e"/"E", for SFT before RL on Tinker. Researched 2026-11-16. Project Gutenberg returned HTTP 503 during this session, so Gutenberg facts come from Wikipedia and the Hugging Face card, not the Gutenberg page itself.

## 1. Gadsby (Wright, 1939): status, size, errata, usefulness

### Takeaway
Gadsby is freely available (Project Gutenberg ebook 47342, plus Wikisource, Internet Archive, Faded Page and LibriVox). It is "over 50,000 words" (50,110 is the usual figure) and has at least four known slips in the 1939 printing. As SFT data for chat answers it has already been tested and transferred poorly. Use it as a vocabulary and phrasing reference, not as SFT targets.

### Cited Findings
- Online copies: Project Gutenberg ebook 47342, Wikisource, Internet Archive scan, Faded Page (Canada) and a LibriVox audiobook marked public domain. Wikipedia does not state a copyright status for the text itself. — [Wikipedia: Gadsby (novel)](https://en.wikipedia.org/wiki/Gadsby_(novel))
- Subtitle: "A Story of Over 50,000 Words Without Using the Letter 'E'". 260 pages, Wetzel Publishing Co., 1939. — [Wikipedia](https://en.wikipedia.org/wiki/Gadsby_(novel))
- 50,110 words is the commonly cited count. — [Today I Found Out](https://www.todayifoundout.com/index.php/2012/05/the-novel-gadsby-has-50110-words-yet-none-of-them-contain-the-letter-e/) (secondary source)
- Known errata in the 1939 printing: "the" on pp. 51, 103 and 124, and "officers" on p. 213. The introduction contains "e" because it is not part of the story. Abbreviations count only if the full form avoids e ("Dr." and "P.S." are fine, "Mr." is not). — [Wikipedia](https://en.wikipedia.org/wiki/Gadsby_(novel))
- The Lipogram-e HF dataset includes all of Gadsby, taken from Project Gutenberg. — [HF Lipogram-e card](https://huggingface.co/datasets/Hellisotherpeople/Lipogram-e)
- Prior transfer test (Qwen3-1.7B, kevinyvv repo): SFT on Gadsby used 48 train and 5 val chunks of up to 2,048 tokens, 7 epochs and 168 updates. Val loss fell from 4.09 to 3.24, but chat passes stayed at 1/50 on validation and went from 2/100 to 8/100 on test. The Gadsby adapter scored 24.5% "usable" on the short eval, versus 26% for the masked baseline with no training. — [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md); [README](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj)
- Same repo: Gadsby adapter plus zeroed e-token output weights gave 100/100 literal passes, and "e-equivalent" answers fell from 15 to 3. — [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)

### Inferences
- The four errata are minor, but any pipeline should run the e-check on every Gadsby chunk and drop or patch those lines, since the book isn't perfectly e-free.
- Gadsby's 1930s narrative register doesn't match the Q&A style. The measured transfer to chat answers was nil to slightly negative. Its best uses are mining an e-free vocabulary and collocation list for teacher prompts, or a small replay share (well under 10%) during SFT.
- Mac check this session: 77,849 of 235,976 entries (33%) in /usr/share/dict/words have no e. So do 4,271 of the 9,894 google-10000 common words (43%), and 465 of the top 1,000. That gives a usable dictionary for OOV checks and synonym hints without Gadsby.

### Gaps
- I could not open the Gutenberg page (503), so I could not confirm its copyright notice ("Public domain in the USA") or file sizes.
- There is no authoritative word count beyond the subtitle and secondary sources. I found no full errata list beyond the four slips Wikipedia gives.

## 2. Existing lipogram corpora and SFT sets

### Takeaway
There is only one ready lipogram corpus, Lipogram-e (MIT on HF, 23.4k rows, 803 kB). Two of its three books, A Void and Eunoia, are modern copyrighted works, so its MIT tag is doubtful for those parts. The only e-free Q&A set is the kevinyvv repo's 200 verified answers (MIT code, prompts inheriting Dolly and No Robots licenses). I found no "prateekfr" lipogram data.

### Cited Findings
- **Lipogram-e** (Roush et al., CAI workshop at COLING 2022): MIT license, 23,419 rows, 803 kB, one `text` column of 0 to 662 characters, train split only. Contents are all of Gadsby, all of Perec's A Void (Adair translation) and all of Bök's Eunoia except its one e-chapter. It was extracted with pandoc from PDF/EPUB. Three conversion-hallucinated e's in A Void were fixed by hand, and the card warns that "at least a few OCR errors exist". — [HF card](https://huggingface.co/datasets/Hellisotherpeople/Lipogram-e)
- The Roush paper reports a GPT-2 ignored-constraint rate of 28.2% with no fine-tuning, 0.5% after 5 epochs on Lipogram-e, 0.3% after 20 epochs, and 0% with its token filter. These numbers come via an aggregator marked "unverified". Code: hellisotherpeople/constrained-text-generation-studio. — [arXiv 2306.15926](https://arxiv.org/pdf/2306.15926); [ACL Anthology 2022.cai-1.2](https://preview.aclanthology.org/setup/2022.cai-1.2)
- **kevinyvv/qwen-3-1.7b-finetune-proj** (MIT): 200 hand-verified e-free answers in `data/rewrites/verified_200_20261015.jsonl`, split into SFT sets of 15 to 180. There is also a 1,235-prompt RL pool (`rl_pool.jsonl`, about 900 kB) and a 50-prompt eval (`short_eval_50.jsonl`), plus `short_eval_dropped.jsonl` with the 25 prompts that can't be answered without e. Prompts come from No Robots (CC BY-NC 4.0) and Dolly (CC BY-SA 3.0), and the repo says those licenses carry over. — [README](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj); file listing via GitHub API (`data/prompts/`)
- The "75 e-answerable prompts" are the 75 validation prompts that can be answered without e. The eval uses the first 50, and 25 others were dropped with reasons. — [experiment_history.md §9](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- Answers were drafted by GPT models running as Codex subagents, then reviewed and e-checked. Each row records the prompt source and the check used. — [README](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj)
- SFT-size curve on Qwen3-1.7B under the mask: 0 examples gave 26.0% usable, 90 gave 31.5%, 180 gave 35.0% (+9.0, CI +0.5 to +17.5). With the instruction alone, 180 examples reached only 9.0% no-e and 4.5% usable. The authors wrote: "The curve is leveling off... More SFT data of this kind is unlikely to be worth writing." 15 examples hurt (10.0%). — [experiment_history.md §11](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- Best result: SFT adapter plus 300 GRPO steps reached 60.5% usable, against 80.5% for unconstrained Qwen3-1.7B. — [README](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj)
- Qwen3-4B was worse than 1.7B under the mask. It keeps its planned words and drops the e's ("you manag an amusrmt park"), writing "th" for "the" in 51/200 answers versus 1/200 for 1.7B. Judge notes flagged misspellings in 130/200 answers versus 68/200. — [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- Earlier rewrite corpora were weak. 95 strict rewrites had mean fluency 3.07/5 with only 12/95 at 4 or above. Looser rewrites reached 3.31/5 (29/95 at 4 or above), but 57/95 had factual-error flags, and 44/95 contained non-ASCII typography. — [experiment_history.md §3](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- GitHub search for "lipogram" found only small or toy repos (LeoHink/lipogramGPT with nanoGPT+PPO, 6 stars; Loujainl/lipogram_e; chloerevery/BelleAbsenteBot, 11 stars), none with a sizable e-free Q&A set. User `prateekfr` has an `easy-rlvr-implementation` repo (MIT, GRPO arithmetic and date tasks), not lipogram data. — GitHub API search (2026-11-16); [prateekfr/easy-rlvr-implementation](https://github.com/prateekfr/easy-rlvr-implementation)
- "Gatsby without the E" (Balasubramanian et al., IJCNLP-AACL 2025) promised to release its lipogram Gatsby translation and code "upon acceptance", but no link appears in v2. — [arXiv 2505.20501v2](https://arxiv.org/html/2505.20501v2); [ACL Anthology](https://aclanthology.org/2025.ijcnlp-short.18)
- The HF "LLM Lipogram Challenge Portal" is a course exercise (logit masking for no-e text and no "Toulouse") with no dataset. — [HF Space](https://llm-course-lipogram-challenge-submission.hf.space/)

### Inferences
- Best reuse: Wu's 200 verified answers as a seed and few-shot pool, the 1,235-prompt pool, and the dropped-prompt reasons, which show how to filter prompts for answerability. The 1.7B SFT curve suggests that just more of the same small data has diminishing returns. The warm start probably needs better-quality and better-matched data (Qwen-style, ASCII-only), not only more of it.
- The 4B misspelling finding matters for Qwen3.5-4B. SFT targets should show whole-word swaps ("the" becomes "this", "that" or "a"), never truncations, to counter the mask-induced "drop the e" habit.
- Lipogram-e's A Void and Eunoia sections are modern copyrighted books (1994 translation, 2001 poetry). If redistribution matters, use only the Gadsby portion.

### Gaps
- I found no Oulipo collections or Kaggle lipogram datasets of note. Searches surfaced none, so this may be a true absence.
- The exact row count and licensing of each kevinyvv file beyond the README were not checked.

## 3. Verifier-in-the-loop generation: evidence and cost

### Takeaway
Programmatic external feedback reliably helps but plateaus. COLLIE's GPT-4 multi-turn feedback went from 50.9% to about 66% by round 3. Self-Refine gains shrink per round (+11.3, +6.4, +3.0). For lipograms specifically, frontier prompting alone leaks e's (GPT-4o 5.94% e%), and even GPT-6-class Codex subagents failed every 2k-token first draft. Near-100% acceptance comes from feedback that names the offending words, plus resampling or rejecting what still fails, with short answers. An unconstrained loop alone won't get there.

### Cited Findings
- **Gatsby paper (2025):** GPT-4o prompted rewriting had e% 5.94 with cosine similarity 0.74 and grammar mistakes 1.62 per paragraph. E-Delete (0% e) had GM 29.26, and WordNet synonym swap (0% e) had GM 24.85. Llama3-8B with fine-tuning, e-filtered beam search (20 beams), multiselection and post-processing reached 0% e, CS 0.71 and GM 1.08. Without fine-tuning, the Llama3 pipeline's OOV rose to 63.17% and GM to 26.47. There was no iterative feedback loop; multiselection is pick-best-of-N. The paper is internally inconsistent: the text says "100% adherence" while the table shows GPT-4o at 5.94, and the e% column is undefined. — [arXiv 2505.20501v2](https://arxiv.org/html/2505.20501v2)
- **COLLIE (Yao et al. 2023):** GPT-4 zero-shot average constraint satisfaction was 50.9%, and pass@20 was above 63%. With automated natural-language feedback after each attempt, round 2 rose about 20% and the rate plateaued around 66% after three feedback rounds, "comparable to pass@5". Character-position and exact-count tasks (word02, sent01, sent02) stayed hard. COLLIE has no letter-exclusion constraint; the closest is para02, which bans the words "the", "and" and "of". — [ar5iv 2307.08689](https://ar5iv.labs.arxiv.org/html/2307.08689)
- **Self-Refine (Madaan et al. 2023):** on Constrained Generation (concept coverage), GPT-4 went from 15.0 to 45.0, ChatGPT from 44.0 to 67.0, and GPT-3.5 from 28.0 to 37.0. Averaged scores across iterations were y0 29.0, y1 40.3, y2 46.7, y3 49.7, with a maximum of 4 iterations. In other tasks, specific feedback beat generic feedback, which beat none (Sentiment Reversal: 43.2, 31.2, 0). — [ar5iv 2303.17651](https://ar5iv.labs.arxiv.org/html/2303.17651)
- **Kamoi et al. (TACL 2024) survey:** self-correction from prompted LLM feedback isn't reliable in general tasks. It "works well in tasks that can use reliable external feedback", and the bottleneck is feedback generation. — [arXiv 2406.01297](https://arxiv.org/html/2406.01297)
- **AutoIF (Qwen team, ICLR 2025):** LLM-written Python verification functions plus rejection sampling (keep only responses that pass) built SFT and DPO data for instruction following. Reported IFEval was up to 90.4% (figure from slides; verify in the paper). — [arXiv 2406.13542](https://arxiv.org/html/2406.13542v3); [ICLR slides](https://www.iclr.cc/media/iclr-2025/Slides/29049_CxG7yOx.pdf)
- **Divide-and-Conquer for lexically constrained generation (EMNLP 2024):** split the constraints into satisfied and missed, regenerate for the missed ones, then merge. It reports about 90%+ success improvement on the hardest task. The paper's two framings of the number are ambiguous. — [arXiv 2410.04628](https://arxiv.org/html/2410.04628v1)
- **Direct lipogram teacher pilot (kevinyvv, 2026-10-06):** GPT-6-class Codex subagents, with one draft and two full repairs, had to write 2,048 to 2,560-token no-e ASCII passages. All 9 first drafts failed the mechanical checks. Accepted results: gpt-6-luna 0/3, gpt-6.1-sol 1/3, gpt-6-astra 2/3. Sol's failing drafts kept 2 and 4 e's and were fixed with extra targeted repairs. The estimated cost was $0.49 to $2.15 per accepted 2k passage, at assumed API rates (an estimate, not a bill). Output tokens including reasoning were 35.8k to 63.2k per model for 3 tasks. — [experiment_history.md §4](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- **Claude subagent pilot (same repo):** all 9 saved short rewrites were e-free, at 53 to 257 seconds each. — [experiment_history.md §3](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- **Tinker prices (saved docs):** Kimi-K2.6 costs $2.205/M prefill ($0.441 cached) and $5.49/M sample. DeepSeek-V4.1-Flash costs $0.60 ($0.12) and $3.60. GPT-OSS-120B costs $0.33 ($0.066) and $0.84. GPT-OSS-20B costs $0.18 and $0.45. Qwen3.5-397B-A17B costs $1.65 and $7.50. With `num_samples > 1`, the shared prompt's extra prefill is billed as cache hits. — Tinker docs `models_models_and_pricing.md`, `sampleresponse.md` (local copies in the session scratchpad, from [tinker-docs.thinkingmachines.ai](https://tinker-docs.thinkingmachines.ai/))
- The saved Tinker SamplingClient docs list temperature, max_tokens and logprob options. I saw no logit-bias or bad-words parameter, so teacher-side e-token masking on Tinker isn't confirmed. — Tinker `api-reference_samplingclient.md` (local copy)

### Inferences (cost model, my estimates, not measured)
- **Short answers are the main cost lever.** The 2k-token pilot failed because the chance of zero e's falls off steeply with length. Q&A targets of 80 to 250 words are much easier. COLLIE's plateau (about 66%) and Self-Refine's shrinking gains suggest 2 to 3 feedback rounds, then drop or resample. Don't loop further.
- **Feedback format:** the verifier should return the exact offending words with positions, plus suggested e-free swaps from the dictionary list. Kamoi and Self-Refine both say specific, reliable feedback is what works. The fix should be "rewrite only these words" (DnC-style), not a full regeneration.
- **Token budget per accepted answer (estimate):** take a non-thinking teacher (Kimi-K2.6 or DeepSeek-V4.1-Flash hybrid with thinking off) and about 250 output tokens per attempt. Assume `num_samples=4` on the first draft, then 2 repair rounds on the best draft, for about 1,500 to 2,000 output tokens per prompt. At $5.49/M that is about $0.008 to $0.011 per prompt. Prefill is about 1 to 2k tokens at $0.44 to $2.2/M, adding around $0.002 to $0.004. With an assumed 50 to 70% acceptance, that comes to about $0.015 to $0.02 per accepted answer on Kimi, so 2,000 answers would cost about $30 to $40. On GPT-OSS-120B at $0.84/M, even with about 2k reasoning tokens per attempt, the cost is about $0.005 to $0.01 per accepted answer, so 2,000 would cost about $10 to $20. DeepSeek-V4.1-Flash falls in between. **"A few dollars" buys a few hundred to about 1,000 answers, not thousands, unless you use GPT-OSS-20B/120B or very short answers.** These numbers are extrapolations and should be confirmed with a 50-prompt pilot that measures real acceptance and tokens.
- **Free alternative on the Mac:** a masked local generator (Qwen3.5-4B or 9B with e-tokens banned via logits) gives 100% literal compliance. The cost moves to quality filtering: as seen above, the 4B model writes misspelled e-dropped words under a mask. The best mix is probably masked local drafts plus a teacher "fix misspellings and fluency" pass, with a re-check after.
- Starting from Qwen's own answers had lower SFT loss (2.82 vs 6.88 on one example, not conclusive). That favors "teacher rewrites the Qwen3.5-4B unconstrained answer" over "teacher answers from scratch". — basis: [experiment_history.md §3](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)

### Gaps
- I found no published measurement of current frontier models (Kimi-K2.6, DeepSeek-V4.x, gpt-oss) on short e-free Q&A, with or without feedback. The per-round acceptance figures above are extrapolated from COLLIE and Self-Refine, not lipogram-specific.
- I could not confirm whether Tinker sampling supports logit bias or banned tokens.
- I could not check whether IFBench's 58 constraints include letter exclusion (the appendix wasn't retrieved). — [arXiv 2507.02833](https://arxiv.org/html/2507.02833v1)

## 4. Quality filtering

### Takeaway
Use a stack. (1) An ASCII-only, case-insensitive e-check that also rejects any non-ASCII character. (2) A dictionary or OOV check that catches e-dropped misspellings ("th", "manag"). (3) A cheap LLM judge for coherence and on-topic answers, with a usable bar of at least 4/5 on both. Measure the judge against hand labels first.

### Cited Findings
- Gatsby paper metrics: OOV % against a vocabulary, LanguageTool grammar-mistake counts, and Proselint in post-processing. E-Delete's OOV of 51.32% and GM of 29.26 show how naive substitutions get caught. — [arXiv 2505.20501v2](https://arxiv.org/html/2505.20501v2)
- kevinyvv: "usable" means no e plus a judge score of at least 4/5 for coherence and answering the prompt. The judge (gpt-6-luna, reasoning off, about 550 input and 43 output tokens per call) was validated against labeled checks (98%, 92%, 0/48 false passes). Judge v3 added a firmer "facts don't count" rule and labels cut-off answers so they aren't penalized for truncation. — [experiment_history.md §9](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- The same repo found 44/95 rewrites had non-ASCII typography (curly quotes and the like), which makes ASCII-only decoding a stricter condition than the training data. — [experiment_history.md §3](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- The same repo saw judge reward hacking under RL (answers shrinking, and a short answer scoring about the same as a usable 4+4). — [experiment_history.md §13](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)

### Inferences
- OOV check: tokenize words and accept a word only if it is in an e-free dictionary (system words list, google-10000 e-free subset) or is a number or proper noun. Flag any word that becomes a dictionary word when an "e" is inserted ("th" becomes "the", "manag" becomes "manage"); that catches the 4B failure mode directly.
- Normalize or reject non-ASCII before the e-check (é, É and the full-width E are not ASCII "e" but read as e). Strip curly quotes so the SFT targets match the ASCII-only eval.
- Also set a minimum length or completeness rule, so the SFT set doesn't teach the "short answer" shortcut the RL judge rewarded.
- The judge can be GPT-OSS-120B on Tinker (about 600 input and 50 output tokens, roughly $0.0003 per call), validated against Wu's 200 verified answers plus deliberately broken variants.

### Gaps
- I found no published accuracy for LLM judges on lipogram fluency specifically, beyond the kevinyvv in-house check.

## 5. Prompt sources for Q&A

### Takeaway
Dolly-15k (CC BY-SA 3.0, commercial OK with share-alike) and No Robots (CC BY-NC 4.0, non-commercial) together give about 25k human prompts. Filter both for e-answerability, which drops prompts that force e-words, such as named entities, quotes or rewrite tasks. Wu's 1,235-prompt pool is a ready, already-filtered subset.

### Cited Findings
- **Dolly-15k:** CC BY-SA 3.0, about 15,011 rows, 8 categories (brainstorming, classification, closed QA, creative writing, general QA, information extraction, open QA, summarization). The `context` field is Wikipedia text, also under CC BY-SA. The card recommends stripping bracketed citation numbers. — [HF card](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- **No Robots:** CC BY-NC 4.0, train_sft 9,500 and test_sft 500. Categories: Generation 4,560, Open QA 1,240, Brainstorm 1,120, Chat 850, Rewrite 660, Summarize 420, Coding 350, Classify 350, Closed QA 260, Extract 190. — [HF card](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- kevinyvv filtered both into a 1,235-prompt RL pool with recorded keep/drop decisions (`rl_pool_decisions.jsonl`, about 585 kB). In the validation set, 75 prompts were judged e-answerable and 25 dropped, with reasons. — [repo](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj); [experiment_history.md §9](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)
- The same history notes a model-based prompt filter mostly gave one identical drop reason, and only 11% of its keeps had been dropped by hand. — [experiment_history.md](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/blob/main/docs/experiment_history.md)

### Inferences
- About 75% of prompts survived the validation filter, so roughly 18k of the 25k Dolly plus No Robots prompts could be usable. Prompt supply is not the bottleneck; generation cost and quality are.
- Prefer Open QA, Brainstorm, General QA and Generation. Drop Closed QA, Extract, Summarize and Rewrite, which force reuse of e-heavy source text.
- To keep the SFT set redistributable, use Dolly only (share-alike). Anything including No Robots must stay non-commercial.
- Hold out Wu's 50 eval prompts and their source items from the SFT set to avoid leakage. The 9-answer Claude pilot was discarded for exactly this reason.

### Gaps
- I didn't measure the per-category e-answerable rate across all of Dolly and No Robots. It would need running a filter; the 75% figure comes from one validation set.
