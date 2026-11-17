# Tinker platform feasibility facts for the lipogram RL project (as of 2026-11-16)

Sources checked: live docs at tinker-docs.thinkingmachines.ai (fetched 2026-11-16), the installed SDK at `.venv/lib/python3.13/site-packages/tinker` (tinker 0.29.1) and `tinker_cookbook` (0.5.5), `lipogram.py` in the same lab, GitHub via `gh` (read-only). No API calls with credentials were made.

Version note: the live docs are generated from a newer SDK (commit 4801e02 of thinking-machines-lab/tinker). PyPI's latest `tinker` is **0.39.1** (uploaded 2026-11-16T18:56) — [PyPI](https://pypi.org/pypi/tinker/json). Several doc features below are **not in 0.29.1**; each is flagged.

## 1. Pricing and cost of one default lipogram.py run

### Takeaway
Qwen/Qwen3.5-4B costs $0.33/M prefill ($0.066 cached), $1.005/M sample, $0.737/M train. A default 10-step run costs roughly **$0.24 without the judge and about $0.40 with Inkling-Small judging all 64 samples per step** (worst case, every sample hits 200 tokens). Using Inkling 2 Small as judge would cut judge cost by about half.

### Cited Findings
- Prices per million tokens (Training table: Prefill / Sample / Train) — [Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md):
  - `Qwen/Qwen3.5-4B` (64K ctx, dense, Hybrid + Vision): $0.33, $0.066 cached / $1.005 / $0.737
  - `Qwen/Qwen3.5-4B:peft:262144` (256K): same prefill/sample, train $1.34
  - `thinkingmachines/Inkling-Small` (276B total, 12B active MoE, 64K): $0.58, $0.116 cached / $1.44 / $1.73
  - `thinkingmachines/Inkling-2-Small` (594B total, 17B active MoE, 64K): $0.30, $0.03 cached / $1.00 / $1.50
  - `thinkingmachines/Inkling` (975B, 41B active): $1.87, $0.374 cached / $4.68 / $5.61
  - Other cheap candidate judges: `Qwen/Qwen3-8B` $0.195/$0.60; `openai/gpt-oss-20b` $0.18/$0.45; `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` $0.184/$0.495 (limited-time 50% discount); `Qwen/Qwen3.6-35B-A3B` $0.45/$1.335; `Qwen/Qwen3.5-9B` $0.30/$0.807.
- Serverless inference (beta, Inkling family only): `thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4` $0.30 in, $0.06 cached, $1.20 out; `thinkingmachines/Inkling-2-Small:peft:262144` $0.30 / $0.03 / $1.00 — [Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md).
- Pricing terms: "Train: Forward and backward pass ... forward only passes invoked by training clients are billed at the training price"; cached prefill applies to input tokens that hit the prompt cache — [Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md).
- With `num_samples > 1`, the shared prompt is counted once; prefill for the remaining `num_samples - 1` samples is billed as cache hits (`SampleResponse.prompt_cache_hit_tokens` docstring) — installed SDK `tinker/types/sample_response.py`.
- Machine-readable prices: `https://tinker-docs.thinkingmachines.ai/tinker/models.json` and `serverless.json` — [Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md).
- Checkpoint storage: free up to **100 GB or 100 checkpoints per organization**, whichever comes first; on hitting a free limit "your session saves one last checkpoint and stops"; paid plan $0.10/GB/month for all storage — [Checkpoint Storage](https://tinker-docs.thinkingmachines.ai/administration/checkpoint-storage-pricing/index.md).
- lipogram.py defaults: `BASE_MODEL="Qwen/Qwen3.5-4B"`, `LORA_RANK=8`, `GROUP_SIZE=8`, 8 `TRAINING_PROMPTS`, `TRAINING_STEPS=10`, `MAX_TOKENS=200`, `LEARNING_RATE=8e-4`, `JUDGE_MODEL="thinkingmachines/Inkling-Small"`, judge `max_tokens=8, temperature=0.0`; it calls `save_weights_for_sampler` 11 times (step 0 plus each step) with `ttl_seconds=7*24*3600`; and in `reward()` the judge call is commented out (`grade = 0  # await judge.grade(prompt, text)`), so by default the judge runs only once, on the final eval — `lipogram.py`.

### Inferences
- Cost per step, worst case (all samples hit 200 tokens). I assumed about 20 prompt tokens per rendered Qwen chat prompt and about 420 judge prompt tokens (about 170 for the grading instructions, plus the question, up to 200 answer tokens, and template overhead). These token counts are my estimates, not measured.
  - Policy sampling: 64 x 200 = 12,800 tokens x $1.005/M = **$0.0129**
  - Policy prefill: 8 x 20 uncached + 56 x 20 cached ≈ $0.0001 (negligible)
  - Training `forward_backward`: 64 datums x ~220 tokens = 14,080 x $0.737/M = **$0.0104**
  - Judge, Inkling-Small: 64 x 420 = 26,880 prefill x $0.58/M = $0.0156, plus 64 x 8 = 512 sample x $1.44/M = $0.0007, for **$0.0163 per step**
  - Same judge on Inkling 2 Small: 26,880 x $0.30/M + 512 x $1.00/M ≈ **$0.0086 per step**
- Per run (10 steps): about **$0.23 to $0.24 with no judge** (the current default, since grade=0), **about $0.40 with the Inkling-Small judge on every sample**, and **about $0.32 with Inkling 2 Small**. The shared system prompt could hit the prefix cache, which would make the judge cheaper still. Real samples usually stop before 200 tokens, so actual cost should come in lower.
- Storage: 11 sampler checkpoints per run means the 100-checkpoint free cap is reached after about 9 runs inside the 7-day TTL window. Shorter TTLs or fewer saves would avoid the "session saves one last checkpoint and stops" failure.

### Gaps
- I found no published per-request minimum charge or per-checkpoint-save fee. Billing granularity was not verified.
- Actual token counts depend on the tokenizer and template. They are not measured here because no API calls were allowed.

## 2. Is LoRA rank 1 accepted?

### Takeaway
The SDK enforces no minimum rank on the client side. `rank` is a plain `int` sent to the server, and TML's own research says rank 1 is enough for RL. Whether the server accepts `rank=1` for Qwen3.5-4B is not documented anywhere I found, so it needs one cheap live test.

### Cited Findings
- Signature (0.29.1): `ServiceClient.create_lora_training_client(base_model: str, rank: int = 32, seed: int | None = None, train_mlp: bool = True, train_attn: bool = True, train_unembed: bool = True, user_metadata=None, optimizer: types.OptimizerConfig | None = None)`; async `create_lora_training_client_async` has the same parameters — installed SDK `tinker/lib/public_interfaces/service_client.py` lines 276-340.
- `types.LoraConfig` has `rank: int`, `seed`, `train_unembed`, `train_mlp`, `train_attn` and no validator or bounds; grep for rank limits in `tinker/` and `tinker_cookbook/` found none — installed SDK `tinker/types/lora_config.py`.
- "The default rank used by Tinker is 32 ... For reinforcement learning, we've found that small ranks give equivalent performance to larger ranks and full fine-tuning"; "the optimal learning rate does *not* depend on the LoRA rank" — [LoRA Primer](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/index.md).
- "Even at rank-1, LoRA has more than enough capacity to absorb all the information provided during training" (RL section) — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/).
- LoRA parameter counts at rank 1 for `Qwen/Qwen3.5-4B`: mlp 1,130,496; attn 897,024; unembed 250,880 (total 2,278,400 per unit of rank) — installed `tinker_cookbook/hyperparam_utils.py` (`get_lora_param_count`).
- The cookbook README calls 16-64 typical and uses 32 as the default — installed `tinker_cookbook/weights/README.md`.

### Inferences
- A rank-1 Qwen3.5-4B adapter has about 2.28M params, about 4.6 MB in bf16 (about 9 MB in fp32). Rank 8 (the lipogram default) is about 18.2M params.
- The cheapest check is one `create_lora_training_client(base_model="Qwen/Qwen3.5-4B", rank=1)` call. A server rejection would show up as a `BadRequestError` there, before any token cost.

### Gaps
- No primary source states the server's allowed rank range or any power-of-two requirement. I found no GitHub issue on rank-1 rejection either.

## 3. Checkpoint download and running LoRA locally

### Takeaway
There are three save methods: `save_state` (resume training), `save_weights_for_sampler` (sampling on Tinker) and `save_weights_external` (download). 0.29.1 has `save_weights_external`, `RestClient.get_external_weights_urls`, and the older `get_checkpoint_archive_url` (tar archive). For Qwen3.5-4B the PEFT conversion has an open bug, issue #1014: with tied embeddings, the unembedding LoRA is silently dropped. The fix, PR #1015, is open and unreviewed. The simplest workaround is `train_unembed=False`.

### Cited Findings
- Method table — [Manage Checkpoints](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/index.md):
  - `save_state` gives `tinker://<run_id>/weights/<name>` (full state including Adam moments; reload with `create_training_client_from_state[_with_optimizer]`)
  - `save_weights_for_sampler` gives `tinker://<run_id>/sampler_weights/<name>` (weights only; SamplingClient, OpenAI/Anthropic-compatible APIs, Playground)
  - `save_weights_external` gives `tinker://<run_id>/external_weights/<name>` ("Download for use outside Tinker")
- Download flow per docs: `save_weights_external("step-100")`, then `rest_client.get_external_weights_urls(path)`, which returns `urls` (file path to signed URL) and `expires`. It "accepts `external_weights` checkpoints, not `weights` or `sampler_weights` paths". CLI: `tinker checkpoint download tinker://<run_id>/external_weights/step-100 --output ./models/` — [Manage Checkpoints](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/index.md).
- The current CLI docs say: "Download and Hugging Face upload commands accept training or external weights checkpoints, not sampler checkpoints"; downloading a `weights` checkpoint "is exported through a temporary external weights checkpoint, which starts a training session on its base model" — [tinker checkpoint](https://tinker-docs.thinkingmachines.ai/tinker/cli/checkpoint/index.md).
- 0.29.1 has `TrainingClient.save_weights_external(name, ttl_seconds=None)` (+`_async`), `RestClient.get_external_weights_urls(tinker_path)` (+`_async`, rejects non-`external_weights/` paths with ValueError), and `RestClient.get_checkpoint_archive_url(training_run_id, checkpoint_id)` returning `CheckpointArchiveUrlResponse{url, expires}` (sync-only decorator; `_async` variant exists) — installed SDK `training_client.py` lines 927-979, `rest_client.py` lines 384-458. The newer docs add `age_encryption_recipients` to `save_weights_external` and `external_weights_upload_start/finish` — [TrainingClient ref](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/index.md), [RestClient ref](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/restclient/index.md). `get_checkpoint_archive_url` no longer appears on the current RestClient reference page (grep of the fetched page).
- Sources conflict on whether sampler checkpoints can be downloaded. The cookbook's PEFT page still shows `weights.download(tinker_path="tinker://<run_id>/sampler_weights/final", ...)` — [Build a PEFT LoRA Adapter](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/lora-adapter/index.md). The installed cookbook 0.5.5 `weights.download` uses the archive-URL path and its docstring uses a `sampler_weights` example (`tinker_cookbook/weights/_download.py`). This conflicts with the CLI docs' "not sampler checkpoints" above.
- Archive contents as checked by the 0.29.1 CLI (`push_hf` path): `adapter_config.json`, `adapter_model.safetensors` (or `.bin`), and a `checkpoint_complete` marker — installed `tinker/cli/commands/checkpoint.py` lines 383-410. The keys are Tinker-internal (`base_model.model.` prefix, `unembed_tokens`, split `in_proj_q/k/v` for Qwen3.5), and `build_lora_adapter` remaps them — installed `tinker_cookbook/weights/README.md`.
- Cookbook utilities: `weights.download`, `weights.build_lora_adapter(base_model, adapter_path, output_path)` (PEFT output: `adapter_config.json` + `adapter_model.safetensors`; loads with `PeftModel.from_pretrained`, vLLM `--enable-lora --lora-modules`, SGLang `--lora-paths`), `weights.build_hf_model` (merge), plus publish-to-Hub. Qwen3.5-4B is listed as Merge ✅ / Adapter ✅, "vLLM 0.18 ✅ Verified (split in_proj_q/k/v works)" — [PEFT page](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/lora-adapter/index.md); installed `tinker_cookbook/weights/README.md`.
- Issue #1014, "build_lora_adapter: unembedding LoRA is silently dropped by PEFT on tied-embedding models (Qwen3.5-4B, Qwen3-4B-Instruct-2507)": **open**, created 2026-11-15, no labels or assignee, and one off-topic third-party comment, with no maintainer reply. Cause: for `tie_word_embeddings=True`, `unembed_tokens` is remapped to `embed_tokens`. PEFT then wraps it as `lora.Embedding` (`lora_embedding_A/B`), so the keys fail to load with only a "missing adapter keys" warning, and the unembed LoRA has no effect. Proposed fix: always map to `lm_head`. The issue also notes that `build_hf_model` on tied models adds the unembed delta to the shared `embed_tokens`, which changes the input embeddings too; whether that matches Tinker's sampler is unknown — [issue #1014](https://github.com/thinking-machines-lab/tinker-cookbook/issues/1014).
- PR #1015, "Target lm_head for the unembedding LoRA in build_lora_adapter on tied-embedding models": **OPEN**, created 2026-11-15, 0 reviews, 0 comments — [PR #1015](https://github.com/thinking-machines-lab/tinker-cookbook/pull/1015).
- Installed `_merge_qwen3_5.py` shows the tied remap (`model.language_model.unembed_tokens` to `model.language_model.embed_tokens`) and the untied `lm_head` path — installed `tinker_cookbook/weights/_merge_qwen3_5.py` lines 72-98.
- A past export bug (server stream reset on `get_checkpoint_archive_url` / `tinker checkpoint download`) was fixed in SDK v0.22.2 — [tinker-feedback #118](https://github.com/thinking-machines-lab/tinker-feedback/issues/118).

### Inferences
- Workarounds for #1014, best first:
  1. Train with `train_unembed=False`. No unembed LoRA exists, so nothing gets dropped, and Tinker sampling and local PEFT should agree. Cost: you lose 250,880 params per unit of rank.
  2. Apply the PR #1015 change locally (map `unembed_tokens` to `lm_head` in `build_lora_adapter`).
  3. Avoid `build_hf_model` for tied models until someone confirms how Tinker applies the unembed LoRA.
- Before trusting a local copy, check parity: compare local PEFT logprobs on a few sampled sequences against Tinker's `seq.logprobs`.
- On 0.29.1, use `save_weights_external` + `get_external_weights_urls` (the documented path) rather than relying on sampler-weights archives.

### Gaps
- I did not see the exact file layout of an `external_weights` export for a LoRA run (whether it is PEFT-ready or raw Tinker keys). The docs say only "external (e.g. HuggingFace) format".
- Whether `get_checkpoint_archive_url` still works server-side for `sampler_weights` is unconfirmed, since the docs conflict.

## 4. Logprobs from sample (top-k, prompt logprobs) and judge expected scores

### Takeaway
On 0.29.1, `sample()` returns per-token logprobs, and it can also return `include_prompt_logprobs`, `topk_prompt_logprobs=k` and `topk_sample_logprobs=k`. `target_prompt_logprobs`, `prompt_alt_tokens_k` and `prompt_logprobs_last_n` exist only in newer SDKs. So you can read a judge's "1".."5" distribution and compute an expected score.

### Cited Findings
- 0.29.1 signature: `SamplingClient.sample(prompt, num_samples, sampling_params, include_prompt_logprobs=False, topk_prompt_logprobs=0, topk_sample_logprobs=0)` (+`sample_async`) — installed `tinker/lib/public_interfaces/sampling_client.py` lines 315-395.
- `SampledSequence` has `tokens`, `logprobs` (per generated token) and `topk_logprobs` (per generated position, a list of up to k `(token_id, logprob)` tuples), plus numpy forms `tokens_np`, `logprobs_np`, `topk_logprobs_np` (`TopkLogprobs` with `(num_tokens, k)` matrices; sentinel `token_id=0, logprob=-99999.0`). `SampleResponse` has `prompt_logprobs` (None at position 0), `topk_prompt_logprobs` and `prompt_cache_hit_tokens` — installed `tinker/types/sampled_sequence.py`, `sample_response.py`, `topk_logprobs.py`.
- `SamplingParams`: `max_tokens`, `seed`, `stop` (str, list of str, or list of token ids; `[]` disables EOS), `temperature=1`, `top_k=-1`, `top_p=1` — installed `tinker/types/_pydantic_types/sampling_params.py`.
- The newer SDK (docs) adds `target_prompt_logprobs` (int64 TensorData `[len(prompt)-1, K]` of token ids to score at each position; `-1` means skip; cost bounded as `len(prompt) * distinct ids`), `prompt_alt_tokens_k`, and `prompt_logprobs_last_n` (score only the last N prompt tokens so the prefix can come from cache) — [SamplingClient ref](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md).
- The server caps top-k width and `len(prompt) * k`, but the docs give no number ("bounds ... the way it bounds a top-k width") — [SamplingClient ref](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md).

### Inferences
- Expected judge score on 0.29.1, option A: call the judge with `topk_sample_logprobs=20`, `temperature=0`. At the generated position whose token is a digit, take the top-k entries whose ids are the tokens for "1".."5", renormalize, and compute E[score]. This adds no extra calls. One caveat: Inkling uses a v0 renderer with channel and header tokens, so the digit may not be the first generated token. Find its index by parsing (lipogram's `parser.parse_tokens`). A digit outside the top-k gets no mass.
- Option B (exact, works on any SDK): render prompt + assistant prefix up to the score, append each candidate digit token, and read the last-token logprob with `compute_logprobs`. That is 5 calls per grade, sharing a cacheable prefix. On the newer SDK, one `target_prompt_logprobs` call with the 5 ids on the final row does the same thing.
- Look up the token ids for "1".."5" in the judge's tokenizer (`O200kBaseChatTokenizer` in lipogram.py) and confirm each digit is a single token.

### Gaps
- No documented maximum k. Whether Inkling-Small sampling honors `topk_sample_logprobs` was not verified without credentials.

## 5. compute_logprobs semantics and cost

### Takeaway
`compute_logprobs(prompt)` is a thin wrapper around `sample()` with `max_tokens=1` and `include_prompt_logprobs=True`. It returns one logprob per prompt token (None at position 0), and it should bill as sampling-side prefill of the full prompt plus 1 sampled token, not at the training price.

### Cited Findings
- 0.29.1 implementation: `_sample_async_impl(prompt, num_samples=1, sampling_params=types.SamplingParams(max_tokens=1), include_prompt_logprobs=True)`, returning `sample_res.prompt_logprobs` (`list[float | None]`) — installed `tinker/lib/public_interfaces/sampling_client.py` lines 397-428.
- The newer SDK adds `compute_logprobs(prompt, prompt_logprobs_last_n=None)`. Scoring only the last N tokens lets "the server ... serve the prefix before them from its prefix cache", and the docs give a log P(completion | context) example — [SamplingClient ref](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md).
- Prefill is "Processing input/prompt tokens when sampling"; forward-only passes on a training client are billed at train price — [Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md).

### Inferences
- Cost for Qwen3.5-4B is about len(prompt) x $0.33/M (or $0.066/M on cache hits) plus 1 token x $1.005/M. A 220-token sequence costs about $0.00007. Scoring with a training client's `forward` would cost $0.737/M instead.
- The SDK is 10 minor versions behind PyPI (0.29.1 vs 0.39.1). In the past the server rejected old SDKs with "Your Tinker SDK version is no longer supported" — [tinker-feedback #120](https://github.com/thinking-machines-lab/tinker-feedback/issues/120). Upgrading would also unlock `target_prompt_logprobs` and `prompt_logprobs_last_n`.

### Gaps
- Docs do not state explicitly how `compute_logprobs` is billed. The figure above comes from the SDK implementation.
