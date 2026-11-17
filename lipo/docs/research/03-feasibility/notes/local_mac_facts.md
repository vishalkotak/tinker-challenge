# Local Mac feasibility: analyzing a Qwen3.5-4B LoRA adapter on an M4 Pro (24 GB)

Researched 2026-11-16. Local measurements were made from files downloaded to `<scratch>` (tokenizer.json, config.json, a public word list). No model weights were downloaded and nothing heavy was run.

## Architecture, transformers version, and runtimes (MPS, MLX, llama.cpp)

### Takeaway
Qwen/Qwen3.5-4B is a vision-language wrapper (`Qwen3_5ForConditionalGeneration`) around a 32-layer text model. 24 of the layers are Gated DeltaNet linear attention and 8 are full attention. Embeddings are tied, the vocab is 248,320, and there is 1 MTP layer. transformers >= 5.2.0 supports it, and the local tinker-challenge env already has 5.5.4. On MPS it runs through a slow pure-PyTorch fallback for the DeltaNet layers. mlx-lm has a native `qwen3_5` model with a Metal kernel for inference (not for training), which makes it the faster local option for forward-only scoring. llama.cpp also supports it (`qwen35.cpp`).

### Cited Findings
- config.json: `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`, `text_config.model_type: qwen3_5_text`, `transformers_version: 4.57.0.dev0` (this is the version that wrote the file, not a usable minimum) — [config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- Text model: `num_hidden_layers 32`, `hidden_size 2560`, `intermediate_size 9216`, SiLU, `vocab_size 248320`, `rms_norm_eps 1e-6`, `max_position_embeddings 262144` — [config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- Hybrid layout: `full_attention_interval 4`. `layer_types` is 3x `linear_attention` then 1x `full_attention`, repeated, so the full-attention layers are 3, 7, 11, 15, 19, 23, 27, 31 (0-indexed) and the other 24 are linear. Full attention: 16 query heads, 4 KV heads, `head_dim 256`, `attn_output_gate true`, partial RoPE (`partial_rotary_factor 0.25`, `rope_theta 1e7`, interleaved mRoPE sections [11, 11, 10]). Linear attention (Gated DeltaNet): `linear_num_key_heads 16`, `linear_num_value_heads 32`, key/value head dim 128, `linear_conv_kernel_dim 4`, `mamba_ssm_dtype float32` — [config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- `tie_word_embeddings: true` at both the top level and the text level. `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` — [config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- Vision tower: depth 24, hidden 1024, 16 heads, intermediate 4096, patch 16, spatial merge 2, `out_hidden_size 2560`. Special token ids: image 248056, video 248057, vision_start 248053, vision_end 248054, eos 248044 (`<|im_end|>`), pad `<|endoftext|>` — [config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json); [HF API](https://huggingface.co/api/models/Qwen/Qwen3.5-4B)
- Size: 4,659,865,088 params (BF16, plus 3,840 F32), covering vision and MTP. The weights ship as two safetensors shards of 5.33 GB and 3.99 GB (about 9.32 GB total). Repo sha `851bf6e`, last modified 2026-03-02, base model `Qwen/Qwen3.5-4B-Base`, Apache-2.0 — [HF API](https://huggingface.co/api/models/Qwen/Qwen3.5-4B?blobs=true)
- transformers added Qwen3.5 in v5.2.0 (released 2026-02-16, "Adding Support for Qwen3.5 (#43830)") — [transformers v5.2.0 release](https://github.com/huggingface/transformers/releases/tag/v5.2.0)
- Local env `uv.lock` pins transformers 5.5.4, torch 2.13.0, tokenizers 0.22.2, tinker 0.29.1, tinker-cookbook 0.5.5. No peft or mlx is pinned there (checked locally).
- transformers `modeling_qwen3_5.py` wraps `causal_conv1d_fn`/`causal_conv1d_update` (hub kernel `causal_conv1d`) and `chunk_gated_delta_rule`/`fused_recurrent_gated_delta_rule` (hub kernel `fla`) with `use_kernel_func_from_hub_with_fallback`. When the kernel is unavailable it uses `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` in pure PyTorch — [modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- causal-conv1d and flash-linear-attention are CUDA-only and "fail silently on MPS, forcing a slow fallback". The third-party `mps-linear-attn` package monkey-patches both with Metal kernels (fp16 Q/K/V, fp32 state). It is tiny and unvetted: 3 stars, last push 2026-03-04, benchmarked only on 0.8B/M1/PyTorch 2.7 — [mpsops/mps-linear-attention](https://github.com/mpsops/mps-linear-attention)
- Known transformers bug: `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-…", dtype=torch.float32)` silently loads bf16 for composite checkpoints, while the concrete `Qwen3_5ForCausalLM` honors dtype. Reported on macOS arm64 with v5.9/5.10.dev; the issue is now closed — [transformers #46459](https://github.com/huggingface/transformers/issues/46459)
- mlx-lm has `mlx_lm/models/qwen3_5.py` (plus `qwen3_5_moe.py`, `qwen3_next.py`). Its `sanitize` drops `mtp.` weights, drops `lm_head.weight` when tied (using `embed_tokens.as_linear`), skips `model.visual`/`vision_tower` keys, and remaps `model.language_model` to `language_model.model`. So the text model loads from the VLM checkpoint — [mlx-lm qwen3_5.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py). Latest release is v0.32.2 (2026-10-16) — [mlx-lm releases](https://github.com/ml-explore/mlx-lm/releases)
- In mlx-lm, `GatedDeltaNet` calls `gated_delta_update(..., use_kernel=not self.training)`, so inference uses the Metal kernel and training falls back to a per-token Python loop. Measured on an M4 Pro 24 GB, 4-bit Qwen3.5-4B-Base, LoRA training ran at 0.073 it/s with 16 layers (14.8 GB peak) and 0.195 it/s with 5 layers (12.6 GB peak) — [mlx-lm #1956](https://github.com/ml-explore/mlx-lm/issues/1956). A backward kernel is requested in [mlx-lm #2050](https://github.com/ml-explore/mlx-lm/issues/2050)
- Reported mlx-lm problem: `load()` on some community VLM-format Qwen3.5/3.6 MoE checkpoints returns garbage with no error, while non-VLM conversions work. Still open — [mlx-lm #1197](https://github.com/ml-explore/mlx-lm/issues/1197)
- mlx-lm cannot load PEFT adapters directly. The PR that would have added PEFT/Unsloth adapter loading to `load_adapters()` was closed without merging — [mlx-lm #1120](https://github.com/ml-explore/mlx-lm/pull/1120)
- llama.cpp has `src/models/qwen35.cpp` and `qwen35moe.cpp`. One open bug reports garbage output on the CPU backend for the qwen35 hybrid family (27B-class models) while Vulkan is fine — [llama.cpp src/models](https://github.com/ggml-org/llama.cpp/tree/master/src/models); [llama.cpp #30906](https://github.com/ggml-org/llama.cpp/issues/30906)

### Inferences
- For hooks (alpha scaling, per-layer ablation, reading hidden states), PyTorch + transformers on MPS is the easiest path, because PEFT and forward hooks just work. The cost is the slow pure-torch DeltaNet path on 24 of 32 layers. Use the concrete `Qwen3_5ForCausalLM` (or the ConditionalGeneration class and ignore vision) and pass dtype explicitly, because of #46459.
- For throughput on logprob scoring, mlx-lm inference uses the Metal gated-delta kernel. The LoRA would have to be merged into the weights or converted to mlx's adapter format by hand, since PEFT loading was never merged. llama.cpp is fine for generation but awkward for per-layer interventions and LoRA alpha sweeps.
- Rough speed (my estimate, not measured): a forward pass costs about 2 x 4.2B FLOPs per token for the text model, embedding/lm_head included. 3,000 samples x 200 tokens = 600k tokens, about 5e15 FLOPs. At an effective 3 to 5 TFLOPS on the M4 Pro GPU that is 15 to 30 minutes per full pass at best. The torch DeltaNet fallback on MPS could easily make it 2 to 5 times slower. Benchmark 50 samples first.

### Gaps
- I found no published tokens/s numbers for Qwen3.5-4B forward passes on MPS in transformers, or for mlx-lm prefill on an M4 Pro. Measure locally.
- I did not confirm whether `kernels`-hub provides an MPS build of `fla` / `causal_conv1d`. It probably does not.
- I did not verify that the official Qwen/Qwen3.5-4B VLM checkpoint loads cleanly through mlx-lm (#1197 is about MoE community conversions). Smoke-test the output.

## Tokenizer statistics (Qwen/Qwen3.5-4B, "e" content)

### Takeaway
The byte-level BPE has 248,044 base tokens plus 26 added tokens (ids up to 248,069; the model vocab_size is 248,320 padded). 27.9% of all base tokens contain "e"/"E", but the share rises to 41.7% among the 10k lowest-id (most frequently merged) tokens. Many very common English words have no e: 61 of the top 100 and 465 of the top 1,000. Nearly all of them are single tokens in Qwen's vocab.

### Cited Findings
- Files: `tokenizer.json` 12.8 MB, `tokenizer_config.json`, `vocab.json` 6.7 MB, `merges.txt` 3.35 MB — [HF API](https://huggingface.co/api/models/Qwen/Qwen3.5-4B?blobs=true)
- The following figures were computed locally (script `<scratch>` on [tokenizer.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/tokenizer.json), decoding the byte-level GPT-2 unicode map). Model type BPE, 248,044 vocab entries, 247,587 merges, 26 added tokens.
  - Tokens whose decoded text contains e or E: all base vocab 69,115 / 248,044 = **27.9%**. Lowest 1k ids 22.3% (mostly single bytes and short pieces). Lowest 5k ids 38.0%. **Lowest 10k ids 41.7%**. Lowest 50k ids 47.1%.
  - ASCII-alphabetic tokens (after stripping whitespace): 53.7% contain e/E overall, 47.1% among ids < 10k. The overall share is low because much of the vocab is CJK and other scripts.
  - Space-prefixed lowercase word tokens (" word", length >= 2 letters): no-e share is 60.0% for ids < 2k (253/422), 49.0% for ids < 10k (1,435/2,930), and 46.4% for ids < 20k.
  - Single-letter ids: `e` = 68, ` e` = 378, `E` = 36, ` E` = 458.
  - Low-id word tokens with e include ` the`, ` be`, ` are`, ` he`, ` we`, ` have`, ` they`, ` their`, ` there`, ` been`, ` were`, ` when`, ` one`, ` more`, ` other`, plus code words ` return`, ` self`, ` def`, ` else`, ` true`, ` false`, ` value`, ` type`.
- Common English words (ranked list [google-10000-english-no-swears](https://github.com/first20hours/google-10000-english), 9,894 words; script `<scratch>`):

  | Top N | Words without e | Share | No-e words that are a single " word" Qwen token |
  |---|---|---|---|
  | 100 | 61 | 61.0% | 61 |
  | 1,000 | 465 | 46.5% | 461 |
  | 5,000 | 2,169 | 43.4% | 1,992 |
  | 9,894 | 4,271 | 43.2% | 3,604 |

  - Top-100 no-e words: of, and, to, a, in, for, is, on, that, by, this, with, i, you, it, not, or, from, at, as, your, all, an, was, will, can, us, about, if, my, has, but, our, do, no, information, up, may, what, which, out, any, only, so, his, contact, who, also, now, first, am, would, how, its, than, find, and others.

### Inferences
- Banning all e-tokens at sampling time would remove about 28% of the vocab, but about 42% of the frequently used token mass by id rank. "the" is the single biggest casualty. Text without e is still easy to write, since e-free function words (of, and, to, a, in, is, that, this, with, it) all exist as single tokens.
- Low id is only a proxy for frequency (it reflects BPE merge order on Qwen's training mix, which includes code and multilingual text). For real frequencies, tokenize a reference corpus.

### Gaps
- I did not compute frequency-weighted e-share on an actual English corpus. Only id-rank and word-list proxies are given above.

## PEFT loading of Tinker LoRA for tied-embedding Qwen3.5 (tinker-cookbook #1014)

### Takeaway
On tied models, the cookbook's `build_lora_adapter` writes the unembedding LoRA as Linear-style `lora_A`/`lora_B` keys on `embed_tokens`. PEFT wraps that module as `lora.Embedding`, which expects `lora_embedding_A/B`, so the keys never load and the unembed LoRA is silently dropped. The fix, and the manual workaround, is to target `lm_head` instead and apply the delta to the logits only.

### Cited Findings
- The converted adapter contains `...embed_tokens.lora_A.weight` (r x hidden), `...embed_tokens.lora_B.weight` (vocab x r), and `target_modules: ["embed_tokens"]`. `PeftModel.from_pretrained` warns "Found missing adapter keys while loading the checkpoint" and `lora_embedding_A` stays zero. Even if the keys loaded, an embed_tokens LoRA would act on the input lookup, not on the logits — [tinker-cookbook #1014](https://github.com/thinking-machines-lab/tinker-cookbook/issues/1014)
- Affected models: Qwen3.5-4B and Qwen3-4B-Instruct-2507 (tied). Untied Qwen3-8B maps to `lm_head` and works. Repro (macOS arm64, torch 2.15 CPU, transformers 5.22.2, peft 0.21.2): the tied model had a max logit diff of 0.814 from the reference, and the untied one 1.19e-7 — [tinker-cookbook #1014](https://github.com/thinking-machines-lab/tinker-cookbook/issues/1014)
- Cause: the adapter path reuses the merge path's name remap (since #528/#593), which sends `unembed_tokens` to `embed_tokens` when the checkpoint has no `lm_head.weight`. PR #1015 adds `unembed_to_lm_head=True` for `build_lora_adapter`, so the keys become `base_model.model.lm_head.lora_{A,B}.weight`. With that change, Qwen3.5 matches the reference to 1.788e-7. PR status is open. The issue has no maintainer response — [tinker-cookbook PR #1015](https://github.com/thinking-machines-lab/tinker-cookbook/pull/1015); [#1014](https://github.com/thinking-machines-lab/tinker-cookbook/issues/1014)
- Separate open question: for tied models, `build_hf_model` (the merge path) adds the unembed delta to the shared `embed_tokens` weight, which also changes the input embeddings. Whether that matches Tinker's sampler is unverified — [tinker-cookbook #1014](https://github.com/thinking-machines-lab/tinker-cookbook/issues/1014)

### Inferences
- Manual workaround without waiting for #1015:
  1. Load the converted adapter safetensors and pop the two `embed_tokens.lora_A/B` tensors.
  2. Load the rest (attention/MLP LoRA) with PEFT, or apply them by hand.
  3. For the unembedding, compute `logits = h @ W_E.T + scale * (h @ A.T) @ B.T` with `scale = lora_alpha / r` from adapter_config.json. Do this with a forward hook on `lm_head` or by post-processing the final hidden states. Do not touch `embed_tokens`.

  Alternatively, rename the keys to `base_model.model.lm_head.lora_A/B.weight`, set `target_modules` to include `lm_head`, and load with PEFT, which is what PR #1015 does.
- This separation also suits the analysis. The unembedding projection of the adapter is just `B` (vocab x r) times `A` (r x hidden). Each of the r rank directions gives a per-token score vector `B[:, k]`, so you can directly check which tokens (for example, e-containing ones) the adapter pushes down. Alpha scaling means multiplying `scale` per module. Per-layer ablation means zeroing a layer's LoRA pair.
- Check the merge path too. If someone used `build_hf_model` merged weights for a tied model, the input embeddings were perturbed as well, so results could differ from Tinker's sampler.

### Gaps
- I could not confirm how the Tinker service applies the unembed LoRA on tied models (logits only, or the shared weight).
- I did not confirm Tinker's default `lora_alpha` / rank convention. Read it from the exported adapter_config.json.

## Memory and time budget on 24 GB (one process at a time)

### Takeaway
bf16 weights are about 9.3 GB on disk with vision and MTP included, and about 8.5 GB for the text model alone. That leaves roughly 8 to 10 GB of usable headroom on a 24 GB Mac, after macOS and the default GPU wired-memory limit. The binding constraint for scoring is the 248k-vocab logits tensor, not activations. Batch size 8 to 16 at 200 tokens is feasible if logprobs are gathered without keeping full fp32 log-softmax copies.

### Cited Findings
- 4.66B BF16 params, about 9.32 GB of safetensors — [HF API](https://huggingface.co/api/models/Qwen/Qwen3.5-4B?blobs=true)
- A real M4 Pro 24 GB data point: 4-bit Qwen3.5-4B LoRA training at batch 1 / 1,024 tokens with grad checkpointing peaked at 12.6 to 14.8 GB in mlx-lm — [mlx-lm #1956](https://github.com/ml-explore/mlx-lm/issues/1956)
- The DeltaNet state dtype is float32 (`mamba_ssm_dtype`) — [config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)

### Inferences
These are my own calculations from the config.
- Embedding / unembedding matrix: 248,320 x 2,560 = 635.7M params, about 1.27 GB bf16. It is shared because embeddings are tied. Vision tower: about 0.3B params (24 x (4 x 1024² + 2 x 1024 x 4096)), about 0.6 GB, which you save by loading text-only.
- Logits per 200-token sequence: 200 x 248,320 x 4 bytes = about 199 MB in fp32 (99 MB in bf16). At batch 16 that is about 3.2 GB in fp32, and log_softmax doubles it. Compute the log-softmax in chunks, or gather the target logit and subtract `logsumexp`, to keep this under about 1.5 GB.
- Activations at 200 tokens are tiny: hidden 2,560 x 200 x 2 bytes = 1 MB per layer per sequence. DeltaNet recurrent state is 32 value heads x 128 x 128 x 4 bytes = 2 MB per layer per sequence, so 24 layers is about 50 MB per sequence. The torch chunked fallback creates extra intermediates but should stay well under 1 GB at batch 16.
- Suggested plan:
  1. Run with `torch.inference_mode()` in bf16 on MPS, loading text-only.
  2. Use batch 8 to start and raise it to 16 if memory pressure stays green.
  3. Run one process at a time and close other GPU apps, consistent with the earlier finding that parallel heavy processes choke this Mac.
  4. Score a few thousand samples once per configuration (budget about 15 to 60 minutes per pass; measure on 50 samples first).
  5. For per-layer ablation across 32 layers, or alpha sweeps, use a subset of about 300 to 500 samples, or cache hidden states before the final norm so that unembedding-only variants need just a cheap lm_head re-run.
- 4-bit or 8-bit MLX quantization would cut the weights to about 2.5 or 4.5 GB, but it perturbs logprobs. Use bf16 for scoring comparisons.

### Gaps
- I did not find a measured MPS or MLX prefill throughput for this exact model. The timing figures above are estimates.
- I did not verify the exact default MPS memory limit for this machine (`recommendedMaxWorkingSetSize` is typically about 65 to 75% of RAM).
