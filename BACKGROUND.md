# BACKGROUND.md — LLM Inference & TTFT: the field guide

Read this twice. Everything in the repo follows from the ideas here, and the
follow-up conversation with the LLM team will draw from exactly this material.

---

## 1. What happens when you send a prompt

A decoder-only transformer serves a request in two phases.

**Prefill.** The engine tokenizes the prompt and runs ONE forward pass over all
prompt tokens simultaneously. This pass does two jobs: it computes the key/value
(K,V) tensors for every prompt token at every layer and stores them in the
**KV cache**, and it produces the logits for the *first* generated token.
Because all tokens are processed at once, the work is large dense matmuls —
this phase is **compute-bound** (limited by the GPU's FLOP/s).

**Decode.** After the first token, generation proceeds one token at a time.
Each step runs the model over a *single* token, attending to the whole KV
cache. The matmuls are skinny (batch of tokens = 1 per sequence), so the GPU
spends its time streaming weights and KV cache from HBM rather than computing —
this phase is **memory-bandwidth-bound**.

TTFT is essentially the cost of prefill plus fixed overheads; TPOT (time per
output token) is the cost of a decode step. They are optimized by *different*
techniques, and confusing the two is the most common mistake in this area.

## 2. The roofline argument (do this math on a whiteboard)

Arithmetic intensity = FLOPs performed per byte moved from memory. A GPU has a
peak FLOP rate and a peak memory bandwidth; whichever you exhaust first is your
bottleneck.

For a weight matrix multiply with batch-of-tokens B: you move the weights once
(bytes ≈ 2·N_params for BF16) and do ≈ 2·B·N_params FLOPs, so intensity ≈ B
FLOPs/byte (times precision factor). RTX 4000 Ada: ~360 GB/s HBM vs ~100+
BF16 TFLOP/s → the crossover B where compute becomes the limit is in the
hundreds of tokens.

- Prefill of a 2048-token prompt: B = 2048 → far past crossover → compute-bound.
- Decode with one sequence: B = 1 → hopelessly bandwidth-bound; the GPU is
  ~99% idle on compute, just streaming 8 GB of weights per token.

Back-of-envelope prefill time for a dense model: FLOPs ≈ 2 · N_params ·
L_prompt. Qwen3-4B, 4096-token prompt → ≈ 2·4e9·4096 ≈ 33 TFLOP; at an
effective 50 TFLOP/s that's ~650 ms. Qwen3.6-27B at FP8 on an RTX 6000 Ada
(~360 FP8 dense TFLOP/s peak, maybe 40-50% achieved): 2·27e9·4096 ≈ 220 TFLOP
→ ~1.2-1.5 s naive... which is why chunking, kernel efficiency, and the fixed
overheads all matter, and why <1s at 4k tokens is a *real* target, not a gimme.
(These are estimation tools, not predictions — measure.)

Consequences you should be able to recite:
- **FP8 helps TTFT** (halves matmul cost where tensor cores support it: Ada
  SM 8.9, Hopper, Blackwell). It also halves weight bytes, helping decode.
- **Weight-only INT4 (AWQ/GPTQ) helps decode** (¼ weight traffic) but for
  prefill the compute still happens in BF16 after dequantization, plus dequant
  overhead → TTFT flat or worse. Activation-quantized schemes (W8A8-FP8, NVFP4
  W4A16 variants) are the ones that touch prefill.
- **Batching decode is nearly free** until you hit the compute roof — the
  foundation of all modern serving economics.

## 3. KV cache — the resource everything fights over

Per token, per layer, you store K and V: bytes = 2 · n_kv_heads · head_dim ·
precision. With GQA (grouped-query attention) n_kv_heads << n_attn_heads,
cutting this 4-8x — that's why GQA exists. Total KV for a request = that ×
n_layers × seq_len. This is why Veysel said 27B "kv cache dahil 100GB'yi geçer":
at the native 262k context the KV cache alone dwarfs the weights. Capping
`--max-model-len` is therefore the first knob on small cards.

**PagedAttention** (vLLM's founding idea): allocate KV in fixed-size blocks
(like OS virtual-memory pages) instead of one contiguous slab per request.
Kills fragmentation, allows near-100% memory utilization, and enables sharing
blocks between requests — which is what makes prefix caching cheap.

**Prefix caching**: hash each full block of prompt tokens; if a new request's
leading blocks match blocks already resident, reuse them and only prefill the
tail. TTFT for the cached portion → ~0. SGLang generalizes this with
**RadixAttention** (a radix tree over token sequences, better partial-match
reuse). The engineering consequence for prompt design: *static content first,
volatile content last* — a timestamp at the top of your system prompt destroys
caching for everything after it.

## 4. Scheduling: continuous batching and chunked prefill

**Continuous (in-flight) batching**: the scheduler admits/retires requests at
every step rather than waiting for a whole batch to finish — the second
foundational serving idea (Orca paper), now universal.

Mixing prefill and decode in one engine creates **head-of-line blocking**: a
monolithic 8k-token prefill occupies the GPU for hundreds of ms during which
every in-flight decode stalls (their inter-token latency spikes), and queued
requests wait (their TTFT spikes). **Chunked prefill** splits prefills into
chunks (budget = `--max-num-batched-tokens`) and co-schedules them with decode
steps. The tradeoff is clean: big budget → best solo TTFT; small budget → best
p99 TTFT and smooth TPOT under load. There is no free lunch; you pick per
workload. (The endgame is **P/D disaggregation** — separate prefill and decode
GPU pools, e.g. Mooncake, vLLM P/D, NVIDIA Dynamo — mention it as "what I'd do
next".)

## 5. The launch-overhead tail: CUDA graphs & compilation

A decode step is thousands of small kernel launches; at ~µs of CPU launch cost
each, the CPU becomes the bottleneck for small models. **CUDA graphs** record
the whole step once and replay it as a single launch. vLLM captures graphs for
a set of batch sizes at startup (that's the warmup pause). `torch.compile`
(vLLM's `-O3`) additionally fuses kernels. Effect on TTFT: visible mainly at
SHORT prompts, where fixed overheads are a large fraction. Hybrid-architecture
models (see §7) have recurrent state that complicates graph capture — hence
the `--max-cudagraph-capture-size` workaround you may need on Qwen3.6.

## 6. What does NOT help TTFT (know these cold)

- **Speculative decoding / MTP**: a draft mechanism proposes tokens, the target
  model verifies them in parallel → 1.2-2x on *decode*. The first token still
  requires the full prefill. Qwen3.6 ships an embedded MTP head; great for
  chat UX, irrelevant to this task's metric.
- **Sampler tricks, streaming granularity, etc.** — cosmetic for TTFT.
- **Tensor parallelism** *can* help TTFT (splits the prefill matmuls across
  GPUs) but adds all-reduce latency; on a single card, moot.

## 7. Hybrid architectures (why Qwen3.5/3.6 are different)

Classic attention costs O(L²) compute in prefill and O(L) KV memory. Linear
attention variants (Mamba/SSMs, DeltaNet, **Gated DeltaNet**) maintain a
fixed-size recurrent state instead of a growing KV cache: O(L) prefill via a
chunk-parallel scan, O(1) state per token in decode. Pure linear attention
loses retrieval precision, so frontier models interleave: Qwen3.5/3.6 use
**3 Gated-DeltaNet layers : 1 full-attention layer**. Practical consequences:
- Long-context prefill (and thus TTFT) scales much more gently — this is what
  Tier B measures.
- Only the ¼ full-attention layers hold a KV cache → long-context memory drops
  massively.
- Prefix caching is trickier: reusing state for the DeltaNet layers means
  checkpointing recurrent states, not just KV blocks — engine support is newer
  and worth verifying empirically (if warm-mode TTFT on Qwen3.5-4B does NOT
  collapse like it does on Qwen3-4B, you've found and can explain exactly this).
- Serving engines route the recurrent state through their Mamba-cache
  machinery (vLLM's `--mamba-*` flags); TGI never implemented any of it.

## 8. Quantization taxonomy (one paragraph each)

**FP8 (E4M3/E5M2), W8A8**: weights AND activations in 8-bit float, matmuls run
on FP8 tensor cores (Ada/Hopper/Blackwell). ~2x compute, ~2x memory, near-zero
quality loss with per-block scales (Qwen ships official FP8 checkpoints,
block size 128). The default choice for this project.
**INT4 weight-only (AWQ/GPTQ)**: 4-bit weights, activations stay 16-bit;
kernels (Marlin/Machete) dequantize on the fly. ~4x weight memory savings →
big decode wins; prefill compute unchanged. AWQ picks salient channels via
activation statistics; GPTQ does layer-wise error minimization.
**NVFP4 / mixed schemes**: 4-bit floating point with FP8 attention/KV (e.g.
nvidia's Qwen3.6-27B-NVFP4, ~22 GB) — Blackwell-native FP4 compute; on Ada it
still helps memory.
**KV-cache quantization** (`--kv-cache-dtype fp8`): halves KV memory and decode
KV bandwidth; mild accuracy risk; mostly a capacity/decode lever, minor for TTFT.
**GGUF/llama.cpp K-quants**: CPU/consumer ecosystem; not used here but know
the name.

## 9. The engine landscape (July 2026, one line each)

**vLLM** — reference open-source server; PagedAttention, continuous batching,
chunked prefill + prefix caching + CUDA graphs on by default (V1 engine);
broadest model support (incl. hybrid GDN as of 0.19). **SGLang** — same class
of performance, RadixAttention prefix reuse, very fast structured
generation; supports Qwen3.6. **TGI** — Hugging Face's server; historically
important, now effectively in maintenance mode, its README points to
vLLM/SGLang; no hybrid-architecture support — which is precisely why it's
benchmarked on Qwen3-4B here. **TensorRT-LLM** — NVIDIA's compiled engine;
best kernels, highest effort; the "next step" for the last 20-30%.
**llama.cpp/Ollama/MLX** — consumer/local ecosystem, GGUF quants, CPU+GPU.
**KTransformers** — CPU-offload MoE specialist.

## 10. Benchmarking methodology (how not to lie to yourself)

1. Measure **client-observed** TTFT via streaming (first SSE chunk with
   content). Server-internal metrics are useful for decomposition, not as the
   headline.
2. Control tokens exactly — build prompts with the model's own tokenizer.
3. **Control the cache.** Unique nonce at prompt START = guaranteed cold.
   Shared prefix = warm. Report both; say which is which. (The single most
   common silent error in published TTFT numbers.)
4. Warmup and discard (graph capture, allocator, JIT all hit request #1).
5. Percentiles (p50/p90/p99) over ≥20 samples; never a lone mean.
6. Sweep the two axes that matter: prompt length and concurrency.
7. Fix sampling (temperature=0, tiny max_tokens) — you're measuring latency.
8. Report versions, flags, GPU, driver; commit raw CSVs. Reproducibility IS
   the credibility.
9. Beware the network path: benchmarking through a proxy adds tens of ms —
   fine if constant across engines, must be stated.
10. For reasoning-mode models, bypass the chat template (`/v1/completions`)
    or you're timing "<think>" emission behavior, not the engine.

## 11. Glossary sprint

TTFT time-to-first-token · TPOT/ITL time-per-output-token / inter-token latency
· E2E end-to-end latency · goodput throughput meeting an SLO · prefill prompt
forward pass · decode autoregressive generation · KV cache stored keys/values
· GQA grouped-query attention · MQA multi-query (1 KV head) · PagedAttention
block-based KV allocation · prefix caching KV reuse across requests ·
RadixAttention tree-structured prefix reuse · continuous batching per-step
admission · chunked prefill prefill split & interleaved with decode · HoL
head-of-line blocking · CUDA graph recorded kernel launch sequence ·
speculative decoding draft-then-verify · MTP multi-token-prediction (embedded
drafter) · GDN Gated DeltaNet linear attention · SSM state-space model · MoE
mixture-of-experts (A3B = 3B active) · AWQ/GPTQ weight-only INT4 · FP8 E4M3
8-bit float · SM 8.9 Ada compute capability (FP8-capable) · TP tensor
parallelism · P/D disaggregation separate prefill/decode fleets · YaRN RoPE
context extension · roofline compute-vs-bandwidth bound model.

## 12. Questions Veysel's team may ask — and the shape of good answers

**"Why is prefill compute-bound and decode bandwidth-bound?"** Arithmetic
intensity: FLOPs/byte ≈ tokens in the batch; prefill has thousands, decode has
~1 per sequence. Draw the roofline.
**"Your TTFT improved 20x with warm cache — is that cheating?"** No, if
labeled: production chat traffic has enormous shared prefixes; report cold and
warm separately and state the workload each represents.
**"Why didn't INT4 improve your TTFT?"** Weight-only quant reduces bytes moved,
not FLOPs; prefill is FLOP-limited; dequant adds overhead. FP8 reduces FLOPs
(native tensor-core support on Ada), so it did help.
**"How would you get TTFT down further?"** In order: prefix-cache-aware prompt
layout (free), bigger chunked-prefill budget if solo-latency-bound, FP8/FP4
kernels via TensorRT-LLM, tensor parallelism across 2 GPUs for the prefill
matmuls, and architecturally: P/D disaggregation so prefills never queue behind
decodes; for long context, hybrid-attention models change the scaling law itself.
**"What's the catch with chunked prefill?"** Solo TTFT vs p99-under-load
tradeoff; show your ablation numbers.
**"Why didn't you use speculative decoding?"** It cannot help the first token
by construction; it's a decode optimization. (Then show you *know* Qwen3.6 has
embedded MTP and roughly what acceptance rate means.)
**"27B on your 20GB card?"** Doesn't fit: FP8 weights ~27GB, even NVFP4 ~22GB;
and KV cache/activations on top. Right answer is a 48GB Ada card at <$1/hr,
which is what I did — knowing the memory math beats heroic squeezing.

## 13. Primary sources worth skimming before the interview

vLLM paper (PagedAttention, SOSP'23) · Orca paper (continuous batching,
OSDI'22) · Sarathi/Sarathi-Serve (chunked prefill) · SGLang paper
(RadixAttention) · Mooncake (P/D disaggregation, KV-centric serving) · AWQ and
GPTQ papers · FP8 formats (NVIDIA Transformer Engine docs) · Gated DeltaNet
paper + Qwen3-Next/Qwen3.5 blog posts (hybrid architecture) · vLLM docs pages:
"Optimization and Tuning", "Automatic Prefix Caching", "Chunked Prefill".
