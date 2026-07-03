# Minimizing TTFT for Local LLM Serving — Results Report

**Author:** Alp · **Date:** ____ · **Repo:** ____

## 1. Setup
- GPU: NVIDIA RTX 4000 Ada, 20 GB, SM 8.9 (FP8 + FlashAttention-2 capable) — RunPod
- Model: Qwen3-4B-Instruct-2507 (BF16) and its official FP8 checkpoint
- Software: vLLM __._._, TGI __._, torch __._, CUDA 12.x, driver ____
- Why not the 27B on this card: weights alone ≥ ~54 GB in BF16; see §7.

## 2. Methodology (1 short paragraph)
Client-observed TTFT = request-send → first streamed token. Exact-length
prompts built with the model tokenizer; cold (unique prefix, guaranteed cache
miss) vs warm (shared prefix) cache modes; temperature=0, max_tokens=16;
5 warmups discarded; N=24/config; p50/p90/p99 reported. Full harness in repo.

## 3. Headline result
> TTFT at 2048-token prompts, concurrency=1:
> naive HF **___ ms** → TGI vanilla **___ ms** → vLLM vanilla **___ ms** →
> vLLM optimized (FP8, tuned) cold **___ ms** / warm-cache **___ ms**
> = **__x** over naive HF, **__x** over vanilla vLLM (cold), **__x** warm.

![ttft vs prompt](plots/ttft_vs_prompt.png)
![speedup](plots/speedup_bar.png)

## 4. Full results table
(paste results/summary.md)

## 5. Ablation — where the speed comes from
| mechanism toggled off | Δ TTFT cold | Δ TTFT warm | Δ p99 @ c=16 | why |
|---|---|---|---|---|
| prefix caching | ~0 | ++++ | | warm path re-prefills everything |
| CUDA graphs / compile (eager) | + (short prompts) | | | per-step launch overhead |
| chunked prefill budget 512 | ++ (solo long prompt) | | −− (improves!) | HoL blocking tradeoff |
| FP8 → BF16 | + | | | prefill is compute-bound; Ada FP8 tensor cores |

## 6. Novel findings
1. **Analytical TTFT model.** Fitted `TTFT = a + b·uncached_tokens`:
   a = ___ ms fixed overhead, effective prefill throughput = ___ tok/s,
   R² = ____; holdout prediction error = ___ %. Practical rule derived:
   *put static content (system prompt, docs) first, volatile content last* —
   quantified warm-path win: ___ x.
2. **INT4-AWQ vs FP8 on prefill.** AWQ improved decode by ___ % but changed
   TTFT by ___ % (flat/worse) — weight-only quant targets bandwidth, prefill
   is compute-bound; FP8 improves TTFT by ___ % on Ada. 
3. (optional) TTFT decomposition via vLLM /metrics vs client clock: ___ ms of
   client-observed TTFT is HTTP/SSE/tokenize, i.e. the floor for tiny prompts.

## 7. What I'd do next
- Reproduce on A100/H100 with Qwen3-30B-A3B-Instruct-2507-FP8 → target <1 s
  TTFT at 4k prompts on a ~30B-class model (original task spec). [or: DONE, numbers here]
- TensorRT-LLM engine build for the last ~20-30% of prefill compute.
- Disaggregated prefill/decode (vLLM P/D, Mooncake-style) for p99 under load.
