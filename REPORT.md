# Minimizing TTFT for Local LLM Serving — Final Results Report

**Author:** Alp  
**Completed:** 2026-07-12  
**Headline model:** `Qwen/Qwen3.6-27B-FP8`  
**Headline GPU:** one NVIDIA RTX 6000 Ada Generation, 49,140 MiB

## Executive result

The assigned target was met:

> **3072-token cold prompt, concurrency 1: 839.7 ms p50 TTFT, 850.6 ms p90, 856.5 ms p99 (N=24).**

This is a client-observed, streamed `/v1/completions` measurement on the full Qwen3.6-27B-FP8 language model. It is not a warm-cache number and not a smaller-model extrapolation.

The boundary matters. At 4096 cold tokens, the model is close but not below one second:

| 4096-token workload, c=1 | p50 | p90 | p99 | N |
|---|---:|---:|---:|---:|
| vanilla, cold | 1193.8 ms | 1197.4 ms | 1198.1 ms | 24 |
| tuned + optimized, cold | **1063.2 ms** | 1080.6 ms | 1085.3 ms | 24 |
| optimized, warm reusable prefix | **117.1 ms** | 118.8 ms | 119.7 ms | 23 |

The defensible conclusion is therefore: **sub-second through 3072 cold tokens; 1.063 s at 4096 cold tokens; 117 ms at 4096 when the long prefix is reusable.**

![Tier C prompt-length result](plots_tier_c/ttft_vs_prompt.png)

## 1. Experimental setup

### Tier C — assigned 27B task

- GPU: NVIDIA RTX 6000 Ada Generation, 49,140 MiB, SM 8.9
- Driver: 570.195.03
- CUDA: 12.8 pod runtime/toolkit; torch wheel `2.10.0+cu128`
- Python: 3.12.3
- vLLM: 0.19.1
- Model: `Qwen/Qwen3.6-27B-FP8`
- Serving: language model only; single GPU; OpenAI-compatible completions endpoint

Qwen3.6 is multimodal and has a native 262k context. `--language-model-only` and an 8192-token context cap are capacity prerequisites on a 48 GB card, not claimed performance tricks. The default CUDA-graph capture range also over-allocates the hybrid recurrent-state cache; 256 was the largest stable capture cap.

### Tier A/B — supporting 4B work

Earlier measurements used an RTX 4000 Ada 20 GB with Qwen3-4B BF16/FP8 and Qwen3.5-4B. Those rows remain in `results/`, but they are not mixed into the Tier C same-hardware speedup table.

## 2. Methodology

TTFT is measured from client request send to the first SSE chunk containing generated text.

- Endpoint: raw `/v1/completions`, avoiding chat-template and `<think>`-block effects.
- Prompt lengths: exact tokenizer round-trip lengths.
- Standard matrix: 128, 512, 1024, 2048, 4096 prompt tokens; concurrency 1, 4, 16.
- Boundary matrix: 3072 cold tokens, concurrency 1.
- Generation: temperature 0, maximum 16 output tokens.
- Cache control:
  - cold prompts put a random nonce first, preventing prefix reuse;
  - warm prompts share the long prefix and put a random nonce at the tail.
- Warmup samples are discarded.
- Standard target is 24 measured requests per cell. The CSV records actual N; three optimized warm cells have N=23 after one disconnected request and still exceeded the harness success threshold.
- A cell below the configured success rate is rejected and no CSV is written.

Raw observations, not rendered Markdown tables, are the source of truth:

- `results_tier_c/qwen36-27b-vanilla.csv`
- `results_tier_c/qwen36-27b-tuned.csv`
- `results_tier_c/qwen36-27b-optimized.csv`
- `results_tier_c/qwen36-27b-aggressive.csv`
- `results_tier_c/qwen36-27b-tuned-optimized.csv`
- `results_tier_c/qwen36-27b-sub1.csv`

## 3. Cold-path scaling

Selected c=1 p50 values:

| prompt tokens | vanilla | tuned | optimized | aggressive |
|---:|---:|---:|---:|---:|
| 128 | 92.2 ms | **84.9 ms** | 93.8 ms | 87.5 ms |
| 512 | **158.5 ms** | 159.1 ms | 166.5 ms | 165.7 ms |
| 1024 | 287.2 ms | **286.5 ms** | 319.4 ms | 288.9 ms |
| 2048 | **590.2 ms** | 591.2 ms | 615.2 ms | 628.2 ms |
| 4096 | **1193.8 ms** | 1193.4 ms | 1251.6 ms | 1244.9 ms |

The full-matrix result is intentionally less exciting than the headline: scheduler flags and FP8-KV do not broadly improve cold c=1 latency. The best 4096 result came from combining the installed RTX 6000 Ada GEMM configurations with the optimized scheduling flags in a focused cell: 1063.2 ms, a 1.12× speedup over vanilla.

The dedicated 3072-token tuned run produced 839.7/850.6/856.5 ms p50/p90/p99. All 24 samples completed.

## 4. Prefix caching

At 4096 tokens and c=1:

- vanilla cold: 1193.8 ms
- optimized warm: 117.1 ms
- speedup: **10.19×**

The benefit persists under load:

| concurrency | vanilla cold p50 | optimized warm p50 | ratio |
|---:|---:|---:|---:|
| 1 | 1193.8 ms | 117.1 ms | 10.19× |
| 4 | 2420.7 ms | 273.1 ms | 8.86× |
| 16 | 10357.2 ms | 1059.1 ms | 9.78× |

The generated `results_tier_c/speedup.md` uses same-cache-mode comparisons and reports 8.88×/9.84× for the latter two cells against vanilla warm. The table above answers a different operational question—reusable optimized prefix versus a cold vanilla request—and is labeled accordingly.

### Failed progressive sweep

`results_tier_c/prefix_sweep_27b-optimized.txt` is preserved because failed experiments are evidence. Its TTFT increased slightly as nominal uncached tokens decreased, producing a negative fitted slope. vLLM logged that hybrid Mamba caching was in experimental `align` mode, and the sweep did not create the expected monotonic hit pattern.

Do **not** quote its intercept, slope, throughput, or holdout error as a model. The harness now rejects non-physical fits. The exact shared-prefix workload in the main matrix is the valid cache result.

## 5. Concurrency

Vanilla 4096-token cold TTFT:

| concurrency | p50 | p90 | p99 |
|---:|---:|---:|---:|
| 1 | 1193.8 ms | 1197.4 ms | 1198.1 ms |
| 4 | 2420.7 ms | 3191.5 ms | 4741.9 ms |
| 16 | 10357.2 ms | 17440.6 ms | 19963.2 ms |

Queueing and head-of-line effects dominate under concurrency. A configuration that is neutral at c=1 can be much worse at c=16. The aggressive mode demonstrates this: FP8 KV cache plus `-O3` is not a safe general-purpose TTFT optimization for this hybrid model.

![Tier C concurrency result](plots_tier_c/ttft_concurrency.png)

## 6. Device-specific FP8 kernel tuning

vLLM 0.19.1 had no block-FP8 launch configurations for the RTX 6000 Ada and the five Qwen3.6 matrix shapes, so it used a generic Triton fallback and logged a warning.

`bench/tune_qwen36_fp8.py` performs a bounded search at `M=4096` and stores one default plus one 4096-specific configuration per shape. Microkernel gains were approximately 1.13–1.16× for four shapes; one shape retained the default. The exact JSON files are in `results_tier_c/fp8_configs/`.

This is the novel kernel-level contribution, but the end-to-end evidence sets its limit: tuned-only full-matrix latency is almost identical to vanilla. The focused tuned-plus-optimized 4096 cell is 1.12× faster. Kernel microbenchmarks are not substituted for request-level TTFT.

## 7. Supporting 4B findings

The RTX 4000 Ada study remains useful for mechanism selection:

1. Qwen3-4B FP8 improved cold TTFT by roughly 1.15–1.6× versus BF16 across the matrix.
2. A fitted 4B cache model was valid: `TTFT = 51.5 ms + 108.8 µs × uncached_tokens`, R² 0.9987, 3.4% holdout error.
3. Qwen3.5-4B hybrid Gated DeltaNet had a higher short-prompt fixed cost but scaled flatter than full attention, becoming 1.31× faster at 16k tokens.
4. Small chunked-prefill budgets hurt the homogeneous long-prompt workload; the expected short-request protection requires a mixed-length benchmark.

These findings motivated the conservative Tier C configuration. They are supporting results, not substitutes for the measured 27B numbers.

## 8. TGI baseline status

No valid TGI number was produced, and none is reported.

Qwen3.6-27B uses hybrid Gated DeltaNet/Mamba layers that TGI does not support. A three-engine comparison therefore has to use the classic `Qwen/Qwen3-4B-Instruct-2507` model. The current pod had no Docker daemon, and nested-container/chroot approaches lacked the required privileges. A native TGI 3.3.7 source build eventually loaded the 4B checkpoint, but the Rust router's embedded Python tokenizer worker failed (`_ctypes`/`charset_normalizer`/`huggingface_hub` initialization), causing every generation POST to close without a response. The harness correctly rejected the run and wrote no CSV.

The reliable next step is not more surgery in this environment: launch the official TGI container on a fresh pod, then run `bench/benchmark_ttft.py --api chat` from the client pod. This is documented in `EXECUTION.md` and `scripts/01_tgi.sh`.

## 9. Conclusion

The assignment is complete on the requested model class and a single 48 GB Ada GPU. Qwen3.6-27B-FP8 reaches **839.7 ms p50 cold TTFT at 3072 tokens**, with p99 still below one second. At 4096 cold tokens, the best measured result is **1063.2 ms**; prefix reuse reduces the 4096 result to **117.1 ms**.

The practical optimization order is:

1. use the official FP8 checkpoint on Ada;
2. cap context and graph capture so the hybrid model fits reliably;
3. preserve reusable prefixes and place volatile tokens at the end;
4. validate scheduler/compile flags under the intended concurrency instead of assuming they help;
5. treat device-specific kernel tuning as a measured, bounded improvement—not a replacement for end-to-end evidence.
