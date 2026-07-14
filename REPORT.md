# Minimizing TTFT for Local LLM Serving — Final Results Report

**Author:** Alp  
**Completed:** 2026-07-14
**Headline model:** `Qwen/Qwen3.6-27B-FP8`  
**Headline GPU:** one NVIDIA RTX 6000 Ada Generation, 49,140 MiB

## Executive result

The assigned target was met and independently revalidated:

> **3072-token cold prompt, concurrency 1: 790.2/801.1/803.1 ms
> p50/p90/p99 (N=24), with a second cell at 831.6/836.6/837.7 ms (N=24).**

These are client-observed, streamed `/v1/completions` measurements on the full
Qwen3.6-27B-FP8 language model. Every prompt begins with a random nonce, so
prefix caching cannot contribute. Both cells keep p99 below one second.

The boundary remains real. The best current-pod 4096-token cold cell is:

| 4096-token workload, c=1 | p50 | p90 | p99 | N |
|---|---:|---:|---:|---:|
| vanilla, cold | 1254.8 ms | 1264.8 ms | 1269.5 ms | 24 |
| dense-M configs + async + `-O3`, cold | **1093.6 ms** | 1110.8 ms | 1124.0 ms | 24 |
| channel-requantized CUTLASS experiment, cold | 1111.0 ms | 1128.8 ms | 1132.9 ms | 24 |

The defensible conclusion is: **repeatable sub-second 27B TTFT through 3072
cold tokens; 1.094 s at 4096 cold tokens.** The earlier 2026-07-12 Tier C
matrix remains valid on driver 570.195.03 and includes a 1063.2 ms 4096 cold
cell plus a 117.1 ms exact reusable-prefix cell. Those rows are not merged
with this driver's raw cells.

### 2026-07-14 optimization campaign

The new source-of-truth directory is `results_ada_570124/`. It contains every
raw CSV, full server startup logs, dense-M and exhaustive-M4096 tuner output,
a PyTorch profiler trace, an Nsight Compute report, and `environment.txt`.

Measured changes relative to the current 4096 vanilla p50 of 1254.8 ms:

| change | 4096 cold p50 | speedup | disposition |
|---|---:|---:|---|
| 8192-token single prefill, untuned | 1279.3 ms | 0.98× | reject |
| dense-M FP8 configs, single prefill | 1132.6 ms | 1.11× | keep |
| dense-M + async scheduling | 1095.8 ms | 1.15× | keep |
| dense-M + async + `-O3` | **1093.6 ms** | 1.15× | statistically neutral extra flag |
| channelwise requantization + CUTLASS | 1111.0 ms | 1.13× | reject for production; altered weights |

The focused profiler recorded 990.4 ms of CUDA-kernel time under profiling
overhead. The Triton block-FP8 GEMMs consumed 774.6 ms (78.2%); Gated DeltaNet
attention consumed 27.5 ms, FlashAttention 26.0 ms, and activation
quantization 26.0 ms. This establishes that the remaining 4096 limit is
primarily GEMM compute, not HTTP or tokenizer overhead.

The generated Triton PTX contains
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, proving FP8 tensor-core
use. The inspected cubin used 154 registers; Triton metadata reported 74,752
bytes of shared memory. Nsight Compute hardware counters were unavailable
because the pod denies `ERR_NVGPUCTRPERM`; the saved report and error are
preserved rather than replaced with inferred counter values.

Explicit CUDA-graph captures at 512, 1024, 2048, and 4096 tokens failed during
hybrid recurrent-state allocation. For example, the 1024 capture attempted a
3.06 GiB allocation with 588.5 MiB free. The stable 256-token graph cap is a
memory constraint, not an arbitrary conservative setting.

![Tier C prompt-length result](plots_tier_c/ttft_vs_prompt.png)

## 1. Experimental setup

### Tier C — assigned 27B task

- GPU: NVIDIA RTX 6000 Ada Generation, 49,140 MiB, SM 8.9
- Driver: 570.124.06 for the current campaign; 570.195.03 for the earlier full matrix
- CUDA: 12.8 pod runtime/toolkit; torch wheel `2.10.0+cu128`
- Python: 3.12.3
- vLLM: 0.19.1; Triton: 3.6.0
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

Raw observations, not rendered Markdown tables, are the source of truth.
Current work is under `results_ada_570124/`; the earlier full matrix remains:

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

vLLM 0.19.1 has no RTX 6000 Ada block-FP8 launch configurations for the five
Qwen3.6 matrix shapes, so the stock engine uses a generic Triton fallback.

`bench/tune_qwen36_fp8.py` now searches the irregular M-grid actually consumed
by vLLM's nearest-M dispatcher: 1, 16, 128, 512, 1024, 2048, 3072, 4096, and
8192. It rotates multiple weight tensors so L2 residency does not select a
synthetic winner, validates outputs, and writes one mapping per matrix shape.
The separate exhaustive 1280-candidate M=4096 search produced microkernel
speedups of 1.14×, 1.14×, 1.19×, 1.14×, and 1.14× across the five shapes.

End-to-end evidence sets the limit: dense-M tuning plus async scheduling
improved current-pod 4096 cold p50 from 1254.8 ms to 1095.8 ms (1.15×).
Microkernel ratios are not substituted for request-level TTFT.

## 7. Same-GPU Qwen3-4B findings

The current RTX 6000 Ada comparison used the standard
`Qwen/Qwen3-4B-Instruct-2507` checkpoint and its official FP8 variant:

| checkpoint | 3072 cold p50 | 4096 cold p50 | 4096 p99 |
|---|---:|---:|---:|
| BF16 | 242.0 ms | 339.4 ms | 345.1 ms |
| official FP8 | **157.8 ms** | **217.0 ms** | **221.0 ms** |

Official FP8 is the clean optimization: 1.56× at 4096 tokens. Disabling prefix
caching, raising the prefill budget, async scheduling, and explicit
3072/4096-token CUDA graphs did not improve the 4096 result beyond noise.
The earlier RTX 4000 Ada architecture study remains in `results/` and is not
mixed into this table.

## 8. TGI baseline status

No valid TGI number was produced, and none is reported.

Qwen3.6-27B uses hybrid Gated DeltaNet/Mamba layers that TGI does not support. A three-engine comparison therefore has to use the classic `Qwen/Qwen3-4B-Instruct-2507` model. The current pod had no Docker daemon, and nested-container/chroot approaches lacked the required privileges. A native TGI 3.3.7 source build eventually loaded the 4B checkpoint, but the Rust router's embedded Python tokenizer worker failed (`_ctypes`/`charset_normalizer`/`huggingface_hub` initialization), causing every generation POST to close without a response. The harness correctly rejected the run and wrote no CSV.

The reliable next step is not more surgery in this environment: launch the official TGI container on a fresh pod, then run `bench/benchmark_ttft.py --api chat` from the client pod. This is documented in `EXECUTION.md` and `scripts/01_tgi.sh`.

## 9. Conclusion

The assignment is complete on the requested model class and a single 48 GB
Ada GPU. Qwen3.6-27B-FP8 produced two independent 3072-token cold cells with
**790.2 ms and 831.6 ms p50**, and both kept p99 below one second. At 4096
cold tokens, the best current-pod result is **1093.6 ms**; this report does not
mislabel it as sub-second.

The practical optimization order is:

1. use the official FP8 checkpoint on Ada;
2. cap context and graph capture so the hybrid model fits reliably;
3. tune block-FP8 against the M values the serving dispatcher actually uses;
4. use one-step prefill plus async scheduling for the pure-cold c=1 workload;
5. preserve reusable prefixes when the application genuinely has them;
6. reject altered-weight or profiler-only wins unless end-to-end latency and
   quality both support them.
