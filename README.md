# ttft-trial — Qwen3.6-27B TTFT on one 48 GB Ada GPU

Measured answer to the assignment: run a Qwen 27B model locally, minimize client-observed time to first token (TTFT), and cross **TTFT < 1 s**.

## Result

**Target met and revalidated on 2026-07-14.** `Qwen/Qwen3.6-27B-FP8`
on one NVIDIA RTX 6000 Ada produced two independent 24-request cold cells:

| workload | configuration | p50 | p90 | p99 | samples |
|---|---|---:|---:|---:|---:|
| 3072-token cold prompt, c=1 | dense-M FP8 configs + cold mode, run A | **790.2 ms** | 801.1 ms | 803.1 ms | 24 |
| 3072-token cold prompt, c=1 | dense-M FP8 configs + cold mode, run B | **831.6 ms** | 836.6 ms | 837.7 ms | 24 |
| 4096-token cold prompt, c=1 | dense-M FP8 configs + async + `-O3` | **1093.6 ms** | 1110.8 ms | 1124.0 ms | 24 |
| Qwen3-4B BF16, 4096 cold, c=1 | vanilla | 339.4 ms | 342.4 ms | 345.1 ms | 24 |
| Qwen3-4B FP8, 4096 cold, c=1 | vanilla | **217.0 ms** | 219.6 ms | 221.0 ms | 24 |

The claim remains deliberately scoped: the full 27B model is repeatably below
one second through 3072 cold tokens, including p99. Its best 4096-token cold
p50 on this pod is 1.094 s, not sub-second. The official 4B FP8 checkpoint is
comfortably sub-second at 4096 tokens and is 1.56× faster than 4B BF16.

Current raw rows, server logs, tuner output, profiler trace, and Nsight Compute
report are in [`results_ada_570124/`](results_ada_570124/). The earlier full
Tier C matrix and plots remain in [`results_tier_c/`](results_tier_c/).

## Hardware and software

The 2026-07-14 validation campaign used:

- NVIDIA RTX 6000 Ada Generation, 49,140 MiB, SM 8.9
- driver 570.124.06; CUDA runtime 12.8
- Python 3.12.3
- vLLM 0.19.1; torch 2.10.0+cu128; Triton 3.6.0
- `Qwen/Qwen3.6-27B-FP8`, text-only serving
- `Qwen/Qwen3-4B-Instruct-2507` BF16 and the official FP8 checkpoint

The earlier full Tier C matrix used driver 570.195.03 on the same GPU model.
Tier A/B data in [`results/`](results/) came from an RTX 4000 Ada 20 GB session
and must not be mixed into same-hardware engine speedups.

## What was measured

Client-observed TTFT is the interval from sending the HTTP request to receiving the first streamed token from `/v1/completions`.

- Exact tokenizer-length prompts: 128, 512, 1024, 2048, and 4096 tokens.
- Concurrency: 1, 4, and 16.
- Cache modes:
  - `cold`: a unique nonce is placed at the beginning, forcing a prefix-cache miss;
  - `warm`: requests share the long prefix and place a unique nonce at the end.
- Temperature 0; 16 generated tokens; warmups discarded; normally 24 measured requests per cell.
- The harness rejects a cell below the configured success fraction instead of writing partial data silently.

The one-cell 3072-token boundary run is in `results_tier_c/qwen36-27b-sub1.csv`. The full vanilla, tuned, optimized, and aggressive matrices are separate CSVs in the same directory.

## Tier C mode definitions

[`scripts/06_qwen36_27b.sh`](scripts/06_qwen36_27b.sh) contains the authoritative flags.

| mode | purpose | measured conclusion |
|---|---|---|
| `vanilla` | Minimum survival flags: text-only model, 8192 context, CUDA-graph capture capped at 256 | Current-pod 4096 cold p50 1254.8 ms |
| `tuned` | Vanilla scheduling plus RTX 6000 Ada block-FP8 GEMM configs | Device configs improve the kernels, but request-level transfer depends on the scheduler token count |
| `cold` | Dense-M FP8 configs, 8192-token prefill budget, async scheduling, prefix caching disabled | Default cold path; repeatable 3072 p99 below one second |
| `optimized` | 8192-token chunk budget, 0.92 GPU utilization, prefix caching | Warm-prefix mode; not the pure-cold default |
| `aggressive` | Optimized plus FP8 KV cache and `-O3` | Not a general win in the earlier full matrix |
| `cache-all` | Optional hybrid-Mamba cache experiment | Implemented but not part of the completed measured matrix |
| `ada-channel` | Requantize block-FP8 weights per output channel and use CUTLASS | Experimental altered-weight path; 4096 p50 1111.0 ms, so it is not promoted |

The current dense-M configurations are preserved in
`results_ada_570124/fp8_configs_v2/`; the exhaustive M=4096 search is in
`fp8_configs_v2_full/`. They are device-specific and not portable performance
claims for other GPUs.

## Main findings

1. **Sub-second 27B cold TTFT is repeatable at 3072 tokens.** Two independent 24-request cells kept p99 at 803.1 ms and 837.7 ms.
2. **4096 cold tokens remain above the target.** The current baseline is 1254.8 ms p50; the best measured current-pod cell is 1093.6 ms, a 1.15× speedup.
3. **The profiler identifies the hard limit.** A focused 4096 request spent 990.4 ms in CUDA kernels under profiler overhead; block-FP8 GEMMs accounted for 774.6 ms (78.2%).
4. **PTX is already using Ada FP8 tensor cores.** Triton emitted `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`; the inspected cubin used 154 registers, and Triton reported 74,752 bytes of shared memory.
5. **Larger prefill CUDA graphs do not fit this hybrid model.** Explicit 512, 1024, 2048, and 4096 captures failed during recurrent-state allocation; the 256 cap remains a capacity constraint.
6. **Official Qwen3-4B FP8 is the clean small-model win.** At 4096 cold tokens it reduced p50 from 339.4 ms BF16 to 217.0 ms FP8 (1.56×). Larger prefill graphs and the cold scheduling flags did not improve that 4096 cell further.
7. **The channelwise CUTLASS experiment is not a production result.** It changes weight quantization, lacks quality evaluation, and did not beat the best unmodified-weight 4096 result.

## TGI status

There is **no valid TGI TTFT number in this repository**. Do not fill one in from failed attempts.

- TGI does not support the hybrid Qwen3.6-27B architecture, so it cannot be the 27B engine baseline.
- A fair TGI comparison must use the classic `Qwen/Qwen3-4B-Instruct-2507` model.
- This pod had no Docker daemon. A source-built TGI 3.3.7 stack loaded the 4B checkpoint, but its embedded Python tokenizer worker failed and every POST connection closed; no CSV was accepted.
- The reliable continuation is a fresh RunPod using the official TGI container, then run the harness with `--api chat`. See `scripts/01_tgi.sh` and `EXECUTION.md`.

## Reproduce

Fresh RTX 6000 Ada pod:

```bash
bash scripts/00_setup.sh tier-c
python bench/tune_qwen36_fp8.py \
  --m-values 1 16 128 512 1024 2048 3072 4096 8192 \
  --out results_ada_570124/fp8_configs_v2 \
  --install
bash scripts/06_qwen36_27b.sh cold
```

In another shell, run the exact cold boundary:

```bash
python bench/benchmark_ttft.py \
  --label qwen36-27b-cold \
  --url http://localhost:8000 \
  --tokenizer Qwen/Qwen3.6-27B-FP8 \
  --prompt-tokens 3072 4096 \
  --concurrency 1 --cache-modes cold --num-requests 24 \
  --out results_ada_570124
```

Focused harness tests:

```bash
python -m unittest tests.test_ttft_harness
```

## Repository map

- `bench/benchmark_ttft.py` — exact-token asynchronous TTFT harness
- `bench/prefix_cache_sweep.py` — cache sweep; rejects non-physical fits
- `bench/tune_qwen36_fp8.py` — dense-M RTX 6000 Ada block-FP8 kernel tuner
- `bench/ada_channel_fp8/sitecustomize.py` — opt-in, altered-weight CUTLASS experiment
- `bench/analyze.py` — summary, speedup tables, and plots
- `scripts/06_qwen36_27b.sh` — Tier C server modes
- `run_tier_c.sh` — earlier full Tier C orchestration
- `results_ada_570124/` — current raw rows, logs, configs, profiler, and Nsight evidence
- `plots_ada_570124/` — current 27B figures
- `results_tier_c/`, `plots_tier_c/` — earlier full 27B matrix
- `results/`, `plots/` — earlier 4B and architecture-study evidence
- `DEBUGLOG.md` — actionable failure and negative-result ledger
