# ttft-trial — Minimizing Time-To-First-Token for Local LLM Deployment

Task (Veysel): deploy locally, hit **TTFT < 1 s on the Qwen 27B**, show vanilla
HF TGI + vLLM baselines first, then quantify how much you can speed it up.

## 0. Model landscape check (July 2026) — this reframes the task

- **`Qwen/Qwen3.6-27B` exists** (dense, multimodal, released 2026-04-22), with an
  official **`Qwen/Qwen3.6-27B-FP8`** checkpoint (~27 GB weights) that runs
  **single-GPU on a 48 GB card** with vLLM ≥ 0.19. So the original "qwen 27B,
  TTFT < 1 s" task is literally doable — no need to permanently downgrade to 4B.
- Qwen3.5/3.6 use a **hybrid architecture: Gated DeltaNet linear attention +
  full attention (3:1)**. Consequences for this project:
  - Prefill over linear-attention layers is ~O(L) with small constants →
    TTFT scaling vs prompt length behaves differently than classic transformers.
    Measuring this IS the novel experiment (Tier B below).
  - **TGI does not support this architecture.** TGI's own README now points
    users to vLLM/SGLang going forward — it's effectively in maintenance mode.
    So the 3-engine "vanilla" comparison must use a classic-architecture model
    that all three engines support: **`Qwen/Qwen3-4B-Instruct-2507`** (+ its
    official FP8 checkpoint). State this explicitly in the report; it's a
    legitimate finding, not a cop-out.
  - Qwen3.6 defaults to *thinking mode* and its chat template opens a `<think>`
    block. Our harness measures TTFT via the raw `/v1/completions` endpoint,
    which bypasses the chat template entirely → clean, comparable numbers.
  - MTP (multi-token prediction) speculative decoding is built in — it speeds
    up *decode*, not TTFT. Say so; knowing what doesn't help is signal.

## 1. Hardware plan

| Tier | GPU | Cost | What runs there |
|---|---|---|---|
| A + B | **RTX 4000 Ada, 20 GB** (already rented) | cheap, keep it | harness dev, 3-engine vanilla comparison, optimization ladder, prefix sweep, architecture study (4B models) |
| C | **L40S or RTX 6000 Ada, 48 GB** (rent for ~3-4 h) | ~$3-5 total | the actual task: Qwen3.6-27B-FP8, TTFT < 1 s, vanilla vs optimized |

Why these cards: all are Ada (SM 8.9) → native **FP8 tensor cores** +
FlashAttention-2. Avoid A100 for Tier C (Ampere = no native FP8 compute).
A 20 GB card cannot host the 27B: even the NVFP4 checkpoint is ~22 GB weights.
Develop cheap on the 4000 Ada, then do one short, well-rehearsed session on the
48 GB card.

## 2. TTFT theory (know this cold)

```
TTFT = HTTP/SSE overhead + tokenization + scheduler queueing
     + PREFILL (dominant) + first sampled token + detokenize/flush
```
- Prefill is **compute-bound** → FP8 helps TTFT on Ada; weight-only INT4
  (AWQ) mainly helps decode (bandwidth-bound) and can even hurt prefill.
- TTFT grows ~linearly with prompt length on classic transformers → always
  report TTFT *vs prompt length*, never one number. Hybrid GDN models should
  show a flatter long-context curve — measure it (Tier B).
- **Prefix caching is the biggest lever**: cached prefix tokens skip prefill.
  Benchmark both cold (unique prefix → guaranteed miss) and warm (shared
  prefix) — most people accidentally benchmark only cache hits.
- Under concurrency, queueing/head-of-line blocking dominates p99 TTFT →
  chunked prefill budget (`--max-num-batched-tokens`) is the tradeoff knob.

## 3. Configuration ladder

Tier A (4B, classic arch — all engines):
| label | stack |
|---|---|
| `naive-hf` | plain transformers + FastAPI (reference floor) |
| `tgi-vanilla` | TGI docker, default flags |
| `vllm-vanilla` | `vllm serve`, defaults, BF16 |
| `ablation-*` | prefix-cache / cudagraphs / chunked-prefill toggled individually |
| `vllm-optimized` | FP8 + tuned flags (scripts/03) |
| `sglang` | optional cross-check (RadixAttention) |

Tier B (novel, 4000 Ada): `Qwen3-4B-Instruct-2507` (full attention) vs
`Qwen/Qwen3.5-4B` (hybrid GDN) — TTFT vs prompt length 128→16k, same vLLM,
same flags. Plot the scaling curves on log-log; fit slopes.

Tier C (48 GB card): `Qwen3.6-27B-FP8` vanilla vs optimized (scripts/06) —
the < 1 s headline, plus the prefix-cache sweep at 27B scale.

## 4. Runbook (RunPod workflow)

**Terminal discipline:** you're on Jupyter Lab — its terminals die when the
tab/kernel does, killing your vLLM server mid-benchmark. Use **tmux** (or SSH
in and use tmux) — `apt install tmux`; one pane for the server, one for the
client. Keep `HF_HOME=/workspace/hf` so weights live on the persistent volume,
not the container disk.

```bash
git clone <your-github>/ttft-trial && cd ttft-trial
bash scripts/00_setup.sh

# Tier A, end-to-end vLLM half (vanilla -> ablations -> optimized -> sweep -> plots):
bash run_all.sh
# TGI runs as its own RunPod pod using the TGI container image (scripts/01_tgi.sh
# header explains); benchmark it remotely, drop the CSV into results/, re-run analyze.

# Tier B (architecture study), same server flags for both models:
vllm serve Qwen/Qwen3.5-4B --max-model-len 16384 &
python bench/benchmark_ttft.py --label qwen35-4b-hybrid --tokenizer Qwen/Qwen3.5-4B \
    --prompt-tokens 128 512 2048 8192 16384 --concurrency 1 --cache-modes cold

# Tier C (on the 48GB pod):
bash scripts/06_qwen36_27b.sh    # contains vanilla + optimized variants
```

## 5. Deliverables

`results/summary.md`, `results/speedup.md`, three plots, prefix-sweep fitted
model (`TTFT = a + b·uncached_tokens`, R², holdout error), ablation table,
architecture-scaling plot, and the one-page report (report_template.md).
Push everything to the `ttft-trial` GitHub repo with the raw CSVs — reviewers
trust raw data.

## 6. Sanity expectations (verify, don't trust)

4B on 4000 Ada, cold, c=1: naive-hf seconds; vanilla vLLM roughly tens-of-ms
at 128 tok to high-hundreds-of-ms at 4k; warm-cache tens of ms regardless of
length. 27B-FP8 on 48 GB: prefill throughput should put 4k-token cold TTFT
comfortably under 1 s; if not, check chunked-prefill budget and that FP8
kernels actually engaged (watch startup logs).
