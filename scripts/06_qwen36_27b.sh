#!/usr/bin/env bash
# ============================================================================
# 06_qwen36_27b.sh — TIER C: the actual task. Qwen3.6-27B-FP8, single 48 GB
# Ada-class GPU (L40S / RTX 6000 Ada), vLLM >= 0.19. Target: TTFT < 1 s.
#
# Architecture notes that matter (Qwen3.6 = hybrid Gated DeltaNet + full
# attention, 3:1, multimodal, thinking-by-default):
#   --language-model-only      skip loading the vision encoder -> frees GBs
#                              for KV cache; we only benchmark text TTFT.
#   --reasoning-parser qwen3   correct parsing of <think> blocks (harmless for
#                              us since we benchmark via /v1/completions).
#   DeltaNet recurrent state is handled via vLLM's mamba cache machinery; if
#   you hit a CUDA-graph/mamba cache size error, reduce
#   --max-cudagraph-capture-size (known issue, vLLM PR #34571).
#   MTP speculative decoding exists (--speculative-config mtp) but improves
#   DECODE, not TTFT -> deliberately excluded; mention in the report.
# ============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3.6-27B-FP8"
MODE="${1:-optimized}"   # usage: bash scripts/06_qwen36_27b.sh [vanilla|optimized]

if [[ "$MODE" == "vanilla" ]]; then
  # Minimal flags: only what's needed to boot at all on 48 GB.
  # (Default 262k max-model-len will OOM the KV cache -> cap it; that cap is
  #  the ONE deviation from stock defaults, document it.)
  vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 \
    --reasoning-parser qwen3
else
  vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --language-model-only \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    -O3
fi

# Benchmark (other tmux pane):
#   python bench/benchmark_ttft.py --label qwen36-27b-$MODE \
#       --url http://localhost:8000 --tokenizer Qwen/Qwen3.6-27B-FP8
#   python bench/prefix_cache_sweep.py --url http://localhost:8000 \
#       --tokenizer Qwen/Qwen3.6-27B-FP8
#
# If --kv-cache-dtype fp8 or -O3 misbehaves with the hybrid arch on your vLLM
# version, drop them one at a time — and record the delta: that's ablation data.
