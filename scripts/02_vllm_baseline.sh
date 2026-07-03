#!/usr/bin/env bash
# ============================================================================
# 02_vllm_baseline.sh — VANILLA BASELINE #2: vLLM with default flags, BF16.
#
# NOTE for the report: modern vLLM (V1 engine) already enables prefix caching,
# chunked prefill and CUDA graphs BY DEFAULT. So "vanilla vLLM" is already a
# strong baseline — which is exactly why the ablation script (04) exists: it
# turns those features OFF one at a time so you can attribute the speedup to
# each mechanism instead of hand-waving.
# ============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct-2507"

vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --disable-log-requests
  # No other flags on purpose. max-model-len will default to the model's
  # native 262k context; on 20GB this may fail to allocate KV cache — if it
  # does, the *minimal* fix is: --max-model-len 16384 (document it).

# Benchmark:
#   python bench/benchmark_ttft.py --label vllm-vanilla --url http://localhost:8000
