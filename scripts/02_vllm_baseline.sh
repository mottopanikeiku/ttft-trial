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
  --max-model-len 16384
  # --max-model-len 16384 is the ONE deviation from stock defaults (report it):
  # the model's native 262k context needs more KV cache than a 20 GB card can
  # spare after weights, so the engine may refuse to boot. 16k still exceeds
  # every Tier-A prompt length by 4x. No other flags on purpose.

# Benchmark:
#   python bench/benchmark_ttft.py --label vllm-vanilla --url http://localhost:8000
