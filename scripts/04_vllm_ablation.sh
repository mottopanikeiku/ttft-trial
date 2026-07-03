#!/usr/bin/env bash
# ============================================================================
# 04_vllm_ablation.sh — attribute the speedup: toggle ONE mechanism at a time.
#
# Runs 3 degraded vLLM servers sequentially and benchmarks each:
#   ablation-no-prefixcache : prefix caching OFF   -> warm-path TTFT collapses
#   ablation-eager          : CUDA graphs/compile OFF -> short-prompt TTFT up
#   ablation-cp-512         : tiny chunked-prefill budget -> solo long-prompt
#                             TTFT up, but p99 under load IMPROVES (show both!)
#
# The ablation table is what turns "vLLM is fast" into "I know WHY it's fast."
# ============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct-2507"
BENCH="python bench/benchmark_ttft.py --url http://localhost:8000"

wait_ready() {
  echo "waiting for server..."
  until curl -sf http://localhost:8000/v1/models > /dev/null; do sleep 2; done
  sleep 3
}

run_case() {  # $1=label, rest = extra vllm flags
  local label="$1"; shift
  echo "=== $label ==="
  vllm serve "$MODEL" --host 0.0.0.0 --port 8000 --max-model-len 8192 \
      --disable-log-requests "$@" &
  local pid=$!
  wait_ready
  $BENCH --label "$label"
  kill "$pid"; wait "$pid" 2>/dev/null || true
  sleep 5
}

run_case ablation-no-prefixcache --no-enable-prefix-caching
run_case ablation-eager          --enforce-eager
run_case ablation-cp-512         --max-num-batched-tokens 512

echo "Ablations done. Run bench/analyze.py to compare."
