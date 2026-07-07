#!/usr/bin/env bash
# ============================================================================
# 04_vllm_ablation.sh — attribute the speedup: toggle ONE mechanism at a time.
#
# Runs 3 degraded vLLM servers sequentially and benchmarks each:
#   ablation-no-prefixcache : prefix caching OFF   -> warm-path TTFT collapses
#   ablation-eager          : CUDA graphs/compile OFF -> short-prompt TTFT up
#   ablation-cp-512         : tiny chunked-prefill budget -> shows the tradeoff;
#                             in this homogeneous matrix it regressed c=16 p99.
#
# The ablation table is what turns "vLLM is fast" into "I know WHY it's fast."
# ============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct-2507"
BENCH="python bench/benchmark_ttft.py --url http://localhost:8000"

wait_ready() {
  local pid="$1"
  echo "waiting for server..."
  for _ in $(seq 1 360); do
    curl -sf http://localhost:8000/v1/models >/dev/null && { sleep 3; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "FATAL: server process $pid died during startup"; return 1; }
    sleep 2
  done
  echo "FATAL: server never became ready"; return 1
}

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  local closed=0
  for _ in $(seq 1 60); do
    if ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
      closed=1
      break
    fi
    sleep 2
  done
  if [[ "$closed" -ne 1 ]]; then
    echo "FATAL: port 8000 still served /v1/models after shutdown"
    return 1
  fi
  sleep 5
}

run_case() {  # $1=label, rest = extra vllm flags
  local label="$1" pid status=0
  shift
  echo "=== $label ==="
  stop_server || true
  vllm serve "$MODEL" --host 0.0.0.0 --port 8000 --max-model-len 8192 "$@" &
  pid=$!
  if wait_ready "$pid"; then
    if $BENCH --label "$label"; then
      status=0
    else
      status=$?
    fi
  else
    status=$?
  fi
  kill "$pid" 2>/dev/null || true
  stop_server || status=$?
  return "$status"
}

run_case ablation-no-prefixcache --no-enable-prefix-caching
run_case ablation-eager          --enforce-eager
run_case ablation-cp-512         --max-num-batched-tokens 512

echo "Ablations done. Run bench/analyze.py to compare."
