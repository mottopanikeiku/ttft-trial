#!/usr/bin/env bash
# ============================================================================
# run_all.sh — end-to-end: vanilla vLLM -> ablations -> optimized -> sweep -> analysis.
# (TGI runs as its own pod/container — see scripts/01_tgi.sh — then drop its
#  CSV into results/ and re-run bench/analyze.py.)
# Budget: on an RTX 4000 Ada the full matrix takes roughly 1-2 hours.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Bounded readiness wait: abort (instead of hanging forever) if the server
# process dies during startup or never comes up within ~12 min.
wait_ready() {
  local pid="$1"
  for _ in $(seq 1 360); do
    curl -sf http://localhost:8000/v1/models >/dev/null && { sleep 3; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "FATAL: server process $pid died during startup"; return 1; }
    sleep 2
  done
  echo "FATAL: server never became ready"; return 1
}

# Killing the wrapper bash does NOT kill its `vllm serve` child — the orphan
# keeps port 8000 and GPU memory, wedging every later server. Kill the actual
# vllm process and wait for the port to actually close before moving on.
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
  sleep 8   # let the process exit fully and GPU memory drain
}

serve_and_bench() {  # $1 = server script, $2 = label, $3.. = extra bench args
  local script="$1" label="$2" pid status=0
  shift 2
  echo "===================== $label ====================="
  stop_server || true
  bash "$script" & pid=$!
  if wait_ready "$pid"; then
    if python bench/benchmark_ttft.py --label "$label" --url http://localhost:8000 "$@"; then
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

serve_and_bench scripts/02_vllm_baseline.sh  vllm-vanilla
bash scripts/04_vllm_ablation.sh

echo "===================== vllm-fp8-only optimized + prefix sweep ====================="
stop_server || true
bash scripts/03_vllm_optimized.sh & PID=$!
STATUS=0
if wait_ready "$PID"; then
  if python bench/benchmark_ttft.py --label vllm-fp8-only --url http://localhost:8000 \
      --tokenizer Qwen/Qwen3-4B-Instruct-2507-FP8; then
    if python bench/prefix_cache_sweep.py --url http://localhost:8000 \
        --tokenizer Qwen/Qwen3-4B-Instruct-2507-FP8 | tee results/prefix_sweep.txt; then
      STATUS=0
    else
      STATUS=$?
    fi
  else
    STATUS=$?
  fi
else
  STATUS=$?
fi
kill "$PID" 2>/dev/null || true
stop_server || STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
  exit "$STATUS"
fi

python bench/analyze.py --baseline vllm-vanilla
echo "DONE — see results/summary.md, results/speedup.md, plots/"
