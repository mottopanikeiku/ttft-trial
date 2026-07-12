#!/usr/bin/env bash
# End-to-end Qwen3.6-27B-FP8 benchmark on one 48 GB Ada GPU.
# Runs each requested mode with bounded startup/shutdown, preserves server logs,
# keeps Tier-C data separate from the 4B results, and regenerates analysis.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/workspace/hf}"

MODEL="Qwen/Qwen3.6-27B-FP8"
RESULTS_DIR="${RESULTS_DIR:-results_tier_c}"
PLOTS_DIR="${PLOTS_DIR:-plots_tier_c}"
SERVER_PID=""

if [[ "$#" -eq 0 ]]; then
  MODES=(vanilla tuned optimized aggressive)
else
  MODES=("$@")
fi
for mode in "${MODES[@]}"; do
  case "$mode" in
    vanilla|tuned|optimized|cache-all|aggressive) ;;
    *)
      echo "usage: $0 [vanilla] [tuned] [optimized] [cache-all] [aggressive]" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$RESULTS_DIR" "$PLOTS_DIR"

wait_ready() {
  local pid="$1"
  # A cold 27B startup may include checkpoint loading and graph capture. Bound
  # it at 30 minutes, while still failing immediately if the server exits.
  for _ in $(seq 1 900); do
    if curl -sf http://localhost:8000/v1/models >/dev/null; then
      sleep 5
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "FATAL: server process $pid died during startup" >&2
      return 1
    fi
    sleep 2
  done
  echo "FATAL: server did not become ready within 30 minutes" >&2
  return 1
}

stop_server() {
  local pid="${SERVER_PID:-}"
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
  fi
  # scripts/06 starts vLLM as a child of its shell wrapper. Killing only the
  # wrapper orphans the GPU-owning process, so stop the actual server too.
  pkill -f "vllm serve $MODEL" 2>/dev/null || true
  if [[ -n "$pid" ]]; then
    wait "$pid" 2>/dev/null || true
  fi
  SERVER_PID=""

  for _ in $(seq 1 90); do
    if ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
      sleep 8
      return 0
    fi
    sleep 2
  done
  echo "FATAL: port 8000 remained live after server shutdown" >&2
  return 1
}

cleanup() {
  stop_server || true
}
trap cleanup EXIT INT TERM

{
  date -Is
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python --version
  vllm --version
} > "$RESULTS_DIR/environment.txt"

run_mode() {
  local mode="$1"
  local label="qwen36-27b-$mode"
  local log="$RESULTS_DIR/server-$mode.log"

  echo "===================== $label ====================="
  stop_server
  rm -f "$RESULTS_DIR/$label.csv"
  bash scripts/06_qwen36_27b.sh "$mode" >"$log" 2>&1 &
  SERVER_PID=$!
  if ! wait_ready "$SERVER_PID"; then
    echo "server log: $log" >&2
    return 1
  fi

  python bench/benchmark_ttft.py \
    --label "$label" \
    --url http://localhost:8000 \
    --tokenizer "$MODEL" \
    --out "$RESULTS_DIR"

  if [[ "$mode" == "optimized" || "$mode" == "cache-all" ]]; then
    python bench/prefix_cache_sweep.py \
      --url http://localhost:8000 \
      --tokenizer "$MODEL" \
      | tee "$RESULTS_DIR/prefix_sweep_27b-$mode.txt"
  fi

  stop_server
}

for mode in "${MODES[@]}"; do
  run_mode "$mode"
done

if [[ -f "$RESULTS_DIR/qwen36-27b-vanilla.csv" ]]; then
  python bench/analyze.py \
    --baseline qwen36-27b-vanilla \
    --results "$RESULTS_DIR" \
    --plots "$PLOTS_DIR" \
    --include-label-regex '^qwen36-27b-'
fi

echo "DONE — see $RESULTS_DIR/ and $PLOTS_DIR/"
