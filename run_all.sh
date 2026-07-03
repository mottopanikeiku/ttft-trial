#!/usr/bin/env bash
# ============================================================================
# run_all.sh — end-to-end: vanilla vLLM -> ablations -> optimized -> sweep -> analysis.
# (TGI runs as its own pod/container — see scripts/01_tgi.sh — then drop its
#  CSV into results/ and re-run bench/analyze.py.)
# Budget: on an RTX 4000 Ada the full matrix takes roughly 1-2 hours.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

wait_ready() { until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 2; done; sleep 3; }

serve_and_bench() {  # $1 = server script, $2 = label, $3.. = extra bench args
  local script="$1" label="$2"; shift 2
  echo "===================== $label ====================="
  bash "$script" & local pid=$!
  wait_ready
  python bench/benchmark_ttft.py --label "$label" --url http://localhost:8000 "$@"
  kill "$pid"; wait "$pid" 2>/dev/null || true
  sleep 8
}

serve_and_bench scripts/02_vllm_baseline.sh  vllm-vanilla
bash scripts/04_vllm_ablation.sh

echo "===================== vllm-optimized + prefix sweep ====================="
bash scripts/03_vllm_optimized.sh & PID=$!
wait_ready
python bench/benchmark_ttft.py --label vllm-optimized --url http://localhost:8000
python bench/prefix_cache_sweep.py --url http://localhost:8000 | tee results/prefix_sweep.txt
kill "$PID"; wait "$PID" 2>/dev/null || true

python bench/analyze.py --baseline vllm-vanilla
echo "DONE — see results/summary.md, results/speedup.md, plots/"
