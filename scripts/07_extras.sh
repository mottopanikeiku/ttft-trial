#!/usr/bin/env bash
# ============================================================================
# 07_extras.sh — two follow-up experiments after the Phase-2 matrix:
#
#  1) vllm-fp8-only: the FP8 checkpoint with OTHERWISE-VANILLA flags.
#     scripts/03 changes five things at once (FP8, max-model-len, chunk
#     budget, mem-util, -O3); this run isolates FP8's own contribution the
#     same way scripts/04 isolated vanilla's features.
#
#  2) Tier B architecture study (EXECUTION.md Phase 4): classic full
#     attention (Qwen3-4B) vs hybrid Gated-DeltaNet (Qwen3.5-4B), IDENTICAL
#     server flags, cold-only, c=1, prompt lengths 128 -> 16384. The plot of
#     these two scaling curves is the novel result of the project.
#     max-model-len 16640 = 16384 prompt + 16 gen + headroom (identical for
#     both models; the ONE flag besides host/port).
#
# Deliberately NOT `set -e`: if the hybrid model fails to boot on this vLLM
# version, we record the failure and keep the rest of the data.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

wait_ready() {
  for _ in $(seq 1 360); do
    curl -sf http://localhost:8000/v1/models >/dev/null && { sleep 3; return 0; }
    pgrep -f "vllm serve" >/dev/null || { echo "FATAL: vllm server process died during startup"; return 1; }
    sleep 2
  done
  echo "FATAL: server never became ready"; return 1
}

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  for _ in $(seq 1 60); do
    curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 || break
    sleep 2
  done
  sleep 8
}

echo "===================== vllm-fp8-only ====================="
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --host 0.0.0.0 --port 8000 --max-model-len 16384 &
if wait_ready; then
  python bench/benchmark_ttft.py --label vllm-fp8-only --url http://localhost:8000 \
      --tokenizer Qwen/Qwen3-4B-Instruct-2507-FP8
fi
stop_server

echo "===================== arch-full-attn ====================="
vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port 8000 --max-model-len 16640 &
if wait_ready; then
  python bench/benchmark_ttft.py --label arch-full-attn --url http://localhost:8000 \
      --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
      --prompt-tokens 128 512 2048 8192 16384 --concurrency 1 --cache-modes cold
fi
stop_server

echo "===================== arch-hybrid-gdn ====================="
vllm serve Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8000 --max-model-len 16640 &
if wait_ready; then
  python bench/benchmark_ttft.py --label arch-hybrid-gdn --url http://localhost:8000 \
      --tokenizer Qwen/Qwen3.5-4B \
      --prompt-tokens 128 512 2048 8192 16384 --concurrency 1 --cache-modes cold
fi
stop_server

echo "EXTRAS DONE"
