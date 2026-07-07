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
MODE="${1:-optimized}"   # usage: bash scripts/06_qwen36_27b.sh [vanilla|optimized|aggressive]

if [[ "$MODE" == "vanilla" ]]; then
  # Minimal flags: only what's needed for text-only boot and the benchmark
  # context. The model is multimodal; skipping the vision encoder is a
  # survival prerequisite, not a TTFT tuning knob. Default 262k context would
  # over-allocate KV cache on a 48 GB card, so cap to the Tier-C prompt window.
  vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --language-model-only \
    --max-model-len 8192 \
    --reasoning-parser qwen3
elif [[ "$MODE" == "optimized" ]]; then
  # Evidence-based config from Tier A (scripts/03 header, results/speedup.md):
  # FP8 checkpoint (inherent to $MODEL) + capped max-model-len + one-step
  # chunk budget carry the win. -O3 and --kv-cache-dtype fp8 are deliberately
  # NOT here: Tier-A data shows -O3 adds nothing over default CUDA-graph
  # capture, and BOTH are the known hybrid-GDN failure modes (EXECUTION.md
  # Phase 5) — don't bet the rented pod's boot on them. Try them via the
  # 'aggressive' mode and record the delta as ablation data.
  vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --language-model-only \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --reasoning-parser qwen3
else
  # 'aggressive': optimized + the two risky flags, run AFTER optimized has
  # produced numbers. If it crashes (CUDA-graph/mamba cache or fp8-KV on the
  # hybrid arch), that's a recorded finding, not lost time.
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
# Run order on the pod: vanilla -> optimized (the headline) -> aggressive
# (delta of -O3 + fp8-KV, keep whatever survives and record the rest).
