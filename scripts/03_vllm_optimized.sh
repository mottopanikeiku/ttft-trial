#!/usr/bin/env bash
# ============================================================================
# 03_vllm_optimized.sh — the TTFT-optimized configuration.
#
# Every flag below exists for a reason you should be able to defend:
#
#   FP8 checkpoint      Prefill is COMPUTE-bound; Ada (SM 8.9) has native FP8
#                       tensor cores -> ~up to 2x matmul throughput vs BF16,
#                       plus half the weight memory (more room for KV cache).
#                       Quality loss for Qwen's official FP8 is negligible.
#
#   --max-model-len     Default is the model's 262k native context. Shrinking
#                       to what the benchmark needs (8k) means smaller KV
#                       allocation, better block utilization, faster startup.
#
#   --max-num-batched-tokens
#                       The chunked-prefill token budget per scheduler step.
#                       Bigger = a single cold prefill finishes in fewer steps
#                       (better solo TTFT); smaller = less head-of-line
#                       blocking under load (better p99 TTFT at concurrency).
#                       8192 lets our largest 4k prompt prefill in ONE step.
#                       Sweep {2048, 8192} and show the tradeoff.
#
#   --gpu-memory-utilization 0.92
#                       More KV cache -> fewer preemptions/requeues under load.
#
#   --enable-prefix-caching
#                       Explicit (default-on in V1) so the config is
#                       self-documenting. This is the warm-path superpower:
#                       cached prefix tokens skip prefill entirely.
#
#   -O3 (compilation)   Full torch.compile + CUDA-graph capture where
#                       supported; removes per-step CPU launch overhead, which
#                       is a visible slice of TTFT for SHORT prompts.
# ============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct-2507-FP8"

vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  -O3 \
  --disable-log-requests

# Benchmark:
#   python bench/benchmark_ttft.py --label vllm-optimized --url http://localhost:8000
#
# Extra credit datapoint (INT4, for the FP8-vs-INT4 prefill study):
#   vllm serve Qwen/Qwen3-4B-AWQ ...   (or quantize yourself with llm-compressor)
#   --label vllm-awq
# Expect: decode throughput up, TTFT flat-to-worse vs BF16. Explain why.
