#!/usr/bin/env bash
# ============================================================================
# 03_vllm_optimized.sh — the current 4B TTFT-optimized configuration.
#
# MEASURED VERDICT (Phase 2 + scripts/07 fp8-isolation run):
#
#   * The promoted optimized config is the official FP8 checkpoint with the
#     minimal context cap needed on the 20 GB dev card. This is the existing
#     raw `vllm-fp8-only` datapoint; keep that label for traceability.
#   * vLLM V1 already enables prefix caching, chunked prefill and CUDA-graph
#     capture by default. Re-stating those defaults did not buy measurable TTFT.
#   * The old 5-flag bundle (`--max-num-batched-tokens 8192`,
#     `--gpu-memory-utilization 0.92`, explicit prefix caching, and `-O3`)
#     regressed several cold-concurrency cells. `-O3` in particular was slower
#     than default graph capture at 512tok/c16 (828 ms vs 428 ms).
#
#   => Do not pass `-O3` in the default optimized path. Keep risky or redundant
#      flags as optional ablation data, not as the baseline we ask people to
#      reproduce before Tier-C runs.
#
# Every active flag below exists for a reason you should be able to defend:
#
#   FP8 checkpoint      Prefill is COMPUTE-bound; Ada (SM 8.9) has native FP8
#                       tensor cores -> ~up to 2x matmul throughput vs BF16,
#                       plus half the weight memory (more room for KV cache).
#                       Quality loss for Qwen's official FP8 is negligible.
#
#   --max-model-len     Default is the model's 262k native context. Shrinking
#                       to 16k is the minimal boot/memory hygiene cap for the
#                       20 GB card and still covers every Tier-A/Tier-B prompt.
#                       It is not counted as a TTFT tuning knob.
#
# Defaults deliberately left alone:
#
#   prefix caching      Default-on in vLLM V1. This is the warm-path superpower:
#                       cached prefix tokens skip prefill entirely.
#
#   chunked prefill     Default budget is already large enough for the Tier-A
#                       homogeneous matrix. The cp-512 ablation showed smaller
#                       chunks made this workload worse, not better.
#
#   CUDA graphs         Default graph capture is valuable; forcing extra
#                       `torch.compile` via `-O3` was not.
# ============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct-2507-FP8"

vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 16384

#   python bench/benchmark_ttft.py --label vllm-fp8-only --url http://localhost:8000 \
#       --tokenizer Qwen/Qwen3-4B-Instruct-2507-FP8
#
# Extra credit datapoint (INT4, for the FP8-vs-INT4 prefill study):
#   vllm serve Qwen/Qwen3-4B-AWQ ...   (or quantize yourself with llm-compressor)
#   --label vllm-awq
# Expect: decode throughput up, TTFT flat-to-worse vs BF16. Explain why.
