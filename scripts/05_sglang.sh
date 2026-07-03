#!/usr/bin/env bash
# ============================================================================
# 05_sglang.sh — OPTIONAL third engine: SGLang.
# Why include it: SGLang's RadixAttention does tree-structured prefix caching;
# on warm-cache workloads it's often the strongest TTFT engine. A 3-engine
# comparison (TGI vs vLLM vs SGLang) reads as thoroughness, not padding.
#   pip install "sglang[all]"
# ============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct-2507"

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --context-length 8192

# Benchmark (SGLang serves the OpenAI API too):
#   python bench/benchmark_ttft.py --label sglang --url http://localhost:8000
