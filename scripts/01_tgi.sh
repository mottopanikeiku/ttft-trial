#!/usr/bin/env bash
# ============================================================================
# 01_tgi.sh — VANILLA BASELINE #1: HF Text Generation Inference, default flags.
#
# IMPORTANT — TGI ships as a Docker image, and most RunPod GPU pods do NOT
# allow docker-in-docker. Two clean options:
#
#   Option A (recommended on RunPod): create a SECOND pod whose *container
#   image* is TGI itself:
#       Image:   ghcr.io/huggingface/text-generation-inference:latest
#       Args:    --model-id Qwen/Qwen3-4B-Instruct-2507
#                --max-input-tokens 8192 --max-total-tokens 8704
#       Expose:  port 80 (map to 8080)
#   Then run the benchmark client from your main pod (or laptop) against its
#   proxy URL. Same GPU type = fair comparison; note this in the report.
#
#   Option B (any machine where you control docker), the command below.
# ============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3-4B-Instruct-2507"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

docker run --rm --gpus all --shm-size 1g -p 8080:80 \
  -v "$HF_CACHE":/data \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id "$MODEL" \
  --max-input-tokens 8192 \
  --max-total-tokens 8704
  # ^ deliberately NO other tuning flags: this is the "vanilla" datapoint.

# Benchmark it (from another shell):
#   python bench/benchmark_ttft.py --label tgi-vanilla \
#       --url http://localhost:8080 --api chat
# TGI exposes an OpenAI-compatible /v1/chat/completions; use --api chat here.
