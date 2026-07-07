#!/usr/bin/env bash
# ============================================================================
# 00_setup.sh — setup on a fresh RunPod pod (works for both the RTX 4000 Ada
# dev pod and the 48GB Tier-C pod).
# Recommended image: runpod/pytorch (CUDA 12.x). Run inside tmux, not a bare
# Jupyter terminal — Jupyter terminals die and take your server with them.
# ============================================================================
set -euo pipefail

nvidia-smi   # 4000 Ada: 20GB SM8.9 | L40S / RTX 6000 Ada: 48GB SM8.9 (FP8 OK)

apt-get update -qq && apt-get install -y -qq tmux git > /dev/null || true

pip install -U pip uv
# Qwen3.5/3.6 hybrid (Gated DeltaNet) support needs vLLM >= 0.19.
# PINNED at 0.19.1: it is the LAST release whose PyPI wheel links CUDA 12
# (pins torch 2.10; vLLM >= 0.20 pins torch 2.11 whose wheels are CUDA-13
# builds needing driver >= r580 — RunPod pods run driver 550 = CUDA 12.4).
# --torch-backend=cu128 pins the matching torch build explicitly.
uv pip install --system --break-system-packages "vllm==0.19.1" --torch-backend=cu128
# numpy<2.3: numba 0.61.x rejects numpy 2.3+; numba/mistral-common are vLLM deps.
# accelerate: required by the naive HF baseline's device_map="cuda".
pip install -U aiohttp transformers pandas matplotlib "numpy<2.3" tabulate accelerate fastapi uvicorn
# optional cross-check engine:  pip install -U "sglang[all]"

# HF cache on the persistent volume so weights survive pod restarts
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$HF_HOME"; grep -q HF_HOME ~/.bashrc || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

echo "== pre-download models for THIS tier =="
python - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
tier_a = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507-FP8",
          "Qwen/Qwen3.5-4B"]                      # dev pod (20GB)
tier_c = ["Qwen/Qwen3.6-27B-FP8"]                 # 48GB pod only (~27GB!)
repos = tier_c if "tier-c" in sys.argv else tier_a
for r in repos:
    print("downloading", r); snapshot_download(r)
PY

echo "== record versions for the report =="
python -c "import torch, vllm, transformers; print('torch', torch.__version__, '| vllm', vllm.__version__, '| transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
