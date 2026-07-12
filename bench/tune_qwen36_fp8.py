#!/usr/bin/env python3
"""Tune Qwen3.6-27B block-FP8 prefill GEMMs for the local Ada GPU.

vLLM 0.19.1 has no RTX 6000 Ada configs for the five Qwen3.6 matrix shapes,
so it falls back to one generic Triton launch configuration. This bounded
search targets M=4096 (the assignment's headline prompt length), preserves the
vLLM default for small M/decode, and emits the JSON format vLLM consumes.
"""

import argparse
import json
import os
import shutil
import time

import torch
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton


SHAPES = [
    (16384, 5120),
    (5120, 6144),
    (34816, 5120),
    (5120, 17408),
    (14336, 5120),
]
BLOCK_SIZE = (128, 128)
TARGET_M = 4096
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}


def candidates(full=False):
    block_ms = (16, 32, 64, 128, 256) if full else (32, 64, 128)
    block_ns = (32, 64, 128, 256) if full else (128, 256)
    group_sizes = (1, 16, 32, 64) if full else (8, 16, 32)
    stage_counts = (2, 3, 4, 5) if full else (2, 3)
    for block_m in block_ms:
        for block_n in block_ns:
            for block_k in (64, 128):
                for group_size in group_sizes:
                    for num_warps in (4, 8):
                        for num_stages in stage_counts:
                            yield {
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": group_size,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }


def allocate_inputs(n, k):
    fp8 = torch.float8_e4m3fn
    a = torch.empty((TARGET_M, k), dtype=fp8, device="cuda")
    b = torch.empty((n, k), dtype=fp8, device="cuda")
    a_scale = torch.ones(
        (TARGET_M, triton.cdiv(k, BLOCK_SIZE[1])), dtype=torch.float32, device="cuda"
    )
    b_scale = torch.ones(
        (triton.cdiv(n, BLOCK_SIZE[0]), triton.cdiv(k, BLOCK_SIZE[1])),
        dtype=torch.float32,
        device="cuda",
    )
    out = torch.empty((TARGET_M, n), dtype=torch.bfloat16, device="cuda")
    return a, b, a_scale, b_scale, out


def launch(inputs, config):
    a, b, a_scale, b_scale, out = inputs
    m, k = a.shape
    n = b.shape[0]

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_SIZE_M"])
            * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        a,
        b,
        out,
        a_scale,
        b_scale,
        m,
        n,
        k,
        BLOCK_SIZE[0],
        BLOCK_SIZE[1],
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        out.stride(0),
        out.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(1),
        b_scale.stride(0),
        **config,
    )


def benchmark(inputs, config, warmup=2, iterations=5):
    for _ in range(warmup):
        launch(inputs, config)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(iterations):
        start.record()
        launch(inputs, config)
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return sum(elapsed) / len(elapsed)


def config_filename(n, k):
    device = current_platform.get_device_name().replace(" ", "_")
    return (
        f"N={n},K={k},device_name={device},dtype=fp8_w8a8,"
        f"block_shape=[{BLOCK_SIZE[0]},{BLOCK_SIZE[1]}].json"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results_tier_c/fp8_configs")
    parser.add_argument(
        "--install",
        action="store_true",
        help="also install generated configs into the active vLLM package",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="search the complete official vLLM configuration space",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    os.makedirs(args.out, exist_ok=True)
    package_configs = os.path.join(os.path.dirname(fp8_utils.__file__), "configs")
    search = list(candidates(args.full))
    print(f"device={current_platform.get_device_name()} candidates={len(search)}")
    started = time.time()

    for n, k in SHAPES:
        print(f"\nshape M={TARGET_M} N={n} K={k}")
        inputs = allocate_inputs(n, k)
        default_ms = benchmark(inputs, DEFAULT_CONFIG)
        best_ms = default_ms
        best = DEFAULT_CONFIG
        for index, config in enumerate(search, 1):
            try:
                latency = benchmark(inputs, config)
            except triton.runtime.autotuner.OutOfResources:
                continue
            if latency < best_ms:
                best_ms = latency
                best = config
            if index % 24 == 0:
                print(
                    f"  {index:3d}/{len(search)} best={best_ms:.3f}ms "
                    f"({default_ms / best_ms:.2f}x)"
                )

        name = config_filename(n, k)
        path = os.path.join(args.out, name)
        with open(path, "w") as handle:
            json.dump({"1": DEFAULT_CONFIG, str(TARGET_M): best}, handle, indent=2)
            handle.write("\n")
        if args.install:
            shutil.copy2(path, os.path.join(package_configs, name))
        print(
            f"  wrote {path}; default={default_ms:.3f}ms best={best_ms:.3f}ms "
            f"speedup={default_ms / best_ms:.2f}x config={best}"
        )
        del inputs
        torch.cuda.empty_cache()

    print(f"\ncompleted in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
