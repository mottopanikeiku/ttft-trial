#!/usr/bin/env python3
"""Tune Qwen3.6-27B block-FP8 GEMMs for the local Ada GPU.

vLLM 0.19.1 has no RTX 6000 Ada configs for the five Qwen3.6 matrix shapes,
so it falls back to one generic Triton launch configuration. This tuner emits
the irregular M-grid consumed by vLLM's nearest-M dispatcher and rotates
multiple weight tensors so the search does not benchmark an L2-resident weight
matrix that cannot occur while the model streams distinct layers.
"""

import argparse
import json
import os
import shutil
import statistics
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
M_VALUES = (1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192)
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
    group_sizes = (1, 4, 8, 64) if full else (1, 4, 8, 16, 32, 64)
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


def allocate_inputs(m, n, k, weight_copies):
    fp8 = torch.float8_e4m3fn
    a = torch.empty((m, k), dtype=fp8, device="cuda")
    weights = [
        torch.empty((n, k), dtype=fp8, device="cuda") for _ in range(weight_copies)
    ]
    a_scale = torch.ones(
        (m, triton.cdiv(k, BLOCK_SIZE[1])), dtype=torch.float32, device="cuda"
    )
    weight_scales = [
        torch.ones(
            (triton.cdiv(n, BLOCK_SIZE[0]), triton.cdiv(k, BLOCK_SIZE[1])),
            dtype=torch.float32,
            device="cuda",
        )
        for _ in range(weight_copies)
    ]
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    return a, weights, a_scale, weight_scales, out


def launch(inputs, config, copy_index=0):
    a, weights, a_scale, weight_scales, out = inputs
    ring_index = copy_index % len(weights)
    b = weights[ring_index]
    b_scale = weight_scales[ring_index]
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
    for index in range(warmup):
        launch(inputs, config, index)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for index in range(iterations):
        start.record()
        launch(inputs, config, index)
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return statistics.median(elapsed)


def validate(inputs, config):
    a, weights, a_scale, weight_scales, out = inputs
    a.fill_(0.25)
    weights[0].fill_(0.25)
    a_scale.uniform_(0.5, 1.5)
    weight_scales[0].uniform_(0.5, 1.5)

    launch(inputs, DEFAULT_CONFIG)
    torch.cuda.synchronize()
    expected = out.clone()
    launch(inputs, config)
    torch.cuda.synchronize()
    max_abs = (out - expected).abs().max().item()
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.5)
    return max_abs


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
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=list(M_VALUES),
        help="matrix row counts to tune; vLLM selects the nearest emitted key",
    )
    parser.add_argument(
        "--weight-copies",
        type=int,
        default=4,
        help="weight/scale ring size used to defeat unrealistic L2 residency",
    )
    parser.add_argument("--coarse-iterations", type=int, default=5)
    parser.add_argument("--final-iterations", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    if (
        any(m <= 0 for m in args.m_values)
        or args.weight_copies < 2
        or args.coarse_iterations <= 0
        or args.final_iterations <= 0
        or args.top_k <= 0
    ):
        raise SystemExit("M values and iteration counts must be positive; weight copies >= 2")

    m_values = sorted(set(args.m_values))
    os.makedirs(args.out, exist_ok=True)
    package_configs = os.path.join(os.path.dirname(fp8_utils.__file__), "configs")
    search = list(candidates(args.full))
    print(
        f"device={current_platform.get_device_name()} candidates={len(search)} "
        f"M={m_values} weight_copies={args.weight_copies}"
    )
    started = time.time()

    for n, k in SHAPES:
        tuned = {}
        for m in m_values:
            print(f"\nshape M={m} N={n} K={k}")
            inputs = allocate_inputs(m, n, k, args.weight_copies)
            default_coarse_ms = benchmark(
                inputs, DEFAULT_CONFIG, iterations=args.coarse_iterations
            )
            ranked = [(default_coarse_ms, DEFAULT_CONFIG)]
            for index, config in enumerate(search, 1):
                try:
                    latency = benchmark(
                        inputs, config, iterations=args.coarse_iterations
                    )
                except triton.runtime.autotuner.OutOfResources:
                    continue
                ranked.append((latency, config))
                if index % 48 == 0:
                    coarse_best = min(value for value, _ in ranked)
                    print(
                        f"  {index:4d}/{len(search)} coarse={coarse_best:.3f}ms "
                        f"({default_coarse_ms / coarse_best:.2f}x)"
                    )

            shortlist = [
                config
                for _, config in sorted(ranked, key=lambda item: item[0])[: args.top_k]
            ]
            if DEFAULT_CONFIG not in shortlist:
                shortlist.append(DEFAULT_CONFIG)
            default_ms = benchmark(
                inputs, DEFAULT_CONFIG, iterations=args.final_iterations
            )
            best_ms = default_ms
            best = DEFAULT_CONFIG
            for config in shortlist:
                if config == DEFAULT_CONFIG:
                    continue
                latency = benchmark(
                    inputs, config, iterations=args.final_iterations
                )
                if latency < best_ms:
                    best_ms = latency
                    best = config

            max_abs = validate(inputs, best)
            tuned[str(m)] = best
            print(
                f"  default={default_ms:.3f}ms best={best_ms:.3f}ms "
                f"speedup={default_ms / best_ms:.2f}x max_abs={max_abs:.4f} "
                f"config={best}"
            )
            del inputs
            torch.cuda.empty_cache()

        name = config_filename(n, k)
        path = os.path.join(args.out, name)
        with open(path, "w") as handle:
            json.dump(tuned, handle, indent=2)
            handle.write("\n")
        if args.install:
            shutil.copy2(path, os.path.join(package_configs, name))
        print(f"  wrote {path}")

    print(f"\ncompleted in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
