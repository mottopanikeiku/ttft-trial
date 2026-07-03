#!/usr/bin/env python3
"""
prefix_cache_sweep.py — the "novel" experiment.

Idea: with prefix caching, only UNCACHED prompt tokens pay prefill cost. So
client-observed TTFT should follow a simple analytical model:

        TTFT ≈ a + b * uncached_tokens

where  a = fixed per-request overhead (HTTP + tokenize + schedule + 1 decode step)
       1/b = effective prefill throughput in tokens/second.

Protocol (total prompt fixed at --total-tokens, e.g. 4096):
  1. For cached fraction f in {0, .25, .50, .75, .95}:
     - build a prompt = [shared prefix of f*T tokens] + [unique tail of (1-f)*T]
     - PRIME the cache once with a request sharing that exact prefix
     - measure TTFT over N requests (each with a fresh unique tail)
  2. Least-squares fit a, b over (uncached_tokens, ttft) points.
  3. VALIDATE: predict TTFT for a held-out fraction (e.g. f=0.60) and report
     prediction error. A model that predicts beats a table that describes.

Deliverables printed at the end are copy-pasteable into the report.

Run against the optimized vLLM server:
    python bench/prefix_cache_sweep.py --url http://localhost:8000
"""

import argparse
import asyncio
import random
import string
import time

import aiohttp
import numpy as np

from benchmark_ttft import FILLER, one_request, pct


def make_prompt(tokenizer, shared_ids, uncached_tokens):
    nonce = "".join(random.choices(string.ascii_lowercase, k=24))
    nonce_ids = tokenizer(f"[{nonce}] ", add_special_tokens=False).input_ids[:8]
    body = tokenizer(FILLER * 200, add_special_tokens=False).input_ids
    tail = nonce_ids + body[: max(0, uncached_tokens - len(nonce_ids))]
    return tokenizer.decode(shared_ids + tail)


async def measure(session, url, model, tokenizer, total, frac, n, max_tokens):
    shared_len = int(total * frac)
    base = tokenizer(FILLER * 200, add_special_tokens=False).input_ids
    shared_ids = base[:shared_len]
    # prime the cache with the shared prefix (one throwaway request)
    await one_request(session, url, "completions", model,
                      make_prompt(tokenizer, shared_ids, total - shared_len),
                      max_tokens)
    ttfts = []
    for _ in range(n):
        ttft, _, _ = await one_request(
            session, url, "completions", model,
            make_prompt(tokenizer, shared_ids, total - shared_len), max_tokens)
        if ttft is not None:
            ttfts.append(ttft)
    return total - shared_len, ttfts


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--total-tokens", type=int, default=4096)
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.0, 0.25, 0.50, 0.75, 0.95])
    ap.add_argument("--holdout", type=float, default=0.60)
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = args.model
        if model is None:
            async with session.get(f"{args.url}/v1/models") as r:
                model = (await r.json())["data"][0]["id"]

        xs, ys = [], []
        print(f"total prompt = {args.total_tokens} tokens, model = {model}\n")
        print(f"{'cached %':>9} | {'uncached tok':>12} | {'p50 TTFT':>10} | {'p90':>9}")
        for f in args.fractions:
            uncached, ttfts = await measure(session, args.url, model, tokenizer,
                                            args.total_tokens, f,
                                            args.num_requests, args.max_tokens)
            p50 = pct(ttfts, 50)
            print(f"{f*100:8.0f}% | {uncached:12d} | {p50*1000:8.1f}ms | {pct(ttfts,90)*1000:7.1f}ms")
            xs.extend([uncached] * len(ttfts))
            ys.extend(ttfts)

        # ---- fit TTFT = a + b * uncached_tokens ----
        A = np.vstack([np.ones(len(xs)), np.array(xs)]).T
        (a, b), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
        r2 = 1 - np.sum((A @ [a, b] - ys) ** 2) / np.sum((ys - np.mean(ys)) ** 2)

        print("\n=== fitted model:  TTFT = a + b * uncached_tokens ===")
        print(f"  a (fixed overhead)          = {a*1000:.1f} ms")
        print(f"  b                           = {b*1e6:.2f} us/token")
        print(f"  1/b (eff. prefill thruput)  = {1/b:,.0f} tokens/s")
        print(f"  R^2                         = {r2:.4f}")

        # ---- holdout validation ----
        uncached, ttfts = await measure(session, args.url, model, tokenizer,
                                        args.total_tokens, args.holdout,
                                        args.num_requests, args.max_tokens)
        pred = a + b * uncached
        meas = pct(ttfts, 50)
        err = abs(pred - meas) / meas * 100
        print(f"\n=== holdout @ cached={args.holdout*100:.0f}% "
              f"(uncached={uncached} tok) ===")
        print(f"  predicted p50 = {pred*1000:.1f} ms | measured p50 = {meas*1000:.1f} ms "
              f"| error = {err:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
