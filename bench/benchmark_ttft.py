#!/usr/bin/env python3
"""
benchmark_ttft.py — async streaming TTFT benchmark against any OpenAI-compatible
server (vLLM, TGI, SGLang, the naive baseline).

Definition used (client-side, the honest one):
    TTFT = time from the instant the HTTP request is sent
           until the first SSE chunk containing generated text arrives.

Design choices worth defending in the report:
  * Prompts are built with the model's own tokenizer to EXACT token counts.
  * cache_mode=cold  -> a unique random nonce is placed at the *start* of every
    prompt, guaranteeing a prefix-cache MISS (pure prefill measurement).
  * cache_mode=warm  -> all requests share the same long prefix, only the last
    ~32 tokens differ (realistic system-prompt / chat-history scenario).
  * temperature=0, max_tokens=16: we measure latency, not generation quality.
  * Warmup requests are discarded (CUDA graph capture, allocator warmup, JIT).
  * Percentiles over >=24 samples per configuration, raw rows kept in CSV.

Usage:
  python bench/benchmark_ttft.py --label vllm-vanilla --url http://localhost:8000
  python bench/benchmark_ttft.py --label tgi-vanilla --url http://localhost:8080 --api chat
"""

import argparse
import asyncio
import csv
import json
import os
import random
import string
import sys
import time

import aiohttp


# --------------------------------------------------------------------------- #
# Prompt construction (exact token counts via the model tokenizer)
# --------------------------------------------------------------------------- #
FILLER = (
    "The history of computing hardware covers developments from simple devices "
    "to aid calculation, to modern day computers. Before the twentieth century, "
    "most calculations were done by humans using mechanical aids. "
)


def build_prompts(tokenizer, n_tokens: int, count: int, cache_mode: str):
    """Return `count` prompt strings, each exactly `n_tokens` tokens long."""
    nonce_tokens = 8  # tokens reserved for the unique nonce
    base_ids = tokenizer(FILLER * 200, add_special_tokens=False).input_ids

    prompts = []
    for _ in range(count):
        nonce = "".join(random.choices(string.ascii_lowercase, k=24))
        nonce_ids = tokenizer(f"[{nonce}] ", add_special_tokens=False).input_ids[:nonce_tokens]
        body = base_ids[: n_tokens - len(nonce_ids)]
        if cache_mode == "cold":
            ids = nonce_ids + body          # unique START -> guaranteed cache miss
        else:  # warm
            ids = body + nonce_ids          # shared prefix, unique TAIL
        prompts.append(tokenizer.decode(ids))
    return prompts


# --------------------------------------------------------------------------- #
# Single streaming request; returns (ttft_s, e2e_s, n_chunks)
# --------------------------------------------------------------------------- #
async def one_request(session, url, api, model, prompt, max_tokens):
    if api == "completions":
        endpoint = f"{url}/v1/completions"
        payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                   "temperature": 0, "stream": True}
    else:
        endpoint = f"{url}/v1/chat/completions"
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0, "stream": True}

    ttft = None
    n_chunks = 0
    buf = b""
    t0 = time.perf_counter()
    t_end = t0

    async with session.post(endpoint, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        async for raw in resp.content.iter_any():
            now = time.perf_counter()
            buf += raw
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("text")
                         or (choices[0].get("delta") or {}).get("content")
                         or "")
                if piece:
                    n_chunks += 1
                    t_end = now
                    if ttft is None:
                        ttft = now - t0
    return ttft, t_end - t0, n_chunks


# --------------------------------------------------------------------------- #
# Worker pool for a given concurrency level
# --------------------------------------------------------------------------- #
async def run_config(session, args, model, prompts, concurrency, results, meta):
    queue = asyncio.Queue()
    for i, p in enumerate(prompts):
        queue.put_nowait((i, p))

    async def worker():
        while True:
            try:
                idx, prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                ttft, e2e, n = await one_request(
                    session, args.url, args.api, model, prompt, args.max_tokens)
                results.append({**meta, "concurrency": concurrency,
                                "request_idx": idx, "ttft_s": ttft,
                                "e2e_s": e2e, "out_chunks": n})
            except Exception as e:
                print(f"  request {idx} failed: {e}", file=sys.stderr)

    await asyncio.gather(*[worker() for _ in range(concurrency)])


def pct(vals, p):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = min(len(vals) - 1, max(0, int(round(p / 100 * (len(vals) - 1)))))
    return vals[k]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", required=True, help="config name, e.g. vllm-vanilla")
    ap.add_argument("--model", default=None, help="model id; auto-detected from /v1/models if omitted")
    ap.add_argument("--api", choices=["completions", "chat"], default="completions",
                    help="use 'chat' for TGI and the naive baseline")
    ap.add_argument("--prompt-tokens", type=int, nargs="+",
                    default=[128, 512, 1024, 2048, 4096])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--cache-modes", nargs="+", default=["cold", "warm"])
    ap.add_argument("--num-requests", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = args.model
        if model is None:
            async with session.get(f"{args.url}/v1/models") as r:
                model = (await r.json())["data"][0]["id"]
        print(f"server={args.url} model={model} label={args.label}")

        results = []
        for cache_mode in args.cache_modes:
            for n_tok in args.prompt_tokens:
                for conc in args.concurrency:
                    meta = {"label": args.label, "api": args.api,
                            "cache_mode": cache_mode, "prompt_tokens": n_tok}
                    # warmup (also primes the shared prefix for warm mode)
                    warm_prompts = build_prompts(tokenizer, n_tok, args.warmup, cache_mode)
                    await run_config(session, args, model, warm_prompts, min(conc, 4), [], meta)
                    # measured
                    prompts = build_prompts(tokenizer, n_tok, args.num_requests, cache_mode)
                    t0 = time.perf_counter()
                    await run_config(session, args, model, prompts, conc, results, meta)
                    dur = time.perf_counter() - t0
                    ttfts = [r["ttft_s"] for r in results
                             if r["cache_mode"] == cache_mode
                             and r["prompt_tokens"] == n_tok
                             and r["concurrency"] == conc
                             and r["ttft_s"] is not None]
                    print(f"  {cache_mode:4s} | {n_tok:5d} tok | c={conc:2d} | "
                          f"p50={pct(ttfts,50)*1000:7.1f}ms  p90={pct(ttfts,90)*1000:7.1f}ms  "
                          f"p99={pct(ttfts,99)*1000:7.1f}ms  ({len(ttfts)} ok, {dur:.1f}s)")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.label}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "api", "cache_mode", "prompt_tokens",
                                          "concurrency", "request_idx", "ttft_s",
                                          "e2e_s", "out_chunks"])
        w.writeheader()
        w.writerows(results)
    print(f"wrote {len(results)} rows -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
