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
import math
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


def decode_exact_prompt(tokenizer, token_ids, n_tokens, padding_ids):
    """Decode token IDs to text that re-tokenizes to exactly ``n_tokens``."""
    text = tokenizer.decode(token_ids)
    for _ in range(8):
        actual_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(actual_ids) == n_tokens:
            return text
        if len(actual_ids) > n_tokens:
            corrected_ids = actual_ids[:n_tokens]
        else:
            missing = n_tokens - len(actual_ids)
            repeats = (missing + len(padding_ids) - 1) // len(padding_ids)
            corrected_ids = actual_ids + (padding_ids * repeats)[:missing]
        text = tokenizer.decode(corrected_ids)
    actual = len(tokenizer(text, add_special_tokens=False).input_ids)
    raise ValueError(
        f"could not construct exact-length prompt: requested {n_tokens}, got {actual}"
    )


def build_prompts(tokenizer, n_tokens: int, count: int, cache_mode: str):
    """Return `count` prompt strings, each exactly `n_tokens` tokens long."""
    nonce_tokens = 8  # tokens reserved for the unique nonce
    # scale the filler pool with the request; FILLER is ~35-40 tokens, so
    # a fixed *200 (~7.5k tokens) silently truncated 8k/16k prompts
    reps = max(200, n_tokens // 20)
    base_ids = tokenizer(FILLER * reps, add_special_tokens=False).input_ids
    if len(base_ids) < n_tokens:
        raise ValueError(f"filler pool too small: {len(base_ids)} < {n_tokens}")

    prompts = []
    for _ in range(count):
        nonce = "".join(random.choices(string.ascii_lowercase, k=24))
        nonce_ids = tokenizer(f"[{nonce}] ", add_special_tokens=False).input_ids[:nonce_tokens]
        body = base_ids[: n_tokens - len(nonce_ids)]
        if cache_mode == "cold":
            ids = nonce_ids + body          # unique START -> guaranteed cache miss
        else:  # warm
            ids = body + nonce_ids          # shared prefix, unique TAIL
        prompts.append(decode_exact_prompt(tokenizer, ids, n_tokens, base_ids))
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
    malformed = 0
    non_content = 0

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
                    malformed += 1
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    non_content += 1
                    continue
                piece = (choices[0].get("text")
                         or (choices[0].get("delta") or {}).get("content")
                         or "")
                if piece:
                    n_chunks += 1
                    t_end = now
                    if ttft is None:
                        ttft = now - t0
                else:
                    non_content += 1

    if ttft is None or n_chunks == 0:
        raise RuntimeError(
            "stream ended without generated text "
            f"(malformed_data={malformed}, non_content_chunks={non_content})"
        )
    return ttft, t_end - t0, n_chunks


# --------------------------------------------------------------------------- #
# Worker pool for a given concurrency level
# --------------------------------------------------------------------------- #
async def run_config(session, args, model, prompts, concurrency, results, meta):
    queue = asyncio.Queue()
    for i, p in enumerate(prompts):
        queue.put_nowait((i, p))

    failures = []

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
                msg = f"request {idx} failed: {e}"
                failures.append(msg)
                print(f"  {msg}", file=sys.stderr)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return len(prompts) - len(failures), failures


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
    ap.add_argument("--min-success-rate", type=float, default=0.90,
                    help="minimum measured request success fraction required per cell")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if not 0 < args.min_success_rate <= 1:
        raise SystemExit("--min-success-rate must be in (0, 1]")

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
        bad_cells = []
        for cache_mode in args.cache_modes:
            for n_tok in args.prompt_tokens:
                for conc in args.concurrency:
                    meta = {"label": args.label, "api": args.api,
                            "cache_mode": cache_mode, "prompt_tokens": n_tok}
                    min_successes = math.ceil(args.num_requests * args.min_success_rate)

                    # warmup (also primes the shared prefix for warm mode). Use
                    # the measured concurrency so c=16 does not include first-use
                    # scheduler/CUDA-graph behavior in the measured rows.
                    warm_count = max(args.warmup, conc) if args.warmup else 0
                    if warm_count:
                        warm_prompts = build_prompts(tokenizer, n_tok, warm_count, cache_mode)
                        warm_ok, warm_failures = await run_config(
                            session, args, model, warm_prompts, conc, [], meta)
                        if warm_ok < warm_count:
                            bad_cells.append(
                                f"{cache_mode}/{n_tok}/c{conc} warmup: "
                                f"{warm_ok}/{warm_count} successful"
                            )

                    # measured
                    prompts = build_prompts(tokenizer, n_tok, args.num_requests, cache_mode)
                    t0 = time.perf_counter()
                    before = len(results)
                    _, failures = await run_config(session, args, model, prompts, conc, results, meta)
                    dur = time.perf_counter() - t0
                    cell_rows = results[before:]
                    ttfts = [r["ttft_s"] for r in cell_rows if r["ttft_s"] is not None]
                    status = "OK" if len(ttfts) >= min_successes else "FAILED"
                    if status == "FAILED":
                        bad_cells.append(
                            f"{cache_mode}/{n_tok}/c{conc}: "
                            f"{len(ttfts)}/{args.num_requests} successful "
                            f"(< required {min_successes}); failures={len(failures)}"
                        )
                    print(f"  {cache_mode:4s} | {n_tok:5d} tok | c={conc:2d} | "
                          f"p50={pct(ttfts,50)*1000:7.1f}ms  p90={pct(ttfts,90)*1000:7.1f}ms  "
                          f"p99={pct(ttfts,99)*1000:7.1f}ms  "
                          f"({len(ttfts)}/{args.num_requests} ok, {dur:.1f}s, {status})")

    if bad_cells:
        print("\nFATAL: insufficient successful samples; not writing CSV.", file=sys.stderr)
        for cell in bad_cells:
            print(f"  - {cell}", file=sys.stderr)
        raise SystemExit(1)
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
