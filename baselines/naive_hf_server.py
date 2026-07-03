#!/usr/bin/env python3
"""
naive_hf_server.py — BASELINE 0: what deployment looks like with zero serving
infrastructure. Plain `transformers` generate(), one request at a time, no
paged KV cache, no batching, no CUDA graphs. Exposes a minimal OpenAI-style
streaming /v1/chat/completions so the same benchmark client works.

This is the number that makes your final speedup chart dramatic — and it is a
FAIR baseline: it is genuinely how many people first deploy a model.

    pip install fastapi uvicorn
    python baselines/naive_hf_server.py            # port 8000
    python bench/benchmark_ttft.py --label naive-hf --api chat \
        --prompt-tokens 128 512 1024 2048 --concurrency 1 --num-requests 12
"""

import asyncio
import json
import threading
import time
import uuid

import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

print("loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="sdpa",   # honest default; not even FA2
)
model.eval()
print("ready.")

app = FastAPI()
gpu_lock = threading.Lock()  # serialize: naive deployments have no batching


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_ID}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    max_tokens = int(body.get("max_tokens", 16))
    text = tok.apply_chat_template(body["messages"], tokenize=False,
                                   add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def generate():
        with gpu_lock, torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_tokens,
                           do_sample=False, streamer=streamer)

    threading.Thread(target=generate, daemon=True).start()
    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def sse():
        loop = asyncio.get_event_loop()
        it = iter(streamer)
        while True:
            piece = await loop.run_in_executor(None, lambda: next(it, None))
            if piece is None:
                break
            chunk = {"id": rid, "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": {"content": piece},
                                  "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    if body.get("stream"):
        return StreamingResponse(sse(), media_type="text/event-stream")
    return JSONResponse({"error": "use stream=true"}, status_code=400)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
