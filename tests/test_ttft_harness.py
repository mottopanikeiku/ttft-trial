import contextlib
import json
import pathlib
import sys
import tempfile
import types
import unittest
from collections import deque
from unittest import mock

import aiohttp
from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import benchmark_ttft
import prefix_cache_sweep


class _Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        tokens = text.split()
        if not tokens and text:
            tokens = [text]
        return _Encoding(list(range(len(tokens))))

    def decode(self, ids):
        return " ".join(f"tok{token_id}" for token_id in ids)


class _FirstDecodeDropsToken(_StubTokenizer):
    """Models a tokenizer whose first decode/encode round trip loses one ID."""

    def __init__(self):
        self.decode_calls = 0

    def decode(self, ids):
        self.decode_calls += 1
        if self.decode_calls == 1:
            ids = ids[:-1]
        return super().decode(ids)


class _StubAutoTokenizer:
    @staticmethod
    def from_pretrained(name):
        del name
        return _StubTokenizer()


@contextlib.contextmanager
def _stub_transformers():
    module = types.ModuleType("transformers")
    module.AutoTokenizer = _StubAutoTokenizer
    with mock.patch.dict(sys.modules, {"transformers": module}):
        yield


class _OpenAIStubServer:
    def __init__(self, outcomes):
        self._outcomes = deque(outcomes)
        self.requests = []
        self._runner = None
        self.url = None

    async def __aenter__(self):
        app = web.Application()
        app.router.add_get("/v1/models", self._models)
        app.router.add_post("/v1/completions", self._completion)
        app.router.add_post("/v1/chat/completions", self._completion)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        await self._runner.cleanup()

    async def _models(self, request):
        del request
        return web.json_response({"data": [{"id": "stub-model"}]})

    async def _completion(self, request):
        self.requests.append(await request.json())
        outcome = self._outcomes.popleft() if self._outcomes else "valid"
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        if outcome == "valid":
            await self._write_sse(response, {"choices": [{"text": "hello"}]})
            await self._write_sse(response, {"choices": [{"text": " world"}]})
            await response.write(b"data: [DONE]\n\n")
        elif outcome == "chat_valid":
            await self._write_sse(response, {"choices": [{"delta": {"content": "hello"}}]})
            await response.write(b"data: [DONE]\n\n")
        elif outcome == "done":
            await response.write(b"data: [DONE]\n\n")
        else:
            raise AssertionError(f"unknown SSE outcome: {outcome}")
        await response.write_eof()
        return response

    @staticmethod
    async def _write_sse(response, payload):
        await response.write(b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n")


class PromptConstructionTests(unittest.TestCase):
    def test_build_prompts_corrects_decode_round_trip_length_changes(self):
        tokenizer = _FirstDecodeDropsToken()

        prompts = benchmark_ttft.build_prompts(tokenizer, 128, 2, "warm")

        self.assertEqual(
            [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts],
            [128, 128],
        )


class TTFTStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_sse_stream_returns_ttft_and_counts_content_chunks(self):
        async with _OpenAIStubServer(["valid"]) as server:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                ttft, e2e, chunks = await benchmark_ttft.one_request(
                    session, server.url, "completions", "stub-model", "prompt", 2
                )

        self.assertEqual(chunks, 2)
        self.assertGreaterEqual(ttft, 0.0)
        self.assertGreaterEqual(e2e, ttft)

    async def test_done_only_stream_raises_instead_of_reporting_missing_ttft(self):
        async with _OpenAIStubServer(["done"]) as server:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                with self.assertRaisesRegex(RuntimeError, "stream ended without generated text"):
                    await benchmark_ttft.one_request(
                        session, server.url, "completions", "stub-model", "prompt", 2
                    )


class HarnessFailureGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_benchmark_main_exits_nonzero_and_writes_no_csv_below_min_success_rate(self):
        with tempfile.TemporaryDirectory() as out_dir:
            async with _OpenAIStubServer(["valid", "done"]) as server:
                argv = [
                    "benchmark_ttft.py",
                    "--url", server.url,
                    "--label", "gate-failure",
                    "--model", "stub-model",
                    "--prompt-tokens", "16",
                    "--concurrency", "1",
                    "--cache-modes", "cold",
                    "--num-requests", "2",
                    "--warmup", "0",
                    "--max-tokens", "2",
                    "--min-success-rate", "0.75",
                    "--tokenizer", "stub-tokenizer",
                    "--out", out_dir,
                ]
                with _stub_transformers(), mock.patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit) as raised:
                        await benchmark_ttft.main()

            self.assertEqual(raised.exception.code, 1)
            self.assertFalse((pathlib.Path(out_dir) / "gate-failure.csv").exists())

    async def test_prefix_sweep_rejects_insufficient_successes_before_fitting(self):
        async with _OpenAIStubServer(["valid", "valid", "done"]) as server:
            argv = [
                "prefix_cache_sweep.py",
                "--url", server.url,
                "--model", "stub-model",
                "--tokenizer", "stub-tokenizer",
                "--total-tokens", "16",
                "--fractions", "0.0",
                "--num-requests", "2",
                "--max-tokens", "2",
                "--min-success-rate", "0.75",
            ]
            with (
                _stub_transformers(),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    prefix_cache_sweep.np.linalg,
                    "lstsq",
                    side_effect=AssertionError("fit should not run"),
                ) as fit,
            ):
                with self.assertRaisesRegex(RuntimeError, r"only 1/2 successful"):
                    await prefix_cache_sweep.main()

        fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
