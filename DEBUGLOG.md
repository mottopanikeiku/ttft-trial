# DEBUGLOG.md — Current Actionable Failure Ledger

This is not a chronological transcript. Obsolete exploratory records were removed during the 2026-07-12 handoff. Each entry below changes how the repository should be run or how its results should be interpreted.

## 1. Qwen3.6-27B failed with the default CUDA-graph capture range

**Symptom**

The model loaded its 66 safetensor shards and then the vLLM engine failed during cache/graph initialization. The default capture range left insufficient memory for the hybrid Gated DeltaNet recurrent-state cache.

**Fix**

All Tier C modes cap graph capture:

```text
--max-cudagraph-capture-size 256
```

This is a survival constraint for the measured 48 GB card. Do not remove it while calling the resulting mode “vanilla”; without it, the model does not become ready.

## 2. Native context length is not viable on 48 GB

Qwen3.6 advertises a 262,144-token context. Allocating for it overcommits KV/recurrent-state memory.

**Fix**

```text
--language-model-only
--max-model-len 8192
```

The first flag omits the vision encoder because this project benchmarks text TTFT. The second bounds the benchmark window. Both are capacity prerequisites.

## 3. Shell wrappers orphaned GPU-owning vLLM children

**Symptom**

Killing `scripts/06_qwen36_27b.sh` stopped the shell but left `vllm serve` running. Port 8000 and tens of GiB of GPU memory remained occupied. Later servers then failed or produced misleading behavior.

**Fix**

`run_tier_c.sh` tracks the wrapper and also terminates the real child with:

```text
pkill -f "vllm serve Qwen/Qwen3.6-27B-FP8"
```

It waits for port 8000 to disappear before starting the next mode. After any interrupted manual run, verify with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`.

## 4. Prompt decode did not initially preserve exact token counts

**Symptom**

Slicing tokenizer IDs and decoding them can retokenize to a different length because token boundaries merge. This invalidates prompt-length labels.

**Fix**

`bench/benchmark_ttft.py` and `bench/prefix_cache_sweep.py` iteratively adjust decoded prompts until re-encoding returns the requested length. Tests include a tokenizer whose decode deliberately drops a token, so a future regression is observable.

## 5. Partial streaming failures were previously easy to overlook

**Symptom**

A server disconnect could leave a cell with fewer samples while analysis still looked plausible.

**Fix**

The harness records failures, checks a minimum measured success fraction, and refuses to write a CSV when a required cell fails. Generated summaries show actual `n`.

Three optimized warm cells contain 23 successful rows after one disconnect and passed the configured threshold. This is disclosed in `REPORT.md`.

## 6. The 27B progressive prefix sweep is not a valid model

**Observed file**

`results_tier_c/prefix_sweep_27b-optimized.txt`

**Observed behavior**

Nominally decreasing uncached tokens from 4096 to 205 changed p50 from 1034.4 ms to 1100.4 ms. The fitted slope was negative. vLLM logged that prefix caching for Mamba layers was experimental in `align` mode.

**Interpretation**

The sweep did not create the assumed progressive cache-hit workload. Its fitted intercept, slope, throughput, and holdout error are not meaningful.

**Fix**

The revised sweep rejects non-finite or non-positive scaling. The main exact shared-prefix matrix remains valid and shows 4096-token optimized warm p50 of 117.1 ms. An optional `cache-all` mode exists for a separate `--mamba-cache-mode all` experiment.

## 7. Prefix-cache telemetry looked inconsistent until workload identity was checked

Cold prompts intentionally randomize the first tokens and should report no prefix hits. Warm prompts share their long body and move the nonce to the tail. Token-ID checks confirmed common prefixes of essentially the full prompt. Do not infer cache behavior from the label alone; inspect server `Prefix cache hit rate` telemetry and verify token-ID prefix identity when changing prompt construction.

## 8. FP8 GEMM fallback warning was real, but microkernel gains did not fully transfer

**Symptom**

vLLM logged that no block-FP8 configuration existed for RTX 6000 Ada and Qwen3.6 matrix shapes, then used a generic default.

**Fix**

`bench/tune_qwen36_fp8.py` searches a bounded subset of the official vLLM configuration space at `M=4096`. It writes five JSON files in `results_tier_c/fp8_configs/` and can install them with `--install`.

**Measured outcome**

Individual microkernels improved by roughly 1.13–1.16× for four shapes, but the full tuned matrix was almost identical to vanilla. The focused tuned-plus-optimized 4096 cold cell improved end-to-end p50 from 1193.8 ms to 1063.2 ms. Do not report the microkernel ratio as request-level TTFT speedup.

## 9. Aggressive flags are not a general win on the hybrid model

The aggressive mode adds `--kv-cache-dtype fp8` and `-O3` to the optimized mode. It booted, but many cold/concurrent cells regressed; examples include 4096 cold c=4 at 4865.1 ms versus vanilla 2420.7 ms.

Keep it as ablation evidence. Do not promote it as the default.

## 10. SGLang cross-check was abandoned cleanly

A CUDA-12-compatible SGLang environment was installed after dependency-resolution and disk-quota work. Qwen3.6 model loading either failed in the RadixAttention path or timed out in the FlashInfer attempt. No valid CSV was produced. The environment was removed to release volume space.

No report claim depends on SGLang.

## 11. TGI 4B source build loaded the model but could not serve requests

**Why source build was attempted**

This RunPod container had no Docker daemon. Nested-container and chroot alternatives lacked the required privileges. TGI also cannot serve Qwen3.6-27B, so the intended comparison model was `Qwen/Qwen3-4B-Instruct-2507`.

**What succeeded**

- TGI 3.3.7 source checkout.
- Python 3.11 environment and locked Rust router/launcher builds.
- Runtime kernel downloads.
- 4B checkpoint download and GPU load.
- `/v1/models` readiness.

**Decisive failure**

The Rust router embedded-Python tokenizer worker initialized an incomplete Python environment. Logs showed missing `_ctypes`, a partially initialized `charset_normalizer`, and failure to import `get_full_repo_name` from `huggingface_hub`. Validation workers panicked; generation POSTs closed without a response. The TTFT harness rejected all cells and wrote no CSV.

**Disposition**

Do not continue patching this native build. Use `ghcr.io/huggingface/text-generation-inference:latest` on a fresh TGI-image pod and run the client with `--api chat`. There is no valid `tgi-vanilla.csv` in this repository.

## 12. Disk quota forced checkpoint cleanup

The 27B cache (~29 GB), TGI virtual environment (~10 GB), and 4B TGI checkpoint approached the 50 GB persistent-volume quota. After all 27B measurements were safely written, the cached `Qwen3.6-27B-FP8` checkpoint was deleted to complete the TGI investigation.

A future Tier C pod must download the 27B model again. Measurement artifacts and device configs are preserved in the repository.

## 13. Final verified environment

Tier C measurement environment:

- NVIDIA RTX 6000 Ada Generation, 49,140 MiB
- driver 570.195.03
- Python 3.12.3
- vLLM 0.19.1
- torch 2.10.0+cu128
- `Qwen/Qwen3.6-27B-FP8`

At handoff, no GPU compute process remained. Focused harness tests passed before documentation cleanup; rerun `python -m unittest tests.test_ttft_harness` after any benchmark-code change.
