# EXECUTION.md — Current Reproduction and Handoff Runbook

This file describes the repository as it exists after the RTX 6000 Ada session on 2026-07-12. It replaces the original speculative phase plan.

## 1. Current state

Completed and preserved:

- Full Qwen3.6-27B-FP8 vanilla matrix.
- Full tuned matrix with RTX 6000 Ada FP8 GEMM configurations.
- Full optimized prefix-cache matrix.
- Full aggressive (`FP8 KV + -O3`) ablation matrix.
- Focused 4096-token tuned-plus-optimized cell.
- Focused 3072-token cold sub-second cell.
- Tier C plots, summaries, raw server logs, environment record, and kernel configs.
- Focused harness unit tests.

Intentionally open:

- TGI 4B comparison. No valid TGI CSV exists; use an official TGI-container pod.
- Optional `cache-all` hybrid-Mamba experiment. The mode exists but is not included in the completed matrix.

The 27B checkpoint was deleted from the current volume after measurements to release quota for the TGI attempt. A fresh session must download it again. All measurement CSVs and logs are already in the repository.

## 2. Source-of-truth artifacts

| artifact | meaning |
|---|---|
| `results_tier_c/qwen36-27b-vanilla.csv` | Full 27B baseline matrix |
| `results_tier_c/qwen36-27b-tuned.csv` | Full matrix with installed RTX 6000 Ada FP8 GEMM configs |
| `results_tier_c/qwen36-27b-optimized.csv` | Prefix-cache/chunk-budget matrix |
| `results_tier_c/qwen36-27b-aggressive.csv` | FP8-KV plus `-O3` ablation |
| `results_tier_c/qwen36-27b-tuned-optimized.csv` | Focused 4096 cold c=1 combination |
| `results_tier_c/qwen36-27b-sub1.csv` | Focused 3072 cold c=1 headline |
| `results_tier_c/fp8_configs/*.json` | Five device-specific vLLM block-FP8 configs |
| `results_tier_c/server-*.log` | Exact vLLM startup/config/kernel evidence |
| `results_tier_c/environment.txt` | Recorded pod environment |
| `results_tier_c/summary.md` | Generated absolute statistics |
| `results_tier_c/speedup.md` | Generated same-cell comparisons |
| `plots_tier_c/*.png` | Final figures |

Do not compare `results/` 4B rows against `results_tier_c/` as engine speedups: the GPU and model differ.

## 3. Fresh Tier C pod

Required hardware: one 48 GB Ada GPU (RTX 6000 Ada or L40S). The checked-in FP8 configuration filenames contain `NVIDIA_RTX_6000_Ada_Generation`; retune on an L40S or any differently named device.

```bash
git clone <repo-url> ttft-trial
cd ttft-trial
bash scripts/00_setup.sh tier-c
```

Expected software from the setup script:

- vLLM 0.19.1
- torch 2.10.0+cu128
- NumPy `<2.3`
- benchmark dependencies
- `Qwen/Qwen3.6-27B-FP8` in `HF_HOME`

Before a paid run:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version
vllm --version
python -m unittest tests.test_ttft_harness
```

## 4. Reproduce the completed matrix

Generate and install configs for the active GPU:

```bash
python bench/tune_qwen36_fp8.py --install
```

The bounded tuner targets the five Qwen3.6 block-FP8 matrix shapes at `M=4096`, writes JSON to `results_tier_c/fp8_configs/`, and copies the files into the active vLLM package. Run it with no vLLM server using the GPU.

Then:

```bash
bash run_tier_c.sh vanilla tuned optimized aggressive
```

`run_tier_c.sh`:

1. records environment versions;
2. launches one mode at a time;
3. allows up to 30 minutes for 27B startup and CUDA-graph capture;
4. verifies `/v1/models` readiness;
5. runs the full exact-token matrix;
6. terminates the actual `vllm serve` child, not only the shell wrapper;
7. regenerates summaries and plots.

The runner refuses unknown modes and keeps 27B output under `results_tier_c/` and `plots_tier_c/`.

## 5. Reproduce only the headline boundary

Start the tuned mode after installing the FP8 configs:

```bash
bash scripts/06_qwen36_27b.sh tuned
```

In another shell:

```bash
python bench/benchmark_ttft.py \
  --label qwen36-27b-sub1 \
  --url http://localhost:8000 \
  --tokenizer Qwen/Qwen3.6-27B-FP8 \
  --prompt-tokens 3072 \
  --concurrency 1 \
  --cache-modes cold \
  --num-requests 24 \
  --out results_tier_c
```

Expected recorded result: p50 839.7 ms, p90 850.6 ms, p99 856.5 ms, 24/24 successful. Small run-to-run differences are expected; preserve raw rows and do not replace the checked-in result unless the full environment is recorded.

## 6. Focused 4096 tuned-plus-optimized cell

Install the FP8 configs, start `optimized`, and label the client run explicitly:

```bash
bash scripts/06_qwen36_27b.sh optimized
```

```bash
python bench/benchmark_ttft.py \
  --label qwen36-27b-tuned-optimized \
  --url http://localhost:8000 \
  --tokenizer Qwen/Qwen3.6-27B-FP8 \
  --prompt-tokens 4096 \
  --concurrency 1 \
  --cache-modes cold \
  --num-requests 24 \
  --out results_tier_c
```

Expected recorded result: p50 1063.2 ms, a 1.12× improvement over the 1193.8 ms vanilla cell.

## 7. Prefix-cache caveat

The normal warm workload in `benchmark_ttft.py` produced real, large reuse benefits. The progressive cache-fraction sweep did not.

`results_tier_c/prefix_sweep_27b-optimized.txt` has a negative slope and is not a valid cost model. vLLM's hybrid Mamba cache was in experimental `align` mode. The revised sweep code rejects non-physical fits; do not weaken that guard to recreate a regression line.

Optional continuation:

```bash
bash run_tier_c.sh cache-all
```

This tests `--mamba-cache-mode all`. Treat it as a new experiment: keep its CSV/log separate and compare cache-hit telemetry before interpreting TTFT.

## 8. Regenerate analysis without a GPU

```bash
python bench/analyze.py \
  --baseline qwen36-27b-vanilla \
  --results results_tier_c \
  --plots plots_tier_c \
  --include-label-regex '^qwen36-27b-'
```

This rewrites `results_tier_c/summary.md`, `results_tier_c/speedup.md`, and the Tier C plots. The summary contains focused one-cell labels in addition to the full matrices; that is intentional.

## 9. TGI continuation — use a separate official-container pod

TGI cannot serve Qwen3.6-27B's hybrid architecture. The comparison model is `Qwen/Qwen3-4B-Instruct-2507`.

Recommended pod:

- container image: `ghcr.io/huggingface/text-generation-inference:latest`
- one Ada GPU
- expose the TGI HTTP port
- model: `Qwen/Qwen3-4B-Instruct-2507`
- `--max-input-tokens 8192 --max-total-tokens 8704`

From the client checkout:

```bash
python bench/benchmark_ttft.py \
  --label tgi-vanilla \
  --url http://<tgi-host>:<port> \
  --api chat \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --out results
```

Then regenerate Tier A analysis with `python bench/analyze.py --baseline vllm-vanilla`.

Do not resume the abandoned native-source build on this pod. It loaded the model but its Rust router's embedded Python tokenizer worker failed (`_ctypes`, `charset_normalizer`, and `huggingface_hub` initialization), every generation POST disconnected, and the harness correctly wrote no CSV. `DEBUGLOG.md` records the decisive evidence.

## 10. Shutdown checklist

Before terminating a paid pod:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
python -m unittest tests.test_ttft_harness
```

Expected current compute-process output after cleanup: empty. Confirm the repository contains `results_tier_c/`, `plots_tier_c/`, the modified benchmark/scripts/tests, and this documentation before pushing.
