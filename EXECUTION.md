# EXECUTION.md — Current Reproduction and Handoff Runbook

This file describes the repository after the RTX 6000 Ada validation campaign
on 2026-07-14. It preserves the earlier full-matrix runbook and adds the exact
current cold-path, profiler, PTX/SASS, and Qwen3-4B procedures.

## 1. Current state

Completed and preserved:

- Two independent Qwen3.6-27B-FP8 3072-token cold cells with p99 below 1 s.
- Current-driver 3072/4096 vanilla and dense-M tuned comparisons.
- Async scheduling, exact token-budget, and `-O3` ablations.
- Dense irregular-M and exhaustive M=4096 block-FP8 tuner searches.
- Focused PyTorch profiler trace and Nsight Compute launch/resource evidence.
- Explicit failed CUDA-graph captures at 512, 1024, 2048, and 4096 tokens.
- Same-GPU Qwen3-4B BF16 versus official-FP8 measurements.
- Opt-in channelwise CUTLASS experiment, kept separate because it changes
  weight quantization and has no quality evaluation.
- The earlier full vanilla/tuned/optimized/aggressive Tier C matrix.

Intentionally open:

- TGI 4B comparison. No valid TGI CSV exists; use an official TGI-container pod.
- Optional `cache-all` hybrid-Mamba experiment.

The current volume contains the 27B and official 4B FP8 checkpoints. The 4B
BF16 checkpoint was deleted, after its CSVs and logs were saved, to stay within
the 50 GB volume quota.

## 2. Source-of-truth artifacts

| artifact | meaning |
|---|---|
| `results_ada_570124/qwen36-27b-vanilla-r570124.csv` | Current-driver 27B cold baseline |
| `results_ada_570124/qwen36-27b-dense-*.csv` | Dense-M scheduler/compile ablations |
| `results_ada_570124/qwen36-27b-cold-final-{a,b}-r570124.csv` | Independent sub-second validation cells |
| `results_ada_570124/qwen3-4b-*.csv` | Same-GPU BF16/official-FP8 comparison |
| `results_ada_570124/fp8_configs_v2/*.json` | Dense irregular-M vLLM configs |
| `results_ada_570124/fp8_configs_v2_full/*.json` | Exhaustive M=4096 configs |
| `results_ada_570124/profile/` | PyTorch trace, table, and focused request row |
| `results_ada_570124/ncu-fp8-default.ncu-rep` | Nsight Compute report with launch/resource metrics |
| `results_ada_570124/server-*.log` | Exact startup, engine config, and kernel evidence |
| `results_ada_570124/environment.txt` | Current software/hardware versions |
| `results_ada_570124/summary.md` | Generated current 27B absolute statistics |
| `results_ada_570124/speedup.md` | Generated same-cell comparisons |
| `plots_ada_570124/*.png` | Current 27B figures |
| `results_tier_c/` | Earlier full 27B matrix on driver 570.195.03 |

Do not merge rows across drivers or GPUs when claiming a configuration
speedup. Qwen3-4B rows answer a separate model-size/quantization question.

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

Generate and install dense-M configs for the active RTX 6000 Ada:

```bash
python bench/tune_qwen36_fp8.py \
  --m-values 1 16 128 512 1024 2048 3072 4096 8192 \
  --out results_ada_570124/fp8_configs_v2 \
  --install
```

The tuner rotates four weight tensors, validates the winning output, and
writes the JSON mapping consumed by vLLM's nearest-M dispatcher. Run it with
no serving process on the GPU. For the much slower exhaustive M=4096 search:

```bash
python bench/tune_qwen36_fp8.py --full --m-values 4096 \
  --out results_ada_570124/fp8_configs_v2_full
```
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

## 5. Reproduce the current cold boundary

After installing the dense-M configs, start the pure-cold mode:

```bash
bash scripts/06_qwen36_27b.sh cold
```

This expands to the capacity flags plus:

```text
--max-num-batched-tokens 8192
--no-enable-prefix-caching
--async-scheduling
--disable-log-stats
```

In another shell:

```bash
python bench/benchmark_ttft.py \
  --label qwen36-27b-cold-final \
  --url http://localhost:8000 \
  --tokenizer Qwen/Qwen3.6-27B-FP8 \
  --prompt-tokens 3072 4096 \
  --concurrency 1 \
  --cache-modes cold \
  --num-requests 24 \
  --out results_ada_570124
```

The two saved validation cells produced 3072 p50/p99 of 790.2/803.1 ms and
831.6/837.7 ms. The saved 4096 cells remained above one second. Preserve raw
rows; do not turn the 3072 result into a 4096 claim.

## 6. Reproduce the same-GPU Qwen3-4B comparison

BF16 baseline:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 --max-model-len 8192
```

Official FP8 baseline:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --host 0.0.0.0 --port 8000 --max-model-len 8192
```

Use the same client command for each checkpoint, changing only label and
tokenizer:

```bash
python bench/benchmark_ttft.py \
  --label qwen3-4b-fp8 \
  --url http://localhost:8000 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --prompt-tokens 3072 4096 \
  --concurrency 1 --cache-modes cold --num-requests 24 \
  --out results_ada_570124
```

Saved 4096 p50 values are 339.4 ms BF16 and 217.0 ms official FP8.

## 7. Run the altered-weight CUTLASS experiment

```bash
bash scripts/06_qwen36_27b.sh ada-channel
```

`bench/ada_channel_fp8/sitecustomize.py` executes in every spawned Python
process. It dequantizes each 128x128-scaled linear weight once, requantizes per
output channel, and selects vLLM's CUTLASS FP8 linear kernel without modifying
the installed package. This is not lossless. The 4096 p50 was 1111.0 ms and
did not beat the 1093.6 ms unmodified-weight result; do not deploy it without
task-specific quality evaluation.

## 8. Prefix-cache caveat

The normal warm workload in `benchmark_ttft.py` produced real, large reuse benefits. The progressive cache-fraction sweep did not.

`results_tier_c/prefix_sweep_27b-optimized.txt` has a negative slope and is not a valid cost model. vLLM's hybrid Mamba cache was in experimental `align` mode. The revised sweep code rejects non-physical fits; do not weaken that guard to recreate a regression line.

Optional continuation:

```bash
bash run_tier_c.sh cache-all
```

This tests `--mamba-cache-mode all`. Treat it as a new experiment: keep its CSV/log separate and compare cache-hit telemetry before interpreting TTFT.

## 9. Regenerate analysis without a GPU

```bash
python bench/analyze.py \
  --baseline qwen36-27b-vanilla \
  --results results_tier_c \
  --plots plots_tier_c \
  --include-label-regex '^qwen36-27b-'
```

This rewrites `results_tier_c/summary.md`, `results_tier_c/speedup.md`, and the Tier C plots. The summary contains focused one-cell labels in addition to the full matrices; that is intentional.

## 10. TGI continuation — use a separate official-container pod

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

## 11. Shutdown checklist

Before terminating a paid pod:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
python -m unittest tests.test_ttft_harness
```

Expected current compute-process output after cleanup: empty. Confirm the repository contains `results_tier_c/`, `plots_tier_c/`, the modified benchmark/scripts/tests, and this documentation before pushing.
