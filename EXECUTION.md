# EXECUTION.md — the exact runbook, start to finish

Total budget: your existing RTX 4000 Ada pod (~$0.30-0.58/hr) for ~1 day of
on-and-off work, plus one **RTX 6000 Ada 48GB** session (~$0.77/hr on-demand,
3-4 hours, <$5). Everything below is literal commands in order.

---

## Phase 0 — repo + pod hygiene (30 min, RTX 4000 Ada pod)

```bash
# On the pod, inside a Jupyter terminal:
apt-get update && apt-get install -y tmux git
tmux new -s ttft          # EVERYTHING from now on happens inside tmux.
                          # (Jupyter terminals die with the tab; tmux survives.
                          #  Detach: Ctrl-b d   Reattach: tmux attach -t ttft
                          #  New pane: Ctrl-b %)

git clone https://github.com/<you>/ttft-trial && cd ttft-trial
bash scripts/00_setup.sh              # installs vllm>=0.19 + deps, downloads 4B models
```

Checklist before continuing: `nvidia-smi` shows the 4000 Ada; `HF_HOME` is
`/workspace/hf`; `python -c "import vllm; print(vllm.__version__)"` prints ≥0.19.

## Phase 1 — naive HF floor (30 min)

```bash
pip install fastapi uvicorn
python baselines/naive_hf_server.py &        # pane 1
# pane 2 (Ctrl-b %):
python bench/benchmark_ttft.py --label naive-hf --api chat \
  --prompt-tokens 128 512 1024 2048 --concurrency 1 --num-requests 12
kill %1
```
Expect seconds-scale TTFT at long prompts. If a run fails, read the stderr line
the harness prints per request — it is almost always the fix.

## Phase 2 — vLLM vanilla + ablations + optimized + prefix sweep (2-3 h, mostly unattended)

```bash
bash run_all.sh
```
This sequentially serves vanilla → three ablations → optimized(FP8), runs the
full matrix (5 prompt lengths × 3 concurrencies × cold/warm × 24 requests)
against each, runs the prefix-cache sweep, then builds tables and plots.
Watch the per-config lines it prints; sanity-check monotonicity (TTFT should
grow with prompt length in cold mode and be ~flat in warm mode).

If vanilla vLLM OOMs at startup (262k default context on 20GB): add
`--max-model-len 16384` in scripts/02 and note it in the report as the single
deviation from stock defaults.

## Phase 3 — TGI vanilla (1 h, separate pod)

RunPod → Deploy → same GPU type (RTX 4000 Ada) → **Custom container image**:
`ghcr.io/huggingface/text-generation-inference:latest`, container args:
`--model-id Qwen/Qwen3-4B-Instruct-2507 --max-input-tokens 8192 --max-total-tokens 8704`,
expose HTTP port 80. Wait for "Connected" then from your main pod:

```bash
python bench/benchmark_ttft.py --label tgi-vanilla --api chat \
  --url https://<tgi-pod-id>-80.proxy.runpod.net
```
Terminate the TGI pod immediately after. Note in the report: benchmarked over
RunPod's proxy → add a same-host localhost run of vLLM vs proxied vLLM if you
want to quantify the proxy's contribution (nice extra rigor: run one vllm config
through the proxy too, so TGI vs vLLM is apples-to-apples).
Why TGI gets the older Qwen3-4B: TGI does not support the Qwen3.5/3.6 hybrid
Gated-DeltaNet architecture (its own README now recommends vLLM/SGLang going
forward). All three engines are compared on the one model they all support.

## Phase 4 — Tier B novel experiment: full attention vs hybrid GDN (1 h)

```bash
# server pane — model 1 (classic transformer):
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 16384
# client pane:
python bench/benchmark_ttft.py --label arch-full-attn \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --prompt-tokens 128 512 2048 8192 16384 --concurrency 1 --cache-modes cold
# kill server; then model 2 (hybrid Gated DeltaNet), IDENTICAL flags:
vllm serve Qwen/Qwen3.5-4B --max-model-len 16384
python bench/benchmark_ttft.py --label arch-hybrid-gdn \
  --tokenizer Qwen/Qwen3.5-4B \
  --prompt-tokens 128 512 2048 8192 16384 --concurrency 1 --cache-modes cold
python bench/analyze.py
```
The plot to make: cold TTFT vs prompt length, log-log, both curves. The story:
how prefill scales for linear-attention hybrids vs classic attention. Caveat to
write down: the two models differ in more than attention (Qwen3.5-4B is also
multimodal-pretrained), so frame it as "architecture generation" comparison.

## Phase 5 — Tier C: the actual task, Qwen3.6-27B (3-4 h, 48GB pod)

Rent **RTX 6000 Ada 48GB** (or L40S). Ada generation = native FP8; do NOT take
an A100 (Ampere, no FP8 tensor cores). Then:

```bash
tmux new -s ttft
git clone https://github.com/<you>/ttft-trial && cd ttft-trial
bash scripts/00_setup.sh tier-c              # downloads Qwen3.6-27B-FP8 (~27GB)

bash scripts/06_qwen36_27b.sh vanilla &      # pane 1
python bench/benchmark_ttft.py --label qwen36-27b-vanilla \
  --tokenizer Qwen/Qwen3.6-27B-FP8           # pane 2
kill %1; sleep 10

bash scripts/06_qwen36_27b.sh optimized &
python bench/benchmark_ttft.py --label qwen36-27b-optimized \
  --tokenizer Qwen/Qwen3.6-27B-FP8
python bench/prefix_cache_sweep.py --tokenizer Qwen/Qwen3.6-27B-FP8 \
  | tee results/prefix_sweep_27b.txt
python bench/analyze.py --baseline qwen36-27b-vanilla
```

Success criterion: cold p50 TTFT < 1000 ms at 4096-token prompts, c=1 —
and report the warm-cache number next to it (it will be dramatically lower).
Copy `results/` + `plots/` off the pod (`runpodctl send`, `scp`, or just
`git add results plots && git commit && git push`), THEN terminate the pod.

Known Tier-C failure modes:
- CUDA graph / Mamba cache size error → add `--max-cudagraph-capture-size 256`
  (hybrid GDN state interacts with graph capture; known vLLM issue).
- `--kv-cache-dtype fp8` or `-O3` errors on the hybrid arch → drop them one at
  a time and RECORD the delta; that's ablation data, not failure.
- OOM → lower `--gpu-memory-utilization` to 0.88, confirm
  `--language-model-only` is set (skips the vision encoder).

## Phase 6 — write-up + repo polish (2 h, no GPU)

Fill `report_template.md` with real numbers → rename to `REPORT.md`, put the
headline table + 2 plots at the top of the repo README, commit raw CSVs.
Final deliverable to Veysel = the GitHub link. The repo should let him
reproduce any number in the report with one script + one command.

## Order of what to cut if short on time

Keep (non-negotiable): Phases 2, 3, 5, 6 — the task as assigned.
Cut first: SGLang cross-check → naive-HF floor → Tier B architecture study.
(But Tier B is your strongest "novel" card; cut it last among the extras.)
