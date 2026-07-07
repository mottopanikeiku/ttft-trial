# DEBUGLOG.md — every incident, diagnosis, fix, and edit (2026-07-05)

A chronological journal of everything that went wrong (or would have gone
wrong) while bringing this project up on the RunPod RTX 4000 Ada pod, and how
each was debugged. Kept deliberately honest — including the debugging
mistakes — because the *process* is as much a deliverable as the numbers.

The method used throughout:

1. **Read the actual error, bottom-up.** The last lines of a traceback name
   the immediate cause; the middle names the path to it.
2. **Identify the layer.** App code? Missing package? Library version?
   Environment/driver? Hardware? Each layer has a different fix and a
   different cheapest test.
3. **Run the cheapest decisive test** for the hypothesis before any
   expensive action (a 1-line python import beats a 400 MB reinstall).
4. **Fix the root cause, not the symptom.** Re-verify by exercising the
   real code path.
5. **Encode the lesson** somewhere it prevents recurrence (a pinned version,
   a comment, this file) — a fix that lives in one person's memory is a
   future incident.

---

## Incident 1 — naive baseline server crash: missing `accelerate`

- **Symptom:** `naive_hf_server.py` exited code 1 immediately after
  "loading model...".
- **Diagnosis:** last traceback line said it all:
  `ValueError: Using a device_map ... requires accelerate. You can install
  it with pip install accelerate`.
- **Root cause:** `transformers.from_pretrained(device_map="cuda")` delegates
  device placement to the `accelerate` package; the setup script never
  installed it.
- **Fix:** `pip install accelerate`; added to `scripts/00_setup.sh` so the
  Tier-C pod gets it automatically.
- **Lesson:** error messages that name their own fix are gifts; read to the
  bottom first.

## Incident 2 — the big one: torch/driver CUDA mismatch

- **Symptom:** server restart crashed with
  `RuntimeError: The NVIDIA driver on your system is too old (found version
  12040)` — yet `nvidia-smi` worked perfectly.
- **Diagnosis:** the driver hadn't changed, so something changed about what
  was *asking* for it. `pip list` showed torch had moved from the image's
  `2.8.0+cu128` to `2.11.0+cu130`: installing vLLM had silently upgraded
  torch to a **CUDA 13** build. Driver 550 speaks CUDA 12.x only; CUDA 13 is
  a major-version jump requiring driver ≥ r580. `nvidia-smi` worked because
  the *driver* was fine — the *user-space wheel* was incompatible.
- **Root cause:** `pip install -U "vllm>=0.19"` grabs the newest release,
  whose wheels (and pinned torch) are CUDA-13 builds. Rented pods have
  host-managed drivers you cannot update — software must match hardware,
  not vice versa.
- **Fix (analytic, not trial-and-error):** instead of downloading 400 MB
  wheels to test one by one, queried PyPI metadata:
  - each vLLM release pins an exact torch (`requires_dist`);
  - each torch release's dependency list reveals its CUDA major
    (`nvidia-cuda-runtime-cu12` vs none).
  Result: vLLM 0.20.2–0.24.0 all pin torch 2.11 (CUDA 13);
  **vLLM 0.19.1 pins torch 2.10.0, whose default wheel is cu128** — the
  newest driver-compatible release, and exactly the project's minimum
  version for hybrid-architecture support. Installed with
  `uv pip install "vllm==0.19.1" --torch-backend=cu128`.
- **Lesson:** before an expensive experiment, look for a cheap source of the
  same information. And always pin the torch/CUDA pairing to the driver
  *before* installing a serving stack.

## Incident 3 — self-inflicted: masked exit codes and discarded stderr

- **Symptom (a):** an `uv pip install` "succeeded" but changed nothing.
  The command was piped through `tail`, and a pipeline reports the *last*
  command's exit code — uv had actually refused to run (Debian PEP-668
  "externally managed environment") and the refusal was invisible.
- **Symptom (b):** twice, a `2>/dev/null` on a probing command threw away
  the very error that explained the failure.
- **Fix:** re-ran with `--break-system-packages`, and switched to
  `echo "EXIT:${PIPESTATUS[0]}"` to preserve the true exit code through
  pipes; stopped discarding stderr on anything diagnostic.
- **Lesson:** the error channel is the product when debugging. Never bury
  it, and never trust a pipeline's aggregate exit code.

## Incident 4 — the lazy-import trap (vLLM 0.23.0)

- **Symptom:** `python -c "import vllm"` succeeded, so the environment was
  declared fixed — then `vllm serve --help` crashed with
  `ImportError: libcudart.so.13: cannot open shared object file`.
- **Root cause:** vLLM loads its compiled CUDA extension (`vllm._C`)
  lazily, on first platform use — not at top-level import. The top-level
  import was a false-positive health check. 0.23.0's extension is also a
  CUDA-13 build.
- **Fix:** dropped to 0.19.1 (see Incident 2) and changed the health check
  to the *actual* code path: `vllm serve --help` reaching argparse proves
  the extension loads.
- **Lesson:** verify the exact code path you will use in production, not a
  proxy for it.

## Incident 5 — removed CLI flag would have crashed every server

- **Symptom (pre-empted):** none — caught by pre-flight check, not failure.
- **Diagnosis:** before launching a 2-hour benchmark chain, verified every
  flag in the server scripts against the installed vLLM
  (`vllm serve --help=<flag>`). `--disable-log-requests` no longer exists —
  the flag's polarity flipped (`--enable-log-requests` now exists, logging
  is off by default). Every server script used the dead flag; each would
  have died at argparse.
- **Fix:** removed the flag everywhere (behavior is unchanged since
  logging now defaults off); verified `-O3` still parses.
- **Lesson:** CLI surfaces churn between major versions. A 10-second
  `--help` check per flag is infinitely cheaper than a failed overnight run.

## Incident 6 — orphan-server and infinite-wait bugs in the runner (code review)

- **Symptom (pre-empted):** none — found by reading `run_all.sh` end-to-end
  before starting it.
- **Bug (a):** `bash scripts/02_...sh & pid=$!` then `kill $pid` kills the
  *bash wrapper*, not its `vllm serve` child. The orphaned server keeps
  port 8000 and ~18 GB of GPU memory; every subsequent server in the chain
  would fail to bind or OOM.
- **Bug (b):** `wait_ready` polled the port in an unbounded `until` loop —
  a server that crashes during startup means the runner hangs *forever*,
  silently.
- **Fix:** added `stop_server()` (pkill the real process, wait for the port
  to actually close) and a bounded `wait_ready` that also aborts if the
  server process disappears. The bounded version paid for itself within
  minutes (Incident 7).
- **Lesson:** when a failure would cost hours, spend ten minutes tracing
  process lifetimes and loop exits at every boundary of the script.

## Incident 7 — vanilla vLLM refused to boot: KV cache vs 262k context

- **Symptom:** `EngineCore failed to start`;
  `ValueError: To serve at least one request with the model's max seq len
  (262144), 36.0 GiB KV cache is needed ... available (9.28 GiB)`.
- **Root cause:** stock defaults use the model's native 262k context; vLLM
  (correctly) refuses to start if a single max-length request cannot fit in
  KV cache. A 20 GB card has ~9.3 GiB spare after BF16 weights.
- **Fix:** `--max-model-len 16384` in the vanilla script — the single
  deviation from stock defaults, documented in the report. (Predicted in
  advance by EXECUTION.md; the error message itself also names the fix.)
- **Lesson:** "vanilla defaults" are defined relative to hardware; document
  every deviation and why it exists.

## Incident 8 — silent benchmark corruption at 16k prompts (code review)

- **Symptom (pre-empted):** none — found by reading the harness before the
  architecture study.
- **Diagnosis:** `build_prompts()` sliced token bodies from a fixed pool of
  `FILLER * 200` ≈ 7.5k tokens. Requesting a 16,384-token prompt would
  silently return a ~7.5k-token one — every "16k" measurement would really
  measure 7.5k, invalidating the scaling experiment while *looking* fine.
- **Fix:** pool now scales with the request (`reps = max(200, n_tokens //
  20)`) plus a hard `ValueError` if the pool is ever short. Phase-2 data
  unaffected (its max was 4096, within the old pool).
- **Lesson:** the worst bugs don't crash — they return plausible wrong
  numbers. Put assertions at the boundary between "what I asked for" and
  "what I got."

---

## Change log (every file edited and why)

| File | Change | Trigger |
|---|---|---|
| `scripts/00_setup.sh` | pin `vllm==0.19.1 --torch-backend=cu128` via uv; add `accelerate`, `fastapi`, `uvicorn`, `numpy<2.4`; explanatory comments | Incidents 1, 2; numpy resolver conflict from install log |
| `scripts/02_vllm_baseline.sh` | drop dead `--disable-log-requests`; add `--max-model-len 16384` + rationale comment | Incidents 5, 7 |
| `scripts/03_vllm_optimized.sh` | drop dead flag | Incident 5 |
| `scripts/04_vllm_ablation.sh` | drop dead flag; bounded `wait_ready`; wait-for-port-close between servers | Incidents 5, 6 |
| `scripts/06_qwen36_27b.sh` | drop dead flag (both variants) | Incident 5 |
| `run_all.sh` | `stop_server()` helper; bounded `wait_ready`; kill real vllm process not wrapper | Incident 6 |
| `bench/benchmark_ttft.py` | scale filler pool with requested length + hard error if short | Incident 8 |
| `README.md`, `EXECUTION.md` | update example commands for removed flag | Incident 5 |
| `scripts/07_extras.sh` | new: FP8-isolation run + Tier-B architecture study, reusing hardened helpers | follow-ups to Phase 2 findings |
| `scripts/03_vllm_optimized.sh`, `run_all.sh`, `results/legacy/` | promote measured `vllm-fp8-only` as optimized; archive old 5-flag `vllm-optimized` CSV outside analysis glob; remove default `-O3` path | Phase-2 evidence: FP8-only beat the 5-flag bundle in most cells and avoided large cold-concurrency regressions |
| `bench/benchmark_ttft.py`, `bench/prefix_cache_sweep.py` | hard-fail empty streams / insufficient successful samples; warm up at measured concurrency; guard prefix-sweep fit | pre-baseline robustness review; avoid silent partial CSVs and NaN fits |
| `bench/analyze.py`, `EXECUTION.md` | add label include/exclude filters and Tier-C separate result/plot directory runbook | prevent cross-model 4B/27B speedup comparisons when new baselines are added |
| `README.md`, `REPORT.md`, `scripts/04_vllm_ablation.sh`, `scripts/06_qwen36_27b.sh`, `scripts/07_extras.sh` | align prose and launch modes with measured FP8-only recommendation; scope cp-512 conclusion; add text-only Tier-C vanilla prerequisite | consistency pass before new baselines |
| `scripts/00_setup.sh` | tighten numpy pin from `<2.4` to `<2.3` | RunPod Tier-C setup pulled numpy 2.3.5; numba 0.61.x requires `<2.3` |

Environment as finally pinned: `vllm 0.19.1` · `torch 2.10.0+cu128` ·
`transformers (as resolved by vllm 0.19.1)` · `numpy < 2.3` · driver
550.127.05 (CUDA 12.4) · RTX 4000 Ada 20 GB (SM 8.9).
