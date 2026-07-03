#!/usr/bin/env python3
"""
analyze.py — aggregate all results/*.csv into report-ready tables and plots.

Outputs:
  results/summary.md          per-config p50/p90/p99 table (markdown)
  results/speedup.md          speedup vs a chosen baseline label
  plots/ttft_vs_prompt.png    TTFT p50 vs prompt length, one line per config
  plots/ttft_concurrency.png  p99 TTFT vs concurrency (head-of-line story)
  plots/speedup_bar.png       headline speedup bar chart

Usage:
  python bench/analyze.py [--baseline vllm-vanilla] [--results results] [--plots plots]
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no CSVs in {results_dir}/ — run benchmarks first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["ttft_s"])
    df["ttft_ms"] = df["ttft_s"] * 1000
    return df


def summarize(df):
    g = (df.groupby(["label", "cache_mode", "prompt_tokens", "concurrency"])["ttft_ms"]
           .agg(n="count",
                p50=lambda s: s.quantile(0.50),
                p90=lambda s: s.quantile(0.90),
                p99=lambda s: s.quantile(0.99))
           .round(1)
           .reset_index())
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--plots", default="plots")
    ap.add_argument("--baseline", default="vllm-vanilla")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    df = load(args.results)
    summ = summarize(df)
    with open(os.path.join(args.results, "summary.md"), "w") as f:
        f.write(summ.to_markdown(index=False))
    print(summ.to_string(index=False))

    # ---- speedup vs baseline (cold, matched configs) ----
    base = summ[summ.label == args.baseline][
        ["cache_mode", "prompt_tokens", "concurrency", "p50"]
    ].rename(columns={"p50": "base_p50"})
    sp = summ.merge(base, on=["cache_mode", "prompt_tokens", "concurrency"])
    sp["speedup"] = (sp["base_p50"] / sp["p50"]).round(2)
    sp = sp[sp.label != args.baseline]
    with open(os.path.join(args.results, "speedup.md"), "w") as f:
        f.write(sp[["label", "cache_mode", "prompt_tokens", "concurrency",
                    "base_p50", "p50", "speedup"]].to_markdown(index=False))

    # ---- plot 1: TTFT vs prompt length (c=1, cold) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    sel = summ[(summ.concurrency == 1) & (summ.cache_mode == "cold")]
    for label, grp in sel.groupby("label"):
        grp = grp.sort_values("prompt_tokens")
        ax.plot(grp.prompt_tokens, grp.p50, marker="o", label=label)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title("Cold-cache TTFT vs prompt length (concurrency=1)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "ttft_vs_prompt.png"), dpi=150)

    # ---- plot 2: p99 TTFT vs concurrency (largest prompt) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    big = summ[summ.cache_mode == "cold"]
    big = big[big.prompt_tokens == big.prompt_tokens.max()]
    for label, grp in big.groupby("label"):
        grp = grp.sort_values("concurrency")
        ax.plot(grp.concurrency, grp.p99, marker="s", label=label)
    ax.set_xlabel("concurrency"); ax.set_ylabel("TTFT p99 (ms)")
    ax.set_title(f"p99 TTFT under load ({int(big.prompt_tokens.max())}-token prompts, cold)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "ttft_concurrency.png"), dpi=150)

    # ---- plot 3: headline speedup bars (c=1, 2048 tok if present) ----
    tgt = 2048 if (sp.prompt_tokens == 2048).any() else sp.prompt_tokens.max()
    bar = sp[(sp.concurrency == 1) & (sp.prompt_tokens == tgt)]
    if not bar.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        piv = bar.pivot_table(index="label", columns="cache_mode", values="speedup")
        piv.plot(kind="bar", ax=ax)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_ylabel(f"TTFT speedup vs {args.baseline} (p50)")
        ax.set_title(f"Speedup at {tgt}-token prompts, concurrency=1")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.plots, "speedup_bar.png"), dpi=150)

    print(f"\nwrote {args.results}/summary.md, {args.results}/speedup.md and plots/*.png")


if __name__ == "__main__":
    main()
