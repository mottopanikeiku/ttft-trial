#!/usr/bin/env python3
"""
analyze.py — aggregate all results/*.csv into report-ready tables and plots.

Outputs:
  results/summary.md          per-config p50/p90/p99 table (markdown)
  results/speedup.md          speedup vs a chosen baseline label
  plots/ttft_vs_prompt.png    the optimization ladder: TTFT p50 vs prompt length
  plots/ttft_concurrency.png  p99 TTFT vs concurrency (head-of-line story)
  plots/speedup_bar.png       headline speedup bars vs baseline
  plots/arch_scaling.png      classic attention vs hybrid GDN scaling (if data)

Usage:
  python bench/analyze.py [--baseline vllm-vanilla] [--results results] [--plots plots]
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style: fixed entity->color map (color follows the config, never plot order)
# Palette slots are a validated CVD-safe set; assignment order is fixed.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary ink: titles
INK2 = "#52514e"         # secondary ink: labels, legends, direct labels
MUTED = "#898781"        # axis tick labels
GRID = "#e1e0d9"         # hairline gridlines
BASELINE = "#c3c2b7"     # axis spines

COLOR = {
    "vllm-vanilla":            "#2a78d6",  # blue   (slot 1, the reference)
    "vllm-optimized":          "#1baf7a",  # aqua   (slot 2)
    "naive-hf":                "#eda100",  # yellow (slot 3)
    "vllm-fp8-only":           "#008300",  # green  (slot 4)
    "ablation-no-prefixcache": "#4a3aa7",  # violet (slot 5)
    "ablation-cp-512":         "#e34948",  # red    (slot 6)
    "ablation-eager":          "#e87ba4",  # magenta(slot 7)
    "arch-hybrid-gdn":         "#eb6834",  # orange (slot 8)
    "arch-full-attn":          "#2a78d6",  # same model as vllm-vanilla -> same hue
    "qwen36-27b-vanilla":       "#2a78d6",
    "qwen36-27b-optimized":     "#1baf7a",
    "qwen36-27b-aggressive":    "#e34948",
}
PRETTY = {
    "vllm-vanilla": "vLLM vanilla (BF16)",
    "vllm-optimized": "legacy optimized label",
    "naive-hf": "naive HF transformers",
    "vllm-fp8-only": "vLLM optimized (FP8-only)",
    "ablation-no-prefixcache": "no prefix cache",
    "ablation-cp-512": "chunked prefill 512",
    "ablation-eager": "eager (no CUDA graphs)",
    "arch-full-attn": "Qwen3-4B (full attention)",
    "arch-hybrid-gdn": "Qwen3.5-4B (hybrid GDN)",
    "qwen36-27b-vanilla": "Qwen3.6-27B vanilla",
    "qwen36-27b-optimized": "Qwen3.6-27B optimized",
    "qwen36-27b-aggressive": "Qwen3.6-27B aggressive",
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def new_fig(w=7.5, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    style_axes(ax)
    return fig, ax


def finish(fig, ax, title, xlabel, ylabel, path):
    ax.set_title(title, color=INK, fontsize=12, fontweight="semibold",
                 loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def line(ax, x, y, label, dashed=False):
    c = COLOR.get(label, MUTED)
    ax.plot(x, y, color=c, linewidth=2,
            linestyle="--" if dashed else "-",
            marker="o", markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2,
            label=PRETTY.get(label, label), zorder=3)


def direct_label(ax, x, y, label, dy=0):
    """Selective end-of-line label in secondary ink (relief rule for low-
    contrast hues); a colored legend swatch still carries identity."""
    ax.annotate(PRETTY.get(label, label), (x, y),
                xytext=(6, dy), textcoords="offset points",
                fontsize=8.5, color=INK2, va="center")


def legend(ax, loc="upper left"):
    ax.legend(loc=loc, frameon=False, fontsize=8.5)


def load(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no CSVs in {results_dir}/ — run benchmarks first")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = raw["ttft_s"].isna()
    if missing.any():
        print(f"WARNING: dropping {missing.sum()} rows with missing ttft_s", file=sys.stderr)
    df = raw[~missing].copy()
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


def tok_ticks(ax, values):
    labels = [f"{v//1024}K" if v >= 1024 and v % 1024 == 0 else str(v) for v in values]
    ax.set_xticks(values, labels)
    ax.minorticks_off()


def ms_ticks(ax, ticks):
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--plots", default="plots")
    ap.add_argument("--baseline", default="vllm-vanilla")
    ap.add_argument("--include-label-regex", default=None,
                    help="only analyze labels matching this regex")
    ap.add_argument("--exclude-label-regex", default=None,
                    help="exclude labels matching this regex")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    df = load(args.results)
    if args.include_label_regex:
        df = df[df["label"].astype(str).str.contains(args.include_label_regex, regex=True, na=False)]
    if args.exclude_label_regex:
        df = df[~df["label"].astype(str).str.contains(args.exclude_label_regex, regex=True, na=False)]
    if df.empty:
        raise SystemExit("no rows left after label filters")
    summ = summarize(df)
    have = set(summ.label.unique())
    if args.baseline not in have:
        raise SystemExit(
            f"baseline {args.baseline!r} absent after label filters; "
            f"available labels: {', '.join(sorted(have))}"
        )
    with open(os.path.join(args.results, "summary.md"), "w") as f:
        f.write(summ.to_markdown(index=False))
    print(summ.to_string(index=False))

    # ---- speedup vs baseline (matched configs) ----
    base = summ[summ.label == args.baseline][
        ["cache_mode", "prompt_tokens", "concurrency", "p50"]
    ].rename(columns={"p50": "base_p50"})
    sp = summ.merge(base, on=["cache_mode", "prompt_tokens", "concurrency"])
    sp["speedup"] = (sp["base_p50"] / sp["p50"]).round(2)
    sp = sp[sp.label != args.baseline]
    with open(os.path.join(args.results, "speedup.md"), "w") as f:
        f.write(sp[["label", "cache_mode", "prompt_tokens", "concurrency",
                    "base_p50", "p50", "speedup"]].to_markdown(index=False))

    have = set(summ.label.unique())

    # ---- plot 1: the dominant mechanism at c=1 ----
    if "qwen36-27b-vanilla" in have and "qwen36-27b-optimized" in have:
        fig, ax = new_fig()
        series = [
            ("qwen36-27b-vanilla", "cold", "#2a78d6", "vanilla — cold", "-"),
            ("qwen36-27b-optimized", "warm", "#1baf7a", "optimized — prefix reused", "-"),
        ]
        token_values = set()
        for lab, mode, color, display_name, linestyle in series:
            grp = summ[
                (summ.concurrency == 1)
                & (summ.cache_mode == mode)
                & (summ.label == lab)
            ].sort_values("prompt_tokens")
            token_values.update(grp.prompt_tokens)
            ax.plot(
                grp.prompt_tokens,
                grp.p50,
                color=color,
                linewidth=2,
                linestyle=linestyle,
                marker="o",
                markersize=5.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=display_name,
                zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        tok_ticks(ax, sorted(token_values))
        ms_ticks(ax, [30, 100, 300, 1000, 3000])
        legend(ax)
        finish(
            fig,
            ax,
            "Prefix reuse, not extra server flags, moves 27B TTFT",
            "prompt length (tokens)",
            "TTFT p50 (ms, log)",
            os.path.join(args.plots, "ttft_vs_prompt.png"),
        )
    else:
        ladder = [
            lab
            for lab in ["naive-hf", "vllm-vanilla", "vllm-fp8-only"]
            if lab in have
        ]
        sel = summ[
            (summ.concurrency == 1)
            & (summ.cache_mode == "cold")
            & (summ.label.isin(ladder))
        ]
        if not sel.empty:
            fig, ax = new_fig()
            for lab in ladder:
                grp = sel[sel.label == lab].sort_values("prompt_tokens")
                if not grp.empty:
                    line(ax, grp.prompt_tokens, grp.p50, lab, dashed=(lab == "vllm-fp8-only"))
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            tok_ticks(ax, sorted(sel.prompt_tokens.unique()))
            ms_ticks(ax, [30, 100, 300, 1000, 3000])
            legend(ax)
            finish(
                fig,
                ax,
                "Cold TTFT vs prompt length — Qwen3-4B, RTX 4000 Ada (c=1, p50)",
                "prompt length (tokens)",
                "TTFT (ms, log)",
                os.path.join(args.plots, "ttft_vs_prompt.png"),
            )

    # ---- plot 2: cache state controls p99 under load ----
    if "qwen36-27b-optimized" in have:
        big = summ[
            (summ.label == "qwen36-27b-optimized")
            & (summ.prompt_tokens == 4096)
        ]
        if not big.empty and big.concurrency.nunique() > 1:
            fig, ax = new_fig()
            for mode, color, display_name in [
                ("cold", "#e34948", "cold prefix"),
                ("warm", "#1baf7a", "reused prefix"),
            ]:
                grp = big[big.cache_mode == mode].sort_values("concurrency")
                ax.plot(
                    grp.concurrency,
                    grp.p99,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=5.5,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    label=display_name,
                    zorder=3,
                )
            ax.set_yscale("log")
            ax.set_xticks(sorted(big.concurrency.unique()))
            ms_ticks(ax, [100, 300, 1000, 3000, 10000, 30000])
            legend(ax)
            finish(
                fig,
                ax,
                "Prefix reuse controls p99 — Qwen3.6-27B, 4096-token prompts",
                "concurrent requests",
                "TTFT p99 (ms, log)",
                os.path.join(args.plots, "ttft_concurrency.png"),
            )
    else:
        story = [
            lab
            for lab in ["vllm-vanilla", "vllm-fp8-only", "ablation-cp-512"]
            if lab in have
        ]
        big = summ[(summ.cache_mode == "cold") & (summ.label.isin(story))]
        if not big.empty and big.concurrency.nunique() > 1:
            tokmax = big.prompt_tokens.max()
            big = big[big.prompt_tokens == tokmax]
            fig, ax = new_fig()
            for lab in story:
                grp = big[big.label == lab].sort_values("concurrency")
                if not grp.empty:
                    line(ax, grp.concurrency, grp.p99, lab)
            ax.set_yscale("log")
            ax.set_xticks(sorted(big.concurrency.unique()))
            ms_ticks(ax, [300, 1000, 3000, 10000])
            legend(ax)
            finish(
                fig,
                ax,
                f"p99 TTFT under load — {int(tokmax)}-token cold prompts",
                "concurrent requests",
                "TTFT p99 (ms, log)",
                os.path.join(args.plots, "ttft_concurrency.png"),
            )

    # ---- plot 3: headline speedup bars (c=1, 2048 tok if present) ----
    tgt = 2048 if (sp.prompt_tokens == 2048).any() else sp.prompt_tokens.max()
    barsel = sp[(sp.concurrency == 1) & (sp.prompt_tokens == tgt)
                & (~sp.label.str.startswith("arch-"))]
    if not barsel.empty:
        piv = (barsel.pivot_table(index="label", columns="cache_mode",
                                  values="speedup")
                     .sort_values("cold"))
        labels = list(piv.index)
        y = np.arange(len(labels))
        fig, ax = new_fig(7.5, 0.75 * len(labels) + 1.6)
        hbar = 0.34
        cold = ax.barh(y + hbar / 2 + 0.02, piv["cold"], height=hbar,
                       color="#2a78d6", label="cold cache",
                       edgecolor=SURFACE, linewidth=1)
        warm = ax.barh(y - hbar / 2 - 0.02, piv.get("warm", piv["cold"]),
                       height=hbar, color="#1baf7a", label="warm cache",
                       edgecolor=SURFACE, linewidth=1)
        for bars in (cold, warm):
            for b in bars:
                w = b.get_width()
                ax.annotate(f"{w:.2f}×", (w, b.get_y() + b.get_height() / 2),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=8.5, color=INK2, va="center")
        ax.axvline(1.0, color=INK, linewidth=0.9, zorder=4)
        ax.set_yticks(y, [PRETTY.get(l, l) for l in labels])
        ax.grid(False, axis="y")
        legend(ax, loc="lower right")
        finish(fig, ax,
               f"TTFT speedup — {tgt}-token prompts, c=1",
               f"p50 speedup vs {args.baseline}  (right of 1.0 = faster)", "",
               os.path.join(args.plots, "speedup_bar.png"))

    # ---- plot 4: architecture scaling (classic vs hybrid GDN) ----
    arch = [l for l in ["arch-full-attn", "arch-hybrid-gdn"] if l in have]
    if len(arch) == 2:
        sel = summ[(summ.concurrency == 1) & (summ.cache_mode == "cold")
                   & (summ.label.isin(arch))]
        fig, ax = new_fig()
        for lab in arch:
            grp = sel[sel.label == lab].sort_values("prompt_tokens")
            line(ax, grp.prompt_tokens, grp.p50, lab)
            # fitted log-log slope over the long-prompt tail (>=2048 tok)
            tail = grp[grp.prompt_tokens >= 2048]
            if len(tail) >= 2:
                k = np.polyfit(np.log(tail.prompt_tokens), np.log(tail.p50), 1)[0]
                last = grp.iloc[-1]
                dy = 10 if lab == "arch-full-attn" else -16
                ax.annotate(f"slope ≈ {k:.2f}",
                            (last.prompt_tokens, last.p50),
                            xytext=(8, dy), textcoords="offset points",
                            fontsize=8.5, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        tok_ticks(ax, sorted(sel.prompt_tokens.unique()))
        ms_ticks(ax, [30, 100, 300, 1000, 3000])
        ax.set_xlim(right=ax.get_xlim()[1] * 2.2)  # room for slope annotations
        legend(ax)
        finish(fig, ax,
               "TTFT scaling: full attention vs hybrid Gated-DeltaNet",
               "prompt length (tokens) — cold, c=1, identical vLLM flags",
               "TTFT p50 (ms, log)",
               os.path.join(args.plots, "arch_scaling.png"))

    print(f"\nwrote {args.results}/summary.md, {args.results}/speedup.md and {args.plots}/*.png")


if __name__ == "__main__":
    main()
