#!/usr/bin/env python3
"""
Statistical analysis and paired figures for the 3 x 3 x 5 cross-grid.

Input:
    results/master_table.csv

Outputs:
    figures/paired_architecture_effect.png
    figures/paired_sampling_effect.png
    results/stats_global.csv
    results/stats_pairwise.csv

Analysis:
    - Friedman test across the 3 architectures using 15 matched blocks:
      (sampling x held-out family)
    - Friedman test across the 3 sampling strategies using 15 matched blocks:
      (architecture x held-out family)
    - Pairwise Wilcoxon signed-rank tests
    - Holm correction within each factor
    - Matched-pairs rank-biserial effect size

All tests operate on seed-averaged cell means from master_table.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


MASTER = Path("results/master_table.csv")
FIGDIR = Path("figures")
OUTDIR = Path("results")
FIGDIR.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)

ARCH_ORDER = ["unconstrained", "powerlaw", "geoaware"]
SAMPLING_ORDER = ["uniform", "geo-axial", "geo-wallstretched"]

ARCH_LABELS = {
    "unconstrained": "Unconstrained",
    "powerlaw": "Power-law-aware",
    "geoaware": "Geometry-aware",
}
SAMPLING_LABELS = {
    "uniform": "Uniform",
    "geo-axial": "Geometry-aware axial",
    "geo-wallstretched": "Geometry-aware + wall-stretched",
}


def holm_adjust(pvalues):
    """Holm step-down correction; returns adjusted p-values in original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * pvalues[idx]
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def rank_biserial(first, second):
    """
    Matched-pairs rank-biserial correlation for d = first - second.

    Positive value means first tends to have larger error than second,
    i.e. the second condition tends to be better if lower error is preferred.
    """
    d = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0

    ranks = rankdata(np.abs(d), method="average")
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom else 0.0


def percent_reduction(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denom = np.mean(first)
    if denom == 0:
        return np.nan
    return 100.0 * (np.mean(first) - np.mean(second)) / denom


def pairwise_tests(block_df, ordered_levels, factor_name):
    rows = []
    pvals = []

    pairs = []
    for i in range(len(ordered_levels)):
        for j in range(i + 1, len(ordered_levels)):
            pairs.append((ordered_levels[i], ordered_levels[j]))

    for first, second in pairs:
        x = block_df[first].to_numpy(dtype=float)
        y = block_df[second].to_numpy(dtype=float)
        d = x - y

        stat = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        row = {
            "factor": factor_name,
            "first": first,
            "second": second,
            "n_blocks": len(x),
            "first_mean_l2": np.mean(x),
            "second_mean_l2": np.mean(y),
            "mean_difference_first_minus_second": np.mean(d),
            "median_difference_first_minus_second": np.median(d),
            "percent_reduction_second_vs_first": percent_reduction(x, y),
            "rank_biserial_first_minus_second": rank_biserial(x, y),
            "wilcoxon_statistic": float(stat.statistic),
            "p_raw": float(stat.pvalue),
        }
        rows.append(row)
        pvals.append(float(stat.pvalue))

    adjusted = holm_adjust(pvals)
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = float(p_holm)
        row["significant_holm_0.05"] = bool(p_holm < 0.05)

    return rows


def build_architecture_blocks(df):
    """
    15 matched blocks:
      each row = one (sampling, holdout) combination
      columns = architectures
    """
    pivot = df.pivot_table(
        index=["sampling", "holdout"],
        columns="arch",
        values="mean_l2",
        aggfunc="first",
    )
    return pivot[ARCH_ORDER].dropna()


def build_sampling_blocks(df):
    """
    15 matched blocks:
      each row = one (arch, holdout) combination
      columns = sampling strategies
    """
    pivot = df.pivot_table(
        index=["arch", "holdout"],
        columns="sampling",
        values="mean_l2",
        aggfunc="first",
    )
    return pivot[SAMPLING_ORDER].dropna()


def save_paired_architecture_plot(blocks):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.arange(len(ARCH_ORDER))

    for _, row in blocks.iterrows():
        ax.plot(
            x,
            [row[a] for a in ARCH_ORDER],
            marker="o",
            linewidth=1,
            alpha=0.45,
        )

    means = blocks.mean(axis=0)
    ax.plot(
        x,
        [means[a] for a in ARCH_ORDER],
        marker="o",
        linewidth=3,
        label="Mean across 15 matched cells",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCH_ORDER])
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Paired effect of architectural inductive bias")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "paired_architecture_effect.png", dpi=300)
    plt.close(fig)


def save_paired_sampling_plot(blocks):
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    x = np.arange(len(SAMPLING_ORDER))

    for _, row in blocks.iterrows():
        ax.plot(
            x,
            [row[s] for s in SAMPLING_ORDER],
            marker="o",
            linewidth=1,
            alpha=0.45,
        )

    means = blocks.mean(axis=0)
    ax.plot(
        x,
        [means[s] for s in SAMPLING_ORDER],
        marker="o",
        linewidth=3,
        label="Mean across 15 matched cells",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SAMPLING_LABELS[s] for s in SAMPLING_ORDER],
        rotation=10,
        ha="right",
    )
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Paired effect of sampling strategy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "paired_sampling_effect.png", dpi=300)
    plt.close(fig)


def main():
    df = pd.read_csv(MASTER)

    arch_blocks = build_architecture_blocks(df)
    sampling_blocks = build_sampling_blocks(df)

    if len(arch_blocks) != 15:
        raise RuntimeError(
            f"Expected 15 architecture blocks, found {len(arch_blocks)}"
        )
    if len(sampling_blocks) != 15:
        raise RuntimeError(
            f"Expected 15 sampling blocks, found {len(sampling_blocks)}"
        )

    arch_friedman = friedmanchisquare(
        *[arch_blocks[a].to_numpy() for a in ARCH_ORDER]
    )
    sampling_friedman = friedmanchisquare(
        *[sampling_blocks[s].to_numpy() for s in SAMPLING_ORDER]
    )

    global_rows = [
        {
            "factor": "architecture",
            "test": "Friedman",
            "n_blocks": len(arch_blocks),
            "statistic": float(arch_friedman.statistic),
            "p_value": float(arch_friedman.pvalue),
        },
        {
            "factor": "sampling",
            "test": "Friedman",
            "n_blocks": len(sampling_blocks),
            "statistic": float(sampling_friedman.statistic),
            "p_value": float(sampling_friedman.pvalue),
        },
    ]
    pd.DataFrame(global_rows).to_csv(
        OUTDIR / "stats_global.csv", index=False
    )

    pairwise_rows = []
    pairwise_rows.extend(
        pairwise_tests(arch_blocks, ARCH_ORDER, "architecture")
    )
    pairwise_rows.extend(
        pairwise_tests(sampling_blocks, SAMPLING_ORDER, "sampling")
    )
    pd.DataFrame(pairwise_rows).to_csv(
        OUTDIR / "stats_pairwise.csv", index=False
    )

    save_paired_architecture_plot(arch_blocks)
    save_paired_sampling_plot(sampling_blocks)

    print("=== GLOBAL TESTS ===")
    for row in global_rows:
        print(
            f"{row['factor']:12s} | Friedman statistic={row['statistic']:.6f} "
            f"| p={row['p_value']:.6g}"
        )

    print("\n=== PAIRWISE TESTS (Holm corrected) ===")
    for row in pairwise_rows:
        print(
            f"{row['factor']:12s} | {row['first']} vs {row['second']} "
            f"| reduction={row['percent_reduction_second_vs_first']:.2f}% "
            f"| r_rb={row['rank_biserial_first_minus_second']:.3f} "
            f"| p_raw={row['p_raw']:.6g} "
            f"| p_holm={row['p_holm']:.6g}"
        )

    print("\nSaved:")
    print(" - figures/paired_architecture_effect.png")
    print(" - figures/paired_sampling_effect.png")
    print(" - results/stats_global.csv")
    print(" - results/stats_pairwise.csv")


if __name__ == "__main__":
    main()