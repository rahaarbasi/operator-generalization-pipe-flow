#!/usr/bin/env python3
"""
Build paper figures directly from results/master_table.csv.

Outputs:
  figures/cross_grid_straight.png
  figures/cross_grid_stenosed.png
  figures/cross_grid_expanded.png
  figures/cross_grid_sinusoidal.png
  figures/cross_grid_hyperbolic.png
  figures/main_effect_architecture.png
  figures/main_effect_sampling.png
  figures/negative_prediction_rate.png
  results/claim_consistency.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MASTER = Path("results/master_table.csv")
ARCH_EFFECT = Path("results/effect_inductive_bias.csv")
SAMPLING_EFFECT = Path("results/effect_sampling.csv")
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

ARCH_ORDER = ["unconstrained", "powerlaw", "geoaware"]
SAMPLING_ORDER = ["uniform", "geo-axial", "geo-wallstretched"]
GEOM_ORDER = ["straight", "stenosed", "expanded", "sinusoidal", "hyperbolic"]

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
GEOM_LABELS = {
    "straight": "Straight",
    "stenosed": "Stenosed",
    "expanded": "Expanded",
    "sinusoidal": "Sinusoidal",
    "hyperbolic": "Hyperbolic constriction",
}

def save_holdout_plot(df, geom):
    sub = df[df["holdout"] == geom].copy()
    x = np.arange(len(ARCH_ORDER))
    offsets = [-0.18, 0.0, 0.18]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for offset, sampling in zip(offsets, SAMPLING_ORDER):
        rows = sub[sub["sampling"] == sampling].set_index("arch").reindex(ARCH_ORDER)
        ax.errorbar(
            x + offset,
            rows["mean_l2"].to_numpy(),
            yerr=rows["mean_l2_seed_std"].to_numpy(),
            marker="o",
            linestyle="none",
            capsize=4,
            label=SAMPLING_LABELS[sampling],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCH_ORDER])
    ax.set_ylabel("Relative L2 error")
    ax.set_title(f"Held-out family: {GEOM_LABELS[geom]}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"cross_grid_{geom}.png", dpi=300)
    plt.close(fig)

def save_main_effect_architecture():
    df = pd.read_csv(ARCH_EFFECT)
    df["arch"] = pd.Categorical(df["arch"], ARCH_ORDER, ordered=True)
    df = df.sort_values("arch")
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.errorbar(
        x,
        df["mean_l2_over_cells"],
        yerr=df["std_across_cells"],
        marker="o",
        linestyle="none",
        capsize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[str(a)] for a in df["arch"]])
    ax.set_ylabel("Mean relative L2 error")
    ax.set_title("Main effect of architectural inductive bias")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "main_effect_architecture.png", dpi=300)
    plt.close(fig)

def save_main_effect_sampling():
    df = pd.read_csv(SAMPLING_EFFECT)
    df["sampling"] = pd.Categorical(df["sampling"], SAMPLING_ORDER, ordered=True)
    df = df.sort_values("sampling")
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(
        x,
        df["mean_l2_over_cells"],
        yerr=df["std_across_cells"],
        marker="o",
        linestyle="none",
        capsize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SAMPLING_LABELS[str(s)] for s in df["sampling"]],
        rotation=10,
        ha="right",
    )
    ax.set_ylabel("Mean relative L2 error")
    ax.set_title("Main effect of sampling strategy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "main_effect_sampling.png", dpi=300)
    plt.close(fig)

def save_negative_prediction_plot(df):
    grouped = (
        df.groupby(["arch", "holdout"], as_index=False)
        .agg(
            negative_rate_mean=("negative_rate_mean", "mean"),
            negative_rate_std=("negative_rate_mean", "std"),
        )
    )

    x = np.arange(len(GEOM_ORDER))
    offsets = [-0.18, 0.0, 0.18]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for offset, arch in zip(offsets, ARCH_ORDER):
        rows = grouped[grouped["arch"] == arch].set_index("holdout").reindex(GEOM_ORDER)
        ax.errorbar(
            x + offset,
            rows["negative_rate_mean"].to_numpy(),
            yerr=rows["negative_rate_std"].fillna(0).to_numpy(),
            marker="o",
            linestyle="none",
            capsize=4,
            label=ARCH_LABELS[arch],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([GEOM_LABELS[g] for g in GEOM_ORDER], rotation=12)
    ax.set_ylabel("Fraction of test cases with negative prediction")
    ax.set_title("Physical admissibility under held-out geometry shift")
    ax.set_ylim(bottom=-0.03)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "negative_prediction_rate.png", dpi=300)
    plt.close(fig)

def save_claim_consistency(df):
    rows = []
    for sampling in SAMPLING_ORDER:
        for geom in GEOM_ORDER:
            sub = df[
                (df["sampling"] == sampling) & (df["holdout"] == geom)
            ].set_index("arch")

            u = float(sub.loc["unconstrained", "mean_l2"])
            p = float(sub.loc["powerlaw", "mean_l2"])
            g = float(sub.loc["geoaware", "mean_l2"])

            rows.append(
                {
                    "sampling": sampling,
                    "holdout": geom,
                    "unconstrained_mean_l2": u,
                    "powerlaw_mean_l2": p,
                    "geoaware_mean_l2": g,
                    "geoaware_better_than_powerlaw": g < p,
                    "powerlaw_better_than_unconstrained": p < u,
                    "strict_order_geo_power_unconstrained": g < p < u,
                    "geoaware_reduction_vs_unconstrained_pct": 100 * (u - g) / u,
                    "geoaware_reduction_vs_powerlaw_pct": 100 * (p - g) / p,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv("results/claim_consistency.csv", index=False)
    n = len(out)
    n_ordered = int(out["strict_order_geo_power_unconstrained"].sum())
    print(f"Strict ordering geoaware < powerlaw < unconstrained: {n_ordered}/{n} cells")

def main():
    df = pd.read_csv(MASTER)
    for geom in GEOM_ORDER:
        save_holdout_plot(df, geom)
    save_main_effect_architecture()
    save_main_effect_sampling()
    save_negative_prediction_plot(df)
    save_claim_consistency(df)

    print("Saved figures and claim_consistency.csv")

if __name__ == "__main__":
    main()