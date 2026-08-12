import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/powerlaw_sampling_ablation_sinusoidal_summary.csv")

label_map = {
    "uniform": "uniform",
    "eta_only_beta": "eta-only beta",
    "z_only": "z-only",
    "z30_eta25_beta": "z30 + beta eta25",
    "z100_eta50_beta_no_uniform": "z100 + beta eta50",
    "z_equal_eta_wallstretched_k2": "z equal + wall-stretched eta",
}

df["label"] = df["sampling"].map(label_map)

os.makedirs("figures", exist_ok=True)

metrics = [
    ("mean_l2", "Mean relative L2 error"),
    ("median_l2", "Median relative L2 error"),
    ("max_l2", "Maximum relative L2 error"),
]

for metric, ylabel in metrics:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(df["label"], df[metric])
    ax.set_title("Power-law-aware DeepONet: sampling ablation on held-out sinusoidal")
    ax.set_xlabel("Training sampling strategy")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    path = f"figures/powerlaw_sampling_ablation_sinusoidal_{metric}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)
