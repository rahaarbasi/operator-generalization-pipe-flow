"""
Compare fixed-domain geometry-channel DeepONet against previous sampling baselines.

Expected inputs:
  results/powerlaw_sampling_ablation_sinusoidal_summary.csv
  results/leave_one_geometry_out_sinusoidal_domain_channel_uniform_summary.csv
  results/leave_one_geometry_out_sinusoidal_domain_channel_mesh_summary.csv

Outputs:
  results/domain_channel_sampling_comparison_sinusoidal.csv
  figures/domain_channel_sampling_comparison_sinusoidal_mean_l2.png
  figures/domain_channel_sampling_comparison_sinusoidal_median_l2.png
  figures/domain_channel_sampling_comparison_sinusoidal_max_l2.png
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def get_summary_row(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if "fluid_model" in df.columns:
        rows = df[df["fluid_model"] == "all"]
        if rows.empty:
            raise ValueError(f"Could not find fluid_model='all' in {path}")
        row = rows.iloc[0]
        return {
            "case": label,
            "mean_l2": float(row["mean_relative_l2"]),
            "median_l2": float(row["median_relative_l2"]),
            "max_l2": float(row["max_relative_l2"]),
            "negative_predictions": int(row.get("negative_prediction_count", 0)),
        }
    raise ValueError(f"Unexpected summary schema for {path}")


def main() -> None:
    results_dir = Path("results")
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    rows = []

    # Previous fixed-model sampling baselines from the power-law-aware ablation.
    powerlaw_path = results_dir / "powerlaw_sampling_ablation_sinusoidal_summary.csv"
    if not powerlaw_path.exists():
        raise FileNotFoundError(f"Missing required file: {powerlaw_path}")

    powerlaw = pd.read_csv(powerlaw_path)
    baseline_map = {
        "uniform": "Power-law-aware | uniform",
        "z_equal_eta_wallstretched_k2": "Power-law-aware | mesh sampling",
        "z30_eta25_beta": "Power-law-aware | z30 + beta eta25",
    }
    for sampling_key, label in baseline_map.items():
        match = powerlaw[powerlaw["sampling"] == sampling_key]
        if match.empty:
            print(f"Warning: skipping missing sampling key {sampling_key} in {powerlaw_path}")
            continue
        row = match.iloc[0]
        rows.append(
            {
                "case": label,
                "mean_l2": float(row["mean_l2"]),
                "median_l2": float(row["median_l2"]),
                "max_l2": float(row["max_l2"]),
                "negative_predictions": int(row.get("negative_predictions", 0)),
            }
        )

    # New geometry-channel model results.
    rows.append(
        get_summary_row(
            results_dir / "leave_one_geometry_out_sinusoidal_domain_channel_uniform_summary.csv",
            "Geometry-channel-aware | uniform",
        )
    )
    rows.append(
        get_summary_row(
            results_dir / "leave_one_geometry_out_sinusoidal_domain_channel_mesh_summary.csv",
            "Geometry-channel-aware | mesh sampling",
        )
    )

    df = pd.DataFrame(rows)
    out_csv = results_dir / "domain_channel_sampling_comparison_sinusoidal.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    metric_specs = [
        ("mean_l2", "Mean relative L2 error"),
        ("median_l2", "Median relative L2 error"),
        ("max_l2", "Maximum relative L2 error"),
    ]

    for metric, ylabel in metric_specs:
        fig, ax = plt.subplots(figsize=(11, 5.2))
        ax.bar(df["case"], df[metric])
        ax.set_title("Held-out sinusoidal: geometry channels vs sampling strategy")
        ax.set_xlabel("Model and training sampling")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out_png = figures_dir / f"domain_channel_sampling_comparison_sinusoidal_{metric}.png"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_png}")
        plt.close(fig)


if __name__ == "__main__":
    main()
