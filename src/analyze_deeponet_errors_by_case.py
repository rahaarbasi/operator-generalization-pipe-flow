"""
Analyze DeepONet validation errors by geometry and fluid model.

This script groups full-field relative L2 errors by:

    geometry type
    fluid model

It helps identify which cases the DeepONet baseline fails on.

Run
---
python src/analyze_deeponet_errors_by_case.py

Outputs
-------
figures/deeponet_error_by_case.png
results/deeponet_error_by_case.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_deeponet_forward_fullfield import (
    build_branch_features,
    get_device,
    load_dataset,
    load_model,
    predict_full_field,
    relative_l2_error,
    train_val_split,
)


def evaluate_errors_by_case(
    dataset_path: Path,
    model_path: Path,
    val_fraction: float = 0.2,
    seed: int = 123,
):
    """Evaluate full-field validation errors and group them by case."""
    device = get_device()
    print(f"Using device: {device}")

    data = load_dataset(dataset_path)
    branch_features = build_branch_features(data)

    _, val_indices = train_val_split(
        n_samples=branch_features.shape[0],
        val_fraction=val_fraction,
        seed=seed,
    )

    model, checkpoint = load_model(model_path, device=device)

    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)

    geometry_names = data["geometry_names"]
    fluid_models = data["fluid_models"]
    geometry_codes = data["geometry_codes"]
    fluid_codes = data["fluid_codes"]

    branch_mean = checkpoint["branch_mean"]
    branch_std = checkpoint["branch_std"]
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])

    rows = []

    for count, sample_idx in enumerate(val_indices, start=1):
        target = velocity_fields[sample_idx]

        prediction = predict_full_field(
            model=model,
            branch_feature=branch_features[sample_idx],
            z=z,
            eta=eta,
            branch_mean=branch_mean,
            branch_std=branch_std,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
        )

        error = relative_l2_error(prediction, target)

        geometry_name = str(geometry_names[geometry_codes[sample_idx]])
        fluid_model = str(fluid_models[fluid_codes[sample_idx]])

        rows.append(
            {
                "sample_idx": int(sample_idx),
                "geometry": geometry_name,
                "fluid_model": fluid_model,
                "relative_l2": float(error),
                "target_max": float(target.max()),
                "prediction_max": float(prediction.max()),
                "prediction_min": float(prediction.min()),
            }
        )

        if count % 10 == 0 or count == len(val_indices):
            print(f"Evaluated {count}/{len(val_indices)} validation samples")

    return rows


def summarize_rows(rows):
    """Create summary statistics for each geometry/fluid pair."""
    case_keys = sorted(
        set((row["geometry"], row["fluid_model"]) for row in rows)
    )

    summary = []

    for geometry, fluid_model in case_keys:
        case_rows = [
            row
            for row in rows
            if row["geometry"] == geometry
            and row["fluid_model"] == fluid_model
        ]

        errors = np.array([row["relative_l2"] for row in case_rows])
        worst_row = max(case_rows, key=lambda row: row["relative_l2"])

        summary.append(
            {
                "geometry": geometry,
                "fluid_model": fluid_model,
                "count": len(case_rows),
                "mean_relative_l2": float(errors.mean()),
                "median_relative_l2": float(np.median(errors)),
                "min_relative_l2": float(errors.min()),
                "max_relative_l2": float(errors.max()),
                "worst_sample_idx": int(worst_row["sample_idx"]),
                "worst_prediction_min": float(worst_row["prediction_min"]),
            }
        )

    return summary


def save_summary_csv(summary, output_path: Path) -> None:
    """Save grouped error summary as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "geometry",
        "fluid_model",
        "count",
        "mean_relative_l2",
        "median_relative_l2",
        "min_relative_l2",
        "max_relative_l2",
        "worst_sample_idx",
        "worst_prediction_min",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved: {output_path}")


def print_summary(summary) -> None:
    """Print grouped error summary."""
    print("")
    print("DeepONet validation error by case")
    print("---------------------------------")

    for row in summary:
        print(
            f"{row['geometry']:24s} | "
            f"{row['fluid_model']:10s} | "
            f"n={row['count']:3d} | "
            f"mean={row['mean_relative_l2']:.3e} | "
            f"median={row['median_relative_l2']:.3e} | "
            f"max={row['max_relative_l2']:.3e} | "
            f"worst sample={row['worst_sample_idx']:4d} | "
            f"pred min={row['worst_prediction_min']:.3e}"
        )


def plot_summary(summary, output_path: Path) -> None:
    """Plot mean and max relative L2 error by case."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        f"{row['geometry']}\n{row['fluid_model']}"
        for row in summary
    ]

    mean_errors = [row["mean_relative_l2"] for row in summary]
    max_errors = [row["max_relative_l2"] for row in summary]

    x = np.arange(len(summary))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)

    ax.bar(x - width / 2, mean_errors, width, label="Mean relative L2")
    ax.bar(x + width / 2, max_errors, width, label="Max relative L2")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("DeepONet validation error grouped by geometry and fluid model")
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    """Run grouped error analysis."""
    rows = evaluate_errors_by_case(
        dataset_path=Path("data/forward_operator_dataset.npz"),
        model_path=Path("models/deeponet_forward_baseline.pt"),
        val_fraction=0.2,
        seed=123,
    )

    summary = summarize_rows(rows)
    print_summary(summary)

    save_summary_csv(
        summary=summary,
        output_path=Path("results/deeponet_error_by_case.csv"),
    )

    plot_summary(
        summary=summary,
        output_path=Path("figures/deeponet_error_by_case.png"),
    )


if __name__ == "__main__":
    main()
