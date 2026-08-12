"""
Aggregate and plot leave-one-geometry-out DeepONet results.

This script reads all files matching:

    results/leave_one_geometry_out_*_summary.csv

and creates:

    results/leave_one_geometry_out_aggregate.csv
    figures/leave_one_geometry_out_mean_error_by_geometry.png
    figures/leave_one_geometry_out_median_error_by_geometry.png
    figures/leave_one_geometry_out_max_error_by_geometry.png
    figures/leave_one_geometry_out_negative_predictions_by_geometry.png
    figures/leave_one_geometry_out_mean_error_by_geometry_and_fluid.png

Run from the repository root:

    python src/aggregate_leave_one_geometry_out.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GEOMETRY_ORDER = [
    "straight",
    "stenosed",
    "expanded",
    "sinusoidal",
    "hyperbolic_constriction",
]

MODEL_ORDER = [
    "Unconstrained",
    "Power-law-aware",
    "Geometry-aware",
]


def extract_held_out_geometry(path: Path) -> str:
    """Extract held-out geometry name from summary CSV filename."""
    prefix = "leave_one_geometry_out_"
    suffix = "_summary.csv"

    name = path.name

    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected filename format: {name}")

    return name[len(prefix):-len(suffix)]


def load_summary_files(results_dir: Path) -> pd.DataFrame:
    """Load all leave-one-geometry-out summary CSV files."""
    files = sorted(results_dir.glob("leave_one_geometry_out_*_summary.csv"))

    if not files:
        raise FileNotFoundError(
            "No leave-one-geometry-out summary files found.\n"
            "Expected files like:\n"
            "results/leave_one_geometry_out_sinusoidal_summary.csv"
        )

    frames = []

    for path in files:
        held_out_geometry = extract_held_out_geometry(path)
        frame = pd.read_csv(path)
        frame.insert(0, "held_out_geometry", held_out_geometry)
        frames.append(frame)

    aggregate = pd.concat(frames, ignore_index=True)

    return aggregate


def keep_model_and_geometry_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply stable categorical ordering for plotting and tables."""
    frame = frame.copy()

    frame["held_out_geometry"] = pd.Categorical(
        frame["held_out_geometry"],
        categories=GEOMETRY_ORDER,
        ordered=True,
    )

    frame["model"] = pd.Categorical(
        frame["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    frame = frame.sort_values(
        ["held_out_geometry", "model", "fluid_model"]
    ).reset_index(drop=True)

    return frame


def print_overall_table(frame: pd.DataFrame) -> None:
    """Print the all-fluid comparison table."""
    all_rows = frame[frame["fluid_model"] == "all"].copy()
    all_rows = keep_model_and_geometry_order(all_rows)

    columns = [
        "held_out_geometry",
        "model",
        "count",
        "mean_relative_l2",
        "median_relative_l2",
        "max_relative_l2",
        "negative_prediction_count",
    ]

    print("")
    print("Leave-one-geometry-out aggregate summary")
    print("----------------------------------------")
    print(all_rows[columns].to_string(index=False))


def plot_metric_by_geometry(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot grouped bars for one metric by held-out geometry and model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = frame[frame["fluid_model"] == "all"].copy()
    all_rows = keep_model_and_geometry_order(all_rows)

    pivot = all_rows.pivot(
        index="held_out_geometry",
        columns="model",
        values=metric,
    )

    present_geometries = [g for g in GEOMETRY_ORDER if g in pivot.index]
    present_models = [m for m in MODEL_ORDER if m in pivot.columns]
    pivot = pivot.loc[present_geometries, present_models]

    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    for i, model in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[model].values, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str), rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_metric_by_fluid(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Plot metric split by fluid model.

    This creates one figure with grouped bars where each x-label combines
    held-out geometry and fluid model.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = frame[frame["fluid_model"] != "all"].copy()
    rows = keep_model_and_geometry_order(rows)

    fluid_values = sorted(rows["fluid_model"].dropna().astype(str).unique())

    x_labels = []
    for geometry in GEOMETRY_ORDER:
        for fluid in fluid_values:
            mask = (
                (rows["held_out_geometry"].astype(str) == geometry)
                & (rows["fluid_model"].astype(str) == fluid)
            )
            if mask.any():
                x_labels.append(f"{geometry}\n{fluid}")

    x = np.arange(len(x_labels))
    width = 0.8 / len(MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    for i, model in enumerate(MODEL_ORDER):
        values = []

        for label in x_labels:
            geometry, fluid = label.split("\n")

            mask = (
                (rows["held_out_geometry"].astype(str) == geometry)
                & (rows["fluid_model"].astype(str) == fluid)
                & (rows["model"].astype(str) == model)
            )

            if mask.any():
                values.append(float(rows.loc[mask, metric].iloc[0]))
            else:
                values.append(np.nan)

        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    """Aggregate CSV files and create comparison figures."""
    results_dir = Path("results")
    figures_dir = Path("figures")

    aggregate = load_summary_files(results_dir)
    aggregate = keep_model_and_geometry_order(aggregate)

    output_csv = results_dir / "leave_one_geometry_out_aggregate.csv"
    aggregate.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    print_overall_table(aggregate)

    plot_metric_by_geometry(
        frame=aggregate,
        metric="mean_relative_l2",
        ylabel="Mean relative L2 error",
        title="Leave-one-geometry-out mean error by held-out geometry",
        output_path=figures_dir / "leave_one_geometry_out_mean_error_by_geometry.png",
    )

    plot_metric_by_geometry(
        frame=aggregate,
        metric="median_relative_l2",
        ylabel="Median relative L2 error",
        title="Leave-one-geometry-out median error by held-out geometry",
        output_path=figures_dir / "leave_one_geometry_out_median_error_by_geometry.png",
    )

    plot_metric_by_geometry(
        frame=aggregate,
        metric="max_relative_l2",
        ylabel="Max relative L2 error",
        title="Leave-one-geometry-out worst-case error by held-out geometry",
        output_path=figures_dir / "leave_one_geometry_out_max_error_by_geometry.png",
    )

    plot_metric_by_geometry(
        frame=aggregate,
        metric="negative_prediction_count",
        ylabel="Negative-prediction count",
        title="Nonphysical negative predictions under geometry extrapolation",
        output_path=figures_dir / "leave_one_geometry_out_negative_predictions_by_geometry.png",
    )

    plot_metric_by_fluid(
        frame=aggregate,
        metric="mean_relative_l2",
        ylabel="Mean relative L2 error",
        title="Leave-one-geometry-out mean error by geometry and fluid",
        output_path=figures_dir / "leave_one_geometry_out_mean_error_by_geometry_and_fluid.png",
    )


if __name__ == "__main__":
    main()
