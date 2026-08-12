"""
Stratified inspection for the forward-operator dataset.

This script creates exactly one inspection figure with 10 panels:

    5 geometries x 2 fluid models = 10 representative samples

Output:
    figures/forward_dataset_stratified_10cases.png
    figures/forward_dataset_stratified_10cases.pdf

Run:
    python src/inspect_forward_operator_dataset_stratified.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(dataset_path: Path):
    """Load the generated NumPy dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Generate it first with:\n"
            "python src/generate_forward_operator_dataset.py --n-samples 500"
        )

    return np.load(dataset_path, allow_pickle=True)


def print_dataset_summary(data) -> None:
    """Print basic dataset information."""
    print("Dataset summary")
    print("---------------")

    for key in [
        "z",
        "eta",
        "radius_profiles",
        "velocity_fields",
        "pressure_gradients",
        "flow_rates",
        "pressure_drops",
        "geometry_codes",
        "fluid_codes",
    ]:
        print(f"{key:22s}: {data[key].shape}")

    print("")
    print("Case counts")

    geometry_names = data["geometry_names"]
    fluid_models = data["fluid_models"]
    geometry_codes = data["geometry_codes"]
    fluid_codes = data["fluid_codes"]

    for geometry_idx, geometry_name in enumerate(geometry_names):
        for fluid_idx, fluid_model in enumerate(fluid_models):
            count = int(
                np.sum(
                    (geometry_codes == geometry_idx)
                    & (fluid_codes == fluid_idx)
                )
            )
            print(
                f"{str(geometry_name):24s} | "
                f"{str(fluid_model):10s}: {count}"
            )


def choose_one_sample_per_case(data, seed: int):
    """
    Select exactly one sample for every geometry/fluid combination.

    Returns
    -------
    selected : dict
        Keys are (geometry_idx, fluid_idx), values are sample indices.
    """
    rng = np.random.default_rng(seed)

    geometry_names = data["geometry_names"]
    fluid_models = data["fluid_models"]
    geometry_codes = data["geometry_codes"]
    fluid_codes = data["fluid_codes"]

    selected = {}

    for geometry_idx, geometry_name in enumerate(geometry_names):
        for fluid_idx, fluid_model in enumerate(fluid_models):
            candidates = np.where(
                (geometry_codes == geometry_idx)
                & (fluid_codes == fluid_idx)
            )[0]

            if len(candidates) == 0:
                raise ValueError(
                    f"No dataset samples found for "
                    f"{str(geometry_name)} / {str(fluid_model)}.\n"
                    "Regenerate with more samples, for example:\n"
                    "python src/generate_forward_operator_dataset.py "
                    "--n-samples 500"
                )

            selected[(geometry_idx, fluid_idx)] = int(rng.choice(candidates))

    return selected


def map_velocity_to_physical_grid(z, eta, radius, velocity, n_y: int = 240):
    """
    Map u(eta, z) to u(r, z) inside the physical pipe geometry.

    Points outside the pipe wall are masked.
    """
    y_max = 1.10 * np.max(radius)
    y = np.linspace(-y_max, y_max, n_y)

    z_grid, y_grid = np.meshgrid(z, y)
    velocity_grid = np.full_like(z_grid, np.nan, dtype=float)

    for i, r_z in enumerate(radius):
        inside = np.abs(y) <= r_z
        eta_values = np.abs(y[inside]) / r_z

        velocity_grid[inside, i] = np.interp(
            eta_values,
            eta,
            velocity[i, :],
        )

    return z_grid, y_grid, np.ma.masked_invalid(velocity_grid)


def plot_stratified_10cases(data, seed: int, output_base: Path) -> None:
    """Create the 10-panel stratified inspection figure."""
    z = data["z"]
    eta = data["eta"]

    radius_profiles = data["radius_profiles"]
    velocity_fields = data["velocity_fields"]
    flow_rates = data["flow_rates"]
    pressure_drops = data["pressure_drops"]

    geometry_names = data["geometry_names"]
    fluid_models = data["fluid_models"]

    selected = choose_one_sample_per_case(data, seed=seed)
    selected_indices = list(selected.values())

    global_umax = velocity_fields[selected_indices].max()

    n_rows = len(geometry_names)
    n_cols = len(fluid_models)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, 14),
        constrained_layout=True,
    )

    contour = None

    for geometry_idx, geometry_name in enumerate(geometry_names):
        for fluid_idx, fluid_model in enumerate(fluid_models):
            ax = axes[geometry_idx, fluid_idx]

            sample_idx = selected[(geometry_idx, fluid_idx)]

            radius = radius_profiles[sample_idx]
            velocity = velocity_fields[sample_idx]

            z_grid, y_grid, velocity_grid = map_velocity_to_physical_grid(
                z=z,
                eta=eta,
                radius=radius,
                velocity=velocity,
            )

            contour = ax.contourf(
                z_grid,
                y_grid,
                velocity_grid,
                levels=50,
                vmin=0.0,
                vmax=global_umax,
            )

            ax.plot(z, radius, linewidth=1.5)
            ax.plot(z, -radius, linewidth=1.5)
            ax.axhline(0.0, linestyle="--", linewidth=0.7)

            ax.set_title(
                f"{str(geometry_name)} | {str(fluid_model)} | sample {sample_idx}\n"
                f"Delta P = {pressure_drops[sample_idx]:.3e}, "
                f"Q = {flow_rates[sample_idx]:.3e}, "
                f"u_max = {velocity.max():.3e}",
                fontsize=8.5,
            )

            ax.set_xlabel(r"$z/L$")
            ax.set_ylabel(r"$r/R_0$")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-1.35, 1.35)
            ax.set_aspect(0.35)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.colorbar(
        contour,
        ax=axes,
        shrink=0.92,
        label=r"Axial velocity, $u(r,z)$",
    )

    fig.suptitle(
        "Stratified inspection: one sample per geometry/fluid pair",
        fontsize=14,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("")
    print("Saved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a stratified 10-case dataset inspection figure."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/forward_operator_dataset.npz",
        help="Path to the generated .npz dataset.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for selecting one sample per case.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="figures/forward_dataset_stratified_10cases",
        help="Output path without extension.",
    )

    return parser.parse_args()


def main() -> None:
    """Run dataset inspection."""
    args = parse_args()

    data = load_dataset(Path(args.dataset))
    print_dataset_summary(data)

    plot_stratified_10cases(
        data=data,
        seed=args.seed,
        output_base=Path(args.output),
    )


if __name__ == "__main__":
    main()
