"""
Inspect and visualize the generated forward-operator dataset.

This script checks the saved .npz dataset and creates a random-sample figure
showing velocity fields mapped inside their corresponding pipe geometries.

Example
-------
python src/inspect_forward_operator_dataset.py

Optional
--------
python src/inspect_forward_operator_dataset.py --dataset data/forward_operator_dataset.npz --n-samples 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(dataset_path: Path):
    """Load the compressed NumPy dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Generate it first with:\n"
            "python src/generate_forward_operator_dataset.py --n-samples 500"
        )

    return np.load(dataset_path, allow_pickle=True)


def print_dataset_summary(data) -> None:
    """Print dataset keys, shapes, and numerical ranges."""
    print("Dataset summary")
    print("---------------")

    print("Keys:")
    for key in data.files:
        print(f"  {key}")

    print("")
    print("Shapes:")
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
        if key in data.files:
            print(f"  {key:20s}: {data[key].shape}")

    print("")
    print("Ranges:")

    flow_rates = data["flow_rates"]
    pressure_drops = data["pressure_drops"]
    velocity_fields = data["velocity_fields"]
    radius_profiles = data["radius_profiles"]

    print(f"  Q:                 {flow_rates.min():.6e} to {flow_rates.max():.6e}")
    print(
        f"  pressure_drop:     "
        f"{pressure_drops.min():.6e} to {pressure_drops.max():.6e}"
    )
    print(
        f"  velocity:          "
        f"{velocity_fields.min():.6e} to {velocity_fields.max():.6e}"
    )
    print(
        f"  radius:            "
        f"{radius_profiles.min():.6e} to {radius_profiles.max():.6e}"
    )

    print("")
    print("Case counts:")

    geometry_names = data["geometry_names"]
    fluid_models = data["fluid_models"]

    geometry_codes = data["geometry_codes"]
    fluid_codes = data["fluid_codes"]

    for idx, name in enumerate(geometry_names):
        count = int(np.sum(geometry_codes == idx))
        print(f"  geometry {str(name):24s}: {count}")

    for idx, name in enumerate(fluid_models):
        count = int(np.sum(fluid_codes == idx))
        print(f"  fluid    {str(name):24s}: {count}")


def map_velocity_to_physical_grid(
    z,
    eta,
    radius,
    velocity,
    n_y: int = 220,
):
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

    velocity_grid = np.ma.masked_invalid(velocity_grid)

    return z_grid, y_grid, velocity_grid


def plot_random_samples(
    data,
    n_samples: int,
    seed: int,
    output_path: Path,
) -> None:
    """Plot random dataset samples as geometry-mapped velocity fields."""
    rng = np.random.default_rng(seed)

    total_samples = data["velocity_fields"].shape[0]
    n_samples = min(n_samples, total_samples)

    selected_indices = rng.choice(total_samples, size=n_samples, replace=False)

    z = data["z"]
    eta = data["eta"]

    radius_profiles = data["radius_profiles"]
    velocity_fields = data["velocity_fields"]
    flow_rates = data["flow_rates"]
    pressure_drops = data["pressure_drops"]
    geometry_codes = data["geometry_codes"]
    fluid_codes = data["fluid_codes"]
    geometry_names = data["geometry_names"]
    fluid_models = data["fluid_models"]

    n_cols = 2
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11, 3.2 * n_rows),
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()

    global_umax = velocity_fields[selected_indices].max()

    contour = None

    for ax_idx, sample_idx in enumerate(selected_indices):
        ax = axes[ax_idx]

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

        geometry_name = str(geometry_names[geometry_codes[sample_idx]])
        fluid_model = str(fluid_models[fluid_codes[sample_idx]])

        ax.set_title(
            f"Sample {sample_idx} | {geometry_name} | {fluid_model}\n"
            f"Delta P = {pressure_drops[sample_idx]:.3e}, "
            f"Q = {flow_rates[sample_idx]:.3e}, "
            f"u_max = {velocity.max():.3e}",
            fontsize=9,
        )

        ax.set_xlabel(r"$z/L$")
        ax.set_ylabel(r"$r/R_0$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect(0.35)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for empty_ax in axes[n_samples:]:
        empty_ax.axis("off")

    if contour is not None:
        fig.colorbar(
            contour,
            ax=axes,
            shrink=0.92,
            label=r"Axial velocity, $u(r,z)$",
        )

    fig.suptitle(
        "Random samples from the forward-operator dataset",
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("")
    print("Saved random-sample figures:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect and visualize the forward-operator dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/forward_operator_dataset.npz",
        help="Path to the generated .npz dataset.",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=6,
        help="Number of random samples to plot.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for choosing samples.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="figures/forward_dataset_random_samples",
        help="Output figure path without extension.",
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    data = load_dataset(dataset_path)
    print_dataset_summary(data)

    plot_random_samples(
        data=data,
        n_samples=args.n_samples,
        seed=args.seed,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
