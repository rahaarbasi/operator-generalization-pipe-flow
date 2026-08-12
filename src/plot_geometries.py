"""
Plot parametric pipe geometries for the forward-operator project.

This script generates a clean figure showing the radius profiles used for
geometry-generalizable pipe-flow surrogate modeling.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geometries import sample_default_geometries


def plot_pipe_geometries() -> None:
    """Generate and save a multi-panel geometry figure."""
    z = np.linspace(0.0, 1.0, 500)
    geometries = sample_default_geometries(z)

    title_map = {
        "straight": "Straight pipe",
        "stenosed": "Stenosed pipe",
        "expanded": "Expanded pipe",
        "sinusoidal": "Sinusoidal pipe",
        "hyperbolic_constriction": "Hyperbolic constriction",
    }

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(
        1,
        len(geometries),
        figsize=(16, 3.6),
        sharey=True,
        constrained_layout=True,
    )

    for ax, (name, radius_profile) in zip(axes, geometries.items()):
        upper_wall = radius_profile
        lower_wall = -radius_profile

        ax.fill_between(z, lower_wall, upper_wall, alpha=0.25)
        ax.plot(z, upper_wall, linewidth=2.0)
        ax.plot(z, lower_wall, linewidth=2.0)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)

        ax.set_title(title_map.get(name, name), fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.35, 1.35)
        ax.set_xlabel(r"Normalized axial coordinate, $z/L$")
        ax.set_aspect(0.55)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"Normalized radius, $r/R_0$")

    fig.suptitle(
        "Parametric pipe geometries for forward-operator learning",
        fontsize=14,
        y=1.05,
    )

    png_path = output_dir / "pipe_geometry_variations.png"
    pdf_path = output_dir / "pipe_geometry_variations.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    plot_pipe_geometries()