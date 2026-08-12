"""
Plot forward solutions for all geometry/fluid combinations.

This script generates two figures with 10 cases:

    5 geometries x 2 fluid models = 10 forward solutions

Figures:
1. normalized_eta_velocity_fields_10cases.png/pdf
2. geometry_mapped_velocity_fields_10cases.png/pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forward_solver import solve_geometry_case


def get_geometry_cases():
    """Return the geometry cases used in the forward-operator benchmark."""
    return [
        {
            "geometry_label": "Straight pipe",
            "geometry_name": "straight",
            "geometry_kwargs": {},
        },
        {
            "geometry_label": "Stenosed pipe",
            "geometry_name": "stenosed",
            "geometry_kwargs": {"severity": 0.35},
        },
        {
            "geometry_label": "Expanded pipe",
            "geometry_name": "expanded",
            "geometry_kwargs": {"expansion": 0.30},
        },
        {
            "geometry_label": "Sinusoidal pipe",
            "geometry_name": "sinusoidal",
            "geometry_kwargs": {"amplitude": 0.15, "modes": 1},
        },
        {
            "geometry_label": "Hyperbolic constriction",
            "geometry_name": "hyperbolic_constriction",
            "geometry_kwargs": {"severity": 0.30},
        },
    ]


def get_fluid_cases():
    """Return the fluid models used for each geometry."""
    return [
        {
            "fluid_label": "Newtonian",
            "fluid_model": "newtonian",
            "fluid_kwargs": {"viscosity": 1.0},
        },
        {
            "fluid_label": "Power-law",
            "fluid_model": "power_law",
            "fluid_kwargs": {"consistency": 1.0, "flow_index": 0.7},
        },
    ]


def compute_all_solutions():
    """Solve all combinations of geometry and fluid model."""
    geometry_cases = get_geometry_cases()
    fluid_cases = get_fluid_cases()

    solutions = []

    for geometry_case in geometry_cases:
        for fluid_case in fluid_cases:
            solution = solve_geometry_case(
                geometry_name=geometry_case["geometry_name"],
                fluid_model=fluid_case["fluid_model"],
                n_z=180,
                n_eta=90,
                pressure_drop=1.0,
                length=1.0,
                geometry_kwargs=geometry_case["geometry_kwargs"],
                fluid_kwargs=fluid_case["fluid_kwargs"],
            )

            solutions.append(
                {
                    "geometry_label": geometry_case["geometry_label"],
                    "geometry_name": geometry_case["geometry_name"],
                    "fluid_label": fluid_case["fluid_label"],
                    "fluid_model": fluid_case["fluid_model"],
                    "solution": solution,
                }
            )

    return solutions


def get_solution_by_case(solutions, geometry_label, fluid_label):
    """Find a solution by geometry label and fluid label."""
    for item in solutions:
        if (
            item["geometry_label"] == geometry_label
            and item["fluid_label"] == fluid_label
        ):
            return item

    raise ValueError(
        f"Could not find solution for geometry={geometry_label}, "
        f"fluid={fluid_label}"
    )


def plot_normalized_eta_fields(solutions, output_dir):
    """
    Plot u(eta, z) for all 10 geometry/fluid cases.

    Rows correspond to geometries.
    Columns correspond to fluid models.
    """
    geometry_cases = get_geometry_cases()
    fluid_cases = get_fluid_cases()

    global_umax = max(item["solution"].velocity.max() for item in solutions)

    fig, axes = plt.subplots(
        len(geometry_cases),
        len(fluid_cases),
        figsize=(10, 14),
        constrained_layout=True,
    )

    contour = None

    for row, geometry_case in enumerate(geometry_cases):
        for col, fluid_case in enumerate(fluid_cases):
            ax = axes[row, col]

            item = get_solution_by_case(
                solutions,
                geometry_case["geometry_label"],
                fluid_case["fluid_label"],
            )

            solution = item["solution"]

            z_grid, eta_grid = np.meshgrid(solution.z, solution.eta)

            contour = ax.contourf(
                z_grid,
                eta_grid,
                solution.velocity.T,
                levels=50,
                vmin=0.0,
                vmax=global_umax,
            )

            ax.set_title(
                f"{item['geometry_label']} | {item['fluid_label']}\n"
                f"Q = {solution.flow_rate:.3e}, "
                f"u_max = {solution.velocity.max():.3e}",
                fontsize=9,
            )

            ax.set_xlabel(r"$z/L$")
            ax.set_ylabel(r"$\eta = r/R(z)$")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.colorbar(
        contour,
        ax=axes,
        shrink=0.92,
        label=r"Axial velocity, $u(\eta,z)$",
    )

    fig.suptitle(
        "Forward solutions in normalized computational coordinates",
        fontsize=14,
    )

    png_path = output_dir / "normalized_eta_velocity_fields_10cases.png"
    pdf_path = output_dir / "normalized_eta_velocity_fields_10cases.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def map_velocity_to_physical_grid(solution, n_y=240):
    """
    Map u(eta, z) to u(r, z) inside the actual pipe geometry.

    The solver stores u on eta in [0, 1]. For visualization, we create
    a physical radial grid y = r/R0 and mask points outside the pipe wall.
    """
    z = solution.z
    radius = solution.radius

    y_max = 1.10 * radius.max()
    y = np.linspace(-y_max, y_max, n_y)

    z_grid, y_grid = np.meshgrid(z, y)
    velocity_grid = np.full_like(z_grid, np.nan, dtype=float)

    for i, r_z in enumerate(radius):
        inside = np.abs(y) <= r_z
        eta_values = np.abs(y[inside]) / r_z

        velocity_grid[inside, i] = np.interp(
            eta_values,
            solution.eta,
            solution.velocity[i, :],
        )

    velocity_grid = np.ma.masked_invalid(velocity_grid)

    return z_grid, y_grid, velocity_grid


def plot_geometry_mapped_fields(solutions, output_dir):
    """
    Plot u(r, z) inside each geometry for all 10 geometry/fluid cases.

    Rows correspond to geometries.
    Columns correspond to fluid models.
    """
    geometry_cases = get_geometry_cases()
    fluid_cases = get_fluid_cases()

    global_umax = max(item["solution"].velocity.max() for item in solutions)

    fig, axes = plt.subplots(
        len(geometry_cases),
        len(fluid_cases),
        figsize=(10, 14),
        constrained_layout=True,
    )

    contour = None

    for row, geometry_case in enumerate(geometry_cases):
        for col, fluid_case in enumerate(fluid_cases):
            ax = axes[row, col]

            item = get_solution_by_case(
                solutions,
                geometry_case["geometry_label"],
                fluid_case["fluid_label"],
            )

            solution = item["solution"]

            z_grid, y_grid, velocity_grid = map_velocity_to_physical_grid(solution)

            contour = ax.contourf(
                z_grid,
                y_grid,
                velocity_grid,
                levels=50,
                vmin=0.0,
                vmax=global_umax,
            )

            ax.plot(solution.z, solution.radius, linewidth=1.6)
            ax.plot(solution.z, -solution.radius, linewidth=1.6)
            ax.axhline(0.0, linestyle="--", linewidth=0.7)

            ax.set_title(
                f"{item['geometry_label']} | {item['fluid_label']}\n"
                f"Q = {solution.flow_rate:.3e}, "
                f"u_max = {solution.velocity.max():.3e}",
                fontsize=9,
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
        "Velocity fields mapped inside variable-radius pipe geometries",
        fontsize=14,
    )

    png_path = output_dir / "geometry_mapped_velocity_fields_10cases.png"
    pdf_path = output_dir / "geometry_mapped_velocity_fields_10cases.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    """Generate both 10-case forward-solution figures."""
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    solutions = compute_all_solutions()

    plot_normalized_eta_fields(solutions, output_dir)
    plot_geometry_mapped_fields(solutions, output_dir)


if __name__ == "__main__":
    main()
