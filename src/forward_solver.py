"""
Forward solvers for geometry-varying pipe flows.

This module builds a lightweight physics-based forward operator for the
forward-operator project.

Given a radius profile R(z), a pressure drop, and fluid parameters, the solver
computes:

- volumetric flow rate Q
- local pressure gradient dp/dz
- axial velocity field u(eta, z)

where eta = r / R(z) is the normalized radial coordinate.

The solver uses a lubrication / locally fully-developed approximation, so it is
best suited for smooth, slowly varying pipe geometries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from geometries import make_geometry


@dataclass
class ForwardSolution:
    """Container for a forward pipe-flow solution."""

    z: np.ndarray
    eta: np.ndarray
    radius: np.ndarray
    velocity: np.ndarray
    pressure_gradient: np.ndarray
    flow_rate: float
    pressure_drop: float
    fluid_model: str


def make_solver_grid(
    n_z: int = 128,
    n_eta: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create normalized axial and radial grids.

    z is the normalized axial coordinate in [0, 1].
    eta is the normalized radial coordinate eta = r / R(z), in [0, 1].
    """
    if n_z < 2:
        raise ValueError("n_z must be at least 2.")

    if n_eta < 2:
        raise ValueError("n_eta must be at least 2.")

    z = np.linspace(0.0, 1.0, n_z)
    eta = np.linspace(0.0, 1.0, n_eta)

    return z, eta


def _validate_radius(radius: np.ndarray) -> None:
    """Check that the radius profile is physically valid."""
    if np.any(radius <= 0.0):
        raise ValueError("Radius profile contains non-positive values.")


def solve_newtonian_forward(
    z: np.ndarray,
    eta: np.ndarray,
    radius: np.ndarray,
    viscosity: float = 1.0,
    pressure_drop: float = 1.0,
    length: float = 1.0,
) -> ForwardSolution:
    """
    Solve Newtonian pressure-driven flow in a smooth variable-radius pipe.

    For a slowly varying pipe, the local relation is

        dp/dz = -8 mu Q / (pi R(z)^4)

    and the local velocity profile is approximated by the Poiseuille profile

        u(eta, z) = 2 Q / (pi R(z)^2) * (1 - eta^2)

    Parameters
    ----------
    z:
        Normalized axial coordinate in [0, 1].
    eta:
        Normalized radial coordinate r / R(z).
    radius:
        Radius profile R(z).
    viscosity:
        Dynamic viscosity mu.
    pressure_drop:
        Total imposed pressure drop Delta P.
    length:
        Physical pipe length. Use 1.0 for nondimensional experiments.

    Returns
    -------
    ForwardSolution
        Newtonian forward solution.
    """
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive.")

    if pressure_drop <= 0.0:
        raise ValueError("pressure_drop must be positive.")

    if length <= 0.0:
        raise ValueError("length must be positive.")

    _validate_radius(radius)

    resistance_integrand = 8.0 * viscosity / (np.pi * radius**4)
    resistance = length * np.trapezoid(resistance_integrand, z)

    flow_rate = pressure_drop / resistance

    local_pressure_gradient = -8.0 * viscosity * flow_rate / (
        np.pi * radius**4
    )

    velocity = np.zeros((len(z), len(eta)))

    for i, r_z in enumerate(radius):
        centerline_factor = 2.0 * flow_rate / (np.pi * r_z**2)
        velocity[i, :] = centerline_factor * (1.0 - eta**2)

    return ForwardSolution(
        z=z,
        eta=eta,
        radius=radius,
        velocity=velocity,
        pressure_gradient=local_pressure_gradient,
        flow_rate=float(flow_rate),
        pressure_drop=float(pressure_drop),
        fluid_model="newtonian",
    )


def solve_power_law_forward(
    z: np.ndarray,
    eta: np.ndarray,
    radius: np.ndarray,
    consistency: float = 1.0,
    flow_index: float = 0.7,
    pressure_drop: float = 1.0,
    length: float = 1.0,
) -> ForwardSolution:
    """
    Solve power-law pressure-driven flow in a smooth variable-radius pipe.

    The power-law model is

        tau = K * gamma_dot^n

    where K is the consistency index and n is the flow behavior index.

    For a circular pipe, the local flow-rate relation is

        Q = [pi n / (3n + 1)] * [G / (2K)]^(1/n) * R^(3 + 1/n)

    where G = -dp/dz > 0.

    For a variable-radius pipe under the lubrication approximation, Q is constant
    along z, and the required local pressure gradient varies with R(z).

    Parameters
    ----------
    z:
        Normalized axial coordinate in [0, 1].
    eta:
        Normalized radial coordinate r / R(z).
    radius:
        Radius profile R(z).
    consistency:
        Power-law consistency index K.
    flow_index:
        Power-law flow behavior index n.
        n < 1 means shear-thinning.
        n = 1 recovers Newtonian behavior if K = mu.
    pressure_drop:
        Total imposed pressure drop Delta P.
    length:
        Physical pipe length. Use 1.0 for nondimensional experiments.

    Returns
    -------
    ForwardSolution
        Power-law forward solution.
    """
    if consistency <= 0.0:
        raise ValueError("consistency must be positive.")

    if flow_index <= 0.0:
        raise ValueError("flow_index must be positive.")

    if pressure_drop <= 0.0:
        raise ValueError("pressure_drop must be positive.")

    if length <= 0.0:
        raise ValueError("length must be positive.")

    _validate_radius(radius)

    n = flow_index
    k = consistency

    coefficient = 2.0 * k * ((3.0 * n + 1.0) / (np.pi * n)) ** n
    resistance_integrand = coefficient / (radius ** (3.0 * n + 1.0))
    resistance = length * np.trapezoid(resistance_integrand, z)

    flow_rate = (pressure_drop / resistance) ** (1.0 / n)

    positive_pressure_gradient = (
        coefficient * flow_rate**n / (radius ** (3.0 * n + 1.0))
    )

    local_pressure_gradient = -positive_pressure_gradient

    velocity = np.zeros((len(z), len(eta)))

    for i, r_z in enumerate(radius):
        g_z = positive_pressure_gradient[i]

        prefactor = (n / (n + 1.0)) * (g_z / (2.0 * k)) ** (1.0 / n)
        radial_shape = r_z ** (1.0 + 1.0 / n) * (
            1.0 - eta ** (1.0 + 1.0 / n)
        )

        velocity[i, :] = prefactor * radial_shape

    return ForwardSolution(
        z=z,
        eta=eta,
        radius=radius,
        velocity=velocity,
        pressure_gradient=local_pressure_gradient,
        flow_rate=float(flow_rate),
        pressure_drop=float(pressure_drop),
        fluid_model="power_law",
    )


def solve_geometry_case(
    geometry_name: str,
    fluid_model: str = "newtonian",
    n_z: int = 128,
    n_eta: int = 64,
    pressure_drop: float = 1.0,
    length: float = 1.0,
    geometry_kwargs: dict | None = None,
    fluid_kwargs: dict | None = None,
) -> ForwardSolution:
    """
    Convenience wrapper for solving one named geometry case.

    Example
    -------
    solution = solve_geometry_case(
        geometry_name="stenosed",
        fluid_model="power_law",
        geometry_kwargs={"severity": 0.35},
        fluid_kwargs={"consistency": 1.0, "flow_index": 0.7},
    )
    """
    geometry_kwargs = geometry_kwargs or {}
    fluid_kwargs = fluid_kwargs or {}

    z, eta = make_solver_grid(n_z=n_z, n_eta=n_eta)
    radius = make_geometry(geometry_name, z, **geometry_kwargs)

    if fluid_model == "newtonian":
        return solve_newtonian_forward(
            z=z,
            eta=eta,
            radius=radius,
            pressure_drop=pressure_drop,
            length=length,
            **fluid_kwargs,
        )

    if fluid_model == "power_law":
        return solve_power_law_forward(
            z=z,
            eta=eta,
            radius=radius,
            pressure_drop=pressure_drop,
            length=length,
            **fluid_kwargs,
        )

    raise ValueError(
        f"Unknown fluid_model '{fluid_model}'. "
        "Available options: 'newtonian', 'power_law'."
    )


if __name__ == "__main__":
    cases = [
        ("straight", "newtonian", {}, {"viscosity": 1.0}),
        ("stenosed", "newtonian", {"severity": 0.35}, {"viscosity": 1.0}),
        (
            "stenosed",
            "power_law",
            {"severity": 0.35},
            {"consistency": 1.0, "flow_index": 0.7},
        ),
        (
            "hyperbolic_constriction",
            "power_law",
            {"severity": 0.30},
            {"consistency": 1.0, "flow_index": 0.7},
        ),
    ]

    for geometry_name, fluid_model, geometry_kwargs, fluid_kwargs in cases:
        solution = solve_geometry_case(
            geometry_name=geometry_name,
            fluid_model=fluid_model,
            geometry_kwargs=geometry_kwargs,
            fluid_kwargs=fluid_kwargs,
        )

        print(
            f"{geometry_name:24s} | "
            f"{fluid_model:10s} | "
            f"Q = {solution.flow_rate:.6e} | "
            f"u_max = {solution.velocity.max():.6e}"
        )