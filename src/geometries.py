"""
Geometry definitions for the forward-operator pipe-flow project.

Each geometry returns a radius profile R(z) over a normalized axial coordinate z in [0, 1].

These are lightweight parametric geometry generators for synthetic forward-operator
datasets. They are not CFD meshes yet.
"""

from __future__ import annotations

from typing import Callable, Dict, Union

import numpy as np


ArrayLike = Union[np.ndarray, float]


def _to_array(z: ArrayLike) -> np.ndarray:
    """Convert scalar or array input to a NumPy array."""
    return np.asarray(z, dtype=float)


def _check_positive_radius(radius_profile: np.ndarray, name: str) -> None:
    """Raise an error if a generated radius profile becomes non-physical."""
    if np.any(radius_profile <= 0.0):
        raise ValueError(
            f"{name} produced non-positive radius values. "
            "Reduce stenosis severity/amplitude."
        )


def straight_pipe(z: ArrayLike, radius: float = 1.0) -> np.ndarray:
    """
    Straight circular pipe.

    Parameters
    ----------
    z:
        Normalized axial coordinate in [0, 1].
    radius:
        Constant pipe radius.

    Returns
    -------
    np.ndarray
        Radius profile R(z).
    """
    z = _to_array(z)
    return radius * np.ones_like(z)


def stenosed_pipe(
    z: ArrayLike,
    radius: float = 1.0,
    severity: float = 0.35,
    center: float = 0.5,
    width: float = 0.12,
) -> np.ndarray:
    """
    Smooth Gaussian stenosis.

    R(z) = R0 * [1 - severity * exp(-0.5 * ((z - center) / width)^2)]

    severity = 0.35 means the radius narrows by 35% at the stenosis center.
    """
    z = _to_array(z)

    if not 0.0 <= severity < 1.0:
        raise ValueError("severity must satisfy 0 <= severity < 1.")

    profile = radius * (
        1.0 - severity * np.exp(-0.5 * ((z - center) / width) ** 2)
    )

    _check_positive_radius(profile, "stenosed_pipe")
    return profile


def expanded_pipe(
    z: ArrayLike,
    radius: float = 1.0,
    expansion: float = 0.30,
    center: float = 0.5,
    width: float = 0.12,
) -> np.ndarray:
    """
    Smooth Gaussian expansion.

    R(z) = R0 * [1 + expansion * exp(-0.5 * ((z - center) / width)^2)]
    """
    z = _to_array(z)

    if expansion < 0.0:
        raise ValueError("expansion must be non-negative.")

    profile = radius * (
        1.0 + expansion * np.exp(-0.5 * ((z - center) / width) ** 2)
    )

    _check_positive_radius(profile, "expanded_pipe")
    return profile


def sinusoidal_pipe(
    z: ArrayLike,
    radius: float = 1.0,
    amplitude: float = 0.15,
    modes: int = 1,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Smooth sinusoidal radius variation.

    R(z) = R0 * [1 + amplitude * sin(2π * modes * z + phase)]
    """
    z = _to_array(z)

    if not 0.0 <= amplitude < 1.0:
        raise ValueError("amplitude must satisfy 0 <= amplitude < 1.")

    profile = radius * (
        1.0 + amplitude * np.sin(2.0 * np.pi * modes * z + phase)
    )

    _check_positive_radius(profile, "sinusoidal_pipe")
    return profile


def hyperbolic_constriction_pipe(
    z: ArrayLike,
    radius: float = 1.0,
    severity: float = 0.30,
    center: float = 0.5,
    width: float = 0.15,
) -> np.ndarray:
    """
    Smooth hyperbolic-secant constriction.

    This is a converging-diverging pipe shape with a smooth throat.

    R(z) = R0 * [1 - severity / cosh((z - center) / width)^2]
    """
    z = _to_array(z)

    if not 0.0 <= severity < 1.0:
        raise ValueError("severity must satisfy 0 <= severity < 1.")

    profile = radius * (
        1.0 - severity / np.cosh((z - center) / width) ** 2
    )

    _check_positive_radius(profile, "hyperbolic_constriction_pipe")
    return profile


GEOMETRY_FUNCTIONS: Dict[str, Callable[..., np.ndarray]] = {
    "straight": straight_pipe,
    "stenosed": stenosed_pipe,
    "expanded": expanded_pipe,
    "sinusoidal": sinusoidal_pipe,
    "hyperbolic_constriction": hyperbolic_constriction_pipe,
}


def make_geometry(name: str, z: ArrayLike, **kwargs) -> np.ndarray:
    """
    Factory function for geometry generation.

    Example
    -------
    z = np.linspace(0, 1, 200)
    R = make_geometry("stenosed", z, severity=0.4)
    """
    if name not in GEOMETRY_FUNCTIONS:
        available = ", ".join(GEOMETRY_FUNCTIONS.keys())
        raise ValueError(f"Unknown geometry '{name}'. Available: {available}")

    return GEOMETRY_FUNCTIONS[name](z, **kwargs)


def sample_default_geometries(
    z: ArrayLike,
    radius: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Generate a small default set of geometry profiles for plotting and debugging.
    """
    return {
        "straight": straight_pipe(z, radius=radius),
        "stenosed": stenosed_pipe(z, radius=radius),
        "expanded": expanded_pipe(z, radius=radius),
        "sinusoidal": sinusoidal_pipe(z, radius=radius),
        "hyperbolic_constriction": hyperbolic_constriction_pipe(z, radius=radius),
    }


if __name__ == "__main__":
    z_grid = np.linspace(0.0, 1.0, 101)
    geometries = sample_default_geometries(z_grid)

    for name, radius_profile in geometries.items():
        print(
            f"{name:24s} | "
            f"min R = {radius_profile.min():.4f} | "
            f"max R = {radius_profile.max():.4f}"
        )