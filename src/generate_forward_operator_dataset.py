"""
Generate a synthetic forward-operator dataset for variable-radius pipe flows.

Each sample contains:

    geometry R(z)
    fluid parameters
    pressure drop
    flow rate Q
    pressure gradient dp/dz
    velocity field u(eta, z)

The dataset is saved as a compressed NumPy .npz file and can later be used
to train DeepONet, FNO, or a coordinate-based neural operator.

Example
-------
python src/generate_forward_operator_dataset.py --n-samples 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from forward_solver import solve_geometry_case


GEOMETRY_NAMES = [
    "straight",
    "stenosed",
    "expanded",
    "sinusoidal",
    "hyperbolic_constriction",
]

FLUID_MODELS = [
    "newtonian",
    "power_law",
]


def sample_geometry_kwargs(
    rng: np.random.Generator,
    geometry_name: str,
) -> Dict[str, Any]:
    """
    Randomly sample geometry parameters.

    The radius is kept close to 1.0 so all geometries live in comparable
    nondimensional ranges.
    """
    radius = float(rng.uniform(0.85, 1.15))

    if geometry_name == "straight":
        return {
            "radius": radius,
        }

    if geometry_name == "stenosed":
        return {
            "radius": radius,
            "severity": float(rng.uniform(0.10, 0.45)),
            "center": float(rng.uniform(0.35, 0.65)),
            "width": float(rng.uniform(0.08, 0.18)),
        }

    if geometry_name == "expanded":
        return {
            "radius": radius,
            "expansion": float(rng.uniform(0.10, 0.40)),
            "center": float(rng.uniform(0.35, 0.65)),
            "width": float(rng.uniform(0.08, 0.18)),
        }

    if geometry_name == "sinusoidal":
        return {
            "radius": radius,
            "amplitude": float(rng.uniform(0.05, 0.20)),
            "modes": int(rng.integers(1, 4)),
            "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        }

    if geometry_name == "hyperbolic_constriction":
        return {
            "radius": radius,
            "severity": float(rng.uniform(0.10, 0.40)),
            "center": float(rng.uniform(0.35, 0.65)),
            "width": float(rng.uniform(0.08, 0.20)),
        }

    raise ValueError(f"Unknown geometry_name: {geometry_name}")


def sample_fluid_kwargs(
    rng: np.random.Generator,
    fluid_model: str,
) -> Dict[str, Any]:
    """Randomly sample fluid parameters."""
    if fluid_model == "newtonian":
        return {
            "viscosity": float(rng.uniform(0.5, 2.0)),
        }

    if fluid_model == "power_law":
        return {
            "consistency": float(rng.uniform(0.5, 2.0)),
            "flow_index": float(rng.uniform(0.55, 1.15)),
        }

    raise ValueError(f"Unknown fluid_model: {fluid_model}")


def encode_geometry_name(geometry_name: str) -> int:
    """Convert geometry name to an integer code."""
    return GEOMETRY_NAMES.index(geometry_name)


def encode_fluid_model(fluid_model: str) -> int:
    """Convert fluid model name to an integer code."""
    return FLUID_MODELS.index(fluid_model)


def generate_dataset(
    n_samples: int,
    n_z: int,
    n_eta: int,
    seed: int,
    pressure_drop_range: Tuple[float, float],
    output_path: Path,
) -> None:
    """Generate and save the forward-operator dataset."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(seed)

    z_all = None
    eta_all = None

    radius_profiles = np.zeros((n_samples, n_z), dtype=np.float64)
    velocity_fields = np.zeros((n_samples, n_z, n_eta), dtype=np.float64)
    pressure_gradients = np.zeros((n_samples, n_z), dtype=np.float64)

    flow_rates = np.zeros(n_samples, dtype=np.float64)
    pressure_drops = np.zeros(n_samples, dtype=np.float64)

    geometry_codes = np.zeros(n_samples, dtype=np.int64)
    fluid_codes = np.zeros(n_samples, dtype=np.int64)

    viscosity_values = np.full(n_samples, np.nan, dtype=np.float64)
    consistency_values = np.full(n_samples, np.nan, dtype=np.float64)
    flow_index_values = np.full(n_samples, np.nan, dtype=np.float64)

    radius_base_values = np.zeros(n_samples, dtype=np.float64)
    severity_values = np.full(n_samples, np.nan, dtype=np.float64)
    expansion_values = np.full(n_samples, np.nan, dtype=np.float64)
    amplitude_values = np.full(n_samples, np.nan, dtype=np.float64)
    modes_values = np.full(n_samples, np.nan, dtype=np.float64)
    center_values = np.full(n_samples, np.nan, dtype=np.float64)
    width_values = np.full(n_samples, np.nan, dtype=np.float64)
    phase_values = np.full(n_samples, np.nan, dtype=np.float64)

    for sample_idx in range(n_samples):
        geometry_name = str(rng.choice(GEOMETRY_NAMES))
        fluid_model = str(rng.choice(FLUID_MODELS))

        geometry_kwargs = sample_geometry_kwargs(rng, geometry_name)
        fluid_kwargs = sample_fluid_kwargs(rng, fluid_model)

        pressure_drop = float(
            rng.uniform(pressure_drop_range[0], pressure_drop_range[1])
        )

        solution = solve_geometry_case(
            geometry_name=geometry_name,
            fluid_model=fluid_model,
            n_z=n_z,
            n_eta=n_eta,
            pressure_drop=pressure_drop,
            length=1.0,
            geometry_kwargs=geometry_kwargs,
            fluid_kwargs=fluid_kwargs,
        )

        if z_all is None:
            z_all = solution.z.copy()
            eta_all = solution.eta.copy()

        radius_profiles[sample_idx, :] = solution.radius
        velocity_fields[sample_idx, :, :] = solution.velocity
        pressure_gradients[sample_idx, :] = solution.pressure_gradient

        flow_rates[sample_idx] = solution.flow_rate
        pressure_drops[sample_idx] = pressure_drop

        geometry_codes[sample_idx] = encode_geometry_name(geometry_name)
        fluid_codes[sample_idx] = encode_fluid_model(fluid_model)

        radius_base_values[sample_idx] = geometry_kwargs.get("radius", np.nan)
        severity_values[sample_idx] = geometry_kwargs.get("severity", np.nan)
        expansion_values[sample_idx] = geometry_kwargs.get("expansion", np.nan)
        amplitude_values[sample_idx] = geometry_kwargs.get("amplitude", np.nan)
        modes_values[sample_idx] = geometry_kwargs.get("modes", np.nan)
        center_values[sample_idx] = geometry_kwargs.get("center", np.nan)
        width_values[sample_idx] = geometry_kwargs.get("width", np.nan)
        phase_values[sample_idx] = geometry_kwargs.get("phase", np.nan)

        viscosity_values[sample_idx] = fluid_kwargs.get("viscosity", np.nan)
        consistency_values[sample_idx] = fluid_kwargs.get("consistency", np.nan)
        flow_index_values[sample_idx] = fluid_kwargs.get("flow_index", np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        z=z_all,
        eta=eta_all,
        radius_profiles=radius_profiles,
        velocity_fields=velocity_fields,
        pressure_gradients=pressure_gradients,
        flow_rates=flow_rates,
        pressure_drops=pressure_drops,
        geometry_codes=geometry_codes,
        fluid_codes=fluid_codes,
        viscosity_values=viscosity_values,
        consistency_values=consistency_values,
        flow_index_values=flow_index_values,
        radius_base_values=radius_base_values,
        severity_values=severity_values,
        expansion_values=expansion_values,
        amplitude_values=amplitude_values,
        modes_values=modes_values,
        center_values=center_values,
        width_values=width_values,
        phase_values=phase_values,
        geometry_names=np.array(GEOMETRY_NAMES),
        fluid_models=np.array(FLUID_MODELS),
    )

    print("Saved dataset:")
    print(f"  {output_path}")
    print("")
    print("Shapes:")
    print(f"  radius_profiles:    {radius_profiles.shape}")
    print(f"  velocity_fields:    {velocity_fields.shape}")
    print(f"  pressure_gradients: {pressure_gradients.shape}")
    print(f"  flow_rates:         {flow_rates.shape}")
    print("")
    print("Ranges:")
    print(f"  Q min/max:          {flow_rates.min():.6e} / {flow_rates.max():.6e}")
    print(
        "  u min/max:          "
        f"{velocity_fields.min():.6e} / {velocity_fields.max():.6e}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a forward-operator pipe-flow dataset."
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples to generate.",
    )

    parser.add_argument(
        "--n-z",
        type=int,
        default=128,
        help="Number of axial grid points.",
    )

    parser.add_argument(
        "--n-eta",
        type=int,
        default=64,
        help="Number of radial eta grid points.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--pressure-drop-min",
        type=float,
        default=0.5,
        help="Minimum pressure drop.",
    )

    parser.add_argument(
        "--pressure-drop-max",
        type=float,
        default=2.0,
        help="Maximum pressure drop.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/forward_operator_dataset.npz",
        help="Output .npz file path.",
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        n_z=args.n_z,
        n_eta=args.n_eta,
        seed=args.seed,
        pressure_drop_range=(
            args.pressure_drop_min,
            args.pressure_drop_max,
        ),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
