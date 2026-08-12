"""
Leave-one-geometry-out DeepONet sampling-ablation experiment.

This script tests whether DeepONet models trained on four geometry families can
generalize to a held-out geometry family.

Default experiment:
    hold out sinusoidal geometries

Main comparison:
    1. unconstrained DeepONet
    2. power-law-aware constrained DeepONet
    3. geometry-aware constrained DeepONet

This is a paper-critical experiment because it evaluates geometry-family
generalization rather than only random train/validation interpolation.

Smoke test:
    python src/train_deeponet_leave_one_geometry_out_sampling_ablation.py --held-out-geometry sinusoidal --epochs 20 --sampling geometry_aware --models power_law_aware,geometry_aware

Main run:
    python src/train_deeponet_leave_one_geometry_out_sampling_ablation.py --held-out-geometry sinusoidal --epochs 1000 --batch-size 8192 --learning-rate 5e-4 --sampling geometry_aware --models power_law_aware,geometry_aware

Outputs:
    results/leave_one_geometry_out_<geometry>_summary.csv
    results/leave_one_geometry_out_<geometry>_per_sample.csv
    figures/leave_one_geometry_out_<geometry>_summary.png
    figures/leave_one_geometry_out_<geometry>_by_fluid.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


MODEL_KINDS = [
    "unconstrained",
    "power_law_aware",
    "geometry_aware",
    "geometry_channel_aware",
]

MODEL_LABELS = {
    "unconstrained": "Unconstrained",
    "power_law_aware": "Power-law-aware",
    "geometry_aware": "Geometry-aware",
    "geometry_channel_aware": "Geometry-channel-aware",
}


@dataclass
class Normalization:
    """Normalization constants used by the models."""
    branch_mean: np.ndarray
    branch_std: np.ndarray
    target_scale: float
    radius_mean: float = 0.0
    radius_std: float = 1.0
    derivative_mean: float = 0.0
    derivative_std: float = 1.0
    curvature_mean: float = 0.0
    curvature_std: float = 1.0
    inverse_radius_mean: float = 0.0
    inverse_radius_std: float = 1.0


class MLP(nn.Module):
    """Simple fully connected network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the MLP."""
        return self.net(x)


class UnconstrainedDeepONet(nn.Module):
    """Standard unconstrained DeepONet."""

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.branch_net = MLP(branch_dim, latent_dim, hidden_dim, n_layers)
        self.trunk_net = MLP(trunk_dim, latent_dim, hidden_dim, n_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scaled velocity without hard constraints."""
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        raw = torch.sum(branch_output * trunk_output, dim=1, keepdim=True)
        return raw + self.bias


class PowerLawAwareConstrainedDeepONet(nn.Module):
    """
    DeepONet with rheology-aware hard output constraint.

    trunk input:
        z, eta, n_eff

    output:
        u_scaled = [1 - eta^(1 + 1/n_eff)] * softplus(raw)
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.branch_net = MLP(branch_dim, latent_dim, hidden_dim, n_layers)
        self.trunk_net = MLP(trunk_dim, latent_dim, hidden_dim, n_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scaled nonnegative velocity."""
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        raw = torch.sum(branch_output * trunk_output, dim=1, keepdim=True)
        raw = raw + self.bias

        eta = trunk_input[:, 1:2]
        n_eff = trunk_input[:, 2:3]

        n_eff = torch.clamp(n_eff, min=0.2, max=5.0)
        exponent = 1.0 + 1.0 / n_eff

        wall_factor = 1.0 - torch.pow(
            torch.clamp(eta, min=0.0, max=1.0),
            exponent,
        )
        wall_factor = torch.clamp(wall_factor, min=0.0)

        return wall_factor * F.softplus(raw)


class GeometryAwareConstrainedDeepONet(nn.Module):
    """
    DeepONet with rheology-aware wall factor and local geometry inputs.

    trunk input:
        z, eta, n_eff, normalized R(z), normalized dR/dz

    output:
        u_scaled = [1 - eta^(1 + 1/n_eff)] * softplus(raw)
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.branch_net = MLP(branch_dim, latent_dim, hidden_dim, n_layers)
        self.trunk_net = MLP(trunk_dim, latent_dim, hidden_dim, n_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scaled nonnegative velocity."""
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        raw = torch.sum(branch_output * trunk_output, dim=1, keepdim=True)
        raw = raw + self.bias

        eta = trunk_input[:, 1:2]
        n_eff = trunk_input[:, 2:3]

        n_eff = torch.clamp(n_eff, min=0.2, max=5.0)
        exponent = 1.0 + 1.0 / n_eff

        wall_factor = 1.0 - torch.pow(
            torch.clamp(eta, min=0.0, max=1.0),
            exponent,
        )
        wall_factor = torch.clamp(wall_factor, min=0.0)

        return wall_factor * F.softplus(raw)


class GeometryChannelAwareConstrainedDeepONet(nn.Module):
    """
    DeepONet with rheology-aware wall factor and richer fixed-domain geometry channels.

    trunk input:
        z, eta, n_eff, normalized R(z), normalized dR/dz,
        normalized d2R/dz2, normalized 1/R(z), distance_to_wall = 1 - eta

    This is a lightweight DAFNO-inspired fixed-domain baseline: geometry is encoded
    through local channels on the mapped (z, eta) computational domain rather than
    through a physical-domain mask.
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.branch_net = MLP(branch_dim, latent_dim, hidden_dim, n_layers)
        self.trunk_net = MLP(trunk_dim, latent_dim, hidden_dim, n_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scaled nonnegative velocity."""
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        raw = torch.sum(branch_output * trunk_output, dim=1, keepdim=True)
        raw = raw + self.bias

        eta = trunk_input[:, 1:2]
        n_eff = trunk_input[:, 2:3]

        n_eff = torch.clamp(n_eff, min=0.2, max=5.0)
        exponent = 1.0 + 1.0 / n_eff

        wall_factor = 1.0 - torch.pow(
            torch.clamp(eta, min=0.0, max=1.0),
            exponent,
        )
        wall_factor = torch.clamp(wall_factor, min=0.0)

        return wall_factor * F.softplus(raw)


def get_device() -> torch.device:
    """Choose the best available PyTorch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_dataset(dataset_path: Path):
    """Load the generated forward-operator dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Generate it first with:\n"
            "python src/generate_forward_operator_dataset.py --n-samples 500"
        )

    return np.load(dataset_path, allow_pickle=True)


def build_branch_features(data) -> np.ndarray:
    """
    Build branch input features.

    Branch input contains:
    - radius profile R(z)
    - pressure drop
    - viscosity, if Newtonian; otherwise 0
    - consistency K, if power-law; otherwise 0
    - flow index n, if power-law; otherwise 0
    - fluid code
    """
    radius_profiles = data["radius_profiles"].astype(np.float32)
    pressure_drops = data["pressure_drops"].astype(np.float32)[:, None]

    viscosity = data["viscosity_values"].astype(np.float32)[:, None]
    consistency = data["consistency_values"].astype(np.float32)[:, None]
    flow_index = data["flow_index_values"].astype(np.float32)[:, None]
    fluid_code = data["fluid_codes"].astype(np.float32)[:, None]

    viscosity = np.nan_to_num(viscosity, nan=0.0)
    consistency = np.nan_to_num(consistency, nan=0.0)
    flow_index = np.nan_to_num(flow_index, nan=0.0)

    branch_features = np.concatenate(
        [
            radius_profiles,
            pressure_drops,
            viscosity,
            consistency,
            flow_index,
            fluid_code,
        ],
        axis=1,
    )

    return branch_features.astype(np.float32)


def build_effective_flow_index(data) -> np.ndarray:
    """
    Build n_eff used by constrained models.

    Newtonian:
        n_eff = 1

    Power-law:
        n_eff = flow_index
    """
    fluid_codes = data["fluid_codes"].astype(np.int64)
    flow_index = data["flow_index_values"].astype(np.float32)

    n_eff = np.ones_like(flow_index, dtype=np.float32)
    power_law_mask = fluid_codes == 1

    n_eff[power_law_mask] = flow_index[power_law_mask]
    n_eff = np.nan_to_num(n_eff, nan=1.0)

    return n_eff.astype(np.float32)


def build_radius_derivative(radius_profiles: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute dR/dz for every radius profile."""
    return np.gradient(radius_profiles, z, axis=1).astype(np.float32)


def build_radius_curvature(radius_derivatives: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute d2R/dz2 for every radius profile."""
    return np.gradient(radius_derivatives, z, axis=1).astype(np.float32)


def normalize_indicator(indicator: np.ndarray) -> np.ndarray:
    """Robustly normalize a nonnegative geometry indicator to O(1)."""
    indicator = np.asarray(indicator, dtype=np.float32)
    indicator = np.abs(indicator)

    scale = float(np.percentile(indicator, 95))
    if scale < 1e-8:
        return np.zeros_like(indicator, dtype=np.float32)

    normalized = indicator / scale
    return np.clip(normalized, 0.0, 5.0).astype(np.float32)


def build_geometry_sampling_weights(
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    radius_curvatures: np.ndarray,
    z: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """
    Build per-sample probability weights over z for geometry-aware sampling.

    score(z) = 1
             + alpha * normalized(|dR/dz|)
             + beta  * normalized(|d2R/dz2|)
             + gamma * normalized(1/R(z))

    The returned array has shape (n_samples, n_z), and each row sums to one.
    """
    curvature = np.gradient(radius_derivatives, z, axis=1).astype(np.float32)

    slope_score = normalize_indicator(radius_derivatives)
    curvature_score = normalize_indicator(curvature)
    inverse_radius_score = normalize_indicator(
        1.0 / np.maximum(radius_profiles.astype(np.float32), 1e-6)
    )

    score = (
        1.0
        + alpha * slope_score
        + beta * curvature_score
        + gamma * inverse_radius_score
    ).astype(np.float64)

    score = np.maximum(score, 1e-12)
    score = score / score.sum(axis=1, keepdims=True)

    return score.astype(np.float64)


def sample_z_indices(
    rng: np.random.Generator,
    sample_indices: np.ndarray,
    n_z: int,
    sampling: str,
    geometry_sampling_weights: np.ndarray | None,
    adaptive_z_fraction: float,
) -> np.ndarray:
    """Sample axial indices uniformly or with geometry-aware importance sampling."""
    batch_size = len(sample_indices)
    z_indices = rng.integers(0, n_z, size=batch_size)

    if sampling == "uniform":
        return z_indices

    if sampling != "geometry_aware":
        raise ValueError(f"Unknown sampling mode: {sampling}")

    if geometry_sampling_weights is None:
        raise ValueError("geometry_sampling_weights is required for geometry_aware sampling")

    adaptive_mask = rng.random(batch_size) < adaptive_z_fraction
    if not np.any(adaptive_mask):
        return z_indices

    adaptive_positions = np.flatnonzero(adaptive_mask)
    adaptive_sample_indices = sample_indices[adaptive_positions]

    for sample_idx in np.unique(adaptive_sample_indices):
        local_positions = adaptive_positions[adaptive_sample_indices == sample_idx]
        z_indices[local_positions] = rng.choice(
            n_z,
            size=len(local_positions),
            replace=True,
            p=geometry_sampling_weights[int(sample_idx)],
        )

    return z_indices


def sample_eta_indices(
    rng: np.random.Generator,
    eta: np.ndarray,
    batch_size: int,
    sampling: str,
    eta_sampling: str,
    wall_fraction: float,
    center_fraction: float,
    eta_stretch_k: float,
) -> np.ndarray:
    """Sample radial coordinate indices.

    The sampling flag controls whether the overall training sampler is uniform
    or geometry-aware. When sampling == "uniform", eta is always uniform.

    When sampling == "geometry_aware", eta_sampling controls the radial rule:
    - uniform: uniform eta sampling.
    - beta_mixture: previous near-center / near-wall beta-mixture sampler.
    - wall_stretched: continuous mesh-inspired wall-stretched eta sampling.

    The wall_stretched rule uses a smooth monotone map
        eta = 1 - (exp(k(1 - s)) - 1) / (exp(k) - 1),  s ~ Uniform(0, 1),
    which keeps coverage from centerline to wall while increasing density near
    eta = 1. For k -> 0 the distribution approaches uniform.
    """
    n_eta = len(eta)

    # Always default to uniform indices. Individual modes can overwrite them.
    eta_indices = rng.integers(0, n_eta, size=batch_size)

    if sampling == "uniform":
        return eta_indices

    if sampling != "geometry_aware":
        raise ValueError(f"Unknown sampling mode: {sampling}")

    if eta_sampling == "uniform":
        return eta_indices

    if eta_sampling == "beta_mixture":
        random_values = rng.random(batch_size)

        center_mask = random_values < center_fraction
        wall_mask = (
            random_values >= center_fraction
        ) & (
            random_values < center_fraction + wall_fraction
        )

        if np.any(center_mask):
            # Concentrate near eta = 0 while keeping some spread.
            eta_center = rng.beta(0.7, 3.0, size=int(center_mask.sum()))
            eta_indices[center_mask] = np.clip(
                np.searchsorted(eta, eta_center),
                0,
                n_eta - 1,
            )

        if np.any(wall_mask):
            # Concentrate near eta = 1 while keeping some spread.
            eta_wall = 1.0 - rng.beta(0.7, 3.0, size=int(wall_mask.sum()))
            eta_indices[wall_mask] = np.clip(
                np.searchsorted(eta, eta_wall),
                0,
                n_eta - 1,
            )

        return eta_indices

    if eta_sampling == "wall_stretched":
        k = float(eta_stretch_k)
        s = rng.uniform(0.0, 1.0, size=batch_size)

        if abs(k) < 1e-8:
            eta_values = s
        else:
            # Continuous mesh-inspired radial stretching:
            # s = 0 -> eta = 0 centerline
            # s = 1 -> eta = 1 wall
            # k > 0 increases sampling density toward the wall.
            eta_values = 1.0 - (
                np.exp(k * (1.0 - s)) - 1.0
            ) / (
                np.exp(k) - 1.0
            )

        return np.clip(
            np.searchsorted(eta, eta_values),
            0,
            n_eta - 1,
        )

    raise ValueError(f"Unknown eta sampling mode: {eta_sampling}")


def relative_l2_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute relative L2 error."""
    denominator = np.linalg.norm(target)

    if denominator < 1e-12:
        return float(np.linalg.norm(prediction - target))

    return float(np.linalg.norm(prediction - target) / denominator)


def split_train_internal_validation(
    train_indices: np.ndarray,
    internal_val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split training geometries into train and internal validation subsets."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(train_indices, copy=True)
    rng.shuffle(shuffled)

    n_val = max(1, int(internal_val_fraction * len(shuffled)))

    internal_val_indices = shuffled[:n_val]
    train_pool_indices = shuffled[n_val:]

    return train_pool_indices, internal_val_indices


def make_normalization(
    branch_features: np.ndarray,
    velocity_fields: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    radius_curvatures: np.ndarray,
    train_indices: np.ndarray,
) -> Normalization:
    """Compute normalization constants using training geometries only."""
    branch_mean = branch_features[train_indices].mean(axis=0, keepdims=True)
    branch_std = branch_features[train_indices].std(axis=0, keepdims=True)
    branch_std = np.where(branch_std < 1e-8, 1.0, branch_std)

    target_scale = float(velocity_fields[train_indices].max())
    if target_scale < 1e-8:
        target_scale = 1.0

    train_radius = radius_profiles[train_indices].reshape(-1)
    train_derivative = radius_derivatives[train_indices].reshape(-1)
    train_curvature = radius_curvatures[train_indices].reshape(-1)
    train_inverse_radius = (1.0 / np.maximum(radius_profiles[train_indices], 1e-6)).reshape(-1)

    radius_mean = float(train_radius.mean())
    radius_std = float(train_radius.std())

    derivative_mean = float(train_derivative.mean())
    derivative_std = float(train_derivative.std())

    curvature_mean = float(train_curvature.mean())
    curvature_std = float(train_curvature.std())

    inverse_radius_mean = float(train_inverse_radius.mean())
    inverse_radius_std = float(train_inverse_radius.std())

    if radius_std < 1e-8:
        radius_std = 1.0

    if derivative_std < 1e-8:
        derivative_std = 1.0

    if curvature_std < 1e-8:
        curvature_std = 1.0

    if inverse_radius_std < 1e-8:
        inverse_radius_std = 1.0

    return Normalization(
        branch_mean=branch_mean,
        branch_std=branch_std,
        target_scale=target_scale,
        radius_mean=radius_mean,
        radius_std=radius_std,
        derivative_mean=derivative_mean,
        derivative_std=derivative_std,
        curvature_mean=curvature_mean,
        curvature_std=curvature_std,
        inverse_radius_mean=inverse_radius_mean,
        inverse_radius_std=inverse_radius_std,
    )


def build_model(
    model_kind: str,
    branch_dim: int,
    hidden_dim: int,
    latent_dim: int,
    n_layers: int,
) -> nn.Module:
    """Build the requested DeepONet variant."""
    if model_kind == "unconstrained":
        return UnconstrainedDeepONet(
            branch_dim=branch_dim,
            trunk_dim=2,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    if model_kind == "power_law_aware":
        return PowerLawAwareConstrainedDeepONet(
            branch_dim=branch_dim,
            trunk_dim=3,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    if model_kind == "geometry_aware":
        return GeometryAwareConstrainedDeepONet(
            branch_dim=branch_dim,
            trunk_dim=5,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    if model_kind == "geometry_channel_aware":
        return GeometryChannelAwareConstrainedDeepONet(
            branch_dim=branch_dim,
            trunk_dim=8,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    raise ValueError(f"Unknown model kind: {model_kind}")


def make_trunk_batch(
    model_kind: str,
    z_values: np.ndarray,
    eta_values: np.ndarray,
    n_eff_values: np.ndarray,
    local_radius: np.ndarray,
    local_derivative: np.ndarray,
    local_curvature: np.ndarray,
    local_inverse_radius: np.ndarray,
    normalization: Normalization,
) -> np.ndarray:
    """Construct trunk input for a batch."""
    if model_kind == "unconstrained":
        return np.stack(
            [
                z_values,
                eta_values,
            ],
            axis=1,
        ).astype(np.float32)

    if model_kind == "power_law_aware":
        return np.stack(
            [
                z_values,
                eta_values,
                n_eff_values,
            ],
            axis=1,
        ).astype(np.float32)

    if model_kind == "geometry_aware":
        local_radius_norm = (
            local_radius - normalization.radius_mean
        ) / normalization.radius_std

        local_derivative_norm = (
            local_derivative - normalization.derivative_mean
        ) / normalization.derivative_std

        return np.stack(
            [
                z_values,
                eta_values,
                n_eff_values,
                local_radius_norm,
                local_derivative_norm,
            ],
            axis=1,
        ).astype(np.float32)

    if model_kind == "geometry_channel_aware":
        local_radius_norm = (
            local_radius - normalization.radius_mean
        ) / normalization.radius_std

        local_derivative_norm = (
            local_derivative - normalization.derivative_mean
        ) / normalization.derivative_std

        local_curvature_norm = (
            local_curvature - normalization.curvature_mean
        ) / normalization.curvature_std

        local_inverse_radius_norm = (
            local_inverse_radius - normalization.inverse_radius_mean
        ) / normalization.inverse_radius_std

        distance_to_wall = 1.0 - eta_values

        return np.stack(
            [
                z_values,
                eta_values,
                n_eff_values,
                local_radius_norm,
                local_derivative_norm,
                local_curvature_norm,
                local_inverse_radius_norm,
                distance_to_wall,
            ],
            axis=1,
        ).astype(np.float32)

    raise ValueError(f"Unknown model kind: {model_kind}")


def sample_training_batch(
    rng: np.random.Generator,
    model_kind: str,
    sample_indices_pool: np.ndarray,
    branch_features: np.ndarray,
    n_eff_values: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    radius_curvatures: np.ndarray,
    velocity_fields: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    batch_size: int,
    normalization: Normalization,
    device: torch.device,
    sampling: str = "uniform",
    geometry_sampling_weights: np.ndarray | None = None,
    adaptive_z_fraction: float = 0.30,
    eta_wall_fraction: float = 0.25,
    eta_center_fraction: float = 0.25,
    eta_sampling: str = "beta_mixture",
    eta_stretch_k: float = 2.0,
):
    """Sample a pointwise training/evaluation batch."""
    sample_indices = rng.choice(sample_indices_pool, size=batch_size, replace=True)

    n_z = len(z)

    z_indices = sample_z_indices(
        rng=rng,
        sample_indices=sample_indices,
        n_z=n_z,
        sampling=sampling,
        geometry_sampling_weights=geometry_sampling_weights,
        adaptive_z_fraction=adaptive_z_fraction,
    )

    eta_indices = sample_eta_indices(
        rng=rng,
        eta=eta,
        batch_size=batch_size,
        sampling=sampling,
        eta_sampling=eta_sampling,
        wall_fraction=eta_wall_fraction,
        center_fraction=eta_center_fraction,
        eta_stretch_k=eta_stretch_k,
    )

    branch_batch = branch_features[sample_indices]
    branch_batch = (
        branch_batch - normalization.branch_mean
    ) / normalization.branch_std

    local_radius = radius_profiles[sample_indices, z_indices]
    local_derivative = radius_derivatives[sample_indices, z_indices]
    local_curvature = radius_curvatures[sample_indices, z_indices]
    local_inverse_radius = 1.0 / np.maximum(local_radius, 1e-6)

    trunk_batch = make_trunk_batch(
        model_kind=model_kind,
        z_values=z[z_indices],
        eta_values=eta[eta_indices],
        n_eff_values=n_eff_values[sample_indices],
        local_radius=local_radius,
        local_derivative=local_derivative,
        local_curvature=local_curvature,
        local_inverse_radius=local_inverse_radius,
        normalization=normalization,
    )

    target_batch = velocity_fields[
        sample_indices,
        z_indices,
        eta_indices,
    ].astype(np.float32)[:, None]

    target_batch = target_batch / normalization.target_scale

    branch_tensor = torch.tensor(
        branch_batch,
        dtype=torch.float32,
        device=device,
    )

    trunk_tensor = torch.tensor(
        trunk_batch,
        dtype=torch.float32,
        device=device,
    )

    target_tensor = torch.tensor(
        target_batch,
        dtype=torch.float32,
        device=device,
    )

    return branch_tensor, trunk_tensor, target_tensor


@torch.no_grad()
def evaluate_pointwise_loss(
    model: nn.Module,
    rng: np.random.Generator,
    model_kind: str,
    sample_indices_pool: np.ndarray,
    branch_features: np.ndarray,
    n_eff_values: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    radius_curvatures: np.ndarray,
    velocity_fields: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    batch_size: int,
    normalization: Normalization,
    device: torch.device,
) -> float:
    """Evaluate random pointwise MSE on a sample pool."""
    model.eval()

    branch_tensor, trunk_tensor, target_tensor = sample_training_batch(
        rng=rng,
        model_kind=model_kind,
        sample_indices_pool=sample_indices_pool,
        branch_features=branch_features,
        n_eff_values=n_eff_values,
        radius_profiles=radius_profiles,
        radius_derivatives=radius_derivatives,
        radius_curvatures=radius_curvatures,
        velocity_fields=velocity_fields,
        z=z,
        eta=eta,
        batch_size=batch_size,
        normalization=normalization,
        device=device,
    )

    prediction = model(branch_tensor, trunk_tensor)
    loss = torch.mean((prediction - target_tensor) ** 2)

    return float(loss.detach().cpu().item())


def train_model(
    model_kind: str,
    branch_features: np.ndarray,
    n_eff_values: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    radius_curvatures: np.ndarray,
    velocity_fields: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    train_pool_indices: np.ndarray,
    internal_val_indices: np.ndarray,
    normalization: Normalization,
    args,
    device: torch.device,
    geometry_sampling_weights: np.ndarray | None = None,
) -> nn.Module:
    """Train one DeepONet variant."""
    rng = np.random.default_rng(args.seed)

    model = build_model(
        model_kind=model_kind,
        branch_dim=branch_features.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("")
    print(f"Training model: {MODEL_LABELS[model_kind]}")
    print("-" * (16 + len(MODEL_LABELS[model_kind])))

    for epoch in range(1, args.epochs + 1):
        model.train()

        branch_tensor, trunk_tensor, target_tensor = sample_training_batch(
            rng=rng,
            model_kind=model_kind,
            sample_indices_pool=train_pool_indices,
            branch_features=branch_features,
            n_eff_values=n_eff_values,
            radius_profiles=radius_profiles,
            radius_derivatives=radius_derivatives,
            radius_curvatures=radius_curvatures,
            velocity_fields=velocity_fields,
            z=z,
            eta=eta,
            batch_size=args.batch_size,
            normalization=normalization,
            device=device,
            sampling=args.sampling,
            geometry_sampling_weights=geometry_sampling_weights,
            adaptive_z_fraction=args.adaptive_z_fraction,
            eta_wall_fraction=args.eta_wall_fraction,
            eta_center_fraction=args.eta_center_fraction,
            eta_sampling=args.eta_sampling,
            eta_stretch_k=args.eta_stretch_k,
        )

        optimizer.zero_grad()

        prediction = model(branch_tensor, trunk_tensor)
        loss = torch.mean((prediction - target_tensor) ** 2)

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            train_loss = float(loss.detach().cpu().item())

            val_loss = evaluate_pointwise_loss(
                model=model,
                rng=rng,
                model_kind=model_kind,
                sample_indices_pool=internal_val_indices,
                branch_features=branch_features,
                n_eff_values=n_eff_values,
                radius_profiles=radius_profiles,
                radius_derivatives=radius_derivatives,
                radius_curvatures=radius_curvatures,
                velocity_fields=velocity_fields,
                z=z,
                eta=eta,
                batch_size=args.batch_size,
                normalization=normalization,
                device=device,
            )

            print(
                f"Epoch {epoch:05d} | "
                f"train MSE = {train_loss:.6e} | "
                f"internal val MSE = {val_loss:.6e}"
            )

    return model


@torch.no_grad()
def predict_full_field(
    model: nn.Module,
    model_kind: str,
    sample_idx: int,
    branch_features: np.ndarray,
    n_eff_values: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    radius_curvatures: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    normalization: Normalization,
    device: torch.device,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Predict full u(z, eta) field for one sample."""
    model.eval()

    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")

    radius_grid = np.repeat(radius_profiles[sample_idx, :, None], len(eta), axis=1)
    derivative_grid = np.repeat(
        radius_derivatives[sample_idx, :, None],
        len(eta),
        axis=1,
    )
    curvature_grid = np.repeat(
        radius_curvatures[sample_idx, :, None],
        len(eta),
        axis=1,
    )
    inverse_radius_grid = 1.0 / np.maximum(radius_grid, 1e-6)

    n_grid = np.full_like(
        z_grid,
        fill_value=float(n_eff_values[sample_idx]),
        dtype=np.float32,
    )

    trunk_input = make_trunk_batch(
        model_kind=model_kind,
        z_values=z_grid.reshape(-1),
        eta_values=eta_grid.reshape(-1),
        n_eff_values=n_grid.reshape(-1),
        local_radius=radius_grid.reshape(-1),
        local_derivative=derivative_grid.reshape(-1),
        local_curvature=curvature_grid.reshape(-1),
        local_inverse_radius=inverse_radius_grid.reshape(-1),
        normalization=normalization,
    )

    branch = branch_features[sample_idx][None, :]
    branch = (branch - normalization.branch_mean) / normalization.branch_std
    branch = np.repeat(branch, trunk_input.shape[0], axis=0)

    predictions = []

    for start in range(0, trunk_input.shape[0], chunk_size):
        end = min(start + chunk_size, trunk_input.shape[0])

        branch_tensor = torch.tensor(
            branch[start:end],
            dtype=torch.float32,
            device=device,
        )

        trunk_tensor = torch.tensor(
            trunk_input[start:end],
            dtype=torch.float32,
            device=device,
        )

        pred = model(branch_tensor, trunk_tensor).detach().cpu().numpy()
        predictions.append(pred)

    prediction = np.concatenate(predictions, axis=0).reshape(len(z), len(eta))
    prediction = prediction * normalization.target_scale

    return prediction.astype(np.float32)


def evaluate_held_out_geometry(
    model: nn.Module,
    model_kind: str,
    held_out_indices: np.ndarray,
    data,
    branch_features: np.ndarray,
    n_eff_values: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    radius_curvatures: np.ndarray,
    normalization: Normalization,
    device: torch.device,
) -> List[Dict]:
    """Evaluate model on all samples from the held-out geometry."""
    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)

    fluid_models = data["fluid_models"]
    fluid_codes = data["fluid_codes"]
    geometry_names = data["geometry_names"]
    geometry_codes = data["geometry_codes"]

    rows = []

    for count, sample_idx in enumerate(held_out_indices, start=1):
        target = velocity_fields[sample_idx]

        prediction = predict_full_field(
            model=model,
            model_kind=model_kind,
            sample_idx=int(sample_idx),
            branch_features=branch_features,
            n_eff_values=n_eff_values,
            radius_profiles=radius_profiles,
            radius_derivatives=radius_derivatives,
            radius_curvatures=radius_curvatures,
            z=z,
            eta=eta,
            normalization=normalization,
            device=device,
        )

        relative_l2 = relative_l2_error(prediction, target)

        rows.append(
            {
                "model": MODEL_LABELS[model_kind],
                "model_key": model_kind,
                "sample_idx": int(sample_idx),
                "geometry": str(geometry_names[geometry_codes[sample_idx]]),
                "fluid_model": str(fluid_models[fluid_codes[sample_idx]]),
                "relative_l2": float(relative_l2),
                "target_max": float(target.max()),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
                "has_negative_prediction": bool(prediction.min() < -1e-8),
            }
        )

        if count % 10 == 0 or count == len(held_out_indices):
            print(
                f"Evaluated {MODEL_LABELS[model_kind]} "
                f"{count}/{len(held_out_indices)} held-out samples"
            )

    return rows


def summarize_rows(rows: List[Dict]) -> List[Dict]:
    """Summarize per-sample rows by model and fluid model."""
    summary = []

    grouping_keys = []

    model_order = [
        "Unconstrained",
        "Power-law-aware",
        "Geometry-aware",
        "Geometry-channel-aware",
    ]

    available_models = [model for model in model_order if any(row["model"] == model for row in rows)]

    for model_label in available_models:
        grouping_keys.append((model_label, "all"))

        for fluid_model in sorted(
            set(row["fluid_model"] for row in rows if row["model"] == model_label)
        ):
            grouping_keys.append((model_label, fluid_model))

    for model_label, fluid_model in grouping_keys:
        if fluid_model == "all":
            group_rows = [row for row in rows if row["model"] == model_label]
        else:
            group_rows = [
                row
                for row in rows
                if row["model"] == model_label
                and row["fluid_model"] == fluid_model
            ]

        if not group_rows:
            continue

        errors = np.array([row["relative_l2"] for row in group_rows])

        summary.append(
            {
                "model": model_label,
                "fluid_model": fluid_model,
                "count": len(group_rows),
                "mean_relative_l2": float(errors.mean()),
                "median_relative_l2": float(np.median(errors)),
                "min_relative_l2": float(errors.min()),
                "max_relative_l2": float(errors.max()),
                "std_relative_l2": float(errors.std()),
                "negative_prediction_count": int(
                    sum(row["has_negative_prediction"] for row in group_rows)
                ),
                "min_prediction": float(
                    min(row["prediction_min"] for row in group_rows)
                ),
            }
        )

    return summary


def save_csv(rows: List[Dict], output_path: Path) -> None:
    """Save rows as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to save for {output_path}")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


def print_summary(summary_rows: List[Dict], held_out_geometry: str) -> None:
    """Print summary table."""
    print("")
    print(f"Leave-one-geometry-out test: held out {held_out_geometry}")
    print("-" * (34 + len(held_out_geometry)))

    for row in summary_rows:
        print(
            f"{row['model']:18s} | "
            f"fluid={row['fluid_model']:10s} | "
            f"n={row['count']:3d} | "
            f"mean={row['mean_relative_l2']:.3e} | "
            f"median={row['median_relative_l2']:.3e} | "
            f"max={row['max_relative_l2']:.3e} | "
            f"neg={row['negative_prediction_count']:3d}"
        )


def plot_summary(summary_rows: List[Dict], held_out_geometry: str, output_path: Path):
    """Plot all-fluid held-out summary by model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [row for row in summary_rows if row["fluid_model"] == "all"]

    model_order = [
        "Unconstrained",
        "Power-law-aware",
        "Geometry-aware",
        "Geometry-channel-aware",
    ]

    rows = sorted(rows, key=lambda row: model_order.index(row["model"]))

    labels = [row["model"] for row in rows]
    mean_errors = [row["mean_relative_l2"] for row in rows]
    median_errors = [row["median_relative_l2"] for row in rows]
    max_errors = [row["max_relative_l2"] for row in rows]

    x = np.arange(len(rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    ax.bar(x - width, mean_errors, width, label="Mean L2")
    ax.bar(x, median_errors, width, label="Median L2")
    ax.bar(x + width, max_errors, width, label="Max L2")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative L2 error")
    ax.set_title(f"Leave-one-geometry-out: held out {held_out_geometry}")
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_by_fluid(summary_rows: List[Dict], held_out_geometry: str, output_path: Path):
    """Plot mean held-out error by model and fluid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [row for row in summary_rows if row["fluid_model"] != "all"]

    model_order = [
        "Unconstrained",
        "Power-law-aware",
        "Geometry-aware",
        "Geometry-channel-aware",
    ]

    fluid_order = sorted(set(row["fluid_model"] for row in rows))

    x = np.arange(len(model_order))
    width = 0.8 / max(1, len(fluid_order))

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for i, fluid_model in enumerate(fluid_order):
        fluid_rows = {
            row["model"]: row
            for row in rows
            if row["fluid_model"] == fluid_model
        }

        means = [
            fluid_rows[model]["mean_relative_l2"]
            if model in fluid_rows else np.nan
            for model in model_order
        ]

        offset = (i - (len(fluid_order) - 1) / 2) * width

        ax.bar(
            x + offset,
            means,
            width,
            label=fluid_model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order)
    ax.set_ylabel("Mean relative L2 error")
    ax.set_title(f"Held-out {held_out_geometry}: error by fluid model")
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DeepONets with leave-one-geometry-out testing."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/forward_operator_dataset.npz",
        help="Path to the generated forward-operator dataset.",
    )

    parser.add_argument(
        "--held-out-geometry",
        type=str,
        default="sinusoidal",
        help="Geometry family to hold out from training.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs for each model.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Number of random coordinate points per training batch.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden width of branch and trunk networks.",
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent dimension for branch/trunk dot product.",
    )

    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of hidden layers in branch and trunk networks.",
    )

    parser.add_argument(
        "--internal-val-fraction",
        type=float,
        default=0.15,
        help="Fraction of non-held-out samples used for internal validation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed.",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print progress every this many epochs.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="unconstrained,power_law_aware,geometry_aware",
        help=(
            "Comma-separated list of models to train. "
            "Options: unconstrained,power_law_aware,geometry_aware,geometry_channel_aware"
        ),
    )

    parser.add_argument(
        "--sampling",
        type=str,
        default="uniform",
        choices=["uniform", "geometry_aware"],
        help="Coordinate sampling strategy used during training.",
    )

    parser.add_argument(
        "--adaptive-z-fraction",
        type=float,
        default=0.30,
        help="Fraction of z coordinates sampled from geometry-aware weights.",
    )

    parser.add_argument(
        "--sampling-alpha",
        type=float,
        default=1.0,
        help="Weight for normalized |dR/dz| in geometry-aware z sampling.",
    )

    parser.add_argument(
        "--sampling-beta",
        type=float,
        default=0.5,
        help="Weight for normalized |d2R/dz2| in geometry-aware z sampling.",
    )

    parser.add_argument(
        "--sampling-gamma",
        type=float,
        default=0.5,
        help="Weight for normalized 1/R(z) in geometry-aware z sampling.",
    )

    parser.add_argument(
        "--eta-wall-fraction",
        type=float,
        default=0.25,
        help="Fraction of eta coordinates biased near the wall for geometry-aware sampling.",
    )

    parser.add_argument(
        "--eta-center-fraction",
        type=float,
        default=0.25,
        help="Fraction of eta coordinates biased near the centerline for geometry-aware sampling.",
    )

    parser.add_argument(
        "--eta-sampling",
        type=str,
        default="beta_mixture",
        choices=["uniform", "beta_mixture", "wall_stretched"],
        help=(
            "Radial eta sampling mode used when --sampling geometry_aware. "
            "Options: uniform, beta_mixture, wall_stretched."
        ),
    )

    parser.add_argument(
        "--eta-stretch-k",
        type=float,
        default=2.0,
        help="Stretching strength for continuous wall-stretched eta sampling.",
    )

    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help=(
            "Optional tag used in output file names. "
            "Use this to avoid overwriting results when comparing sampling variants."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run leave-one-geometry-out experiment."""
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()

    print(f"Using device: {device}")

    data = load_dataset(Path(args.dataset))

    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)
    radius_profiles = data["radius_profiles"].astype(np.float32)
    geometry_names = data["geometry_names"]
    geometry_codes = data["geometry_codes"]

    branch_features = build_branch_features(data)
    n_eff_values = build_effective_flow_index(data)
    radius_derivatives = build_radius_derivative(radius_profiles, z)
    radius_curvatures = build_radius_curvature(radius_derivatives, z)

    geometry_sampling_weights = None
    if args.sampling == "geometry_aware":
        geometry_sampling_weights = build_geometry_sampling_weights(
            radius_profiles=radius_profiles,
            radius_derivatives=radius_derivatives,
            radius_curvatures=radius_curvatures,
            z=z,
            alpha=args.sampling_alpha,
            beta=args.sampling_beta,
            gamma=args.sampling_gamma,
        )

    geometry_name_to_code = {
        str(name): int(code)
        for code, name in enumerate(geometry_names)
    }

    if args.held_out_geometry not in geometry_name_to_code:
        available = ", ".join(str(name) for name in geometry_names)
        raise ValueError(
            f"Unknown held-out geometry: {args.held_out_geometry}\n"
            f"Available geometries: {available}"
        )

    held_out_code = geometry_name_to_code[args.held_out_geometry]

    all_indices = np.arange(len(geometry_codes))
    held_out_indices = all_indices[geometry_codes == held_out_code]
    non_held_out_indices = all_indices[geometry_codes != held_out_code]

    train_pool_indices, internal_val_indices = split_train_internal_validation(
        train_indices=non_held_out_indices,
        internal_val_fraction=args.internal_val_fraction,
        seed=args.seed,
    )

    normalization = make_normalization(
        branch_features=branch_features,
        velocity_fields=velocity_fields,
        radius_profiles=radius_profiles,
        radius_derivatives=radius_derivatives,
        radius_curvatures=radius_curvatures,
        train_indices=train_pool_indices,
    )

    requested_models = [
        item.strip()
        for item in args.models.split(",")
        if item.strip()
    ]

    for model_kind in requested_models:
        if model_kind not in MODEL_KINDS:
            raise ValueError(
                f"Unknown model kind: {model_kind}. "
                f"Available: {', '.join(MODEL_KINDS)}"
            )

    print("")
    print("Leave-one-geometry-out setup")
    print("----------------------------")
    print(f"Held-out geometry:      {args.held_out_geometry}")
    print(f"Held-out samples:       {len(held_out_indices)}")
    print(f"Training-pool samples:  {len(train_pool_indices)}")
    print(f"Internal-val samples:   {len(internal_val_indices)}")
    print(f"Branch dimension:       {branch_features.shape[1]}")
    print(f"Target scale:           {normalization.target_scale:.6e}")
    print(f"Models:                 {', '.join(requested_models)}")
    print(f"Training sampling:      {args.sampling}")
    if args.sampling == "geometry_aware":
        print(f"Adaptive z fraction:    {args.adaptive_z_fraction:.2f}")
        print(
            "Sampling weights:       "
            f"alpha={args.sampling_alpha:.2f}, "
            f"beta={args.sampling_beta:.2f}, "
            f"gamma={args.sampling_gamma:.2f}"
        )
        print(f"Eta sampling mode:      {args.eta_sampling}")
        if args.eta_sampling == "beta_mixture":
            print(
                "Eta bias fractions:     "
                f"wall={args.eta_wall_fraction:.2f}, "
                f"center={args.eta_center_fraction:.2f}"
            )
        elif args.eta_sampling == "wall_stretched":
            print(f"Eta stretch k:          {args.eta_stretch_k:.2f}")

    all_sample_rows: List[Dict] = []

    for model_kind in requested_models:
        model = train_model(
            model_kind=model_kind,
            branch_features=branch_features,
            n_eff_values=n_eff_values,
            radius_profiles=radius_profiles,
            radius_derivatives=radius_derivatives,
            radius_curvatures=radius_curvatures,
            velocity_fields=velocity_fields,
            z=z,
            eta=eta,
            train_pool_indices=train_pool_indices,
            internal_val_indices=internal_val_indices,
            normalization=normalization,
            args=args,
            device=device,
            geometry_sampling_weights=geometry_sampling_weights,
        )

        model_rows = evaluate_held_out_geometry(
            model=model,
            model_kind=model_kind,
            held_out_indices=held_out_indices,
            data=data,
            branch_features=branch_features,
            n_eff_values=n_eff_values,
            radius_profiles=radius_profiles,
            radius_derivatives=radius_derivatives,
            radius_curvatures=radius_curvatures,
            normalization=normalization,
            device=device,
        )

        all_sample_rows.extend(model_rows)

    summary_rows = summarize_rows(all_sample_rows)
    print_summary(summary_rows, args.held_out_geometry)

    safe_geometry = args.held_out_geometry.replace("/", "_").replace(" ", "_")
    safe_sampling = args.output_tag if args.output_tag else args.sampling
    safe_sampling = safe_sampling.replace("/", "_").replace(" ", "_")

    per_sample_path = Path(
        f"results/leave_one_geometry_out_{safe_geometry}_{safe_sampling}_per_sample.csv"
    )
    summary_path = Path(
        f"results/leave_one_geometry_out_{safe_geometry}_{safe_sampling}_summary.csv"
    )

    save_csv(all_sample_rows, per_sample_path)
    save_csv(summary_rows, summary_path)

    plot_summary(
        summary_rows=summary_rows,
        held_out_geometry=args.held_out_geometry,
        output_path=Path(
            f"figures/leave_one_geometry_out_{safe_geometry}_{args.sampling}_summary.png"
        ),
    )

    plot_by_fluid(
        summary_rows=summary_rows,
        held_out_geometry=args.held_out_geometry,
        output_path=Path(
            f"figures/leave_one_geometry_out_{safe_geometry}_{args.sampling}_by_fluid.png"
        ),
    )


if __name__ == "__main__":
    main()
