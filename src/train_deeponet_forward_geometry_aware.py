"""
Train a rheology-aware and local-geometry-aware constrained DeepONet.

Standalone version:
    This file does NOT import from train_deeponet_forward_powerlaw_constraint.py.
    It is self-contained to avoid circular-import problems.

Model idea
----------
Branch input:
    global radius profile R(z) + pressure drop + fluid parameters

Trunk input:
    (z, eta, n_eff, R(z), dR/dz)

Hard output structure:
    u_pred = scale * [1 - eta^(1 + 1/n_eff)] * softplus(raw)

This enforces:
    u >= 0
    u(eta=1) = 0

Small test:
    python src/train_deeponet_forward_geometry_aware.py --epochs 20

Main run:
    python src/train_deeponet_forward_geometry_aware.py --epochs 1000 --batch-size 8192 --learning-rate 5e-4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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
    - fluid code: 0 for Newtonian, 1 for power-law

    Shape:
        (n_samples, n_z + 5)
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
    Build n_eff used in the wall factor.

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


def train_val_split(
    n_samples: int,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create train/validation sample indices."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_val = max(1, int(val_fraction * n_samples))

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return train_indices, val_indices


def relative_l2_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute relative L2 error."""
    denominator = np.linalg.norm(target)

    if denominator < 1e-12:
        return float(np.linalg.norm(prediction - target))

    return float(np.linalg.norm(prediction - target) / denominator)


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


def build_radius_derivative(radius_profiles: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute dR/dz for every radius profile."""
    return np.gradient(radius_profiles, z, axis=1).astype(np.float32)


def compute_local_geometry_normalization(
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    train_indices: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute normalization constants for local R(z) and dR/dz."""
    train_radius = radius_profiles[train_indices].reshape(-1)
    train_derivative = radius_derivatives[train_indices].reshape(-1)

    radius_mean = float(train_radius.mean())
    radius_std = float(train_radius.std())
    derivative_mean = float(train_derivative.mean())
    derivative_std = float(train_derivative.std())

    if radius_std < 1e-8:
        radius_std = 1.0

    if derivative_std < 1e-8:
        derivative_std = 1.0

    return radius_mean, radius_std, derivative_mean, derivative_std


class GeometryAwareConstrainedDeepONet(nn.Module):
    """
    DeepONet with rheology-aware wall factor and local geometry inputs.

    Branch input:
        global radius profile and physical parameters

    Trunk input:
        z, eta, n_eff, normalized R(z), normalized dR/dz

    Output:
        u_scaled = [1 - eta^(1 + 1/n_eff)] * softplus(raw)
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int = 5,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()

        self.branch_net = MLP(
            input_dim=branch_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        self.trunk_net = MLP(
            input_dim=trunk_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

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

        positive_amplitude = F.softplus(raw)

        return wall_factor * positive_amplitude


def sample_training_batch(
    rng: np.random.Generator,
    train_indices: np.ndarray,
    branch_features: np.ndarray,
    n_eff_values: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    velocity_fields: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    batch_size: int,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    radius_mean: float,
    radius_std: float,
    derivative_mean: float,
    derivative_std: float,
    target_scale: float,
    device: torch.device,
):
    """Sample a random pointwise training batch."""
    sample_indices = rng.choice(train_indices, size=batch_size, replace=True)

    n_z = len(z)
    n_eta = len(eta)

    z_indices = rng.integers(0, n_z, size=batch_size)
    eta_indices = rng.integers(0, n_eta, size=batch_size)

    branch_batch = branch_features[sample_indices]
    branch_batch = (branch_batch - branch_mean) / branch_std

    local_radius = radius_profiles[sample_indices, z_indices]
    local_derivative = radius_derivatives[sample_indices, z_indices]

    local_radius_norm = (local_radius - radius_mean) / radius_std
    local_derivative_norm = (local_derivative - derivative_mean) / derivative_std

    trunk_batch = np.stack(
        [
            z[z_indices],
            eta[eta_indices],
            n_eff_values[sample_indices],
            local_radius_norm,
            local_derivative_norm,
        ],
        axis=1,
    ).astype(np.float32)

    target_batch = velocity_fields[
        sample_indices,
        z_indices,
        eta_indices,
    ].astype(np.float32)[:, None]

    target_batch = target_batch / target_scale

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
    val_indices: np.ndarray,
    branch_features: np.ndarray,
    n_eff_values: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    velocity_fields: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    batch_size: int,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    radius_mean: float,
    radius_std: float,
    derivative_mean: float,
    derivative_std: float,
    target_scale: float,
    device: torch.device,
) -> float:
    """Evaluate validation MSE on random point samples."""
    model.eval()

    branch_tensor, trunk_tensor, target_tensor = sample_training_batch(
        rng=rng,
        train_indices=val_indices,
        branch_features=branch_features,
        n_eff_values=n_eff_values,
        radius_profiles=radius_profiles,
        radius_derivatives=radius_derivatives,
        velocity_fields=velocity_fields,
        z=z,
        eta=eta,
        batch_size=batch_size,
        branch_mean=branch_mean,
        branch_std=branch_std,
        radius_mean=radius_mean,
        radius_std=radius_std,
        derivative_mean=derivative_mean,
        derivative_std=derivative_std,
        target_scale=target_scale,
        device=device,
    )

    prediction = model(branch_tensor, trunk_tensor)
    loss = torch.mean((prediction - target_tensor) ** 2)

    return float(loss.detach().cpu().item())


@torch.no_grad()
def predict_full_field(
    model: nn.Module,
    branch_feature: np.ndarray,
    n_eff: float,
    radius_profile: np.ndarray,
    radius_derivative: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    radius_mean: float,
    radius_std: float,
    derivative_mean: float,
    derivative_std: float,
    target_scale: float,
    device: torch.device,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Predict the full u(z, eta) field for one sample."""
    model.eval()

    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")

    radius_grid = np.repeat(radius_profile[:, None], len(eta), axis=1)
    derivative_grid = np.repeat(radius_derivative[:, None], len(eta), axis=1)

    radius_norm = (radius_grid - radius_mean) / radius_std
    derivative_norm = (derivative_grid - derivative_mean) / derivative_std

    n_grid = np.full_like(z_grid, fill_value=n_eff, dtype=np.float32)

    coords = np.stack(
        [
            z_grid.reshape(-1),
            eta_grid.reshape(-1),
            n_grid.reshape(-1),
            radius_norm.reshape(-1),
            derivative_norm.reshape(-1),
        ],
        axis=1,
    ).astype(np.float32)

    branch_normalized = (branch_feature[None, :] - branch_mean) / branch_std
    branch_repeated = np.repeat(branch_normalized, coords.shape[0], axis=0)

    predictions = []

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])

        branch_tensor = torch.tensor(
            branch_repeated[start:end],
            dtype=torch.float32,
            device=device,
        )

        trunk_tensor = torch.tensor(
            coords[start:end],
            dtype=torch.float32,
            device=device,
        )

        pred = model(branch_tensor, trunk_tensor)
        pred = pred.detach().cpu().numpy()

        predictions.append(pred)

    prediction = np.concatenate(predictions, axis=0).reshape(len(z), len(eta))
    prediction = prediction * target_scale

    return prediction.astype(np.float32)


def plot_training_history(train_losses, val_losses, output_path: Path) -> None:
    """Plot training and validation loss curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    ax.semilogy(train_losses, label="Train loss")
    ax.semilogy(val_losses, label="Validation loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on scaled velocity")
    ax.set_title("Geometry-aware constrained DeepONet training")
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_prediction_example(
    model: nn.Module,
    sample_idx: int,
    branch_features: np.ndarray,
    n_eff_values: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    velocity_fields: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    radius_mean: float,
    radius_std: float,
    derivative_mean: float,
    derivative_std: float,
    target_scale: float,
    device: torch.device,
    output_path: Path,
) -> float:
    """Plot target, prediction, and absolute error for one validation sample."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target = velocity_fields[sample_idx]

    prediction = predict_full_field(
        model=model,
        branch_feature=branch_features[sample_idx],
        n_eff=float(n_eff_values[sample_idx]),
        radius_profile=radius_profiles[sample_idx],
        radius_derivative=radius_derivatives[sample_idx],
        z=z,
        eta=eta,
        branch_mean=branch_mean,
        branch_std=branch_std,
        radius_mean=radius_mean,
        radius_std=radius_std,
        derivative_mean=derivative_mean,
        derivative_std=derivative_std,
        target_scale=target_scale,
        device=device,
    )

    error = np.abs(prediction - target)
    relative_l2 = relative_l2_error(prediction, target)

    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")
    vmax = max(float(target.max()), float(prediction.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    panels = [
        ("Target", target, 0.0, vmax),
        ("Prediction", prediction, 0.0, vmax),
        ("Absolute error", error, 0.0, float(error.max())),
    ]

    for ax, (title, field, vmin, vmax_panel) in zip(axes, panels):
        contour = ax.contourf(
            z_grid,
            eta_grid,
            field,
            levels=50,
            vmin=vmin,
            vmax=vmax_panel,
        )

        ax.set_title(title)
        ax.set_xlabel(r"$z/L$")
        ax.set_ylabel(r"$\eta = r/R(z)$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        fig.colorbar(contour, ax=ax, shrink=0.9)

    fig.suptitle(
        f"Geometry-aware constrained DeepONet | sample {sample_idx} | "
        f"relative L2 = {relative_l2:.3e}",
        fontsize=12,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Example relative L2 error: {relative_l2:.6e}")
    print(f"Prediction min/max: {prediction.min():.6e} / {prediction.max():.6e}")

    return float(relative_l2)


def evaluate_full_validation(
    model: nn.Module,
    data,
    branch_features: np.ndarray,
    n_eff_values: np.ndarray,
    radius_profiles: np.ndarray,
    radius_derivatives: np.ndarray,
    val_indices: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    radius_mean: float,
    radius_std: float,
    derivative_mean: float,
    derivative_std: float,
    target_scale: float,
    device: torch.device,
):
    """Evaluate full-field relative L2 error over all validation samples."""
    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)

    geometry_names = data["geometry_names"]
    fluid_models = data["fluid_models"]
    geometry_codes = data["geometry_codes"]
    fluid_codes = data["fluid_codes"]

    rows = []
    predictions_for_plot = {}

    for count, sample_idx in enumerate(val_indices, start=1):
        target = velocity_fields[sample_idx]

        prediction = predict_full_field(
            model=model,
            branch_feature=branch_features[sample_idx],
            n_eff=float(n_eff_values[sample_idx]),
            radius_profile=radius_profiles[sample_idx],
            radius_derivative=radius_derivatives[sample_idx],
            z=z,
            eta=eta,
            branch_mean=branch_mean,
            branch_std=branch_std,
            radius_mean=radius_mean,
            radius_std=radius_std,
            derivative_mean=derivative_mean,
            derivative_std=derivative_std,
            target_scale=target_scale,
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
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
            }
        )

        predictions_for_plot[int(sample_idx)] = prediction

        if count % 10 == 0 or count == len(val_indices):
            print(f"Full-field validation {count}/{len(val_indices)}")

    return rows, predictions_for_plot


def summarize_error_rows(rows):
    """Summarize errors by geometry/fluid pair."""
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


def print_validation_summary(rows, summary) -> None:
    """Print full validation and grouped summaries."""
    errors = np.array([row["relative_l2"] for row in rows])

    print("")
    print("Geometry-aware constrained DeepONet validation relative L2 errors")
    print("-----------------------------------------------------------------")
    print(f"Mean:   {errors.mean():.6e}")
    print(f"Median: {np.median(errors):.6e}")
    print(f"Min:    {errors.min():.6e}")
    print(f"Max:    {errors.max():.6e}")
    print(f"Std:    {errors.std():.6e}")

    print("")
    print("Geometry-aware constrained DeepONet error by case")
    print("-------------------------------------------------")

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


def plot_validation_histogram(rows, output_path: Path) -> None:
    """Plot histogram of validation relative L2 errors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    errors = np.array([row["relative_l2"] for row in rows])

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(errors, bins=20)

    ax.axvline(
        errors.mean(),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {errors.mean():.3e}",
    )

    ax.axvline(
        np.median(errors),
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {np.median(errors):.3e}",
    )

    ax.set_xlabel("Relative L2 error")
    ax.set_ylabel("Validation sample count")
    ax.set_title("Geometry-aware constrained DeepONet validation errors")
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_best_median_worst_examples(
    rows,
    predictions_for_plot,
    data,
    output_path: Path,
) -> None:
    """Plot target/prediction/error for best, median, and worst samples."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)

    errors = np.array([row["relative_l2"] for row in rows])
    order = np.argsort(errors)

    chosen = [
        ("Best", int(order[0])),
        ("Median", int(order[len(order) // 2])),
        ("Worst", int(order[-1])),
    ]

    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)

    for row_id, (label, position) in enumerate(chosen):
        row = rows[position]
        sample_idx = int(row["sample_idx"])

        target = velocity_fields[sample_idx]
        prediction = predictions_for_plot[sample_idx]
        abs_error = np.abs(prediction - target)

        shared_vmax = max(float(target.max()), float(prediction.max()))
        error_vmax = max(float(abs_error.max()), 1e-12)

        panels = [
            ("Target", target, 0.0, shared_vmax),
            ("Prediction", prediction, 0.0, shared_vmax),
            ("Absolute error", abs_error, 0.0, error_vmax),
        ]

        for col, (panel_title, field, vmin, vmax) in enumerate(panels):
            ax = axes[row_id, col]

            contour = ax.contourf(
                z_grid,
                eta_grid,
                field,
                levels=50,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(
                f"{label} sample {sample_idx}\n"
                f"{panel_title} | rel L2 = {row['relative_l2']:.3e}",
                fontsize=9,
            )

            ax.set_xlabel(r"$z/L$")
            ax.set_ylabel(r"$\eta = r/R(z)$")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

            fig.colorbar(contour, ax=ax, shrink=0.84)

    fig.suptitle(
        "Geometry-aware constrained DeepONet validation examples",
        fontsize=14,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_error_by_case(summary, output_path: Path) -> None:
    """Plot mean and max relative L2 error by case."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [f"{row['geometry']}\n{row['fluid_model']}" for row in summary]
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
    ax.set_title("Geometry-aware constrained DeepONet validation error by case")
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a geometry-aware constrained DeepONet."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/forward_operator_dataset.npz",
        help="Path to the generated dataset.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Number of random coordinate samples per batch.",
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
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    """Train and evaluate the geometry-aware constrained DeepONet."""
    args = parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    data = load_dataset(Path(args.dataset))

    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    radius_profiles = data["radius_profiles"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)

    branch_features = build_branch_features(data)
    n_eff_values = build_effective_flow_index(data)
    radius_derivatives = build_radius_derivative(radius_profiles, z)

    n_samples = branch_features.shape[0]
    train_indices, val_indices = train_val_split(
        n_samples=n_samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    branch_mean = branch_features[train_indices].mean(axis=0, keepdims=True)
    branch_std = branch_features[train_indices].std(axis=0, keepdims=True)
    branch_std = np.where(branch_std < 1e-8, 1.0, branch_std)

    (
        radius_mean,
        radius_std,
        derivative_mean,
        derivative_std,
    ) = compute_local_geometry_normalization(
        radius_profiles=radius_profiles,
        radius_derivatives=radius_derivatives,
        train_indices=train_indices,
    )

    target_scale = float(velocity_fields[train_indices].max())
    if target_scale < 1e-8:
        target_scale = 1.0

    print(f"Number of samples: {n_samples}")
    print(f"Train samples:     {len(train_indices)}")
    print(f"Validation samples:{len(val_indices)}")
    print(f"Branch dimension:  {branch_features.shape[1]}")
    print(f"Trunk dimension:   5 = (z, eta, n_eff, R(z), dR/dz)")
    print(f"Target scale:      {target_scale:.6e}")
    print(
        "Constraint:        "
        "u = scale * [1 - eta^(1 + 1/n_eff)] * softplus(raw)"
    )
    print("Local geometry:    normalized R(z), normalized dR/dz")

    model = GeometryAwareConstrainedDeepONet(
        branch_dim=branch_features.shape[1],
        trunk_dim=5,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        branch_tensor, trunk_tensor, target_tensor = sample_training_batch(
            rng=rng,
            train_indices=train_indices,
            branch_features=branch_features,
            n_eff_values=n_eff_values,
            radius_profiles=radius_profiles,
            radius_derivatives=radius_derivatives,
            velocity_fields=velocity_fields,
            z=z,
            eta=eta,
            batch_size=args.batch_size,
            branch_mean=branch_mean,
            branch_std=branch_std,
            radius_mean=radius_mean,
            radius_std=radius_std,
            derivative_mean=derivative_mean,
            derivative_std=derivative_std,
            target_scale=target_scale,
            device=device,
        )

        optimizer.zero_grad()
        prediction = model(branch_tensor, trunk_tensor)
        loss = torch.mean((prediction - target_tensor) ** 2)
        loss.backward()
        optimizer.step()

        train_loss = float(loss.detach().cpu().item())

        val_loss = evaluate_pointwise_loss(
            model=model,
            rng=rng,
            val_indices=val_indices,
            branch_features=branch_features,
            n_eff_values=n_eff_values,
            radius_profiles=radius_profiles,
            radius_derivatives=radius_derivatives,
            velocity_fields=velocity_fields,
            z=z,
            eta=eta,
            batch_size=args.batch_size,
            branch_mean=branch_mean,
            branch_std=branch_std,
            radius_mean=radius_mean,
            radius_std=radius_std,
            derivative_mean=derivative_mean,
            derivative_std=derivative_std,
            target_scale=target_scale,
            device=device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:05d} | "
                f"train MSE = {train_loss:.6e} | "
                f"val MSE = {val_loss:.6e}"
            )

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "deeponet_forward_geometry_aware.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "branch_mean": branch_mean,
            "branch_std": branch_std,
            "radius_mean": radius_mean,
            "radius_std": radius_std,
            "derivative_mean": derivative_mean,
            "derivative_std": derivative_std,
            "target_scale": target_scale,
            "branch_dim": branch_features.shape[1],
            "trunk_dim": 5,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "z": z,
            "eta": eta,
            "constraint": "u_scaled = [1 - eta^(1 + 1/n_eff)] * softplus(raw)",
            "trunk_inputs": "(z, eta, n_eff, normalized R(z), normalized dR/dz)",
        },
        model_path,
    )

    print(f"Saved model: {model_path}")

    plot_training_history(
        train_losses=train_losses,
        val_losses=val_losses,
        output_path=Path("figures/deeponet_geometry_aware_training_loss.png"),
    )

    example_sample_idx = int(val_indices[0])

    plot_prediction_example(
        model=model,
        sample_idx=example_sample_idx,
        branch_features=branch_features,
        n_eff_values=n_eff_values,
        radius_profiles=radius_profiles,
        radius_derivatives=radius_derivatives,
        velocity_fields=velocity_fields,
        z=z,
        eta=eta,
        branch_mean=branch_mean,
        branch_std=branch_std,
        radius_mean=radius_mean,
        radius_std=radius_std,
        derivative_mean=derivative_mean,
        derivative_std=derivative_std,
        target_scale=target_scale,
        device=device,
        output_path=Path("figures/deeponet_geometry_aware_prediction_example.png"),
    )

    rows, predictions_for_plot = evaluate_full_validation(
        model=model,
        data=data,
        branch_features=branch_features,
        n_eff_values=n_eff_values,
        radius_profiles=radius_profiles,
        radius_derivatives=radius_derivatives,
        val_indices=val_indices,
        branch_mean=branch_mean,
        branch_std=branch_std,
        radius_mean=radius_mean,
        radius_std=radius_std,
        derivative_mean=derivative_mean,
        derivative_std=derivative_std,
        target_scale=target_scale,
        device=device,
    )

    summary = summarize_error_rows(rows)
    print_validation_summary(rows, summary)

    plot_validation_histogram(
        rows=rows,
        output_path=Path("figures/deeponet_geometry_aware_validation_errors.png"),
    )

    plot_best_median_worst_examples(
        rows=rows,
        predictions_for_plot=predictions_for_plot,
        data=data,
        output_path=Path("figures/deeponet_geometry_aware_best_median_worst_examples.png"),
    )

    save_summary_csv(
        summary=summary,
        output_path=Path("results/deeponet_geometry_aware_error_by_case.csv"),
    )

    plot_error_by_case(
        summary=summary,
        output_path=Path("figures/deeponet_geometry_aware_error_by_case.png"),
    )


if __name__ == "__main__":
    main()
