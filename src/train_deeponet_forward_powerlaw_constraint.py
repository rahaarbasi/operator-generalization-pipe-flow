"""
Train a power-law-aware constrained DeepONet for the forward-operator dataset.

This is the next step after the simple positive/no-slip DeepONet.

Output form:
    u_pred = scale * [1 - eta^(1 + 1/n_eff)] * softplus(raw)

where:
    n_eff = 1.0 for Newtonian cases
    n_eff = power-law flow index n for power-law cases

This enforces:
    u >= 0
    u(eta=1) = 0

and uses the correct idealized radial exponent for Newtonian and power-law
pipe profiles.

Important:
    This is a rheology/radial constraint, not yet a geometry constraint.
    Geometry variation is still learned from R(z) through the branch network.

Run:
    python src/train_deeponet_forward_powerlaw_constraint.py --epochs 20
    python src/train_deeponet_forward_powerlaw_constraint.py --epochs 1000 --batch-size 8192 --learning-rate 5e-4
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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset(dataset_path: Path):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Generate it first with:\n"
            "python src/generate_forward_operator_dataset.py --n-samples 500"
        )
    return np.load(dataset_path, allow_pickle=True)


def build_branch_features(data) -> np.ndarray:
    radius_profiles = data["radius_profiles"].astype(np.float32)
    pressure_drops = data["pressure_drops"].astype(np.float32)[:, None]

    viscosity = data["viscosity_values"].astype(np.float32)[:, None]
    consistency = data["consistency_values"].astype(np.float32)[:, None]
    flow_index = data["flow_index_values"].astype(np.float32)[:, None]
    fluid_code = data["fluid_codes"].astype(np.float32)[:, None]

    viscosity = np.nan_to_num(viscosity, nan=0.0)
    consistency = np.nan_to_num(consistency, nan=0.0)
    flow_index = np.nan_to_num(flow_index, nan=0.0)

    return np.concatenate(
        [radius_profiles, pressure_drops, viscosity, consistency, flow_index, fluid_code],
        axis=1,
    ).astype(np.float32)


def build_effective_flow_index(data) -> np.ndarray:
    """n_eff = 1 for Newtonian, and n_eff = n for power-law cases."""
    fluid_codes = data["fluid_codes"].astype(np.int64)
    flow_index = data["flow_index_values"].astype(np.float32)

    n_eff = np.ones_like(flow_index, dtype=np.float32)
    power_law_mask = fluid_codes == 1
    n_eff[power_law_mask] = flow_index[power_law_mask]
    n_eff = np.nan_to_num(n_eff, nan=1.0)

    return n_eff.astype(np.float32)


def train_val_split(n_samples: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_val = max(1, int(val_fraction * n_samples))
    return indices[n_val:], indices[:n_val]


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_layers: int):
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
        return self.net(x)


class PowerLawConstrainedDeepONet(nn.Module):
    """DeepONet with a power-law-aware positive/no-slip wall factor."""

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int = 3,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.branch_net = MLP(branch_dim, latent_dim, hidden_dim, n_layers)
        self.trunk_net = MLP(trunk_dim, latent_dim, hidden_dim, n_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        raw = torch.sum(branch_output * trunk_output, dim=1, keepdim=True) + self.bias

        eta = torch.clamp(trunk_input[:, 1:2], min=0.0, max=1.0)
        n_eff = torch.clamp(trunk_input[:, 2:3], min=0.2, max=5.0)
        exponent = 1.0 + 1.0 / n_eff

        wall_factor = 1.0 - torch.pow(eta, exponent)
        wall_factor = torch.clamp(wall_factor, min=0.0)

        return wall_factor * F.softplus(raw)


def sample_batch(
    rng,
    sample_indices_pool,
    branch_features,
    n_eff_values,
    velocity_fields,
    z,
    eta,
    batch_size,
    branch_mean,
    branch_std,
    target_scale,
    device,
):
    sample_indices = rng.choice(sample_indices_pool, size=batch_size, replace=True)
    z_indices = rng.integers(0, len(z), size=batch_size)
    eta_indices = rng.integers(0, len(eta), size=batch_size)

    branch_batch = (branch_features[sample_indices] - branch_mean) / branch_std
    trunk_batch = np.stack(
        [z[z_indices], eta[eta_indices], n_eff_values[sample_indices]],
        axis=1,
    ).astype(np.float32)

    target_batch = velocity_fields[sample_indices, z_indices, eta_indices].astype(np.float32)[:, None]
    target_batch = target_batch / target_scale

    return (
        torch.tensor(branch_batch, dtype=torch.float32, device=device),
        torch.tensor(trunk_batch, dtype=torch.float32, device=device),
        torch.tensor(target_batch, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def evaluate_pointwise_loss(model, rng, val_indices, branch_features, n_eff_values, velocity_fields, z, eta, batch_size, branch_mean, branch_std, target_scale, device):
    model.eval()
    branch, trunk, target = sample_batch(
        rng,
        val_indices,
        branch_features,
        n_eff_values,
        velocity_fields,
        z,
        eta,
        batch_size,
        branch_mean,
        branch_std,
        target_scale,
        device,
    )
    pred = model(branch, trunk)
    return float(torch.mean((pred - target) ** 2).detach().cpu().item())


@torch.no_grad()
def predict_full_field(model, branch_feature, n_eff, z, eta, branch_mean, branch_std, target_scale, device, chunk_size=4096):
    model.eval()
    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")
    n_grid = np.full_like(z_grid, fill_value=n_eff, dtype=np.float32)
    coords = np.stack([z_grid.reshape(-1), eta_grid.reshape(-1), n_grid.reshape(-1)], axis=1).astype(np.float32)

    branch_normalized = (branch_feature[None, :] - branch_mean) / branch_std
    branch_repeated = np.repeat(branch_normalized, coords.shape[0], axis=0)

    predictions = []
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        branch_tensor = torch.tensor(branch_repeated[start:end], dtype=torch.float32, device=device)
        trunk_tensor = torch.tensor(coords[start:end], dtype=torch.float32, device=device)
        pred = model(branch_tensor, trunk_tensor)
        predictions.append(pred.detach().cpu().numpy())

    prediction = np.concatenate(predictions, axis=0).reshape(len(z), len(eta))
    return (prediction * target_scale).astype(np.float32)


def relative_l2(prediction, target) -> float:
    denom = np.linalg.norm(target)
    if denom < 1e-12:
        return float(np.linalg.norm(prediction - target))
    return float(np.linalg.norm(prediction - target) / denom)


def plot_training(train_losses, val_losses, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.semilogy(train_losses, label="Train loss")
    ax.semilogy(val_losses, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on scaled velocity")
    ax.set_title("Power-law-aware constrained DeepONet training")
    ax.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_prediction_example(model, sample_idx, branch_features, n_eff_values, velocity_fields, z, eta, branch_mean, branch_std, target_scale, device, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = velocity_fields[sample_idx]
    prediction = predict_full_field(model, branch_features[sample_idx], float(n_eff_values[sample_idx]), z, eta, branch_mean, branch_std, target_scale, device)
    error = np.abs(prediction - target)
    rel = relative_l2(prediction, target)

    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")
    vmax = max(float(target.max()), float(prediction.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    panels = [("Target", target, 0.0, vmax), ("Prediction", prediction, 0.0, vmax), ("Absolute error", error, 0.0, float(error.max()))]
    for ax, (title, field, vmin, vmax_panel) in zip(axes, panels):
        contour = ax.contourf(z_grid, eta_grid, field, levels=50, vmin=vmin, vmax=vmax_panel)
        ax.set_title(title)
        ax.set_xlabel(r"$z/L$")
        ax.set_ylabel(r"$\eta = r/R(z)$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        fig.colorbar(contour, ax=ax, shrink=0.9)

    fig.suptitle(f"Power-law-aware constrained DeepONet | sample {sample_idx} | relative L2 = {rel:.3e}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Example relative L2 error: {rel:.6e}")
    print(f"Prediction min/max: {prediction.min():.6e} / {prediction.max():.6e}")


def evaluate_full_validation(model, data, branch_features, n_eff_values, val_indices, branch_mean, branch_std, target_scale, device):
    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)
    geometry_names = data["geometry_names"]
    fluid_models = data["fluid_models"]
    geometry_codes = data["geometry_codes"]
    fluid_codes = data["fluid_codes"]

    rows = []
    predictions = {}
    for count, sample_idx in enumerate(val_indices, start=1):
        target = velocity_fields[sample_idx]
        prediction = predict_full_field(model, branch_features[sample_idx], float(n_eff_values[sample_idx]), z, eta, branch_mean, branch_std, target_scale, device)
        err = relative_l2(prediction, target)
        rows.append(
            {
                "sample_idx": int(sample_idx),
                "geometry": str(geometry_names[geometry_codes[sample_idx]]),
                "fluid_model": str(fluid_models[fluid_codes[sample_idx]]),
                "relative_l2": float(err),
                "target_max": float(target.max()),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
            }
        )
        predictions[int(sample_idx)] = prediction
        if count % 10 == 0 or count == len(val_indices):
            print(f"Full-field validation {count}/{len(val_indices)}")
    return rows, predictions


def summarize(rows):
    case_keys = sorted(set((row["geometry"], row["fluid_model"]) for row in rows))
    summary = []
    for geometry, fluid_model in case_keys:
        case_rows = [row for row in rows if row["geometry"] == geometry and row["fluid_model"] == fluid_model]
        errors = np.array([row["relative_l2"] for row in case_rows])
        worst = max(case_rows, key=lambda row: row["relative_l2"])
        summary.append(
            {
                "geometry": geometry,
                "fluid_model": fluid_model,
                "count": len(case_rows),
                "mean_relative_l2": float(errors.mean()),
                "median_relative_l2": float(np.median(errors)),
                "min_relative_l2": float(errors.min()),
                "max_relative_l2": float(errors.max()),
                "worst_sample_idx": int(worst["sample_idx"]),
                "worst_prediction_min": float(worst["prediction_min"]),
            }
        )
    return summary


def print_summary(rows, summary):
    errors = np.array([row["relative_l2"] for row in rows])
    print("\nPower-law-aware constrained DeepONet validation relative L2 errors")
    print("------------------------------------------------------------------")
    print(f"Mean:   {errors.mean():.6e}")
    print(f"Median: {np.median(errors):.6e}")
    print(f"Min:    {errors.min():.6e}")
    print(f"Max:    {errors.max():.6e}")
    print(f"Std:    {errors.std():.6e}")

    print("\nPower-law-aware constrained DeepONet error by case")
    print("--------------------------------------------------")
    for row in summary:
        print(
            f"{row['geometry']:24s} | {row['fluid_model']:10s} | "
            f"n={row['count']:3d} | mean={row['mean_relative_l2']:.3e} | "
            f"median={row['median_relative_l2']:.3e} | max={row['max_relative_l2']:.3e} | "
            f"worst sample={row['worst_sample_idx']:4d} | pred min={row['worst_prediction_min']:.3e}"
        )


def save_summary_csv(summary, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["geometry", "fluid_model", "count", "mean_relative_l2", "median_relative_l2", "min_relative_l2", "max_relative_l2", "worst_sample_idx", "worst_prediction_min"]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved: {output_path}")


def plot_histogram(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = np.array([row["relative_l2"] for row in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(errors, bins=20)
    ax.axvline(errors.mean(), linestyle="--", linewidth=1.5, label=f"Mean = {errors.mean():.3e}")
    ax.axvline(np.median(errors), linestyle=":", linewidth=1.5, label=f"Median = {np.median(errors):.3e}")
    ax.set_xlabel("Relative L2 error")
    ax.set_ylabel("Validation sample count")
    ax.set_title("Power-law-aware constrained DeepONet validation errors")
    ax.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_best_median_worst(rows, predictions, data, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)
    errors = np.array([row["relative_l2"] for row in rows])
    order = np.argsort(errors)
    chosen = [("Best", int(order[0])), ("Median", int(order[len(order) // 2])), ("Worst", int(order[-1]))]
    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row_id, (label, position) in enumerate(chosen):
        row = rows[position]
        sample_idx = int(row["sample_idx"])
        target = velocity_fields[sample_idx]
        prediction = predictions[sample_idx]
        abs_error = np.abs(prediction - target)
        shared_vmax = max(float(target.max()), float(prediction.max()))
        error_vmax = max(float(abs_error.max()), 1e-12)
        panels = [("Target", target, 0.0, shared_vmax), ("Prediction", prediction, 0.0, shared_vmax), ("Absolute error", abs_error, 0.0, error_vmax)]
        for col, (panel_title, field, vmin, vmax) in enumerate(panels):
            ax = axes[row_id, col]
            contour = ax.contourf(z_grid, eta_grid, field, levels=50, vmin=vmin, vmax=vmax)
            ax.set_title(f"{label} sample {sample_idx}\n{panel_title} | rel L2 = {row['relative_l2']:.3e}", fontsize=9)
            ax.set_xlabel(r"$z/L$")
            ax.set_ylabel(r"$\eta = r/R(z)$")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            fig.colorbar(contour, ax=ax, shrink=0.84)
    fig.suptitle("Power-law-aware constrained DeepONet validation examples", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_error_by_case(summary, output_path: Path):
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
    ax.set_title("Power-law-aware constrained DeepONet validation error by case")
    ax.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a power-law-aware constrained DeepONet.")
    parser.add_argument("--dataset", type=str, default="data/forward_operator_dataset.npz")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    data = load_dataset(Path(args.dataset))
    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)
    branch_features = build_branch_features(data)
    n_eff_values = build_effective_flow_index(data)

    train_indices, val_indices = train_val_split(branch_features.shape[0], args.val_fraction, args.seed)
    branch_mean = branch_features[train_indices].mean(axis=0, keepdims=True)
    branch_std = branch_features[train_indices].std(axis=0, keepdims=True)
    branch_std = np.where(branch_std < 1e-8, 1.0, branch_std)
    target_scale = float(velocity_fields[train_indices].max())
    if target_scale < 1e-8:
        target_scale = 1.0

    print(f"Number of samples: {branch_features.shape[0]}")
    print(f"Train samples:     {len(train_indices)}")
    print(f"Validation samples:{len(val_indices)}")
    print(f"Branch dimension:  {branch_features.shape[1]}")
    print("Trunk dimension:   3 = (z, eta, n_eff)")
    print(f"Target scale:      {target_scale:.6e}")
    print("Constraint:        u = scale * [1 - eta^(1 + 1/n_eff)] * softplus(raw)")

    model = PowerLawConstrainedDeepONet(
        branch_dim=branch_features.shape[1],
        trunk_dim=3,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        branch, trunk, target = sample_batch(
            rng,
            train_indices,
            branch_features,
            n_eff_values,
            velocity_fields,
            z,
            eta,
            args.batch_size,
            branch_mean,
            branch_std,
            target_scale,
            device,
        )
        optimizer.zero_grad()
        pred = model(branch, trunk)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu().item())
        val_loss = evaluate_pointwise_loss(
            model,
            rng,
            val_indices,
            branch_features,
            n_eff_values,
            velocity_fields,
            z,
            eta,
            args.batch_size,
            branch_mean,
            branch_std,
            target_scale,
            device,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:05d} | train MSE = {train_loss:.6e} | val MSE = {val_loss:.6e}")

    Path("models").mkdir(exist_ok=True)
    model_path = Path("models/deeponet_forward_powerlaw_constraint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "branch_mean": branch_mean,
            "branch_std": branch_std,
            "target_scale": target_scale,
            "branch_dim": branch_features.shape[1],
            "trunk_dim": 3,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "z": z,
            "eta": eta,
            "constraint": "u_scaled = [1 - eta^(1 + 1/n_eff)] * softplus(raw)",
        },
        model_path,
    )
    print(f"Saved model: {model_path}")

    plot_training(train_losses, val_losses, Path("figures/deeponet_powerlaw_constraint_training_loss.png"))
    plot_prediction_example(
        model,
        int(val_indices[0]),
        branch_features,
        n_eff_values,
        velocity_fields,
        z,
        eta,
        branch_mean,
        branch_std,
        target_scale,
        device,
        Path("figures/deeponet_powerlaw_constraint_prediction_example.png"),
    )
    rows, predictions = evaluate_full_validation(model, data, branch_features, n_eff_values, val_indices, branch_mean, branch_std, target_scale, device)
    summary = summarize(rows)
    print_summary(rows, summary)
    plot_histogram(rows, Path("figures/deeponet_powerlaw_constraint_validation_errors.png"))
    plot_best_median_worst(rows, predictions, data, Path("figures/deeponet_powerlaw_constraint_best_median_worst_examples.png"))
    save_summary_csv(summary, Path("results/deeponet_powerlaw_constraint_error_by_case.csv"))
    plot_error_by_case(summary, Path("figures/deeponet_powerlaw_constraint_error_by_case.png"))


if __name__ == "__main__":
    main()
