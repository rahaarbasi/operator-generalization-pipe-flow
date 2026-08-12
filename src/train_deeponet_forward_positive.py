
"""
Train a physics-constrained positive DeepONet for the forward-operator dataset.

Compared with the unconstrained DeepONet baseline, this model enforces:

    u(z, eta) >= 0
    u(z, eta=1) = 0

using:

    u_pred_scaled(z, eta) = (1 - eta^2) * softplus(network_output)

Small smoke test:
    python src/train_deeponet_forward_positive.py --epochs 20

Main run:
    python src/train_deeponet_forward_positive.py --epochs 1000 --batch-size 8192 --learning-rate 5e-4
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


class PositiveDeepONet(nn.Module):
    """
    Positive DeepONet.

    Output:
        u_scaled = (1 - eta^2) * softplus(raw)

    This enforces:
        u >= 0
        u(eta=1) = 0
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int = 2,
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
        eta = trunk_input[:, 1:2]
        wall_factor = torch.clamp(1.0 - eta**2, min=0.0)
        return wall_factor * F.softplus(raw)


def sample_batch(
    rng: np.random.Generator,
    sample_pool: np.ndarray,
    branch_features: np.ndarray,
    velocity_fields: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    batch_size: int,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    target_scale: float,
    device: torch.device,
):
    sample_indices = rng.choice(sample_pool, size=batch_size, replace=True)
    z_indices = rng.integers(0, len(z), size=batch_size)
    eta_indices = rng.integers(0, len(eta), size=batch_size)

    branch_batch = (branch_features[sample_indices] - branch_mean) / branch_std
    trunk_batch = np.stack([z[z_indices], eta[eta_indices]], axis=1).astype(np.float32)
    target_batch = velocity_fields[sample_indices, z_indices, eta_indices].astype(np.float32)[:, None]
    target_batch = target_batch / target_scale

    return (
        torch.tensor(branch_batch, dtype=torch.float32, device=device),
        torch.tensor(trunk_batch, dtype=torch.float32, device=device),
        torch.tensor(target_batch, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def pointwise_val_loss(
    model: nn.Module,
    rng: np.random.Generator,
    val_indices: np.ndarray,
    branch_features: np.ndarray,
    velocity_fields: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    batch_size: int,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    target_scale: float,
    device: torch.device,
) -> float:
    model.eval()
    branch, trunk, target = sample_batch(
        rng, val_indices, branch_features, velocity_fields, z, eta, batch_size,
        branch_mean, branch_std, target_scale, device
    )
    pred = model(branch, trunk)
    return float(torch.mean((pred - target) ** 2).detach().cpu().item())


@torch.no_grad()
def predict_full_field(
    model: nn.Module,
    branch_feature: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    target_scale: float,
    device: torch.device,
    chunk_size: int = 4096,
) -> np.ndarray:
    model.eval()
    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")
    coords = np.stack([z_grid.reshape(-1), eta_grid.reshape(-1)], axis=1).astype(np.float32)

    branch_normalized = (branch_feature[None, :] - branch_mean) / branch_std
    branch_repeated = np.repeat(branch_normalized, coords.shape[0], axis=0)

    preds = []
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        branch_tensor = torch.tensor(branch_repeated[start:end], dtype=torch.float32, device=device)
        trunk_tensor = torch.tensor(coords[start:end], dtype=torch.float32, device=device)
        pred = model(branch_tensor, trunk_tensor).detach().cpu().numpy()
        preds.append(pred)

    pred = np.concatenate(preds, axis=0).reshape(len(z), len(eta))
    return (pred * target_scale).astype(np.float32)


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.linalg.norm(target)
    if denom < 1e-12:
        return float(np.linalg.norm(pred - target))
    return float(np.linalg.norm(pred - target) / denom)


def plot_training(train_losses, val_losses, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.semilogy(train_losses, label="Train loss")
    ax.semilogy(val_losses, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on scaled velocity")
    ax.set_title("Positive DeepONet training")
    ax.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def evaluate_validation(
    model,
    data,
    branch_features,
    val_indices,
    branch_mean,
    branch_std,
    target_scale,
    device,
):
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
        pred = predict_full_field(
            model, branch_features[sample_idx], z, eta,
            branch_mean, branch_std, target_scale, device
        )
        err = relative_l2(pred, target)
        rows.append({
            "sample_idx": int(sample_idx),
            "geometry": str(geometry_names[geometry_codes[sample_idx]]),
            "fluid_model": str(fluid_models[fluid_codes[sample_idx]]),
            "relative_l2": float(err),
            "prediction_min": float(pred.min()),
            "prediction_max": float(pred.max()),
            "target_max": float(target.max()),
        })
        predictions[int(sample_idx)] = pred
        if count % 10 == 0 or count == len(val_indices):
            print(f"Full-field validation {count}/{len(val_indices)}")

    return rows, predictions


def summarize_by_case(rows):
    summary = []
    keys = sorted(set((r["geometry"], r["fluid_model"]) for r in rows))
    for geometry, fluid_model in keys:
        case_rows = [r for r in rows if r["geometry"] == geometry and r["fluid_model"] == fluid_model]
        errors = np.array([r["relative_l2"] for r in case_rows])
        worst = max(case_rows, key=lambda r: r["relative_l2"])
        summary.append({
            "geometry": geometry,
            "fluid_model": fluid_model,
            "count": len(case_rows),
            "mean_relative_l2": float(errors.mean()),
            "median_relative_l2": float(np.median(errors)),
            "min_relative_l2": float(errors.min()),
            "max_relative_l2": float(errors.max()),
            "worst_sample_idx": int(worst["sample_idx"]),
            "worst_prediction_min": float(worst["prediction_min"]),
        })
    return summary


def print_summary(rows, summary) -> None:
    errors = np.array([r["relative_l2"] for r in rows])
    print("")
    print("Positive DeepONet validation relative L2 errors")
    print("------------------------------------------------")
    print(f"Mean:   {errors.mean():.6e}")
    print(f"Median: {np.median(errors):.6e}")
    print(f"Min:    {errors.min():.6e}")
    print(f"Max:    {errors.max():.6e}")
    print(f"Std:    {errors.std():.6e}")

    print("")
    print("Positive DeepONet error by case")
    print("--------------------------------")
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


def save_case_csv(summary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "geometry", "fluid_model", "count", "mean_relative_l2", "median_relative_l2",
        "min_relative_l2", "max_relative_l2", "worst_sample_idx", "worst_prediction_min"
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved: {output_path}")


def plot_histogram(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = np.array([r["relative_l2"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(errors, bins=20)
    ax.axvline(errors.mean(), linestyle="--", linewidth=1.5, label=f"Mean = {errors.mean():.3e}")
    ax.axvline(np.median(errors), linestyle=":", linewidth=1.5, label=f"Median = {np.median(errors):.3e}")
    ax.set_xlabel("Relative L2 error")
    ax.set_ylabel("Validation sample count")
    ax.set_title("Positive DeepONet full-field validation errors")
    ax.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_case_bar(summary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['geometry']}\n{r['fluid_model']}" for r in summary]
    mean_errors = [r["mean_relative_l2"] for r in summary]
    max_errors = [r["max_relative_l2"] for r in summary]
    x = np.arange(len(summary))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    ax.bar(x - width / 2, mean_errors, width, label="Mean relative L2")
    ax.bar(x + width / 2, max_errors, width, label="Max relative L2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Positive DeepONet validation error by case")
    ax.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_best_median_worst(rows, predictions, data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)

    errors = np.array([r["relative_l2"] for r in rows])
    order = np.argsort(errors)
    chosen = [("Best", int(order[0])), ("Median", int(order[len(order) // 2])), ("Worst", int(order[-1]))]
    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)

    for row_id, (label, pos) in enumerate(chosen):
        row = rows[pos]
        sample_idx = int(row["sample_idx"])
        target = velocity_fields[sample_idx]
        pred = predictions[sample_idx]
        abs_err = np.abs(pred - target)
        shared_vmax = max(float(target.max()), float(pred.max()))
        error_vmax = max(float(abs_err.max()), 1e-12)
        panels = [
            ("Target", target, 0.0, shared_vmax),
            ("Prediction", pred, 0.0, shared_vmax),
            ("Absolute error", abs_err, 0.0, error_vmax),
        ]
        for col, (title, field, vmin, vmax) in enumerate(panels):
            ax = axes[row_id, col]
            contour = ax.contourf(z_grid, eta_grid, field, levels=50, vmin=vmin, vmax=vmax)
            ax.set_title(f"{label} sample {sample_idx}\n{title} | rel L2 = {row['relative_l2']:.3e}", fontsize=9)
            ax.set_xlabel(r"$z/L$")
            ax.set_ylabel(r"$\eta = r/R(z)$")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            fig.colorbar(contour, ax=ax, shrink=0.84)

    fig.suptitle("Positive DeepONet validation examples: best, median, and worst", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_prediction_example(
    model,
    sample_idx,
    branch_features,
    velocity_fields,
    z,
    eta,
    branch_mean,
    branch_std,
    target_scale,
    device,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = velocity_fields[sample_idx]
    pred = predict_full_field(model, branch_features[sample_idx], z, eta, branch_mean, branch_std, target_scale, device)
    err = np.abs(pred - target)
    rel = relative_l2(pred, target)
    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")
    vmax = max(float(target.max()), float(pred.max()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    panels = [("Target", target, 0.0, vmax), ("Prediction", pred, 0.0, vmax), ("Absolute error", err, 0.0, float(err.max()))]
    for ax, (title, field, vmin, vmax_panel) in zip(axes, panels):
        contour = ax.contourf(z_grid, eta_grid, field, levels=50, vmin=vmin, vmax=vmax_panel)
        ax.set_title(title)
        ax.set_xlabel(r"$z/L$")
        ax.set_ylabel(r"$\eta = r/R(z)$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        fig.colorbar(contour, ax=ax, shrink=0.9)
    fig.suptitle(f"Positive DeepONet example | sample {sample_idx} | relative L2 = {rel:.3e}", fontsize=12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Example relative L2 error: {rel:.6e}")
    print(f"Prediction min/max: {pred.min():.6e} / {pred.max():.6e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a positive DeepONet.")
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


def main() -> None:
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
    print(f"Target scale:      {target_scale:.6e}")
    print("Constraint:        u = target_scale * (1 - eta^2) * softplus(raw)")

    model = PositiveDeepONet(
        branch_dim=branch_features.shape[1],
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
            rng, train_indices, branch_features, velocity_fields, z, eta,
            args.batch_size, branch_mean, branch_std, target_scale, device
        )

        optimizer.zero_grad()
        pred = model(branch, trunk)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        optimizer.step()

        train_loss = float(loss.detach().cpu().item())
        val_loss = pointwise_val_loss(
            model, rng, val_indices, branch_features, velocity_fields, z, eta,
            args.batch_size, branch_mean, branch_std, target_scale, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:05d} | train MSE = {train_loss:.6e} | val MSE = {val_loss:.6e}")

    Path("models").mkdir(exist_ok=True)
    model_path = Path("models/deeponet_forward_positive.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "branch_mean": branch_mean,
            "branch_std": branch_std,
            "target_scale": target_scale,
            "branch_dim": branch_features.shape[1],
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "z": z,
            "eta": eta,
            "constraint": "u_scaled = (1 - eta^2) * softplus(raw)",
        },
        model_path,
    )
    print(f"Saved model: {model_path}")

    plot_training(train_losses, val_losses, Path("figures/deeponet_positive_training_loss.png"))
    plot_prediction_example(
        model, int(val_indices[0]), branch_features, velocity_fields, z, eta,
        branch_mean, branch_std, target_scale, device,
        Path("figures/deeponet_positive_prediction_example.png")
    )

    rows, predictions = evaluate_validation(
        model, data, branch_features, val_indices, branch_mean, branch_std, target_scale, device
    )
    summary = summarize_by_case(rows)
    print_summary(rows, summary)

    plot_histogram(rows, Path("figures/deeponet_positive_validation_errors.png"))
    plot_best_median_worst(rows, predictions, data, Path("figures/deeponet_positive_best_median_worst_examples.png"))
    save_case_csv(summary, Path("results/deeponet_positive_error_by_case.csv"))
    plot_case_bar(summary, Path("figures/deeponet_positive_error_by_case.png"))


if __name__ == "__main__":
    main()
