"""
Full-field evaluation for the DeepONet forward-operator baseline.

This script evaluates the trained DeepONet on validation samples and saves:

1. figures/deeponet_forward_validation_errors.png
2. figures/deeponet_forward_best_median_worst_examples.png

Run:
    python src/evaluate_deeponet_forward_fullfield.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


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
    """Build branch input features. This must match train_deeponet_forward.py."""
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


def train_val_split(
    n_samples: int,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create the same train/validation split used during training."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_val = max(1, int(val_fraction * n_samples))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return train_indices, val_indices


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


class DeepONet(nn.Module):
    """DeepONet with branch/trunk dot product."""

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int = 2,
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
        """Predict normalized velocity."""
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        return torch.sum(branch_output * trunk_output, dim=1, keepdim=True) + self.bias


def load_model(model_path: Path, device: torch.device):
    """Load saved model and normalization metadata."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train it first with:\n"
            "python src/train_deeponet_forward.py --epochs 500"
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = DeepONet(
        branch_dim=int(checkpoint["branch_dim"]),
        trunk_dim=2,
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        n_layers=int(checkpoint["n_layers"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def predict_full_field(
    model: nn.Module,
    branch_feature: np.ndarray,
    z: np.ndarray,
    eta: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Predict the full u(z, eta) field for one sample."""
    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")

    coords = np.stack(
        [z_grid.reshape(-1), eta_grid.reshape(-1)],
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
        predictions.append(pred.detach().cpu().numpy())

    prediction = np.concatenate(predictions, axis=0).reshape(len(z), len(eta))
    prediction = prediction * target_std + target_mean

    return prediction.astype(np.float32)


def relative_l2_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute relative L2 error."""
    denominator = np.linalg.norm(target)
    if denominator < 1e-12:
        return float(np.linalg.norm(prediction - target))
    return float(np.linalg.norm(prediction - target) / denominator)


def evaluate_validation_samples(
    model: nn.Module,
    data,
    branch_features: np.ndarray,
    val_indices: np.ndarray,
    checkpoint,
    device: torch.device,
    max_samples: int | None,
):
    """Evaluate validation samples and keep predictions for plotting."""
    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)

    branch_mean = checkpoint["branch_mean"]
    branch_std = checkpoint["branch_std"]
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])

    if max_samples is not None:
        val_indices = val_indices[:max_samples]

    errors = []
    predictions = {}

    for count, sample_idx in enumerate(val_indices, start=1):
        target = velocity_fields[sample_idx]

        prediction = predict_full_field(
            model=model,
            branch_feature=branch_features[sample_idx],
            z=z,
            eta=eta,
            branch_mean=branch_mean,
            branch_std=branch_std,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
        )

        error = relative_l2_error(prediction, target)

        errors.append(error)
        predictions[int(sample_idx)] = prediction

        if count % 10 == 0 or count == len(val_indices):
            print(f"Evaluated {count}/{len(val_indices)} validation samples")

    return np.array(errors, dtype=np.float64), predictions, val_indices


def plot_error_histogram(errors: np.ndarray, output_path: Path) -> None:
    """Plot histogram of validation relative L2 errors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    ax.set_title("DeepONet full-field validation errors")
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_best_median_worst_examples(
    errors: np.ndarray,
    predictions: dict,
    val_indices: np.ndarray,
    data,
    output_path: Path,
) -> None:
    """Plot target, prediction, and error for best, median, and worst cases."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)

    order = np.argsort(errors)
    chosen_positions = [
        ("Best", int(order[0])),
        ("Median", int(order[len(order) // 2])),
        ("Worst", int(order[-1])),
    ]

    z_grid, eta_grid = np.meshgrid(z, eta, indexing="ij")

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 10),
        constrained_layout=True,
    )

    for row, (case_label, error_position) in enumerate(chosen_positions):
        sample_idx = int(val_indices[error_position])

        target = velocity_fields[sample_idx]
        prediction = predictions[int(sample_idx)]
        abs_error = np.abs(prediction - target)

        shared_vmax = max(float(target.max()), float(prediction.max()))
        error_vmax = max(float(abs_error.max()), 1e-12)

        panels = [
            ("Target", target, 0.0, shared_vmax),
            ("Prediction", prediction, 0.0, shared_vmax),
            ("Absolute error", abs_error, 0.0, error_vmax),
        ]

        for col, (panel_title, field, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]

            contour = ax.contourf(
                z_grid,
                eta_grid,
                field,
                levels=50,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(
                f"{case_label} sample {sample_idx}\n"
                f"{panel_title} | rel L2 = {errors[error_position]:.3e}",
                fontsize=9,
            )

            ax.set_xlabel(r"$z/L$")
            ax.set_ylabel(r"$\eta = r/R(z)$")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

            fig.colorbar(contour, ax=ax, shrink=0.84)

    fig.suptitle(
        "DeepONet validation examples: best, median, and worst",
        fontsize=14,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DeepONet forward-operator model."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/forward_operator_dataset.npz",
        help="Path to the generated dataset.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/deeponet_forward_baseline.pt",
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction used during training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed used for train/validation split.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of validation samples to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    """Run full-field evaluation."""
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    data = load_dataset(Path(args.dataset))
    branch_features = build_branch_features(data)

    _, val_indices = train_val_split(
        n_samples=branch_features.shape[0],
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    model, checkpoint = load_model(Path(args.model), device=device)

    errors, predictions, evaluated_indices = evaluate_validation_samples(
        model=model,
        data=data,
        branch_features=branch_features,
        val_indices=val_indices,
        checkpoint=checkpoint,
        device=device,
        max_samples=args.max_samples,
    )

    print("")
    print("Validation relative L2 errors")
    print("-----------------------------")
    print(f"Mean:   {errors.mean():.6e}")
    print(f"Median: {np.median(errors):.6e}")
    print(f"Min:    {errors.min():.6e}")
    print(f"Max:    {errors.max():.6e}")
    print(f"Std:    {errors.std():.6e}")

    plot_error_histogram(
        errors=errors,
        output_path=Path("figures/deeponet_forward_validation_errors.png"),
    )

    plot_best_median_worst_examples(
        errors=errors,
        predictions=predictions,
        val_indices=evaluated_indices,
        data=data,
        output_path=Path("figures/deeponet_forward_best_median_worst_examples.png"),
    )


if __name__ == "__main__":
    main()
