#!/usr/bin/env python3
"""
Run exactly one cell of the paper cross-grid experiment.

Folder convention:
runs/{arch}__{sampling}__holdout-{geom}__seed-{n}/

Architectures:
    unconstrained | powerlaw | geoaware

Sampling:
    uniform | geo-axial | geo-wallstretched

Held-out geometry:
    straight | stenosed | expanded | sinusoidal | hyperbolic
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from train_deeponet_leave_one_geometry_out_sampling_ablation import (
    MODEL_LABELS,
    build_branch_features,
    build_effective_flow_index,
    build_geometry_sampling_weights,
    build_model,
    build_radius_derivative,
    evaluate_pointwise_loss,
    get_device,
    load_dataset,
    make_normalization,
    predict_full_field,
    relative_l2_error,
    sample_training_batch,
    split_train_internal_validation,
)


ARCH_TO_MODEL = {
    "unconstrained": "unconstrained",
    "powerlaw": "power_law_aware",
    "geoaware": "geometry_aware",
}

GEOM_TO_DATASET = {
    "straight": "straight",
    "stenosed": "stenosed",
    "expanded": "expanded",
    "sinusoidal": "sinusoidal",
    "hyperbolic": "hyperbolic_constriction",
}


def parse_args():
    p = argparse.ArgumentParser(description="Run one cross-grid cell.")
    p.add_argument("--arch", required=True, choices=ARCH_TO_MODEL)
    p.add_argument(
        "--sampling",
        required=True,
        choices=["uniform", "geo-axial", "geo-wallstretched"],
    )
    p.add_argument("--held-out", required=True, choices=GEOM_TO_DATASET)
    p.add_argument("--seed", required=True, type=int)

    p.add_argument("--dataset", default="data/forward_operator_dataset.npz")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--internal-val-fraction", type=float, default=0.15)
    p.add_argument("--print-every", type=int, default=50)

    # Geometry-aware axial sampling hyperparameters: preserve current experiment defaults.
    p.add_argument("--adaptive-z-fraction", type=float, default=0.30)
    p.add_argument("--sampling-alpha", type=float, default=1.0)
    p.add_argument("--sampling-beta", type=float, default=0.5)
    p.add_argument("--sampling-gamma", type=float, default=0.5)
    p.add_argument("--eta-stretch-k", type=float, default=2.0)

    p.add_argument("--runs-dir", default="runs")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing run folder.",
    )
    return p.parse_args()


def underlying_sampling(label: str):
    if label == "uniform":
        return "uniform", "uniform"
    if label == "geo-axial":
        return "geometry_aware", "uniform"
    if label == "geo-wallstretched":
        return "geometry_aware", "wall_stretched"
    raise ValueError(label)


def run_folder_name(arch: str, sampling: str, held_out: str, seed: int) -> str:
    return f"{arch}__{sampling}__holdout-{held_out}__seed-{seed:02d}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    def command(args):
        try:
            return subprocess.check_output(
                args, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    commit = command(["git", "rev-parse", "HEAD"])
    status = command(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def hardware_info(device: torch.device):
    info = {
        "device": str(device),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if device.type == "cuda":
        info["accelerator_name"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        info["accelerator_name"] = "Apple Metal Performance Shaders (MPS)"
    else:
        info["accelerator_name"] = platform.processor() or "CPU"
    return info


def normalization_to_dict(normalization):
    return {
        "branch_mean": normalization.branch_mean,
        "branch_std": normalization.branch_std,
        "target_scale": normalization.target_scale,
        "radius_mean": normalization.radius_mean,
        "radius_std": normalization.radius_std,
        "derivative_mean": normalization.derivative_mean,
        "derivative_std": normalization.derivative_std,
    }


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_model(
    *,
    model_kind,
    branch_features,
    n_eff_values,
    radius_profiles,
    radius_derivatives,
    velocity_fields,
    z,
    eta,
    train_pool_indices,
    internal_val_indices,
    normalization,
    args,
    device,
    geometry_sampling_weights,
    sampling_mode,
    eta_sampling_mode,
    train_curve_path,
):
    rng = np.random.default_rng(args.seed)

    model = build_model(
        model_kind=model_kind,
        branch_dim=branch_features.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    with train_curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_mse", "internal_val_mse"],
        )
        writer.writeheader()

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
                velocity_fields=velocity_fields,
                z=z,
                eta=eta,
                batch_size=args.batch_size,
                normalization=normalization,
                device=device,
                sampling=sampling_mode,
                geometry_sampling_weights=geometry_sampling_weights,
                adaptive_z_fraction=args.adaptive_z_fraction,
                eta_wall_fraction=0.25,
                eta_center_fraction=0.25,
                eta_sampling=eta_sampling_mode,
                eta_stretch_k=args.eta_stretch_k,
            )

            optimizer.zero_grad()
            prediction = model(branch_tensor, trunk_tensor)
            loss = torch.mean((prediction - target_tensor) ** 2)
            loss.backward()
            optimizer.step()

            train_loss = float(loss.detach().cpu().item())
            val_loss = ""

            # Preserve the validation cadence of the existing experiment so
            # validation RNG calls do not change the training trajectory.
            if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
                val_loss_value = evaluate_pointwise_loss(
                    model=model,
                    rng=rng,
                    model_kind=model_kind,
                    sample_indices_pool=internal_val_indices,
                    branch_features=branch_features,
                    n_eff_values=n_eff_values,
                    radius_profiles=radius_profiles,
                    radius_derivatives=radius_derivatives,
                    velocity_fields=velocity_fields,
                    z=z,
                    eta=eta,
                    batch_size=args.batch_size,
                    normalization=normalization,
                    device=device,
                )
                val_loss = float(val_loss_value)
                print(
                    f"Epoch {epoch:05d} | "
                    f"train MSE = {train_loss:.6e} | "
                    f"internal val MSE = {val_loss:.6e}"
                )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_mse": f"{train_loss:.12e}",
                    "internal_val_mse": (
                        "" if val_loss == "" else f"{val_loss:.12e}"
                    ),
                }
            )
            f.flush()

    return model


@torch.no_grad()
def evaluate_and_collect(
    *,
    model,
    model_kind,
    held_out_indices,
    data,
    branch_features,
    n_eff_values,
    radius_profiles,
    radius_derivatives,
    normalization,
    device,
):
    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)
    fluid_models = data["fluid_models"]
    fluid_codes = data["fluid_codes"]
    geometry_names = data["geometry_names"]
    geometry_codes = data["geometry_codes"]

    rows = []
    predictions = []
    targets = []

    for count, sample_idx in enumerate(held_out_indices, start=1):
        sample_idx = int(sample_idx)
        target = velocity_fields[sample_idx]

        prediction = predict_full_field(
            model=model,
            model_kind=model_kind,
            sample_idx=sample_idx,
            branch_features=branch_features,
            n_eff_values=n_eff_values,
            radius_profiles=radius_profiles,
            radius_derivatives=radius_derivatives,
            z=z,
            eta=eta,
            normalization=normalization,
            device=device,
        )

        rel_l2 = relative_l2_error(prediction, target)

        rows.append(
            {
                "sample_idx": sample_idx,
                "geometry": str(geometry_names[geometry_codes[sample_idx]]),
                "fluid_model": str(fluid_models[fluid_codes[sample_idx]]),
                "relative_l2": float(rel_l2),
                "target_max": float(target.max()),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
                "has_negative_prediction": bool(prediction.min() < -1e-8),
            }
        )
        predictions.append(prediction)
        targets.append(target)

        if count % 10 == 0 or count == len(held_out_indices):
            print(f"Evaluated {count}/{len(held_out_indices)} held-out samples")

    return (
        rows,
        np.stack(predictions).astype(np.float32),
        np.stack(targets).astype(np.float32),
    )


def save_per_sample(rows, path: Path):
    if not rows:
        raise RuntimeError("No held-out rows were produced.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")

    set_seeds(args.seed)

    model_kind = ARCH_TO_MODEL[args.arch]
    held_out_dataset_name = GEOM_TO_DATASET[args.held_out]
    sampling_mode, eta_sampling_mode = underlying_sampling(args.sampling)

    run_dir = Path(args.runs_dir) / run_folder_name(
        args.arch, args.sampling, args.held_out, args.seed
    )

    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Run folder already exists: {run_dir}\n"
                "Refusing to overwrite. Use --overwrite only intentionally."
            )
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=False)

    dataset_path = Path(args.dataset)
    device = get_device()
    hardware = hardware_info(device)
    git = git_info()

    config = {
        "arch": args.arch,
        "model_kind": model_kind,
        "sampling": args.sampling,
        "sampling_mode": sampling_mode,
        "eta_sampling_mode": eta_sampling_mode,
        "held_out": args.held_out,
        "held_out_dataset_name": held_out_dataset_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "n_layers": args.n_layers,
        "internal_val_fraction": args.internal_val_fraction,
        "print_every": args.print_every,
        "adaptive_z_fraction": args.adaptive_z_fraction,
        "sampling_alpha": args.sampling_alpha,
        "sampling_beta": args.sampling_beta,
        "sampling_gamma": args.sampling_gamma,
        "eta_stretch_k": args.eta_stretch_k,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "hardware": hardware,
    }

    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Run folder: {run_dir}")
    print(f"Architecture: {args.arch} ({MODEL_LABELS[model_kind]})")
    print(f"Sampling: {args.sampling}")
    print(f"Held out: {args.held_out} -> {held_out_dataset_name}")
    print(f"Seed: {args.seed:02d}")
    print(f"Device: {device}")

    total_start = time.perf_counter()

    data = load_dataset(dataset_path)
    z = data["z"].astype(np.float32)
    eta = data["eta"].astype(np.float32)
    velocity_fields = data["velocity_fields"].astype(np.float32)
    radius_profiles = data["radius_profiles"].astype(np.float32)
    geometry_names = data["geometry_names"]
    geometry_codes = data["geometry_codes"]

    branch_features = build_branch_features(data)
    n_eff_values = build_effective_flow_index(data)
    radius_derivatives = build_radius_derivative(radius_profiles, z)

    geometry_name_to_code = {
        str(name): int(code)
        for code, name in enumerate(geometry_names)
    }

    if held_out_dataset_name not in geometry_name_to_code:
        available = ", ".join(str(name) for name in geometry_names)
        raise ValueError(
            f"Dataset does not contain geometry '{held_out_dataset_name}'. "
            f"Available: {available}"
        )

    held_out_code = geometry_name_to_code[held_out_dataset_name]
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
        train_indices=train_pool_indices,
    )

    geometry_sampling_weights = None
    if sampling_mode == "geometry_aware":
        geometry_sampling_weights = build_geometry_sampling_weights(
            radius_profiles=radius_profiles,
            radius_derivatives=radius_derivatives,
            z=z,
            alpha=args.sampling_alpha,
            beta=args.sampling_beta,
            gamma=args.sampling_gamma,
        )

    training_start = time.perf_counter()
    model = train_one_model(
        model_kind=model_kind,
        branch_features=branch_features,
        n_eff_values=n_eff_values,
        radius_profiles=radius_profiles,
        radius_derivatives=radius_derivatives,
        velocity_fields=velocity_fields,
        z=z,
        eta=eta,
        train_pool_indices=train_pool_indices,
        internal_val_indices=internal_val_indices,
        normalization=normalization,
        args=args,
        device=device,
        geometry_sampling_weights=geometry_sampling_weights,
        sampling_mode=sampling_mode,
        eta_sampling_mode=eta_sampling_mode,
        train_curve_path=run_dir / "train_curve.csv",
    )
    training_seconds = time.perf_counter() - training_start

    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": args.arch,
            "model_kind": model_kind,
            "normalization": normalization_to_dict(normalization),
            "config": config,
        },
        run_dir / "model.pt",
    )

    evaluation_start = time.perf_counter()
    rows, predictions, targets = evaluate_and_collect(
        model=model,
        model_kind=model_kind,
        held_out_indices=held_out_indices,
        data=data,
        branch_features=branch_features,
        n_eff_values=n_eff_values,
        radius_profiles=radius_profiles,
        radius_derivatives=radius_derivatives,
        normalization=normalization,
        device=device,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    save_per_sample(rows, run_dir / "per_sample.csv")

    np.savez_compressed(
        run_dir / "predictions.npz",
        sample_indices=held_out_indices.astype(np.int64),
        predictions=predictions,
        targets=targets,
        z=z,
        eta=eta,
    )

    total_seconds = time.perf_counter() - total_start
    timing = {
        "wall_clock_seconds": total_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "device": str(device),
        "accelerator_name": hardware["accelerator_name"],
    }
    with (run_dir / "timing.json").open("w") as f:
        json.dump(timing, f, indent=2)

    print("")
    print("Completed successfully.")
    print(f"Saved six standard outputs in: {run_dir}")
    for filename in (
        "config.json",
        "per_sample.csv",
        "predictions.npz",
        "train_curve.csv",
        "model.pt",
        "timing.json",
    ):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()