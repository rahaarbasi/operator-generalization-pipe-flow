import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from plot_sampling_on_geometry import get_radius, compute_geometry_score

os.makedirs("figures", exist_ok=True)


def wall_stretched_eta(rng, n, k=2.0):
    s = rng.uniform(0.0, 1.0, size=n)
    eta = 1.0 - (np.exp(k * (1.0 - s)) - 1.0) / (np.exp(k) - 1.0)
    return eta


def wall_stretched_eta_pdf(eta, k=2.0):
    eta = np.asarray(eta)
    pdf = (np.exp(k) - 1.0) / (
        k * (1.0 + (1.0 - eta) * (np.exp(k) - 1.0))
    )
    pdf = pdf / np.trapezoid(pdf, eta)
    return pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=str, default="sinusoidal")
    parser.add_argument("--n-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sampling-alpha", type=float, default=1.0)
    parser.add_argument("--sampling-beta", type=float, default=1.0)
    parser.add_argument("--sampling-gamma", type=float, default=1.0)
    parser.add_argument("--eta-stretch-k", type=float, default=2.0)

    parser.add_argument(
        "--output",
        type=str,
        default="figures/combined_sampling_function_sinusoidal_z_equal_eta_wall_stretched.png",
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # z geometry score
    z_grid = np.linspace(0.0, 1.0, 2000)
    R_grid = get_radius(args.geometry, z_grid)

    score, p_z, dR_dz, d2R_dz2 = compute_geometry_score(
        z_grid,
        R_grid,
        alpha=args.sampling_alpha,
        beta=args.sampling_beta,
        gamma=args.sampling_gamma,
    )

    # sample z from score-based p_z
    z = rng.choice(z_grid, size=args.n_points, replace=True, p=p_z)

    # eta from continuous wall-stretched sampling
    eta = wall_stretched_eta(rng, args.n_points, k=args.eta_stretch_k)

    Rz = get_radius(args.geometry, z)
    r = eta * Rz

    # eta pdf for plotting
    eta_grid = np.linspace(0.0, 1.0, 1000)
    pdf_eta = wall_stretched_eta_pdf(eta_grid, k=args.eta_stretch_k)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    # 1. Geometry
    ax = axes[0]
    ax.plot(z_grid, R_grid, linewidth=2, label="R(z)")
    ax.plot(z_grid, -R_grid, linewidth=2, label="-R(z)")
    ax.fill_between(z_grid, -R_grid, R_grid, alpha=0.08)
    ax.set_title(f"Geometry: {args.geometry}")
    ax.set_xlabel("z")
    ax.set_ylabel("r")
    ax.legend()

    # 2. Combined samples on geometry
    ax = axes[1]
    ax.plot(z_grid, R_grid, linewidth=2)
    ax.plot(z_grid, -R_grid, linewidth=2)
    ax.fill_between(z_grid, -R_grid, R_grid, alpha=0.08)
    ax.scatter(z, r, s=5, alpha=0.25)
    ax.scatter(z, -r, s=5, alpha=0.12)
    ax.set_title("Combined z-score + wall-stretched eta samples")
    ax.set_xlabel("z")
    ax.set_ylabel("r")

    # 3. z sampling function
    ax = axes[2]
    ax.plot(z_grid, score / np.max(score), linewidth=2, label="normalized score(z)")
    ax.plot(z_grid, p_z / np.max(p_z), linestyle="--", linewidth=2, label="normalized p_z(z)")
    ax.set_title("z sampling function")
    ax.set_xlabel("z")
    ax.set_ylabel("relative probability")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. z histogram
    ax = axes[3]
    ax.hist(z, bins=70, range=(0, 1), density=True, alpha=0.75, label="sampled z")
    ax.plot(z_grid, p_z / np.max(p_z) * np.max(ax.get_ylim()), linewidth=2, label="scaled p_z(z)")
    ax.set_title("Sampled z distribution")
    ax.set_xlabel("z")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)

    # 5. eta histogram + density
    ax = axes[4]
    ax.hist(eta, bins=70, range=(0, 1), density=True, alpha=0.75, label="sampled eta")
    ax.plot(eta_grid, pdf_eta, linewidth=2.5, label=f"wall-stretched pdf, k={args.eta_stretch_k}")
    ax.axhline(1.0, linestyle=":", label="uniform density")
    ax.set_title("eta sampling distribution")
    ax.set_xlabel("eta = r / R(z)")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. geometry indicators
    ax = axes[5]
    g1 = np.abs(dR_dz)
    g2 = np.abs(d2R_dz2)
    g3 = 1.0 / np.maximum(R_grid, 1e-12)

    ax.plot(z_grid, g1 / np.max(g1 + 1e-12), label="normalized |dR/dz|")
    ax.plot(z_grid, g2 / np.max(g2 + 1e-12), label="normalized |d²R/dz²|")
    ax.plot(z_grid, g3 / np.max(g3 + 1e-12), label="normalized 1/R(z)")
    ax.set_title("Equal-weight geometry indicators")
    ax.set_xlabel("z")
    ax.set_ylabel("relative magnitude")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Combined sampling: equal-weight z score + continuous wall-stretched eta (k={args.eta_stretch_k})",
        fontsize=15,
    )

    plt.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
