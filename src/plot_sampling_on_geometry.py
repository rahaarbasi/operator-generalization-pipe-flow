import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Geometry definitions
# NOTE:
# If your repo already has exact geometry formulas in src/geometries.py,
# replace these with imports from there.
# ============================================================

def radius_straight(z):
    return np.ones_like(z)


def radius_sinusoidal(z, base=1.0, amp=0.25):
    return base + amp * np.sin(2.0 * np.pi * z)


def radius_stenosed(z, base=1.0, depth=0.35, center=0.5, width=0.12):
    return base - depth * np.exp(-((z - center) ** 2) / (2.0 * width ** 2))


def radius_expanded(z, base=1.0, height=0.35, center=0.5, width=0.12):
    return base + height * np.exp(-((z - center) ** 2) / (2.0 * width ** 2))


def radius_hyperbolic_constriction(z, base=1.0, depth=0.35, center=0.5, width=0.18):
    return base - depth / (1.0 + ((z - center) / width) ** 2)


def get_radius(geometry_name, z):
    if geometry_name == "straight":
        return radius_straight(z)
    elif geometry_name == "sinusoidal":
        return radius_sinusoidal(z)
    elif geometry_name == "stenosed":
        return radius_stenosed(z)
    elif geometry_name == "expanded":
        return radius_expanded(z)
    elif geometry_name == "hyperbolic_constriction":
        return radius_hyperbolic_constriction(z)
    else:
        raise ValueError(f"Unknown geometry: {geometry_name}")


# ============================================================
# Sampling helpers
# ============================================================

def normalize_nonnegative(x, eps=1e-12):
    x = np.asarray(x)
    x = x - np.min(x)
    xmax = np.max(x)
    if xmax < eps:
        return np.ones_like(x)
    return x / xmax


def compute_geometry_score(z_grid, R_grid, alpha=1.0, beta=0.5, gamma=0.5):
    dR_dz = np.gradient(R_grid, z_grid)
    d2R_dz2 = np.gradient(dR_dz, z_grid)

    term1 = normalize_nonnegative(np.abs(dR_dz))
    term2 = normalize_nonnegative(np.abs(d2R_dz2))
    term3 = normalize_nonnegative(1.0 / np.maximum(R_grid, 1e-12))

    score = 1.0 + alpha * term1 + beta * term2 + gamma * term3
    score = np.maximum(score, 1e-12)
    p = score / np.sum(score)
    return score, p, dR_dz, d2R_dz2


def sample_z_uniform(rng, n_points):
    return rng.uniform(0.0, 1.0, size=n_points)


def sample_z_geometry_aware(
    rng,
    n_points,
    z_grid,
    prob_grid,
    adaptive_fraction=0.30,
):
    n_adaptive = int(adaptive_fraction * n_points)
    n_uniform = n_points - n_adaptive

    z_uniform = rng.uniform(0.0, 1.0, size=n_uniform)
    z_adaptive = rng.choice(z_grid, size=n_adaptive, replace=True, p=prob_grid)

    z = np.concatenate([z_uniform, z_adaptive])
    rng.shuffle(z)
    return z


def sample_eta_uniform(rng, n_points):
    return rng.uniform(0.0, 1.0, size=n_points)


def sample_eta_geometry_aware(
    rng,
    n_points,
    wall_fraction=0.25,
    center_fraction=0.25,
):
    n_wall = int(wall_fraction * n_points)
    n_center = int(center_fraction * n_points)
    n_uniform = n_points - n_wall - n_center

    # near wall => eta ~ 1
    eta_wall = rng.beta(a=8.0, b=2.0, size=n_wall)

    # near centerline => eta ~ 0
    eta_center = rng.beta(a=2.0, b=8.0, size=n_center)

    eta_uniform = rng.uniform(0.0, 1.0, size=n_uniform)

    eta = np.concatenate([eta_uniform, eta_wall, eta_center])
    rng.shuffle(eta)
    return eta


def sample_points(
    geometry_name,
    n_points,
    seed,
    sampling_mode,
    adaptive_z_fraction=0.30,
    alpha=1.0,
    beta=0.5,
    gamma=0.5,
    eta_wall_fraction=0.25,
    eta_center_fraction=0.25,
):
    rng = np.random.default_rng(seed)

    z_grid = np.linspace(0.0, 1.0, 2000)
    R_grid = get_radius(geometry_name, z_grid)
    score, prob_grid, dR_dz, d2R_dz2 = compute_geometry_score(
        z_grid, R_grid, alpha=alpha, beta=beta, gamma=gamma
    )

    if sampling_mode == "uniform":
        z = sample_z_uniform(rng, n_points)
        eta = sample_eta_uniform(rng, n_points)
    elif sampling_mode == "geometry_aware":
        z = sample_z_geometry_aware(
            rng,
            n_points,
            z_grid,
            prob_grid,
            adaptive_fraction=adaptive_z_fraction,
        )
        eta = sample_eta_geometry_aware(
            rng,
            n_points,
            wall_fraction=eta_wall_fraction,
            center_fraction=eta_center_fraction,
        )
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    Rz = get_radius(geometry_name, z)
    r = eta * Rz

    return {
        "z": z,
        "eta": eta,
        "r": r,
        "z_grid": z_grid,
        "R_grid": R_grid,
        "score": score,
        "prob_grid": prob_grid,
        "dR_dz": dR_dz,
        "d2R_dz2": d2R_dz2,
    }


# ============================================================
# Plotting
# ============================================================

def plot_geometry_boundaries(ax, z_grid, R_grid, title=None):
    ax.plot(z_grid, R_grid, linewidth=2, label="+R(z)")
    ax.plot(z_grid, -R_grid, linewidth=2, label="-R(z)")
    ax.fill_between(z_grid, -R_grid, R_grid, alpha=0.08)
    ax.set_xlim(0.0, 1.0)
    y_max = 1.1 * np.max(R_grid)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("z")
    ax.set_ylabel("r")
    if title is not None:
        ax.set_title(title)


def scatter_sampling(ax, z_grid, R_grid, z, r, title=None):
    plot_geometry_boundaries(ax, z_grid, R_grid, title=title)
    ax.scatter(z, r, s=8, alpha=0.35)
    ax.scatter(z, -r, s=8, alpha=0.15)  # mirror for full pipe intuition


def plot_score(ax, z_grid, score, prob_grid, title=None):
    ax.plot(z_grid, score, linewidth=2, label="score(z)")
    ax.plot(z_grid, prob_grid / np.max(prob_grid), linewidth=2, linestyle="--", label="normalized p(z)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("z")
    ax.set_ylabel("relative magnitude")
    if title is not None:
        ax.set_title(title)
    ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=str, default="sinusoidal")
    parser.add_argument("--n-points", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--adaptive-z-fraction", type=float, default=0.30)
    parser.add_argument("--sampling-alpha", type=float, default=1.0)
    parser.add_argument("--sampling-beta", type=float, default=0.5)
    parser.add_argument("--sampling-gamma", type=float, default=0.5)
    parser.add_argument("--eta-wall-fraction", type=float, default=0.25)
    parser.add_argument("--eta-center-fraction", type=float, default=0.25)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. If omitted, auto-generated in figures/.",
    )

    args = parser.parse_args()

    uniform = sample_points(
        geometry_name=args.geometry,
        n_points=args.n_points,
        seed=args.seed,
        sampling_mode="uniform",
        adaptive_z_fraction=args.adaptive_z_fraction,
        alpha=args.sampling_alpha,
        beta=args.sampling_beta,
        gamma=args.sampling_gamma,
        eta_wall_fraction=args.eta_wall_fraction,
        eta_center_fraction=args.eta_center_fraction,
    )

    geometry_aware = sample_points(
        geometry_name=args.geometry,
        n_points=args.n_points,
        seed=args.seed,
        sampling_mode="geometry_aware",
        adaptive_z_fraction=args.adaptive_z_fraction,
        alpha=args.sampling_alpha,
        beta=args.sampling_beta,
        gamma=args.sampling_gamma,
        eta_wall_fraction=args.eta_wall_fraction,
        eta_center_fraction=args.eta_center_fraction,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    plot_geometry_boundaries(
    axes[0],
    uniform["z_grid"],
    uniform["R_grid"],
    title=f"Geometry: {args.geometry}",
    )

    scatter_sampling(
    axes[1],
    uniform["z_grid"],
    uniform["R_grid"],
    uniform["z"],
    uniform["r"],
    title="Uniform sampling",
    )

    scatter_sampling(
    axes[2],
    geometry_aware["z_grid"],
    geometry_aware["R_grid"],
    geometry_aware["z"],
    geometry_aware["r"],
    title="Geometry-aware sampling",
    )

    plot_z_histogram(
    axes[3],
    uniform,
    geometry_aware,
    )

    plot_eta_histogram(
    axes[4],
    uniform,
    geometry_aware,
    )

    plot_score(
    axes[5],
    geometry_aware["z_grid"],
    geometry_aware["score"],
    geometry_aware["prob_grid"],
    title="Sampling score along z",
    )
    
    fig.suptitle(
        f"Sampling diagnostic on geometry '{args.geometry}' (n={args.n_points}, seed={args.seed})",
        fontsize=16,
    )
    plt.tight_layout()

    if args.output is None:
        os.makedirs("figures", exist_ok=True)
        output_path = f"figures/sampling_diagnostic_{args.geometry}_n{args.n_points}_seed{args.seed}.png"
    else:
        output_path = args.output

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")

def plot_z_histogram(ax, uniform, geometry_aware, bins=60):
    ax.hist(
        uniform["z"],
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        alpha=0.45,
        label="uniform",
    )
    ax.hist(
        geometry_aware["z"],
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        alpha=0.45,
        label="geometry-aware",
    )
    ax.set_xlabel("z")
    ax.set_ylabel("density")
    ax.set_title("Sampled z distribution")
    ax.legend()


def plot_eta_histogram(ax, uniform, geometry_aware, bins=60):
    ax.hist(
        uniform["eta"],
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        alpha=0.45,
        label="uniform",
    )
    ax.hist(
        geometry_aware["eta"],
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        alpha=0.45,
        label="geometry-aware",
    )
    ax.set_xlabel("eta = r / R(z)")
    ax.set_ylabel("density")
    ax.set_title("Sampled eta distribution")
    ax.legend()

if __name__ == "__main__":
    main()