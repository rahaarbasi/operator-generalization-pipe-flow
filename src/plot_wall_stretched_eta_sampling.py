import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

rng = np.random.default_rng(42)

n = 30000
stretch_k = 2.0

# Continuous mesh-inspired wall-stretched sampling
# s = 0 -> eta = 0 centerline
# s = 1 -> eta = 1 wall
s = rng.uniform(0.0, 1.0, size=n)
eta = 1.0 - (np.exp(stretch_k * (1.0 - s)) - 1.0) / (np.exp(stretch_k) - 1.0)

# Analytical density for the mapping
x = np.linspace(0.0, 1.0, 1000)
pdf = (np.exp(stretch_k) - 1.0) / (
    stretch_k * (1.0 + (1.0 - x) * (np.exp(stretch_k) - 1.0))
)
pdf = pdf / np.trapezoid(pdf, x)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(eta, bins=80, range=(0, 1), density=True, alpha=0.75)
axes[0].set_title(f"Continuous wall-stretched eta sampling\nstretch k={stretch_k:.1f}")
axes[0].set_xlabel("eta = r / R(z)")
axes[0].set_ylabel("density")
axes[0].grid(alpha=0.3)

axes[1].plot(x, pdf, linewidth=2.5, label="continuous wall-stretched density")
axes[1].axhline(1.0, linestyle=":", label="uniform density")
axes[1].set_title("Mesh-inspired continuous radial sampling")
axes[1].set_xlabel("eta = r / R(z)")
axes[1].set_ylabel("density")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
path = "figures/wall_stretched_eta_sampling_continuous_k2.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
print(f"Saved: {path}")
