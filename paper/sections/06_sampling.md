# 6. Sampling Strategies

## 6.1 Motivation

Neural operators are typically trained using samples drawn from a computational domain.

For geometry-varying flows, the distribution of training points may strongly influence the learned operator because velocity gradients are not uniformly distributed throughout the domain.

Two observations motivate the sampling study:

1. Velocity gradients are generally strongest near solid walls.
2. Geometry-induced flow variations are concentrated near regions where the pipe radius changes rapidly.

Therefore, uniform sampling may allocate excessive training effort to regions with relatively little physical variation.

The objective of this work is to determine whether physically informed sampling strategies improve geometry extrapolation.

---

## 6.2 Uniform Sampling

The baseline strategy samples points uniformly in both coordinates:

(z,\eta)

with:

z ∈ [0,1]

η ∈ [0,1]

All locations have equal probability of being selected.

This approach is simple and widely used in operator-learning benchmarks.

---

## 6.3 Geometry-Aware Axial Sampling

Geometry-aware sampling concentrates training points in regions with stronger geometric variation.

The sampling density is increased in locations where:

|dR/dz|

is large.

As a result:

* constrictions receive more samples,
* expansions receive more samples,
* nearly straight regions receive fewer samples.

The goal is to allocate training effort to physically important locations.

---

## 6.4 Wall-Stretched Radial Sampling

Velocity gradients are largest near the wall.

To better resolve near-wall behavior, radial coordinates are sampled using a stretched distribution that concentrates samples near:

η = 1

while maintaining coverage of the entire cross-section.

Compared with uniform sampling, this strategy places more training points in regions where velocity changes most rapidly.

---

## 6.5 Combined Sampling

The most physically informed strategy combines:

* geometry-aware sampling in the axial direction,
* wall-stretched sampling in the radial direction.

This produces a training distribution that preferentially samples:

* regions of strong geometric variation,
* regions of strong velocity gradients.

The combined strategy represents the strongest sampling prior investigated in this study.

---

## 6.6 Sampling Versus Inductive Bias

The central hypothesis of this work is that sampling and inductive bias play different roles.

Sampling affects:

* where information is observed during training.

Inductive bias affects:

* how the model represents physical structure.

The experiments are designed to determine which of these factors has the larger impact on geometry extrapolation.

Specifically, we compare:

* improvements obtained from geometry-aware sampling,
* improvements obtained from power-law-aware constraints,
* improvements obtained from geometry-channel-aware architectures.

This allows the relative importance of data allocation and architectural design to be quantified under identical training conditions.
