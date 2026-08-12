# 4. Dataset

## 4.1 Geometry Families

The dataset contains five parametric pipe-geometry families:

* Straight pipe
* Stenosed pipe
* Expanded pipe
* Sinusoidal pipe
* Hyperbolic-constriction pipe

Each geometry is represented by a radius profile R(z) defined along the normalized axial coordinate z ∈ [0,1].

The geometry families were selected to provide increasing geometric complexity while remaining analytically interpretable.

---

## 4.2 Fluid Models

Two fluid classes are considered:

### Newtonian fluids

The viscosity is constant throughout the domain.

### Power-law fluids

The constitutive relation is:

τ = K γ̇ⁿ

where K is the consistency index and n is the flow index.

Both shear-thinning and near-Newtonian behaviors are represented in the dataset.

---

## 4.3 Forward Operator

The learning task is:

R(z), fluid parameters, ΔP, (z,η) → u(z,η)

where:

* R(z) is the geometry profile,
* fluid parameters are μ for Newtonian fluids or (K,n) for power-law fluids,
* ΔP is the pressure drop,
* η = r / R(z) is the normalized radial coordinate,
* u(z,η) is the axial velocity field.

The velocity fields are generated using an analytical forward solver.

---

## 4.4 Training and Evaluation Protocol

The primary evaluation uses a leave-one-geometry-family-out strategy.

For each experiment:

* one geometry family is completely excluded from training,
* all remaining geometry families are used for training,
* testing is performed only on the held-out geometry family.

This protocol evaluates geometry extrapolation rather than interpolation.

The sinusoidal geometry family is used as the primary benchmark because it consistently produces the largest extrapolation errors.

---

## 4.5 Sampling Strategies

The computational domain is represented in normalized coordinates:

(z,η)

with:

z ∈ [0,1]
η ∈ [0,1]

Several sampling strategies are investigated.

### Uniform Sampling

Training points are sampled uniformly in both z and η.

This serves as the baseline strategy.

### Geometry-Aware Axial Sampling

Axial locations are sampled according to local geometric variation.

Regions with stronger geometric deformation receive more training samples.

The goal is to allocate learning capacity to locations where velocity gradients are expected to be larger.

### Wall-Stretched Radial Sampling

Radial samples are concentrated near η = 1.

This targets the near-wall region where velocity gradients are strongest and where geometry-induced effects are most significant.

### Combined Sampling

Geometry-aware axial sampling and wall-stretched radial sampling are combined.

This produces the most physically informed training-point distribution considered in this work.

---

## 4.6 Dataset Motivation

The dataset is intentionally simple enough to isolate the effects of:

* operator architecture,
* physical inductive bias,
* sampling strategy,
* geometry extrapolation.

This allows a controlled study of how neural operators behave when confronted with previously unseen geometry families.
