# Paper Outline Draft

## Working title

**Physics-constrained DeepONets for geometry-generalizable surrogate modeling of non-Newtonian pipe flow**

Alternative titles:

1. **Geometry extrapolation in physics-constrained neural operators for non-Newtonian pipe flow**
2. **Rheology-aware and geometry-aware DeepONets for surrogate modeling of variable-radius pipe flows**
3. **Physics-informed operator learning for generalized Newtonian flow in axisymmetric geometries**

---

## Central research question

Can physics-constrained neural operators learn surrogate mappings for pressure-driven pipe flow that generalize across fluid rheology and variable-radius geometries?

More specifically:

* Can a DeepONet predict velocity fields (u(z,\eta)) from a geometry profile (R(z)), pressure forcing, and fluid parameters?
* Do hard physical constraints improve reliability compared with an unconstrained DeepONet?
* Do local geometry features improve generalization to unseen geometry families?
* Is geometry extrapolation harder than random-split interpolation?

---

## Proposed contribution

This work develops and evaluates DeepONet-based forward surrogates for pressure-driven Newtonian and power-law pipe flows in variable-radius axisymmetric geometries.

The main contributions are:

1. A reduced forward-model dataset for Newtonian and power-law pipe flows across multiple variable-radius geometries.
2. A comparison of unconstrained, positivity-constrained, power-law-aware, and geometry-aware DeepONet architectures.
3. A leave-one-geometry-family-out generalization benchmark for testing extrapolation to unseen geometry classes.
4. Evidence that hard physical constraints eliminate nonphysical negative velocity predictions.
5. Evidence that geometry-aware features improve average extrapolation performance in most held-out geometries, while simpler rheology-aware constraints can be more robust in difficult extrapolation regimes.

---

## Abstract draft

Neural operator methods provide a promising route for fast surrogate modeling of parameterized fluid-flow problems, but their reliability under geometry extrapolation remains an important challenge. In this work, we study DeepONet-based surrogate models for pressure-driven Newtonian and power-law pipe flows in variable-radius axisymmetric geometries. The models learn the mapping from a radius profile (R(z)), pressure forcing, and fluid parameters to the normalized velocity field (u(z,\eta)), where (\eta=r/R(z)).

We compare an unconstrained DeepONet with physically constrained variants that enforce nonnegative velocity and no-slip behavior at the wall. We further evaluate rheology-aware and geometry-aware constraints by incorporating the effective power-law exponent and local geometric quantities such as (R(z)) and (dR/dz). In random train-validation splits, geometry-aware constraints improve average prediction accuracy. However, leave-one-geometry-family-out tests reveal that geometry extrapolation is substantially harder than interpolation within known geometry families. The unconstrained model frequently produces nonphysical negative velocities under geometry extrapolation, while constrained models eliminate these failures. Across held-out geometry families, the geometry-aware model achieves the best mean error in most cases, whereas the simpler power-law-aware model provides stronger robustness for the difficult held-out sinusoidal case.

These results suggest that physics-based inductive biases are essential for reliable neural-operator surrogates in geometry-parametric flow problems, and that geometry-aware features improve interpolation and many extrapolation settings but do not automatically guarantee robust generalization to all unseen geometry families.

---

## 1. Introduction

### Motivation

Fast surrogate models are increasingly important for computational fluid dynamics, inverse problems, biomedical flows, and design optimization. Full numerical simulation can be expensive when many geometries, pressure conditions, or material parameters must be evaluated.

Neural operators, including DeepONets, are attractive because they aim to learn mappings between functions rather than only finite-dimensional input-output pairs. For pipe and vessel-like flows, a natural operator-learning task is to map a geometry profile and fluid parameters to a velocity field.

### Problem

A major challenge is geometry generalization. A surrogate may perform well when train and test samples come from the same geometry distribution, but fail when tested on an unseen geometry family.

This work asks whether hard physical constraints and geometry-aware features improve reliability under this harder extrapolation setting.

### Research gap

Many neural-operator demonstrations rely on random train-test splits. These splits mainly evaluate interpolation within the sampled data distribution. For geometry-parametric flow problems, a more demanding test is whether the model can generalize to a geometry family excluded from training.

### Our approach

We construct a controlled benchmark using pressure-driven Newtonian and power-law pipe flows in several variable-radius geometries. We train DeepONet variants and compare them under both random-split validation and leave-one-geometry-family-out extrapolation.

---

## 2. Physical problem

### Geometry

We consider axisymmetric pipe-like geometries parameterized by an axial coordinate (z) and a radius profile (R(z)). The radial coordinate is normalized as:

[
\eta = \frac{r}{R(z)}
]

so that all geometries are represented on the fixed computational domain:

[
z \in [0,1], \qquad \eta \in [0,1].
]

The model predicts:

[
u(z,\eta)
]

rather than directly predicting on a variable physical radial grid.

### Fluid models

Two fluid classes are considered:

1. Newtonian fluids.
2. Power-law generalized Newtonian fluids.

The dataset includes pressure forcing and fluid parameters such as viscosity for Newtonian cases and consistency/index parameters for power-law cases.

### Geometry families

The current benchmark includes:

* straight
* stenosed
* expanded
* sinusoidal
* hyperbolic constriction

These geometry families provide a controlled setting for testing both interpolation and extrapolation.

---

## 3. Dataset generation

### Forward solver

A reduced forward solver is used to generate velocity fields for each geometry-fluid-parameter sample. Each sample contains:

* axial grid (z)
* normalized radial grid (\eta)
* radius profile (R(z))
* pressure forcing
* fluid parameters
* velocity field (u(z,\eta))

### Operator-learning input-output structure

The learned operator maps:

[
R(z), \Delta P, \text{fluid parameters}, (z,\eta) \rightarrow u(z,\eta).
]

The geometry profile (R(z)) is provided as a functional input to the branch network. The pointwise coordinate ((z,\eta)) is provided to the trunk network.

### Important clarification

In leave-one-geometry-family-out tests, the held-out geometry profile (R(z)) is still provided at test time. The model is not asked to guess the geometry. Instead, the model is tested on whether it can use a radius profile from a geometry family never seen during training.

---

## 4. DeepONet architectures

### 4.1 Unconstrained DeepONet

The baseline DeepONet receives the global geometry profile and fluid parameters through the branch network and the coordinate ((z,\eta)) through the trunk network.

It has no hard physical constraint on the output. Therefore, it can produce negative velocity predictions.

### 4.2 Positive/no-slip constrained DeepONet

A constrained version multiplies the network output by a radial wall factor and applies a positive activation to enforce:

* nonnegative velocity
* zero velocity at the wall

This removes nonphysical negative predictions but may be too restrictive if the assumed radial structure is not sufficiently expressive.

### 4.3 Power-law-aware constrained DeepONet

The power-law-aware model introduces an effective flow index (n_{\mathrm{eff}}) into the trunk input and uses a rheology-aware wall factor.

This model enforces physical admissibility while adapting the radial structure to Newtonian and power-law cases.

### 4.4 Geometry-aware constrained DeepONet

The geometry-aware model further includes local geometric quantities in the trunk input:

[
R(z), \qquad \frac{dR}{dz}.
]

This gives the model direct pointwise access to local radius and slope information, in addition to the global radius profile supplied to the branch network.

---

## 5. Experiments

### Experiment 1: Random train-validation split

The first experiment evaluates interpolation when train and validation samples are drawn from the same geometry-family distribution.

Purpose:

* measure ordinary predictive accuracy
* compare unconstrained and constrained models
* test whether geometry-aware features improve interpolation

Expected result summary:

* geometry-aware constrained DeepONet gives the best overall random-split performance
* unconstrained DeepONet may fit many samples but can produce nonphysical outputs
* constrained models improve physical reliability

### Experiment 2: Leave-one-geometry-family-out generalization

The second experiment evaluates geometry extrapolation.

For each run:

* one geometry family is removed from training
* the model is trained on the remaining geometry families
* the model is tested on the held-out geometry family

This is repeated for:

* straight
* stenosed
* expanded
* sinusoidal
* hyperbolic constriction

Purpose:

* identify which geometries are hardest to extrapolate to
* test whether local geometry features improve unseen-family generalization
* evaluate physical reliability under distribution shift

### Metrics

The main metrics are:

* mean relative (L^2) error
* median relative (L^2) error
* maximum relative (L^2) error
* number of negative velocity predictions

The negative-prediction count is included because a surrogate can have moderate average error while still violating basic physical admissibility.

---

## 6. Results

### 6.1 Random-split results

The random-split results show that geometry-aware local features improve interpolation performance across the sampled geometry distribution. The geometry-aware constrained DeepONet achieves the best mean and median validation errors among the tested architectures.

This indicates that providing local radius and slope information helps the model learn geometry-dependent variations when the test geometries are drawn from the same distribution as the training samples.

### 6.2 Leave-one-geometry-family-out results

The leave-one-geometry-family-out experiment reveals a more nuanced picture.

The unconstrained DeepONet frequently produces nonphysical negative velocity predictions under geometry extrapolation. This occurs even when its mean error is not always the worst. Therefore, physical admissibility must be evaluated separately from error magnitude.

The constrained models eliminate negative velocity predictions across all held-out geometry tests.

The geometry-aware constrained DeepONet gives the best mean error for most held-out geometry families, including stenosed, expanded, and hyperbolic-constriction cases. This suggests that local geometric quantities generally improve average extrapolation performance.

However, the held-out sinusoidal case is different. In this case, the power-law-aware constrained DeepONet gives the lowest mean, median, and maximum error. This suggests that richer local-geometry features do not automatically guarantee better extrapolation to every unseen geometry family. Simpler physics-based inductive bias can sometimes generalize more robustly.

### 6.3 Geometry-dependent extrapolation difficulty

The aggregate leave-one-geometry-out plots show that extrapolation difficulty depends strongly on the held-out geometry family.

This supports the conclusion that geometry generalization should not be evaluated using only one held-out geometry or a random train-test split. Different geometry families expose different failure modes.

### 6.4 Physical reliability

The negative-prediction comparison shows that the unconstrained model is unreliable under geometry distribution shift. The constrained models avoid this failure mode by construction.

This result supports the use of hard physical constraints in operator-learning surrogates for fluid-flow problems.

---

## 7. Discussion

### Main interpretation

The results show that adding physics-based structure changes the behavior of neural operators in important ways.

The unconstrained model has flexibility but can violate physical constraints. The constrained models reduce this failure mode. Geometry-aware features improve accuracy in most cases, but they do not fully solve geometry extrapolation.

### Why geometry-aware is not always best

The geometry-aware model receives (R(z)) and (dR/dz) locally. This helps interpolation and many extrapolation cases, but it may also make the model sensitive to local geometry patterns represented in the training families.

The sinusoidal held-out case suggests that an unseen oscillatory radius pattern can still challenge the geometry-aware model.

### Implication

For robust geometry-generalizable surrogate modeling, it may not be enough to provide geometry features. The architecture may also need stronger geometry-scaled physical constraints, conservation-aware structure, or broader random geometry sampling during training.

---

## 8. Limitations

This benchmark uses reduced forward solutions rather than full Navier-Stokes or full finite-element simulations. The geometries are axisymmetric and described by one-dimensional radius profiles (R(z)). The model predicts velocity fields on a normalized coordinate system rather than on arbitrary unstructured meshes.

The current geometry families are still limited. A stronger next benchmark would train on random smooth geometries generated from Fourier modes, Gaussian-process samples, or spline control points, then test on structured held-out geometry families.

The current constraints enforce nonnegativity and wall behavior, but do not yet explicitly enforce mass conservation or pressure-flow consistency in the neural-operator architecture.

---

## 9. Next steps

The next technical step is to introduce broader unknown-geometry sampling, such as random Fourier radius profiles:

[
R(z) = R_0 \left[1 + \sum_k a_k \sin(2\pi k z + \phi_k)\right],
]

with constraints on minimum radius, smoothness, and slope.

This would allow training on a broader function space of smooth geometries and testing whether the learned operator generalizes to named geometry families such as stenosed, sinusoidal, and hyperbolic-constriction cases.

A second next step is to develop a geometry-scaled constrained DeepONet that incorporates expected pressure-flow and radius-scaling behavior more directly into the output structure.

---

## 10. Possible paper figures

Figure 1. Overview of geometry families and normalized coordinate system.

Figure 2. Example velocity fields for Newtonian and power-law fluids across variable-radius geometries.

Figure 3. Random-split model comparison: unconstrained vs positive/no-slip vs power-law-aware vs geometry-aware DeepONet.

Figure 4. Leave-one-geometry-family-out mean error by held-out geometry and model.

Figure 5. Leave-one-geometry-family-out median and maximum error comparison.

Figure 6. Negative velocity predictions under geometry extrapolation.

Figure 7. Best/median/worst prediction examples for the final constrained models.

---

## 11. Possible conclusion

This study shows that physics-constrained neural operators provide more reliable surrogates than unconstrained DeepONets for pressure-driven Newtonian and power-law pipe flows in variable-radius geometries. Hard constraints eliminate nonphysical negative velocity predictions, while geometry-aware features improve average accuracy in most interpolation and extrapolation settings. However, leave-one-geometry-family-out tests reveal that geometry extrapolation remains challenging and depends strongly on the held-out geometry family. These results highlight the importance of evaluating neural operators under geometry distribution shifts and motivate future work on broader random geometry sampling and stronger conservation-aware architectures.
