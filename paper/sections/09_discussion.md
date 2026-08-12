# 9. Discussion

## 9.1 Inductive Bias Dominates Sampling

The central result of this study is that inductive bias has a substantially larger impact on geometry extrapolation than sampling strategy.

Geometry-aware sampling consistently improved performance, but the magnitude of the improvement remained modest.

For example, the power-law-aware DeepONet improved from:

Mean L2 = 0.319

to

Mean L2 = 0.298

after introducing geometry-aware and wall-focused sampling.

In contrast, introducing a physically motivated architectural constraint reduced the mean error from:

0.573

for the unconstrained model to:

0.319

for the power-law-aware model.

The gain from inductive bias was therefore much larger than the gain from sampling.

---

## 9.2 Why Sampling Helps Only Modestly

Sampling changes where information is observed during training.

However, sampling does not change how the model represents the underlying physical process.

A better sampling strategy may expose the model to more informative regions of the domain, such as:

* near-wall regions,
* regions with strong geometric variation.

Nevertheless, the model must still learn the correct physical structure from data.

If the architecture itself does not encode useful physical assumptions, improved sampling alone cannot fully compensate.

This explains why geometry-aware sampling improves performance but does not fundamentally change extrapolation behavior.

---

## 9.3 Why Inductive Bias Helps More

Inductive bias directly modifies the hypothesis space available to the model.

The power-law-aware architecture constrains the predicted velocity field to follow physically meaningful radial behavior.

As a result, the model does not need to learn this structure entirely from data.

Instead, the optimization process focuses on learning the remaining variability associated with geometry and fluid parameters.

The elimination of negative velocity predictions provides further evidence that physically motivated constraints improve model reliability.

These observations are consistent with the broader literature on physics-informed machine learning, where architectural constraints often provide stronger gains than purely data-centric modifications.

---

## 9.4 Geometry-Aware Features Are Not Automatically Better

An unexpected result is that the geometry-channel-aware DeepONet does not outperform the simpler power-law-aware model on the held-out sinusoidal geometry family.

The geometry-channel model receives additional information:

* local radius,
* local radius gradient,
* local curvature.

Despite this richer representation, the held-out sinusoidal case remains difficult.

This suggests that providing more geometric information does not automatically improve extrapolation.

A more complex representation may increase flexibility but may also increase sensitivity to geometry distributions observed during training.

In contrast, the simpler power-law-aware architecture may generalize more robustly because it relies more heavily on physically motivated structure.

---

## 9.5 Implications for Neural Operator Design

The results suggest a practical hierarchy for developing neural operators for geometry-varying flows.

First, physically meaningful inductive biases should be incorporated into the architecture.

Second, geometry-aware sampling can be used as a secondary refinement.

The experiments indicate that data allocation alone is unlikely to overcome deficiencies in architectural design.

For geometry-varying flow problems, model structure appears to be more important than sampling density.

---

## 9.6 Limitations

Several limitations should be acknowledged.

First, the study is restricted to laminar pipe-flow problems.

Second, only DeepONet-based architectures are considered.

Third, geometry variation is represented through a limited set of parametric geometry families.

Finally, only one held-out geometry family is examined in detail within the sampling experiments.

Future work should investigate whether the same conclusions remain valid for:

* more complex geometries,
* higher-dimensional flows,
* Fourier Neural Operators (FNO),
* Physics-Informed Neural Operators (PINO),
* experimental datasets.

---

## 9.7 Main Takeaway

The principal conclusion of this work is that geometry extrapolation performance is driven primarily by architectural inductive bias rather than sampling strategy.

Sampling improvements are beneficial, but they provide incremental gains.

Physically informed model structure produces substantially larger improvements in both accuracy and physical consistency.

For the benchmark considered here, choosing the right inductive bias matters more than choosing the right sampling distribution.
