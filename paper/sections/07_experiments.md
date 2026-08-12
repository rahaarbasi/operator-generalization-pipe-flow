# 7. Experimental Setup

## 7.1 Objective

The objective of the experiments is to evaluate the relative impact of:

* sampling strategy,
* architectural inductive bias,

on geometry extrapolation performance in neural operators.

All experiments use the same forward-operator learning task:

R(z), fluid parameters, ΔP, (z,η) → u(z,η)

The target quantity is the axial velocity field.

---

## 7.2 Geometry Families

The dataset contains five geometry families:

* straight,
* stenosed,
* expanded,
* sinusoidal,
* hyperbolic constriction.

Each sample contains:

* a radius profile R(z),
* fluid parameters,
* pressure-drop information,
* velocity-field targets.

Both Newtonian and power-law fluids are included.

---

## 7.3 Leave-One-Geometry-Out Protocol

To evaluate geometry extrapolation, a leave-one-geometry-family-out protocol is used.

One geometry family is completely removed from the training set and reserved for testing.

The model is still provided with the corresponding radius profile R(z) during inference, but it has never observed examples from that geometry family during training.

This setup evaluates the ability of the learned operator to generalize beyond geometry classes represented in the training data.

---

## 7.4 Held-Out Sinusoidal Benchmark

The primary benchmark in this study uses:

Held-out geometry = sinusoidal

This geometry family was selected because it consistently produced the largest extrapolation errors among the investigated geometries.

The sinusoidal family therefore provides a challenging test of both sampling strategies and inductive biases.

---

## 7.5 Neural Operator Models

The following models are evaluated:

1. Unconstrained DeepONet

2. Power-law-aware DeepONet

3. Geometry-channel-aware DeepONet

Architectural details are provided in Section 5.

---

## 7.6 Sampling Configurations

The following sampling configurations are compared:

1. Uniform sampling

2. Geometry-aware sampling

3. Geometry-aware sampling with wall-stretched radial sampling

Sampling definitions are provided in Section 6.

---

## 7.7 Evaluation Metrics

Prediction accuracy is measured using relative L2 error.

For each experiment, the following statistics are reported:

* Mean relative L2 error,
* Median relative L2 error,
* Maximum relative L2 error.

These metrics provide complementary information:

* Mean error measures overall performance,
* Median error measures typical performance,
* Maximum error measures worst-case behavior.

---

## 7.8 Physical Validity Metric

In addition to prediction error, physical validity is evaluated.

The number of test cases producing negative velocity predictions is recorded.

Negative velocity values are considered nonphysical for the benchmark problems considered here.

This metric is used to distinguish physically plausible operators from models that achieve low numerical error while violating basic physical constraints.

---

## 7.9 Implementation

All models are implemented in Python using PyTorch.

Training is performed using identical optimization settings wherever possible in order to isolate the effects of:

* sampling strategy,
* architectural inductive bias.

The comparison therefore focuses on differences arising from model design and data allocation rather than optimizer tuning.
