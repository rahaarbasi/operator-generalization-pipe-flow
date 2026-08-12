# 8. Results

## 8.1 Main Comparison

Table 1 summarizes the held-out sinusoidal results.

The unconstrained DeepONet produces the largest errors and frequently generates nonphysical negative velocity predictions.

Adding physics-based inductive biases substantially improves both accuracy and physical consistency.

Among the tested models, the power-law-aware DeepONet achieves the lowest overall extrapolation error.

| Model                           | Sampling                |   Mean L2 | Median L2 |    Max L2 | Negative Predictions |
| ------------------------------- | ----------------------- | --------: | --------: | --------: | -------------------: |
| Unconstrained DeepONet          | Uniform                 |     0.573 |     0.360 |     5.279 |                   50 |
| Power-law-aware DeepONet        | Uniform                 |     0.319 |     0.238 |     2.122 |                    0 |
| Power-law-aware DeepONet        | Geometry-aware sampling | **0.298** | **0.224** | **1.889** |                    0 |
| Geometry-channel-aware DeepONet | Uniform                 |     0.356 |     0.233 |     3.502 |                    0 |
| Geometry-channel-aware DeepONet | Geometry-aware sampling |     0.362 |     0.211 |     3.419 |                    0 |

---

## 8.2 Effect of Inductive Bias

The largest improvement is obtained by introducing physics-based inductive bias.

Moving from the unconstrained DeepONet to the power-law-aware DeepONet reduces:

* mean error by approximately 44%,
* maximum error by approximately 60%,
* negative predictions from 50 cases to zero.

These improvements occur even before modifying the sampling strategy.

The results therefore indicate that architectural structure has a strong influence on geometry extrapolation performance.

---

## 8.3 Effect of Geometry-Aware Sampling

Geometry-aware sampling provides a consistent but modest improvement for the power-law-aware DeepONet.

The mean relative L2 error decreases from:

0.319 → 0.298

while the maximum error decreases from:

2.122 → 1.889

The improvement is measurable but significantly smaller than the gains obtained through inductive bias.

---

## 8.4 Geometry-Channel-Aware Model

The geometry-channel-aware DeepONet eliminates negative velocity predictions and achieves competitive median errors.

However, for the held-out sinusoidal geometry family it does not outperform the simpler power-law-aware model.

The geometry-channel-aware architecture produces:

* a lower median error,
* but a larger mean error,
* and a substantially larger worst-case error.

This suggests that additional geometric information alone is not sufficient to guarantee better extrapolation.

---

## 8.5 Negative Velocity Analysis

The unconstrained DeepONet generates nonphysical negative velocity predictions in 50 of the 100 held-out sinusoidal test cases.

All constrained models eliminate negative velocity predictions entirely.

This result demonstrates that physical constraints improve reliability even when prediction error is used as the primary optimization objective.

---

## 8.6 Summary

Two observations emerge consistently across all experiments.

First, inductive bias produces substantially larger improvements than sampling modifications.

Second, physically constrained architectures improve both numerical accuracy and physical plausibility.

The strongest overall performance is obtained by the power-law-aware DeepONet combined with geometry-aware sampling.
