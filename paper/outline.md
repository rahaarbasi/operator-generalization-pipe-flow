# Sampling and Inductive Bias Effects in Neural Operators for Geometry-Varying Pipe Flow

## Abstract

* Problem
* Gap in current neural-operator literature
* Investigated factors:

  * sampling strategy
  * inductive bias
* Geometry-varying pipe-flow benchmark
* Leave-one-geometry-out evaluation
* Main findings
* Contributions

---

## 1. Introduction

### 1.1 Neural operators for PDE surrogates

* DeepONet
* FNO
* operator learning

### 1.2 Geometry generalization challenge

Most papers evaluate interpolation.

Few evaluate extrapolation to unseen geometries.

### 1.3 Sampling vs inductive bias

Two common approaches:

* better sampling
* better architecture

Open question:

Which contributes more to geometry generalization?

### 1.4 Contributions

* geometry-varying benchmark
* leave-one-geometry-out protocol
* sampling comparison
* inductive-bias comparison
* quantitative analysis

---

## 2. Related Work

### 2.1 Neural operators

DeepONet
FNO
PINO

### 2.2 Physics-informed inductive biases

hard constraints
boundary-aware architectures

### 2.3 Adaptive and non-uniform sampling

residual-based
importance sampling
boundary-focused sampling

### 2.4 Geometry generalization

existing limitations

---

## 3. Problem Definition

### 3.1 Geometry-varying pipe flow

Forward operator:

R(z), fluid parameters, ΔP, (z,η) → u(z,η)

### 3.2 Fluid models

* Newtonian
* power-law

### 3.3 Geometry families

* straight
* stenosed
* expanded
* sinusoidal
* hyperbolic constriction

### 3.4 Leave-one-geometry-out protocol

Motivation and setup

---

## 4. Dataset Construction

### 4.1 Parametric geometry generation

(Figure: pipe_geometry_variations)

### 4.2 Forward solver

Dataset generation procedure

### 4.3 Training and test splits

LOGO protocol

---

## 5. Sampling Strategies

### 5.1 Uniform sampling

### 5.2 Geometry-aware sampling

(Figure: sampling_diagnostic_sinusoidal_histograms)

### 5.3 Wall-stretched eta sampling

(Figure: wall_stretched_eta_sampling_continuous_k2)

### 5.4 Expected benefits

Near-wall resolution
Geometry emphasis

---

## 6. Neural Operator Models

### 6.1 Unconstrained DeepONet

### 6.2 Power-law-aware DeepONet

Physics-informed radial factor

### 6.3 Geometry-channel-aware DeepONet

Local geometry channels

---

## 7. Experimental Setup

### 7.1 Training settings

### 7.2 Evaluation metrics

* Mean L2
* Median L2
* Max L2
* Negative predictions

### 7.3 Held-out sinusoidal benchmark

Reason for selecting sinusoidal case

---

## 8. Results

### 8.1 Main comparison

(Table: master_results_table.csv)

### 8.2 Effect of sampling

(Figure: powerlaw_sampling_ablation_sinusoidal_mean_l2)

### 8.3 Effect of inductive bias

Comparison across architectures

### 8.4 Negative velocity analysis

Physical consistency

### 8.5 Error distribution

Worst-case behavior

---

## 9. Discussion

### 9.1 Why inductive bias dominates

Interpretation

### 9.2 Why sampling gives limited gains

Interpretation

### 9.3 Failure modes

Sinusoidal geometry

### 9.4 Implications for neural operators

Lessons for operator-learning workflows

---

## 10. Limitations

* synthetic benchmark
* laminar flow only
* DeepONet only
* no experimental validation
* no FNO/PINO comparison

---

## 11. Conclusion

Main findings

1. Geometry extrapolation is difficult.
2. Inductive bias improves generalization more than sampling.
3. Sampling helps but does not replace physics-aware structure.
4. Negative-prediction analysis reveals failures hidden by average error metrics.

Future work:
geometry-scaled architectures, FNO/PINO, experimental datasets.
