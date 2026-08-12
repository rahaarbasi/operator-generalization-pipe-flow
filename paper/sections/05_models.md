# 5. Neural Operator Models

## 5.1 DeepONet Formulation

All models are based on the DeepONet framework.

The operator is represented as:

u(z,\eta) = \mathcal{G}(R(z), \text{fluid parameters}, \Delta P, z, \eta)

where:

* the branch network encodes geometry and fluid information,
* the trunk network encodes the spatial location (z,\eta),
* the final prediction is obtained from the interaction between branch and trunk representations.

All models share the same training procedure and differ only in their inductive biases.

---

## 5.2 Unconstrained DeepONet

The baseline model directly predicts the velocity field:

u(z,\eta)

without any physical constraints.

The model is free to learn arbitrary velocity profiles from the training data.

Advantages:

* maximum flexibility,
* simplest architecture.

Limitations:

* may produce nonphysical velocity fields,
* may generate negative velocities,
* no explicit enforcement of wall behavior.

---

## 5.3 Power-Law-Aware DeepONet

The first constrained model incorporates prior knowledge about pipe-flow velocity profiles.

The predicted velocity is written as:

u(z,\eta) = (1-\eta)^p N(z,\eta)

where:

* N(z,\eta) is the neural-network output,
* p is determined from the power-law flow structure.

This formulation guarantees:

* non-negative velocity predictions,
* zero velocity at the wall.

The constraint introduces a rheology-inspired radial structure into the model.

The goal is to improve physical consistency while preserving flexibility.

---

## 5.4 Geometry-Channel-Aware DeepONet

The second constrained model augments the operator input using local geometric information.

Additional channels include:

* local radius R(z),
* local radius gradient dR/dz,
* local radius curvature d²R/dz².

These quantities are provided to the branch network together with the original geometry description.

The objective is to expose local geometric features that may influence the velocity field.

Unlike the power-law-aware model, the geometry-channel-aware model does not directly impose a velocity-shape constraint. Instead, it introduces additional geometric information that may improve geometry extrapolation.

---

## 5.5 Inductive-Bias Hierarchy

The three models represent increasing levels of physical structure.

| Model                           | Inductive bias                  |
| ------------------------------- | ------------------------------- |
| Unconstrained DeepONet          | None                            |
| Power-law-aware DeepONet        | Rheology-aware radial structure |
| Geometry-channel-aware DeepONet | Local geometric information     |

This hierarchy allows us to investigate a central question:

Does performance improve more from better sampling or from stronger inductive bias?

The remainder of the paper evaluates this question under leave-one-geometry-family-out generalization.
