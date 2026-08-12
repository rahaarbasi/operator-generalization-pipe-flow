# 3. Problem Definition

We study neural-operator learning for geometry-varying laminar pipe flows.

The objective is to learn the forward operator

R(z), fluid parameters, ΔP, (z,η) → u(z,η)

where:

* R(z) is the radius profile of the pipe,
* ΔP is the imposed pressure drop,
* η = r / R(z) is the normalized radial coordinate,
* u(z,η) is the axial velocity field.

The dataset contains multiple geometry families including straight, stenosed, expanded, sinusoidal, and hyperbolic-constriction pipes.

Both Newtonian and power-law fluids are considered.

The main scientific question is:

How important are sampling strategies compared with architectural inductive biases when learning operators across varying geometries?

To answer this question, we compare:

1. Different DeepONet architectures with different levels of physical inductive bias.
2. Different sampling strategies in the computational domain.
3. Generalization to previously unseen geometry families through leave-one-geometry-family-out evaluation.

The held-out sinusoidal geometry is used as the primary benchmark because it represents the most difficult extrapolation case in our dataset.
