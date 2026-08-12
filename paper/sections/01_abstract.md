# Abstract

Neural operators have emerged as a promising framework for surrogate modeling of partial differential equations, but their ability to generalize across previously unseen geometries remains an open challenge. In geometry-varying flow problems, both data sampling strategy and architectural inductive bias may influence extrapolation performance, yet their relative importance is not well understood.

In this work, we investigate this question using a controlled benchmark based on laminar pipe-flow problems with variable geometry. Analytical and semi-analytical forward solutions are generated for five geometry families, including straight, stenosed, expanded, sinusoidal, and hyperbolic-constriction pipes, under both Newtonian and power-law fluid assumptions. A leave-one-geometry-family-out evaluation protocol is used to assess geometry extrapolation.

Three DeepONet architectures are considered: an unconstrained baseline, a power-law-aware constrained model, and a geometry-channel-aware model incorporating local geometric features. These architectures are combined with multiple sampling strategies, including uniform sampling, geometry-aware axial sampling, and wall-focused radial sampling.

The results show that physically motivated inductive bias produces substantially larger improvements than sampling modifications. Relative to the unconstrained baseline, constrained architectures reduce prediction error and eliminate nonphysical negative velocity predictions. Geometry-aware sampling provides consistent but modest gains, whereas inductive bias yields the dominant improvement in extrapolation performance. For the most challenging held-out sinusoidal geometry family, the power-law-aware DeepONet combined with geometry-aware sampling achieves the best overall accuracy.

These findings suggest that, for geometry-varying neural-operator learning, architectural inductive bias is more influential than sampling strategy, and should therefore be prioritized when designing surrogate models for unseen geometries.
