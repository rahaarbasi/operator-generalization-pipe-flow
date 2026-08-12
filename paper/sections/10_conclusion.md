# 10. Conclusion

This work investigated the relative importance of sampling strategy and architectural inductive bias in neural operators for geometry-varying laminar pipe-flow problems.

A controlled benchmark was developed using five geometry families and both Newtonian and power-law fluids. Geometry extrapolation was evaluated using a leave-one-geometry-family-out protocol, with particular focus on the challenging sinusoidal geometry family.

Three DeepONet variants were compared:

* an unconstrained baseline,
* a power-law-aware constrained model,
* a geometry-channel-aware model.

In addition, multiple sampling strategies were investigated, including geometry-aware axial sampling and wall-focused radial sampling.

The experiments show that architectural inductive bias has a substantially larger impact on geometry extrapolation than sampling strategy.

Geometry-aware sampling produced measurable improvements, but the gains remained modest compared with those obtained from physically informed architectural constraints.

The power-law-aware DeepONet reduced both prediction error and nonphysical behavior relative to the unconstrained baseline, while geometry-aware features improved performance for several geometry families but did not consistently outperform simpler physically motivated constraints.

The results suggest that, for geometry-varying flow problems, incorporating physical structure into the neural operator is more effective than modifying the sampling distribution alone.

Future work will investigate:

* geometry-scaled architectures,
* Fourier Neural Operators,
* Physics-Informed Neural Operators,
* more complex geometries,
* experimental validation using physical flow measurements.

Overall, the study demonstrates that physically informed inductive biases are a key ingredient for reliable geometry extrapolation in neural-operator learning.
