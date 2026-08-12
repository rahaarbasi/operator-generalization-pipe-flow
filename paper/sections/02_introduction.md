# 2. Introduction

Machine-learning surrogates for partial differential equations have received increasing attention as a means of accelerating scientific computing and engineering simulation. In particular, neural operators have emerged as a promising class of models because they aim to learn mappings between function spaces rather than approximating solutions on a single fixed discretization. This capability makes neural operators attractive for applications involving repeated evaluations of computationally expensive forward models.

Among neural-operator architectures, Fourier Neural Operators (FNOs), DeepONets, and related operator-learning methods have demonstrated strong performance on a wide range of problems in fluid mechanics, heat transfer, porous-media flow, and computational physics. Despite this progress, generalization beyond the training distribution remains a major challenge. In many practical applications, the geometry encountered during deployment differs from the geometries observed during training, creating a geometry-extrapolation problem rather than a simple interpolation task.

Geometry variation is particularly important in flow systems. Examples include biomedical flows in vessels with varying diameters, capillary and microfluidic devices, industrial piping systems, and engineering components with geometric imperfections or design modifications. A neural operator intended for such applications must not only reproduce solutions on familiar geometries but also generalize reliably to previously unseen geometric configurations.

Two broad strategies are commonly used to improve generalization. The first is to modify the training data distribution through improved sampling. More informative sampling can allocate computational resources toward physically important regions such as boundary layers, geometric constrictions, or regions with strong gradients. The second is to modify the model architecture through inductive bias. Physically motivated constraints, symmetry assumptions, and geometry-aware representations can restrict the hypothesis space and encourage physically meaningful predictions.

While both strategies are widely used, their relative importance is not well understood. In particular, it remains unclear whether improvements in geometry extrapolation are driven primarily by better sampling or by better architectural design. This question is difficult to isolate in realistic computational-fluid-dynamics benchmarks because many sources of complexity are present simultaneously.

To address this issue, we construct a controlled benchmark based on geometry-varying laminar pipe-flow problems. The benchmark includes five geometry families and both Newtonian and power-law fluid models. Because the governing physics is relatively simple and high-quality reference solutions are available, the framework provides a suitable environment for studying the effect of sampling and inductive bias in isolation.

Within this benchmark, we compare three DeepONet architectures with increasing levels of physical structure:

* an unconstrained DeepONet baseline,
* a power-law-aware constrained DeepONet,
* a geometry-channel-aware constrained DeepONet.

We additionally compare multiple sampling strategies, including uniform sampling, geometry-aware axial sampling, and wall-focused radial sampling.

A leave-one-geometry-family-out protocol is used to evaluate geometry extrapolation. In this setting, an entire geometry family is excluded from training and used only during testing, providing a more demanding evaluation than standard random train-test splits.

The main objective of this work is to determine whether sampling strategy or architectural inductive bias plays the dominant role in geometry extrapolation. The results show that physically motivated inductive bias produces substantially larger improvements than sampling modifications, while geometry-aware sampling provides smaller but consistent gains. These findings suggest that, for geometry-varying neural-operator learning, architectural design should be prioritized before investing effort in increasingly sophisticated sampling schemes.

The contributions of this work are:

1. A controlled geometry-varying pipe-flow benchmark for studying neural-operator generalization.
2. A systematic comparison between sampling-based and inductive-bias-based approaches for improving geometry extrapolation.
3. A leave-one-geometry-family-out evaluation framework for assessing robustness to unseen geometries.
4. Evidence that physically motivated inductive bias has a larger impact on extrapolation performance than sampling strategy within the benchmark considered here.
