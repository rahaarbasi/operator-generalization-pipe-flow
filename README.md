# Operator Generalization in Geometry-Varying Pipe Flow

Research code and manuscript materials for studying **generalization of neural operators to unseen pipe geometries**, with particular emphasis on the relative roles of:

- architectural / physical inductive bias,
- geometry representation,
- spatial sampling strategy,
- extrapolation across geometry families.

The main model family considered here is **DeepONet**, evaluated on Newtonian and power-law internal-flow problems over varying pipe geometries.

## Research Question

When a neural operator is required to predict flow in a geometry family that was not observed during training, what matters more for generalization:

1. the structure imposed on the model, or
2. the way training points are sampled from the physical domain?

The project uses leave-one-geometry-family-out experiments to investigate this question.

## Repository Structure

```text
.
├── data/        # Forward-operator dataset
├── docs/        # Supporting research notes
├── figures/     # Generated figures
├── models/      # Trained DeepONet checkpoints
├── paper/       # Manuscript, figures, and master result table
├── results/     # Raw and summarized experiment results
├── src/         # Dataset, training, evaluation, and plotting code
└── requirements.txt
```

## Main Experimental Components

### Neural-operator baselines

The repository includes several DeepONet variants, including:

- unconstrained baseline,
- positivity-constrained model,
- power-law-informed constraint,
- geometry-aware model.

### Geometry extrapolation

Generalization is evaluated using **leave-one-geometry-family-out** experiments.

The available geometry families include straight and non-uniform pipe configurations such as constricted, expanded, sinusoidal, and related varying-radius geometries.

### Sampling ablations

The repository also contains experiments comparing different spatial sampling strategies, including uniform and geometry-aware sampling schemes.

## Key Code

Dataset generation:

```text
src/generate_forward_operator_dataset.py
```

Baseline and constrained DeepONet training:

```text
src/train_deeponet_forward.py
src/train_deeponet_forward_positive.py
src/train_deeponet_forward_powerlaw_constraint.py
src/train_deeponet_forward_geometry_aware.py
```

Geometry-extrapolation experiments:

```text
src/train_deeponet_leave_one_geometry_out.py
src/train_deeponet_leave_one_geometry_out_sampling_ablation.py
src/train_deeponet_leave_one_geometry_out_domain_channels.py
```

Evaluation and analysis:

```text
src/evaluate_deeponet_forward.py
src/evaluate_deeponet_forward_fullfield.py
src/analyze_deeponet_errors_by_case.py
src/aggregate_leave_one_geometry_out.py
```

## Manuscript

The developing manuscript is located in:

```text
paper/
```

Individual manuscript sections are stored under:

```text
paper/sections/
```

and the consolidated experimental results used for the manuscript are available under:

```text
paper/results/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Current Status

This repository is an active research project.

The present manuscript focuses specifically on **geometry extrapolation, sampling strategy, and inductive bias**. Inverse rheology and the earlier MLP/PINN pipe-flow benchmarks are maintained as separate research tracks and are not part of this repository.
