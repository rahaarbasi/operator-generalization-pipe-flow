# Cross-grid operator-generalization experiment

Final experiment snapshot generated on 2026-08-12.

## Purpose

Test whether architectural inductive bias or sampling strategy has the larger
effect on out-of-family geometry generalization in the pipe-flow operator model.

## Experimental grid

Architectures:
- unconstrained
- power-law-aware
- geometry-aware

Sampling strategies:
- uniform
- geometry-aware axial
- geometry-aware axial + wall-stretched radial

Held-out geometry families:
- straight
- stenosed
- expanded
- sinusoidal
- hyperbolic constriction

Seeds:
- 0, 1, 2, 3, 4

Total:
3 architectures × 3 sampling strategies × 5 held-out families × 5 seeds
= 225 training runs.

Training configuration:
- epochs: 1000
- batch size: 8192
- learning rate: 5e-4
- device: Apple MPS

See manifest.csv and individual config.json files for exact dataset hashes,
Git commits, hardware, and run-level settings.

## Main result

Mean relative L2 error over the 15 matched sampling × held-out cells:

Unconstrained:   0.409074
Power-law-aware: 0.215228
Geometry-aware:  0.173079

Geometry-aware < Power-law-aware < Unconstrained in all 15/15 matched cells.

Mean relative L2 error by sampling strategy:

Uniform:              0.266033
Geometry-aware axial: 0.264053
Wall-stretched:       0.267297

Sampling changes were small compared with architecture changes.

## Statistics

Architecture:
Friedman p = 3.059e-07

Holm-corrected paired Wilcoxon comparisons:
- unconstrained vs power-law-aware: p = 1.831e-04
- unconstrained vs geometry-aware: p = 1.831e-04
- power-law-aware vs geometry-aware: p = 1.831e-04

Matched rank-biserial effect size = 1.0 for all three architecture comparisons.

Sampling:
Friedman p = 0.07427

No pairwise sampling comparison was significant after Holm correction.

## Physical admissibility

Power-law-aware and geometry-aware architectures produced zero negative-velocity
predictions across the reported held-out tests.

The unconstrained architecture frequently produced negative predictions,
particularly for stenosed and hyperbolic-constriction cases.

## Interpretation

The experiment supports the conclusion that architectural inductive bias has a
large and consistent effect on held-out-geometry generalization, whereas changing
the sampling strategy alone produces comparatively small changes.

This experiment evaluates generalization relative to the reduced forward solver
used to generate the dataset. It does not constitute validation against full CFD.
A separate CFD spot-check is planned.

## Repository contents

aggregate/
    Derived tables used for figures and statistical analysis.

per_run/
    Lightweight raw evidence for all 225 runs:
    config.json
    per_sample.csv
    train_curve.csv
    timing.json

manifest.csv
    Run-level dataset hashes, Git commits, training settings, hardware and timing.

Large model checkpoints and predictions.npz files remain in the local ignored
runs/ directory and are intentionally not committed to Git because of their size.
