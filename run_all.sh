#!/usr/bin/env bash
set -euo pipefail

DATASET="data/forward_operator_dataset.npz"

echo "=== 1. Generate forward-operator dataset ==="
python src/generate_forward_operator_dataset.py \
  --output "$DATASET"

echo "=== 2. Train baseline DeepONet ==="
python src/train_deeponet_forward.py \
  --dataset "$DATASET"

echo "=== 3. Leave-one-geometry-family-out experiments ==="

for GEOMETRY in straight stenosed expanded hyperbolic_constriction sinusoidal
do
  echo "--- Held out: $GEOMETRY ---"

  python src/train_deeponet_leave_one_geometry_out.py \
    --dataset "$DATASET" \
    --held-out-geometry "$GEOMETRY" \
    --models unconstrained,power_law_aware,geometry_aware
done

echo "=== 4. Sampling ablation on sinusoidal geometry ==="

python src/train_deeponet_leave_one_geometry_out_sampling_ablation.py \
  --dataset "$DATASET" \
  --held-out-geometry sinusoidal \
  --models power_law_aware \
  --sampling uniform

python src/train_deeponet_leave_one_geometry_out_sampling_ablation.py \
  --dataset "$DATASET" \
  --held-out-geometry sinusoidal \
  --models power_law_aware \
  --sampling geometry_aware

echo "=== Done ==="
