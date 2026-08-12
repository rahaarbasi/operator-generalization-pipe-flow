#!/usr/bin/env python3
"""
Resume-safe launcher for the full 225-run crossed grid.

Calls src/run_cross_grid_cell.py.

Grid:
    3 architectures x 3 sampling strategies x 5 held-out families x 5 seeds

Existing complete run folders are skipped.
Existing incomplete run folders cause a hard stop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ARCHES = ("unconstrained", "powerlaw", "geoaware")
SAMPLINGS = ("uniform", "geo-axial", "geo-wallstretched")
GEOMS = ("sinusoidal", "straight", "stenosed", "expanded", "hyperbolic")
SEEDS = (0, 1, 2, 3, 4)

REQUIRED_FILES = (
    "config.json",
    "per_sample.csv",
    "predictions.npz",
    "train_curve.csv",
    "model.pt",
    "timing.json",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def run_name(arch, sampling, geom, seed):
    return f"{arch}__{sampling}__holdout-{geom}__seed-{seed:02d}"


def is_complete(folder: Path) -> bool:
    return folder.is_dir() and all((folder / f).exists() for f in REQUIRED_FILES)


def validate_complete_config(folder: Path, arch, sampling, geom, seed, epochs):
    cfg = json.loads((folder / "config.json").read_text())
    expected = {
        "arch": arch,
        "sampling": sampling,
        "held_out": geom,
        "seed": seed,
        "epochs": epochs,
    }
    mismatches = []
    for key, wanted in expected.items():
        got = cfg.get(key)
        if got != wanted:
            mismatches.append(f"{key}: found {got!r}, expected {wanted!r}")
    return mismatches


def require_clean_git():
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise SystemExit(
            "Refusing to launch paper runs because git is dirty.\n"
            "Commit/stash changes first.\n\n"
            + status
        )


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        require_clean_git()

    total = len(ARCHES) * len(SAMPLINGS) * len(GEOMS) * len(SEEDS)
    skipped = 0
    launched = 0

    for geom in GEOMS:
        for arch in ARCHES:
            for sampling in SAMPLINGS:
                for seed in SEEDS:
                    name = run_name(arch, sampling, geom, seed)
                    folder = runs_dir / name

                    if folder.exists():
                        if not is_complete(folder):
                            raise SystemExit(
                                f"INCOMPLETE RUN FOLDER FOUND: {folder}\n"
                                "Stopping rather than overwriting evidence."
                            )

                        mismatches = validate_complete_config(
                            folder, arch, sampling, geom, seed, args.epochs
                        )
                        if mismatches:
                            raise SystemExit(
                                f"CONFIG MISMATCH IN {folder}:\n  "
                                + "\n  ".join(mismatches)
                            )

                        skipped += 1
                        print(f"[SKIP] {name}")
                        continue

                    if args.limit is not None and launched >= args.limit:
                        print(
                            f"\nReached --limit {args.limit}. "
                            f"Skipped {skipped}, launched {launched}."
                        )
                        return

                    cmd = [
                        sys.executable,
                        "src/run_cross_grid_cell.py",
                        "--arch", arch,
                        "--sampling", sampling,
                        "--held-out", geom,
                        "--seed", str(seed),
                        "--epochs", str(args.epochs),
                        "--print-every", str(args.print_every),
                        "--batch-size", str(args.batch_size),
                        "--learning-rate", str(args.learning_rate),
                    ]

                    print(f"\n[RUN {skipped + launched + 1}/{total}] {name}", flush=True)
                    print(" ".join(cmd), flush=True)

                    if args.dry_run:
                        launched += 1
                        continue

                    subprocess.run(cmd, check=True)
                    launched += 1

    print(
        f"\nSweep traversal complete. "
        f"Existing complete runs skipped: {skipped}; "
        f"new runs launched: {launched}."
    )

    if not args.dry_run:
        subprocess.run([sys.executable, "build_master_table.py"], check=True)


if __name__ == "__main__":
    main()