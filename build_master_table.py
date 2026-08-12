#!/usr/bin/env python3
"""
Aggregate the 3 x 3 x 5 x 5 operator-generalization experiment grid.

Expected run folder:
    runs/{arch}__{sampling}__holdout-{geom}__seed-{nn}/

Expected per-run file:
    per_sample.csv

Outputs:
    results/master_table.csv
    results/effect_inductive_bias.csv
    results/effect_sampling.csv
    results/fluid_breakdown.csv
    results/completeness_report.csv

The script is safe to re-run while the sweep is in progress. It reports missing
runs explicitly and never fabricates or hand-enters summary values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from itertools import product
from pathlib import Path

ARCHES = ("unconstrained", "powerlaw", "geoaware")
SAMPLINGS = ("uniform", "geo-axial", "geo-wallstretched")
GEOMS = ("straight", "stenosed", "expanded", "sinusoidal", "hyperbolic")
SEEDS = tuple(range(5))

TRUE_VALUES = {"true", "1", "yes"}
EXPECTED_PER_SAMPLE_COLUMNS = {
    "relative_l2",
    "fluid_model",
    "has_negative_prediction",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="results")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error unless all 225 expected runs are complete.",
    )
    return p.parse_args()


def folder_name(arch, sampling, geom, seed):
    return f"{arch}__{sampling}__holdout-{geom}__seed-{seed:02d}"


def parse_folder(name):
    parts = name.split("__")
    if len(parts) != 4:
        raise ValueError(f"Invalid run folder name: {name}")
    arch, sampling, holdout, seed = parts
    if not holdout.startswith("holdout-") or not seed.startswith("seed-"):
        raise ValueError(f"Invalid run folder name: {name}")
    return (
        arch,
        sampling,
        holdout.removeprefix("holdout-"),
        int(seed.removeprefix("seed-")),
    )


def mean_std(values):
    values = list(values)
    if not values:
        return math.nan, math.nan
    return (
        sum(values) / len(values),
        st.stdev(values) if len(values) > 1 else 0.0,
    )


def median(values):
    return st.median(values)


def validate_config(folder, expected):
    path = folder / "config.json"
    if not path.exists():
        return ["missing config.json"]

    try:
        cfg = json.loads(path.read_text())
    except Exception as e:
        return [f"invalid config.json: {e}"]

    arch, sampling, geom, seed = expected
    problems = []
    checks = {
        "arch": arch,
        "sampling": sampling,
        "held_out": geom,
        "seed": seed,
    }
    for key, wanted in checks.items():
        got = cfg.get(key)
        if got != wanted:
            problems.append(f"config {key}={got!r}, expected {wanted!r}")
    return problems


def read_run(folder):
    """Return a run-level summary derived only from raw per_sample.csv."""
    path = folder / "per_sample.csv"
    if not path.exists():
        raise FileNotFoundError("missing per_sample.csv")

    l2 = []
    neg = 0
    by_fluid = defaultdict(lambda: {"l2": [], "neg": 0})

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = EXPECTED_PER_SAMPLE_COLUMNS - cols
        if missing:
            raise ValueError(
                f"per_sample.csv missing columns: {sorted(missing)}"
            )

        for row in reader:
            err = float(row["relative_l2"])
            is_neg = str(row["has_negative_prediction"]).strip().lower() in TRUE_VALUES
            fluid = str(row["fluid_model"]).strip()

            l2.append(err)
            neg += int(is_neg)
            by_fluid[fluid]["l2"].append(err)
            by_fluid[fluid]["neg"] += int(is_neg)

    if not l2:
        raise ValueError("per_sample.csv contains no rows")

    summary = {
        "n_test": len(l2),
        "mean_l2": sum(l2) / len(l2),
        "median_l2": median(l2),
        "max_l2": max(l2),
        "neg_count": neg,
        "neg_rate": neg / len(l2),
        "fluids": {},
    }

    for fluid, d in by_fluid.items():
        vals = d["l2"]
        summary["fluids"][fluid] = {
            "n_test": len(vals),
            "mean_l2": sum(vals) / len(vals),
            "median_l2": median(vals),
            "max_l2": max(vals),
            "neg_count": d["neg"],
            "neg_rate": d["neg"] / len(vals),
        }

    return summary


def write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = list(product(ARCHES, SAMPLINGS, GEOMS, SEEDS))
    cells = defaultdict(list)
    run_records = {}
    completeness_rows = []
    bad_runs = []

    for key in expected:
        arch, sampling, geom, seed = key
        name = folder_name(*key)
        folder = runs_dir / name

        if not folder.exists():
            completeness_rows.append([arch, sampling, geom, seed, "missing", ""])
            continue

        problems = validate_config(folder, key)
        try:
            metrics = read_run(folder)
        except Exception as e:
            problems.append(str(e))
            metrics = None

        if problems:
            detail = " | ".join(problems)
            completeness_rows.append([arch, sampling, geom, seed, "invalid", detail])
            bad_runs.append((name, detail))
            continue

        completeness_rows.append([arch, sampling, geom, seed, "complete", ""])
        run_records[key] = metrics
        cells[(arch, sampling, geom)].append((seed, metrics))

    write_csv(
        out_dir / "completeness_report.csv",
        ["arch", "sampling", "holdout", "seed", "status", "detail"],
        completeness_rows,
    )

    # ------------------------------------------------------------------
    # Master table: one row per (architecture, sampling, held-out geometry).
    # Each statistic is first computed within each seed run from raw samples,
    # then mean +/- sample std is taken across available seeds.
    # ------------------------------------------------------------------
    master_rows = []
    cell_mean_l2 = {}

    for arch, sampling, geom in product(ARCHES, SAMPLINGS, GEOMS):
        runs = sorted(cells.get((arch, sampling, geom), []))
        if not runs:
            continue

        metrics = [m for _, m in runs]
        seed_ids = [s for s, _ in runs]

        mm, ms = mean_std(m["mean_l2"] for m in metrics)
        dm, ds = mean_std(m["median_l2"] for m in metrics)
        xm, xs = mean_std(m["max_l2"] for m in metrics)
        nm, ns = mean_std(m["neg_count"] for m in metrics)
        rm, rs = mean_std(m["neg_rate"] for m in metrics)
        ntm, nts = mean_std(m["n_test"] for m in metrics)

        cell_mean_l2[(arch, sampling, geom)] = mm

        master_rows.append([
            arch,
            sampling,
            geom,
            len(runs),
            ",".join(f"{s:02d}" for s in seed_ids),
            f"{ntm:.2f}",
            f"{nts:.2f}",
            f"{mm:.8f}",
            f"{ms:.8f}",
            f"{dm:.8f}",
            f"{ds:.8f}",
            f"{xm:.8f}",
            f"{xs:.8f}",
            f"{nm:.4f}",
            f"{ns:.4f}",
            f"{rm:.8f}",
            f"{rs:.8f}",
        ])

    write_csv(
        out_dir / "master_table.csv",
        [
            "arch",
            "sampling",
            "holdout",
            "n_seeds",
            "seed_ids",
            "n_test_mean",
            "n_test_std",
            "mean_l2",
            "mean_l2_seed_std",
            "median_l2",
            "median_l2_seed_std",
            "max_l2",
            "max_l2_seed_std",
            "negative_count_mean",
            "negative_count_seed_std",
            "negative_rate_mean",
            "negative_rate_seed_std",
        ],
        master_rows,
    )

    # ------------------------------------------------------------------
    # Symmetric main effects.
    #
    # Architecture effect:
    #   for each arch, average the 15 seed-averaged cell means
    #   (3 samplings x 5 held-out geometries).
    #
    # Sampling effect:
    #   for each sampling, average the 15 seed-averaged cell means
    #   (3 architectures x 5 held-out geometries).
    #
    # This deliberately uses cell means rather than pooling all raw samples,
    # so large test families do not silently receive extra weight.
    # ------------------------------------------------------------------
    arch_effect_rows = []
    for arch in ARCHES:
        vals = [
            v for (a, s, g), v in cell_mean_l2.items()
            if a == arch
        ]
        if vals:
            m, sd = mean_std(vals)
            arch_effect_rows.append([
                arch,
                len(vals),
                f"{m:.8f}",
                f"{sd:.8f}",
            ])

    write_csv(
        out_dir / "effect_inductive_bias.csv",
        ["arch", "n_cells", "mean_l2_over_cells", "std_across_cells"],
        arch_effect_rows,
    )

    sampling_effect_rows = []
    for sampling in SAMPLINGS:
        vals = [
            v for (a, s, g), v in cell_mean_l2.items()
            if s == sampling
        ]
        if vals:
            m, sd = mean_std(vals)
            sampling_effect_rows.append([
                sampling,
                len(vals),
                f"{m:.8f}",
                f"{sd:.8f}",
            ])

    write_csv(
        out_dir / "effect_sampling.csv",
        ["sampling", "n_cells", "mean_l2_over_cells", "std_across_cells"],
        sampling_effect_rows,
    )

    # ------------------------------------------------------------------
    # Fluid-model breakdown:
    # one row per cell x fluid model, mean +/- std across seeds.
    # ------------------------------------------------------------------
    fluid_rows = []
    for (arch, sampling, geom), runs in sorted(cells.items()):
        fluids = sorted({
            fluid
            for _, m in runs
            for fluid in m["fluids"]
        })

        for fluid in fluids:
            vals = [
                m["fluids"][fluid]
                for _, m in runs
                if fluid in m["fluids"]
            ]
            if not vals:
                continue

            mm, ms = mean_std(v["mean_l2"] for v in vals)
            dm, ds = mean_std(v["median_l2"] for v in vals)
            xm, xs = mean_std(v["max_l2"] for v in vals)
            rm, rs = mean_std(v["neg_rate"] for v in vals)

            fluid_rows.append([
                arch,
                sampling,
                geom,
                fluid,
                len(vals),
                f"{mm:.8f}",
                f"{ms:.8f}",
                f"{dm:.8f}",
                f"{ds:.8f}",
                f"{xm:.8f}",
                f"{xs:.8f}",
                f"{rm:.8f}",
                f"{rs:.8f}",
            ])

    write_csv(
        out_dir / "fluid_breakdown.csv",
        [
            "arch",
            "sampling",
            "holdout",
            "fluid_model",
            "n_seeds",
            "mean_l2",
            "mean_l2_seed_std",
            "median_l2",
            "median_l2_seed_std",
            "max_l2",
            "max_l2_seed_std",
            "negative_rate_mean",
            "negative_rate_seed_std",
        ],
        fluid_rows,
    )

    n_complete = len(run_records)
    n_missing = sum(1 for row in completeness_rows if row[4] == "missing")
    n_invalid = sum(1 for row in completeness_rows if row[4] == "invalid")
    n_cells = len(cells)
    complete_cells = sum(
        1
        for cell in product(ARCHES, SAMPLINGS, GEOMS)
        if len(cells.get(cell, [])) == len(SEEDS)
    )

    print(f"Expected runs : {len(expected)}")
    print(f"Complete runs : {n_complete}")
    print(f"Missing runs  : {n_missing}")
    print(f"Invalid runs  : {n_invalid}")
    print(f"Expected cells: {len(ARCHES) * len(SAMPLINGS) * len(GEOMS)}")
    print(f"Complete cells: {complete_cells}")
    print(f"Cells present : {n_cells}")

    if bad_runs:
        print("\nInvalid runs:")
        for name, detail in bad_runs:
            print(f"  - {name}: {detail}")

    if args.strict and (n_complete != len(expected) or n_invalid):
        raise SystemExit(
            "Strict completeness check failed: the 225-run grid is not complete."
        )

    print(f"\nWrote aggregate tables to: {out_dir}")


if __name__ == "__main__":
    main()
    