from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

FILES = [
    (
        "Unconstrained DeepONet",
        "Uniform",
        ROOT / "results/leave_one_geometry_out_sinusoidal_summary.csv",
        "Unconstrained",
    ),
    (
        "Power-law-aware DeepONet",
        "Uniform",
        ROOT / "results/leave_one_geometry_out_sinusoidal_sampling_baseline_uniform_summary.csv",
        "Power-law-aware",
    ),
    (
        "Power-law-aware DeepONet",
        "Geometry-aware sampling",
        ROOT / "results/leave_one_geometry_out_sinusoidal_sampling_zequal_eta_wallstretched_k2_powerlaw_summary.csv",
        "Power-law-aware",
    ),
    (
        "Geometry-channel-aware DeepONet",
        "Uniform",
        ROOT / "results/leave_one_geometry_out_sinusoidal_domain_channel_uniform_summary.csv",
        "Geometry-channel-aware",
    ),
    (
        "Geometry-channel-aware DeepONet",
        "Geometry-aware sampling",
        ROOT / "results/leave_one_geometry_out_sinusoidal_domain_channel_mesh_summary.csv",
        "Geometry-channel-aware",
    ),
]

rows = []

for model_name, sampling, csv_file, internal_model_name in FILES:

    df = pd.read_csv(csv_file)

    row = df[
        (df["model"] == internal_model_name)
        & (df["fluid_model"] == "all")
    ].iloc[0]

    rows.append(
        {
            "Model": model_name,
            "Sampling": sampling,
            "Mean_L2": row["mean_relative_l2"],
            "Median_L2": row["median_relative_l2"],
            "Max_L2": row["max_relative_l2"],
            "Negative_Predictions": int(row["negative_prediction_count"]),
            "Held_Out_Geometry": "sinusoidal",
        }
    )

master = pd.DataFrame(rows)

output_file = (
    ROOT
    / "papers/sampling_inductive_bias/results/master_results_table.csv"
)

master.to_csv(output_file, index=False)

print(master)
print()
print(f"Saved: {output_file}")
