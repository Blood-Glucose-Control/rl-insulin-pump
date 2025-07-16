#!/usr/bin/env python3
"""
Analysis script for the retroactively organized training run.
Analyzes the SLURM output and training progress.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_slurm_output(slurm_file):
    """Parse the SLURM output file to extract training metrics."""
    if not os.path.exists(slurm_file):
        print(f"SLURM file not found: {slurm_file}")
        return None

    with open(slurm_file, "r") as f:
        content = f.read()

    # Extract training metrics
    metrics = []
    patterns = {
        "episode_length": r"ep_len_mean\s+\|\s+([\d.]+)",
        "episode_reward": r"ep_rew_mean\s+\|\s+([\d.]+)",
        "episodes": r"episodes\s+\|\s+(\d+)",
        "total_timesteps": r"total_timesteps\s+\|\s+(\d+)",
        "actor_loss": r"actor_loss\s+\|\s+([-\d.]+)",
        "critic_loss": r"critic_loss\s+\|\s+([\d.e+]+)",
    }

    # Find all metric blocks
    metric_blocks = re.findall(r"-{30,}.*?-{30,}", content, re.DOTALL)

    for block in metric_blocks:
        metric_dict = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, block)
            if match:
                try:
                    metric_dict[key] = float(match.group(1))
                except ValueError:
                    metric_dict[key] = match.group(1)

        if metric_dict:
            metrics.append(metric_dict)

    return pd.DataFrame(metrics) if metrics else None


def plot_training_progress(df, save_dir):
    """Plot training progress from metrics."""
    if df is None or df.empty:
        print("No metrics data to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Training Progress Analysis", fontsize=16)

    # Episode length over time
    if "episode_length" in df.columns:
        axes[0, 0].plot(df.index, df["episode_length"], "b-")
        axes[0, 0].set_title("Episode Length")
        axes[0, 0].set_xlabel("Training Block")
        axes[0, 0].set_ylabel("Mean Episode Length")

    # Episode reward over time
    if "episode_reward" in df.columns:
        axes[0, 1].plot(df.index, df["episode_reward"], "g-")
        axes[0, 1].set_title("Episode Reward")
        axes[0, 1].set_xlabel("Training Block")
        axes[0, 1].set_ylabel("Mean Episode Reward")

    # Total timesteps
    if "total_timesteps" in df.columns:
        axes[0, 2].plot(df.index, df["total_timesteps"], "r-")
        axes[0, 2].set_title("Total Timesteps")
        axes[0, 2].set_xlabel("Training Block")
        axes[0, 2].set_ylabel("Total Timesteps")

    # Actor loss
    if "actor_loss" in df.columns:
        axes[1, 0].plot(df.index, df["actor_loss"], "purple")
        axes[1, 0].set_title("Actor Loss")
        axes[1, 0].set_xlabel("Training Block")
        axes[1, 0].set_ylabel("Actor Loss")

    # Critic loss
    if "critic_loss" in df.columns:
        axes[1, 1].plot(df.index, df["critic_loss"], "orange")
        axes[1, 1].set_title("Critic Loss")
        axes[1, 1].set_xlabel("Training Block")
        axes[1, 1].set_ylabel("Critic Loss")

    # Episodes over time
    if "episodes" in df.columns:
        axes[1, 2].plot(df.index, df["episodes"], "brown")
        axes[1, 2].set_title("Number of Episodes")
        axes[1, 2].set_xlabel("Training Block")
        axes[1, 2].set_ylabel("Episodes Completed")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_analysis.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Training analysis plot saved to {save_dir}/training_analysis.png")


def analyze_patient_switching(slurm_file):
    """Analyze patient switching pattern from SLURM output."""
    if not os.path.exists(slurm_file):
        return None

    with open(slurm_file, "r") as f:
        content = f.read()

    # Extract patient switching information
    patient_switches = re.findall(
        r"Training on patient (\w+#\d+) \((\d+)/(\d+)\)", content
    )
    forced_switches = re.findall(
        r"PatientSwitchCallback: Forcing patient switch after (\d+) steps", content
    )

    return {
        "patient_switches": patient_switches,
        "forced_switches": forced_switches,
        "total_patients": len(set([p[0] for p in patient_switches])),
        "total_switches": len(patient_switches),
    }


def main():
    """Main analysis function."""
    run_dir = Path(__file__).parent
    slurm_file = run_dir / "slurm" / "slurm-1335771.out"

    print("Analyzing training run...")
    print(f"Run directory: {run_dir}")

    # Parse SLURM output
    df = parse_slurm_output(slurm_file)
    if df is not None:
        print(f"\nFound {len(df)} training metric blocks")
        print("\nTraining Summary:")
        print(f"Final episode reward: {df['episode_reward'].iloc[-1]:.2f}")
        print(f"Final episode length: {df['episode_length'].iloc[-1]:.2f}")
        print(f"Total timesteps: {df['total_timesteps'].iloc[-1]:.0f}")
        print(f"Final actor loss: {df['actor_loss'].iloc[-1]:.2f}")
        print(f"Final critic loss: {df['critic_loss'].iloc[-1]:.2f}")

        # Save metrics to CSV
        df.to_csv(run_dir / "results" / "training_metrics.csv", index=False)
        print(f"Metrics saved to: {run_dir}/results/training_metrics.csv")

        # Create plots
        plot_training_progress(df, run_dir / "results")

    # Analyze patient switching
    patient_info = analyze_patient_switching(slurm_file)
    if patient_info:
        print("\nPatient Switching Analysis:")
        print(f"Total unique patients: {patient_info['total_patients']}")
        print(f"Total switches: {patient_info['total_switches']}")
        print(f"Forced switches: {len(patient_info['forced_switches'])}")

    # List all available files
    print("\nRun Directory Contents:")
    for root, dirs, files in os.walk(run_dir):
        level = root.replace(str(run_dir), "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            if file != os.path.basename(__file__):  # Don't list this script
                print(f"{subindent}{file}")


if __name__ == "__main__":
    main()
