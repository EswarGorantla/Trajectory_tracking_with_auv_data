import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any, Optional

# Colorblind-safe palette (Tol bright / Okabe-Ito inspired)
COLORS = {
    "hybrid_mpc": "#2563eb",       # Blue
    "physics_mpc": "#dc2626",      # Red
    "gru_mpc": "#7c3aed",          # Purple
    "reactive_los": "#f59e0b",     # Amber
    "potential_field": "#059669",  # Green
    "obstacle": "#dc2626",
    "safe_buffer": "#f87171",
    "goal": "#16a34a",
    "boundary": "#64748b"
}

def plot_2d_topdown(
    df: pd.DataFrame,
    scenario: Dict[str, Any],
    title: str,
    output_path: str,
    method_name: str = "hybrid_mpc"
):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    b = scenario.get("boundaries", {"north_min": 0, "north_max": 120, "east_min": 0, "east_max": 100})
    bw = b["east_max"] - b["east_min"]
    bh = b["north_max"] - b["north_min"]

    # Boundary rectangle
    rect = plt.Rectangle((b["east_min"], b["north_min"]), bw, bh, fill=True, facecolor="#f8fafc", edgecolor="#94a3b8", linestyle="--", lw=1.5, label="Boundary")
    ax.add_patch(rect)

    # Obstacles
    for i, obs in enumerate(scenario.get("obstacles", [])):
        circle_obs = plt.Circle((obs["east"], obs["north"]), obs["radius"], color=COLORS["obstacle"], alpha=0.7, label="Physical Obstacle" if i == 0 else "")
        circle_buf = plt.Circle((obs["east"], obs["north"]), obs["radius"] + obs.get("margin", 4.0), fill=False, edgecolor=COLORS["safe_buffer"], linestyle=":", lw=1.5, label="Safe Margin" if i == 0 else "")
        ax.add_patch(circle_obs)
        ax.add_patch(circle_buf)

    # Direct line
    start = scenario["start"]
    goal = scenario["goal"]
    ax.plot([start[1], goal[1]], [start[0], goal[0]], 'k:', lw=1.2, alpha=0.4, label="Direct LOS")

    # Realized True Trajectory
    color = COLORS.get(method_name, "#2563eb")
    ax.plot(df["true_east"], df["true_north"], '-', color=color, lw=2.2, label=f"Realized Path ({method_name})")

    # Start & Goal markers
    ax.plot(start[1], start[0], 'go', ms=9, label="Start A")
    ax.plot(goal[1], goal[0], 'r*', ms=14, label="Goal B (5m tol)")
    goal_circle = plt.Circle((goal[1], goal[0]), 5.0, fill=False, edgecolor="#16a34a", linestyle="--", lw=1.2)
    ax.add_patch(goal_circle)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("East Position (meters)", fontsize=10)
    ax.set_ylabel("North Position (meters)", fontsize=10)
    ax.set_xlim(b["east_min"] - 5, b["east_max"] + 5)
    ax.set_ylim(b["north_min"] - 5, b["north_max"] + 5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_trajectory_comparison_2d(
    trajs_dict: Dict[str, pd.DataFrame],
    scenario: Dict[str, Any],
    title: str,
    output_path: str
):
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    b = scenario.get("boundaries", {"north_min": 0, "north_max": 120, "east_min": 0, "east_max": 100})
    bw = b["east_max"] - b["east_min"]
    bh = b["north_max"] - b["north_min"]

    rect = plt.Rectangle((b["east_min"], b["north_min"]), bw, bh, fill=True, facecolor="#f8fafc", edgecolor="#94a3b8", linestyle="--", lw=1.5, label="Boundary")
    ax.add_patch(rect)

    for i, obs in enumerate(scenario.get("obstacles", [])):
        circle_obs = plt.Circle((obs["east"], obs["north"]), obs["radius"], color=COLORS["obstacle"], alpha=0.6, label="Physical Obstacle" if i == 0 else "")
        circle_buf = plt.Circle((obs["east"], obs["north"]), obs["radius"] + obs.get("margin", 4.0), fill=False, edgecolor=COLORS["safe_buffer"], linestyle=":", lw=1.5, label="Safe Margin" if i == 0 else "")
        ax.add_patch(circle_obs)
        ax.add_patch(circle_buf)

    start = scenario["start"]
    goal = scenario["goal"]
    ax.plot([start[1], goal[1]], [start[0], goal[0]], 'k:', lw=1.2, alpha=0.4, label="Direct LOS")

    for method, df in trajs_dict.items():
        col = COLORS.get(method, "#334155")
        ax.plot(df["true_east"], df["true_north"], '-', color=col, lw=2.0, label=method)

    ax.plot(start[1], start[0], 'go', ms=9, label="Start A")
    ax.plot(goal[1], goal[0], 'r*', ms=14, label="Goal B")
    goal_circle = plt.Circle((goal[1], goal[0]), 5.0, fill=False, edgecolor="#16a34a", linestyle="--", lw=1.2)
    ax.add_patch(goal_circle)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("East Position (meters)", fontsize=10)
    ax.set_ylabel("North Position (meters)", fontsize=10)
    ax.set_xlim(b["east_min"] - 5, b["east_max"] + 5)
    ax.set_ylim(b["north_min"] - 5, b["north_max"] + 5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_flight_telemetry(df: pd.DataFrame, title: str, output_path: str):
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), dpi=300, sharex=True)
    t = df["time_sec"]

    # 1. Depth
    axes[0].plot(t, df["true_depth"], color="#0284c7", lw=2, label="True Depth")
    axes[0].plot(t, df["meas_depth"], 'k:', alpha=0.5, label="Measured Depth")
    axes[0].set_ylabel("Depth (m)", fontsize=10)
    axes[0].invert_yaxis()  # Depth positive down
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    # 2. Surge Speed
    axes[1].plot(t, df["true_speed"], color="#16a34a", lw=2, label="True Surge Speed")
    axes[1].plot(t, df["cmd_speed"], '--', color="#ca8a04", lw=1.5, label="Commanded Speed")
    axes[1].axhline(0.5, color="r", linestyle=":", label="Min Limit (0.5 m/s)")
    axes[1].axhline(2.0, color="r", linestyle=":", label="Max Limit (2.0 m/s)")
    axes[1].set_ylabel("Speed (m/s)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    # 3. Pitch Angle
    axes[2].plot(t, np.rad2deg(df["true_pitch"]), color="#7c3aed", lw=2, label="True Pitch")
    axes[2].plot(t, np.rad2deg(df["cmd_pitch"]), '--', color="#ea580c", lw=1.5, label="Commanded Pitch")
    axes[2].axhline(20.0, color="r", linestyle=":", label="Limit (+-20 deg)")
    axes[2].axhline(-20.0, color="r", linestyle=":")
    axes[2].set_ylabel("Pitch (deg)", fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8)

    # 4. Distance to Goal
    axes[3].plot(t, df["dist_to_goal"], color="#dc2626", lw=2, label="Distance to Goal")
    axes[3].axhline(5.0, color="#16a34a", linestyle="--", label="Goal Acceptance (5m)")
    axes[3].set_ylabel("Distance (m)", fontsize=10)
    axes[3].set_xlabel("Elapsed Mission Time (seconds)", fontsize=10)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_method_boxplots(df_summary: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    methods = df_summary["method"].unique()
    data = [df_summary[df_summary["method"] == m][metric].dropna().values for m in methods]
    
    bp = ax.boxplot(data, labels=methods, patch_artist=True, medianprops=dict(color="black", lw=1.5))
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(COLORS.get(method, "#94a3b8"))
        patch.set_alpha(0.7)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
