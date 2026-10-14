import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
phase2_dir = os.path.dirname(current_dir)
if phase2_dir not in sys.path:
    sys.path.insert(0, phase2_dir)

from src.state import AUVState
from src.goal import Goal
from src.constraints import Boundaries3D, ConstraintManager
from src.physics_dynamics import PhysicsDynamics
from src.learned_dynamics import LearnedDynamics
from src.dynamics_interface import DynamicsInterface
from src.cost_function import MPCCostFunction
from src.mpc_controller import MPCController
from src.simulator import AUVSimulator
from src.trajectory import TrajectoryManager
from src.evaluation import TrajectoryVisualizer

def main():
    print("============================================================")
    print("PHASE 2: STEP 17 — ROBUSTNESS EXPERIMENT (CURRENT & NOISE)")
    print("============================================================")
    print("NOTE: Simulated ocean currents and sensor jitter are artificially introduced")
    print("to test closed-loop disturbance rejection, distinctly separate from AURORA telemetry.")

    config_path = os.path.join(phase2_dir, "config", "mpc_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    start_pos = np.array([0.0, 0.0, 20.0])
    goal_pos = np.array([100.0, 80.0, 40.0])

    boundaries = Boundaries3D(
        north_min=cfg["default_boundaries"]["north_min"],
        north_max=cfg["default_boundaries"]["north_max"],
        east_min=cfg["default_boundaries"]["east_min"],
        east_max=cfg["default_boundaries"]["east_max"],
        depth_min=cfg["default_boundaries"]["depth_min"],
        depth_max=cfg["default_boundaries"]["depth_max"]
    )
    constraint_mgr = ConstraintManager(boundaries=boundaries)

    start_heading = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0]) % (2 * np.pi)
    start_state = AUVState(
        north=start_pos[0], east=start_pos[1], depth=start_pos[2],
        roll=0.0, pitch=0.0, heading=start_heading, speed=1.2
    )
    goal = Goal(north=goal_pos[0], east=goal_pos[1], depth=goal_pos[2], tolerance=float(cfg["goal_tolerance"]))

    learned = LearnedDynamics()
    physics = PhysicsDynamics(dt=cfg["dt"], u_cruise=cfg["cruise_speed"])
    dynamics = DynamicsInterface(
        physics=physics,
        learned=learned,
        max_pitch_rate=cfg["max_pitch_rate"],
        max_heading_rate=cfg["max_heading_rate"],
        max_speed=cfg["maximum_speed"],
        min_speed=cfg["minimum_speed"]
    )
    cost_func = MPCCostFunction(
        w_goal=cfg["weight_goal"],
        w_tracking=cfg["weight_tracking"],
        w_control=cfg["weight_control"],
        w_smooth=cfg["weight_smoothness"],
        w_constraint=cfg["weight_constraints"],
        u_cruise=cfg["cruise_speed"]
    )

    # 3 Experimental Robustness Test Conditions:
    conditions = [
        {
            "name": "Nominal",
            "current": np.array([0.0, 0.0, 0.0]),
            "noise_std": 0.0,
            "desc": "Calm water, perfect state feedback"
        },
        {
            "name": "Ocean_Current",
            "current": np.array([0.15, -0.12, 0.05]),  # 0.20 m/s cross-current drift
            "noise_std": 0.0,
            "desc": "Simulated 0.20 m/s constant ocean current drift"
        },
        {
            "name": "Current_Plus_Noise",
            "current": np.array([0.15, -0.12, 0.05]),
            "noise_std": 0.50,  # 0.5m acoustic USBL jitter
            "desc": "Simulated ocean current + 0.5m acoustic position jitter"
        }
    ]

    results_dir = os.path.join(phase2_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary_rows = []
    trajectories_dict = {}

    for cond in conditions:
        print(f"\n>>> Running Robustness Test: {cond['name']} ({cond['desc']})")
        np.random.seed(42)

        controller = MPCController(
            dynamics_interface=dynamics,
            cost_function=cost_func,
            prediction_horizon=cfg["prediction_horizon"],
            control_horizon=cfg["control_horizon"],
            num_candidates=cfg["optimizer"]["num_candidates"],
            elite_fraction=cfg["optimizer"]["elite_fraction"],
            iterations=cfg["optimizer"]["iterations"],
            use_ai=True
        )

        simulator = AUVSimulator(
            dynamics=dynamics,
            controller=controller,
            start_state=start_state,
            goal=goal,
            constraint_manager=constraint_mgr,
            dt=cfg["dt"],
            current_disturbance=cond["current"],
            measurement_noise_std=cond["noise_std"]
        )

        df_sim = simulator.run(max_steps=cfg["max_simulation_steps"], verbose=False)
        trajectories_dict[cond["name"]] = df_sim

        traj_path = os.path.join(results_dir, "trajectories", f"robustness_{cond['name']}.csv")
        TrajectoryManager.save_trajectory(df_sim, traj_path)

        metrics = TrajectoryManager.compute_metrics(df_sim, start_pos, goal, boundaries)

        summary_rows.append({
            "Condition": cond["name"],
            "Description": cond["desc"],
            "Goal Reached": "YES" if metrics["goal_reached"] else "NO",
            "Time to Goal (s)": metrics["time_to_goal_s"],
            "Final Error (m)": round(metrics["final_distance_to_goal_m"], 2),
            "Path Length (m)": round(metrics["total_path_length_m"], 2),
            "Boundary Violations": metrics["boundary_violations"],
            "Total Control Effort": round(metrics["total_control_effort"], 2)
        })

    df_robust = pd.DataFrame(summary_rows)
    robust_csv_path = os.path.join(results_dir, "metrics", "robustness_comparison.csv")
    df_robust.to_csv(robust_csv_path, index=False)
    print(f"\nSaved robustness comparison table to: {robust_csv_path}\n")

    # Plot Comparison of Robustness Tracks
    fig, ax = plt.subplots(figsize=(10, 8))
    bw = boundaries.east_max - boundaries.east_min
    bh = boundaries.north_max - boundaries.north_min
    rect = plt.Rectangle((boundaries.east_min, boundaries.north_min), bw, bh, fill=True, facecolor='#f8fafc', edgecolor='#94a3b8', linestyle='--', lw=2.0, label='Boundary')
    ax.add_patch(rect)

    colors = {'Nominal': '#2563eb', 'Ocean_Current': '#059669', 'Current_Plus_Noise': '#d97706'}
    styles = {'Nominal': '-', 'Ocean_Current': '-.', 'Current_Plus_Noise': ':'}

    for name, df_t in trajectories_dict.items():
        ax.plot(df_t["east"], df_t["north"], styles[name], color=colors[name], lw=2.4, label=f'MPC: {name}')

    ax.plot(start_pos[1], start_pos[0], 'go', ms=11, label='START A')
    ax.plot(goal.east, goal.north, 'r*', ms=16, label=f'GOAL B (Tol={goal.tolerance}m)')
    goal_circ = plt.Circle((goal.east, goal.north), goal.tolerance, fill=False, color='#16a34a', linestyle='--')
    ax.add_patch(goal_circ)

    ax.set_title("Robustness Evaluation: MPC Disturbance Rejection", fontsize=13, fontweight='bold')
    ax.set_xlabel("East Position (meters)", fontsize=11)
    ax.set_ylabel("North Position (meters)", fontsize=11)
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 125)
    ax.set_aspect('equal', adjustable='box')

    plot_path = os.path.join(plots_dir, "robustness_comparison_2D.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness plot to: {plot_path}")

    print("============================================================")
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("============================================================")
    print(df_robust.to_string(index=False))
    print("============================================================\n")

if __name__ == "__main__":
    main()
