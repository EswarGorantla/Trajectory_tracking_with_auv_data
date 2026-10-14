import os
import sys
import yaml
import numpy as np
import pandas as pd

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
    print("PHASE 2: STEP 13 — MULTI-SCENARIO START/GOAL EXPERIMENTS")
    print("============================================================")

    config_path = os.path.join(phase2_dir, "config", "mpc_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 3 Configured Scenarios from Project Specification
    scenarios = [
        {
            "name": "Case_1",
            "start": np.array([0.0, 0.0, 20.0]),
            "goal": np.array([100.0, 80.0, 40.0]),
            "desc": "Diagonal climbing dive (North-East)"
        },
        {
            "name": "Case_2",
            "start": np.array([20.0, 10.0, 30.0]),
            "goal": np.array([80.0, 90.0, 50.0]),
            "desc": "Interior boundary transit (Eastward bias)"
        },
        {
            "name": "Case_3",
            "start": np.array([100.0, 20.0, 50.0]),
            "goal": np.array([20.0, 80.0, 20.0]),
            "desc": "South-East ascent transit (High depth to shallow)"
        }
    ]

    boundaries = Boundaries3D(
        north_min=cfg["default_boundaries"]["north_min"],
        north_max=cfg["default_boundaries"]["north_max"],
        east_min=cfg["default_boundaries"]["east_min"],
        east_max=cfg["default_boundaries"]["east_max"],
        depth_min=cfg["default_boundaries"]["depth_min"],
        depth_max=cfg["default_boundaries"]["depth_max"]
    )
    constraint_mgr = ConstraintManager(boundaries=boundaries)

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

    results_dir = os.path.join(phase2_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    vis = TrajectoryVisualizer(plots_dir)

    summary_rows = []

    for sc in scenarios:
        print(f"\n>>> Running Scenario: {sc['name']} ({sc['desc']})")
        start_p = sc["start"]
        goal_p = sc["goal"]

        start_heading = np.arctan2(goal_p[1] - start_p[1], goal_p[0] - start_p[0]) % (2 * np.pi)
        start_state = AUVState(
            north=start_p[0], east=start_p[1], depth=start_p[2],
            roll=0.0, pitch=0.0, heading=start_heading, speed=1.2
        )
        goal = Goal(north=goal_p[0], east=goal_p[1], depth=goal_p[2], tolerance=float(cfg["goal_tolerance"]))

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
            dt=cfg["dt"]
        )

        df_sim = simulator.run(max_steps=cfg["max_simulation_steps"], verbose=False)

        # Save trajectory & metrics
        traj_path = os.path.join(results_dir, "trajectories", f"{sc['name']}.csv")
        TrajectoryManager.save_trajectory(df_sim, traj_path)

        metrics = TrajectoryManager.compute_metrics(df_sim, start_p, goal, boundaries)
        metrics_path = os.path.join(results_dir, "metrics", f"{sc['name']}_metrics.json")
        TrajectoryManager.save_metrics(metrics, metrics_path)

        # Plots
        vis.plot_2d_trajectory(df_sim, start_p, goal, boundaries, title=f"AUV Trajectory ({sc['name']})", filename=f"{sc['name']}_2D.png")
        vis.plot_3d_trajectory(df_sim, start_p, goal, boundaries, title=f"3D AUV Trajectory ({sc['name']})", filename=f"{sc['name']}_3D.png")

        summary_rows.append({
            "Scenario": sc["name"],
            "Description": sc["desc"],
            "Start": f"({start_p[0]:.0f}, {start_p[1]:.0f}, {start_p[2]:.0f})",
            "Goal": f"({goal_p[0]:.0f}, {goal_p[1]:.0f}, {goal_p[2]:.0f})",
            "Success": "YES" if metrics["goal_reached"] else "NO",
            "Time to Goal (s)": metrics["time_to_goal_s"],
            "Path Length (m)": round(metrics["total_path_length_m"], 2),
            "Final Error (m)": round(metrics["final_distance_to_goal_m"], 2),
            "Boundary Violations": metrics["boundary_violations"],
            "Constraint Violations": metrics["total_constraint_violations"],
            "Control Effort": round(metrics["total_control_effort"], 2)
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "metrics", "multi_scenario_results.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved multi-scenario summary table to: {summary_path}\n")

    print("============================================================")
    print("MULTI-SCENARIO EXPERIMENT SUMMARY")
    print("============================================================")
    print(df_summary.to_string(index=False))
    print("============================================================\n")

if __name__ == "__main__":
    main()
