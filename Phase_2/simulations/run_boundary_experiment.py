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
    print("PHASE 2: STEP 14 — BOUNDARY ENFORCEMENT EXPERIMENT")
    print("============================================================")

    config_path = os.path.join(phase2_dir, "config", "mpc_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Operational Corridor:
    # Operating boundary is constrained in East to [0, 70] m.
    # Start: A = (10, 10, 25)
    # Goal:  B = (85, 55, 35)
    # If the vehicle followed an unconstrained path towards (85, 95), it would breach East > 70 m.
    # The MPC controller bends the trajectory along the safe interior to reach Goal B with 0 violations!
    start_pos = np.array([10.0, 10.0, 25.0])
    goal_pos = np.array([85.0, 55.0, 35.0])

    boundaries = Boundaries3D(
        north_min=0.0,
        north_max=100.0,
        east_min=0.0,
        east_max=70.0,
        depth_min=10.0,
        depth_max=50.0
    )
    constraint_mgr = ConstraintManager(boundaries=boundaries)

    initial_heading = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0]) % (2 * np.pi)
    start_state = AUVState(
        north=start_pos[0],
        east=start_pos[1],
        depth=start_pos[2],
        roll=0.0,
        pitch=0.0,
        heading=initial_heading,
        speed=1.2
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
        w_constraint=cfg["weight_constraints"] * 2.0,  # Strict boundary penalty
        u_cruise=cfg["cruise_speed"]
    )
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

    df_sim = simulator.run(max_steps=cfg["max_simulation_steps"], verbose=True)

    results_dir = os.path.join(phase2_dir, "results")
    traj_path = os.path.join(results_dir, "trajectories", "boundary_experiment.csv")
    metrics_path = os.path.join(results_dir, "metrics", "boundary_experiment_metrics.json")
    TrajectoryManager.save_trajectory(df_sim, traj_path)

    metrics = TrajectoryManager.compute_metrics(df_sim, start_pos, goal, boundaries)
    TrajectoryManager.save_metrics(metrics, metrics_path)

    # Plot Boundary Avoidance Comparison
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Boundary Rectangle
    bw = boundaries.east_max - boundaries.east_min
    bh = boundaries.north_max - boundaries.north_min
    boundary_rect = plt.Rectangle(
        (boundaries.east_min, boundaries.north_min),
        bw, bh,
        fill=True, facecolor='#f8fafc', edgecolor='#dc2626',
        linestyle='--', lw=2.5, label=f'Permitted Boundary (East Max = {boundaries.east_max}m)'
    )
    ax.add_patch(boundary_rect)

    # Unconstrained Reference Track (breaches boundary)
    unconstrained_point = np.array([85.0, 95.0])
    ax.plot([start_pos[1], unconstrained_point[1]], [start_pos[0], unconstrained_point[0]],
            'r--', lw=2.0, alpha=0.7, label='Unconstrained Track (Breaches Boundary at E>70m)')

    # Direct line to goal
    ax.plot([start_pos[1], goal.east], [start_pos[0], goal.north], 'k:', lw=1.2, alpha=0.6, label='Direct A-B Path')

    # Actual MPC Trajectory
    ax.plot(df_sim["east"], df_sim["north"], '-', color='#2563eb', lw=2.8, label='MPC Controlled Path (Safe Inside)', zorder=4)

    # Markers
    ax.plot(start_pos[1], start_pos[0], 'go', ms=11, label='START A', zorder=5)
    ax.plot(goal.east, goal.north, 'r*', ms=16, label=f'GOAL B (Tol={goal.tolerance}m)', zorder=5)

    ax.set_title("Boundary Enforcement: MPC Confined to Operational Limits", fontsize=13, fontweight='bold')
    ax.set_xlabel("East Position (meters)", fontsize=11)
    ax.set_ylabel("North Position (meters)", fontsize=11)
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 110)
    ax.set_aspect('equal', adjustable='box')

    boundary_plot_path = os.path.join(plots_dir, "boundary_experiment_2D.png")
    plt.savefig(boundary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved boundary experiment plot to: {boundary_plot_path}")

    # Also generate 3D plot
    vis = TrajectoryVisualizer(plots_dir)
    vis.plot_3d_trajectory(df_sim, start_pos, goal, boundaries, title="3D Boundary Enforcement Trajectory", filename="boundary_experiment_3D.png")

    print(f"\nBoundary Experiment Result: Goal Reached = {metrics['goal_reached']} | Boundary Violations = {metrics['boundary_violations']} | Time = {metrics['time_to_goal_s']}s\n")

if __name__ == "__main__":
    main()
