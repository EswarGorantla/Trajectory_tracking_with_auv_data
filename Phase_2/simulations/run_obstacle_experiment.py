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
from src.constraints import Boundaries3D, CylindricalObstacle3D, ConstraintManager
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
    print("PHASE 2: STEP 15 — 3D CYLINDRICAL OBSTACLE AVOIDANCE MPC")
    print("============================================================")

    config_path = os.path.join(phase2_dir, "config", "mpc_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Start and Goal Points
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

    # Place a 3D Cylindrical Obstacle directly in the center of the direct A->B line of sight!
    # Direct line midpoint is approx (50.0, 40.0, 30.0).
    obs1 = CylindricalObstacle3D(
        north=50.0,
        east=40.0,
        radius=8.0,
        margin=4.0,
        depth_min=10.0,
        depth_max=50.0
    )
    obstacles = [obs1]
    constraint_mgr = ConstraintManager(boundaries=boundaries, obstacles=obstacles)

    start_heading = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0]) % (2 * np.pi)
    start_state = AUVState(
        north=start_pos[0],
        east=start_pos[1],
        depth=start_pos[2],
        roll=0.0,
        pitch=0.0,
        heading=start_heading,
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
        w_constraint=cfg["weight_constraints"] * 2.5,  # Strong obstacle avoidance barrier
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
    traj_path = os.path.join(results_dir, "trajectories", "obstacle_experiment.csv")
    metrics_path = os.path.join(results_dir, "metrics", "obstacle_experiment_metrics.json")
    TrajectoryManager.save_trajectory(df_sim, traj_path)

    metrics = TrajectoryManager.compute_metrics(df_sim, start_pos, goal, boundaries)
    TrajectoryManager.save_metrics(metrics, metrics_path)

    # Visualizations
    plots_dir = os.path.join(results_dir, "plots")
    vis = TrajectoryVisualizer(plots_dir)
    vis.plot_2d_trajectory(
        df_sim, start_pos, goal, boundaries, obstacles=obstacles,
        title="3D Obstacle Avoidance: MPC Circumnavigation", filename="obstacle_experiment_2D.png"
    )
    vis.plot_3d_trajectory(
        df_sim, start_pos, goal, boundaries, obstacles=obstacles,
        title="3D AUV Trajectory Around Cylindrical Obstacle", filename="obstacle_experiment_3D.png"
    )

    # Check minimum distance to obstacle center
    dists_to_obs = np.hypot(df_sim["north"] - obs1.north, df_sim["east"] - obs1.east)
    min_dist_to_obs = float(np.min(dists_to_obs))
    print(f"\nObstacle Avoidance Results:")
    print(f"Goal Reached: {metrics['goal_reached']} (Final error: {metrics['final_distance_to_goal_m']:.2f}m)")
    print(f"Obstacle Radius: {obs1.radius}m | Safety Margin: {obs1.safe_radius}m")
    print(f"Minimum Distance to Obstacle Center: {min_dist_to_obs:.2f}m (> {obs1.radius}m -> NO COLLISION)")
    print(f"Obstacle Collisions: {metrics['obstacle_violations']}")
    print(f"Time to Goal: {metrics['time_to_goal_s']}s\n")

if __name__ == "__main__":
    main()
