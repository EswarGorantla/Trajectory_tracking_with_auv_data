import os
import sys
import yaml
import numpy as np

# Add Phase_2 to path
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
    print("PHASE 2: FIRST EXPERIMENT — BASIC A -> B GOAL-DIRECTED MPC")
    print("============================================================")

    # Load Configuration
    config_path = os.path.join(phase2_dir, "config", "mpc_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Setup Start Point A & Goal Point B
    start_pos = np.array([0.0, 0.0, 20.0])
    goal_pos = np.array([100.0, 80.0, 40.0])
    goal_tol = float(cfg.get("goal_tolerance", 5.0))

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
    goal = Goal(north=goal_pos[0], east=goal_pos[1], depth=goal_pos[2], tolerance=goal_tol)

    # 2. Setup Operational Boundaries
    boundaries = Boundaries3D(
        north_min=cfg["default_boundaries"]["north_min"],
        north_max=cfg["default_boundaries"]["north_max"],
        east_min=cfg["default_boundaries"]["east_min"],
        east_max=cfg["default_boundaries"]["east_max"],
        depth_min=cfg["default_boundaries"]["depth_min"],
        depth_max=cfg["default_boundaries"]["depth_max"]
    )
    constraint_mgr = ConstraintManager(boundaries=boundaries)

    # 3. Instantiate Phase 1 Learned Model & Dynamics Interface
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

    # 4. Instantiate Cost Function & MPC Controller
    cost_func = MPCCostFunction(
        w_goal=cfg["weight_goal"],
        w_tracking=cfg["weight_tracking"],
        w_control=cfg["weight_control"],
        w_smooth=cfg["weight_smoothness"],
        w_constraint=cfg["weight_constraints"],
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

    # 5. Initialize Closed-Loop Simulator
    simulator = AUVSimulator(
        dynamics=dynamics,
        controller=controller,
        start_state=start_state,
        goal=goal,
        constraint_manager=constraint_mgr,
        dt=cfg["dt"]
    )

    # 6. Execute Closed-Loop Simulation
    df_sim = simulator.run(max_steps=cfg["max_simulation_steps"], verbose=True)

    # 7. Compute & Save Metrics
    results_dir = os.path.join(phase2_dir, "results")
    traj_path = os.path.join(results_dir, "trajectories", "A_to_B.csv")
    metrics_path = os.path.join(results_dir, "metrics", "A_to_B_metrics.json")
    
    TrajectoryManager.save_trajectory(df_sim, traj_path)
    metrics = TrajectoryManager.compute_metrics(df_sim, start_pos, goal, boundaries)
    TrajectoryManager.save_metrics(metrics, metrics_path)

    # 8. Render Publication-Quality Visualizations
    plots_dir = os.path.join(results_dir, "plots")
    vis = TrajectoryVisualizer(plots_dir)
    vis.plot_2d_trajectory(df_sim, start_pos, goal, boundaries, filename="A_to_B_2D.png")
    vis.plot_depth_profile(df_sim, start_pos, goal, boundaries, filename="A_to_B_depth.png")
    vis.plot_3d_trajectory(df_sim, start_pos, goal, boundaries, filename="A_to_B_3D.png")

    # 9. Print Step 12 Required Result Format
    final_p = metrics["final_position"]
    print("\n=====================================")
    print("A -> B MPC RESULT")
    print("=====================================")
    print(f"Start:\n({start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f})")
    print(f"\nGoal:\n({goal.north:.1f}, {goal.east:.1f}, {goal.depth:.1f})")
    print(f"\nFinal position:\n({final_p[0]:.2f}, {final_p[1]:.2f}, {final_p[2]:.2f})")
    print(f"\nFinal distance to goal:\n{metrics['final_distance_to_goal_m']:.2f} m")
    print(f"\nGoal reached:\n{'YES' if metrics['goal_reached'] else 'NO'}")
    print(f"\nTime to goal:\n{metrics['time_to_goal_s']:.1f} s")
    print(f"\nPath length:\n{metrics['total_path_length_m']:.2f} m")
    print(f"\nBoundary violations:\n{metrics['boundary_violations']}")
    print(f"\nConstraint violations:\n{metrics['total_constraint_violations']}")
    print(f"\nTotal control effort:\n{metrics['total_control_effort']:.4f}")
    print(f"\nLearned Model Inference Calls:\n{learned.call_count} batches ({learned.total_samples_inferred:,} states predicted)")
    print("=====================================\n")

if __name__ == "__main__":
    main()
