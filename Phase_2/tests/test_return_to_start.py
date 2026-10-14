import os
import sys
import numpy as np

# Ensure Phase_2 path is loaded
current_dir = os.path.dirname(os.path.abspath(__file__))
phase2_dir = os.path.dirname(current_dir)
if phase2_dir not in sys.path:
    sys.path.insert(0, phase2_dir)

from src.simulation_runner import ClosedLoopSimulation

def test_return_disabled():
    """Case 1: Return disabled (r = 0): A -> B -> Stop"""
    print("\n" + "=" * 70)
    print("TEST 1: Return Disabled (r = 0): Outbound A -> B only")
    print("=" * 70)

    scenario_cfg = {
        "name": "test_return_disabled",
        "start": [0.0, 0.0, 20.0],
        "goal": [60.0, 50.0, 30.0],
        "boundaries": {
            "north_min": -10.0, "north_max": 100.0,
            "east_min": -10.0, "east_max": 100.0,
            "depth_min": 10.0, "depth_max": 60.0
        },
        "obstacles": [],
        "current": [0.0, 0.0, 0.0]
    }

    sim = ClosedLoopSimulation(
        method="hybrid_mpc",
        scenario_config=scenario_cfg,
        seed=101,
        dt_control=2.0,
        max_steps=60,
        goal_tolerance_primary=5.0,
        horizon=10,
        num_candidates=25,
        iterations=2,
        return_to_start=False
    )

    df_traj, metrics = sim.run()

    print(f"  Reached B:                 {metrics['reached_b']}")
    print(f"  Time reached B:            {metrics['time_reached_b_sec']} s")
    print(f"  Final distance to B:       {metrics['final_distance_to_b_m']:.2f} m")
    print(f"  Return to start enabled:   {metrics['return_to_start']}")
    print(f"  Reached A (Return):        {metrics['reached_a']}")
    print(f"  Total path length:         {metrics['total_path_length_m']:.2f} m")
    print(f"  Legs present in log:       {list(df_traj['leg'].unique())}")
    print(f"  Final Movement Phase:      {metrics['movement_phase']}")

    # Assertions
    assert metrics['reached_b'] is True, "Failed: Robot should reach Point B within tolerance"
    assert metrics['final_distance_to_b_m'] <= 5.0, f"Failed: Final distance to B ({metrics['final_distance_to_b_m']}) > tolerance 5.0m"
    assert metrics['return_to_start'] is False, "Failed: return_to_start should be False"
    assert metrics['reached_a'] is None, "Failed: reached_a should be None when return disabled"
    assert list(df_traj['leg'].unique()) == [1], f"Failed: Only leg 1 should be present, got {df_traj['leg'].unique()}"
    print("  --> [PASS] Case 1 passed: Robot moved from A to B and stopped without initiating return.")
    return metrics


def test_return_enabled():
    """Case 2: Return enabled (r = 1): A -> B -> A -> Stop"""
    print("\n" + "=" * 70)
    print("TEST 2: Return Enabled (r = 1): Round-trip A -> B -> A")
    print("=" * 70)

    scenario_cfg = {
        "name": "test_return_enabled",
        "start": [0.0, 0.0, 20.0],
        "goal": [60.0, 50.0, 30.0],
        "boundaries": {
            "north_min": -10.0, "north_max": 100.0,
            "east_min": -10.0, "east_max": 100.0,
            "depth_min": 10.0, "depth_max": 60.0
        },
        "obstacles": [],
        "current": [0.0, 0.0, 0.0]
    }

    sim = ClosedLoopSimulation(
        method="hybrid_mpc",
        scenario_config=scenario_cfg,
        seed=101,
        dt_control=2.0,
        max_steps=120,
        goal_tolerance_primary=5.0,
        horizon=10,
        num_candidates=25,
        iterations=2,
        return_to_start=True
    )

    df_traj, metrics = sim.run()

    print(f"  Reached B (Outbound):      {metrics['reached_b']}")
    print(f"  Step reached B:            {metrics['step_reached_b']}")
    print(f"  Time reached B:            {metrics['time_reached_b_sec']} s")
    print(f"  Reached A (Return):        {metrics['reached_a']}")
    print(f"  Step reached A:            {metrics['step_reached_a']}")
    print(f"  Time reached A:            {metrics['time_reached_a_sec']} s")
    print(f"  Final distance to A:       {metrics['final_distance_to_a_m']:.2f} m")
    print(f"  Forward path length:       {metrics['forward_path_length_m']:.2f} m")
    print(f"  Return path length:        {metrics['return_path_length_m']:.2f} m")
    print(f"  Total path length:         {metrics['total_path_length_m']:.2f} m")
    print(f"  Legs present in log:       {list(df_traj['leg'].unique())}")
    print(f"  Final Movement Phase:      {metrics['movement_phase']}")
    print(f"  Success (5m tolerance):    {metrics['success_5m']}")

    # Assertions
    assert metrics['reached_b'] is True, "Failed: Robot should reach Point B within tolerance"
    assert 'ROTATING_BACK' in df_traj['phase'].unique(), "Failed: ROTATING_BACK phase must be executed at Point B"
    assert metrics['reached_a'] is True, "Failed: Robot should reach Point A on return leg within tolerance"
    assert metrics['final_distance_to_a_m'] <= 5.0, f"Failed: Final distance to A ({metrics['final_distance_to_a_m']}) > tolerance 5.0m"
    assert 1 in df_traj['leg'].unique() and 2 in df_traj['leg'].unique(), "Failed: Both legs (1 and 2) must be logged in telemetry"
    assert metrics['return_path_length_m'] > 0.0, "Failed: Return path length must be positive"
    assert metrics['success_5m'] is True, "Failed: Mission overall success should be True"
    print("  --> [PASS] Case 2 passed: Rover rotated back at Point B and returned to original position A.")
    return metrics


if __name__ == "__main__":
    m1 = test_return_disabled()
    m2 = test_return_enabled()
    print("\n" + "=" * 70)
    print("ALL POINT-TO-POINT RETURN UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
