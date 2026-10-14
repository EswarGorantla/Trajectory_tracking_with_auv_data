import os
import sys
import hashlib
import numpy as np
import unittest
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p1_dir = os.path.join(root_dir, "presentationphase1")
p2_dir = os.path.join(root_dir, "Phase_2")
sys.path.insert(0, p1_dir)
sys.path.insert(0, p2_dir)

from src.state import AUVState, AUVControl, wrap_angle, wrap_to_2pi
from src.goal import Goal
from src.constraints import Boundaries3D, CylindricalObstacle3D, ConstraintManager
from src.physics_dynamics import PhysicsDynamics
from src.learned_dynamics import LearnedDynamics
from src.dynamics_interface import DynamicsInterface
from src.mpc_controller import MPCController
from src.cost_function import MPCCostFunction
from src.true_plant import TrueAUVPlant, PlantParameters, CylindricalObstacle

# 1. Checkpoint Hash Immutability Assertion
FROZEN_HASHES = {
    os.path.join(p1_dir, "models", "best_dynamics_model.pt"): "9BF7305EDD14FB2BA61FD9A293E687D604F1DD9B76592D835A1016E242AA9FF2",
    os.path.join(p1_dir, "models", "scaler.pkl"): "AE84FFFF1E0FA8F7CE21B9A3E6C99FC117BAA712E16F60551536976F71A14670",
    os.path.join(p1_dir, "models", "model_config.json"): "569A9BC2E694D644C59DA854B64C206D48BFF748E6C401A79A236D000F331C41"
}

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()

def test_checkpoint_hash_immutability():
    for fpath, expected_hash in FROZEN_HASHES.items():
        actual_hash = get_sha256(fpath)
        assert actual_hash == expected_hash, f"Hash mismatch for {fpath}! Expected {expected_hash}, got {actual_hash}"

# 2. Angle Wrapping Unit Tests
def test_angle_wrapping():
    assert np.isclose(wrap_to_2pi(0.0), 0.0)
    assert np.isclose(wrap_to_2pi(2 * np.pi), 0.0)
    assert np.isclose(wrap_to_2pi(-0.1), 2 * np.pi - 0.1)
    assert np.isclose(wrap_to_2pi(2.5 * np.pi), 0.5 * np.pi)

    assert np.isclose(wrap_angle(0.0), 0.0)
    assert np.isclose(wrap_angle(np.pi), np.pi) or np.isclose(wrap_angle(np.pi), -np.pi)
    assert np.isclose(wrap_angle(2 * np.pi), 0.0)
    assert np.isclose(wrap_angle(3 * np.pi), np.pi) or np.isclose(wrap_angle(3 * np.pi), -np.pi)
    assert np.isclose(wrap_angle(-np.pi - 0.1), np.pi - 0.1)

# 3. Sign Conventions Unit Tests
def test_sign_conventions_and_geometry():
    physics = PhysicsDynamics(dt=2.0, u_cruise=1.2)
    
    # Positive pitch = climb -> depth decreases
    st = np.array([0.0, 0.0, 50.0, 0.0, 0.15, 0.0, 1.2]) # theta = +0.15 rad
    st_next = physics.step(st)
    assert st_next[2] < 50.0, "Positive pitch must decrease depth (climb towards surface)"
    
    # Negative pitch = dive -> depth increases
    st_dive = np.array([0.0, 0.0, 50.0, 0.0, -0.15, 0.0, 1.2])
    st_next_dive = physics.step(st_dive)
    assert st_next_dive[2] > 50.0, "Negative pitch must increase depth (dive deeper)"

    # LOS Pitch Sign: when goal is deeper (z_goal > z_current), pitch command must be negative (dive)
    cur_auv = AUVState(north=0.0, east=0.0, depth=20.0, roll=0.0, pitch=0.0, heading=0.0, speed=1.2)
    goal_deeper = Goal(north=50.0, east=50.0, depth=40.0)
    los_pitch = goal_deeper.los_pitch_to(cur_auv)
    assert los_pitch < 0.0, "LOS pitch toward deeper target must be negative (nose down)"

# 4. Slew Limiter Tests
def test_slew_limiter():
    dyn = DynamicsInterface(max_pitch_rate=0.10, max_heading_rate=0.20, max_speed_rate=0.15)
    st = AUVState(north=0.0, east=0.0, depth=20.0, roll=0.0, pitch=0.0, heading=0.0, speed=1.2)
    hist = np.tile(st.to_feature_vector(), (10, 1))
    
    # Request extreme control changes
    extreme_ctrl = AUVControl(speed_cmd=2.0, pitch_cmd=0.50, heading_cmd=1.0)
    next_st, _ = dyn.predict_next_state(st, extreme_ctrl, hist, use_ai=False)
    
    # Check rate clipping
    assert np.isclose(next_st.pitch - st.pitch, 0.10), "Pitch slew rate limit exceeded"
    assert np.isclose(next_st.heading - st.heading, 0.20), "Heading slew rate limit exceeded"
    assert (next_st.speed - st.speed) <= 0.15 + 1e-5, "Speed slew rate limit exceeded"

# 5. Batch vs Reference Rollout Numerical Equivalence
def test_batch_vs_reference_rollout_equivalence():
    physics = PhysicsDynamics(dt=2.0, u_cruise=1.2)
    learned = LearnedDynamics()
    dyn = DynamicsInterface(physics=physics, learned=learned)
    
    init_state = AUVState(north=10.0, east=20.0, depth=30.0, roll=0.2, pitch=0.05, heading=1.2, speed=1.2)
    init_hist = np.tile(init_state.to_feature_vector(), (10, 1))
    
    rng = np.random.default_rng(42)
    K = 5
    H = 4
    candidates = rng.uniform(low=[0.8, -0.2, 0.0], high=[1.5, 0.2, 2 * np.pi], size=(K, H, 3))
    
    # 1. Vectorized Batch Rollout
    batch_trajs, _ = dyn.predict_rollout_batch(init_state, init_hist, candidates, use_ai=True)
    
    # 2. Sequential Single-Step Rollout Loop
    seq_trajs = np.zeros_like(batch_trajs)
    for k in range(K):
        cur_st = init_state
        cur_h = init_hist.copy()
        seq_trajs[k, 0, :] = cur_st.to_array()
        for h in range(H):
            ctrl = AUVControl(speed_cmd=candidates[k, h, 0], pitch_cmd=candidates[k, h, 1], heading_cmd=candidates[k, h, 2])
            next_st, cur_h = dyn.predict_next_state(cur_st, ctrl, cur_h, use_ai=True)
            seq_trajs[k, h + 1, :] = next_st.to_array()
            cur_st = next_st
            
    max_diff = np.max(np.abs(batch_trajs - seq_trajs))
    assert max_diff < 1e-5, f"Vectorized batch rollout differs from sequential reference by {max_diff:.2e}!"

# 6. Obstacle and Boundary Penalties
def test_obstacle_and_boundary_penalties():
    boundaries = Boundaries3D(north_min=0.0, north_max=100.0, east_min=0.0, east_max=100.0, depth_min=10.0, depth_max=50.0)
    obstacle = CylindricalObstacle3D(north=50.0, east=50.0, radius=8.0, margin=4.0)
    mgr = ConstraintManager(boundaries=boundaries, obstacles=[obstacle])
    
    # Safe trajectory inside boundaries, far from obstacle, with valid speed
    safe_traj = np.zeros((1, 5, 7))
    safe_traj[0, :, :3] = np.array([[10, 10, 20], [20, 20, 20], [30, 30, 20], [40, 30, 20], [45, 30, 20]])
    safe_traj[0, :, 6] = 1.2  # Valid cruise speed
    cost_safe = mgr.evaluate_batch_penalties(safe_traj)
    assert np.isclose(cost_safe[0], 0.0), "Safe trajectory should have zero constraint penalty"
    
    # Violating boundary trajectory
    wall_traj = np.zeros((1, 5, 7))
    wall_traj[0, :, :3] = np.array([[10, 10, 20], [50, 50, 20], [80, 80, 20], [105, 80, 20], [110, 80, 20]])
    cost_wall = mgr.evaluate_batch_penalties(wall_traj)
    assert cost_wall[0] > 0.0, "Boundary breach must incur positive barrier penalty"
    
    # Obstacle buffer breach trajectory
    obs_traj = np.zeros((1, 5, 7))
    obs_traj[0, :, :3] = np.array([[50, 40, 20], [50, 42, 20], [50, 45, 20], [50, 48, 20], [50, 50, 20]])
    cost_obs = mgr.evaluate_batch_penalties(obs_traj)
    assert cost_obs[0] >= 100.0, "Obstacle buffer breach must incur strong barrier penalty"

# 7. Deterministic Seed Reproducibility
def test_deterministic_seed_reproducibility():
    dyn = DynamicsInterface()
    ctrl1 = MPCController(dyn, seed=12345)
    ctrl2 = MPCController(dyn, seed=12345)
    
    st = AUVState(north=0.0, east=0.0, depth=20.0, roll=0.0, pitch=0.0, heading=0.0, speed=1.2)
    hist = np.tile(st.to_feature_vector(), (10, 1))
    goal = Goal(north=100.0, east=80.0, depth=40.0)
    mgr = ConstraintManager()
    
    act1, traj1, cost1, _ = ctrl1.compute_control(st, hist, goal, mgr)
    act2, traj2, cost2, _ = ctrl2.compute_control(st, hist, goal, mgr)
    
    assert np.isclose(cost1, cost2), "Same seed must yield identical MPC cost"
    assert np.allclose(traj1, traj2), "Same seed must yield identical predicted optimal trajectory"
    assert np.isclose(act1.speed_cmd, act2.speed_cmd)
    assert np.isclose(act1.pitch_cmd, act2.pitch_cmd)
    assert np.isclose(act1.heading_cmd, act2.heading_cmd)

if __name__ == "__main__":
    print("Running Phase 2 Simulation & Safety Unit Test Suite...")
    test_checkpoint_hash_immutability()
    print("  [PASS] 1. Checkpoint SHA-256 Immutability")
    test_angle_wrapping()
    print("  [PASS] 2. Modular Angle Wrapping")
    test_sign_conventions_and_geometry()
    print("  [PASS] 3. Marine Sign Conventions & 3D LOS Geometry")
    test_slew_limiter()
    print("  [PASS] 4. Actuator Slew Rate Limiter")
    test_batch_vs_reference_rollout_equivalence()
    print("  [PASS] 5. Vectorized CEM Batch vs Reference Rollout Equivalence")
    test_obstacle_and_boundary_penalties()
    print("  [PASS] 6. Obstacle & Boundary Barrier Penalties")
    test_deterministic_seed_reproducibility()
    print("  [PASS] 7. Deterministic Seed Reproducibility")
    print("\nALL 7 TESTS PASSED SUCCESSFULLY!")
