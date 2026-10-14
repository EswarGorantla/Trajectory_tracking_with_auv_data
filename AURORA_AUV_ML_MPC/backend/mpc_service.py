import time
import numpy as np
from typing import Dict, Any, List

from backend.schemas import MPCRequest, MPCResponse, AUVState
from backend.model_service import model_service
from src.simulator import AUVPlant
from src.mpc import MonteCarloMPC, PIDController
from src.state_estimation import StateEstimator

class MPCService:
    def __init__(self):
        self.ms = model_service

    def _generate_reference_path(self, traj_type: str, n_steps: int, dt: float):
        """Generates desired target path coordinates."""
        t_axis = np.arange(n_steps) * dt
        v_des = 1.2
        s_des = v_des * t_axis

        if traj_type == "step_waypoint":
            # Square/corner waypoint pattern
            x_des = np.zeros(n_steps)
            y_des = np.zeros(n_steps)
            z_des = 12.0 + 3.0 * np.sin(2.0 * np.pi * t_axis / (n_steps * dt))
            for i in range(n_steps):
                if i < n_steps // 3:
                    x_des[i] = v_des * (i * dt)
                    y_des[i] = 0.0
                elif i < 2 * n_steps // 3:
                    x_des[i] = x_des[n_steps // 3 - 1]
                    y_des[i] = v_des * ((i - n_steps // 3) * dt)
                else:
                    x_des[i] = x_des[n_steps // 3 - 1] + v_des * ((i - 2 * n_steps // 3) * dt)
                    y_des[i] = y_des[2 * n_steps // 3 - 1]
        elif traj_type == "straight_dive":
            # Straight survey line with depth ramp
            x_des = 0.9 * s_des
            y_des = np.zeros(n_steps)
            z_des = 10.0 + 0.1 * s_des
        else: # 3d_sinusoid (default)
            x_des = 0.8 * s_des
            y_des = 15.0 * np.sin(2.0 * np.pi * x_des / 150.0)
            z_des = 12.0 - 4.0 * np.cos(2.0 * np.pi * x_des / 150.0)

        return t_axis, x_des, y_des, z_des

    def run_mpc_experiment(self, req: MPCRequest) -> MPCResponse:
        t0 = time.time()
        
        # Load simulation parameters from config
        dt = self.ms.config.get("sampling_time_dt", 2.0)
        n_steps = min(max(req.simulation_steps, 20), 80)
        k_d = self.ms.config.get("drag_coefficient_kd", 0.022498)
        u_cruise = self.ms.config.get("cruise_speed_u", 1.2)
        
        t_axis, x_des, y_des, z_des = self._generate_reference_path(req.trajectory_type, n_steps, dt)
        
        # Setup disturbance
        disturbance = None
        if req.enable_disturbance:
            disturbance = np.array([0.0, req.disturbance_current_east, 0.0])

        # Helper to simulate a controller
        def simulate_controller(use_ai_mode: Any, ctrl_type: str):
            # Create synthetic plant using learned dynamics
            config_dict = {
                "paths": {"models_dir": self.ms.models_dir},
                "hyperparameters": {"seq_len": self.ms.config.get("sequence_length", 10), "hidden_dim": 64},
                "simulation": {"dt": dt, "U_cruise": u_cruise, "k_d_default": k_d},
                "mpc": {"horizon": 5, "num_candidates": 80}
            }
            
            plant = AUVPlant(config_dict, use_ai=True, k_d=k_d, dt=dt, u_cruise=u_cruise, device=self.ms.device)
            
            # Initial state
            psi0 = np.arctan2(y_des[1] - y_des[0], x_des[1] - x_des[0])
            if req.initial_state:
                plant.reset(initial_state=[
                    req.initial_state.north, req.initial_state.east, req.initial_state.depth,
                    req.initial_state.roll, req.initial_state.pitch, req.initial_state.heading,
                    req.initial_state.speed
                ])
            else:
                plant.reset(initial_state=[0.0, 0.0, float(z_des[0]), 0.0, 0.0, float(psi0), u_cruise])

            states = [plant.state.copy()]
            controls = []

            if ctrl_type == "mpc":
                ctrl = MonteCarloMPC(config_dict, use_ai=use_ai_mode, device=self.ms.device)
            else:
                ctrl = PIDController(config_dict)

            for step in range(n_steps - 1):
                if ctrl_type == "mpc":
                    u_opt = ctrl.solve(plant.state, plant.history, step, x_des, y_des, z_des)
                else:
                    u_opt = ctrl.solve(plant.state, step, x_des, y_des, z_des)

                next_st = plant.step(u_opt[0], u_opt[1], u_opt[2], disturbance=disturbance)
                states.append(next_st)
                controls.append(u_opt)

            return np.array(states), np.array(controls)

        # 1. Run PID
        st_pid, ctrl_pid = simulate_controller(None, "pid")
        # 2. Run Physics-only MPC
        st_phys, ctrl_phys = simulate_controller(None, "mpc")
        # 3. Run AI-enhanced MPC
        ai_choice = "lstm" if req.ai_model_name.upper() == "LSTM" else "gru"
        st_ai, ctrl_ai = simulate_controller(ai_choice, "mpc")

        # Compute Metrics
        def compute_metrics(states, controls):
            dist_err = np.sqrt((states[:, 0] - x_des)**2 + (states[:, 1] - y_des)**2)
            pos_rmse = float(np.sqrt(np.mean(dist_err**2)))
            depth_err = np.abs(states[:, 2] - z_des)
            depth_rmse = float(np.sqrt(np.mean(depth_err**2)))
            
            dy, dx = np.diff(y_des), np.diff(x_des)
            path_headings = np.append(np.arctan2(dy, dx), np.arctan2(dy[-1], dx[-1]))
            heading_err = StateEstimator.wrap_angle(states[:, 5] - path_headings)
            heading_rmse = float(np.sqrt(np.mean(heading_err**2)))

            effort = float(np.sum(
                np.diff(controls[:, 0])**2 + 
                np.diff(controls[:, 1])**2 + 
                StateEstimator.wrap_angle(np.diff(controls[:, 2]))**2
            ))
            return {
                "position_rmse_m": round(pos_rmse, 3),
                "depth_rmse_m": round(depth_rmse, 3),
                "heading_rmse_rad": round(heading_rmse, 3),
                "control_effort": round(effort, 3)
            }

        metrics = {
            "PID": compute_metrics(st_pid, ctrl_pid),
            "Physics_MPC": compute_metrics(st_phys, ctrl_phys),
            "AI_MPC": compute_metrics(st_ai, ctrl_ai)
        }

        # Convert trajectories to JSON dictionaries
        desired_traj = []
        pid_traj = []
        phys_traj = []
        ai_traj = []

        for i in range(n_steps):
            t_curr = float(i * dt)
            desired_traj.append({
                "time_s": t_curr,
                "north": float(x_des[i]),
                "east": float(y_des[i]),
                "depth": float(z_des[i])
            })
            pid_traj.append({
                "time_s": t_curr,
                "north": float(st_pid[i, 0]),
                "east": float(st_pid[i, 1]),
                "depth": float(st_pid[i, 2]),
                "heading_deg": float(np.degrees(st_pid[i, 5])),
                "speed": float(st_pid[i, 6])
            })
            phys_traj.append({
                "time_s": t_curr,
                "north": float(st_phys[i, 0]),
                "east": float(st_phys[i, 1]),
                "depth": float(st_phys[i, 2]),
                "heading_deg": float(np.degrees(st_phys[i, 5])),
                "speed": float(st_phys[i, 6])
            })
            ai_traj.append({
                "time_s": t_curr,
                "north": float(st_ai[i, 0]),
                "east": float(st_ai[i, 1]),
                "depth": float(st_ai[i, 2]),
                "heading_deg": float(np.degrees(st_ai[i, 5])),
                "speed": float(st_ai[i, 6])
            })

        dt_ms = (time.time() - t0) * 1000.0
        return MPCResponse(
            success=True,
            trajectory_type=req.trajectory_type,
            desired_trajectory=desired_traj,
            pid_trajectory=pid_traj,
            physics_mpc_trajectory=phys_traj,
            ai_mpc_trajectory=ai_traj,
            metrics=metrics,
            execution_time_ms=round(dt_ms, 2)
        )

mpc_service = MPCService()
