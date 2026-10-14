import numpy as np
from typing import Tuple, List, Union, Optional
from src.state import AUVState, AUVControl, wrap_angle, wrap_to_2pi
from src.physics_dynamics import PhysicsDynamics
from src.learned_dynamics import LearnedDynamics

class DynamicsInterface:
    """
    Unified Modular Dynamics Interface for Model Predictive Control (MPC).
    Connects receding-horizon optimization to:
      1. 'hybrid': 3D Marine Kinematics/Drag + Phase 1 Deep GRU Residual Model
      2. 'physics_only': Pure First-Principles Marine Kinematics & Drag
      3. 'gru_only': Pure Data-Driven Residual Delta Propagation
    """

    def __init__(
        self,
        physics: Optional[PhysicsDynamics] = None,
        learned: Optional[LearnedDynamics] = None,
        max_pitch_rate: float = 0.10,
        max_heading_rate: float = 0.20,
        max_speed_rate: float = 0.15,
        max_speed: float = 2.0,
        min_speed: float = 0.5,
        dynamics_mode: str = "hybrid",
        include_ne_residual: bool = True
    ):
        self.physics = physics or PhysicsDynamics()
        self.learned = learned or LearnedDynamics()
        self.max_pitch_rate = max_pitch_rate
        self.max_heading_rate = max_heading_rate
        self.max_speed_rate = max_speed_rate
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.dynamics_mode = dynamics_mode
        self.include_ne_residual = include_ne_residual

    def predict_next_state(
        self,
        current_state: AUVState,
        control: AUVControl,
        history_features: np.ndarray,
        use_ai: Optional[bool] = None
    ) -> Tuple[AUVState, np.ndarray]:
        """
        Advances AUV state by one timestep (dt = 2.0s) given control command and 10-step history.
        """
        st_arr = current_state.to_array()
        mode = self.dynamics_mode
        if use_ai is not None:
            mode = "hybrid" if use_ai else "physics_only"

        # 1. Autopilot Steering Actuation Dynamics (Fin rate limits)
        theta_act = current_state.pitch + np.clip(
            control.pitch_cmd - current_state.pitch,
            -self.max_pitch_rate,
            self.max_pitch_rate
        )
        heading_err = wrap_angle(control.heading_cmd - current_state.heading)
        psi_act = wrap_to_2pi(
            current_state.heading + np.clip(heading_err, -self.max_heading_rate, self.max_heading_rate)
        )
        u_target = np.clip(control.speed_cmd, self.min_speed, self.max_speed)
        u_act = current_state.speed + np.clip(
            u_target - current_state.speed,
            -self.max_speed_rate,
            self.max_speed_rate
        )

        act_state = st_arr.copy()
        act_state[4] = theta_act
        act_state[5] = psi_act
        act_state[6] = u_act

        # 2. Physics Baseline
        phys_next = self.physics.step(act_state)

        # 3. Residual Prediction
        if mode in ["hybrid", "gru_only"] and history_features is not None and len(history_features) == 10:
            residuals = self.learned.predict_residual(history_features)
        else:
            residuals = np.zeros(7, dtype=np.float64)

        # 4. Synthesize Next State
        if mode == "gru_only":
            x_next = current_state.north + residuals[0]
            y_next = current_state.east + residuals[1]
            z_next = current_state.depth + residuals[2]
            phi_next = current_state.roll + residuals[3]
            theta_next = np.clip(theta_act + residuals[4], -0.35, 0.35)
            psi_next = wrap_to_2pi(psi_act + residuals[5])
            speed_next = np.clip(current_state.speed + residuals[6], self.min_speed, self.max_speed)
        elif mode == "physics_only":
            x_next = phys_next[0]
            y_next = phys_next[1]
            z_next = phys_next[2]
            phi_next = phys_next[3]
            theta_next = theta_act
            psi_next = psi_act
            speed_next = np.clip(phys_next[6], self.min_speed, self.max_speed)
        else: # Hybrid
            res_ne = residuals if self.include_ne_residual else np.zeros(7)
            x_next = phys_next[0] + res_ne[0]
            y_next = phys_next[1] + res_ne[1]
            z_next = phys_next[2] + residuals[2]
            phi_next = phys_next[3] + residuals[3]
            theta_next = theta_act + residuals[4] * 0.5
            psi_next = wrap_to_2pi(psi_act + residuals[5] * 0.5)
            speed_next = np.clip(phys_next[6] + residuals[6], self.min_speed, self.max_speed)

        next_state = AUVState(
            north=float(x_next),
            east=float(y_next),
            depth=float(z_next),
            roll=float(phi_next),
            pitch=float(theta_next),
            heading=float(psi_next),
            speed=float(speed_next)
        )

        new_feat = next_state.to_feature_vector()
        if history_features is not None:
            new_history = np.vstack([history_features[1:], new_feat])
        else:
            new_history = np.tile(new_feat, (10, 1))

        return next_state, new_history

    def predict_rollout_batch(
        self,
        initial_state: AUVState,
        initial_history: np.ndarray,
        candidate_controls: np.ndarray,  # (K, H, 3)
        use_ai: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized simulation of K candidate control sequences across Horizon H.
        Returns:
            trajectories: (K, H+1, 7)
            final_histories: (K, 10, 6)
        """
        K, H, _ = candidate_controls.shape
        mode = self.dynamics_mode
        if use_ai is not None:
            mode = "hybrid" if use_ai else "physics_only"

        trajectories = np.zeros((K, H + 1, 7), dtype=np.float64)
        trajectories[:, 0, :] = initial_state.to_array()

        current_states = np.tile(initial_state.to_array(), (K, 1))
        histories = np.tile(initial_history, (K, 1, 1))

        for h in range(H):
            speed_cmds = candidate_controls[:, h, 0]
            pitch_cmds = candidate_controls[:, h, 1]
            heading_cmds = candidate_controls[:, h, 2]

            # Actuator response
            theta_acts = current_states[:, 4] + np.clip(
                pitch_cmds - current_states[:, 4],
                -self.max_pitch_rate,
                self.max_pitch_rate
            )
            heading_errs = wrap_angle(heading_cmds - current_states[:, 5])
            psi_acts = wrap_to_2pi(
                current_states[:, 5] + np.clip(heading_errs, -self.max_heading_rate, self.max_heading_rate)
            )
            u_targets = np.clip(speed_cmds, self.min_speed, self.max_speed)
            u_acts = current_states[:, 6] + np.clip(
                u_targets - current_states[:, 6],
                -self.max_speed_rate,
                self.max_speed_rate
            )

            act_states = current_states.copy()
            act_states[:, 4] = theta_acts
            act_states[:, 5] = psi_acts
            act_states[:, 6] = u_acts

            phys_next = self.physics.step_batch(act_states)

            if mode in ["hybrid", "gru_only"]:
                residuals = self.learned.predict_residual_batch(histories)
            else:
                residuals = np.zeros((K, 7), dtype=np.float64)

            next_states = np.zeros_like(current_states)
            if mode == "gru_only":
                next_states[:, 0] = current_states[:, 0] + residuals[:, 0]
                next_states[:, 1] = current_states[:, 1] + residuals[:, 1]
                next_states[:, 2] = current_states[:, 2] + residuals[:, 2]
                next_states[:, 3] = current_states[:, 3] + residuals[:, 3]
                next_states[:, 4] = np.clip(theta_acts + residuals[:, 4], -0.35, 0.35)
                next_states[:, 5] = wrap_to_2pi(psi_acts + residuals[:, 5])
                next_states[:, 6] = np.clip(current_states[:, 6] + residuals[:, 6], self.min_speed, self.max_speed)
            elif mode == "physics_only":
                next_states[:, 0] = phys_next[:, 0]
                next_states[:, 1] = phys_next[:, 1]
                next_states[:, 2] = phys_next[:, 2]
                next_states[:, 3] = phys_next[:, 3]
                next_states[:, 4] = theta_acts
                next_states[:, 5] = psi_acts
                next_states[:, 6] = np.clip(phys_next[:, 6], self.min_speed, self.max_speed)
            else: # Hybrid
                ne_res = residuals[:, :2] if self.include_ne_residual else np.zeros((K, 2))
                next_states[:, 0] = phys_next[:, 0] + ne_res[:, 0]
                next_states[:, 1] = phys_next[:, 1] + ne_res[:, 1]
                next_states[:, 2] = phys_next[:, 2] + residuals[:, 2]
                next_states[:, 3] = phys_next[:, 3] + residuals[:, 3]
                next_states[:, 4] = theta_acts + residuals[:, 4] * 0.5
                next_states[:, 5] = wrap_to_2pi(psi_acts + residuals[:, 5] * 0.5)
                next_states[:, 6] = np.clip(phys_next[:, 6] + residuals[:, 6], self.min_speed, self.max_speed)

            trajectories[:, h + 1, :] = next_states
            current_states = next_states

            # Slide history window
            new_feats = np.column_stack([
                next_states[:, 6],
                next_states[:, 3],
                next_states[:, 4],
                np.cos(next_states[:, 5]),
                np.sin(next_states[:, 5]),
                next_states[:, 2]
            ])
            histories = np.concatenate([histories[:, 1:, :], new_feats[:, np.newaxis, :]], axis=1)

        return trajectories, histories
