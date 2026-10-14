import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from src.goal import Goal
from src.constraints import Boundaries3D

class TrajectoryManager:
    """Manages trajectory logging, metrics computation, and serialization."""

    @staticmethod
    def save_trajectory(df: pd.DataFrame, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Saved trajectory log to: {file_path}")

    @staticmethod
    def compute_metrics(
        df: pd.DataFrame,
        start_pos: np.ndarray,
        goal: Goal,
        boundaries: Boundaries3D
    ) -> Dict[str, Any]:
        """Calculates exact performance metrics from the simulated telemetry."""
        if len(df) == 0:
            return {}

        final_row = df.iloc[-1]
        final_pos = np.array([final_row["north"], final_row["east"], final_row["depth"]])
        final_dist = float(goal.distance_to(final_pos))
        goal_reached = bool(final_dist <= goal.tolerance)

        # Path Length
        coords = df[["north", "east", "depth"]].values
        deltas = np.diff(coords, axis=0)
        step_lengths = np.linalg.norm(deltas, axis=1)
        total_path_length = float(np.sum(step_lengths))

        # Max distance from goal
        all_dists = df["distance_to_goal"].values
        max_dist = float(np.max(all_dists))

        # Time to goal
        time_to_goal = float(final_row["timestamp"]) if goal_reached else None

        # Violations
        boundary_violations = int(np.sum(~df["in_boundaries"]))
        obstacle_violations = int(np.sum(df["obstacle_collision"])) if "obstacle_collision" in df else 0
        total_constraint_violations = int(np.sum(df["constraint_status"] == "VIOLATION"))

        # Control Effort: sum of squared differences between consecutive control inputs
        ctrls = df[["speed_cmd", "pitch_cmd", "heading_cmd"]].values
        if len(ctrls) > 1:
            diffs = np.diff(ctrls, axis=0)
            control_effort = float(np.sum(diffs**2))
        else:
            control_effort = 0.0

        metrics = {
            "start_position": [float(x) for x in start_pos],
            "goal_position": [float(x) for x in goal.to_array()],
            "goal_tolerance_m": float(goal.tolerance),
            "final_position": [float(x) for x in final_pos],
            "final_distance_to_goal_m": final_dist,
            "goal_reached": goal_reached,
            "time_to_goal_s": time_to_goal,
            "total_simulation_steps": int(len(df)),
            "total_simulation_time_s": float(final_row["timestamp"]),
            "total_path_length_m": total_path_length,
            "maximum_distance_to_goal_m": max_dist,
            "boundary_violations": boundary_violations,
            "obstacle_violations": obstacle_violations,
            "total_constraint_violations": total_constraint_violations,
            "total_control_effort": control_effort
        }
        return metrics

    @staticmethod
    def save_metrics(metrics: Dict[str, Any], file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {file_path}")
