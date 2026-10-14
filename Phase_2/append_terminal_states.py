import os
import sys
import numpy as np
import pandas as pd

p2_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, p2_dir)

from src.true_plant import TrueAUVPlant, PlantParameters, CylindricalObstacle
from src.state import AUVState, AUVControl
from src.scenarios import get_canonical_scenarios

def sync_terminal_states():
    trajs_dir = os.path.join(p2_dir, "results_v2", "trajectories")
    master_csv = os.path.join(p2_dir, "results_v2", "master_summary.csv")
    
    if not os.path.exists(master_csv):
        print("master_summary.csv not found!")
        return

    df_master = pd.read_csv(master_csv)
    scenarios = get_canonical_scenarios()

    print(f"Replaying plant for {len(df_master)} trajectories to sync terminal state...")

    for idx, row in df_master.iterrows():
        scen_name = row["scenario"]
        method = row["method"]
        seed = int(row["seed"])

        traj_file = os.path.join(trajs_dir, f"{scen_name}_{method}_seed{seed}.csv")
        if not os.path.exists(traj_file):
            continue

        df_raw = pd.read_csv(traj_file)
        scen = scenarios[scen_name]
        start_pos = np.array(scen["start"])
        goal_pos = np.array(scen["goal"])
        start_heading = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0]) % (2 * np.pi)

        # Check if already has terminal row (NaN cmd_speed)
        if pd.isna(df_raw.iloc[-1]["cmd_speed"]):
            continue

        # Replay plant with exact same seed
        rng = np.random.default_rng(seed)
        delta = float(rng.uniform(-0.25, 0.25))
        plant_params = PlantParameters(kd_mismatch=delta)
        obs_objs = [
            CylindricalObstacle(
                north=o["north"],
                east=o["east"],
                radius=o["radius"],
                margin=o.get("margin", 4.0)
            )
            for o in scen.get("obstacles", [])
        ]
        plant = TrueAUVPlant(
            params=plant_params,
            current_velocity=scen["current"],
            obstacles=obs_objs,
            seed=seed
        )
        init_state = AUVState(
            north=start_pos[0],
            east=start_pos[1],
            depth=start_pos[2],
            roll=0.0,
            pitch=0.0,
            heading=start_heading,
            speed=1.20
        )
        plant.reset(init_state)

        for _, r in df_raw.iterrows():
            cmd = AUVControl(speed_cmd=r["cmd_speed"], pitch_cmd=r["cmd_pitch"], heading_cmd=r["cmd_heading"])
            plant.step(cmd)

        final_s = plant.state
        final_dist = float(np.linalg.norm(final_s[:3] - goal_pos))

        term_row = {
            "step": len(df_raw),
            "time_sec": len(df_raw) * 2.0,
            "true_north": final_s[0],
            "true_east": final_s[1],
            "true_depth": final_s[2],
            "true_roll": final_s[3],
            "true_pitch": final_s[4],
            "true_heading": final_s[5],
            "true_speed": final_s[6],
            "meas_north": final_s[0],
            "meas_east": final_s[1],
            "meas_depth": final_s[2],
            "meas_heading": final_s[5],
            "meas_speed": final_s[6],
            "cmd_speed": np.nan,
            "cmd_pitch": np.nan,
            "cmd_heading": np.nan,
            "dist_to_goal": final_dist,
            "min_obstacle_dist": float(plant.min_obstacle_dist),
            "collision": False,
            "buffer_violation": False,
            "boundary_violation_m": 0.0,
            "is_in_bounds": True,
            "ood_flag": False,
            "solve_time_sec": 0.0
        }
        for k_cost in ["j_goal", "j_tracking", "j_control", "j_smooth", "j_constraint", "total_cost"]:
            term_row[f"cost_{k_cost}"] = 0.0

        df_updated = pd.concat([df_raw, pd.DataFrame([term_row])], ignore_index=True)
        df_updated.to_csv(traj_file, index=False)

    print("All trajectories successfully synchronized with terminal states!")

if __name__ == "__main__":
    sync_terminal_states()
