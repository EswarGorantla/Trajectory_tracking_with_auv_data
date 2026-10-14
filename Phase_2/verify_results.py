import os
import sys
import json
import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p2_dir = os.path.join(root_dir, "Phase_2")
sys.path.insert(0, p2_dir)

from src.scenarios import get_canonical_scenarios

def verify_all_results():
    print("============================================================")
    print("INDEPENDENT METRIC RE-COMPUTATION & AUDIT VERIFICATION")
    print("============================================================")
    print("Recomputing every headline metric directly from raw CSV flight logs...")

    results_dir = os.path.join(p2_dir, "results_v2")
    trajs_dir = os.path.join(results_dir, "trajectories")
    master_csv = os.path.join(results_dir, "master_summary.csv")

    if not os.path.exists(master_csv):
        raise FileNotFoundError(f"Master summary CSV not found at: {master_csv}")

    df_master = pd.read_csv(master_csv)
    scenarios = get_canonical_scenarios()

    total_checked = 0
    mismatches = []

    for idx, row in df_master.iterrows():
        scen_name = row["scenario"]
        method = row["method"]
        seed = int(row["seed"])

        traj_file = os.path.join(trajs_dir, f"{scen_name}_{method}_seed{seed}.csv")
        if not os.path.exists(traj_file):
            mismatches.append(f"Missing trajectory file: {traj_file}")
            continue

        df_raw = pd.read_csv(traj_file)
        scen = scenarios[scen_name]
        start_pos = np.array(scen["start"])
        goal_pos = np.array(scen["goal"])
        straight_dist = float(np.linalg.norm(goal_pos - start_pos))

        # 1. Independent Recomputations
        # Path length
        coords = df_raw[["true_north", "true_east", "true_depth"]].values
        step_diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        recomputed_path_len = float(np.sum(step_diffs))

        # Efficiency
        recomputed_eff = (straight_dist / recomputed_path_len * 100.0) if recomputed_path_len > 0 else 0.0

        # Final Distance
        recomputed_final_dist = float(np.linalg.norm(coords[-1] - goal_pos))

        # Success
        dists = np.linalg.norm(coords - goal_pos, axis=1)
        recomputed_success_5m = bool(np.any(dists <= 5.0))
        recomputed_success_2m = bool(np.any(dists <= 2.0))

        # Boundary violations
        recomputed_bound_viol = int(np.sum(df_raw["boundary_violation_m"] > 0))

        # Physical collisions
        recomputed_collisions = int(np.sum(df_raw["collision"] == True))

        # Compare against master CSV
        # 1. Final distance check
        if abs(recomputed_final_dist - row["final_distance_m"]) > 1e-3:
            mismatches.append(f"[{scen_name} {method} {seed}] Final dist mismatch: recomputed={recomputed_final_dist:.4f}, recorded={row['final_distance_m']:.4f}")

        # 2. Path length check
        if abs(recomputed_path_len - row["total_path_length_m"]) > 1e-2:
            mismatches.append(f"[{scen_name} {method} {seed}] Path length mismatch: recomputed={recomputed_path_len:.4f}, recorded={row['total_path_length_m']:.4f}")

        # 3. Efficiency check
        if abs(recomputed_eff - row["path_efficiency_pct"]) > 1e-2:
            mismatches.append(f"[{scen_name} {method} {seed}] Efficiency mismatch: recomputed={recomputed_eff:.4f}, recorded={row['path_efficiency_pct']:.4f}")

        # 4. Success 5m check
        if recomputed_success_5m != bool(row["success_5m"]):
            mismatches.append(f"[{scen_name} {method} {seed}] Success 5m mismatch: recomputed={recomputed_success_5m}, recorded={row['success_5m']}")

        # 5. Collisions check
        if recomputed_collisions != int(row["physical_collisions"]):
            mismatches.append(f"[{scen_name} {method} {seed}] Collisions mismatch: recomputed={recomputed_collisions}, recorded={row['physical_collisions']}")

        total_checked += 1

    print(f"\nAudit complete across {total_checked} trajectory flight logs.")
    if len(mismatches) > 0:
        print(f"FAILED! Found {len(mismatches)} mismatches:")
        for m in mismatches[:10]:
            print("  ", m)
        raise AssertionError("Independent recomputation audit failed!")
    else:
        print("PERFECT MATCH! All headline metrics independently verified with zero discrepancies.")

if __name__ == "__main__":
    verify_all_results()
