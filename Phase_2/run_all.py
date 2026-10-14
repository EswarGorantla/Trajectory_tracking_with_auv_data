import os
import sys
import time
import json
import argparse
import hashlib
import numpy as np
import pandas as pd

# Paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p1_dir = os.path.join(root_dir, "presentationphase1")
p2_dir = os.path.join(root_dir, "Phase_2")
sys.path.insert(0, p1_dir)
sys.path.insert(0, p2_dir)

from src.scenarios import get_canonical_scenarios
from src.simulation_runner import ClosedLoopSimulation
from src.statistics import (
    compute_bootstrap_ci,
    compute_wilson_interval,
    compute_paired_comparison,
    apply_holm_bonferroni
)
from src.plotting import (
    plot_2d_topdown,
    plot_trajectory_comparison_2d,
    plot_flight_telemetry,
    plot_method_boxplots
)

# Frozen Phase 1 Hashes
FROZEN_HASHES = {
    os.path.join(p1_dir, "models", "best_dynamics_model.pt"): "9BF7305EDD14FB2BA61FD9A293E687D604F1DD9B76592D835A1016E242AA9FF2",
    os.path.join(p1_dir, "models", "scaler.pkl"): "AE84FFFF1E0FA8F7CE21B9A3E6C99FC117BAA712E16F60551536976F71A14670",
    os.path.join(p1_dir, "models", "model_config.json"): "569A9BC2E694D644C59DA854B64C206D48BFF748E6C401A79A236D000F331C41"
}

def verify_phase1_integrity():
    for fpath, expected in FROZEN_HASHES.items():
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        actual = h.hexdigest().upper()
        if actual != expected:
            raise RuntimeError(f"FATAL: Phase 1 artifact {fpath} was modified! Hash={actual}, Expected={expected}")
    print("[SECURITY] Phase 1 artifacts SHA-256 integrity verified (zero modification).")

def main():
    parser = argparse.ArgumentParser(description="Run research-grade Phase 2 AUV MPC benchmarks.")
    parser.add_argument("--quick", action="store_true", help="Run quick verification with few seeds (minutes).")
    parser.add_argument("--full", action="store_true", help="Run full Monte Carlo evaluation (30 seeds).")
    parser.add_argument("--seeds", type=int, default=None, help="Explicit number of seeds to run.")
    args = parser.parse_args()

    num_seeds = 3
    if args.full:
        num_seeds = 30
    elif args.seeds is not None:
        num_seeds = args.seeds
    elif args.quick:
        num_seeds = 3

    print("============================================================")
    print("PHASE 2 AUV MODEL PREDICTIVE CONTROL: RESEARCH BENCHMARK")
    print(f"Mode: {'FULL (30 seeds)' if num_seeds >= 30 else f'QUICK ({num_seeds} seeds)'}")
    print("============================================================")

    # 1. Assert Phase 1 Immutability
    verify_phase1_integrity()

    # Seed pool: evaluation seeds start at 1000
    seed_pool = [1000 + i for i in range(num_seeds)]

    # Setup directories
    results_dir = os.path.join(p2_dir, "results_v2")
    trajs_dir = os.path.join(results_dir, "trajectories")
    metrics_dir = os.path.join(results_dir, "metrics")
    plots_dir = os.path.join(results_dir, "plots")
    ablations_dir = os.path.join(results_dir, "ablations")

    for d in [results_dir, trajs_dir, metrics_dir, plots_dir, ablations_dir]:
        os.makedirs(d, exist_ok=True)

    scenarios = get_canonical_scenarios()
    methods = [
        "hybrid_mpc",       # Proposed
        "physics_mpc",      # Kinematics-only ablation
        "gru_mpc",          # Data-driven GRU ablation
        "potential_field",  # Strong reactive baseline (APF)
        "reactive_los"      # Basic reactive baseline
    ]

    all_metric_records = []
    representative_trajectories = {}

    start_total_time = time.time()

    # 2. RUN MAIN BENCHMARK MATRIX
    print("\n--- 1. Executing Main Benchmark Matrix ---")
    for scen_name, scen_cfg in scenarios.items():
        print(f"\n>> Scenario: {scen_name} ({scen_cfg['desc']})")
        for method in methods:
            for seed in seed_pool:
                sim = ClosedLoopSimulation(
                    method=method,
                    scenario_config=scen_cfg,
                    seed=seed,
                    dt_control=2.0,
                    max_steps=120,
                    randomize_plant_mismatch=True
                )
                df_traj, metrics = sim.run()
                all_metric_records.append(metrics)

                # Save trajectory CSV
                traj_fname = f"{scen_name}_{method}_seed{seed}.csv"
                df_traj.to_csv(os.path.join(trajs_dir, traj_fname), index=False)

                # Store first seed trajectory for visualization
                if seed == seed_pool[0]:
                    key = f"{scen_name}_{method}"
                    representative_trajectories[key] = df_traj

            # Summary for this (scenario, method) cell
            cell_metrics = [m for m in all_metric_records if m["scenario"] == scen_name and m["method"] == method]
            success_count = sum(1 for m in cell_metrics if m["success_5m"])
            collisions = sum(m["physical_collisions"] for m in cell_metrics)
            avg_time = np.mean([m["time_to_goal_sec"] for m in cell_metrics])
            avg_eff = np.mean([m["path_efficiency_pct"] for m in cell_metrics])
            print(f"  {method:<16} | Success: {success_count}/{len(cell_metrics)} | Collisions: {collisions} | Mean Time: {avg_time:.1f}s | Path Eff: {avg_eff:.1f}%")

    # 3. RUN ABLATIONS SUITE (on obstacle_avoidance scenario)
    print("\n--- 2. Executing Ablations Suite ---")
    base_scen = scenarios["obstacle_avoidance"]
    ablation_records = []

    # A. Horizon Ablations: H in {5, 10, 15}
    for h in [5, 10, 15]:
        for seed in seed_pool[:min(num_seeds, 5)]:
            sim = ClosedLoopSimulation("hybrid_mpc", base_scen, seed=seed, horizon=h)
            _, m = sim.run()
            m["ablation_type"] = "horizon"
            m["ablation_val"] = h
            ablation_records.append(m)

    # B. Candidates Ablations: K in {30, 70, 150}
    for k in [30, 70, 150]:
        for seed in seed_pool[:min(num_seeds, 5)]:
            sim = ClosedLoopSimulation("hybrid_mpc", base_scen, seed=seed, num_candidates=k)
            _, m = sim.run()
            m["ablation_type"] = "candidates"
            m["ablation_val"] = k
            ablation_records.append(m)

    # C. Warm-Start Ablation
    for ws in [True, False]:
        for seed in seed_pool[:min(num_seeds, 5)]:
            sim = ClosedLoopSimulation("hybrid_mpc", base_scen, seed=seed, use_warm_start=ws)
            _, m = sim.run()
            m["ablation_type"] = "warm_start"
            m["ablation_val"] = str(ws)
            ablation_records.append(m)

    # D. Kalman Filter vs Raw Sensor Ablation
    for kf in [True, False]:
        for seed in seed_pool[:min(num_seeds, 5)]:
            sim = ClosedLoopSimulation("hybrid_mpc", base_scen, seed=seed, use_kalman_filter=kf)
            _, m = sim.run()
            m["ablation_type"] = "kalman_filter"
            m["ablation_val"] = str(kf)
            ablation_records.append(m)

    # E. Ocean Current Magnitude Sweep: 0.0, 0.1, 0.2, 0.4 m/s
    for c_mag in [0.0, 0.10, 0.20, 0.40]:
        c_vec = np.array([c_mag * 0.707, -c_mag * 0.707, 0.0])
        for seed in seed_pool[:min(num_seeds, 5)]:
            sim = ClosedLoopSimulation("hybrid_mpc", base_scen, seed=seed, current_velocity=c_vec)
            _, m = sim.run()
            m["ablation_type"] = "current_mag"
            m["ablation_val"] = c_mag
            ablation_records.append(m)

    # Save Ablation CSV
    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(os.path.join(ablations_dir, "ablation_results.csv"), index=False)
    print(f"Saved ablation results to: {os.path.join(ablations_dir, 'ablation_results.csv')}")

    # 4. CONSOLIDATE MASTER SUMMARY CSV & METRICS JSON
    df_master = pd.DataFrame(all_metric_records)
    master_csv_path = os.path.join(results_dir, "master_summary.csv")
    df_master.to_csv(master_csv_path, index=False)
    print(f"\nSaved master summary CSV to: {master_csv_path}")

    # Aggregate statistical table with Wilson intervals & Bootstrap CIs
    agg_table = []
    for scen_name in scenarios.keys():
        scen_df = df_master[df_master["scenario"] == scen_name]
        for method in methods:
            m_df = scen_df[scen_df["method"] == method]
            if len(m_df) == 0:
                continue

            n_ep = len(m_df)
            succ_5m = int(m_df["success_5m"].sum())
            succ_rate = succ_5m / n_ep
            w_low, w_high = compute_wilson_interval(succ_5m, n_ep)

            times = m_df["time_to_goal_sec"].values
            t_low, t_high = compute_bootstrap_ci(times)

            effs = m_df["path_efficiency_pct"].values
            eff_low, eff_high = compute_bootstrap_ci(effs)

            cols = int(m_df["physical_collisions"].sum())
            buffs = int(m_df["buffer_violations"].sum())

            agg_table.append({
                "scenario": scen_name,
                "method": method,
                "episodes": n_ep,
                "success_5m_count": succ_5m,
                "success_rate_pct": succ_rate * 100.0,
                "wilson_ci_95": [w_low * 100.0, w_high * 100.0],
                "time_to_goal_mean": float(np.mean(times)),
                "time_ci_95": [t_low, t_high],
                "path_efficiency_mean": float(np.mean(effs)),
                "path_eff_ci_95": [eff_low, eff_high],
                "total_physical_collisions": cols,
                "total_buffer_violations": buffs,
                "solve_time_mean_sec": float(m_df["solve_time_mean_sec"].mean()),
                "solve_time_p95_sec": float(m_df["solve_time_p95_sec"].mean())
            })

    agg_json_path = os.path.join(metrics_dir, "benchmark_aggregated_metrics.json")
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump(agg_table, f, indent=2)
    print(f"Saved aggregated metrics JSON to: {agg_json_path}")

    # 5. GENERATE PUBLICATION-GRADE PLOTS
    print("\n--- 3. Generating Publication-Quality Figures ---")
    # A. 2D Top-down for all scenarios (Hybrid MPC)
    for scen_name, scen_cfg in scenarios.items():
        k = f"{scen_name}_hybrid_mpc"
        if k in representative_trajectories:
            out_p = os.path.join(plots_dir, f"{scen_name}_2d_hybrid.png")
            plot_2d_topdown(representative_trajectories[k], scen_cfg, f"AUV Trajectory (Hybrid MPC) - {scen_name}", out_p, "hybrid_mpc")

    # B. Multi-controller comparison on Obstacle Avoidance
    obs_trajs = {m: representative_trajectories[f"obstacle_avoidance_{m}"] for m in methods if f"obstacle_avoidance_{m}" in representative_trajectories}
    plot_trajectory_comparison_2d(obs_trajs, scenarios["obstacle_avoidance"], "Obstacle Avoidance: Controller Comparison", os.path.join(plots_dir, "obstacle_comparison_2d.png"))

    # C. Flight Telemetry Plot (Hybrid MPC on obstacle_avoidance)
    if "obstacle_avoidance_hybrid_mpc" in representative_trajectories:
        plot_flight_telemetry(representative_trajectories["obstacle_avoidance_hybrid_mpc"], "AUV Flight Telemetry & Envelope (Hybrid MPC)", os.path.join(plots_dir, "telemetry_profile_obstacle.png"))

    # D. Boxplots of Time-To-Goal and Path Efficiency
    plot_method_boxplots(df_master[df_master["scenario"] == "obstacle_avoidance"], "time_to_goal_sec", "Time to Goal (s)", "Time to Goal Distribution (Obstacle Avoidance)", os.path.join(plots_dir, "boxplot_time_obstacle.png"))
    plot_method_boxplots(df_master[df_master["scenario"] == "obstacle_avoidance"], "path_efficiency_pct", "Path Efficiency (%)", "Path Efficiency Distribution (Obstacle Avoidance)", os.path.join(plots_dir, "boxplot_eff_obstacle.png"))

    # E. Worst-Case Episode Gallery (Worst 3 runs by distance error)
    worst_df = df_master.sort_values(by="final_distance_m", ascending=False).head(3)
    worst_summary = worst_df[["scenario", "method", "seed", "final_distance_m", "physical_collisions", "boundary_violations"]].to_dict(orient="records")
    with open(os.path.join(metrics_dir, "worst_case_episodes.json"), "w", encoding="utf-8") as f:
        json.dump(worst_summary, f, indent=2)

    # 6. Final Security Assertion
    verify_phase1_integrity()

    elapsed = time.time() - start_total_time
    print(f"\n============================================================")
    print(f"BENCHMARK COMPLETE! Total Elapsed Time: {elapsed:.1f} seconds")
    print(f"Results Directory: {results_dir}")
    print("============================================================\n")

if __name__ == "__main__":
    main()
