import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ensure Phase_2 path is loaded
current_dir = os.path.dirname(os.path.abspath(__file__))
phase2_dir = os.path.dirname(current_dir)
if phase2_dir not in sys.path:
    sys.path.insert(0, phase2_dir)

from src.simulation_runner import ClosedLoopSimulation

def run_point_to_point_return_simulation():
    print("=" * 72)
    print("  PHASE 2: CANONICAL ROUND-TRIP MISSION (A -> B -> A)")
    print("  Autosub6000 Receding-Horizon Neural MPC (Decoupled RK4 Plant)")
    print("=" * 72)

    scenario_cfg = {
        "name": "point_to_point_return",
        "start": [0.0, 0.0, 20.0],
        "goal": [100.0, 80.0, 40.0],
        "boundaries": {
            "north_min": 0.0, "north_max": 120.0,
            "east_min": 0.0, "east_max": 100.0,
            "depth_min": 10.0, "depth_max": 60.0
        },
        "obstacles": [],
        "current": [0.0, 0.0, 0.0]
    }

    sim = ClosedLoopSimulation(
        method="hybrid_mpc",
        scenario_config=scenario_cfg,
        seed=1000,
        dt_control=2.0,
        max_steps=180,
        goal_tolerance_primary=5.0,
        goal_tolerance_tight=2.0,
        horizon=10,
        num_candidates=20,
        iterations=2,
        use_warm_start=True,
        use_los_seed=True,
        use_kalman_filter=True,
        usbl_sigma=0.50,
        return_to_start=True
    )

    df_traj, metrics = sim.run()

    print(f"  Reached Point B:           {metrics['reached_b']} (Step {metrics['step_reached_b']}, t = {metrics['time_reached_b_sec']}s)")
    print(f"  Reached Point A (Return):  {metrics['reached_a']} (Step {metrics['step_reached_a']}, t = {metrics['time_reached_a_sec']}s)")
    print(f"  Final Distance to Point A: {metrics['final_distance_to_a_m']:.2f} m")
    print(f"  Forward Path Length:       {metrics['forward_path_length_m']:.2f} m")
    print(f"  Return Path Length:        {metrics['return_path_length_m']:.2f} m")
    print(f"  Total Realized Path:       {metrics['total_path_length_m']:.2f} m")
    print(f"  Path Efficiency:           {metrics['path_efficiency_pct']:.1f} %")
    print(f"  Boundary Violations:       {metrics['boundary_violations']}")
    print(f"  Physical Collisions:       {metrics['physical_collisions']}")
    print(f"  Mission Status:            {metrics['movement_phase']}")

    # 1. Save Trajectory CSV
    traj_dir = os.path.join(phase2_dir, "results", "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    traj_path = os.path.join(traj_dir, "A_to_B_to_A.csv")
    df_traj.to_csv(traj_path, index=False)
    print(f"  [+] Saved trajectory CSV:  {traj_path}")

    # 2. Save Metrics JSON
    metrics_dir = os.path.join(phase2_dir, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "A_to_B_to_A_metrics.json")
    
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (bool, np.bool_)):
            clean_metrics[k] = bool(v)
        elif isinstance(v, (float, np.floating)):
            clean_metrics[k] = 0.0 if np.isnan(v) or np.isinf(v) else float(v)
        elif isinstance(v, (int, np.integer)):
            clean_metrics[k] = int(v)
        elif isinstance(v, (list, tuple)):
            clean_metrics[k] = list(v)
        elif isinstance(v, str):
            clean_metrics[k] = str(v)
        elif v is None:
            clean_metrics[k] = None

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2)
    print(f"  [+] Saved metrics JSON:     {metrics_path}")

    # 3. Generate High-Quality Publication Graphs
    plots_dir = os.path.join(phase2_dir, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Split into Leg 1 and Leg 2
    step_b = metrics.get('step_reached_b', len(df_traj) // 2)
    df_leg1 = df_traj[df_traj['step'] <= step_b]
    df_leg2 = df_traj[df_traj['step'] >= step_b]

    b_cfg = scenario_cfg["boundaries"]
    start_pt = scenario_cfg["start"]
    goal_pt = scenario_cfg["goal"]

    # --- 2D Trajectory Plot ---
    plt.style.use('default')
    fig2d, ax2d = plt.subplots(figsize=(10, 8.5), dpi=300)
    fig2d.patch.set_facecolor('#ffffff')
    ax2d.set_facecolor('#f8fafc')

    # Geofence boundary
    bw = b_cfg["east_max"] - b_cfg["east_min"]
    bh = b_cfg["north_max"] - b_cfg["north_min"]
    rect = plt.Rectangle(
        (b_cfg["east_min"], b_cfg["north_min"]), bw, bh,
        fill=True, facecolor='#f1f5f9', edgecolor='#64748b',
        linestyle='--', lw=2.0, label='Geofence Boundary'
    )
    ax2d.add_patch(rect)

    # Reference straight line A <-> B
    ax2d.plot([start_pt[1], goal_pt[1]], [start_pt[0], goal_pt[0]],
              'k:', lw=1.2, alpha=0.4, label='Direct LOS Axis (78.7m)')

    # Leg 1: Outbound Path
    ax2d.plot(df_leg1['true_east'], df_leg1['true_north'], '-',
              color='#0284c7', lw=3.0, label=f'Outbound Leg 1: A ➔ B ({metrics["forward_path_length_m"]:.1f}m)', zorder=4)

    # Leg 2: Return Path
    ax2d.plot(df_leg2['true_east'], df_leg2['true_north'], '--',
              color='#f59e0b', lw=3.0, label=f'Return Leg 2: B ➔ A ({metrics["return_path_length_m"]:.1f}m)', zorder=5)

    # Markers
    ax2d.plot(start_pt[1], start_pt[0], 'o', color='#10b981', ms=12, label='Point A: Start & Return Origin', zorder=6)
    tol_circle_a = plt.Circle((start_pt[1], start_pt[0]), 5.0, fill=False, color='#10b981', linestyle='--', lw=1.8, label='Arrival Envelope (±5.0m)')
    ax2d.add_patch(tol_circle_a)

    ax2d.plot(goal_pt[1], goal_pt[0], '*', color='#ef4444', ms=16, label='Point B: Destination & Turnaround', zorder=6)
    tol_circle_b = plt.Circle((goal_pt[1], goal_pt[0]), 5.0, fill=False, color='#ef4444', linestyle='--', lw=1.8)
    ax2d.add_patch(tol_circle_b)

    # 180 Pivot Marker
    if len(df_leg1) > 0:
        p_b = df_leg1.iloc[-1]
        ax2d.annotate(
            "180° Dynamic Turn\n(t = {:.1f}s)".format(metrics['time_reached_b_sec']),
            xy=(p_b['true_east'], p_b['true_north']),
            xytext=(p_b['true_east'] + 8, p_b['true_north'] - 10),
            arrowprops=dict(facecolor='#f59e0b', shrink=0.08, width=1.5, headwidth=6),
            fontsize=9.5, fontweight='bold', color='#b45309',
            bbox=dict(boxstyle="round,pad=0.4", fc="#fef3c7", ec="#f59e0b", lw=1.2)
        )

    # Terminal settling marker
    final_pt = df_traj.iloc[-1]
    ax2d.plot(final_pt['true_east'], final_pt['true_north'], 's', color='#047857', ms=9, label=f'Terminal Docked ({metrics["final_distance_to_a_m"]:.2f}m to A)', zorder=7)

    ax2d.set_title("Autosub6000 Round-Trip Navigation (A ➔ B ➔ A)\nReceding-Horizon Neural MPC with Dynamic Return Replanning", fontsize=13, fontweight='bold', pad=12)
    ax2d.set_xlabel("East Position (meters)", fontsize=11, fontweight='semibold')
    ax2d.set_ylabel("North Position (meters)", fontsize=11, fontweight='semibold')
    ax2d.legend(loc='upper left', fontsize=9.5, framealpha=0.95, edgecolor='#cbd5e1')
    ax2d.grid(True, linestyle=':', alpha=0.6)
    ax2d.set_xlim(b_cfg["east_min"] - 8, b_cfg["east_max"] + 8)
    ax2d.set_ylim(b_cfg["north_min"] - 8, b_cfg["north_max"] + 8)
    ax2d.set_aspect('equal')

    plot_2d_path = os.path.join(plots_dir, "A_to_B_to_A_2D.png")
    fig2d.savefig(plot_2d_path, dpi=300, bbox_inches='tight')
    plt.close(fig2d)
    print(f"  [+] Saved 2D Trajectory Plot: {plot_2d_path}")

    # --- 3D Spatial Trajectory Plot ---
    fig3d = plt.figure(figsize=(11, 8.5), dpi=300)
    ax3d = fig3d.add_subplot(111, projection='3d')
    fig3d.patch.set_facecolor('#ffffff')
    ax3d.set_facecolor('#ffffff')

    # Wireframe Geofence Boundary Box
    eb = [b_cfg["east_min"], b_cfg["east_max"]]
    nb = [b_cfg["north_min"], b_cfg["north_max"]]
    db = [b_cfg["depth_min"], b_cfg["depth_max"]]

    for e in eb:
        for n in nb:
            ax3d.plot([e, e], [n, n], db, 'k--', alpha=0.25, lw=1.0)
    for e in eb:
        for d in db:
            ax3d.plot([e, e], nb, [d, d], 'k--', alpha=0.25, lw=1.0)
    for n in nb:
        for d in db:
            ax3d.plot(eb, [n, n], [d, d], 'k--', alpha=0.25, lw=1.0)

    # Leg 1: Outbound 3D Path
    ax3d.plot(df_leg1['true_east'], df_leg1['true_north'], df_leg1['true_depth'], '-',
              color='#0284c7', lw=3.2, label=f'Outbound Leg 1: A ➔ B ({metrics["forward_path_length_m"]:.1f}m)')

    # Leg 2: Return 3D Path
    ax3d.plot(df_leg2['true_east'], df_leg2['true_north'], df_leg2['true_depth'], '--',
              color='#f59e0b', lw=3.2, label=f'Return Leg 2: B ➔ A ({metrics["return_path_length_m"]:.1f}m)')

    # 3D Markers
    ax3d.scatter([start_pt[1]], [start_pt[0]], [start_pt[2]], color='#10b981', s=130, depthshade=False, label='START / RETURN ORIGIN A')
    ax3d.scatter([goal_pt[1]], [goal_pt[0]], [goal_pt[2]], color='#ef4444', marker='*', s=240, depthshade=False, label='DESTINATION / PIVOT B')
    ax3d.scatter([final_pt['true_east']], [final_pt['true_north']], [final_pt['true_depth']], color='#047857', marker='s', s=110, depthshade=False, label=f'FINAL DOCK ({metrics["final_distance_to_a_m"]:.2f}m error)')

    ax3d.invert_zaxis() # Depth positive downwards (0m surface, deeper downwards)
    ax3d.set_title("3D Flight Trajectory: Point A ➔ Point B ➔ Point A\nClosed-Loop Autosub6000 Receding-Horizon Control", fontsize=13, fontweight='bold', pad=18)
    ax3d.set_xlabel("East Position X (meters)", fontsize=11, labelpad=12, fontweight='bold')
    ax3d.set_ylabel("North Position Y (meters)", fontsize=11, labelpad=12, fontweight='bold')
    ax3d.set_zlabel("Depth Below Surface Z (meters)", fontsize=11, labelpad=16, fontweight='bold')
    ax3d.zaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.0f}m"))
    ax3d.legend(loc='upper right', fontsize=9.2, framealpha=0.95)
    ax3d.view_init(elev=26, azim=-60)

    plot_3d_path = os.path.join(plots_dir, "A_to_B_to_A_3D.png")
    fig3d.savefig(plot_3d_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig3d)
    print(f"  [+] Saved 3D Spatial Plot:    {plot_3d_path}")

    # --- Depth Profile Plot ---
    fig_dp, ax_dp = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig_dp.patch.set_facecolor('#ffffff')
    ax_dp.set_facecolor('#f8fafc')

    ax_dp.axhline(b_cfg["depth_min"], color='#94a3b8', linestyle='--', lw=1.5, label=f'Depth Boundary Min ({b_cfg["depth_min"]}m)')
    ax_dp.axhline(b_cfg["depth_max"], color='#94a3b8', linestyle='--', lw=1.5, label=f'Depth Boundary Max ({b_cfg["depth_max"]}m)')
    ax_dp.axhspan(b_cfg["depth_min"], b_cfg["depth_max"], color='#f1f5f9', alpha=0.5)

    # Target depths
    ax_dp.axhline(start_pt[2], color='#10b981', linestyle=':', lw=1.8, label=f'Point A Depth ({start_pt[2]}m)')
    ax_dp.axhline(goal_pt[2], color='#ef4444', linestyle=':', lw=1.8, label=f'Point B Depth ({goal_pt[2]}m)')

    ax_dp.plot(df_leg1['time_sec'], df_leg1['true_depth'], '-', color='#0284c7', lw=2.8, label='Leg 1 Dive Profile (A ➔ B)')
    ax_dp.plot(df_leg2['time_sec'], df_leg2['true_depth'], '--', color='#f59e0b', lw=2.8, label='Leg 2 Ascent Profile (B ➔ A)')

    ax_dp.plot(metrics['time_reached_b_sec'], df_traj[df_traj['step'] == step_b]['true_depth'].values[0], '*', color='#ef4444', ms=14, label='Reached B (Turnaround)')
    ax_dp.plot(df_traj.iloc[-1]['time_sec'], df_traj.iloc[-1]['true_depth'], 's', color='#047857', ms=10, label=f'Docked at A (t={df_traj.iloc[-1]["time_sec"]:.0f}s)')

    ax_dp.invert_yaxis()
    ax_dp.set_title("Vertical Depth Profile Across Round-Trip Mission (A ➔ B ➔ A)", fontsize=13, fontweight='bold', pad=10)
    ax_dp.set_xlabel("Mission Elapsed Time (seconds)", fontsize=11, fontweight='semibold')
    ax_dp.set_ylabel("Depth Below Surface (meters)", fontsize=11, fontweight='semibold')
    ax_dp.legend(loc='lower left', fontsize=9.5, framealpha=0.95)
    ax_dp.grid(True, linestyle=':', alpha=0.6)

    plot_depth_path = os.path.join(plots_dir, "A_to_B_to_A_depth.png")
    fig_dp.savefig(plot_depth_path, dpi=300, bbox_inches='tight')
    plt.close(fig_dp)
    print(f"  [+] Saved Depth Profile Plot: {plot_depth_path}")

    print("=" * 72)
    print("  ROUND-TRIP SIMULATION, METRICS, AND 2D/3D PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 72)

if __name__ == "__main__":
    run_point_to_point_return_simulation()
