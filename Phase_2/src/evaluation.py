import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
from src.goal import Goal
from src.constraints import Boundaries3D, CylindricalObstacle3D

class TrajectoryVisualizer:
    """Publication-Quality Visualizer for 2D, Depth, and 3D AUV Trajectories."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_2d_trajectory(
        self,
        df: pd.DataFrame,
        start_pos: np.ndarray,
        goal: Goal,
        boundaries: Boundaries3D,
        obstacles: Optional[List[CylindricalObstacle3D]] = None,
        title: str = "AUV Horizontal Trajectory (MPC Control)",
        filename: str = "trajectory_2D.png"
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))

        # 1. Plot Boundaries
        bw = boundaries.east_max - boundaries.east_min
        bh = boundaries.north_max - boundaries.north_min
        boundary_rect = plt.Rectangle(
            (boundaries.east_min, boundaries.north_min),
            bw, bh,
            fill=True, facecolor='#f1f5f9', edgecolor='#94a3b8',
            linestyle='--', lw=2.0, label='Operating Boundary'
        )
        ax.add_patch(boundary_rect)

        # 2. Plot Obstacles if any
        if obstacles:
            for i, obs in enumerate(obstacles):
                obs_circle = plt.Circle(
                    (obs.east, obs.north), obs.radius,
                    color='#ef4444', alpha=0.6, label='Obstacle' if i == 0 else ""
                )
                margin_circle = plt.Circle(
                    (obs.east, obs.north), obs.safe_radius,
                    fill=False, color='#dc2626', linestyle=':', lw=1.5, label='Safety Margin' if i == 0 else ""
                )
                ax.add_patch(obs_circle)
                ax.add_patch(margin_circle)

        # 3. Direct LOS reference line
        ax.plot([start_pos[1], goal.east], [start_pos[0], goal.north],
                'k:', lw=1.2, alpha=0.5, label='Direct A-B Line')

        # 4. Actual MPC Trajectory
        ax.plot(df["east"], df["north"], '-', color='#2563eb', lw=2.5, label='AUV MPC Trajectory', zorder=4)

        # 5. Start and Goal Markers
        ax.plot(start_pos[1], start_pos[0], 'go', ms=11, label='START A', zorder=5)
        ax.plot(goal.east, goal.north, 'r*', ms=16, label=f'GOAL B (Tol={goal.tolerance}m)', zorder=5)

        # Acceptance circle around goal
        goal_circle = plt.Circle((goal.east, goal.north), goal.tolerance, fill=False, color='#16a34a', linestyle='--', lw=1.5)
        ax.add_patch(goal_circle)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("East Position (meters)", fontsize=11)
        ax.set_ylabel("North Position (meters)", fontsize=11)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        # Add margin to axes
        ax.set_xlim(boundaries.east_min - 10, boundaries.east_max + 10)
        ax.set_ylim(boundaries.north_min - 10, boundaries.north_max + 10)
        ax.set_aspect('equal', adjustable='box')

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_depth_profile(
        self,
        df: pd.DataFrame,
        start_pos: np.ndarray,
        goal: Goal,
        boundaries: Boundaries3D,
        title: str = "AUV Vertical Depth Trajectory (MPC Control)",
        filename: str = "trajectory_depth.png"
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Depth boundaries
        ax.axhline(boundaries.depth_min, color='#94a3b8', linestyle='--', lw=1.8, label=f'Depth Min ({boundaries.depth_min}m)')
        ax.axhline(boundaries.depth_max, color='#94a3b8', linestyle='--', lw=1.8, label=f'Depth Max ({boundaries.depth_max}m)')
        ax.axhspan(boundaries.depth_min, boundaries.depth_max, color='#f1f5f9', alpha=0.5)

        # Goal target depth
        ax.axhline(goal.depth, color='#16a34a', linestyle=':', lw=2.0, label=f'Goal Depth ({goal.depth}m)')

        # Vehicle depth trajectory
        ax.plot(df["timestamp"], df["depth"], '-', color='#0284c7', lw=2.5, label='AUV Depth Profile')

        # Markers
        ax.plot(0.0, start_pos[2], 'go', ms=10, label='Start Depth')
        final_row = df.iloc[-1]
        ax.plot(final_row["timestamp"], final_row["depth"], 'r*', ms=14, label='Final Depth')

        ax.invert_yaxis()  # Standard oceanographic convention: depth increases downwards
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
        ax.set_ylabel("Depth Below Surface (meters)", fontsize=11)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_3d_trajectory(
        self,
        df: pd.DataFrame,
        start_pos: np.ndarray,
        goal: Goal,
        boundaries: Boundaries3D,
        obstacles: Optional[List[CylindricalObstacle3D]] = None,
        title: str = "3D AUV Goal-Directed Trajectory (MPC)",
        filename: str = "trajectory_3D.png"
    ) -> str:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111, projection='3d')

        # 1. 3D Trajectory line
        ax.plot(df["east"], df["north"], df["depth"], '-', color='#2563eb', lw=3.0, label='AUV 3D Flight Path')

        # 2. Start and Goal points
        ax.scatter([start_pos[1]], [start_pos[0]], [start_pos[2]], color='green', s=120, label='START A')
        ax.scatter([goal.east], [goal.north], [goal.depth], color='red', marker='*', s=220, label='GOAL B')

        # 3. Cylindrical obstacles
        if obstacles:
            for obs in obstacles:
                z_cyl = np.linspace(obs.depth_min, obs.depth_max, 20)
                theta_cyl = np.linspace(0, 2 * np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl)
                x_grid = obs.east + obs.radius * np.cos(theta_grid)
                y_grid = obs.north + obs.radius * np.sin(theta_grid)
                ax.plot_surface(x_grid, y_grid, z_grid, color='#ef4444', alpha=0.35)

        # 4. Boundary Box Outline
        e_bounds = [boundaries.east_min, boundaries.east_max]
        n_bounds = [boundaries.north_min, boundaries.north_max]
        d_bounds = [boundaries.depth_min, boundaries.depth_max]

        # Draw wireframe boundary
        for e in e_bounds:
            for n in n_bounds:
                ax.plot([e, e], [n, n], d_bounds, 'k--', alpha=0.3)
        for e in e_bounds:
            for d in d_bounds:
                ax.plot([e, e], n_bounds, [d, d], 'k--', alpha=0.3)
        for n in n_bounds:
            for d in d_bounds:
                ax.plot(e_bounds, [n, n], [d, d], 'k--', alpha=0.3)

        ax.invert_zaxis()  # Invert depth: surface at 0m, deeper downwards
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("East Position X (meters)", fontsize=10.5, labelpad=12, fontweight='bold')
        ax.set_ylabel("North Position Y (meters)", fontsize=10.5, labelpad=12, fontweight='bold')
        ax.set_zlabel("Depth Below Surface Z (meters)", fontsize=10.5, labelpad=16, fontweight='bold')
        ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.0f}m"))
        ax.legend(loc='upper right', fontsize=9.2, framealpha=0.95)
        ax.view_init(elev=26, azim=-60)

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        return save_path
