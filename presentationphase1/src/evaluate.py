import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any

class ModelEvaluator:
    """Computes comparison metrics and exports all trajectory and state time-series plots."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.preds_dir = os.path.join(self.results_dir, "predictions")
        os.makedirs(self.preds_dir, exist_ok=True)

        self.state_meta = [
            ("North Position", "m", 0, "north"),
            ("East Position", "m", 1, "east"),
            ("Depth", "m", 2, "depth"),
            ("Roll Angle", "rad", 3, "roll"),
            ("Pitch Angle", "rad", 4, "pitch"),
            ("Heading Angle", "rad", 5, "heading"),
            ("Surge Speed", "m/s", 6, "speed")
        ]

    def compute_comparison_table(self, raw_preds: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Computes MAE, RMSE, and R² across Model A (Physics), Model B (ML-Only), Model C (Hybrid)."""
        act = raw_preds["actual"]
        phy = raw_preds["physics"]
        ml = raw_preds["ml_only"]
        hyb = raw_preds["hybrid"]

        records = []
        for label, unit, idx, _ in self.state_meta:
            y_t = act[:, idx]
            y_p = phy[:, idx]
            y_m = ml[:, idx]
            y_h = hyb[:, idx]

            if "Heading" in label:
                e_p = (y_p - y_t + np.pi) % (2 * np.pi) - np.pi
                e_m = (y_m - y_t + np.pi) % (2 * np.pi) - np.pi
                e_h = (y_h - y_t + np.pi) % (2 * np.pi) - np.pi

                mae_p, rmse_p = float(np.mean(np.abs(e_p))), float(np.sqrt(np.mean(e_p**2)))
                r2_p = float(r2_score(y_t, y_t + e_p))

                mae_m, rmse_m = float(np.mean(np.abs(e_m))), float(np.sqrt(np.mean(e_m**2)))
                r2_m = float(r2_score(y_t, y_t + e_m))

                mae_h, rmse_h = float(np.mean(np.abs(e_h))), float(np.sqrt(np.mean(e_h**2)))
                r2_h = float(r2_score(y_t, y_t + e_h))
            else:
                mae_p, rmse_p = float(mean_absolute_error(y_t, y_p)), float(np.sqrt(mean_squared_error(y_t, y_p)))
                r2_p = float(r2_score(y_t, y_p))

                mae_m, rmse_m = float(mean_absolute_error(y_t, y_m)), float(np.sqrt(mean_squared_error(y_t, y_m)))
                r2_m = float(r2_score(y_t, y_m))

                mae_h, rmse_h = float(mean_absolute_error(y_t, y_h)), float(np.sqrt(mean_squared_error(y_t, y_h)))
                r2_h = float(r2_score(y_t, y_h))

            imp = ((rmse_p - rmse_h) / rmse_p) * 100.0

            records.append({
                "State Variable": f"{label} ({unit})",
                "Physics MAE": round(mae_p, 4),
                "Physics RMSE": round(rmse_p, 4),
                "Physics R²": round(r2_p, 4),

                "ML-Only MAE": round(mae_m, 4),
                "ML-Only RMSE": round(rmse_m, 4),
                "ML-Only R²": round(r2_m, 4),

                "Hybrid MAE": round(mae_h, 4),
                "Hybrid RMSE": round(rmse_h, 4),
                "Hybrid R²": round(r2_h, 4),

                "Hybrid Improvement (%)": round(imp, 2)
            })

        df_comp = pd.DataFrame(records)
        comp_csv = os.path.join(self.results_dir, "model_comparison.csv")
        df_comp.to_csv(comp_csv, index=False)
        print(f"Saved model comparison table to: {comp_csv}")
        return df_comp

    def plot_trajectory_comparison(self, df_preds: pd.DataFrame):
        """Generates results/trajectory_comparison.png clearly labeling ACTUAL, PHYSICS, ML, and PHYSICS + ML."""
        sub = df_preds.iloc[100:450].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(11, 8.5))

        # 1. Plot lines with proper z-order: ACTUAL in background, predictions on top
        ax.plot(sub["actual_east"], sub["actual_north"], '-', color='#111827', lw=3.0, alpha=0.5, label='ACTUAL (AURORA Test Telemetry)', zorder=1)
        ax.plot(sub["predicted_east_physics"], sub["predicted_north_physics"], '--', color='#d97706', lw=1.6, label='PHYSICS (Kinematics Baseline)', zorder=2)
        ax.plot(sub["predicted_east_ml"], sub["predicted_north_ml"], ':', color='#dc2626', lw=1.6, label='ML (Direct Learned Dynamics)', zorder=3)
        ax.plot(sub["predicted_east_hybrid"], sub["predicted_north_hybrid"], '-.', color='#2563eb', lw=1.8, label='PHYSICS + ML (Hybrid Residual)', zorder=4)

        ax.set_title("AUV Horizontal Trajectory Reconstruction (Unseen Test Data)", fontsize=13, fontweight='bold')
        ax.set_xlabel("East Position (meters)", fontsize=11)
        ax.set_ylabel("North Position (meters)", fontsize=11)
        ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
        ax.grid(True, alpha=0.3)

        # 2. Add Zoomed-In Inset focusing on the sharp hairpin turn to clearly show all 4 lines separating
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        ax_ins = inset_axes(ax, width="42%", height="38%", loc='center', bbox_to_anchor=(0.08, 0.05, 0.9, 0.9), bbox_transform=ax.transAxes)

        ax_ins.plot(sub["actual_east"], sub["actual_north"], '-', color='#111827', lw=3.5, alpha=0.5, label='Actual', zorder=1)
        ax_ins.plot(sub["predicted_east_physics"], sub["predicted_north_physics"], '--', color='#d97706', lw=2.0, label='Physics', zorder=2)
        ax_ins.plot(sub["predicted_east_ml"], sub["predicted_north_ml"], ':', color='#dc2626', lw=2.0, label='ML-Only', zorder=3)
        ax_ins.plot(sub["predicted_east_hybrid"], sub["predicted_north_hybrid"], '-.', color='#2563eb', lw=2.2, label='Hybrid', zorder=4)

        # Zoom limits on the sharp turning maneuver
        ax_ins.set_xlim(310, 335)
        ax_ins.set_ylim(-195, -145)
        ax_ins.set_title("Zoom: Turning Maneuver (Scale: ~25m)", fontsize=9, fontweight='bold', color='#1e3a8a')
        ax_ins.grid(True, linestyle=':', alpha=0.5)
        ax_ins.tick_params(labelsize=8)
        mark_inset(ax, ax_ins, loc1=1, loc2=3, fc="none", ec="0.5", lw=1.0, linestyle="--")

        out_path = os.path.join(self.results_dir, "trajectory_comparison.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved trajectory comparison plot with zoom inset to: {out_path}")

    def plot_individual_state_predictions(self, df_preds: pd.DataFrame):
        """Generates state-by-state Actual vs Predicted plots under results/predictions/."""
        sub = df_preds.iloc[150:350].reset_index(drop=True)
        t_axis = sub["timestamp"].values - sub["timestamp"].values[0]

        plots_info = [
            ("North Position (m)", "actual_north", "predicted_north_physics", "predicted_north_hybrid", "north_prediction.png", "North (m)"),
            ("East Position (m)", "actual_east", "predicted_east_physics", "predicted_east_hybrid", "east_prediction.png", "East (m)"),
            ("Depth (m)", "actual_depth", "predicted_depth_physics", "predicted_depth_hybrid", "depth_prediction.png", "Depth (m)"),
            ("Surge Speed (m/s)", "actual_surge_speed", "predicted_surge_speed_physics", "predicted_surge_speed_hybrid", "speed_prediction.png", "Speed (m/s)"),
            ("Pitch Angle (deg)", "actual_pitch", "predicted_pitch_physics", "predicted_pitch_hybrid", "pitch_prediction.png", "Pitch (deg)"),
            ("Heading Angle (deg)", "actual_heading", "predicted_heading_physics", "predicted_heading_hybrid", "heading_prediction.png", "Heading (deg)")
        ]

        for title, act_col, phy_col, hyb_col, fname, ylabel in plots_info:
            plt.figure(figsize=(9, 4.5))

            y_act = sub[act_col].values
            y_phy = sub[phy_col].values
            y_hyb = sub[hyb_col].values

            if "deg" in title:
                y_act = np.degrees(y_act)
                y_phy = np.degrees(y_phy)
                y_hyb = np.degrees(y_hyb)

            plt.plot(t_axis, y_act, 'k-', lw=2.2, label='ACTUAL Telemetry')
            plt.plot(t_axis, y_phy, '--', color='#ffb703', lw=1.8, label='PHYSICS Prediction')
            plt.plot(t_axis, y_hyb, '-.', color='#0077b6', lw=2.0, label='PHYSICS + ML Hybrid')

            if "Depth" in title:
                plt.gca().invert_yaxis()

            plt.title(f"State Dynamic Prediction: {title}", fontsize=12, fontweight='bold')
            plt.xlabel("Time Elapsed (seconds)", fontsize=10)
            plt.ylabel(ylabel, fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(self.preds_dir, fname)
            plt.savefig(save_path, dpi=300)
            plt.close()

        print(f"Saved 6 individual state prediction plots to: {self.preds_dir}")

    def plot_multistep_degradation(self, multistep_res: Dict[str, Any]):
        """Generates results/multistep_degradation.png comparing recursive compounding error."""
        plt.figure(figsize=(8, 5))
        horizons_sec = multistep_res["horizon_seconds"]
        plt.plot(horizons_sec, multistep_res["physics_position_rmse"], 'o--', color='#ffb703', lw=2.0, label='PHYSICS-Only Rollout')
        plt.plot(horizons_sec, multistep_res["hybrid_position_rmse"], 's-', color='#0077b6', lw=2.5, label='PHYSICS + ML Hybrid Rollout')

        plt.title("Multi-Step Prediction Error Propagation (Position RMSE)", fontsize=12, fontweight='bold')
        plt.xlabel("Prediction Horizon (seconds)", fontsize=10)
        plt.ylabel("Position RMSE (meters)", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(self.results_dir, "multistep_degradation.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved multi-step degradation plot to: {out_path}")
