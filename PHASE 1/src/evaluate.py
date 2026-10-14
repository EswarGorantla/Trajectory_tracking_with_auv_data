import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List

class ModelEvaluator:
    """Computes regression metrics and exports publication-ready comparison charts."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.preds_dir = os.path.join(self.results_dir, "predictions")

        for d in [self.metrics_dir, self.plots_dir, self.preds_dir]:
            os.makedirs(d, exist_ok=True)

        self.state_labels = [
            ("North Position", "m", 0),
            ("East Position", "m", 1),
            ("Depth", "m", 2),
            ("Roll Angle", "rad", 3),
            ("Pitch Angle", "rad", 4),
            ("Heading Angle", "rad", 5),
            ("Surge Speed", "m/s", 6)
        ]

    def compute_metrics_table(self, raw_preds: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Computes MAE, RMSE, and R² for every state variable across all 3 models.
        """
        act = raw_preds["actual"]
        phy = raw_preds["physics"]
        ml = raw_preds["ml_only"]
        hyb = raw_preds["hybrid"]

        records = []
        for name, unit, idx in self.state_labels:
            y_true = act[:, idx]
            y_phy = phy[:, idx]
            y_ml = ml[:, idx]
            y_hyb = hyb[:, idx]

            # Angular wrapping for heading metric calculations
            if "Heading" in name:
                y_phy_err = (y_phy - y_true + np.pi) % (2 * np.pi) - np.pi
                y_ml_err = (y_ml - y_true + np.pi) % (2 * np.pi) - np.pi
                y_hyb_err = (y_hyb - y_true + np.pi) % (2 * np.pi) - np.pi

                mae_phy = float(np.mean(np.abs(y_phy_err)))
                rmse_phy = float(np.sqrt(np.mean(y_phy_err**2)))
                r2_phy = float(r2_score(y_true, y_true + y_phy_err))

                mae_ml = float(np.mean(np.abs(y_ml_err)))
                rmse_ml = float(np.sqrt(np.mean(y_ml_err**2)))
                r2_ml = float(r2_score(y_true, y_true + y_ml_err))

                mae_hyb = float(np.mean(np.abs(y_hyb_err)))
                rmse_hyb = float(np.sqrt(np.mean(y_hyb_err**2)))
                r2_hyb = float(r2_score(y_true, y_true + y_hyb_err))
            else:
                mae_phy = float(mean_absolute_error(y_true, y_phy))
                rmse_phy = float(np.sqrt(mean_squared_error(y_true, y_phy)))
                r2_phy = float(r2_score(y_true, y_phy))

                mae_ml = float(mean_absolute_error(y_true, y_ml))
                rmse_ml = float(np.sqrt(mean_squared_error(y_true, y_ml)))
                r2_ml = float(r2_score(y_true, y_ml))

                mae_hyb = float(mean_absolute_error(y_true, y_hyb))
                rmse_hyb = float(np.sqrt(mean_squared_error(y_true, y_hyb)))
                r2_hyb = float(r2_score(y_true, y_hyb))

            records.append({
                "State Variable": f"{name} ({unit})",
                "Physics MAE": round(mae_phy, 4),
                "Physics RMSE": round(rmse_phy, 4),
                "Physics R²": round(r2_phy, 4),

                "ML-Only MAE": round(mae_ml, 4),
                "ML-Only RMSE": round(rmse_ml, 4),
                "ML-Only R²": round(r2_ml, 4),

                "Hybrid MAE": round(mae_hyb, 4),
                "Hybrid RMSE": round(rmse_hyb, 4),
                "Hybrid R²": round(r2_hyb, 4),

                "RMSE Improvement (%)": round(((rmse_phy - rmse_hyb) / rmse_phy) * 100, 2)
            })

        df_metrics = pd.DataFrame(records)
        csv_path = os.path.join(self.metrics_dir, "model_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"Saved model evaluation metrics to {csv_path}")
        return df_metrics

    def generate_all_plots(self, df_preds: pd.DataFrame, df_metrics: pd.DataFrame, multistep_res: Dict[str, Any]):
        """Generates trajectory plots, state time-series, and metric comparison bar charts."""
        # 1. 2D Trajectory Plot (First 300 test steps for visual clarity)
        sub_df = df_preds.iloc[100:400].reset_index(drop=True)
        plt.figure(figsize=(10, 8))
        plt.plot(sub_df["Actual_East"], sub_df["Actual_North"], 'k-', label='Actual AURORA Telemetry', lw=2.5, zorder=3)
        plt.plot(sub_df["Physics_East"], sub_df["Physics_North"], '--', color='#ffb703', label='Physics-Only Model', lw=1.8, zorder=2)
        plt.plot(sub_df["ML_Only_East"], sub_df["ML_Only_North"], ':', color='#e63946', label='ML-Only Dynamics', lw=1.8, zorder=2)
        plt.plot(sub_df["Hybrid_East"], sub_df["Hybrid_North"], '-.', color='#0077b6', label='Physics + ML Hybrid Dynamics', lw=2.0, zorder=4)
        
        plt.title("AUV Horizontal Trajectory: Actual vs Predicted (Unseen Test Data)", fontsize=13, fontweight='bold')
        plt.xlabel("East Position (m)", fontsize=11)
        plt.ylabel("North Position (m)", fontsize=11)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        traj_plot_path = os.path.join(self.plots_dir, "trajectory_actual_vs_predicted.png")
        plt.savefig(traj_plot_path, dpi=300)
        plt.close()
        print(f"Saved trajectory plot to {traj_plot_path}")

        # 2. State Variable Time Series Comparison (Speed, Depth, Heading, Pitch)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        t_axis = sub_df["Time_s"].values - sub_df["Time_s"].values[0]

        # Speed
        axes[0, 0].plot(t_axis, sub_df["Actual_Speed"], 'k-', label='Actual', lw=2)
        axes[0, 0].plot(t_axis, sub_df["Physics_Speed"], '--', color='#ffb703', label='Physics')
        axes[0, 0].plot(t_axis, sub_df["Hybrid_Speed"], '-.', color='#0077b6', label='Hybrid (Physics+ML)')
        axes[0, 0].set_title("Surge Ground Speed (m/s)", fontweight='bold')
        axes[0, 0].set_ylabel("Speed (m/s)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Depth
        axes[0, 1].plot(t_axis, sub_df["Actual_Depth"], 'k-', label='Actual', lw=2)
        axes[0, 1].plot(t_axis, sub_df["Physics_Depth"], '--', color='#ffb703', label='Physics')
        axes[0, 1].plot(t_axis, sub_df["Hybrid_Depth"], '-.', color='#0077b6', label='Hybrid')
        axes[0, 1].set_title("Depth Profile (m)", fontweight='bold')
        axes[0, 1].set_ylabel("Depth (m)")
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Heading
        axes[1, 0].plot(t_axis, np.degrees(sub_df["Actual_Heading"]), 'k-', label='Actual', lw=2)
        axes[1, 0].plot(t_axis, np.degrees(sub_df["Physics_Heading"]), '--', color='#ffb703', label='Physics')
        axes[1, 0].plot(t_axis, np.degrees(sub_df["Hybrid_Heading"]), '-.', color='#0077b6', label='Hybrid')
        axes[1, 0].set_title("Heading Angle (deg)", fontweight='bold')
        axes[1, 0].set_xlabel("Time Elapsed (s)")
        axes[1, 0].set_ylabel("Heading (deg)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Pitch
        axes[1, 1].plot(t_axis, np.degrees(sub_df["Actual_Pitch"]), 'k-', label='Actual', lw=2)
        axes[1, 1].plot(t_axis, np.degrees(sub_df["Physics_Pitch"]), '--', color='#ffb703', label='Physics')
        axes[1, 1].plot(t_axis, np.degrees(sub_df["Hybrid_Pitch"]), '-.', color='#0077b6', label='Hybrid')
        axes[1, 1].set_title("Pitch Angle (deg)", fontweight='bold')
        axes[1, 1].set_xlabel("Time Elapsed (s)")
        axes[1, 1].set_ylabel("Pitch (deg)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.suptitle("AUV State Dynamic Predictions vs Actual Sensor Telemetry (Test Set)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        state_plot_path = os.path.join(self.plots_dir, "state_variables_comparison.png")
        plt.savefig(state_plot_path, dpi=300)
        plt.close()
        print(f"Saved state variables comparison plot to {state_plot_path}")

        # 3. Model RMSE Comparison Bar Chart
        plt.figure(figsize=(11, 5))
        x = np.arange(len(df_metrics))
        width = 0.25

        plt.bar(x - width, df_metrics["Physics RMSE"], width, label="Physics-Only", color="#ffb703")
        plt.bar(x, df_metrics["ML-Only RMSE"], width, label="ML-Only", color="#e63946")
        plt.bar(x + width, df_metrics["Hybrid RMSE"], width, label="Physics + ML Hybrid", color="#0077b6")

        plt.xticks(x, [s.split(" (")[0] for s in df_metrics["State Variable"]], rotation=15, ha='right', fontsize=9)
        plt.ylabel("Test RMSE", fontsize=10)
        plt.title("Test Set RMSE Comparison: Physics vs ML-Only vs Hybrid Dynamics", fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        bar_plot_path = os.path.join(self.plots_dir, "metrics_comparison_bar.png")
        plt.savefig(bar_plot_path, dpi=300)
        plt.close()
        print(f"Saved metrics bar chart to {bar_plot_path}")

        # 4. Multi-Step Prediction Degradation Plot
        plt.figure(figsize=(8, 5))
        horizons_sec = multistep_res["horizon_seconds"]
        plt.plot(horizons_sec, multistep_res["physics_position_rmse"], 'o--', color='#ffb703', lw=2, label='Physics-Only')
        plt.plot(horizons_sec, multistep_res["hybrid_position_rmse"], 's-', color='#0077b6', lw=2.5, label='Physics + ML Hybrid')
        plt.title("Multi-Step Prediction Error Propagation (Position RMSE)", fontsize=12, fontweight='bold')
        plt.xlabel("Prediction Horizon (seconds)", fontsize=10)
        plt.ylabel("Position RMSE (meters)", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        deg_plot_path = os.path.join(self.preds_dir, "multistep_degradation.png")
        plt.savefig(deg_plot_path, dpi=300)
        plt.close()
        print(f"Saved multi-step degradation plot to {deg_plot_path}")
