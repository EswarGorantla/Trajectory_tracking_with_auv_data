import matplotlib.pyplot as plt
import numpy as np
import os

class Plotter:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def save_and_close(self, filename, subdir=""):
        path = os.path.join(self.results_dir, subdir)
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        plt.tight_layout()
        plt.savefig(full_path, dpi=150)
        plt.close()
        print(f"Saved plot to {full_path}")

    def plot_open_loop_physics(self, sub_df, x_pred, y_pred, z_pred):
        """Plots open-loop trajectory comparison from Step A."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(sub_df['East_act'], sub_df['North_act'], 'b-', label='Actual')
        ax1.plot(y_pred, x_pred, 'r--', label='Physics Predict')
        ax1.set_xlabel('East (m)')
        ax1.set_ylabel('North (m)')
        ax1.set_title('2D Trajectory (Open-loop)')
        ax1.grid(True)
        ax1.legend()
        ax1.axis('equal')

        ax2.plot(sub_df['depth'], 'b-', label='Actual')
        ax2.plot(z_pred, 'r--', label='Physics Predict')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title('Depth Profile (Open-loop)')
        ax2.invert_yaxis()
        ax2.grid(True)
        ax2.legend()
        self.save_and_close("physics_vs_actual.png", "predictions")

    def plot_training_history(self, history):
        """Plots neural network training and validation curves."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, (tr, va) in history.items():
            ax.plot(tr, label=f"{name.upper()} Train")
            ax.plot(va, '--', label=f"{name.upper()} Val")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Neural Network Training & Validation History')
        ax.grid(True)
        ax.legend()
        self.save_and_close("training_curves.png", "training")

    def plot_predictor_rmse(self, state_vars, rmse_phys, rmse_mlp, rmse_lstm, rmse_gru):
        """Plots relative prediction error bar chart relative to Physics-only baseline."""
        rel_mlp = (rmse_mlp / rmse_phys) * 100
        rel_lstm = (rmse_lstm / rmse_phys) * 100
        rel_gru = (rmse_gru / rmse_phys) * 100
        
        x = np.arange(len(state_vars))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, [100]*len(state_vars), width, label='Physics-only (100%)', color='gray', alpha=0.5)
        ax.bar(x, rel_mlp, width, label='Physics+MLP', color='orange')
        ax.bar(x + width, rel_lstm, width, label='Physics+LSTM', color='blue')
        ax.bar(x + 2*width, rel_gru, width, label='Physics+GRU', color='green')
        
        ax.set_ylabel('Relative RMSE (%)')
        ax.set_title('Predictor RMSE Comparison (Relative to Physics-only Baseline)')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(state_vars, rotation=15, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        self.save_and_close("predictor_rmse_comparison.png", "model_comparison")

    def plot_closed_loop_comparison(self, x_des, y_des, z_des, t_axis, states_pid, states_phys, states_lstm, states_gru, suffix=""):
        """Plots tracking trajectories and depth tracking profiles."""
        fig = plt.figure(figsize=(13, 5))
        ax1 = fig.add_subplot(121)
        ax1.plot(y_des, x_des, 'k--', label='Desired Path')
        ax1.plot(states_pid[:, 1], states_pid[:, 0], 'g-', label='PID')
        ax1.plot(states_phys[:, 1], states_phys[:, 0], 'b-', label='Physics MPC')
        ax1.plot(states_lstm[:, 1], states_lstm[:, 0], 'r-', label='LSTM MPC')
        ax1.plot(states_gru[:, 1], states_gru[:, 0], 'c-', label='GRU MPC')
        ax1.set_xlabel('East (m)')
        ax1.set_ylabel('North (m)')
        ax1.set_title('Horizontal Path Tracking')
        ax1.grid(True)
        ax1.legend()
        ax1.axis('equal')

        ax2 = fig.add_subplot(122)
        ax2.plot(t_axis, z_des, 'k--', label='Desired Depth')
        ax2.plot(t_axis, states_pid[:, 2], 'g-', label='PID')
        ax2.plot(t_axis, states_phys[:, 2], 'b-', label='Physics MPC')
        ax2.plot(t_axis, states_lstm[:, 2], 'r-', label='LSTM MPC')
        ax2.plot(t_axis, states_gru[:, 2], 'c-', label='GRU MPC')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title('Depth Tracking')
        ax2.grid(True)
        ax2.legend()
        ax2.invert_yaxis()
        self.save_and_close(f"controller_comparison{suffix}.png", "control")

    def plot_tracking_errors(self, t_axis, err_pid, err_phys, err_lstm, err_gru, metric_name, filename):
        """Plots errors (e.g. cross-track, depth, or heading) over time."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_axis, err_pid, 'g-', label='PID')
        ax.plot(t_axis, err_phys, 'b-', label='Physics MPC')
        ax.plot(t_axis, err_lstm, 'r-', label='LSTM MPC')
        ax.plot(t_axis, err_gru, 'c-', label='GRU MPC')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Profile")
        ax.grid(True)
        ax.legend()
        self.save_and_close(filename, "control")
        
    def plot_multi_step_predictions(self, multi_results):
        """Plots prediction trajectories over multiple forecast steps (1-step, 5-step, 10-step, 20-step)."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        horizons = list(multi_results.keys())
        for idx, h in enumerate(horizons[:4]):
            ax = axes[idx]
            res = multi_results[h]
            if 'actual' not in res:
                continue
            ax.plot(res['actual'][:, 1], res['actual'][:, 0], 'k-', label='Actual', marker='o')
            ax.plot(res['physics'][:, 1], res['physics'][:, 0], 'b--', label='Physics Rollout', marker='x')
            ax.plot(res['lstm'][:, 1], res['lstm'][:, 0], 'r-.', label='LSTM Rollout', marker='^')
            ax.set_title(f"{h}-Step Prediction Horizon")
            ax.set_xlabel('East (m)')
            ax.set_ylabel('North (m)')
            ax.grid(True)
            ax.legend()
            ax.axis('equal')
            
        self.save_and_close("multi_step_degradation.png", "predictions")
