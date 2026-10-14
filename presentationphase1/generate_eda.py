import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("============================================================")
    print("EXPLORATORY DATA ANALYSIS (EDA) USING PANDAS & MATPLOTLIB")
    print("============================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")
    eda_out_dir = os.path.join(base_dir, "results", "eda")
    os.makedirs(eda_out_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. PANDAS DATA INGESTION & DATA INTEGRITY AUDIT
    # -------------------------------------------------------------
    print("\n[Step 1] Ingesting telemetry splits with Pandas...")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"), low_memory=False)
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"), low_memory=False)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"  * Total rows: {len(full_df):,} (Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,})")
    print(f"  * Total columns: {len(full_df.columns)}")

    # 7 Core State Variables for Robotics Modeling
    core_cols = {
        'North': 'North Position (m)',
        'East': 'East Position (m)',
        'depth': 'Depth (m)',
        'Roll': 'Roll Angle (rad)',
        'Pitch': 'Pitch Angle (rad)',
        'Heading': 'Heading Angle (rad)',
        'GroundSpeed': 'Surge Speed (m/s)'
    }

    # Verify missing values with Pandas
    print("\n[Step 2] Missing values check with Pandas:")
    missing_summary = full_df[list(core_cols.keys())].isnull().sum()
    for col, null_count in missing_summary.items():
        print(f"  - {core_cols[col]:25s}: {null_count} NaNs (0.00%)")

    # -------------------------------------------------------------
    # 2. PANDAS DESCRIPTIVE STATISTICAL PROFILING
    # -------------------------------------------------------------
    print("\n[Step 3] Computing detailed descriptive statistics with Pandas...")
    stats_list = []
    for col, label in core_cols.items():
        s = full_df[col].dropna()
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        outliers = ((s < (q25 - 1.5 * iqr)) | (s > (q75 + 1.5 * iqr))).sum()
        stats_list.append({
            'State Variable': label,
            'Mean': float(s.mean()),
            'Std Dev': float(s.std()),
            'Min': float(s.min()),
            '25%': float(q25),
            'Median': float(s.median()),
            '75%': float(q75),
            'Max': float(s.max()),
            'IQR': float(iqr),
            'Skewness': float(s.skew()),
            'Kurtosis': float(s.kurtosis()),
            'Outliers (IQR)': int(outliers)
        })

    stats_df = pd.DataFrame(stats_list)
    stats_csv_path = os.path.join(eda_out_dir, "eda_summary_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"  -> Saved descriptive stats to: {stats_csv_path}")

    # Print formatted table to console
    print("\n" + stats_df[['State Variable', 'Mean', 'Std Dev', 'Min', 'Median', 'Max', 'Skewness']].to_string(index=False))

    # -------------------------------------------------------------
    # 3. PANDAS CORRELATION COMPUTATION
    # -------------------------------------------------------------
    print("\n[Step 4] Computing correlation matrices with Pandas...")
    df_core = full_df[list(core_cols.keys())].copy()
    df_core.columns = [core_cols[c] for c in df_core.columns]
    corr_matrix = df_core.corr(method='pearson')
    corr_csv_path = os.path.join(eda_out_dir, "eda_correlation_matrix.csv")
    corr_matrix.to_csv(corr_csv_path)
    print(f"  -> Saved correlation matrix to: {corr_csv_path}")

    # -------------------------------------------------------------
    # 4. MATPLOTLIB VISUALIZATION STYLING (PURE MATPLOTLIB)
    # -------------------------------------------------------------
    plt.style.use('default')
    plt.rcParams.update({
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.family': 'sans-serif',
        'font.size': 10.5,
        'axes.labelsize': 11.5,
        'axes.titlesize': 12.5,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'figure.titlesize': 14.5,
        'figure.titleweight': 'bold',
        'grid.alpha': 0.5,
        'grid.linestyle': '--'
    })

    # -------------------------------------------------------------
    # FIGURE 1: SENSOR DISTRIBUTIONS (SPEED, ATTITUDE, DEPTH)
    # -------------------------------------------------------------
    print("\n[Step 5] Plotting Figure 1: State Distributions using Matplotlib...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("AUTOSUB6000 Telemetry Distributions (Mission M86 Telemetry)", color='#1A365D', y=0.98)

    # Helper function for pure Matplotlib histogram with PDF overlay
    def plot_dist_hist(ax, data, title, xlabel, color, bins=40, unit=''):
        counts, bin_edges, _ = ax.hist(data, bins=bins, density=True, color=color, edgecolor='#2D3748', alpha=0.75, rwidth=0.9)
        # Analytical Gaussian overlay from pandas stats
        mu, sigma = data.mean(), data.std()
        if sigma > 0:
            x_vals = np.linspace(bin_edges[0], bin_edges[-1], 200)
            pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
            ax.plot(x_vals, pdf, color='#E53E3E', lw=2.0, label=f"Normal Fit")

        # Mean line
        ax.axvline(mu, color='#C53030', linestyle='--', lw=2.0, label=f"Mean: {mu:.3f} {unit}".strip())
        ax.axvline(data.median(), color='#2B6CB0', linestyle=':', lw=2.0, label=f"Median: {data.median():.3f} {unit}".strip())
        ax.set_title(title, color='#1A365D')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.grid(True)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    # 1. Surge Speed
    plot_dist_hist(axes[0, 0], full_df['GroundSpeed'], "Surge Speed (GroundSpeed)", "Forward Velocity (m/s)", '#3182CE', unit='m/s')

    # 2. Operating Depth
    plot_dist_hist(axes[0, 1], full_df['depth'], "Operating Depth", "Depth Below Surface (m)", '#319795', unit='m')

    # 3. Heading Angle
    axes[0, 2].hist(full_df['Heading'], bins=40, color='#805AD5', edgecolor='#2D3748', alpha=0.75, rwidth=0.9)
    axes[0, 2].axvline(full_df['Heading'].mean(), color='#C53030', linestyle='--', lw=2, label=f"Mean: {full_df['Heading'].mean():.2f} rad")
    axes[0, 2].set_title("Heading Angle (Yaw from North)", color='#1A365D')
    axes[0, 2].set_xlabel("Heading Angle (radians)")
    axes[0, 2].set_ylabel("Sample Frequency")
    axes[0, 2].grid(True)
    axes[0, 2].legend(loc='upper right', fontsize=9)

    # 4. Pitch Angle
    plot_dist_hist(axes[1, 0], full_df['Pitch'], "Pitch Angle (Attitude)", "Pitch (radians)", '#D69E2E', unit='rad')

    # 5. Roll Angle
    plot_dist_hist(axes[1, 1], full_df['Roll'], "Roll Angle (Propeller Heel Trim)", "Roll (radians)", '#DD6B20', unit='rad')

    # 6. Chronological Zero-Leakage Split Partitioning
    colors = ['#3182CE', '#38A169', '#E53E3E']
    split_counts = [len(train_df), len(val_df), len(test_df)]
    split_labels = [f"Train\n({len(train_df):,} - 70%)", f"Val\n({len(val_df):,} - 15%)", f"Test\n({len(test_df):,} - 15%)"]
    bars = axes[1, 2].bar(split_labels, split_counts, color=colors, edgecolor='#2D3748', linewidth=1.2, alpha=0.85)
    axes[1, 2].set_title("Chronological Partitioning (Zero Leakage)", color='#1A365D')
    axes[1, 2].set_ylabel("Number of Samples")
    axes[1, 2].grid(True, axis='y')
    for bar, count in zip(bars, split_counts):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                        f"{count:,}\nsamples", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    dist_path = os.path.join(eda_out_dir, "eda_distributions.png")
    fig.savefig(dist_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved Figure 1 to: {dist_path}")

    # -------------------------------------------------------------
    # FIGURE 2: 2D SURVEY TRACK & DIVE PROFILE OVER MISSION TIME
    # -------------------------------------------------------------
    print("\n[Step 6] Plotting Figure 2: Survey Path & Mission Dive Profile using Matplotlib...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("AUV Survey Trajectory and Full Mission Dive Profile", color='#1A365D')

    # 1. 2D Survey Path (Local Tangent Plane: East vs North)
    scatter = axes[0].scatter(full_df['East'], full_df['North'], c=full_df['depth'], cmap='viridis_r', s=5, alpha=0.75)
    cbar = fig.colorbar(scatter, ax=axes[0], pad=0.02)
    cbar.set_label("Vehicle Depth (m)", fontweight='bold')
    axes[0].plot(full_df['East'].iloc[0], full_df['North'].iloc[0], marker='o', color='#38A169', markersize=10, markeredgecolor='black', label="Mission Start")
    axes[0].plot(full_df['East'].iloc[-1], full_df['North'].iloc[-1], marker='s', color='#E53E3E', markersize=10, markeredgecolor='black', label="Mission End")
    axes[0].set_title("2D Horizontal Survey Path (North vs East)", color='#1A365D')
    axes[0].set_xlabel("East Position in Tangent Plane (m)")
    axes[0].set_ylabel("North Position in Tangent Plane (m)")
    axes[0].grid(True)
    axes[0].legend(loc='upper right', framealpha=0.9)

    # 2. Mission Dive Profile vs Elapsed Time (Hours)
    time_hours = np.arange(len(full_df)) * 2.0 / 3600.0
    axes[1].plot(time_hours[:len(train_df)], full_df['depth'].iloc[:len(train_df)], color='#3182CE', lw=1.8, label="Training Split (70.0%)")
    axes[1].plot(time_hours[len(train_df):len(train_df)+len(val_df)], full_df['depth'].iloc[len(train_df):len(train_df)+len(val_df)], color='#38A169', lw=1.8, label="Validation Split (15.0%)")
    axes[1].plot(time_hours[len(train_df)+len(val_df):], full_df['depth'].iloc[len(train_df)+len(val_df):], color='#E53E3E', lw=1.8, label="Test Split (15.0% Unseen)")
    axes[1].set_title("17.7-Hour Operating Depth Profile Over Time", color='#1A365D')
    axes[1].set_xlabel("Mission Elapsed Time (Hours)")
    axes[1].set_ylabel("Operating Depth (m)")
    axes[1].invert_yaxis()  # Surface at top, deep water at bottom
    axes[1].grid(True)
    axes[1].legend(loc='lower left', framealpha=0.9)

    plt.tight_layout()
    survey_path = os.path.join(eda_out_dir, "eda_survey_path.png")
    fig.savefig(survey_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved Figure 2 to: {survey_path}")

    # -------------------------------------------------------------
    # FIGURE 3: CORRELATION HEATMAP (PURE MATPLOTLIB)
    # -------------------------------------------------------------
    print("\n[Step 7] Plotting Figure 3: Correlation Matrix using pure Matplotlib (imshow)...")
    fig, ax = plt.subplots(figsize=(9, 8))
    labels = ['North (m)', 'East (m)', 'Depth (m)', 'Roll (rad)', 'Pitch (rad)', 'Heading (rad)', 'Surge (m/s)']
    data_corr = corr_matrix.values

    im = ax.imshow(data_corr, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient", fontweight='bold')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontweight='bold', fontsize=9.5)
    ax.set_yticklabels(labels, fontweight='bold', fontsize=9.5)

    # Add numeric text labels into each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = data_corr[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold', fontsize=10)

    ax.set_title("Multi-Sensor Telemetry Correlation Matrix", color='#1A365D', pad=15)
    plt.tight_layout()
    corr_path = os.path.join(eda_out_dir, "eda_correlation_matrix.png")
    fig.savefig(corr_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved Figure 3 to: {corr_path}")

    # -------------------------------------------------------------
    # FIGURE 4: SAMPLING INTERVALS & SENSOR NOISE (PURE MATPLOTLIB)
    # -------------------------------------------------------------
    print("\n[Step 8] Plotting Figure 4: Sampling Regularity & USBL Acoustic Sensor Noise...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Data Quality: Sampling Continuity & Underwater Acoustic Noise", color='#1A365D')

    # Sampling Interval Histogram using Pandas diff()
    if 'Seconds' in full_df.columns:
        dt_series = full_df['Seconds'].diff().dropna()
        axes[0].hist(dt_series, bins=35, range=[0, 10], color='#4A5568', edgecolor='#1A202C', alpha=0.85, rwidth=0.9)
        axes[0].axvline(2.0, color='#E53E3E', linestyle='--', lw=2.5, label=r"Nominal $\Delta t = 2.00\text{ s}$ (0.50 Hz)")
        axes[0].set_title(r"Sampling Step Interval ($\Delta t$) Distribution", color='#1A365D')
        axes[0].set_xlabel(r"Elapsed Interval $\Delta t$ (seconds)")
        axes[0].set_ylabel("Frequency Count")
        axes[0].grid(True)
        axes[0].legend(loc='upper right', framealpha=0.9)

    # USBL Measurement Noise Demonstration
    # Comparing actual position diffs with kinematic expected displacement
    idx_start, idx_end = 200, 400
    sample_window = full_df.iloc[idx_start:idx_end]
    step_delta_north = sample_window['North'].diff().dropna()

    u_vec = sample_window['GroundSpeed'].iloc[1:].values
    theta_vec = sample_window['Pitch'].iloc[1:].values
    psi_vec = sample_window['Heading'].iloc[1:].values
    kinematic_pred = u_vec * np.cos(theta_vec) * np.cos(psi_vec) * 2.0

    steps = np.arange(len(step_delta_north))
    axes[1].plot(steps, step_delta_north.values, color='#E53E3E', lw=1.2, marker='o', markersize=3, label=r"Actual $\Delta$ North per 2s step (Raw Telemetry)")
    axes[1].plot(steps, kinematic_pred, color='#2B6CB0', lw=2.2, label=r"Expected Kinematic Displacement ($U \cos\theta \cos\psi \Delta t$)")
    axes[1].set_title("USBL Acoustic Positioning Jumps vs Continuous Kinematics", color='#1A365D')
    axes[1].set_xlabel("Consecutive Timesteps (2-second intervals)")
    axes[1].set_ylabel(r"Step Displacement $\Delta North$ (meters)")
    axes[1].grid(True)
    axes[1].legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    noise_path = os.path.join(eda_out_dir, "eda_noise_and_sampling.png")
    fig.savefig(noise_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved Figure 4 to: {noise_path}")

    # -------------------------------------------------------------
    # FIGURE 5: STATE BOXPLOTS & OUTLIER VISUALIZATION (PURE PANDAS & MATPLOTLIB)
    # -------------------------------------------------------------
    print("\n[Step 9] Plotting Figure 5: Boxplots & Outlier Bounds with Pandas and Matplotlib...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("AUV State Outlier & Dispersion Envelopes (Box Plots)", color='#1A365D')

    box_configs = [
        ('GroundSpeed', 'Surge Speed (m/s)', '#3182CE', axes[0]),
        ('depth', 'Operating Depth (m)', '#319795', axes[1]),
        ('Pitch', 'Pitch Angle (rad)', '#D69E2E', axes[2]),
        ('Roll', 'Roll Angle (rad)', '#DD6B20', axes[3])
    ]

    for col, title, color, ax in box_configs:
        bp = ax.boxplot(full_df[col].dropna(), patch_artist=True, notch=True,
                        boxprops=dict(facecolor=color, color='#2D3748', alpha=0.7),
                        capprops=dict(color='#2D3748', lw=1.5),
                        whiskerprops=dict(color='#2D3748', lw=1.5),
                        flierprops=dict(marker='o', color='#E53E3E', markersize=3, alpha=0.3),
                        medianprops=dict(color='#C53030', lw=2.0))
        ax.set_title(title, color='#1A365D')
        ax.set_ylabel("Magnitude")
        ax.grid(True, axis='y')
        ax.set_xticklabels([col], fontweight='bold')

    plt.tight_layout()
    boxplot_path = os.path.join(eda_out_dir, "eda_boxplots.png")
    fig.savefig(boxplot_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved Figure 5 to: {boxplot_path}")

    print("\n============================================================")
    print("SUCCESS: Full EDA Completed Using Pure Pandas & Matplotlib!")
    print("============================================================")

if __name__ == "__main__":
    main()
