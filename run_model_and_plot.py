import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set file paths
csv_path = r"c:\Users\eswar\OneDrive\Dokumen\RoboticsSEM5\original_data\M86\multibeam_echosounder\multi-beam-echosounder-index.csv"
output_plot_path = r"C:\Users\eswar\.gemini\antigravity\brain\54a8d0c6-8ff3-4ae2-a5f4-65100c13a45d\trajectory_plot.png"

# 1. Load and clean data
print("Loading data...")
df = pd.read_csv(csv_path, low_memory=False)
df.columns = df.columns.str.strip()

cols_to_parse = ['Latitude', 'Longitude', 'Heading', 'Pitch', 'Roll', 'GroundSpeed', 'depth', 'Date', 'Time']
for col in cols_to_parse[:-2]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=cols_to_parse[:-2]).copy()
df['DateTime'] = pd.to_datetime(df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip(), format='%d-%b-%Y %H:%M:%S')
df = df.sort_values('DateTime').reset_index(drop=True)

# 2. Select the turning segment (starts at index 6268, length 500 rows = 1000 seconds)
start_idx = 6268
n_steps = 500
segment = df.iloc[start_idx:start_idx+n_steps].copy().reset_index(drop=True)

# 3. Establish local coordinate system (North, East in meters)
R = 6371000.0 # Earth's radius in meters
lat0 = np.radians(segment['Latitude'].iloc[0])
lon0 = np.radians(segment['Longitude'].iloc[0])

segment['North_act'] = R * np.radians(segment['Latitude'] - segment['Latitude'].iloc[0])
segment['East_act'] = R * np.radians(segment['Longitude'] - segment['Longitude'].iloc[0]) * np.cos(lat0)
segment['Depth_act'] = segment['depth'].values

# 4. AUV Rigid-body Dynamics Simulation
# State vector: [North, East, Depth, Speed]
# Initial conditions
x_pred = np.zeros(n_steps)
y_pred = np.zeros(n_steps)
z_pred = np.zeros(n_steps)
u_pred = np.zeros(n_steps)

x_pred[0] = segment['North_act'].iloc[0]
y_pred[0] = segment['East_act'].iloc[0]
z_pred[0] = segment['Depth_act'].iloc[0]
u_pred[0] = segment['GroundSpeed'].iloc[0]

# Model parameters
dt = 2.0 # sample rate
U_cruise = segment['GroundSpeed'].mean() # nominal thrust balances drag at cruise speed
k_d = 0.05 # drag-to-mass dynamic coefficient (time constant ~ 20s)

# Integrate over time
for t in range(n_steps - 1):
    # Retrieve inputs (attitude at current step)
    theta = segment['Pitch'].iloc[t]
    psi = segment['Heading'].iloc[t]
    
    # 1. Forward Speed Dynamics (Surge):
    # dU/dt = k_d * (U_cruise^2 - U_t^2)
    # Using Forward Euler Integration:
    du = k_d * (U_cruise**2 - u_pred[t]**2) * dt
    u_pred[t+1] = u_pred[t] + du
    
    # Ensure speed doesn't go negative (physical limit)
    u_pred[t+1] = max(u_pred[t+1], 0.0)
    
    # 2. Kinematics (Body velocities to local North-East-Down):
    # dx/dt = U * cos(pitch) * cos(heading)
    # dy/dt = U * cos(pitch) * sin(heading)
    # dz/dt = -U * sin(pitch) => depth rate = U * sin(pitch)
    dx = u_pred[t] * np.cos(theta) * np.cos(psi) * dt
    dy = u_pred[t] * np.cos(theta) * np.sin(psi) * dt
    dz = -u_pred[t] * np.sin(theta) * dt # depth change
    
    # Integrate position
    x_pred[t+1] = x_pred[t] + dx
    y_pred[t+1] = y_pred[t] + dy
    z_pred[t+1] = z_pred[t] - dz # depth increases when moving downward (dz is negative)

# Store predictions in the DataFrame
segment['North_pred'] = x_pred
segment['East_pred'] = y_pred
segment['Depth_pred'] = z_pred
segment['Speed_pred'] = u_pred

# 5. Calculate Errors
segment['Pos_error'] = np.sqrt((segment['North_act'] - segment['North_pred'])**2 + (segment['East_act'] - segment['East_pred'])**2)
rmse_horiz = np.sqrt(np.mean(segment['Pos_error']**2))
rmse_depth = np.sqrt(np.mean((segment['Depth_act'] - segment['Depth_pred'])**2))
rmse_speed = np.sqrt(np.mean((segment['GroundSpeed'] - segment['Speed_pred'])**2))

print("\n--- Simulation Results ---")
print(f"Cruising Speed (U_cruise): {U_cruise:.4f} m/s")
print(f"Horizontal Position Drift (RMSE): {rmse_horiz:.4f} meters over 1000s")
print(f"Depth Drift (RMSE): {rmse_depth:.4f} meters over 1000s")
print(f"Speed Error (RMSE): {rmse_speed:.4f} m/s")
print(f"Max Horizontal Error: {segment['Pos_error'].max():.4f} meters")

# 6. Generate Plot
fig = plt.figure(figsize=(14, 10))
grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel 1: 2D Horizontal Trajectory
ax1 = fig.add_subplot(grid[0, 0])
ax1.plot(segment['East_act'], segment['North_act'], 'b-', label='Actual (USBL/DVL)', linewidth=2)
ax1.plot(segment['East_pred'], segment['North_pred'], 'r--', label='Predicted (Physics Model)', linewidth=2)
ax1.plot(segment['East_act'].iloc[0], segment['North_act'].iloc[0], 'go', label='Start')
ax1.plot(segment['East_act'].iloc[-1], segment['North_act'].iloc[-1], 'kx', label='End')
ax1.set_xlabel('East Position (meters)')
ax1.set_ylabel('North Position (meters)')
ax1.set_title('AUV 2D Horizontal Trajectory (360° Turn)')
ax1.grid(True)
ax1.legend()
ax1.axis('equal')

# Panel 2: Depth vs Time
ax2 = fig.add_subplot(grid[0, 1])
time_axis = np.arange(0, n_steps * 2, 2)
ax2.plot(time_axis, segment['Depth_act'], 'b-', label='Actual Depth', linewidth=2)
ax2.plot(time_axis, segment['Depth_pred'], 'r--', label='Predicted Depth', linewidth=2)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Depth (meters)')
ax2.set_title('Depth Profile over Time')
ax2.grid(True)
ax2.legend()
ax2.invert_yaxis() # Depth increases downwards

# Panel 3: Speed vs Time
ax3 = fig.add_subplot(grid[1, 0])
ax3.plot(time_axis, segment['GroundSpeed'], 'b-', label='Actual Speed (DVL)', linewidth=1.5)
ax3.plot(time_axis, segment['Speed_pred'], 'r--', label='Predicted Speed (Surge Model)', linewidth=2)
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Speed (m/s)')
ax3.set_title('Vehicle Surge Speed over Time')
ax3.grid(True)
ax3.legend()

# Panel 4: Position Error over Time
ax4 = fig.add_subplot(grid[1, 1])
ax4.plot(time_axis, segment['Pos_error'], 'm-', label='Horizontal Error', linewidth=2)
ax4.plot(time_axis, np.abs(segment['Depth_act'] - segment['Depth_pred']), 'c-', label='Depth Error', linewidth=1.5)
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Error (meters)')
ax4.set_title('Position Error Accumulation (Drift) over Time')
ax4.grid(True)
ax4.legend()

plt.suptitle(f"AURORA AUV Dynamics Model Evaluation (RMSE Horiz: {rmse_horiz:.2f}m, Depth: {rmse_depth:.2f}m)", fontsize=14, weight='bold')
plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_plot_path}")
plt.close()
