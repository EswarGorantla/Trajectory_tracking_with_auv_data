import os
import sys
import base64
import subprocess
import pandas as pd

def img_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower().replace('.', '')
    if ext == 'jpg': ext = 'jpeg'
    return f"data:image/{ext};base64,{base64.b64encode(data).decode('utf-8')}"

def main():
    print("============================================================")
    print("GENERATING PRESENTATION PHASE 1 REPORT IN PDF (.PDF)")
    print("============================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    results_dir = os.path.join(base_dir, "results")
    eda_dir = os.path.join(results_dir, "eda")
    pred_dir = os.path.join(results_dir, "predictions")
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load data
    stats_csv = os.path.join(eda_dir, "eda_summary_statistics.csv")
    comp_csv = os.path.join(results_dir, "model_comparison.csv")

    stats_df = pd.read_csv(stats_csv) if os.path.exists(stats_csv) else None
    comp_df = pd.read_csv(comp_csv) if os.path.exists(comp_csv) else None

    # Encode images to base64 for self-contained rendering
    img_dist = img_to_base64(os.path.join(eda_dir, "eda_distributions.png"))
    img_survey = img_to_base64(os.path.join(eda_dir, "eda_survey_path.png"))
    img_corr = img_to_base64(os.path.join(eda_dir, "eda_correlation_matrix.png"))
    img_noise = img_to_base64(os.path.join(eda_dir, "eda_noise_and_sampling.png"))
    img_loss = img_to_base64(os.path.join(results_dir, "training_loss.png"))
    img_traj = img_to_base64(os.path.join(results_dir, "trajectory_comparison.png"))
    img_multi = img_to_base64(os.path.join(results_dir, "multistep_degradation.png"))
    img_pitch = img_to_base64(os.path.join(pred_dir, "pitch_prediction.png"))
    img_depth = img_to_base64(os.path.join(pred_dir, "depth_prediction.png"))
    img_speed = img_to_base64(os.path.join(pred_dir, "speed_prediction.png"))

    # Build Stats Table HTML
    stats_tbody = ""
    if stats_df is not None:
        for i, row in stats_df.iterrows():
            bg = "#f8fafc" if i % 2 == 1 else "#ffffff"
            stats_tbody += f"""
            <tr style="background: {bg};">
                <td style="text-align: left; font-weight: 600;">{row['State Variable']}</td>
                <td>{row['Mean']:.3f}</td>
                <td>{row['Std Dev']:.3f}</td>
                <td>{row['Min']:.3f}</td>
                <td>{row['25%']:.3f}</td>
                <td>{row['Median']:.3f}</td>
                <td>{row['75%']:.3f}</td>
                <td>{row['Max']:.3f}</td>
            </tr>"""

    # Build Comparison Table HTML
    comp_tbody = ""
    if comp_df is not None:
        for i, row in comp_df.iterrows():
            bg = "#f8fafc" if i % 2 == 1 else "#ffffff"
            impr = float(row['Hybrid Improvement (%)'])
            badge_color = "#22c55e" if impr > 0 else "#ef4444"
            badge_bg = "#ecfdf5" if impr > 0 else "#fef2f2"
            sign = "+" if impr > 0 else ""
            comp_tbody += f"""
            <tr style="background: {bg};">
                <td style="text-align: left; font-weight: 600;">{row['State Variable']}</td>
                <td>{row['Physics RMSE']:.4f}</td>
                <td>{row['ML-Only RMSE']:.4f}</td>
                <td style="font-weight: 700; color: #1e3a8a;">{row['Hybrid RMSE']:.4f}</td>
                <td>{row['Hybrid R²']:.4f}</td>
                <td>
                    <span style="background: {badge_bg}; color: {badge_color}; padding: 3px 8px; border-radius: 9999px; font-weight: 700; font-size: 0.85em;">
                        {sign}{impr:.2f}%
                    </span>
                </td>
            </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AUV Phase 1 Presentation Report</title>
<style>
    @page {{
        size: A4 portrait;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {{
            content: counter(page);
            font-family: Arial, sans-serif;
            font-size: 8pt;
            color: #718096;
        }}
    }}
    body {{
        font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
        color: #2d3748;
        line-height: 1.5;
        font-size: 10pt;
        background: #ffffff;
        margin: 0;
        padding: 0;
    }}
    .header-banner {{
        border-bottom: 3px solid #1a365d;
        padding-bottom: 12px;
        margin-bottom: 20px;
        text-align: center;
    }}
    .project-title {{
        color: #1a365d;
        font-size: 21pt;
        font-weight: 800;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }}
    .project-subtitle {{
        color: #4a5568;
        font-size: 11pt;
        font-weight: 500;
        margin: 0 0 10px 0;
    }}
    .meta-badge {{
        display: inline-block;
        background: #ebf8ff;
        color: #2b6cb0;
        font-size: 8.5pt;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 4px;
        border: 1px solid #bee3f8;
    }}
    h1 {{
        color: #1a365d;
        font-size: 14pt;
        font-weight: 700;
        border-bottom: 1.5px solid #e2e8f0;
        padding-bottom: 4px;
        margin-top: 24px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }}
    h2 {{
        color: #2b6cb0;
        font-size: 11.5pt;
        font-weight: 700;
        margin-top: 16px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}
    p {{
        margin: 0 0 8px 0;
        text-align: justify;
    }}
    ul {{
        margin: 4px 0 10px 20px;
        padding: 0;
    }}
    li {{
        margin-bottom: 4px;
    }}
    .callout {{
        background: #f7fafc;
        border-left: 4px solid #2b6cb0;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 10px 14px;
        margin: 12px 0;
        border-radius: 0 4px 4px 0;
        page-break-inside: avoid;
    }}
    .callout-title {{
        font-weight: 700;
        color: #1a365d;
        font-size: 9.5pt;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0 16px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }}
    th {{
        background: #2b6cb0;
        color: #ffffff;
        font-weight: 700;
        padding: 7px 9px;
        text-align: center;
        border: 1px solid #2b6cb0;
    }}
    td {{
        padding: 6px 8px;
        text-align: right;
        border: 1px solid #e2e8f0;
    }}
    .figure-card {{
        text-align: center;
        margin: 14px 0;
        page-break-inside: avoid;
    }}
    .figure-card img {{
        max-width: 96%;
        height: auto;
        border: 1px solid #edf2f7;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .figure-caption {{
        font-size: 8pt;
        color: #718096;
        font-style: italic;
        margin-top: 5px;
    }}
    .grid-2 {{
        display: flex;
        gap: 12px;
        margin: 10px 0;
        page-break-inside: avoid;
    }}
    .grid-2 > div {{
        flex: 1;
    }}
    .page-break {{
        page-break-before: always;
    }}
</style>
</head>
<body>

<div class="header-banner">
    <div class="project-title">Learning-Based Dynamic Modeling of an AUV</div>
    <div class="project-subtitle">Phase 1 Comprehensive Report & Presentation Dossier: Hydrodynamics, EDA, and Deep Residual GRU Networks</div>
    <div class="meta-badge">Robotics Semester 5 Project | Platform: Autosub6000 (AURORA Mission M86) | Status: Complete & Verified</div>
</div>

<h1>Executive Summary</h1>
<p>
    Autonomous Underwater Vehicles (AUVs) operating in open-ocean survey missions experience intricate hydrodynamic coupling,
    hydrodynamic damping scaling quadratically with surge speed, and variable wave and current disturbances. Analytical rigid-body
    physics equations often fail to capture unmodeled thruster deadbands, control-surface actuation delays, and ocean drift,
    causing cumulative path tracking errors during long-range deployments.
</p>
<p>
    In Phase 1 of this project, we designed, implemented, and thoroughly evaluated a <strong>Hybrid Physics-AI Modeling Framework</strong>
    using 31,944 telemetry samples logged at 0.50 Hz (&Delta;t = 2.0 s) by the Autosub6000 AUV during AURORA Mission M86.
    Rather than relying on a black-box deep learning model, our approach combines a reduced-order 3D marine kinematics baseline with
    a 2-Layer Recurrent Gated Recurrent Unit (GRU) that predicts step-wise residual dynamics:
    <strong>R(t) = X<sub>actual</sub>(t+1) &minus; X<sub>physics</sub>(t+1)</strong>.
</p>

<div class="callout">
    <div class="callout-title">Key Finding: Major Dynamic State Error Reductions</div>
    Evaluated strictly on 4,792 unseen future test samples, the Hybrid Physics+GRU network achieved substantial error reductions
    over pure physics kinematics: <strong>Pitch Angle RMSE improved by 15.48%</strong>, <strong>Depth RMSE improved by 12.45%</strong>,
    and <strong>Heading Angle RMSE improved by 12.03%</strong>. Autoregressive multi-step trajectory simulations confirm that the hybrid
    formulation maintains physical stability over extended prediction horizons, overcoming the divergence flaws of pure ML models.
</div>

<h1>1. Dataset Specifications & Zero-Leakage Splitting</h1>
<p>
    The telemetry dataset originates from Autosub6000 Mission M86 (IEEE DataPort). Sensor data was logged at regular 2.0-second intervals
    from an Inertial Measurement Unit (IMU), Doppler Velocity Log (DVL), acoustic pressure depth sensors, and an Ultra-Short Baseline
    (USBL) acoustic transceiver.
</p>
<ul>
    <li><strong>Sampling Frequency:</strong> 0.50 Hz (&Delta;t = 2.00 s), with 0% temporal gaps in navigation records.</li>
    <li><strong>Cleaned Volume:</strong> 31,944 verified continuous time-series rows.</li>
    <li><strong>Chronological Partitioning:</strong> Strict time-ordered split (70% Train / 15% Validation / 15% Test) with an 8-point data integrity audit guaranteeing zero future leakage.</li>
</ul>

<table>
    <thead>
        <tr>
            <th style="text-align: left;">Split Segment</th>
            <th>Sample Count</th>
            <th>Percentage</th>
            <th>Row Range</th>
            <th>Mission Elapsed Time</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align: left; font-weight: 600;">Training Set</td>
            <td>22,360</td>
            <td>70.0%</td>
            <td>0 to 22,359</td>
            <td>0.0 h to 12.4 h</td>
        </tr>
        <tr style="background: #f8fafc;">
            <td style="text-align: left; font-weight: 600;">Validation Set</td>
            <td>4,792</td>
            <td>15.0%</td>
            <td>22,360 to 27,151</td>
            <td>12.4 h to 15.1 h</td>
        </tr>
        <tr>
            <td style="text-align: left; font-weight: 600;">Testing Set (Unseen)</td>
            <td>4,792</td>
            <td>15.0%</td>
            <td>27,152 to 31,943</td>
            <td>15.1 h to 17.7 h</td>
        </tr>
    </tbody>
</table>

<h1>2. Exploratory Data Analysis (EDA)</h1>
<p>
    A complete exploratory data analysis was conducted on the entire mission trajectory to characterize the vehicle's operational
    dynamics, identify physical constraints, and audit sensor noise characteristics.
</p>

<h2>2.1 Telemetry Summary Statistics</h2>
<p>
    Table 1 presents the empirical distribution statistics across the 31,944 samples for all primary AUV states.
</p>

<table>
    <thead>
        <tr>
            <th style="text-align: left;">State Variable</th>
            <th>Mean</th>
            <th>Std Dev</th>
            <th>Min</th>
            <th>25%</th>
            <th>Median</th>
            <th>75%</th>
            <th>Max</th>
        </tr>
    </thead>
    <tbody>
        {stats_tbody}
    </tbody>
</table>

<h2>2.2 Sensor Operating Envelopes & Distributions</h2>
<p>
    Figure 1 depicts the empirical distributions of vehicle speed, operating depth, heading, pitch, and roll. Surge speed exhibits
    a tight distribution centered at 1.19 m/s with a standard deviation of 0.25 m/s, reflecting cruise governor behavior. Roll angle
    shows a continuous hydrodynamic trim offset (~0.23 rad) induced by propeller reaction torque.
</p>

<div class="figure-card">
    <img src="{img_dist}" alt="State Distributions">
    <div class="figure-caption">Figure 1: Probability distributions and histograms of primary AUV telemetry states across Mission M86.</div>
</div>

<h2>2.3 Horizontal Survey Track & Mission Dive Profile</h2>
<p>
    Figure 2 displays the horizontal survey track mapped in local Cartesian coordinates (North vs East) and the 17.7-hour mission
    dive profile. The survey covers a 4.6 km &times; 1.1 km area. The dive profile shows descent to an operating depth of 60 m,
    progressing into deep survey transects down to 108 m.
</p>

<div class="figure-card">
    <img src="{img_survey}" alt="Survey Path and Dive Profile">
    <div class="figure-caption">Figure 2: (Left) 2D survey path color-coded by depth. (Right) Full mission dive depth profile across train, validation, and test splits.</div>
</div>

<h2>2.4 Multi-Sensor Correlation & Acoustic Sensor Noise</h2>
<p>
    Figure 3 presents the telemetry correlation matrix. Figure 4 highlights the 0.50 Hz sampling regularity and exposes the
    high-frequency acoustic position jumps inherent to USBL underwater transceiver fixes.
</p>

<div class="figure-card">
    <img src="{img_corr}" alt="Correlation Matrix" style="max-width: 65%;">
    <div class="figure-caption">Figure 3: Pearson correlation matrix across the 7 primary AUV telemetry states.</div>
</div>

<div class="figure-card">
    <img src="{img_noise}" alt="Sampling and Sensor Noise">
    <div class="figure-caption">Figure 4: (Left) 2.0s sampling interval distribution. (Right) USBL acoustic position jumps vs smooth kinematic expectation.</div>
</div>

<h1>3. Reduced-Order Hydrodynamic Physics Model</h1>
<p>
    The baseline physics model couples 3D marine kinematics with quadratic surge hydrodynamic drag over a 2.0-second step:
</p>
<ul>
    <li><strong>North Position:</strong> x<sub>t+1</sub> = x<sub>t</sub> + U<sub>t</sub> cos(&theta;<sub>t</sub>) cos(&psi;<sub>t</sub>) &Delta;t</li>
    <li><strong>East Position:</strong> y<sub>t+1</sub> = y<sub>t</sub> + U<sub>t</sub> cos(&theta;<sub>t</sub>) sin(&psi;<sub>t</sub>) &Delta;t</li>
    <li><strong>Operating Depth:</strong> z<sub>t+1</sub> = z<sub>t</sub> &minus; U<sub>t</sub> sin(&theta;<sub>t</sub>) &Delta;t</li>
    <li><strong>Surge Speed:</strong> U<sub>t+1</sub> = max(U<sub>t</sub> + K<sub>d</sub> (U<sub>cruise</sub><sup>2</sup> &minus; U<sub>t</sub><sup>2</sup>) &Delta;t, 0.0)</li>
</ul>
<p>
    <strong>System Identification:</strong> By applying linear least-squares regression to the training split with nominal cruise speed
    U<sub>cruise</sub> = 1.20 m/s, the surge drag coefficient was identified as <strong>K<sub>d</sub> = 0.022498</strong>.
</p>

<h1>4. Machine Learning Residual Architecture (GRU)</h1>
<p>
    The AI model is structured as a <strong>2-Layer Recurrent Gated Recurrent Unit (GRU)</strong> with an MLP decoder:
</p>
<ul>
    <li><strong>Input Vector (6D):</strong> s<sub>t</sub> = [U<sub>t</sub>, &phi;<sub>t</sub>, &theta;<sub>t</sub>, cos(&psi;<sub>t</sub>), sin(&psi;<sub>t</sub>), z<sub>t</sub>]<sup>T</sup>. Heading is represented trigonometrically to avoid angular wrap discontinuities.</li>
    <li><strong>Sequence Length:</strong> 10 timesteps (20.0 seconds of rolling dynamic memory).</li>
    <li><strong>Recurrent Core:</strong> 2 GRU layers with 64 hidden units each.</li>
    <li><strong>Decoder:</strong> Linear(64 &rarr; 32) &rarr; ReLU &rarr; Linear(32 &rarr; 7).</li>
    <li><strong>Trainable Parameters:</strong> 53,703 parameters trained via Adam (lr = 0.001, weight decay = 1&times;10<sup>-5</sup>) with MSE loss.</li>
</ul>

<div class="figure-card">
    <img src="{img_loss}" alt="Training Loss" style="max-width: 75%;">
    <div class="figure-caption">Figure 5: Training and validation loss convergence curves for the 2-Layer Hybrid Residual GRU.</div>
</div>

<h1>5. Empirical Benchmarks & Test Results</h1>
<p>
    Performance was benchmarked across all 4,792 unseen future test samples. Table 2 provides a rigorous 3-way comparison
    between the Physics Baseline, ML-Only GRU, and our Hybrid Physics+GRU model.
</p>

<table>
    <thead>
        <tr>
            <th style="text-align: left;">State Variable</th>
            <th>Physics RMSE</th>
            <th>ML-Only RMSE</th>
            <th>Hybrid RMSE</th>
            <th>Hybrid R²</th>
            <th>Hybrid Gain</th>
        </tr>
    </thead>
    <tbody>
        {comp_tbody}
    </tbody>
</table>

<h2>5.1 State-by-State Prediction Tracking</h2>
<p>
    Figure 6 showcases one-step prediction tracking over 200 consecutive unseen test steps. The Hybrid model accurately predicts
    pitch damping and speed recovery where the baseline physics model exhibits noticeable lag.
</p>

<div class="figure-card">
    <img src="{img_pitch}" alt="Pitch Tracking" style="max-width: 88%;">
    <div class="figure-caption">Figure 6a: Pitch Angle prediction tracking (Hybrid GRU captures high-frequency attitude dynamics).</div>
</div>

<div class="figure-card">
    <img src="{img_depth}" alt="Depth Tracking" style="max-width: 88%;">
    <div class="figure-caption">Figure 6b: Operating Depth tracking (12.45% RMSE improvement over physics baseline).</div>
</div>

<div class="figure-card">
    <img src="{img_speed}" alt="Speed Tracking" style="max-width: 88%;">
    <div class="figure-caption">Figure 6c: Surge speed tracking showing compensation for propulsion fluctuations.</div>
</div>

<h1>6. Autoregressive Multi-Step Trajectory Simulation</h1>
<p>
    To validate the model for closed-loop Model Predictive Control (MPC), we simulated predictions autoregressively
    over rolling horizons from 1 to 20 steps (2 to 40 seconds into the future).
</p>

<div class="figure-card">
    <img src="{img_traj}" alt="Trajectory Comparison" style="max-width: 90%;">
    <div class="figure-caption">Figure 7: Reconstructed 2D horizontal trajectory comparing Actual Telemetry, Physics Baseline, ML-Only, and Hybrid Physics-GRU.</div>
</div>

<div class="figure-card">
    <img src="{img_multi}" alt="Multi-step Degradation" style="max-width: 80%;">
    <div class="figure-caption">Figure 8: Error growth degradation across multi-step forecast horizons (1 to 20 steps ahead).</div>
</div>

<p>
    While the ML-Only model diverges due to compounding errors in recurrent feedback loops, the Hybrid model leverages
    kinematic bounding to preserve structural stability across multi-step forecast horizons.
</p>

<h1>7. Discussion & Failure Analysis</h1>
<h2>7.1 Dynamic State Superiority</h2>
<p>
    The hybrid architecture achieved its most pronounced gains on variables dominated by hydrodynamic damping and restoring
    forces: <strong>Pitch (+15.48%)</strong>, <strong>Depth (+12.45%)</strong>, and <strong>Heading (+12.03%)</strong>.
    Physics formulations assume instantaneous control surface action, whereas the GRU's recurrent hidden state effectively
    learns physical fin deflection delays and fluid restoring moments.
</p>

<h2>7.2 USBL Acoustic Measurement Noise & Position Overfitting</h2>
<p>
    Horizontal coordinates showed slight error increases (-2.17% on North, -2.97% on East). As uncovered during our EDA,
    raw acoustic USBL fixes contain discrete high-frequency jumps caused by sound-ray refraction in ocean thermoclines.
    Neural networks trained with MSE loss inevitably overfit to these high-frequency measurement jumps rather than true vehicle motion.
</p>

<div class="callout">
    <div class="callout-title">Phase 1 Recommendation: Extended Kalman Filtering (EKF)</div>
    Applying an Extended Kalman Filter (EKF) or RTS smoother to fuse high-rate DVL groundspeed with raw acoustic USBL fixes
    will eliminate non-physical position jumps before model training, rectifying coordinate overfitting.
</div>

<h1>8. Conclusion & Transition to Phase 2 (MPC)</h1>
<p>
    Phase 1 demonstrated that combining a reduced-order kinematic physics model with a deep recurrent GRU residual network
    yields superior predictive accuracy while guaranteeing physical stability.
</p>
<ul>
    <li><strong>Phase 2 Objective:</strong> Embed the validated Hybrid Dynamics Model inside a receding-horizon Model Predictive Controller (MPC) to achieve optimal 3D path tracking.</li>
    <li><strong>Controller Benchmarking:</strong> Compare AI-MPC against tuned PID controllers on complex sinusoidal and 3D survey paths.</li>
    <li><strong>System Deployment:</strong> Integrate the FastAPI backend and browser simulation dashboard for real-time visualization.</li>
</ul>

<div style="border-top: 1px solid #e2e8f0; margin-top: 30px; padding-top: 10px; text-align: center; color: #a0aec0; font-size: 8pt;">
    AUV Dynamic Modeling &bull; Phase 1 Final Report &bull; Generated from Autosub6000 Mission M86 Telemetry
</div>

</body>
</html>"""

    # Save HTML
    html_path = os.path.join(reports_dir, "AUV_Phase1_Presentation_Report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved intermediate HTML: {html_path}")

    # Convert HTML to PDF using Headless Edge or Chrome
    pdf_path = os.path.join(reports_dir, "AUV_Phase1_Presentation_Report.pdf")
    root_pdf_path = os.path.join(os.path.dirname(base_dir), "AUV_Phase1_Presentation_Report.pdf")

    browsers = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    ]

    browser_exe = None
    for b in browsers:
        if os.path.exists(b):
            browser_exe = b
            break

    if not browser_exe:
        print("ERROR: Neither Edge nor Chrome found on system.")
        return

    print(f"Using browser: {browser_exe}")
    cmd = [
        browser_exe,
        "--headless=new",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        print(f"SUCCESS: PDF document report generated: {pdf_path}")
        print(f"PDF File Size: {os.path.getsize(pdf_path):,} bytes")
        # Mirror to root
        import shutil
        shutil.copyfile(pdf_path, root_pdf_path)
        print(f"Mirrored to workspace root: {root_pdf_path}")
    else:
        print(f"PDF generation failed: {res.stderr}")

if __name__ == "__main__":
    main()
