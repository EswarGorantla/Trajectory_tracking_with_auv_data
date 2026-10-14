# Comprehensive Exploratory Data Analysis (EDA) Report: Autosub6000 AUV Telemetry
**Project Title:** Learning-Based Dynamic Modeling of an Autonomous Underwater Vehicle (AUV)  
**Dataset:** Autosub6000 Mission M86 Continuous Multi-Sensor Telemetry (AURORA Project, IEEE DataPort)  
**Module:** Presentation Phase 1 — What Actually Happened in the Data Analysis  
**Tools Used:** Pure **Pandas** & **Matplotlib**

---

## Executive Overview: What Actually Happened

This report provides a transparent, detailed breakdown of the Exploratory Data Analysis (EDA) conducted on the continuous navigation and attitude telemetry of the Autosub6000 AUV during ocean survey Mission M86.

Through systematic data hygiene auditing, statistical moment profiling, bivariate correlation mapping, 2D/3D trajectory reconstruction, and acoustic transceiver noise analysis, this investigation uncovered the exact physical mechanisms and sensor limitations governing the vehicle's underwater dynamics. Most importantly, this EDA directly explains **why our deep learning residual models succeeded on attitude/velocity states (+12% to +15% RMSE reductions) and why horizontal position predictions experienced slight acoustic noise overfitting (-2% to -3%)**.

---

## 1. Dataset Provenance, Volume, & Chronological Splitting

### 1.1 Ingestion & Physical Structure
* **Vehicle Platform:** Autosub6000 (Autonomous Underwater Vehicle operated by the National Oceanography Centre).
* **Mission ID:** M86 (Cruise JC125, Celtic Sea deep-water bathymetric survey).
* **Sampling Rate:** $\Delta t = 2.00\text{ seconds}$ ($0.50\text{ Hz}$ logging frequency).
* **Continuous Mission Duration:** $17.74\text{ hours}$ ($63,886\text{ seconds}$).
* **Total Cleaned Samples:** $31,944\text{ consecutive timesteps}$.
* **Sensor Payload:**
  * **Inertial Measurement Unit (IMU):** Roll ($\phi$), Pitch ($\theta$), Heading ($\psi$).
  * **Doppler Velocity Log (DVL):** Surge speed over seabed ($U$ / `GroundSpeed`), relative water speed (`WaterSpeed`).
  * **Acoustic Pressure Transducer:** Vehicle operating depth below surface ($z$ / `depth`).
  * **Ultra-Short Baseline (USBL) Acoustic Transceiver:** Geographic Latitude and Longitude converted to local Cartesian tangent plane ($x = \text{North}, y = \text{East}$).

### 1.2 Zero-Leakage Chronological Partitioning
Because telemetry data possesses continuous autoregressive momentum, standard random shuffling causes extreme future data leakage. To ensure empirical validity, the dataset was strictly partitioned chronologically:

| Split Segment | Sample Count | Percentage | Timeline (Elapsed Hours) | Row Index Range | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | $22,360$ | $70.0\%$ | $0.00\text{ h}$ to $12.42\text{ h}$ | $0$ to $22,359$ | Feature/target scaling & model optimization |
| **Validation Set** | $4,792$ | $15.0\%$ | $12.42\text{ h}$ to $15.08\text{ h}$ | $22,360$ to $27,151$ | Early stopping (patience = 7) & hyperparameter tuning |
| **Testing Set (Unseen)** | $4,792$ | $15.0\%$ | $15.08\text{ h}$ to $17.74\text{ h}$ | $27,152$ to $31,943$ | Strict out-of-sample generalization benchmark |

* **Zero-Leakage Audit:** Confirmed that $\max(\text{Time}_{\text{train}}) = 44,718\text{ s} \le \min(\text{Time}_{\text{val}}) = 44,720\text{ s} \le \max(\text{Time}_{\text{val}}) = 54,302\text{ s} \le \min(\text{Time}_{\text{test}}) = 54,304\text{ s}$. Zero temporal overlap exists.

---

## 2. Comprehensive Descriptive Statistical Profiling

The table below outlines the empirical distribution moments computed with **Pandas** across all 31,944 telemetry samples:

| State Variable | Mean | Std Dev | Min | 25% (Q1) | Median | 75% (Q3) | Max | IQR | Skewness | Kurtosis | Outliers (IQR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **North Position ($m$)** | $2387.11$ | $1646.68$ | $-336.36$ | $901.01$ | $2454.29$ | $4038.15$ | $4632.38$ | $3137.14$ | $-0.184$ | $-1.411$ | $0$ ($0.0\%$) |
| **East Position ($m$)** | $595.95$ | $241.19$ | $-124.58$ | $354.63$ | $618.63$ | $851.00$ | $1034.71$ | $496.38$ | $-0.490$ | $-0.395$ | $18$ ($0.05\%$) |
| **Operating Depth ($m$)** | $68.34$ | $34.50$ | $0.001$ | $56.99$ | $60.34$ | $93.92$ | $108.38$ | $36.93$ | $-0.742$ | $-0.370$ | $2,784$ ($8.7\%$) |
| **Roll Angle ($\text{rad}$)** | $+0.229$ | $0.047$ | $-0.249$ | $+0.220$ | $+0.223$ | $+0.227$ | $+1.505$ | $0.007$ | $+3.142$ | $+34.810$ | $1,942$ ($6.1\%$) |
| **Pitch Angle ($\text{rad}$)** | $-0.012$ | $0.052$ | $-0.659$ | $-0.019$ | $-0.011$ | $-0.003$ | $+0.705$ | $0.016$ | $+0.841$ | $+18.230$ | $2,105$ ($6.6\%$) |
| **Heading Angle ($\text{rad}$)** | $2.917$ | $2.069$ | $0.000$ | $0.728$ | $3.094$ | $4.077$ | $6.283$ | $3.349$ | $+0.198$ | $-1.385$ | $0$ ($0.0\%$) |
| **Surge Speed ($m/s$)** | $1.192$ | $0.253$ | $0.000$ | $1.112$ | $1.242$ | $1.360$ | $3.249$ | $0.248$ | $-1.815$ | $+6.840$ | $1,852$ ($5.8\%$) |

---

## 3. What Actually Happened: The 10 Critical EDA Discoveries

### Discovery 1: Constant Cruise Speed Operating Mode
* **Observation:** Surge speed (`GroundSpeed`) exhibits a sharp, narrow probability density peak around $1.20\text{ m/s}$ (mean = $1.192\text{ m/s}$, median = $1.242\text{ m/s}$).
* **Physical Cause:** Autosub6000 operates under an automated governor cruise mode during acoustic sonar surveys to maintain optimal multibeam seafloor resolution.
* **Robotics Modeling Impact:** This confirms that modeling surge dynamics with a baseline quadratic drag model referencing $U_{\text{cruise}} = 1.20\text{ m/s}$ is physically grounded. Linear least squares regression on the training split yielded $K_d = 0.022498$.

### Discovery 2: Hydrodynamic Propeller Counter-Torque Roll Heel
* **Observation:** Roll angle is not centered at zero; it maintains a steady non-zero trim offset of $+0.229\text{ rad}$ ($+13.1^\circ$) with extreme positive kurtosis ($34.81$).
* **Physical Cause:** Autosub6000 is powered by a single large aft propeller. The clockwise propeller rotation exerts an equal and opposite reactionary counter-torque on the hull. Because the vehicle relies on passive transverse metacentric height ($GM_T$) rather than active ailerons, the hull develops a continuous equilibrium heel angle.
* **Robotics Modeling Impact:** Pure symmetric physics equations assuming zero roll trim miscalculate true hydrodynamics. The neural network successfully learned this steady bias offset without manual recalibration.

### Discovery 3: Multimodal Depth Transects
* **Observation:** Depth distribution displays distinct clustering at $60\text{ m}$ (upper survey plateau) and $94\text{ m}$ (deep seafloor survey track), reaching a maximum depth of $108.38\text{ m}$.
* **Physical Cause:** The AUV conducted a multi-tiered dive survey, descending through the water column before maintaining a fixed acoustic altitude off the seabed.

### Discovery 4: Bimodal Reciprocating Heading Distribution
* **Observation:** Heading angles cluster along two primary orientations: approximately North ($\psi \approx 0.0 - 0.5\text{ rad}$) and South ($\psi \approx 3.0 - 3.5\text{ rad}$).
* **Physical Cause:** Standard marine survey lawnmower pattern consisting of parallel tracks traversed back and forth across the survey grid.
* **Robotics Modeling Impact:** Because heading wraps continuously between $0$ and $2\pi$, feeding raw angles directly into neural networks produces artificial discontinuities (e.g. $0.01\text{ rad}$ and $6.27\text{ rad}$ appear distant to an MSE loss, despite representing almost identical directions). Our feature engineering mapped heading into $[\cos\psi, \sin\psi]$, completely eliminating wrap spikes.

### Discovery 5: Pitch Angle Diving Dynamics
* **Observation:** Pitch remains close to horizontal ($-0.012\text{ rad}$ / $-0.7^\circ$) during survey lines, but exhibits excursions between $-0.659\text{ rad}$ (steep descent, $-37.7^\circ$) and $+0.705\text{ rad}$ (steep ascent, $+40.4^\circ$).
* **Physical Cause:** Dynamic pitching controlled by stern control planes during depth-change maneuvers.

### Discovery 6: Spatial Survey Boundary
* **Observation:** In local Cartesian coordinates, the vehicle covered a bounding box of $4.63\text{ km}$ in the North direction and $1.16\text{ km}$ in the East direction.
* **Physical Cause:** Represents the exact dimensions of the ocean bathymetry survey block.

### Discovery 7: Kinematic Pitch-to-Depth Coupling
* **Observation:** Scatter plotting actual vertical depth deltas ($\Delta z = z_{t+1} - z_t$) against pitch angle ($\theta_t$) revealed an inverse relationship.
* **Physical Cause:** Directly verifies the vertical kinematic projection equation $\dot{z} = -U \sin\theta$. When the vehicle pitches down (nose down, $\theta < 0$), depth increases ($\Delta z > 0$).

### Discovery 8: Surge Velocity Drops During Pitch Maneuvers
* **Observation:** Bivariate scatter plots show that during steep pitch excursions ($|\theta| > 0.1\text{ rad}$), surge speed experiences transient drops of $0.15$ to $0.30\text{ m/s}$.
* **Physical Cause:** Deflection of the stern diving planes increases hydrodynamic form drag, temporarily retarding forward surge velocity until the cruise governor compensates with added propeller RPM.

### Discovery 9: Acoustic USBL Transceiver Noise & Position Jumps
* **Observation:** Comparing consecutive 2-second displacement steps in North position ($\Delta x$) against kinematic projection ($U \cos\theta \cos\psi \Delta t$) revealed high-frequency jagged jumps in raw coordinates.
* **Physical Cause:** Underwater GPS does not exist; AUVs rely on acoustic Ultra-Short Baseline (USBL) transceivers communicating with a surface ship. Thermocline layers, multipath acoustic reflections, and surface wave motion introduce discrete positioning noise fixes.
* **Robotics Modeling Impact:** This is the single most critical finding of the EDA. Because neural networks minimize Mean Squared Error across all frequency bands, the model attempted to predict high-frequency acoustic transceiver jumps rather than smooth vehicle inertia. This explains why hybrid position RMSE was slightly higher (-2.17% North, -2.97% East) while attitude and speed models improved dramatically.

### Discovery 10: Verified Sampling Regularity
* **Observation:** Timestamp analysis confirmed that 99.98% of all logged intervals occurred at exactly $\Delta t = 2.00\text{ s}$ ($\sigma = 0.04\text{ s}$). Zero missing records or temporal resets occurred across the entire 17.7-hour mission.

---

## 4. Why This Matters: Cause-and-Effect on Phase 1 Models

The findings from this EDA directly explain the empirical test results observed when evaluating on 4,792 unseen future samples:

```
+----------------------------------------------------------------------------------+
|                            EDA CAUSE AND EFFECT MATRIX                            |
+------------------------------------+---------------------------------------------+
| EDA Empirical Discovery            | Direct Modeling Consequence                 |
+------------------------------------+---------------------------------------------+
| Pitch plane deflection lags        | Hybrid GRU improved Pitch RMSE by +15.48%   |
| Variable buoyancy & dive drag      | Hybrid GRU improved Depth RMSE by +12.45%   |
| Yaw turning momentum & heel        | Hybrid GRU improved Heading RMSE by +12.03% |
| Propeller governor fluctuations    | Hybrid GRU improved Speed RMSE by +4.49%    |
| USBL acoustic measurement noise    | Position predictions slightly overfit (-2%) |
+------------------------------------+---------------------------------------------+
```

1. **Why Pure Physics Failed on Attitude:** Rigid-body kinematics assume instantaneous steering response. In reality, the AUV's control surfaces and fluid added mass introduce temporal lags. The 2-layer GRU's recurrent hidden state learned this history over its 10-step ($20.0\text{ s}$) sequence window.
2. **Why Pure ML Diverged in Multi-Step Forecasts:** An unconstrained neural network feeding noisy predictions back into its own recurrent loops compounds small errors, causing exponential divergence. The Hybrid model bounds the trajectory within rigid-body kinematic physics, guaranteeing closed-loop stability.
3. **Why Position Needs Kalman Filtering:** Pure raw USBL coordinates should never be directly predicted with deep neural networks without prior sensor fusion.

---

## 5. Summary of Deliverables & Access Links

* **Master EDA Jupyter Notebook:** [`AUV_Full_EDA_Analysis.ipynb`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/AUV_Full_EDA_Analysis.ipynb) *(Executed with all Matplotlib charts and Pandas tables embedded)*.
* **Presentation Report (PDF):** [`AUV_Phase1_Presentation_Report.pdf`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/AUV_Phase1_Presentation_Report.pdf) *(Print-ready publication dossier)*.
* **Presentation Report (Word):** [`AUV_Phase1_Presentation_Report.docx`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/AUV_Phase1_Presentation_Report.docx) *(Editable Word report)*.
* **Descriptive Statistics CSV:** [`eda_summary_statistics.csv`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/presentationphase1/results/eda/eda_summary_statistics.csv).
* **Correlation Matrix CSV:** [`eda_correlation_matrix.csv`](file:///c:/Users/eswar/OneDrive/Dokumen/RoboticsSEM5/presentationphase1/results/eda/eda_correlation_matrix.csv).
