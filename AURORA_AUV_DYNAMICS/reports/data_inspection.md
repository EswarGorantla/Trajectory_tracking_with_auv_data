# AURORA Dataset Inspection Report

**Project Title:** Learning-Based Dynamic Modeling of an Autonomous Underwater Vehicle Using AURORA Multi-Sensor Data  
**Dataset Reference:** Autosub6000 AUV Survey, Mission M86, Cruise JC125 (Porcupine Abyssal Plain / Celtic Sea)

---

## 1. Available Files & Formats

* **Primary Telemetry File:** `original_data/M86/multibeam_echosounder/multi-beam-echosounder-index.csv`
* **Format:** Comma-Separated Values (CSV), ASCII text.
* **File Size:** ~7.12 MB
* **Total Rows:** 36,444 rows
* **Time Duration:** ~20.25 hours continuous mission deployment
* **Sampling Rate:** $0.50\text{ Hz}$ ($\Delta t = 2.0\text{ seconds}$)

---

## 2. Sensor Audit & Variable Inventory

| Raw Column Name | Physical Meaning | Instrument / Sensor | Raw Unit | Coordinate System | Missing Values (%) |
| :--- | :--- | :--- | :---: | :--- | :---: |
| `Date` | Calendar date | System Clock | `YYYY-MM-DD` | UTC | 0.00% |
| `Time` | Time of measurement | System Clock | `HH:MM:SS.mmm` | UTC | 0.00% |
| `Seconds` | Mission timestamp | System Clock | `seconds` | Elapsed | 0.00% |
| `Latitude` | Geodetic latitude | USBL Acoustic Transponder / GPS | `degrees` | WGS84 | 0.00% |
| `Longitude` | Geodetic longitude | USBL Acoustic Transponder / GPS | `degrees` | WGS84 | 0.00% |
| `depth` | Submergence depth below sea level | Paroscientific Pressure Sensor | `meters (m)` | Earth Z (positive-down) | 0.00% |
| `altitude` | Distance above seabed | Multibeam / Altimeter | `meters (m)` | Seabed relative | 0.00% |
| `Roll` | Longitudinal tilt angle | Inertial Measurement Unit (IMU) | `radians (rad)` | Body Frame ($\phi$) | 0.00% |
| `Pitch` | Lateral tilt angle (nose up/down) | Inertial Measurement Unit (IMU) | `radians (rad)` | Body Frame ($\theta$) | 0.00% |
| `Heading` | Compass direction clockwise from North | Fiber Optic Gyrocompass / IMU | `radians (rad)` | Earth Frame ($\psi$) | 0.00% |
| `GroundSpeed` | Forward linear velocity over seabed | Doppler Velocity Log (DVL) | `m/s` | Body Surge ($U$) | 0.00% |

---

## 3. Sensors & Actuation Assessment

1. **Sensors Available:**
   * Acoustic Positioning (USBL / GPS) $\rightarrow$ Latitude & Longitude
   * Pressure Depth Sensor $\rightarrow$ Depth ($z$)
   * Inertial Measurement Unit (IMU) $\rightarrow$ Roll ($\phi$), Pitch ($\theta$), Heading ($\psi$)
   * Doppler Velocity Log (DVL) $\rightarrow$ GroundSpeed ($U$)
2. **Sensors NOT Present in Navigation Index:**
   * 3-Axis body angular velocity rates ($p, q, r$) are not logged as separate columns.
   * Separate sway ($v$) and heave ($w$) velocity channels are not broken out (only total GroundSpeed surge is present).
   * Accelerometers 3-axis raw acceleration values ($a_x, a_y, a_z$) are not exported in the index CSV.
3. **Actuator / Control Input Telemetry:**
   * **Actuator Status:** No rudder fin deflection angles, elevator angles, motor RPM, or thruster voltage commands exist in the telemetry file.
   * **Scientific Handling:** As mandated by scientific honesty, we do not invent control inputs. We formulate the dynamics problem using state history:
     $$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{x}_{t-1}, \dots, \mathbf{x}_{t-k})$$

---

## 4. Coordinate Transformations Required

1. **Latitude/Longitude $\rightarrow$ Local Cartesian $(x, y)$:**
   Using equirectangular projection centered at the mission start origin $(\text{Lat}_0, \text{Lon}_0)$:
   $$x = (\text{Lat} - \text{Lat}_0) \times \frac{\pi}{180} \times R_{\text{earth}}$$
   $$y = (\text{Lon} - \text{Lon}_0) \times \frac{\pi}{180} \times R_{\text{earth}} \times \cos\left(\text{Lat}_0 \times \frac{\pi}{180}\right)$$
2. **Angle Wrapping:**
   Ensure heading remains in $[0, 2\pi)$ and angular differences are wrapped to $[-\pi, \pi]$.
