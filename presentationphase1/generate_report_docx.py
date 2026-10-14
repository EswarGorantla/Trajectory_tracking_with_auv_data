import os
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_border(cell, **kwargs):
    """
    kwargs can be top, bottom, left, right
    values: dict(val='single', sz='4', color='CCCCCC')
    """
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = OxmlElement(tag)
            element.set(qn('w:val'), edge_data.get('val', 'single'))
            element.set(qn('w:sz'), str(edge_data.get('sz', 4)))
            element.set(qn('w:color'), edge_data.get('color', 'CCCCCC'))
            tcBorders.append(element)
    tcPr.append(tcBorders)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(26, 54, 93) # #1A365D Navy
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(43, 108, 176) # #2B6CB0 Slate Blue
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(45, 55, 72)
    return p

def add_body_paragraph(doc, text, bold_prefix=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Arial'
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(26, 32, 44)
    r = p.add_run(text)
    r.font.name = 'Arial'
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(45, 55, 72)
    return p

def add_bullet_point(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Arial'
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(26, 32, 44)
    r = p.add_run(text)
    r.font.name = 'Arial'
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(45, 55, 72)
    return p

def add_callout(doc, title, text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "EDF2F7")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    set_cell_border(cell, left=dict(val='single', sz=24, color='2B6CB0'),
                          top=dict(val='single', sz=4, color='CBD5E0'),
                          right=dict(val='single', sz=4, color='CBD5E0'),
                          bottom=dict(val='single', sz=4, color='CBD5E0'))
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r1 = p.add_run(f"KEY HIGHLIGHT: {title}\n")
    r1.font.name = 'Arial'
    r1.font.bold = True
    r1.font.size = Pt(10.5)
    r1.font.color.rgb = RGBColor(26, 54, 93)

    r2 = p.add_run(text)
    r2.font.name = 'Arial'
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(45, 55, 72)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def add_image_with_caption(doc, img_path, caption, width=Inches(6.2)):
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    run = p_img.add_run()
    run.add_picture(img_path, width=width)

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = 'Arial'
    r_cap.font.size = Pt(9.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(113, 128, 150)

def main():
    print("============================================================")
    print("GENERATING PRESENTATION PHASE 1 REPORT IN WORD (.DOCX)")
    print("============================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    results_dir = os.path.join(base_dir, "results")
    eda_dir = os.path.join(results_dir, "eda")
    os.makedirs(reports_dir, exist_ok=True)

    doc = Document()

    # Set page margins (0.75 in / ~19 mm)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # -------------------------------------------------------------
    # DOCUMENT COVER / HEADER
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    run_t = p_title.add_run("Learning-Based Dynamic Modeling of an Autonomous Underwater Vehicle (AUV)")
    run_t.font.name = 'Arial'
    run_t.font.size = Pt(22)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(26, 54, 93)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    run_s = p_sub.add_run("Phase 1 Project Report & Presentation Comprehensive Dossier\n"
                          "Hydrodynamic Kinematics, Telemetry EDA, and Deep Residual GRU Networks")
    run_s.font.name = 'Arial'
    run_s.font.size = Pt(13)
    run_s.font.color.rgb = RGBColor(74, 85, 104)

    # Meta banner
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_after = Pt(16)
    r_meta = p_meta.add_run("Robotics Semester 5 Project | Platform: Autosub6000 (AURORA Mission M86) | Status: Complete")
    r_meta.font.name = 'Arial'
    r_meta.font.size = Pt(9.5)
    r_meta.font.bold = True
    r_meta.font.color.rgb = RGBColor(43, 108, 176)

    # Horizontal dividing line
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(10)
    run_d = p_div.add_run("―" * 58)
    run_d.font.color.rgb = RGBColor(203, 213, 224)
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # -------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # -------------------------------------------------------------
    add_heading_1(doc, "Executive Summary")
    add_body_paragraph(
        doc,
        "Autonomous Underwater Vehicles (AUVs) operating in real ocean environments experience highly complex, "
        "nonlinear hydrodynamic forces that analytical physics equations fail to capture accurately. Unmodeled thruster "
        "deadbands, control-surface deflection lags, wave action, and variable buoyancy lead to substantial cumulative drift "
        "when relying solely on rigid-body kinematics."
    )
    add_body_paragraph(
        doc,
        "In Phase 1 of this project, we designed and experimentally validated a Hybrid Physics-AI Modeling Framework using real-world "
        "telemetry logged during the Autosub6000 AUV Mission M86 (recorded at 0.50 Hz / dt = 2.0 s). Instead of replacing physics "
        "principles with a black-box model, our architecture uses a nominal 3D kinematics baseline combined with a 2-Layer Recurrent "
        "Gated Recurrent Unit (GRU) trained strictly to predict residual mismatches: R(t) = X_actual(t+1) - X_physics(t+1)."
    )

    add_callout(
        doc,
        "Substantial Dynamics Improvements over Baseline",
        "Evaluated on 4,792 unseen future test samples, the Hybrid Physics+GRU network achieved dramatic error reductions "
        "over pure physics: Pitch Angle RMSE improved by 15.48%, Operating Depth RMSE improved by 12.45%, and Heading Angle RMSE "
        "improved by 12.03%. Multi-step autoregressive trajectory simulations demonstrate that hybrid modeling preserves physical "
        "plausibility while eliminating cumulative dynamic drift."
    )

    # -------------------------------------------------------------
    # SECTION 1: DATASET SPECIFICATIONS & TIME PARTITIONING
    # -------------------------------------------------------------
    add_heading_1(doc, "1. Dataset Specifications & Zero-Leakage Splitting")
    add_body_paragraph(
        doc,
        "The project utilizes sensor telemetry from Autosub6000 Mission M86, an operational ocean survey vehicle. "
        "The vehicle logged high-rate navigation data using a Doppler Velocity Log (DVL), an Inertial Measurement Unit (IMU), "
        "pressure depth transducers, and Ultra-Short Baseline (USBL) acoustic positioning transponders."
    )

    add_bullet_point(doc, "Raw Telemetry Source: IEEE DataPort Autosub6000 Mission M86 multibeam echosounder logs.", "Dataset Source: ")
    add_bullet_point(doc, "31,944 verified samples logged every 2.00 seconds (0.50 Hz sampling frequency).", "Cleaned Sample Volume: ")
    add_bullet_point(doc, "Strict chronological train/validation/test ordering ensuring zero future data leakage.", "Temporal Partitioning: ")

    # Dataset split table
    t_split = doc.add_table(rows=4, cols=5)
    t_split.alignment = WD_TABLE_ALIGNMENT.CENTER
    split_headers = ["Split Segment", "Sample Count", "Percentage", "Row Index Range", "Mission Elapsed Time"]
    for j, h in enumerate(split_headers):
        cell = t_split.cell(0, j)
        set_cell_background(cell, "2B6CB0")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.font.name = 'Arial'
        run.font.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(255, 255, 255)

    split_data = [
        ["Training Set", "22,360", "70.0%", "0 to 22,359", "0.0 h to 12.4 h"],
        ["Validation Set", "4,792", "15.0%", "22,360 to 27,151", "12.4 h to 15.1 h"],
        ["Testing Set (Unseen)", "4,792", "15.0%", "27,152 to 31,943", "15.1 h to 17.7 h"]
    ]
    for i, row in enumerate(split_data):
        fill = "F7FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = t_split.cell(i + 1, j)
            set_cell_background(cell, fill)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            set_cell_border(cell, bottom=dict(val='single', sz=4, color='E2E8F0'))
            p = cell.paragraphs[0]
            if j in [1, 2]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = 'Arial'
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(45, 55, 72)
            if j == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    add_heading_1(doc, "2. Exploratory Data Analysis (EDA)")
    add_body_paragraph(
        doc,
        "Rigorous exploratory data analysis was conducted on the full continuous mission telemetry to characterize vehicle operating "
        "envelopes, detect measurement noise, and establish physical consistency before training neural networks."
    )

    add_heading_2(doc, "2.1 Telemetry Summary Statistics")
    add_body_paragraph(
        doc,
        "Table 1 outlines the empirical distribution metrics for all seven primary AUV state variables across the 31,944 telemetry samples."
    )

    # Statistics table from CSV
    stats_csv_path = os.path.join(eda_dir, "eda_summary_statistics.csv")
    if os.path.exists(stats_csv_path):
        df_stats = pd.read_csv(stats_csv_path)
        t_stats = doc.add_table(rows=len(df_stats) + 1, cols=8)
        t_stats.alignment = WD_TABLE_ALIGNMENT.CENTER
        stat_cols = ["State Variable", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"]
        for j, c in enumerate(stat_cols):
            cell = t_stats.cell(0, j)
            set_cell_background(cell, "2B6CB0")
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(c)
            r.font.name = 'Arial'
            r.font.bold = True
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(255, 255, 255)

        for i, r_val in df_stats.iterrows():
            fill = "F7FAFC" if i % 2 == 1 else "FFFFFF"
            vals = [
                str(r_val['State Variable']),
                f"{r_val['Mean']:.3f}",
                f"{r_val['Std Dev']:.3f}",
                f"{r_val['Min']:.3f}",
                f"{r_val['25%']:.3f}",
                f"{r_val['Median']:.3f}",
                f"{r_val['75%']:.3f}",
                f"{r_val['Max']:.3f}"
            ]
            for j, val in enumerate(vals):
                cell = t_stats.cell(i + 1, j)
                set_cell_background(cell, fill)
                set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
                set_cell_border(cell, bottom=dict(val='single', sz=4, color='E2E8F0'))
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
                r = p.add_run(val)
                r.font.name = 'Arial'
                r.font.size = Pt(8.5)
                r.font.color.rgb = RGBColor(45, 55, 72)
                if j == 0:
                    r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_heading_2(doc, "2.2 State Distributions & Physical Operating Envelopes")
    add_body_paragraph(
        doc,
        "Figure 1 displays the distributions of surge speed, depth, heading, pitch, and roll. Surge speed exhibits a sharp operating mode "
        "around 1.20 m/s (cruise propulsion setting), with an overall mean of 1.19 m/s. Pitch values stay centered near zero (-0.012 rad) "
        "with small excursions during depth changes. Roll exhibits a steady hydrodynamic heel trim angle (~0.23 rad) caused by the single-screw "
        "propeller counter-torque."
    )
    add_image_with_caption(
        doc,
        os.path.join(eda_dir, "eda_distributions.png"),
        "Figure 1: Probability distributions and histograms of primary AUV telemetry states across Mission M86."
    )

    add_heading_2(doc, "2.3 Survey Path & Mission Dive Profile")
    add_body_paragraph(
        doc,
        "Figure 2 illustrates the horizontal 2D survey path in local Cartesian coordinates (North vs East) and the mission depth dive "
        "profile over 17.7 hours of continuous operation. The vehicle surveyed a track of 4.6 km North by 1.1 km East. The dive profile shows "
        "initial descent to ~60 m depth, followed by an operating deep lawnmower pattern extending down to 108 m."
    )
    add_image_with_caption(
        doc,
        os.path.join(eda_dir, "eda_survey_path.png"),
        "Figure 2: (Left) Horizontal 2D trajectory color-coded by depth. (Right) Full mission dive depth profile across train, val, and test splits."
    )

    add_heading_2(doc, "2.4 Multi-Sensor Telemetry Correlation Analysis")
    add_body_paragraph(
        doc,
        "The state correlation matrix (Figure 3) uncovers key hydrodynamic couplings. Strong kinematic correlation exists between "
        "North/East positions and Heading (0.61 and 0.47). Depth shows negative correlation with surge speed, indicating slight velocity drops "
        "during vertical diving operations due to increased plane drag."
    )
    add_image_with_caption(
        doc,
        os.path.join(eda_dir, "eda_correlation_matrix.png"),
        "Figure 3: Pearson correlation matrix between 7 state telemetry variables."
    )

    add_heading_2(doc, "2.5 Sampling Regularity & USBL Acoustic Sensor Noise")
    add_body_paragraph(
        doc,
        "The sampling interval distribution confirms highly consistent 0.50 Hz telemetry (dt = 2.0 s). However, time-series inspection "
        "reveals high-frequency position jumps in the raw North and East coordinates (Figure 4). These jumps stem from acoustic Ultra-Short "
        "Baseline (USBL) underwater transceiver noise, which causes instantaneous step velocities inconsistent with vehicle momentum."
    )
    add_image_with_caption(
        doc,
        os.path.join(eda_dir, "eda_noise_and_sampling.png"),
        "Figure 4: (Left) Sampling interval frequency distribution. (Right) High-frequency USBL acoustic transceiver jumps vs continuous kinematic projection."
    )

    # -------------------------------------------------------------
    # SECTION 3: REDUCED-ORDER HYDRODYNAMIC PHYSICS MODEL
    # -------------------------------------------------------------
    add_heading_1(doc, "3. Reduced-Order Hydrodynamic Physics Model")
    add_body_paragraph(
        doc,
        "We formulated a 3D kinematics baseline combined with quadratic surge damping to represent nominal AUV motion over a 2.0-second window:"
    )
    add_bullet_point(doc, "North Position: x_{t+1} = x_t + U_t * cos(theta_t) * cos(psi_t) * dt", "Kinematic Integration: ")
    add_bullet_point(doc, "East Position:  y_{t+1} = y_t + U_t * cos(theta_t) * sin(psi_t) * dt", "Kinematic Integration: ")
    add_bullet_point(doc, "Vertical Depth: z_{t+1} = z_t - U_t * sin(theta_t) * dt", "Vertical Integration: ")
    add_bullet_point(doc, "Surge Drag Model: U_{t+1} = max(U_t + K_d * (U_cruise^2 - U_t^2) * dt, 0.0)", "Quadratic Damping: ")

    add_body_paragraph(
        doc,
        "System Identification: Using linear least-squares regression on the training split with nominal cruise speed U_cruise = 1.20 m/s, "
        "the hydrodynamic surge damping coefficient was estimated as Kd = 0.022498. While this baseline projects motion accurately in calm "
        "straight-line segments, it completely omits thruster dynamics, cross-flow ocean currents, and pitch-depth buoyancy coupling."
    )

    # -------------------------------------------------------------
    # SECTION 4: MACHINE LEARNING RESIDUAL ARCHITECTURE (GRU)
    # -------------------------------------------------------------
    add_heading_1(doc, "4. Deep Recurrent Residual Architecture (GRU)")
    add_body_paragraph(
        doc,
        "To compensate for physics baseline deficiencies, we engineered a deep recurrent neural network. Instead of learning full dynamics "
        "from scratch, the network learns the residual error vector: R(t) = X_actual(t+1) - X_physics(t+1)."
    )

    add_heading_2(doc, "4.1 State & Feature Representations")
    add_bullet_point(doc, "s_t = [U_t, phi_t, theta_t, cos(psi_t), sin(psi_t), z_t]^T in R^6. Heading is represented via trigonometric projection to eliminate 0-to-2pi wrap discontinuities.", "Continuous 6D Input Features: ")
    add_bullet_point(doc, "r_t = [Delta North, Delta East, Delta Depth, Delta Roll, Delta Pitch, Delta Heading, Delta Speed]^T in R^7.", "Residual Target Outputs: ")
    add_bullet_point(doc, "Rolling history of 10 timesteps (T = 10 steps = 20.0 seconds of historical dynamics).", "Temporal Sequence Window: ")

    add_heading_2(doc, "4.2 Network Specifications & Hyperparameters")
    # Architecture table
    t_arch = doc.add_table(rows=7, cols=2)
    t_arch.alignment = WD_TABLE_ALIGNMENT.CENTER
    arch_data = [
        ["Model Architecture", "2-Layer Recurrent Gated Recurrent Unit (GRU) with MLP Decoder"],
        ["Hidden Units", "64 units per recurrent layer (2 layers deep)"],
        ["Decoder Subnetwork", "Linear(64 -> 32) -> ReLU -> Dropout(0.1) -> Linear(32 -> 7)"],
        ["Total Trainable Parameters", "53,703 parameters"],
        ["Loss Function", "Normalized Mean Squared Error (MSELoss)"],
        ["Optimization", "Adam (learning rate = 0.001, weight decay = 1e-5), ReduceLROnPlateau"],
    ]
    for i, (k, v) in enumerate(arch_data):
        cell_k = t_arch.cell(i, 0)
        cell_v = t_arch.cell(i, 1)
        cell_k.width = Inches(2.2)
        cell_v.width = Inches(4.3)
        fill = "F7FAFC" if i % 2 == 1 else "FFFFFF"
        set_cell_background(cell_k, fill)
        set_cell_background(cell_v, fill)
        set_cell_margins(cell_k, top=60, bottom=60, left=80, right=80)
        set_cell_margins(cell_v, top=60, bottom=60, left=80, right=80)
        set_cell_border(cell_k, bottom=dict(val='single', sz=4, color='E2E8F0'))
        set_cell_border(cell_v, bottom=dict(val='single', sz=4, color='E2E8F0'))

        pk = cell_k.paragraphs[0]
        rk = pk.add_run(k)
        rk.font.name = 'Arial'
        rk.font.bold = True
        rk.font.size = Pt(9)
        rk.font.color.rgb = RGBColor(26, 54, 93)

        pv = cell_v.paragraphs[0]
        rv = pv.add_run(v)
        rv.font.name = 'Arial'
        rv.font.size = Pt(9)
        rv.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_heading_2(doc, "4.3 Training Convergence")
    add_body_paragraph(
        doc,
        "The model converged smoothly within 30 epochs (Figure 5). Early stopping prevented overfitting, achieving a best validation "
        "loss of 0.4489 MSE. The training and validation loss curves tracked closely without divergent gap, proving excellent generalization."
    )
    add_image_with_caption(
        doc,
        os.path.join(results_dir, "training_loss.png"),
        "Figure 5: Training and validation loss convergence curves for the 2-Layer Hybrid Residual GRU."
    )

    # -------------------------------------------------------------
    # SECTION 5: EMPIRICAL BENCHMARKS & MODEL COMPARISON
    # -------------------------------------------------------------
    add_heading_1(doc, "5. Empirical Test Benchmarks & Results")
    add_body_paragraph(
        doc,
        "Models were evaluated strictly on 4,792 unseen future test telemetry samples. Table 2 provides a complete three-way benchmark "
        "comparing the nominal Physics Baseline, an ML-Only GRU (predicting state deltas directly), and our Hybrid Physics+GRU model."
    )

    comp_csv = os.path.join(results_dir, "model_comparison.csv")
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        t_comp = doc.add_table(rows=len(df_comp) + 1, cols=6)
        t_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
        headers = ["State Variable", "Physics RMSE", "ML-Only RMSE", "Hybrid RMSE", "Hybrid R²", "Hybrid Improvement"]
        for j, h in enumerate(headers):
            cell = t_comp.cell(0, j)
            set_cell_background(cell, "2B6CB0")
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(h)
            r.font.name = 'Arial'
            r.font.bold = True
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(255, 255, 255)

        for i, row in df_comp.iterrows():
            fill = "F7FAFC" if i % 2 == 1 else "FFFFFF"
            impr_val = float(row['Hybrid Improvement (%)'])
            impr_str = f"+{impr_val:.2f}%" if impr_val > 0 else f"{impr_val:.2f}%"
            vals = [
                str(row['State Variable']),
                f"{row['Physics RMSE']:.4f}",
                f"{row['ML-Only RMSE']:.4f}",
                f"{row['Hybrid RMSE']:.4f}",
                f"{row['Hybrid R²']:.4f}",
                impr_str
            ]
            for j, v in enumerate(vals):
                cell = t_comp.cell(i + 1, j)
                set_cell_background(cell, fill)
                set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
                set_cell_border(cell, bottom=dict(val='single', sz=4, color='E2E8F0'))
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
                r = p.add_run(v)
                r.font.name = 'Arial'
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(45, 55, 72)
                if j == 0:
                    r.font.bold = True
                if j == 5:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(40, 167, 69) if impr_val > 0 else RGBColor(220, 53, 69)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_heading_2(doc, "5.1 Prediction Tracking Comparison")
    add_body_paragraph(
        doc,
        "Figure 6 details the telemetry tracking performance across 200 consecutive test steps for all primary states. The Hybrid model "
        "captures rapid pitch adjustments and heading transitions where pure physics lags significantly."
    )

    pred_dir = os.path.join(results_dir, "predictions")
    add_image_with_caption(
        doc,
        os.path.join(pred_dir, "pitch_prediction.png"),
        "Figure 6a: Pitch Angle prediction tracking (Hybrid GRU captures high-frequency attitude dynamics)."
    )
    add_image_with_caption(
        doc,
        os.path.join(pred_dir, "depth_prediction.png"),
        "Figure 6b: Operating Depth tracking comparison (12.45% RMSE improvement over physics baseline)."
    )
    add_image_with_caption(
        doc,
        os.path.join(pred_dir, "speed_prediction.png"),
        "Figure 6c: Surge speed tracking showing compensation for propeller speed fluctuations."
    )

    # -------------------------------------------------------------
    # SECTION 6: MULTI-STEP TRAJECTORY RECONSTRUCTION
    # -------------------------------------------------------------
    add_heading_1(doc, "6. Multi-Step Autoregressive Trajectory Simulation")
    add_body_paragraph(
        doc,
        "To test suitability for closed-loop Model Predictive Control (MPC), models were simulated autoregressively across rolling "
        "prediction horizons (1 to 20 steps = 2 to 40 seconds into the future)."
    )

    add_image_with_caption(
        doc,
        os.path.join(results_dir, "trajectory_comparison.png"),
        "Figure 7: Reconstructed 2D horizontal trajectory comparing Actual Telemetry, Physics Baseline, ML-Only, and Hybrid Physics-GRU."
    )

    add_body_paragraph(
        doc,
        "Figure 8 depicts error accumulation over increasing forecast horizons. The pure ML model diverges rapidly due to error compounding "
        "when feeding noisy predictions back into recurrent hidden states. In contrast, the Hybrid model maintains physical bounding, "
        "ensuring stability over extended planning horizons."
    )
    add_image_with_caption(
        doc,
        os.path.join(results_dir, "multistep_degradation.png"),
        "Figure 8: Error growth degradation across multi-step forecast horizons (1 to 20 steps ahead)."
    )

    # -------------------------------------------------------------
    # SECTION 7: DISCUSSION, LIMITATIONS & FAILURE ANALYSIS
    # -------------------------------------------------------------
    add_heading_1(doc, "7. Discussion, Key Findings & Failure Analysis")

    add_heading_2(doc, "7.1 Why the Hybrid Model Excels on Dynamic States")
    add_body_paragraph(
        doc,
        "The Hybrid model achieved its largest gains on variables governed by complex hydrodynamic damping and restoring forces: "
        "Pitch (+15.48%), Depth (+12.45%), and Heading (+12.03%). Physics equations assume instantaneous control-surface response, "
        "whereas the GRU network's recurrent hidden state successfully captures fin actuation lag and buoyancy restoring moments."
    )

    add_heading_2(doc, "7.2 Acoustic USBL Noise & Position Prediction Overfitting")
    add_body_paragraph(
        doc,
        "Interestingly, horizontal position predictions showed a slight degradation (-2.17% on North and -2.97% on East). Our EDA "
        "revealed that raw USBL acoustic coordinates suffer from discrete position jumps caused by multipath acoustic reflections in water. "
        "Because neural networks attempt to minimize squared error across all frequencies, the network partially overfits to high-frequency "
        "measurement noise rather than true vehicle motion."
    )

    add_callout(
        doc,
        "Phase 1 Recommendation: Kalman Filter Preprocessing",
        "To resolve acoustic USBL position overfitting, an Extended Kalman Filter (EKF) or Rauch-Tung-Striebel (RTS) smoother must be applied "
        "to raw Latitude and Longitude before model training. This will fuse high-rate DVL ground velocities with low-rate acoustic fixes, "
        "eliminating non-physical coordinate jumps."
    )

    # -------------------------------------------------------------
    # SECTION 8: CONCLUSION & TRANSITION TO PHASE 2 (MPC)
    # -------------------------------------------------------------
    add_heading_1(doc, "8. Conclusion & Transition to Phase 2 (MPC)")
    add_body_paragraph(
        doc,
        "Phase 1 successfully established and benchmarked a learning-based dynamic model for the Autosub6000 AUV. By combining reduced-order "
        "physics with a 2-Layer Recurrent GRU residual predictor, we achieved superior attitude and velocity accuracy while preserving physical "
        "stability during multi-step forecasts."
    )

    add_heading_2(doc, "8.1 Phase 2 Implementation Roadmap")
    add_bullet_point(doc, "Formulate a receding-horizon optimization problem using the validated Hybrid GRU model to predict internal dynamics.", "Model Predictive Control (MPC): ")
    add_bullet_point(doc, "Benchmark AI-enhanced MPC against standard PID feedback control on complex 3D sinusoidal and lawnmower paths.", "Benchmark vs PID: ")
    add_bullet_point(doc, "Deploy the FastAPI backend and interactive frontend dashboard for real-time operator visualization and telemetry injection.", "Software Deployment: ")

    # Save document
    out_docx_path = os.path.join(reports_dir, "AUV_Phase1_Presentation_Report.docx")
    doc.save(out_docx_path)
    print(f"\nSUCCESS: Word document report generated: {out_docx_path}")

    # Mirror to workspace root
    root_docx_path = os.path.join(os.path.dirname(base_dir), "AUV_Phase1_Presentation_Report.docx")
    doc.save(root_docx_path)
    print(f"Mirrored to workspace root: {root_docx_path}")

if __name__ == "__main__":
    main()
