import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Setup path to Phase_2
app_dir = os.path.dirname(os.path.abspath(__file__))
phase2_dir = os.path.dirname(app_dir)
root_dir = os.path.dirname(phase2_dir)
if phase2_dir not in sys.path:
    sys.path.insert(0, phase2_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from src.simulation_runner import ClosedLoopSimulation
from src.scenarios import get_canonical_scenarios
from src.state import AUVState, wrap_angle
from app.report_generator import generate_markdown_report, generate_html_report, generate_csv_report, generate_proof_data

app = FastAPI(
    title="Autosub6000 Goal-Directed MPC Mission Terminal",
    description="Research-Grade Receding-Horizon Control Powered by Phase 1 Deep GRU Residual Plant",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(app_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Request Schemas
class Position3D(BaseModel):
    north: float = 0.0
    east: float = 0.0
    depth: float = 20.0

class GoalConfig(BaseModel):
    north: float = 100.0
    east: float = 80.0
    depth: float = 40.0
    tolerance: float = 5.0

class BoundaryConfig(BaseModel):
    north_min: float = 0.0
    north_max: float = 120.0
    east_min: float = 0.0
    east_max: float = 100.0
    depth_min: float = 10.0
    depth_max: float = 60.0

class ObstacleItem(BaseModel):
    north: float = 50.0
    east: float = 40.0
    radius: float = 8.0
    margin: float = 4.0
    depth_min: float = 0.0
    depth_max: float = 80.0

class SimulationRequest(BaseModel):
    start: Position3D = Position3D(north=0.0, east=0.0, depth=20.0)
    goal: GoalConfig = GoalConfig(north=100.0, east=80.0, depth=40.0, tolerance=5.0)
    boundaries: BoundaryConfig = BoundaryConfig()
    obstacles: List[ObstacleItem] = []
    method: str = "hybrid_mpc" # hybrid_mpc, physics_mpc, gru_mpc, potential_field, reactive_los
    horizon: int = 10
    num_candidates: int = 30 # Default 30 for fast interactive response, 70 for nominal
    iterations: int = 3
    max_steps: int = 75
    current_velocity: List[float] = [0.0, 0.0, 0.0]
    use_kalman_filter: bool = True
    randomize_plant_mismatch: bool = False
    usbl_sigma: float = 0.50
    seed: int = 1000
    return_to_start: bool = False

@app.get("/")
def get_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Autosub6000 MPC Mission Control</h1><p>Static index.html is being prepared...</p>")

@app.get("/api/scenarios")
def get_scenarios():
    """Returns canonical benchmark scenarios for 1-click loading."""
    scens = get_canonical_scenarios()
    output = {}
    for k, v in scens.items():
        output[k] = {
            "name": v["name"],
            "desc": v.get("desc", ""),
            "start": v["start"],
            "goal": v["goal"],
            "boundaries": v["boundaries"],
            "obstacles": v["obstacles"],
            "current": v.get("current", [0.0, 0.0, 0.0]),
            "return_to_start": v.get("return_to_start", False)
        }
    return output

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    """
    Executes research-grade closed-loop AUV mission using decoupled continuous-time true plant
    and the Phase 1 hybrid neural dynamics model.
    """
    try:
        pad = 12.0
        b_n_min = min(req.boundaries.north_min, req.start.north - pad, req.goal.north - pad)
        b_n_max = max(req.boundaries.north_max, req.start.north + pad, req.goal.north + pad)
        b_e_min = min(req.boundaries.east_min, req.start.east - pad, req.goal.east - pad)
        b_e_max = max(req.boundaries.east_max, req.start.east + pad, req.goal.east + pad)

        scenario_dict = {
            "name": "interactive_mission",
            "start": [req.start.north, req.start.east, req.start.depth],
            "goal": [req.goal.north, req.goal.east, req.goal.depth],
            "boundaries": {
                "north_min": b_n_min,
                "north_max": b_n_max,
                "east_min": b_e_min,
                "east_max": b_e_max,
                "depth_min": req.boundaries.depth_min,
                "depth_max": req.boundaries.depth_max,
            },
            "obstacles": [
                {
                    "north": o.north,
                    "east": o.east,
                    "radius": o.radius,
                    "margin": o.margin,
                    "depth_min": o.depth_min,
                    "depth_max": o.depth_max
                }
                for o in req.obstacles
            ],
            "current": req.current_velocity
        }

        # Step budget: calculate dynamically to guarantee sufficient steps for round trip A -> B -> A
        if req.return_to_start:
            dist_ab = np.linalg.norm(np.array([req.goal.north - req.start.north, req.goal.east - req.start.east, req.goal.depth - req.start.depth]))
            # 2.0s per step, nominal speed 1.2 m/s -> ~2.4m/step. Double for round trip + turns & safety margin
            needed_steps = int((2.0 * dist_ab / 2.4) * 1.4) + 40
            effective_max_steps = max(req.max_steps * 2, needed_steps, 200)
        else:
            effective_max_steps = req.max_steps

        # Initialize decoupled closed loop simulation harness
        sim = ClosedLoopSimulation(
            method=req.method,
            scenario_config=scenario_dict,
            seed=req.seed,
            dt_control=2.0,
            max_steps=effective_max_steps,
            goal_tolerance_primary=req.goal.tolerance,
            goal_tolerance_tight=2.0,
            horizon=req.horizon,
            num_candidates=req.num_candidates,
            iterations=req.iterations,
            use_warm_start=True,
            use_los_seed=True,
            use_kalman_filter=req.use_kalman_filter,
            usbl_sigma=req.usbl_sigma,
            current_velocity=np.array(req.current_velocity, dtype=np.float64),
            randomize_plant_mismatch=req.randomize_plant_mismatch,
            return_to_start=req.return_to_start
        )

        df_traj, metrics = sim.run()

        def safe_num(v, default=0.0):
            if v is None or pd.isna(v) or np.isnan(v) or np.isinf(v):
                return default
            return float(v)

        # Format trajectory records
        traj_records = []
        for idx, r in df_traj.iterrows():
            pt = {
                "step": int(r["step"]),
                "time_sec": safe_num(r["time_sec"]),
                "phase": str(r.get("phase", "FORWARD")),
                "leg": int(r.get("leg", 1)),
                "true_north": safe_num(r["true_north"]),
                "true_east": safe_num(r["true_east"]),
                "true_depth": safe_num(r["true_depth"]),
                "true_roll": safe_num(r["true_roll"]),
                "true_pitch": safe_num(r["true_pitch"]),
                "true_heading": safe_num(r["true_heading"]),
                "true_speed": safe_num(r["true_speed"], 1.2),
                "meas_north": safe_num(r["meas_north"]),
                "meas_east": safe_num(r["meas_east"]),
                "meas_depth": safe_num(r["meas_depth"]),
                "meas_heading": safe_num(r["meas_heading"]),
                "meas_speed": safe_num(r["meas_speed"], 1.2),
                "cmd_speed": safe_num(r["cmd_speed"]) if not pd.isna(r["cmd_speed"]) else None,
                "cmd_pitch": safe_num(r["cmd_pitch"]) if not pd.isna(r["cmd_pitch"]) else None,
                "cmd_heading": safe_num(r["cmd_heading"]) if not pd.isna(r["cmd_heading"]) else None,
                "target_north": safe_num(r.get("target_north", req.goal.north)),
                "target_east": safe_num(r.get("target_east", req.goal.east)),
                "target_depth": safe_num(r.get("target_depth", req.goal.depth)),
                "dist_to_target": safe_num(r.get("dist_to_target", r["dist_to_goal"])),
                "dist_to_goal": safe_num(r["dist_to_goal"]),
                "dist_to_b": safe_num(r.get("dist_to_b", r["dist_to_goal"])),
                "dist_to_a": safe_num(r.get("dist_to_a", 0.0)),
                "reached_b": bool(r.get("reached_b", False)),
                "reached_a": bool(r.get("reached_a", False)),
                "min_obstacle_dist": safe_num(r["min_obstacle_dist"], 999.0) if not np.isinf(r["min_obstacle_dist"]) else 999.0,
                "collision": bool(r["collision"]),
                "buffer_violation": bool(r["buffer_violation"]),
                "boundary_violation_m": safe_num(r["boundary_violation_m"]),
                "is_in_bounds": bool(r["is_in_bounds"]),
                "ood_flag": bool(r["ood_flag"]),
                "solve_time_sec": safe_num(r["solve_time_sec"]),
                "cost_j_goal": safe_num(r.get("cost_j_goal", 0.0)),
                "cost_j_tracking": safe_num(r.get("cost_j_tracking", 0.0)),
                "cost_j_control": safe_num(r.get("cost_j_control", 0.0)),
                "cost_j_smooth": safe_num(r.get("cost_j_smooth", 0.0)),
                "cost_j_constraint": safe_num(r.get("cost_j_constraint", 0.0)),
                "cost_total_cost": safe_num(r.get("cost_total_cost", 0.0)),
                "predicted_traj": r["predicted_traj"] if r["predicted_traj"] is not None else []
            }
            traj_records.append(pt)

        # Sanitize metrics
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (bool, np.bool_)):
                clean_metrics[k] = bool(v)
            elif isinstance(v, (float, np.floating)):
                clean_metrics[k] = 0.0 if np.isnan(v) or np.isinf(v) else float(v)
            elif isinstance(v, (int, np.integer)):
                clean_metrics[k] = int(v)
            elif isinstance(v, (list, tuple)):
                clean_metrics[k] = list(v)
            elif isinstance(v, str):
                clean_metrics[k] = str(v)
            elif v is None:
                clean_metrics[k] = None

        # Automatically generate the exhaustive mathematical calculation report and save it to disk
        results_dir = os.path.join(phase2_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Automatically generate all report formats (.md, .txt, .csv, .html, .json)
        report_md = generate_markdown_report(clean_metrics, scenario_dict, traj_records)
        proof_data = generate_proof_data(clean_metrics, scenario_dict, traj_records)
        report_csv = generate_csv_report(traj_records)
        report_html = generate_html_report(clean_metrics, scenario_dict, traj_records)
        report_json = json.dumps({"metrics": clean_metrics, "scenario": scenario_dict, "trajectory": traj_records}, indent=2)

        try:
            with open(os.path.join(results_dir, "mission_calculation_report.md"), "w", encoding="utf-8") as f:
                f.write(report_md)
            with open(os.path.join(results_dir, "mission_calculation_report.txt"), "w", encoding="utf-8") as f:
                f.write(report_md)
            with open(os.path.join(results_dir, "mission_telemetry_calculations.csv"), "w", encoding="utf-8") as f:
                f.write(report_csv)
            with open(os.path.join(results_dir, "mission_calculation_report.html"), "w", encoding="utf-8") as f:
                f.write(report_html)
            with open(os.path.join(results_dir, "autosub6000_mission_metrics.json"), "w", encoding="utf-8") as f:
                f.write(report_json)
        except Exception as err_save:
            print(f"Warning: Failed to save local reports: {err_save}")

        return {
            "success": True,
            "metrics": clean_metrics,
            "trajectory": traj_records,
            "scenario": scenario_dict,
            "report_markdown": report_md,
            "proof_data": proof_data,
            "report_download_url": "/api/download_report_md"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ReportRequest(BaseModel):
    metrics: Dict[str, Any]
    scenario: Dict[str, Any]
    trajectory_summary: List[Dict[str, Any]]

@app.get("/api/download_report_md")
@app.get("/api/download_calculation_file")
def download_report_md():
    """Returns the calculation report directly as Markdown (.md)."""
    results_dir = os.path.join(phase2_dir, "results")
    p = os.path.join(results_dir, "mission_calculation_report.md")
    if os.path.exists(p):
        return FileResponse(
            p,
            filename="mission_calculation_report.md",
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="mission_calculation_report.md"'}
        )
    raise HTTPException(status_code=404, detail="No calculation report generated yet. Run a simulation first.")

@app.get("/api/download_report_txt")
def download_report_txt():
    """Returns the calculation report directly as Plain Text (.txt) that opens in Notepad."""
    results_dir = os.path.join(phase2_dir, "results")
    p = os.path.join(results_dir, "mission_calculation_report.txt")
    if not os.path.exists(p):
        p_md = os.path.join(results_dir, "mission_calculation_report.md")
        if os.path.exists(p_md):
            with open(p_md, "r", encoding="utf-8") as f_in, open(p, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
    if os.path.exists(p):
        return FileResponse(
            p,
            filename="mission_calculation_report.txt",
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="mission_calculation_report.txt"'}
        )
    raise HTTPException(status_code=404, detail="No calculation text file generated yet. Run a simulation first.")

@app.get("/api/download_telemetry_csv")
def download_telemetry_csv_file():
    """Returns the step-by-step numeric telemetry directly as CSV (.csv) that opens in Excel."""
    results_dir = os.path.join(phase2_dir, "results")
    p = os.path.join(results_dir, "mission_telemetry_calculations.csv")
    if os.path.exists(p):
        return FileResponse(
            p,
            filename="mission_telemetry_calculations.csv",
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="mission_telemetry_calculations.csv"'}
        )
    raise HTTPException(status_code=404, detail="No telemetry CSV generated yet. Run a simulation first.")

@app.get("/api/download_report_html")
def download_report_html_file():
    """Returns the rich publication report directly as HTML (.html)."""
    results_dir = os.path.join(phase2_dir, "results")
    p = os.path.join(results_dir, "mission_calculation_report.html")
    if os.path.exists(p):
        return FileResponse(
            p,
            filename="mission_calculation_report.html",
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="mission_calculation_report.html"'}
        )
    raise HTTPException(status_code=404, detail="No HTML report generated yet. Run a simulation first.")

@app.get("/api/download_metrics_json")
def download_metrics_json_file():
    """Returns the mission performance metrics directly as JSON (.json)."""
    results_dir = os.path.join(phase2_dir, "results")
    p = os.path.join(results_dir, "autosub6000_mission_metrics.json")
    if os.path.exists(p):
        return FileResponse(
            p,
            filename="autosub6000_mission_metrics.json",
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="autosub6000_mission_metrics.json"'}
        )
    raise HTTPException(status_code=404, detail="No metrics JSON generated yet. Run a simulation first.")

@app.post("/api/export_report")
def export_mission_report(req: ReportRequest):
    """
    Generates an exhaustive, downloadable mission calculation document in Markdown format
    with complete mathematical equations, cost weights, performance indices, and all step records.
    """
    report_md = generate_markdown_report(req.metrics, req.scenario, req.trajectory_summary)
    return Response(
        content=report_md,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=mission_calculation_report.md"}
    )

@app.post("/api/export_html_report")
def export_mission_html_report(req: ReportRequest):
    """
    Generates a publication-grade HTML report with interactive tables, progress bars,
    and printable PDF formatting.
    """
    report_html = generate_html_report(req.metrics, req.scenario, req.trajectory_summary)
    return Response(
        content=report_html,
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=mission_calculation_report.html"}
    )

@app.post("/api/export_csv_report")
def export_mission_csv_report(req: ReportRequest):
    """
    Generates a comprehensive CSV log containing all numeric calculations, states,
    costs, and actuator commands for every step.
    """
    report_csv = generate_csv_report(req.trajectory_summary)
    return Response(
        content=report_csv,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=mission_telemetry_calculations.csv"}
    )

@app.get("/api/download_research_report")
def download_research_report():
    p = os.path.join(phase2_dir, "PHASE2_RESEARCH_REPORT.md")
    if os.path.exists(p):
        return FileResponse(p, filename="PHASE2_RESEARCH_REPORT.md", media_type="text/markdown")
    raise HTTPException(status_code=404, detail="PHASE2_RESEARCH_REPORT.md not found.")

@app.get("/api/download_results_summary")
def download_results_summary():
    p = os.path.join(phase2_dir, "RESULTS_SUMMARY.md")
    if os.path.exists(p):
        return FileResponse(p, filename="RESULTS_SUMMARY.md", media_type="text/markdown")
    raise HTTPException(status_code=404, detail="RESULTS_SUMMARY.md not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
