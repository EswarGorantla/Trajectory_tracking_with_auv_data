from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class AUVState(BaseModel):
    north: float = Field(..., description="North Cartesian displacement (m)")
    east: float = Field(..., description="East Cartesian displacement (m)")
    depth: float = Field(..., description="Depth below surface (m, positive-down)")
    roll: float = Field(0.0, description="Roll angle (rad)")
    pitch: float = Field(0.0, description="Pitch angle (rad, positive nose-up)")
    heading: float = Field(0.0, description="Heading angle clockwise from North (rad)")
    speed: float = Field(1.2, description="Surge forward ground speed (m/s)")

class ControlInput(BaseModel):
    u_cmd: float = Field(1.2, description="Commanded surge speed (m/s)")
    theta_cmd: float = Field(0.0, description="Commanded pitch angle (rad)")
    psi_cmd: float = Field(0.0, description="Commanded heading angle (rad)")

class PredictRequest(BaseModel):
    current_state: AUVState
    commands: ControlInput
    sequence_history: Optional[List[AUVState]] = Field(None, description="Optional 10 previous states. If omitted, synthesized from current_state.")
    model_name: str = Field("LSTM", description="Model architecture: 'LSTM', 'GRU', 'MLP', 'Linear', 'Random Forest'")

class PredictResponse(BaseModel):
    success: bool
    physics_prediction: AUVState
    ai_residual_correction: Dict[str, float]
    final_prediction: AUVState
    model_used: str
    prediction_horizon: int = 1
    execution_time_ms: float

class PredictSequenceRequest(BaseModel):
    start_index: int = Field(100, description="Start index in test dataset")
    steps: int = Field(20, description="Number of forecasting steps")
    model_name: str = Field("LSTM", description="Model to use")

class PredictSequenceResponse(BaseModel):
    success: bool
    timesteps: List[float]
    actual_trajectory: List[Dict[str, float]]
    physics_trajectory: List[Dict[str, float]]
    ai_trajectory: List[Dict[str, float]]
    rmse_physics: Dict[str, float]
    rmse_ai: Dict[str, float]

class MPCRequest(BaseModel):
    initial_state: Optional[AUVState] = None
    trajectory_type: str = Field("3d_sinusoid", description="'3d_sinusoid', 'step_waypoint', or 'straight_dive'")
    simulation_steps: int = Field(50, description="Number of closed-loop simulation steps (2.0s per step)")
    enable_disturbance: bool = Field(False, description="Enable simulated ocean current disturbance")
    disturbance_current_east: float = Field(0.15, description="Eastward current disturbance (m/s)")
    ai_model_name: str = Field("LSTM", description="AI model to use in MPC loop: 'LSTM' or 'GRU'")

class MPCResponse(BaseModel):
    success: bool
    trajectory_type: str
    desired_trajectory: List[Dict[str, float]]
    pid_trajectory: List[Dict[str, float]]
    physics_mpc_trajectory: List[Dict[str, float]]
    ai_mpc_trajectory: List[Dict[str, float]]
    metrics: Dict[str, Dict[str, float]]
    execution_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    available_models: List[str]
    device: str
    sequence_length: int
    dataset: str
    identified_drag_kd: float

class SampleResponse(BaseModel):
    sample_index: int
    current_state: AUVState
    commands: ControlInput
    sequence_history: List[AUVState]
    actual_next_state: AUVState
