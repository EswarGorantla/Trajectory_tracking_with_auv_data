import os
import io
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.schemas import (
    HealthResponse, SampleResponse,
    PredictRequest, PredictResponse,
    PredictSequenceRequest, PredictSequenceResponse,
    MPCRequest, MPCResponse
)
from backend.model_service import model_service
from backend.prediction_service import prediction_service
from backend.mpc_service import mpc_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all models, scalers, and configs into memory on startup."""
    try:
        model_service.load_all()
    except Exception as e:
        print(f"Error loading models during startup: {e}")
    yield

app = FastAPI(
    title="AURORA AUV ML-MPC Dynamics API",
    description="Learning-Based Dynamic Modeling & Model Predictive Control for Autonomous Underwater Vehicles",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns model health, device status, and configuration parameters."""
    return HealthResponse(
        status="healthy" if model_service.loaded else "initializing",
        model_loaded=model_service.loaded,
        model_type=model_service.config.get("model_type", "LSTM"),
        available_models=list(model_service.models.keys()),
        device=str(model_service.device).upper(),
        sequence_length=model_service.config.get("sequence_length", 10),
        dataset=model_service.config.get("dataset", "AURORA Autosub6000 M86"),
        identified_drag_kd=float(model_service.config.get("drag_coefficient_kd", 0.022498))
    )

@app.get("/sample", response_model=SampleResponse, tags=["Data"])
async def get_test_sample(index: int = Query(None, description="Optional row index in test set")):
    """Fetches a real state sequence and actual next state from the processed AURORA test dataset."""
    try:
        sample = model_service.get_sample(index)
        return SampleResponse(**sample)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse, tags=["Dynamics Prediction"])
async def predict_next_state(request: PredictRequest):
    """
    Predicts the future AUV state (at t+1) from current state and sequence history.
    Returns Physics Prediction, AI Residual Correction, and Final Improved Prediction.
    """
    try:
        return prediction_service.predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-sequence", response_model=PredictSequenceResponse, tags=["Dynamics Prediction"])
async def predict_trajectory_sequence(request: PredictSequenceRequest):
    """Runs a multi-step rolling forecast over a real AURORA test segment."""
    try:
        return prediction_service.predict_sequence(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence forecast error: {str(e)}")

@app.post("/upload-csv", tags=["Batch Inference"])
async def upload_csv_prediction(
    file: UploadFile = File(...),
    model_name: str = Query("LSTM", description="AI model to use: 'LSTM', 'GRU', or 'MLP'")
):
    """
    Accepts any uploaded CSV file containing AUV columns (North, East, depth, Speed, Pitch, Roll, Heading).
    Executes the trained Hybrid Physics-ML model on all rows and returns the predicted future trajectory.
    """
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
        return prediction_service.predict_dataframe(df, model_name=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")

@app.post("/mpc", response_model=MPCResponse, tags=["Trajectory Control"])
async def run_mpc_control(request: MPCRequest):
    """
    Executes a closed-loop trajectory tracking simulation comparing PID, Physics MPC, and AI-Enhanced MPC.
    """
    try:
        return mpc_service.run_mpc_experiment(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MPC simulation error: {str(e)}")

# Mount static frontend assets
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serves the main single-page web dashboard."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Please place index.html in frontend/"}
