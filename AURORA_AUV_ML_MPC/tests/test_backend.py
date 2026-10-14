import os
import sys
import unittest
import asyncio
import httpx

# Ensure root directory is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.main import app
from backend.model_service import model_service

class TestAURORABackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force model loading
        model_service.load_all()

    def _run(self, coro):
        return asyncio.run(coro)

    def test_01_health_endpoint(self):
        """Test GET /health returns healthy status and loaded models."""
        async def _test():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/health")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "healthy")
                self.assertTrue(data["model_loaded"])
                self.assertIn("LSTM", data["available_models"])
                self.assertIn("GRU", data["available_models"])
                self.assertEqual(data["sequence_length"], 10)
        self._run(_test())

    def test_02_sample_endpoint(self):
        """Test GET /sample returns real test telemetry and 10-step history."""
        async def _test():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/sample")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn("current_state", data)
                self.assertIn("commands", data)
                self.assertIn("sequence_history", data)
                self.assertEqual(len(data["sequence_history"]), 10)
                self.assertIn("actual_next_state", data)
        self._run(_test())

    def test_03_predict_endpoint_lstm(self):
        """Test POST /predict with LSTM model."""
        async def _test():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                sample_res = (await client.get("/sample")).json()
                payload = {
                    "current_state": sample_res["current_state"],
                    "commands": sample_res["commands"],
                    "sequence_history": sample_res["sequence_history"],
                    "model_name": "LSTM"
                }
                response = await client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["success"])
                self.assertEqual(data["model_used"], "LSTM")
                self.assertIn("physics_prediction", data)
                self.assertIn("ai_residual_correction", data)
                self.assertIn("final_prediction", data)
                
                final_state = data["final_prediction"]
                self.assertFalse(any(v is None for v in final_state.values()))
        self._run(_test())

    def test_04_predict_endpoint_gru(self):
        """Test POST /predict with GRU model."""
        async def _test():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                sample_res = (await client.get("/sample")).json()
                payload = {
                    "current_state": sample_res["current_state"],
                    "commands": sample_res["commands"],
                    "sequence_history": sample_res["sequence_history"],
                    "model_name": "GRU"
                }
                response = await client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["success"])
                self.assertEqual(data["model_used"], "GRU")
        self._run(_test())

    def test_05_predict_sequence_endpoint(self):
        """Test POST /predict-sequence multi-step forecasting."""
        async def _test():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                payload = {
                    "start_index": 100,
                    "steps": 15,
                    "model_name": "LSTM"
                }
                response = await client.post("/predict-sequence", json=payload)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["success"])
                self.assertEqual(len(data["actual_trajectory"]), 16)
                self.assertEqual(len(data["physics_trajectory"]), 16)
                self.assertEqual(len(data["ai_trajectory"]), 16)
                self.assertIn("rmse_physics", data)
                self.assertIn("rmse_ai", data)
        self._run(_test())

    def test_06_mpc_endpoint(self):
        """Test POST /mpc closed-loop control simulation."""
        async def _test():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                payload = {
                    "trajectory_type": "3d_sinusoid",
                    "simulation_steps": 25,
                    "enable_disturbance": False,
                    "ai_model_name": "LSTM"
                }
                response = await client.post("/mpc", json=payload)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["success"])
                self.assertIn("desired_trajectory", data)
                self.assertIn("pid_trajectory", data)
                self.assertIn("physics_mpc_trajectory", data)
                self.assertIn("ai_mpc_trajectory", data)
                self.assertIn("metrics", data)
                self.assertIn("PID", data["metrics"])
                self.assertIn("Physics_MPC", data["metrics"])
                self.assertIn("AI_MPC", data["metrics"])
        self._run(_test())

if __name__ == "__main__":
    unittest.main()
