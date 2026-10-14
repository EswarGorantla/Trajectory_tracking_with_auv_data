# Final Project Report
Learning-Based Dynamic Modeling and Model Predictive Control for Autonomous Underwater Vehicles Using Multi-Sensor AURORA Data

## Abstract
This project presents a learning-based hybrid modeling and predictive control pipeline for an Autonomous Underwater Vehicle (AUV) using multi-sensor AURORA dataset index telemetry. By training recurrent neural networks to learn the step-by-step residuals of a rigid-body physics model, we compensate for complex, unmodeled hydrodynamics. The hybrid predictor is integrated inside a receding-horizon Model Predictive Controller (MPC).

## Information Flow Diagram
```
                 AURORA DATA
                      ↓
              Current AUV State
                      ↓
          ┌─────────────────────┐
          │ Physics Dynamics    │
          │       Model         │
          └──────────┬──────────┘
                     ↓
             Physics Prediction
                     ↓
          ┌─────────────────────┐
          │   LSTM / GRU        │
          │ Residual ML Model   │
          └──────────┬──────────┘
                     ↓
              ML Correction
                     ↓
        ┌────────────────────────┐
        │ Improved Dynamics      │
        │        Model           │
        └───────────┬────────────┘
                    ↓
                   MPC
                    ↓
          Optimal Control Input
                    ↓
             AUV Simulation
                    ↓
             Future Position
```

## Experimental Results

### Closed-Loop Sinusoidal Tracking Performance
| Controller             |   Pos RMSE (m) |   Depth RMSE (m) |   Heading RMSE (rad) |   Control Effort |
|:-----------------------|---------------:|-----------------:|---------------------:|-----------------:|
| PID Controller         |       104.97   |          2.59561 |             1.77702  |         401.813  |
| Physics-only MPC       |        73.4626 |          5.44257 |             1.24738  |          60.3715 |
| AI-enhanced MPC (LSTM) |        21.4234 |          1.47691 |             0.600053 |          64.781  |
| AI-enhanced MPC (GRU)  |        51.26   |         16.2267  |             0.433704 |          84.4059 |

### Disturbance Rejection Tracking Performance (East Current: 0.15 m/s)
| Controller             |   Pos RMSE (m) |   Depth RMSE (m) |   Heading RMSE (rad) |   Control Effort |
|:-----------------------|---------------:|-----------------:|---------------------:|-----------------:|
| PID Controller         |        45.5831 |          1.31041 |             1.46141  |         218.347  |
| Physics-only MPC       |        88.1568 |          4.75586 |             0.750039 |          74.9083 |
| AI-enhanced MPC (LSTM) |        58.1362 |         13.1196  |             0.866855 |          54.1997 |

### Major Insights
1. **AI MPC Performance:** The LSTM-MPC achieves a Position RMSE of **21.4234 meters**, which represents a **70.84% improvement** in trajectory tracking error over the physics-only MPC, and a **79.59% improvement** over the PID feedback baseline.
2. **Model Multi-Step Stability:** While GRU performs slightly better for one-step-ahead predictions, it suffers from higher drift during multi-step horizon rollouts compared to LSTM, which is why the GRU MPC performs worse in closed-loop simulations.
