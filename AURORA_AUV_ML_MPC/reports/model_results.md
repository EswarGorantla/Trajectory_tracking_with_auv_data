# Model Evaluation & Predictor Results

### One-Step-Ahead Predictor Test RMSE
| State Variable     |   Physics-only |   Physics+Linear |   Physics+Forest |   Physics+MLP |   Physics+LSTM |   Physics+GRU |
|:-------------------|---------------:|-----------------:|-----------------:|--------------:|---------------:|--------------:|
| North Position (m) |      0.711185  |        1.11957   |        0.988244  |     1.92724   |      1.00589   |     1.01729   |
| East Position (m)  |      0.692716  |        0.696698  |        0.881288  |     0.714063  |      0.727583  |     0.700045  |
| Depth (m)          |      0.122318  |        0.114986  |        0.105644  |     0.114098  |      0.103571  |     0.104927  |
| Roll (rad)         |      0.0769959 |        0.0739103 |        0.0740158 |     0.0892747 |      0.0728697 |     0.0724379 |
| Pitch (rad)        |      0.0453854 |        0.0433526 |        0.0386559 |     0.0387994 |      0.0380332 |     0.0375599 |
| Heading (rad)      |      0.0571978 |        0.0576066 |        0.0540833 |     0.0990315 |      0.0492535 |     0.0484137 |
| Speed (m/s)        |      0.227555  |        0.226025  |        0.203524  |     0.237422  |      0.186187  |     0.172196  |

### Key Observations
* The GRU model is the best one-step-ahead predictor.
* Position coordinates (North/East) increase in error when neural networks attempt to learn residuals. This is due to noisy transpond Updates (USBL transceiver noise) in raw Latitude and Longitude.
