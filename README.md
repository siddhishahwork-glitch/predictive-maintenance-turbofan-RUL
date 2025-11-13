# predictive-maintenance-turbofan-RUL
Developed Remaining Useful Life (RUL) prediction model for turbofan engines using multivariate sensor data. 
This project develops a Remaining Useful Life (RUL) prediction model for turbofan jet engines using the NASA CMAPSS FD001 dataset.
The goal is to forecast how many operating cycles remain before an engine fails, enabling proactive maintenance and improved reliability.

The final model uses a leakage-free LSTM neural network, achieving:

📉 RMSE: 23.14 cycles

📉 MAE: 16.28 cycles



├── predictive_maintenance_lstm.py   
├── requirements.txt                    # Python dependencies
├── README.md                           # Project overview (this file)
└── Dataset cannot be redistributed, so it is excluded from this repo.

Methodology:
Step 1: Data Cleaning & Sorting
Load sensor and operational settings
Sort by engine_id and cycle

Step 2: Add RUL Labels
max_cycle_per_engine = df.groupby('engine_id')['cycle'].transform('max')
df['RUL'] = max_cycle_per_engine - df['cycle']

Step 3: 
Engines are split, not rows:
80% engines → training
20% engines → validation

Step 4: Feature Scaling
StandardScaler applied only on training data.

Step 5: Create Time-Series Sequences
LSTM input windows:
Window size: 30 cycles
Each window → RUL of last timestep

Step 6: LSTM Model
Two stacked LSTM layers + dropout + dense layers.

Step 7: Model Evaluation
Metrics:
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)


LSTM Architecture:
Input: 30 timesteps × N features
↓
LSTM(64, return_sequences=True)
↓
Dropout(0.2)
↓
LSTM(32)
↓
Dropout(0.2)
↓
Dense(16, activation='relu')
↓
Dense(1)  # RUL output

Optimized with:
Adam optimizer (lr = 0.001)
EarlyStopping (patience 5)

Final Model Performance (FD001)
RMSE	23.14 cycles
MAE	16.28 cycles


Interpretation:

Predictions are within ~20–30 cycles error band
Error decreases as the engine approaches failure
Behaviour matches academic benchmarks for CMAPSS FD001
Suitable for reliability decision-making and health monitoring

Technologies Used:

TensorFlow / Keras — LSTM deep learning
NumPy & Pandas — data processing
Matplotlib — result visualization
Scikit-Learn — scaling & metrics
Python 3.11

Key Engineering Skills Demonstrated:
Predictive Maintenance
Reliability Engineering
Time-Series Modelling
Deep Learning (LSTM)
Feature Engineering
Information Leakage Prevention
Model Evaluation (RMSE/MAE)
Clean, reproducible ML pipeline

Future Improvements:

Try Bi-LSTM or GRU networks
Add attention mechanism
Perform hyperparameter optimization
Combine classical ML (XGBoost) with deep learning
Use FD002–FD004 for multi-condition modelling
