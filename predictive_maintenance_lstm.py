import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ----------------------------------------------------------
# 0. REPRODUCIBILITY
# ----------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------------------------------------
# 1. LOAD NASA CMAPSS FD001 DATA
# ----------------------------------------------------------
def load_data():
    path = "archive/train_FD001.txt"

    cols = ["engine_id", "cycle",
            "op_setting_1", "op_setting_2", "op_setting_3"]
    cols += [f"sensor_{i}" for i in range(1, 22)]

    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------
# 2. ADD RUL LABEL
# ----------------------------------------------------------
def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["max_cycle"] = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


# ----------------------------------------------------------
# 3. SPLIT BY ENGINE 
# ----------------------------------------------------------
def split_by_engine(df: pd.DataFrame,
                    test_size: float = 0.2,
                    random_state: int = 42):
    engines = df["engine_id"].unique()
    train_eng, val_eng = train_test_split(
        engines, test_size=test_size, random_state=random_state
    )
    train_df = df[df.engine_id.isin(train_eng)].copy()
    val_df = df[df.engine_id.isin(val_eng)].copy()
    return train_df, val_df


# ----------------------------------------------------------
# 4. CREATE SEQUENCE WINDOWS FOR LSTM
# ----------------------------------------------------------
def create_sequences(df: pd.DataFrame,
                     feature_cols,
                     window_size: int = 30):
    """
    For each engine:
      - take sliding windows of length `window_size`
      - X = features for these window_size timesteps
      - y = RUL at the last timestep in the window
    """
    X_list = []
    y_list = []

    for eng_id, eng_df in df.groupby("engine_id"):
        eng_df = eng_df.sort_values("cycle")
        values = eng_df[feature_cols].values
        rul = eng_df["RUL"].values

        if len(eng_df) < window_size:
            continue

        for start in range(0, len(eng_df) - window_size + 1):
            end = start + window_size
            X_list.append(values[start:end, :])
            y_list.append(rul[end - 1])  # RUL at last step

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


# ----------------------------------------------------------
# 5. BUILD LSTM MODEL
# ----------------------------------------------------------
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)  # RUL
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ----------------------------------------------------------
# 6. PIPELINE
# ----------------------------------------------------------
def main():
    set_seed(42)

    print("Loading data...")
    df = load_data()
    print("Raw shape:", df.shape)

    print("Adding RUL labels...")
    df = add_rul(df)

    print("Splitting by engine (no leakage)...")
    train_df, val_df = split_by_engine(df, test_size=0.2, random_state=42)
    print("Train engines:", train_df.engine_id.nunique())
    print("Val engines  :", val_df.engine_id.nunique())

    # ------- Feature selection -------
    # Drop low-information sensors
    drop_sensors = [
        "sensor_1", "sensor_5", "sensor_6",
        "sensor_10", "sensor_16", "sensor_18", "sensor_19"
    ]
    feature_cols = [
        c for c in df.columns
        if c not in ["engine_id", "cycle", "RUL"] + drop_sensors + ["max_cycle"]
    ]

    print("Using features:", feature_cols)

    # ------- Scaling -------
    scaler = StandardScaler()

    # Fit scaler on TRAIN ONLY (flattened) to avoid leakage
    train_features_flat = train_df[feature_cols].values
    scaler.fit(train_features_flat)

    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()

    train_df_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df_scaled[feature_cols] = scaler.transform(val_df[feature_cols])

    # ------- Sequence creation -------
    window_size = 30
    print(f"Creating sequences with window size = {window_size}...")

    X_train, y_train = create_sequences(train_df_scaled, feature_cols, window_size)
    X_val, y_val = create_sequences(val_df_scaled, feature_cols, window_size)

    print("Train sequences:", X_train.shape, "Val sequences:", X_val.shape)

    # ------- Build & train LSTM -------
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    # ------- Evaluate -------
    y_pred = model.predict(X_val).flatten()
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    print("\n=== LSTM MODEL PERFORMANCE (NO LEAKAGE) ===")
    print(f"Validation RMSE: {rmse:.2f} cycles")
    print(f"Validation MAE : {mae:.2f} cycles\n")

    # ------- Plot True vs Predicted -------
    plt.figure()
    plt.scatter(y_val, y_pred, alpha=0.4)
    plt.xlabel("True RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title("True vs Predicted RUL – LSTM (No-Leakage)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
