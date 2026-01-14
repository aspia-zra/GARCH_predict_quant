import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from arch import arch_model
from joblib import Parallel, delayed
import multiprocessing
import math
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Function to create sequences for LSTM model
def create_sequences(series, window=30):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i-window:i])
        y.append(1 if series[i] > 0 else 0)  # 1 if positive return, 0 if negative
    return np.array(X), np.array(y)

# Function to train LSTM and return signals
def train_lstm(log_returns):
    # Drop NaN values and keep the index intact
    log_returns = log_returns.dropna()
    
    # Preserve the original index and reshape for LSTM
    original_index = log_returns.index
    log_returns_values = log_returns.values.reshape(-1, 1)

    # Scale the log returns (values only)
    scaler = StandardScaler()
    scaled_log_returns = scaler.fit_transform(log_returns_values)

    # Create sequences for LSTM
    X, y = create_sequences(scaled_log_returns)

    # Reshape X to be compatible with LSTM input shape
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(16, input_shape=(X.shape[1], 1)))
    model.add(Dense(1, activation="sigmoid"))  # Binary classification (0 or 1)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    # Train the LSTM model
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Predict probabilities and generate signals
    probs = model.predict(X, verbose=0).flatten()
    signals = np.where(probs > 0.6, 1, -1)  # Threshold to generate signals (+1 or -1)

    # Return the signals as a Pandas Series with the original index
    signal_series = pd.Series(
        signals,
        index=original_index[-len(signals):]  # Use the original index for alignment
    )

    return signal_series
