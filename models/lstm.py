import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# -----------------------------
# Create rolling sequences for multi-feature input
# -----------------------------
def make_sequences(X, y, window=30):
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i - window:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# -----------------------------
# LSTM training + evaluation
# -----------------------------
def train_lstm(df, target_col="log_ret", features=None, window=30, train_frac=0.7):
    """
    df: DataFrame with log returns and optional features
    target_col: column to predict (by default log returns)
    features: list of columns to use as input features
    """
    df = df.dropna()
    
    # Default: use only log returns
    if features is None:
        features = [target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].values)
    
    # Target: up/down
    y = (df[target_col].values > 0).astype(int)
    
    X_seq, y_seq = make_sequences(X_scaled, y, window)
    
    split = int(len(X_seq) * train_frac)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    # -----------------------------
    # Model
    # -----------------------------
    model = Sequential([
        Input(shape=(window, len(features))),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy"
    )
    
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # -----------------------------
    # Evaluation
    # -----------------------------
    probs_test = model.predict(X_test, verbose=0).flatten()
    preds_test = (probs_test > 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds_test)
    baseline_acc = max(np.mean(y_test), 1 - np.mean(y_test))
    
    print(f"LSTM accuracy:     {acc:.3f}")
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    
    # -----------------------------
    # Predict full series for strategy
    # -----------------------------
    probs_all = model.predict(X_seq, verbose=0).flatten()
    prob_series = pd.Series(probs_all, index=df.index[window:])
    
    return prob_series
