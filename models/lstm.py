import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# -----------------------------
# Create rolling sequences
# -----------------------------
def make_sequences(series, window=30):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(1 if series[i] > 0 else 0)
    return np.array(X), np.array(y)


# -----------------------------
# LSTM training + evaluation
# -----------------------------
def train_lstm(log_returns, window=30, train_frac=0.7):
    log_returns = log_returns.dropna()

    values = log_returns.values.reshape(-1, 1)

    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)

    X, y = make_sequences(values_scaled, window)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * train_frac)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -----------------------------
    # Model
    # -----------------------------
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(16),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy"
    )

    model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=32,
        verbose=0
    )

    # -----------------------------
    # Evaluation
    # -----------------------------
    probs_test = model.predict(X_test, verbose=0).flatten()
    preds_test = (probs_test > 0.5).astype(int)

    acc = accuracy_score(y_test, preds_test)

    baseline_acc = max(
        np.mean(y_test),
        1 - np.mean(y_test)
    )

    print(f"LSTM accuracy:     {acc:.3f}")
    print(f"Baseline accuracy: {baseline_acc:.3f}")

    # -----------------------------
    # Predict full series (for strategy)
    # -----------------------------
    probs_all = model.predict(X, verbose=0).flatten()

    prob_series = pd.Series(
        probs_all,
        index=log_returns.index[window:]
    )

    return prob_series
