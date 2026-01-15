import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from models.garch import run_garch_pipeline
from models.lstm import train_lstm


# -----------------------------
# Sharpe ratio utility
# -----------------------------
def sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    excess = returns - risk_free_rate / periods
    if excess.std() == 0:
        return np.nan
    return np.sqrt(periods) * excess.mean() / excess.std()


# -----------------------------
# Run GARCH pipeline
# -----------------------------
daily_df = run_garch_pipeline()
daily_df = daily_df.dropna(subset=["log_ret", "signal_daily"])


# -----------------------------
# Add extra features for LSTM
# -----------------------------
daily_df["ret_1"] = daily_df["log_ret"].shift(1)
daily_df["ret_5"] = daily_df["log_ret"].rolling(5).mean()
daily_df["vol_5"] = daily_df["log_ret"].rolling(5).std()
daily_df.dropna(inplace=True)

features = ["log_ret", "ret_1", "ret_5", "vol_5"]


# -----------------------------
# Run LSTM (probabilities)
# -----------------------------
daily_df["lstm_prob"] = train_lstm(daily_df, features=features)


# -----------------------------
# Confidence-weighted + threshold signal
# -----------------------------
threshold = 0.55  # only trade if LSTM confident
daily_df["final_signal"] = daily_df["signal_daily"] * np.where(
    (daily_df["lstm_prob"] > threshold) | (daily_df["lstm_prob"] < 1-threshold),
    (daily_df["lstm_prob"] - 0.5) * 2,
    0
)
daily_df["final_signal"] = daily_df["final_signal"].shift(1)  # execute next day


# -----------------------------
# Backtest
# -----------------------------
daily_df["forward_return"] = daily_df["log_ret"].shift(-1)
daily_df["strategy_return"] = daily_df["final_signal"] * daily_df["forward_return"]
daily_df["baseline_return"] = daily_df["forward_return"]

strategy_cum = (1 + daily_df["strategy_return"].fillna(0)).cumprod() - 1
baseline_cum = (1 + daily_df["baseline_return"].fillna(0)).cumprod() - 1

strategy_sharpe = sharpe_ratio(daily_df["strategy_return"].dropna())
baseline_sharpe = sharpe_ratio(daily_df["baseline_return"].dropna())

print(f"Trades taken: {(daily_df['final_signal'].abs() > 0).sum()}")
print(f"Final strategy return: {strategy_cum.iloc[-1]:.2%}")
print(f"Final baseline return: {baseline_cum.iloc[-1]:.2%}")
print(f"Strategy Sharpe ratio: {strategy_sharpe:.2f}")
print(f"Baseline Sharpe ratio: {baseline_sharpe:.2f}")


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(14, 5))
strategy_cum.plot(label="GARCH + LSTM Strategy")
baseline_cum.plot(label="Buy & Hold")
plt.legend()
plt.title("Strategy vs Baseline")
plt.ylabel("Return")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.show()


# -----------------------------
# Save output
# -----------------------------
daily_df.to_csv("final_strategy_output.csv")
