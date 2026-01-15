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
# Load GARCH signals
# -----------------------------
daily_df = run_garch_pipeline()
daily_df = daily_df.dropna(subset=["log_ret", "signal_daily"])


# -----------------------------
# Add LSTM features
# -----------------------------
daily_df["ret_1"] = daily_df["log_ret"].shift(1)
daily_df["ret_3"] = daily_df["log_ret"].rolling(3).mean()
daily_df["ret_5"] = daily_df["log_ret"].rolling(5).mean()
daily_df["vol_5"] = daily_df["log_ret"].rolling(5).std()
daily_df["mom_3"] = daily_df["ret_3"] / daily_df["vol_5"]
daily_df["mom_5"] = daily_df["ret_5"] / daily_df["vol_5"]
daily_df.dropna(inplace=True)

features = ["log_ret", "ret_1", "ret_3", "ret_5", "vol_5", "mom_3", "mom_5"]


# -----------------------------
# Walk-forward backtest
# -----------------------------
window_size = 500   # training window
test_size = 100     # testing window
all_probs = []

for start in range(0, len(daily_df) - window_size - test_size + 1, test_size):
    train_df = daily_df.iloc[start:start+window_size]
    test_df = daily_df.iloc[start+window_size:start+window_size+test_size]
    
    # Concatenate to keep index intact
    combined = pd.concat([train_df, test_df])
    
    # Train LSTM and get probabilities
    probs = train_lstm(combined, features=features)
    
    # Take only the test period probabilities
    probs_test = probs.loc[test_df.index]
    all_probs.append(probs_test)

# Combine all test window probabilities
daily_df["lstm_prob"] = pd.concat(all_probs).sort_index()


# -----------------------------
# Confidence-weighted + thresholded signal
# -----------------------------
threshold = 0.55
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
strategy_cum.plot(label="GARCH + LSTM Strategy (Walk-Forward)")
baseline_cum.plot(label="Buy & Hold")
plt.legend()
plt.title("Walk-Forward Strategy vs Baseline")
plt.ylabel("Return")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.show()


# -----------------------------
# Save output
# -----------------------------
daily_df.to_csv("walkforward_strategy_output_fixed.csv")
