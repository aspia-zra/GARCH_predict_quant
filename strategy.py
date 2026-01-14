import numpy as np
from models.garch import run_garch_pipeline
from models.lstm import train_lstm

daily_df = run_garch_pipeline()

lstm_signal = train_lstm(daily_df["log_ret"])

daily_df["signal_lstm"] = lstm_signal
daily_df["signal_lstm"] = daily_df["signal_lstm"].shift(1)

daily_df["final_signal"] = np.where(
    daily_df["signal_daily"] == daily_df["signal_lstm"],
    daily_df["signal_daily"],
    0
)
print(daily_df[["signal_daily", "signal_lstm", "final_signal"]].tail())
print(f"Number of trading days: {(daily_df['final_signal'] != 0).sum()}")
daily_df.to_csv("final_signals.csv")

