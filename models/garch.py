import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from arch import arch_model
from joblib import Parallel, delayed
import multiprocessing
import os

import os
import pandas as pd

from pathlib import Path
import pandas as pd

import pandas as pd
from pathlib import Path

def load_data():
    # Construct the path dynamically for daily data
    daily_file_path = Path(__file__).resolve().parent.parent / 'data' / 'daily.csv'
    
    # Check if the file path is correct
    print("Daily data file path:", daily_file_path)

    # Load the daily data and perform operations
    daily_df = (
        pd.read_csv(daily_file_path)
        .drop(columns=["Unnamed: 7"], errors="ignore")  # Ignore if not present
        .assign(Date=lambda x: pd.to_datetime(x["Date"]))
        .set_index("Date")
    )

    # Construct the path dynamically for intraday data
    intraday_file_path = Path(__file__).resolve().parent.parent / 'data' / 'intraday.csv'
    
    # Check if the file path is correct
    print("Intraday data file path:", intraday_file_path)

    # Load the intraday data and perform operations
    intraday_df = (
        pd.read_csv(intraday_file_path)
        .drop(columns=["Unnamed: 6"], errors="ignore")  # Ignore if not present
        .assign(datetime=lambda x: pd.to_datetime(x["datetime"]))
        .set_index("datetime")
        .assign(date=lambda x: pd.to_datetime(x.index.date))
    )

    return daily_df, intraday_df


def compute_returns_and_variance(df):
    df["log_ret"] = np.log(df["Adj Close"]).diff()
    df["variance"] = df["log_ret"].rolling(180).var()
    return df


# GARCH fit for a single window
def fit_garch_window(x):
    # Scale up to avoid poor scaling warning
    x_scaled = x * 100
    model = arch_model(x_scaled, p=1, q=3)
    fit = model.fit(disp="off", options={"maxiter": 5000})
    return fit.forecast(horizon=1).variance.iloc[-1, 0] / (100 ** 2)


# Rolling parallel GARCH
def run_garch_parallel(df, window=180):
    returns = df["log_ret"].dropna()
    windows = [returns.iloc[i - window:i] for i in range(window, len(returns))]

    n_jobs = multiprocessing.cpu_count()
    garch_vars = Parallel(n_jobs=n_jobs)(
        delayed(fit_garch_window)(x) for x in windows
    )

    df["prediction"] = np.nan
    df.loc[returns.index[window:], "prediction"] = garch_vars
    return df.dropna()

def generate_daily_signal(df, threshold=0.01):
    df["prediction_premium"] = (df["prediction"] - df["variance"]) / df["variance"]
    df["premium_std"] = df["prediction_premium"].rolling(180).std()

    # Apply threshold to avoid tiny signals
    df["signal_daily"] = np.where(
        df["prediction_premium"] > df["premium_std"] * threshold, 1,
        np.where(df["prediction_premium"] < -df["premium_std"] * threshold, -1, 0)
    )

    # Shift signal for next day execution
    df["signal_daily"] = df["signal_daily"].shift().fillna(0)
    return df

def plot_daily_signal_distribution(df):
    plt.style.use("ggplot")
    df["signal_daily"].plot(kind="hist", bins=5, edgecolor="k")
    plt.title("Daily Signal Distribution")
    plt.xlabel("Signal")
    plt.ylabel("Frequency")
    plt.show()
    
# Intraday indicators
def add_intraday_indicators(df):
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(20).mean()
    avg_loss = loss.rolling(20).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    ma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["uband"] = ma + 2 * std
    df["lband"] = ma - 2 * std

    return df

def generate_intraday_signal(df):
    df["signal_intraday"] = np.where(
        (df["rsi"] > 70) & (df["close"] > df["uband"]), 1,
        np.where((df["rsi"] < 30) & (df["close"] < df["lband"]), -1, 0)
    )
    return df

# Merge daily signals into intraday dataframe

def merge_daily_intraday(daily_df, intraday_df):
    df = intraday_df.reset_index().merge(
        daily_df[["signal_daily"]].reset_index(),
        left_on="date",
        right_on="Date"
    ).drop(columns=["date", "Date"]).set_index("datetime")
    return df

def compute_strategy_returns(df):
    df["return"] = np.log(df["close"]).diff()

    df["return_sign"] = np.where(
        (df["signal_daily"] == 1) & (df["signal_intraday"] == 1), -1,
        np.where((df["signal_daily"] == -1) & (df["signal_intraday"] == -1), 1, 0)
    )

    df["return_sign"] = df.groupby(pd.Grouper(freq="D"))["return_sign"].ffill()
    df["forward_return"] = df["return"].shift(-1)
    df["strategy_return"] = df["forward_return"] * df["return_sign"]

    return df

def compute_daily_returns(df):
    return df.groupby(pd.Grouper(freq="D"))["strategy_return"].sum()

def plot_cumulative_returns(daily_returns):
    cumulative = np.exp(np.log1p(daily_returns).cumsum()) - 1
    cumulative.plot(figsize=(16, 6))
    plt.title("Intraday Strategy Returns")
    plt.ylabel("Return")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.show()

def run_garch_pipeline():
    daily_df, _ = load_data()
    daily_df = daily_df["2020":]
    daily_df = compute_returns_and_variance(daily_df)
    daily_df = run_garch_parallel(daily_df)
    daily_df = generate_daily_signal(daily_df)
    return daily_df



# if __name__ == "__main__":
#     daily_df, intraday_df = load_data()

#     # Only use data from 2020
#     daily_df = daily_df["2020":]

#     daily_df = compute_returns_and_variance(daily_df)
#     daily_df = run_garch_parallel(daily_df)
#     daily_df = generate_daily_signal(daily_df)

#     plot_daily_signal_distribution(daily_df)

#     final_df = merge_daily_intraday(daily_df, intraday_df)
#     final_df = add_intraday_indicators(final_df)
#     final_df = generate_intraday_signal(final_df)
#     final_df = compute_strategy_returns(final_df)

#     daily_returns = compute_daily_returns(final_df)
#     plot_cumulative_returns(daily_returns)
