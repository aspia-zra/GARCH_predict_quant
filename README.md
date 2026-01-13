# Intraday volatility GARCH model prediction strategy

This project implements a simple volatility-driven trading strategy using a GARCH model to generate daily trade signals, demonstrating:

- time series modelling
- rolling-window estimation
- signal generation
- backtesting

Financial vocabulary explained at the bottom.

Note: This project is a research project combining statistical models with rule-based execution. The GARCH model is used as a feature generator; rolling volatility forecasts are treated as learned signals derived from historical data and combined with intraday indicators to drive execution decisions.

This strategy follows a supervised-style pipeline:
- feature construction
- signal generation
- evaluation with out-of-sample backtesting

This is done using a two-timeframe strategy using daily and intraday data on a given asset, wherein daily data is used to decide whether to trade today, and intraday signals are used to decide when to enter, and positions are opened once per day, and held until the end of the trading day.

## Strategy

This strategy uses a GARCH model using volatility to generate a daily prediction premium, comparing this prediction to recent realised volatility: trades are made when predicted volatility is unusually high.

The intraday data uses technical indicators for the price action pattern to generate our final signal to determine the singular time to enter and hold the trade during the day until the end of the day.

## Logic

Daily signals:
- calculate daily log returns
- calculate rolling 6 month variance
- fit a GARCH model in a rolling window
- predict variance for the next day
- signal generation to execute trades, if predicted variance > realised variance
- backtest: apply the signal to daily returns

Intraday signals:
- merge daily signal data with intraday data
- compute intraday indicators
- enter one position per day when intraday conditions align
- exit at end of day

## Files

main.py
daily.csv
intraday.csv

## Requirements
Python 3.11
pandas
numpy
arch

### Install dependencies

``` pip install numpy pandas matplotlib arch joblib ```

### Running the files

``` python main.py ```

The script prints total return.

## Notes

Optimised for speed:

We have used a parallel rolling GARCH model, where each rolling window is independent and use their own CPU




Financial Vocabulary explained here:

* * insert

The script outputs a daily time series, and strategy returns as a cumulative return and per-day returns as a Series.

# Future improvements:

Data:
- splitting data into training validation and test periods to avoid overfitting
- apply strategy across several stocks to test diversification

Modelling:
- GARCH variants to capture asymmetry, such as EGARCH
- alternative stochastic volatility models for speed
- automatic standardisation pipelines to normalise

Performance Evaluation:
- Sharpe ratio to measure risk-adjusted performance
- Monte Carlo simulations to test robustness

Visualisation:
- use Plotly for interactive visualisation

AI:
- using LSTM models on returns
- reinforcement learning to optimise signal weighting as trading policy