# Intraday Volatility Trading Strategy using GARCH and LSTM

## Overview

This project investigates a **volatility-driven trading strategy** that combines classical econometric modelling with modern machine learning techniques.

The system uses:

- a **GARCH model** to forecast daily volatility regimes  
- an **LSTM neural network** to learn directional patterns in returns  
- a **two-timeframe trading framework** combining daily and intraday data  
- **out-of-sample backtesting** with risk-adjusted evaluation  

The project is designed as a **research-oriented MSc-level study**, emphasising methodology, correctness, and evaluation rather than optimising for raw profit.

---

## Motivation

Financial markets exhibit:

- volatility clustering  
- regime changes  
- non-stationary behaviour  

Traditional models (e.g. **GARCH**) are effective at volatility forecasting, while neural networks are better suited to **pattern recognition in sequences**.

This project explores how:

- statistical volatility forecasts can act as **features**  
- machine learning confidence can **scale risk**  
- combining both can improve **decision-making**

---

## Strategy Design

### High-Level Architecture
```
Daily Prices
↓
Log Returns
↓
Rolling GARCH Volatility Forecast
↓
Daily Trade Direction (GARCH)
↓
LSTM Confidence Scaling
↓
Position Sizing
↓
Backtest & Evaluation
```

---

### Daily Layer (GARCH)

- Compute daily log returns  
- Estimate rolling 6-month realised variance  
- Fit a **GARCH(1,3)** model in a rolling window  
- Forecast next-day variance  
- Compute a prediction premium  
- Generate a directional daily signal  

> The GARCH model is used **only as a feature generator**.

---

### Machine Learning Layer (LSTM)

An LSTM network is trained on sequences of historical log returns to estimate the **probability of a positive return**.

**Key characteristics:**

- sequence-based learning  
- binary directional target  
- probability output (0–1)  
- chronological train/test split  

The LSTM **does not decide direction directly**.  
Instead, it:

- provides confidence  
- scales position size  
- reduces exposure during uncertain periods  

This reflects real-world quantitative trading practice.

---

### Signal Combination Logic

Final position sizing:
Final Signal = GARCH Direction × LSTM Confidence


Where:

- **GARCH** determines long or short  
- **LSTM** determines position size  

Positions are:

- entered at the next trading day  
- held for one day  
- capped to avoid excessive leverage  

---

## Backtesting & Evaluation

### Metrics Used

- Cumulative return  
- Buy-and-hold baseline  
- Number of trades  
- Sharpe ratio (risk-adjusted return)  

> Accuracy alone is not used, as it is misleading in financial contexts.

---

### Results Interpretation

- Strategy evaluated against a **strong baseline** (buy-and-hold)  
- Many strategies underperform in raw returns  

This project prioritises **correct research methodology** over profit optimisation.

---

## Project Structure

```
garch/
│
├── data/
│ ├── daily.csv
│ ├── intraday.csv
│
├── models/
│ ├── garch.py
│ ├── lstm.py
│
├── strategy.py
├── final_strategy_output.csv
├── README.md

```

---

## Requirements

- Python 3.11  
- numpy  
- pandas  
- matplotlib  
- arch  
- joblib  
- tensorflow / keras  
- scikit-learn  

### Install dependencies

```bash
pip install numpy pandas matplotlib arch joblib tensorflow scikit-learn
```

## Running the Project

```bash
python strategy.py
```

## Outputs

- Cumulative return plot  
- Sharpe ratio  
- Number of trades  
- CSV containing signals and returns  

---

## Financial Terminology (Brief)

- **Log return**: Percentage price change using logarithms  
- **Variance**: Measure of return dispersion  
- **Volatility**: Square root of variance  
- **GARCH**: Model capturing volatility clustering  
- **Sharpe ratio**: Risk-adjusted return metric  
- **Look-ahead bias**: Using future data unintentionally  

---

## Limitations

- Single asset  
- No transaction costs  
- No slippage  
- Simplified execution assumptions  

These are intentional for clarity.

---

## Future Extensions

- Transaction cost modelling  
- Volatility-targeted position sizing  
- Multi-asset portfolios  
- EGARCH / GJR-GARCH  
- Regime classification  
- Reinforcement learning policy optimisation  

---

## Academic Positioning

This project demonstrates:

- Econometric modelling  
- ML sequence learning  
- Signal fusion  
- Backtesting discipline  
- Research-grade evaluation  

