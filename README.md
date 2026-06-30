# Quantitative Risk & Performance Analytics Engine

A production-grade, cross-asset Portfolio Risk and Performance Analytics Engine built locally in Python with a clean, decoupled modular architecture.

## 🚀 Target Project File Architecture
```
quantitative_risk_engine/
├── config/
│   └── portfolio_config.yaml         # Tickers, share holdings, risk limits, scenarios
├── src/
│   ├── data/
│   │   ├── downloader.py             # yfinance API client wrapper
│   │   ├── transformer.py            # Missing data treatment & return math
│   │   └── validator.py              # pricing validation & anomalies
│   ├── analytics/
│   │   └── performance.py            # Cumulative returns, Sharpe, Drawdowns
│   ├── portfolio/
│   │   ├── constructor.py            # Weight calculations & portfolio snapshot
│   │   └── exposure.py               # HHI concentration & sector exposures
│   └── risk/
│       ├── engine.py                 # Parametric/Historical/Monte Carlo VaR & ES
│       ├── stress.py                 # Historical & hypothetical stress scenarios
│       ├── monitoring.py             # Rolling risk metrics & drawdown tracking
│       └── limits.py                 # Risk limits & severity breach reporting
├── app.py                            # Streamlit main dashboard orchestration
└── requirements.txt                  # Pinned package dependencies
```

## 🛠️ Features & Modules
1. **Data Layer**: Robust yfinance price downloader, forward-fill data imputation, and validation checks (negative price detection, single-day anomaly flags).
2. **Portfolio Engine**: Position weighting, sector concentration reporting, Herfindahl-Hirschman Index (HHI), and Marginal Contribution to Risk (MCTR).
3. **Performance Analytics**: Annualized Returns, Volatility, Sharpe Ratio, and Drawdowns.
4. **VaR & ES Engine**: Historical, Parametric (Gaussian), and Monte Carlo simulation models for both Value-at-Risk and Expected Shortfall.
5. **Stress Testing**: Replay historical stress windows (e.g., COVID Crash, 2022 Rate Hikes) or run custom per-asset hypothetical shock scenarios.
6. **Risk Limits & Monitoring**: Pre-defined threshold checks (VaR limit, exposure limit, HHI limit) and rolling volatility/VaR tracking.

## 💻 Local Setup
1. Clone this repository.
2. Initialize virtual environment:
   ```bash
   python -m venv env
   .\env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```
