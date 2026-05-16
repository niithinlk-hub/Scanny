# Scanny

Streamlit app for stock screening and backtesting on Yahoo Finance data.

## Strategies

- **MACD Money Map** — trend (zero-line bias + distance-filtered crossover with 1-2 bar confirmation) + reversal (price/MACD divergence + first histogram flip). Optional weekly higher-timeframe MACD bias.
- **Triple Threat** — MACD crossover + RSI(14) trend filter (50 mid-line) + Stochastic(14,3) extremes.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- Bulk Yahoo download via `yf.download(group_by="ticker", threads=True)` — no per-ticker loops.
- Invalid / delisted tickers skipped gracefully and listed in the UI.
- `@st.cache_data` caching keyed on tickers + dates + interval.
- Manual ticker entry or CSV upload (`ticker` column).
- Sidebar: date range, timeframe (1d/1wk/1h), strategy, "active signal today" filter.
- Screener table: latest close + signal + key indicators per ticker.
- Per-ticker backtest: summary metrics, equity curve, candlestick chart with entry/exit markers, MACD subplot, trades CSV download.
- Backtest engine: 1 unit per signal, 2R target with 50% partial + breakeven stop on remainder, swing-low/high initial stop, opposite-cross early exit.

## Deploy

Pushed to GitHub and Streamlit Community Cloud as **Scanny**.
