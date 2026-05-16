"""
Scanny - Stock Screener & Backtester
Streamlit app for multi-ticker screening and backtesting using Yahoo Finance.

Strategies:
    1. MACD Money Map (trend + reversal, multi-timeframe MACD)
    2. Triple Threat (MACD + Stochastic + RSI)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

APP_NAME = "Scanny"
APP_TAGLINE = "Stock Screener & Backtester — MACD Money Map / Triple Threat"

# =============================================================================
# Indicator helpers
# =============================================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3
               ) -> Tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    rng = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / rng
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k.fillna(50.0), d.fillna(50.0)


def swing_low(low: pd.Series, lookback: int) -> pd.Series:
    return low.rolling(lookback, min_periods=1).min()


def swing_high(high: pd.Series, lookback: int) -> pd.Series:
    return high.rolling(lookback, min_periods=1).max()


def find_pivots(series: pd.Series, left: int = 3, right: int = 3,
                kind: str = "low") -> List[int]:
    """Return integer index positions of pivot lows/highs."""
    vals = series.values
    n = len(vals)
    pivots: List[int] = []
    for i in range(left, n - right):
        window = vals[i - left:i + right + 1]
        center = vals[i]
        if np.isnan(center):
            continue
        if kind == "low" and center == np.nanmin(window):
            pivots.append(i)
        elif kind == "high" and center == np.nanmax(window):
            pivots.append(i)
    return pivots


# =============================================================================
# Strategy framework
# =============================================================================

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    direction: int  # 1 long, -1 short
    entry_price: float
    exit_price: Optional[float]
    stop: float
    target: float
    size: float = 1.0
    r_multiple: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4) if self.exit_price is not None else None,
            "stop": round(self.stop, 4),
            "target": round(self.target, 4),
            "r_multiple": round(self.r_multiple, 3),
            "pnl_pct": round(self.pnl_pct * 100, 3),
            "exit_reason": self.exit_reason,
        }


class Strategy(ABC):
    """Base strategy class. Subclasses implement generate_signals + per-bar logic."""

    name: str = "Base"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with 'signal' column (1 long, -1 short, 0 flat) and indicator cols."""

    @abstractmethod
    def initial_stop(self, df: pd.DataFrame, i: int, direction: int) -> float:
        ...

    def initial_target(self, entry: float, stop: float, direction: int,
                       rr: float = 2.0) -> float:
        risk = abs(entry - stop)
        return entry + rr * risk * direction

    def early_exit(self, df: pd.DataFrame, i: int, trade: Trade) -> Optional[str]:
        """Return reason string if early exit triggered this bar; else None."""
        return None

    def backtest(self, df: pd.DataFrame, swing_lookback: int = 10,
                 partial_at_2r: bool = True) -> Tuple[List[Trade], pd.DataFrame]:
        df = self.generate_signals(df).copy()
        df["_swing_low"] = df["Low"].rolling(swing_lookback, min_periods=1).min()
        df["_swing_high"] = df["High"].rolling(swing_lookback, min_periods=1).max()

        trades: List[Trade] = []
        open_trade: Optional[Trade] = None
        partial_taken = False
        remaining_size = 1.0

        signals = df["signal"].values
        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values
        idx = df.index

        for i in range(len(df)):
            price_h = highs[i]
            price_l = lows[i]
            price_c = closes[i]

            if open_trade is not None:
                d = open_trade.direction
                stop = open_trade.stop
                target = open_trade.target
                risk0 = abs(open_trade.entry_price - stop)
                two_r_price = open_trade.entry_price + 2 * risk0 * d

                hit_stop = (price_l <= stop) if d == 1 else (price_h >= stop)
                hit_2r = (price_h >= two_r_price) if d == 1 else (price_l <= two_r_price)

                early = self.early_exit(df, i, open_trade)

                if hit_stop:
                    exit_price = stop
                    open_trade.exit_date = idx[i]
                    open_trade.exit_price = exit_price
                    open_trade.exit_reason = "stop"
                    risk = abs(open_trade.entry_price - open_trade.stop)
                    pnl = (exit_price - open_trade.entry_price) * d
                    open_trade.r_multiple = pnl / risk if risk > 0 else 0.0
                    open_trade.pnl_pct = pnl / open_trade.entry_price
                    trades.append(open_trade)
                    open_trade = None
                    partial_taken = False
                    remaining_size = 1.0
                    continue

                if partial_at_2r and hit_2r and not partial_taken:
                    partial_taken = True
                    remaining_size = 0.5
                    open_trade.stop = open_trade.entry_price  # move to breakeven
                    stop = open_trade.stop

                if early is not None:
                    exit_price = price_c
                    open_trade.exit_date = idx[i]
                    open_trade.exit_price = exit_price
                    open_trade.exit_reason = early
                    risk = abs(open_trade.entry_price - open_trade.stop) or abs(open_trade.entry_price - stop)
                    risk = risk if risk > 0 else max(1e-9, abs(open_trade.entry_price * 0.01))
                    pnl = (exit_price - open_trade.entry_price) * d
                    open_trade.r_multiple = pnl / risk if risk > 0 else 0.0
                    open_trade.pnl_pct = pnl / open_trade.entry_price
                    trades.append(open_trade)
                    open_trade = None
                    partial_taken = False
                    remaining_size = 1.0
                    continue

            if open_trade is None and signals[i] != 0:
                direction = int(signals[i])
                entry = price_c
                stop = self.initial_stop(df, i, direction)
                if (direction == 1 and stop >= entry) or (direction == -1 and stop <= entry):
                    pad = entry * 0.02
                    stop = entry - pad if direction == 1 else entry + pad
                target = self.initial_target(entry, stop, direction, rr=2.0)
                open_trade = Trade(
                    entry_date=idx[i], exit_date=None, direction=direction,
                    entry_price=entry, exit_price=None, stop=stop, target=target,
                )
                partial_taken = False
                remaining_size = 1.0

        if open_trade is not None:
            last = len(df) - 1
            exit_price = closes[last]
            open_trade.exit_date = idx[last]
            open_trade.exit_price = exit_price
            open_trade.exit_reason = "open_at_end"
            risk = abs(open_trade.entry_price - open_trade.stop) or 1e-9
            pnl = (exit_price - open_trade.entry_price) * open_trade.direction
            open_trade.r_multiple = pnl / risk
            open_trade.pnl_pct = pnl / open_trade.entry_price
            trades.append(open_trade)

        equity = build_equity_curve(df.index, trades)
        return trades, equity


# =============================================================================
# Strategy 1: MACD Money Map
# =============================================================================

class MoneyMapStrategy(Strategy):
    name = "MACD Money Map"

    def __init__(self, dist_mult: float = 0.5, confirm_bars: int = 2,
                 use_htf_bias: bool = True, swing_lookback: int = 10):
        super().__init__(dist_mult=dist_mult, confirm_bars=confirm_bars,
                         use_htf_bias=use_htf_bias, swing_lookback=swing_lookback)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        m, s, h = macd(df["Close"])
        df["macd"] = m
        df["macd_signal"] = s
        df["macd_hist"] = h

        std = m.rolling(50, min_periods=20).std().bfill().fillna(m.std())
        thr = self.params["dist_mult"] * std
        df["macd_thr"] = thr

        # Higher timeframe bias (weekly resample)
        if self.params["use_htf_bias"]:
            wk = df["Close"].resample("W").last().dropna()
            wm, ws, _ = macd(wk)
            htf = (wm > ws).astype(int) - (wm < ws).astype(int)
            df["htf_bias"] = htf.reindex(df.index, method="ffill").fillna(0)
        else:
            df["htf_bias"] = 0

        cross_up = (m > s) & (m.shift(1) <= s.shift(1))
        cross_dn = (m < s) & (m.shift(1) >= s.shift(1))

        cb = int(self.params["confirm_bars"])
        confirm_long = cross_up.shift(cb).fillna(False) & (m > s) & (m > thr) & (m > 0)
        confirm_short = cross_dn.shift(cb).fillna(False) & (m < s) & (m < -thr) & (m < 0)

        # Reversal: bullish divergence + first green histogram flip
        rev_long = self._reversal_signals(df, kind="bull")
        rev_short = self._reversal_signals(df, kind="bear")

        long_sig = (confirm_long | rev_long)
        short_sig = (confirm_short | rev_short)

        if self.params["use_htf_bias"]:
            long_sig = long_sig & (df["htf_bias"] >= 0)
            short_sig = short_sig & (df["htf_bias"] <= 0)

        sig = np.where(long_sig, 1, np.where(short_sig, -1, 0))
        df["signal"] = sig
        return df

    def _reversal_signals(self, df: pd.DataFrame, kind: str) -> pd.Series:
        out = pd.Series(False, index=df.index)
        price = df["Close"]
        m = df["macd"]
        hist = df["macd_hist"]

        if kind == "bull":
            piv = find_pivots(price, 3, 3, "low")
            for j in range(1, len(piv)):
                i_prev, i_cur = piv[j - 1], piv[j]
                if price.iloc[i_cur] < price.iloc[i_prev] and m.iloc[i_cur] > m.iloc[i_prev]:
                    # find first green hist bar after i_cur
                    for k in range(i_cur, min(len(df), i_cur + 10)):
                        if hist.iloc[k] > 0 and (k == 0 or hist.iloc[k - 1] <= 0):
                            out.iloc[k] = True
                            break
        else:
            piv = find_pivots(price, 3, 3, "high")
            for j in range(1, len(piv)):
                i_prev, i_cur = piv[j - 1], piv[j]
                if price.iloc[i_cur] > price.iloc[i_prev] and m.iloc[i_cur] < m.iloc[i_prev]:
                    for k in range(i_cur, min(len(df), i_cur + 10)):
                        if hist.iloc[k] < 0 and (k == 0 or hist.iloc[k - 1] >= 0):
                            out.iloc[k] = True
                            break
        return out

    def initial_stop(self, df: pd.DataFrame, i: int, direction: int) -> float:
        lb = int(self.params["swing_lookback"])
        lo = max(0, i - lb)
        if direction == 1:
            return float(df["Low"].iloc[lo:i + 1].min())
        return float(df["High"].iloc[lo:i + 1].max())

    def early_exit(self, df: pd.DataFrame, i: int, trade: Trade) -> Optional[str]:
        if i == 0:
            return None
        m = df["macd"].iloc[i]
        s = df["macd_signal"].iloc[i]
        m_prev = df["macd"].iloc[i - 1]
        s_prev = df["macd_signal"].iloc[i - 1]
        if trade.direction == 1 and m < s and m_prev >= s_prev:
            return "macd_flip"
        if trade.direction == -1 and m > s and m_prev <= s_prev:
            return "macd_flip"
        return None


# =============================================================================
# Strategy 2: Triple Threat (MACD + Stoch + RSI)
# =============================================================================

class TripleThreatStrategy(Strategy):
    name = "Triple Threat"

    def __init__(self, swing_lookback: int = 10, stoch_extreme_exit: bool = True):
        super().__init__(swing_lookback=swing_lookback,
                         stoch_extreme_exit=stoch_extreme_exit)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        m, s, h = macd(df["Close"])
        df["macd"] = m
        df["macd_signal"] = s
        df["macd_hist"] = h
        df["rsi"] = rsi(df["Close"], 14)
        k, d = stochastic(df["High"], df["Low"], df["Close"], 14, 3)
        df["stoch_k"] = k
        df["stoch_d"] = d

        cross_up = (m > s) & (m.shift(1) <= s.shift(1))
        cross_dn = (m < s) & (m.shift(1) >= s.shift(1))

        long_sig = cross_up & (df["rsi"] > 50) & (k < 80) & (d < 80) & \
                   ((k.shift(1) < 20) | (d.shift(1) < 20) | (k < 50))
        short_sig = cross_dn & (df["rsi"] < 50) & (k > 20) & (d > 20) & \
                    ((k.shift(1) > 80) | (d.shift(1) > 80) | (k > 50))

        sig = np.where(long_sig, 1, np.where(short_sig, -1, 0))
        df["signal"] = sig
        return df

    def initial_stop(self, df: pd.DataFrame, i: int, direction: int) -> float:
        lb = int(self.params["swing_lookback"])
        lo = max(0, i - lb)
        if direction == 1:
            return float(df["Low"].iloc[lo:i + 1].min())
        return float(df["High"].iloc[lo:i + 1].max())

    def early_exit(self, df: pd.DataFrame, i: int, trade: Trade) -> Optional[str]:
        if i == 0:
            return None
        m = df["macd"].iloc[i]
        s = df["macd_signal"].iloc[i]
        m_prev = df["macd"].iloc[i - 1]
        s_prev = df["macd_signal"].iloc[i - 1]
        if trade.direction == 1 and m < s and m_prev >= s_prev:
            return "macd_flip"
        if trade.direction == -1 and m > s and m_prev <= s_prev:
            return "macd_flip"
        if self.params.get("stoch_extreme_exit"):
            k = df["stoch_k"].iloc[i]
            d = df["stoch_d"].iloc[i]
            k_prev = df["stoch_k"].iloc[i - 1]
            if trade.direction == 1 and k > 80 and d > 80 and k < k_prev:
                return "stoch_overbought"
            if trade.direction == -1 and k < 20 and d < 20 and k > k_prev:
                return "stoch_oversold"
        return None


STRATEGIES: Dict[str, type] = {
    "MACD Money Map": MoneyMapStrategy,
    "Triple Threat": TripleThreatStrategy,
}


# =============================================================================
# Data download & cleaning
# =============================================================================

def _parse_tickers(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip().upper() for p in text.replace("\n", ",").replace(" ", ",").split(",")]
    return [p for p in parts if p]


@st.cache_data(show_spinner=False, ttl=60 * 30)
def download_prices(tickers: Tuple[str, ...], start: str, end: str,
                    interval: str = "1d") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    return raw


def clean_download_result(raw: pd.DataFrame, tickers: List[str]
                          ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    out: Dict[str, pd.DataFrame] = {}
    skipped: List[str] = []
    if raw is None or raw.empty:
        return out, list(tickers)

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            if is_multi:
                if t not in raw.columns.get_level_values(0):
                    skipped.append(t)
                    continue
                sub = raw[t].copy()
            else:
                sub = raw.copy()
            sub = sub.dropna(how="all")
            needed = {"Open", "High", "Low", "Close"}
            if not needed.issubset(set(sub.columns)):
                skipped.append(t)
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty or len(sub) < 30:
                skipped.append(t)
                continue
            if "Volume" not in sub.columns:
                sub["Volume"] = 0
            sub.index = pd.to_datetime(sub.index)
            out[t] = sub
        except Exception:
            skipped.append(t)
    return out, skipped


# =============================================================================
# Backtest metrics & equity curve
# =============================================================================

def build_equity_curve(idx: pd.DatetimeIndex, trades: List[Trade]) -> pd.DataFrame:
    eq = pd.Series(0.0, index=idx)
    cum = 0.0
    for t in trades:
        if t.exit_date is None:
            continue
        cum += t.r_multiple
        if t.exit_date in eq.index:
            eq.loc[t.exit_date] = cum
        else:
            after = eq.index[eq.index >= t.exit_date]
            if len(after) > 0:
                eq.loc[after[0]] = cum
    eq = eq.replace(0.0, np.nan).ffill().fillna(0.0)
    df = pd.DataFrame({"equity_R": eq})
    df["peak"] = df["equity_R"].cummax()
    df["drawdown_R"] = df["equity_R"] - df["peak"]
    return df


def summarize(trades: List[Trade], equity: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"total_trades": 0, "win_rate": 0, "avg_R": 0, "total_R": 0,
                "max_drawdown_R": 0, "profit_factor": 0, "wins": 0, "losses": 0}
    rs = np.array([t.r_multiple for t in trades])
    wins = int((rs > 0).sum())
    losses = int((rs <= 0).sum())
    gp = float(rs[rs > 0].sum()) if (rs > 0).any() else 0.0
    gl = float(-rs[rs < 0].sum()) if (rs < 0).any() else 0.0
    pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0.0
    mdd = float(equity["drawdown_R"].min()) if not equity.empty else 0.0
    return {
        "total_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(100 * wins / n, 2),
        "avg_R": round(float(rs.mean()), 3),
        "total_R": round(float(rs.sum()), 3),
        "max_drawdown_R": round(mdd, 3),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
    }


# =============================================================================
# Screener
# =============================================================================

def screen_tickers(data: Dict[str, pd.DataFrame], strategy: Strategy
                   ) -> pd.DataFrame:
    rows = []
    for t, df in data.items():
        try:
            sigdf = strategy.generate_signals(df)
            last = sigdf.iloc[-1]
            sig_today = int(last.get("signal", 0))
            sig_label = "LONG" if sig_today == 1 else ("SHORT" if sig_today == -1 else "—")
            row = {
                "ticker": t,
                "close": round(float(last["Close"]), 4),
                "signal": sig_label,
                "macd": round(float(last.get("macd", np.nan)), 4),
                "macd_signal": round(float(last.get("macd_signal", np.nan)), 4),
                "macd_hist": round(float(last.get("macd_hist", np.nan)), 4),
            }
            if "rsi" in sigdf.columns:
                row["rsi"] = round(float(last["rsi"]), 2)
            if "stoch_k" in sigdf.columns:
                row["stoch_k"] = round(float(last["stoch_k"]), 2)
                row["stoch_d"] = round(float(last["stoch_d"]), 2)
            rows.append(row)
        except Exception as e:
            rows.append({"ticker": t, "close": None, "signal": "ERR", "error": str(e)[:60]})
    return pd.DataFrame(rows)


# =============================================================================
# Charts
# =============================================================================

def make_price_chart(df: pd.DataFrame, trades: List[Trade], ticker: str,
                     strategy_name: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.03,
                        subplot_titles=(f"{ticker} — Price", "MACD", "Volume"))
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
        close=df["Close"], name="Price", showlegend=False), row=1, col=1)

    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    if longs:
        fig.add_trace(go.Scatter(
            x=[t.entry_date for t in longs], y=[t.entry_price for t in longs],
            mode="markers", marker_symbol="triangle-up", marker_size=12,
            marker_color="#16a34a", name="Long entry"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t.exit_date for t in longs if t.exit_date],
            y=[t.exit_price for t in longs if t.exit_price],
            mode="markers", marker_symbol="x", marker_size=10,
            marker_color="#16a34a", name="Long exit"), row=1, col=1)
    if shorts:
        fig.add_trace(go.Scatter(
            x=[t.entry_date for t in shorts], y=[t.entry_price for t in shorts],
            mode="markers", marker_symbol="triangle-down", marker_size=12,
            marker_color="#dc2626", name="Short entry"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t.exit_date for t in shorts if t.exit_date],
            y=[t.exit_price for t in shorts if t.exit_price],
            mode="markers", marker_symbol="x", marker_size=10,
            marker_color="#dc2626", name="Short exit"), row=1, col=1)

    if "macd" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
                                 line=dict(color="#2563eb")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                                 line=dict(color="#f59e0b")), row=2, col=1)
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Hist",
                             marker_color=colors, opacity=0.5), row=2, col=1)
        fig.add_hline(y=0, line_color="#888", line_dash="dot", row=2, col=1)

    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color="#64748b", opacity=0.5), row=3, col=1)

    fig.update_layout(
        height=720, template="plotly_white", showlegend=True,
        title=f"{strategy_name} — {ticker}",
        xaxis_rangeslider_visible=False, hovermode="x unified",
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def make_equity_chart(equity: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity["equity_R"],
                             name="Equity (R)", line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(x=equity.index, y=equity["drawdown_R"],
                             name="Drawdown (R)", line=dict(color="#dc2626", width=1),
                             fill="tozeroy", opacity=0.3))
    fig.update_layout(
        title=f"Equity Curve — {ticker} (cumulative R)",
        template="plotly_white", height=380, hovermode="x unified",
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


# =============================================================================
# Streamlit UI
# =============================================================================

def sidebar_inputs() -> dict:
    st.sidebar.title(f"⚡ {APP_NAME}")
    st.sidebar.caption(APP_TAGLINE)
    st.sidebar.markdown("---")

    default_tickers = "AAPL, MSFT, NVDA, RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS"
    text = st.sidebar.text_area("Tickers (comma / space separated)",
                                value=default_tickers, height=110)

    uploaded = st.sidebar.file_uploader("...or upload CSV (column: ticker)",
                                        type=["csv"])
    csv_tickers: List[str] = []
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            col = next((c for c in df_up.columns if c.lower() == "ticker"), None)
            if col is None:
                st.sidebar.error("CSV must have a 'ticker' column.")
            else:
                csv_tickers = [str(x).strip().upper() for x in df_up[col].dropna()]
        except Exception as e:
            st.sidebar.error(f"CSV read failed: {e}")

    tickers = sorted(set(_parse_tickers(text) + csv_tickers))

    st.sidebar.markdown("---")
    today = date.today()
    default_start = today - timedelta(days=365 * 3)
    start = st.sidebar.date_input("Start date", value=default_start,
                                  max_value=today - timedelta(days=1))
    end = st.sidebar.date_input("End date", value=today, max_value=today)

    interval = st.sidebar.selectbox("Timeframe",
                                    options=["1d", "1wk", "1h"], index=0,
                                    help="Daily recommended. Intraday limited by Yahoo history.")

    strat_name = st.sidebar.selectbox("Strategy", list(STRATEGIES.keys()))

    only_active = st.sidebar.checkbox("Show only tickers with active signal today",
                                      value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: Yahoo Finance (yfinance). No commissions/slippage modeled.")

    return {
        "tickers": tickers, "start": start, "end": end,
        "interval": interval, "strategy_name": strat_name,
        "only_active": only_active,
    }


def render_data_status(loaded: Dict[str, pd.DataFrame], skipped: List[str]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickers loaded", len(loaded))
    c2.metric("Tickers skipped", len(skipped))
    if loaded:
        sample_df = next(iter(loaded.values()))
        c3.metric("Bars (sample)", len(sample_df))
    if skipped:
        st.warning(f"Skipped tickers (no data / invalid): {', '.join(skipped)}")


def render_screener(loaded: Dict[str, pd.DataFrame], strategy: Strategy,
                    only_active: bool) -> None:
    st.subheader("📡 Screener — latest bar")
    if not loaded:
        st.info("Load tickers from the sidebar to populate the screener.")
        return
    table = screen_tickers(loaded, strategy)
    if only_active:
        table = table[table["signal"].isin(["LONG", "SHORT"])]
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_backtest(loaded: Dict[str, pd.DataFrame], strategy: Strategy,
                    strategy_name: str) -> None:
    st.subheader("🧪 Per-ticker backtest")
    if not loaded:
        st.info("Load tickers first.")
        return
    c1, c2 = st.columns([2, 1])
    ticker = c1.selectbox("Ticker", sorted(loaded.keys()))
    run = c2.button("Run backtest", type="primary", use_container_width=True)
    if not run:
        return

    df = loaded[ticker]
    with st.spinner(f"Backtesting {ticker}…"):
        trades, equity = strategy.backtest(df)
        summary = summarize(trades, equity)
        sigdf = strategy.generate_signals(df)

    m = st.columns(6)
    m[0].metric("Trades", summary["total_trades"])
    m[1].metric("Win rate %", summary["win_rate"])
    m[2].metric("Total R", summary["total_R"])
    m[3].metric("Avg R", summary["avg_R"])
    m[4].metric("Max DD (R)", summary["max_drawdown_R"])
    m[5].metric("Profit factor", summary["profit_factor"])

    st.plotly_chart(make_price_chart(sigdf, trades, ticker, strategy_name),
                    use_container_width=True)
    st.plotly_chart(make_equity_chart(equity, ticker), use_container_width=True)

    if trades:
        st.markdown("**Trades**")
        trades_df = pd.DataFrame([t.as_dict() for t in trades])
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        st.download_button("Download trades CSV",
                           data=trades_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"{ticker}_{strategy_name.replace(' ', '_')}_trades.csv",
                           mime="text/csv")
    else:
        st.info("No trades generated for this ticker over the selected range.")


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} — Screener & Backtester",
                       layout="wide", page_icon="⚡")
    st.markdown(f"## ⚡ {APP_NAME}")
    st.caption(APP_TAGLINE)

    cfg = sidebar_inputs()

    if not cfg["tickers"]:
        st.info("Enter tickers in the sidebar to begin.")
        return
    if cfg["start"] >= cfg["end"]:
        st.error("Start date must be before end date.")
        return

    strategy_cls = STRATEGIES[cfg["strategy_name"]]
    strategy = strategy_cls()

    with st.spinner(f"Downloading {len(cfg['tickers'])} tickers from Yahoo…"):
        try:
            raw = download_prices(
                tuple(cfg["tickers"]),
                start=cfg["start"].isoformat(),
                end=cfg["end"].isoformat(),
                interval=cfg["interval"],
            )
        except Exception as e:
            st.error(f"Download failed: {e}")
            return
        loaded, skipped = clean_download_result(raw, cfg["tickers"])

    render_data_status(loaded, skipped)
    st.markdown("---")
    render_screener(loaded, strategy, cfg["only_active"])
    st.markdown("---")
    render_backtest(loaded, strategy, cfg["strategy_name"])


if __name__ == "__main__":
    main()
