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
import requests
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
    default_interval: str = "1d"
    is_intraday: bool = False

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


# =============================================================================
# Strategy 3: Intraday Opening Range Breakout (ORB) on "stocks in play"
# =============================================================================

class IntradayORBStrategy(Strategy):
    name = "Intraday ORB"
    default_interval = "5m"
    is_intraday = True

    def __init__(self, or_minutes: int = 15, bar_minutes: int = 5,
                 gap_threshold_pct: float = 1.0, rv_threshold: float = 3.0,
                 breakout_buffer: float = 0.0015, vol_multiplier: float = 1.5,
                 sl_mode: str = "or_mid", r_multiple: float = 2.0,
                 rv_lookback: int = 10, require_directional_gap: bool = False):
        super().__init__(or_minutes=or_minutes, bar_minutes=bar_minutes,
                         gap_threshold_pct=gap_threshold_pct,
                         rv_threshold=rv_threshold,
                         breakout_buffer=breakout_buffer,
                         vol_multiplier=vol_multiplier,
                         sl_mode=sl_mode, r_multiple=r_multiple,
                         rv_lookback=rv_lookback,
                         require_directional_gap=require_directional_gap)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        bars_per_or = max(1, int(self.params["or_minutes"] // self.params["bar_minutes"]))
        df["_date"] = df.index.date

        for c in ("or_high", "or_low", "or_mid", "or_height",
                  "gap_pct", "rel_vol_15", "entry_price", "stop_price",
                  "target_price"):
            df[c] = np.nan
        df["signal"] = 0
        df["reason"] = ""

        day_groups = list(df.groupby("_date", sort=True))
        day_dates = [d for d, _ in day_groups]

        first_or_vol: Dict = {}
        first_open: Dict = {}
        last_close: Dict = {}
        for d, g in day_groups:
            first_or_vol[d] = float(g.iloc[:bars_per_or]["Volume"].sum())
            first_open[d] = float(g.iloc[0]["Open"])
            last_close[d] = float(g.iloc[-1]["Close"])

        rv_lb = int(self.params["rv_lookback"])
        buf = float(self.params["breakout_buffer"])
        vm = float(self.params["vol_multiplier"])
        gap_thr = float(self.params["gap_threshold_pct"])
        rv_thr = float(self.params["rv_threshold"])

        for di, (d, g) in enumerate(day_groups):
            or_bars = g.iloc[:bars_per_or]
            if len(or_bars) < bars_per_or:
                continue
            or_high = float(or_bars["High"].max())
            or_low = float(or_bars["Low"].min())
            or_mid = (or_high + or_low) / 2.0
            or_height = or_high - or_low
            or_vol_avg = float(or_bars["Volume"].mean()) if len(or_bars) else 0.0

            prev_close = last_close[day_dates[di - 1]] if di > 0 else None
            today_open = first_open[d]
            gap_pct = ((today_open / prev_close - 1) * 100) if prev_close else 0.0

            prior_vols = [first_or_vol[day_dates[j]]
                          for j in range(max(0, di - rv_lb), di)]
            avg_prior = float(np.mean(prior_vols)) if prior_vols else np.nan
            rv_15 = (first_or_vol[d] / avg_prior) if avg_prior and avg_prior > 0 else 0.0

            mask = df["_date"] == d
            df.loc[mask, "or_high"] = or_high
            df.loc[mask, "or_low"] = or_low
            df.loc[mask, "or_mid"] = or_mid
            df.loc[mask, "or_height"] = or_height
            df.loc[mask, "gap_pct"] = gap_pct
            df.loc[mask, "rel_vol_15"] = rv_15

            in_play = (abs(gap_pct) >= gap_thr) and (rv_15 >= rv_thr)
            if not in_play:
                continue

            post = g.iloc[bars_per_or:]
            triggered = False
            for ts, row in post.iterrows():
                if triggered:
                    break
                bar_close = float(row["Close"])
                bar_vol = float(row["Volume"])
                vol_ok = bar_vol >= vm * or_vol_avg if or_vol_avg > 0 else True

                long_brk = bar_close > or_high * (1 + buf)
                short_brk = bar_close < or_low * (1 - buf)
                if self.params["require_directional_gap"]:
                    long_brk = long_brk and gap_pct > 0
                    short_brk = short_brk and gap_pct < 0

                if long_brk and vol_ok:
                    entry = bar_close
                    stop = or_mid if self.params["sl_mode"] == "or_mid" else or_low * (1 - buf)
                    if stop >= entry:
                        stop = entry * 0.99
                    risk = entry - stop
                    target = entry + self.params["r_multiple"] * risk
                    df.at[ts, "signal"] = 1
                    df.at[ts, "entry_price"] = entry
                    df.at[ts, "stop_price"] = stop
                    df.at[ts, "target_price"] = target
                    df.at[ts, "reason"] = (
                        f"5m ORB LONG: gap {gap_pct:.1f}%, rel_vol_15={rv_15:.1f}x; "
                        f"breakout above OR_high={or_high:.2f} with vol "
                        f"{(bar_vol / (or_vol_avg or 1)):.1f}x OR avg. "
                        f"Entry={entry:.2f}, SL={stop:.2f}, TP={target:.2f}."
                    )
                    triggered = True
                elif short_brk and vol_ok:
                    entry = bar_close
                    stop = or_mid if self.params["sl_mode"] == "or_mid" else or_high * (1 + buf)
                    if stop <= entry:
                        stop = entry * 1.01
                    risk = stop - entry
                    target = entry - self.params["r_multiple"] * risk
                    df.at[ts, "signal"] = -1
                    df.at[ts, "entry_price"] = entry
                    df.at[ts, "stop_price"] = stop
                    df.at[ts, "target_price"] = target
                    df.at[ts, "reason"] = (
                        f"5m ORB SHORT: gap {gap_pct:.1f}%, rel_vol_15={rv_15:.1f}x; "
                        f"breakdown below OR_low={or_low:.2f} with vol "
                        f"{(bar_vol / (or_vol_avg or 1)):.1f}x OR avg. "
                        f"Entry={entry:.2f}, SL={stop:.2f}, TP={target:.2f}."
                    )
                    triggered = True

        df.drop(columns=["_date"], inplace=True)
        return df

    def initial_stop(self, df: pd.DataFrame, i: int, direction: int) -> float:
        s = df["stop_price"].iloc[i]
        if pd.isna(s):
            return float(df["Low"].iloc[i] * 0.99) if direction == 1 else float(df["High"].iloc[i] * 1.01)
        return float(s)

    def backtest(self, df: pd.DataFrame, **kwargs) -> Tuple[List[Trade], pd.DataFrame]:
        df = self.generate_signals(df).copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df["_d"] = df.index.date

        trades: List[Trade] = []
        for d, g in df.groupby("_d", sort=True):
            open_trade: Optional[Trade] = None
            for ts, row in g.iterrows():
                price_h = float(row["High"])
                price_l = float(row["Low"])

                if open_trade is not None:
                    d_dir = open_trade.direction
                    hit_stop = (price_l <= open_trade.stop) if d_dir == 1 else (price_h >= open_trade.stop)
                    hit_tgt = (price_h >= open_trade.target) if d_dir == 1 else (price_l <= open_trade.target)
                    if hit_stop:
                        exit_price = open_trade.stop
                        open_trade.exit_date = ts
                        open_trade.exit_price = exit_price
                        open_trade.exit_reason = "stop"
                        risk = abs(open_trade.entry_price - open_trade.stop) or 1e-9
                        open_trade.r_multiple = (exit_price - open_trade.entry_price) * d_dir / risk
                        open_trade.pnl_pct = (exit_price - open_trade.entry_price) * d_dir / open_trade.entry_price
                        trades.append(open_trade)
                        open_trade = None
                        continue
                    if hit_tgt:
                        exit_price = open_trade.target
                        open_trade.exit_date = ts
                        open_trade.exit_price = exit_price
                        open_trade.exit_reason = "target"
                        risk = abs(open_trade.entry_price - open_trade.stop) or 1e-9
                        open_trade.r_multiple = (exit_price - open_trade.entry_price) * d_dir / risk
                        open_trade.pnl_pct = (exit_price - open_trade.entry_price) * d_dir / open_trade.entry_price
                        trades.append(open_trade)
                        open_trade = None
                        continue

                if open_trade is None and int(row.get("signal", 0)) != 0:
                    direction = int(row["signal"])
                    open_trade = Trade(
                        entry_date=ts, exit_date=None, direction=direction,
                        entry_price=float(row["entry_price"]), exit_price=None,
                        stop=float(row["stop_price"]),
                        target=float(row["target_price"]),
                    )

            if open_trade is not None:
                last_ts = g.index[-1]
                last_c = float(g.iloc[-1]["Close"])
                open_trade.exit_date = last_ts
                open_trade.exit_price = last_c
                open_trade.exit_reason = "session_end"
                d_dir = open_trade.direction
                risk = abs(open_trade.entry_price - open_trade.stop) or 1e-9
                open_trade.r_multiple = (last_c - open_trade.entry_price) * d_dir / risk
                open_trade.pnl_pct = (last_c - open_trade.entry_price) * d_dir / open_trade.entry_price
                trades.append(open_trade)

        df.drop(columns=["_d"], inplace=True)
        equity = build_equity_curve(df.index, trades)
        return trades, equity


# =============================================================================
# Strategy 4: BTST (Buy Today, Sell Tomorrow) — overnight momentum
# =============================================================================

class BTSTStrategy(Strategy):
    name = "BTST"
    default_interval = "1d"
    is_intraday = False

    def __init__(self, min_day_pct: float = 2.0, close_in_range_thr: float = 0.8,
                 vol_rel_min: float = 2.0, vol_lookback: int = 20,
                 target_pct: float = 2.5, sl_mode: str = "day_extreme"):
        super().__init__(min_day_pct=min_day_pct,
                         close_in_range_thr=close_in_range_thr,
                         vol_rel_min=vol_rel_min, vol_lookback=vol_lookback,
                         target_pct=target_pct, sl_mode=sl_mode)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        prev_close = df["Close"].shift(1)
        df["day_return_pct"] = (df["Close"] / prev_close - 1) * 100
        vol_lb = int(self.params["vol_lookback"])
        df["vol_avg"] = df["Volume"].rolling(vol_lb, min_periods=max(5, vol_lb // 2)).mean()
        df["vol_rel"] = df["Volume"] / df["vol_avg"].replace(0, np.nan)
        rng = (df["High"] - df["Low"]).replace(0, np.nan)
        df["close_pos"] = (df["Close"] - df["Low"]) / rng

        long_cond = (
            (df["day_return_pct"] >= self.params["min_day_pct"]) &
            (df["close_pos"] >= self.params["close_in_range_thr"]) &
            (df["vol_rel"] >= self.params["vol_rel_min"])
        )
        short_cond = (
            (df["day_return_pct"] <= -self.params["min_day_pct"]) &
            ((1 - df["close_pos"]) >= self.params["close_in_range_thr"]) &
            (df["vol_rel"] >= self.params["vol_rel_min"])
        )
        df["signal"] = np.where(long_cond.fillna(False), 1,
                                np.where(short_cond.fillna(False), -1, 0))

        df["entry_price"] = np.where(df["signal"] != 0, df["Close"], np.nan)
        df["stop_price"] = np.where(df["signal"] == 1, df["Low"],
                            np.where(df["signal"] == -1, df["High"], np.nan))
        tgt_pct = float(self.params["target_pct"]) / 100.0
        df["target_price"] = np.where(
            df["signal"] == 1, df["Close"] * (1 + tgt_pct),
            np.where(df["signal"] == -1, df["Close"] * (1 - tgt_pct), np.nan)
        )

        reasons: List[str] = []
        for _, r in df.iterrows():
            s = int(r["signal"]) if not pd.isna(r["signal"]) else 0
            if s == 1:
                reasons.append(
                    f"BTST LONG: +{r['day_return_pct']:.1f}% day, close near HOD "
                    f"({r['close_pos']:.0%} of range), vol {r['vol_rel']:.1f}x {vol_lb}D avg. "
                    f"Entry={r['entry_price']:.2f}, SL={r['stop_price']:.2f}, TP={r['target_price']:.2f}. "
                    f"Expect overnight continuation."
                )
            elif s == -1:
                reasons.append(
                    f"BTST SHORT: {r['day_return_pct']:.1f}% day, close near LOD "
                    f"({(1 - r['close_pos']):.0%} of range), vol {r['vol_rel']:.1f}x {vol_lb}D avg. "
                    f"Entry={r['entry_price']:.2f}, SL={r['stop_price']:.2f}, TP={r['target_price']:.2f}."
                )
            else:
                reasons.append("")
        df["reason"] = reasons
        return df

    def initial_stop(self, df: pd.DataFrame, i: int, direction: int) -> float:
        s = df["stop_price"].iloc[i]
        if pd.isna(s):
            return float(df["Low"].iloc[i]) if direction == 1 else float(df["High"].iloc[i])
        return float(s)

    def backtest(self, df: pd.DataFrame, **kwargs) -> Tuple[List[Trade], pd.DataFrame]:
        df = self.generate_signals(df).copy()
        trades: List[Trade] = []
        sigs = df["signal"].values
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        stops = df["stop_price"].values
        targets = df["target_price"].values
        entries = df["entry_price"].values
        idx = df.index

        for i in range(len(df) - 1):
            s = int(sigs[i])
            if s == 0:
                continue
            entry = float(entries[i])
            stop = float(stops[i])
            target = float(targets[i])
            d2_open = float(opens[i + 1])
            d2_high = float(highs[i + 1])
            d2_low = float(lows[i + 1])
            d2_close = float(closes[i + 1])

            if s == 1:
                gap_hits_stop = d2_open <= stop
                gap_hits_tgt = d2_open >= target
                intraday_stop = d2_low <= stop
                intraday_tgt = d2_high >= target
            else:
                gap_hits_stop = d2_open >= stop
                gap_hits_tgt = d2_open <= target
                intraday_stop = d2_high >= stop
                intraday_tgt = d2_low <= target

            if gap_hits_stop:
                exit_price, reason = d2_open, "gap_stop"
            elif gap_hits_tgt:
                exit_price, reason = d2_open, "gap_target"
            elif intraday_tgt and not intraday_stop:
                exit_price, reason = target, "target"
            elif intraday_stop and not intraday_tgt:
                exit_price, reason = stop, "stop"
            elif intraday_stop and intraday_tgt:
                exit_price, reason = stop, "stop_first_conservative"
            else:
                exit_price, reason = d2_close, "day2_close"

            risk = abs(entry - stop) or 1e-9
            pnl = (exit_price - entry) * s
            trades.append(Trade(
                entry_date=idx[i], exit_date=idx[i + 1], direction=s,
                entry_price=entry, exit_price=exit_price,
                stop=stop, target=target,
                r_multiple=pnl / risk, pnl_pct=pnl / entry,
                exit_reason=reason,
            ))

        equity = build_equity_curve(df.index, trades)
        return trades, equity


STRATEGIES: Dict[str, type] = {
    "MACD Money Map": MoneyMapStrategy,
    "Triple Threat": TripleThreatStrategy,
    "Intraday ORB": IntradayORBStrategy,
    "BTST": BTSTStrategy,
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
# Ticker universes (NASDAQ-100, S&P 500, NSE NIFTY 500)
# =============================================================================

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Minimal embedded fallbacks if network/wikipedia/NSE fetch fails
_FALLBACK_SP500 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "TSLA","JPM","V","WMT","UNH","XOM","MA","PG","JNJ","ORCL","HD","COST",
    "ABBV","BAC","KO","CVX","NFLX","CRM","ADBE","MRK","PEP","TMO","AMD",
    "LIN","CSCO","ACN","ABT","WFC","MCD","DHR","DIS","TXN","PM","IBM","CAT",
    "INTU","GE","VZ","NOW","QCOM","UNP","NEE","SPGI","PFE","RTX","CMCSA",
    "AMGN","ISRG","HON","T","MS","AMAT","GS","BLK","BKNG","LOW","NKE","UPS",
]
_FALLBACK_NDX = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST",
    "NFLX","ADBE","PEP","AMD","CSCO","TMUS","CMCSA","INTC","QCOM","TXN",
    "AMGN","INTU","ISRG","HON","BKNG","AMAT","SBUX","VRTX","ADI","LRCX",
    "MU","GILD","ADP","PANW","REGN","PYPL","KLAC","MELI","SNPS","CDNS",
    "MAR","CTAS","FTNT","ASML","ABNB","ORLY","CRWD","ROP","CHTR","NXPI",
]
_FALLBACK_NSE = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","BAJFINANCE.NS",
    "AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","HCLTECH.NS","SUNPHARMA.NS",
    "TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","NESTLEIND.NS","ONGC.NS","NTPC.NS",
    "POWERGRID.NS","M&M.NS","TATAMOTORS.NS","TATASTEEL.NS","ADANIENT.NS",
    "JSWSTEEL.NS","COALINDIA.NS","BAJAJFINSV.NS","HDFCLIFE.NS","SBILIFE.NS",
    "DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","BRITANNIA.NS","EICHERMOT.NS",
    "GRASIM.NS","HEROMOTOCO.NS","HINDALCO.NS","INDUSINDBK.NS","BPCL.NS",
    "TECHM.NS","UPL.NS","APOLLOHOSP.NS","TATACONSUM.NS","BAJAJ-AUTO.NS",
    "ADANIPORTS.NS","SHRIRAMFIN.NS",
]


def _wiki_table(url: str, attempts: int = 2) -> List[pd.DataFrame]:
    for _ in range(attempts):
        try:
            r = requests.get(url, headers=_HTTP_HEADERS, timeout=15)
            r.raise_for_status()
            return pd.read_html(io.StringIO(r.text))
        except Exception:
            continue
    return []


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_sp500_tickers() -> List[str]:
    tables = _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    for tbl in tables:
        col = next((c for c in tbl.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
        if col is not None:
            tickers = [str(x).strip().upper().replace(".", "-") for x in tbl[col].dropna()]
            tickers = [t for t in tickers if t.isascii() and 1 <= len(t) <= 8]
            if len(tickers) > 100:
                return sorted(set(tickers))
    return list(_FALLBACK_SP500)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_nasdaq100_tickers() -> List[str]:
    tables = _wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100")
    for tbl in tables:
        cols_l = [str(c).strip().lower() for c in tbl.columns]
        col = None
        for cand in ("ticker", "symbol"):
            if cand in cols_l:
                col = tbl.columns[cols_l.index(cand)]
                break
        if col is not None:
            tickers = [str(x).strip().upper().replace(".", "-") for x in tbl[col].dropna()]
            tickers = [t for t in tickers if t.isascii() and 1 <= len(t) <= 8]
            if 50 <= len(tickers) <= 200:
                return sorted(set(tickers))
    return list(_FALLBACK_NDX)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_nse500_tickers() -> List[str]:
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if col is not None:
            tickers = [f"{str(x).strip().upper()}.NS" for x in df[col].dropna()]
            tickers = [t for t in tickers if len(t) > 3]
            if len(tickers) > 100:
                return sorted(set(tickers))
    except Exception:
        pass
    return list(_FALLBACK_NSE)


UNIVERSE_LOADERS = {
    "NASDAQ-100": fetch_nasdaq100_tickers,
    "S&P 500": fetch_sp500_tickers,
    "NSE (NIFTY 500)": fetch_nse500_tickers,
}


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

def _latest_signal_row(sigdf: pd.DataFrame, intraday: bool) -> Optional[pd.Series]:
    """Return latest row in the most recent session day that has signal != 0; else None.

    Daily strategies: only check the last bar.
    Intraday strategies: scan the last available session day for any signal.
    """
    if sigdf.empty or "signal" not in sigdf.columns:
        return None
    nz = sigdf[sigdf["signal"] != 0]
    if nz.empty:
        return None
    if not intraday:
        last = sigdf.iloc[-1]
        return last if int(last["signal"]) != 0 else None
    last_date = sigdf.index[-1].date() if hasattr(sigdf.index[-1], "date") else None
    if last_date is not None and hasattr(nz.index[-1], "date"):
        same_day = nz[[ts.date() == last_date for ts in nz.index]]
        if not same_day.empty:
            return same_day.iloc[-1]
    return nz.iloc[-1]


def screen_tickers(data: Dict[str, pd.DataFrame], strategy: Strategy
                   ) -> pd.DataFrame:
    rows = []
    intraday = bool(strategy.is_intraday)
    for t, df in data.items():
        try:
            sigdf = strategy.generate_signals(df)
            last_bar = sigdf.iloc[-1]
            close_now = round(float(last_bar["Close"]), 4)

            sig_row = _latest_signal_row(sigdf, intraday)
            sig_today = int(sig_row["signal"]) if sig_row is not None else 0
            sig_label = "LONG" if sig_today == 1 else ("SHORT" if sig_today == -1 else "—")

            row = {"ticker": t, "close": close_now, "signal": sig_label}

            if sig_row is not None and sig_today != 0:
                row["signal_time"] = sig_row.name
                for c in ("entry_price", "stop_price", "target_price"):
                    if c in sig_row and not pd.isna(sig_row[c]):
                        row[c] = round(float(sig_row[c]), 4)
                if "reason" in sig_row:
                    row["reason"] = str(sig_row["reason"])[:300]

            for c in ("gap_pct", "rel_vol_15"):
                if c in sigdf.columns and not pd.isna(last_bar.get(c, np.nan)):
                    row[c] = round(float(last_bar[c]), 3)
            for c in ("day_return_pct", "close_pos", "vol_rel"):
                if c in sigdf.columns and not pd.isna(last_bar.get(c, np.nan)):
                    row[c] = round(float(last_bar[c]), 3)
            for c in ("macd", "macd_signal", "macd_hist", "rsi", "stoch_k", "stoch_d"):
                if c in sigdf.columns and not pd.isna(last_bar.get(c, np.nan)):
                    row[c] = round(float(last_bar[c]), 3)

            rows.append(row)
        except Exception as e:
            rows.append({"ticker": t, "close": None, "signal": "ERR", "error": str(e)[:80]})
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

    if "or_high" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["or_high"], name="OR High",
                                 line=dict(color="#16a34a", width=1, dash="dot"),
                                 connectgaps=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["or_low"], name="OR Low",
                                 line=dict(color="#dc2626", width=1, dash="dot"),
                                 connectgaps=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["or_mid"], name="OR Mid",
                                 line=dict(color="#9ca3af", width=1, dash="dash"),
                                 connectgaps=False), row=1, col=1)
        if "rel_vol_15" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["rel_vol_15"], name="RelVol_15",
                                     line=dict(color="#7c3aed")), row=2, col=1)
            fig.add_hline(y=1.0, line_color="#888", line_dash="dot", row=2, col=1)

    if "day_return_pct" in df.columns and "macd" not in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["day_return_pct"], name="Day %",
                             marker_color=["#16a34a" if v >= 0 else "#dc2626"
                                           for v in df["day_return_pct"].fillna(0)],
                             opacity=0.6), row=2, col=1)
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

    st.sidebar.markdown("### 1. Universe")
    universes = st.sidebar.multiselect(
        "Preloaded universes",
        list(UNIVERSE_LOADERS.keys()),
        default=["NASDAQ-100"],
        help="Constituent lists fetched once and cached for 24h.",
    )

    with st.sidebar.expander("Universe preview / counts"):
        for u in universes:
            try:
                lst = UNIVERSE_LOADERS[u]()
                st.write(f"**{u}** — {len(lst)} tickers (e.g. {', '.join(lst[:6])} …)")
            except Exception as e:
                st.warning(f"{u}: fetch failed — {e}")

    st.sidebar.markdown("### 2. Custom tickers (optional)")
    text = st.sidebar.text_area(
        "Add tickers (comma / space / newline separated)",
        value="", height=80,
        help="Merged with selected universes. Use Yahoo suffixes for non-US (.NS, .BO, .L).",
    )

    uploaded = st.sidebar.file_uploader("…or upload CSV (column: ticker)",
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

    st.sidebar.markdown("### 3. Strategy")
    strat_name = st.sidebar.selectbox("Strategy", list(STRATEGIES.keys()))
    strat_cls = STRATEGIES[strat_name]
    forced_interval = strat_cls.default_interval
    is_intraday = strat_cls.is_intraday

    strat_params: dict = {}
    with st.sidebar.expander("Strategy parameters", expanded=False):
        if strat_name == "Intraday ORB":
            strat_params["or_minutes"] = st.number_input("OR window (min)",
                value=15, min_value=5, max_value=60, step=5)
            strat_params["bar_minutes"] = 5
            strat_params["gap_threshold_pct"] = st.number_input("Gap threshold %",
                value=1.0, min_value=0.0, max_value=10.0, step=0.25)
            strat_params["rv_threshold"] = st.number_input("RelVol 15m threshold (x)",
                value=3.0, min_value=0.5, max_value=20.0, step=0.5)
            strat_params["breakout_buffer"] = st.number_input("Breakout buffer (frac)",
                value=0.0015, min_value=0.0, max_value=0.01, step=0.0005, format="%.4f")
            strat_params["vol_multiplier"] = st.number_input("Breakout vol mult (x OR avg)",
                value=1.5, min_value=0.0, max_value=10.0, step=0.25)
            strat_params["r_multiple"] = st.number_input("R multiple (target)",
                value=2.0, min_value=0.5, max_value=10.0, step=0.5)
            strat_params["sl_mode"] = st.selectbox("Stop-loss mode",
                options=["or_mid", "or_extreme"], index=0)
            strat_params["require_directional_gap"] = st.checkbox(
                "Require gap aligned with breakout direction", value=False)
        elif strat_name == "BTST":
            strat_params["min_day_pct"] = st.number_input("Min day return %",
                value=2.0, min_value=0.5, max_value=15.0, step=0.5)
            strat_params["close_in_range_thr"] = st.number_input("Close-in-range threshold",
                value=0.8, min_value=0.5, max_value=1.0, step=0.05)
            strat_params["vol_rel_min"] = st.number_input("Min relative volume (x)",
                value=2.0, min_value=0.5, max_value=10.0, step=0.5)
            strat_params["vol_lookback"] = st.number_input("Vol lookback (days)",
                value=20, min_value=5, max_value=60, step=5)
            strat_params["target_pct"] = st.number_input("Target % (overnight)",
                value=2.5, min_value=0.5, max_value=10.0, step=0.5)
        else:
            st.caption("Defaults used (MACD strategies).")

    st.sidebar.markdown("### 4. Date range & timeframe")
    today = date.today()
    if is_intraday:
        default_start = today - timedelta(days=30)
        max_back = today - timedelta(days=58)  # Yahoo ~60-day cap for 5m
        st.sidebar.caption("⚠️ Intraday (5m) — Yahoo limits history to ~60 days.")
    else:
        default_start = today - timedelta(days=365 * 3)
        max_back = today - timedelta(days=365 * 10)
    start = st.sidebar.date_input("Start date", value=default_start,
                                  min_value=max_back,
                                  max_value=today - timedelta(days=1))
    end = st.sidebar.date_input("End date", value=today, max_value=today)
    st.sidebar.caption(f"Interval (auto from strategy): **{forced_interval}**")
    interval = forced_interval

    only_active = st.sidebar.checkbox("Show only tickers with active signal today",
                                      value=False)

    st.sidebar.markdown("---")
    load_clicked = st.sidebar.button("📥 Load Data", type="primary",
                                     use_container_width=True)
    clear_clicked = st.sidebar.button("🗑️ Clear loaded data",
                                      use_container_width=True)

    st.sidebar.caption("Data: Yahoo Finance (yfinance). No commissions/slippage modeled.")

    universe_tickers: List[str] = []
    for u in universes:
        try:
            universe_tickers.extend(UNIVERSE_LOADERS[u]())
        except Exception:
            pass

    tickers = sorted(set(universe_tickers + _parse_tickers(text) + csv_tickers))

    return {
        "tickers": tickers, "universes": universes,
        "start": start, "end": end,
        "interval": interval, "strategy_name": strat_name,
        "strat_params": strat_params,
        "only_active": only_active,
        "load_clicked": load_clicked, "clear_clicked": clear_clicked,
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


def _do_load(cfg: dict) -> None:
    if cfg["start"] >= cfg["end"]:
        st.error("Start date must be before end date.")
        return
    if not cfg["tickers"]:
        st.warning("No tickers selected. Pick a universe or add custom tickers.")
        return

    by_group: Dict[str, List[str]] = {u: list(UNIVERSE_LOADERS[u]()) for u in cfg["universes"]}
    custom = [t for t in cfg["tickers"]
              if all(t not in v for v in by_group.values())]
    if custom:
        by_group["Custom"] = custom

    all_loaded: Dict[str, pd.DataFrame] = {}
    all_skipped: List[str] = []
    group_meta: Dict[str, dict] = {}

    bar = st.progress(0.0, text="Starting download…")
    n_groups = max(1, len(by_group))
    for gi, (gname, glist) in enumerate(by_group.items()):
        if not glist:
            continue
        bar.progress(gi / n_groups, text=f"Downloading {gname} ({len(glist)} tickers)…")
        try:
            raw = download_prices(
                tuple(sorted(set(glist))),
                start=cfg["start"].isoformat(),
                end=cfg["end"].isoformat(),
                interval=cfg["interval"],
            )
            loaded, skipped = clean_download_result(raw, glist)
        except Exception as e:
            st.error(f"{gname} download failed: {e}")
            loaded, skipped = {}, list(glist)
        all_loaded.update(loaded)
        all_skipped.extend(skipped)
        group_meta[gname] = {"requested": len(glist), "loaded": len(loaded),
                             "skipped": len(skipped)}
    bar.progress(1.0, text="Done.")
    bar.empty()

    st.session_state["scanny_loaded"] = all_loaded
    st.session_state["scanny_skipped"] = sorted(set(all_skipped))
    st.session_state["scanny_groups"] = group_meta
    st.session_state["scanny_meta"] = {
        "start": cfg["start"].isoformat(),
        "end": cfg["end"].isoformat(),
        "interval": cfg["interval"],
        "universes": list(cfg["universes"]),
        "n_custom": len(by_group.get("Custom", [])),
    }


def render_group_breakdown(group_meta: Dict[str, dict]) -> None:
    if not group_meta:
        return
    rows = [{"Group": g, "Requested": m["requested"],
             "Loaded": m["loaded"], "Skipped": m["skipped"]}
            for g, m in group_meta.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} — Screener & Backtester",
                       layout="wide", page_icon="⚡")
    st.markdown(f"## ⚡ {APP_NAME}")
    st.caption(APP_TAGLINE)

    cfg = sidebar_inputs()

    if cfg["clear_clicked"]:
        for k in ("scanny_loaded", "scanny_skipped", "scanny_groups", "scanny_meta"):
            st.session_state.pop(k, None)
        st.success("Cleared loaded data.")

    if cfg["load_clicked"]:
        with st.spinner("Fetching data from Yahoo Finance…"):
            _do_load(cfg)

    loaded: Dict[str, pd.DataFrame] = st.session_state.get("scanny_loaded", {})
    skipped: List[str] = st.session_state.get("scanny_skipped", [])
    group_meta: Dict[str, dict] = st.session_state.get("scanny_groups", {})
    meta = st.session_state.get("scanny_meta", {})

    if not loaded:
        st.info("Pick universes / add tickers in the sidebar, then click **📥 Load Data**.")
        if cfg["tickers"]:
            st.caption(f"Currently selected: {len(cfg['tickers'])} tickers across "
                       f"{len(cfg['universes'])} universe(s) + "
                       f"custom inputs. Nothing downloaded yet.")
        return

    if meta:
        st.caption(
            f"Loaded {len(loaded)} tickers · {meta.get('interval')} · "
            f"{meta.get('start')} → {meta.get('end')} · "
            f"universes: {', '.join(meta.get('universes', []) or ['—'])}"
            + (f" · {meta.get('n_custom')} custom" if meta.get('n_custom') else "")
        )

    strategy_cls = STRATEGIES[cfg["strategy_name"]]
    try:
        strategy = strategy_cls(**cfg.get("strat_params", {}))
    except TypeError:
        strategy = strategy_cls()

    render_data_status(loaded, skipped)
    if group_meta:
        with st.expander("Per-universe load breakdown", expanded=False):
            render_group_breakdown(group_meta)
    st.markdown("---")
    render_screener(loaded, strategy, cfg["only_active"])
    st.markdown("---")
    render_backtest(loaded, strategy, cfg["strategy_name"])


if __name__ == "__main__":
    main()
