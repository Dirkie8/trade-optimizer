"""
ATR Channel Breakout (Chandelier-style)

Builds channels based on highest high/lowest low and ATR multiplier. Signals on price breaking channels.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class ATRChannelBreakout(BaseStrategy): 
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        lookback = int(self.config.get("lookback", 22))
        atr_period = int(self.config.get("atr_period", 14))
        mult = float(self.config.get("atr_multiplier", 3.0))
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        need = max(lookback, atr_period)
        if len(df) < need + 2:
            return "HOLD", None, None

        hh = df["High"].rolling(lookback).max()
        ll = df["Low"].rolling(lookback).min()
        atr = _atr(df, atr_period)
        upper = hh - mult * atr
        lower = ll + mult * atr

        prev_close, last_close = df["Close"].iloc[-2], df["Close"].iloc[-1]
        prev_upper, last_upper = upper.iloc[-2], upper.iloc[-1]
        prev_lower, last_lower = lower.iloc[-2], lower.iloc[-1]
        if any(pd.isna(x) for x in [prev_upper, last_upper, prev_lower, last_lower]):
            return "HOLD", None, None

        if prev_close <= prev_upper and last_close > last_upper:
            return "BUY", sl_pips, tp_pips
        if prev_close >= prev_lower and last_close < last_lower:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
