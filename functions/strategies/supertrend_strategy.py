"""
Supertrend (Simplified) Strategy

Uses ATR-based bands around the average price. Signals when price breaks the bands.
This is a simplified version approximating supertrend behavior.
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


class SupertrendStrategy(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        atr_period = int(self.config.get("atr_period", 14))
        multiplier = float(self.config.get("multiplier", 2.0))
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        if len(df) < atr_period + 2:
            return "HOLD", None, None

        atr = _atr(df, atr_period)
        hl2 = (df["High"] + df["Low"]) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr

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
