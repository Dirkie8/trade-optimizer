"""
Keltner Channel Breakout Strategy

EMA-based midline with ATR bands; signal on breakouts.
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


class KeltnerChannelBreakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        ema_period = int(self.config.get("ema_period", 20))
        atr_period = int(self.config.get("atr_period", 14))
        multiplier = float(self.config.get("multiplier", 2.0))
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        if len(df) < max(ema_period, atr_period) + 2:
            return "HOLD", None, None

        ema = df["Close"].ewm(span=ema_period, adjust=False).mean()
        atr = _atr(df, atr_period)
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr

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
