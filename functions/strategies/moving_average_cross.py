"""
MovingAverageCross Strategy

Simple moving average crossover strategy.
Buy when short MA crosses above long MA, sell when it crosses below.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class MovingAverageCross(BaseStrategy):
    """Simple Moving Average Crossover strategy.

    Logic:
    - Compute short and long simple moving averages on Close.
    - If short crosses above long -> BUY, else if short crosses below long -> SELL, otherwise HOLD.
    - Always returns SL/TP in pips for BUY/SELL, taken from config keys:
        - short_ma_period, long_ma_period, take_profit_pips, stop_loss_pips
    """

    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        short_period = int(self.config["short_ma_period"])
        long_period = int(self.config["long_ma_period"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        if short_period <= 0 or long_period <= 0:
            raise ValueError("MA periods must be positive integers")
        if short_period >= long_period:
            # Enforce conventional constraint; otherwise crosses get noisy
            return "HOLD", None, None

        df = self.data.copy()
        if len(df) < long_period + 2:
            # Not enough data to produce a reliable crossover signal
            return "HOLD", None, None

        df["ma_short"] = df["Close"].rolling(short_period, min_periods=short_period).mean()
        df["ma_long"] = df["Close"].rolling(long_period, min_periods=long_period).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if pd.isna(prev["ma_short"]) or pd.isna(prev["ma_long"]) or pd.isna(last["ma_short"]) or pd.isna(last["ma_long"]):
            return "HOLD", None, None

        crossed_up = prev["ma_short"] < prev["ma_long"] and last["ma_short"] > last["ma_long"]
        crossed_down = prev["ma_short"] > prev["ma_long"] and last["ma_short"] < last["ma_long"]

        if crossed_up:
            return "BUY", sl_pips, tp_pips
        elif crossed_down:
            return "SELL", sl_pips, tp_pips
        else:
            return "HOLD", None, None
