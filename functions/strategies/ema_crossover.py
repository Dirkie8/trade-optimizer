"""
EMA Crossover Strategy

Trend-following strategy using two Exponential Moving Averages (EMAs).
Buy when fast EMA crosses above slow EMA; sell when fast crosses below slow.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class EMACrossover(BaseStrategy):
    """Exponential Moving Average crossover strategy.

    Parameters:
    - fast_period: EMA period for the fast average
    - slow_period: EMA period for the slow average
    - take_profit_pips: Target profit in pips
    - stop_loss_pips: Maximum loss in pips
    """

    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        fast = int(self.config["fast_period"])
        slow = int(self.config["slow_period"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        if fast <= 0 or slow <= 0 or fast >= slow:
            return "HOLD", None, None

        df = self.data.copy()
        if len(df) < slow + 2:
            return "HOLD", None, None

        df["ema_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
        df["ema_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if pd.isna(prev["ema_fast"]) or pd.isna(prev["ema_slow"]) or pd.isna(last["ema_fast"]) or pd.isna(last["ema_slow"]):
            return "HOLD", None, None

        crossed_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        crossed_down = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

        if crossed_up:
            return "BUY", sl_pips, tp_pips
        if crossed_down:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
