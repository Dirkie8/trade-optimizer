"""
Inside Bar Breakout Strategy

Trades breakouts of inside bars (bar with range fully inside previous bar).
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


class InsideBarBreakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < 3:
            return "HOLD", None, None

        prev_high = df["High"].iloc[-2]
        prev_low = df["Low"].iloc[-2]
        prev2_high = df["High"].iloc[-3]
        prev2_low = df["Low"].iloc[-3]

        # Inside bar condition for previous bar
        inside = (prev_high <= prev2_high) and (prev_low >= prev2_low)
        cl = df["Close"].iloc[-1]
        if inside and cl > prev_high:
            return "BUY", sl_pips, tp_pips
        if inside and cl < prev_low:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
