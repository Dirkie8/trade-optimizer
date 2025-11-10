"""
Pivot Reversal Strategy (Daily Pivots on Intraday)

Computes prior day's pivot points and signals when price crosses R1/S1.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


class PivotReversal(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < 100:
            return "HOLD", None, None

        # Resample to daily to compute yesterday's pivots
        daily = df.resample('1D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
        if len(daily) < 2:
            return "HOLD", None, None

        y_high = daily["High"].iloc[-2]
        y_low = daily["Low"].iloc[-2]
        y_close = daily["Close"].iloc[-2]
        if any(pd.isna(x) for x in [y_high, y_low, y_close]):
            return "HOLD", None, None

        P = (y_high + y_low + y_close) / 3.0
        R1 = 2 * P - y_low
        S1 = 2 * P - y_high

        cl = df["Close"].iloc[-1]
        if cl > R1:
            return "BUY", sl_pips, tp_pips
        if cl < S1:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
