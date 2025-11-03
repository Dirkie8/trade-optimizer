"""
MACD Zero Cross Strategy

Trend/momentum strategy triggering on MACD line crossing the zero axis.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class MACDZeroCross(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        fast = int(self.config.get("fast_period", 12))
        slow = int(self.config.get("slow_period", 26))
        signal = int(self.config.get("signal_period", 9))
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        if not (fast < slow and fast > 0 and slow > 0 and signal > 0):
            return "HOLD", None, None

        df = self.data.copy()
        if len(df) < slow + signal + 2:
            return "HOLD", None, None

        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        # Optionally compute signal line (not strictly needed for zero cross)
        # macd_signal = macd.ewm(span=signal, adjust=False).mean()

        prev_macd, last_macd = macd.iloc[-2], macd.iloc[-1]
        if pd.isna(prev_macd) or pd.isna(last_macd):
            return "HOLD", None, None

        if prev_macd <= 0 and last_macd > 0:
            return "BUY", sl_pips, tp_pips
        if prev_macd >= 0 and last_macd < 0:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
