"""
NR7 Breakout Strategy

Identifies the Narrow Range 7 (NR7) pattern on the previous bar and trades a
breakout of that bar's high/low on the current close.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


class NR7Breakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        lookback = int(cfg.get("lookback", 7))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < lookback + 3:
            return "HOLD", None, None

        rng = (df["High"] - df["Low"]).copy()
        # Determine if the previous bar had the narrowest range of the last N
        prev_range = rng.iloc[-2]
        window = rng.iloc[-(lookback + 1):-1]
        if window.isna().any():
            return "HOLD", None, None
        nr7 = prev_range == window.min()

        prev_high = df["High"].iloc[-2]
        prev_low = df["Low"].iloc[-2]
        c = df["Close"].iloc[-1]

        if nr7 and c > prev_high:
            return "BUY", sl_pips, tp_pips
        if nr7 and c < prev_low:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
