"""
Fractal Breakout Strategy

Uses Bill Williams fractals: breakout beyond the most recent confirmed fractal
high/low within a lookback window.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _fractal_high(high: pd.Series) -> pd.Series:
    return (high.shift(2) < high.shift(1)) & (high.shift(1) < high) & (high.shift(1) > high.shift(3)) & (high.shift(1) > high.shift(4))


def _fractal_low(low: pd.Series) -> pd.Series:
    return (low.shift(2) > low.shift(1)) & (low.shift(1) > low) & (low.shift(1) < low.shift(3)) & (low.shift(1) < low.shift(4))


class FractalBreakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        lookback = int(cfg.get("lookback", 30))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < 10:
            return "HOLD", None, None

        fh = _fractal_high(df["High"]).fillna(False)
        fl = _fractal_low(df["Low"]).fillna(False)

        recent_high_idx = fh.iloc[-(lookback + 5):-1][fh.iloc[-(lookback + 5):-1]].last_valid_index()
        recent_low_idx = fl.iloc[-(lookback + 5):-1][fl.iloc[-(lookback + 5):-1]].last_valid_index()

        c_last = df["Close"].iloc[-1]
        if recent_high_idx is not None:
            level = df.loc[recent_high_idx, "High"]
            if c_last > level:
                return "BUY", sl_pips, tp_pips
        if recent_low_idx is not None:
            level = df.loc[recent_low_idx, "Low"]
            if c_last < level:
                return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
