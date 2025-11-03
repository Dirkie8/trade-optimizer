"""
Triple Moving Average Strategy

Uses three EMAs (short, mid, long).
Signals when the ordering aligns strongly (bullish: short > mid > long, bearish: short < mid < long)
and the ordering changed recently.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class TripleMovingAverage(BaseStrategy):
    """Triple EMA alignment strategy."""

    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        short = int(self.config["short_period"])
        mid = int(self.config["mid_period"])
        long = int(self.config["long_period"])
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        if not (short < mid < long):
            return "HOLD", None, None

        df = self.data.copy()
        if len(df) < long + 2:
            return "HOLD", None, None

        df["ema_s"] = df["Close"].ewm(span=short, adjust=False).mean()
        df["ema_m"] = df["Close"].ewm(span=mid, adjust=False).mean()
        df["ema_l"] = df["Close"].ewm(span=long, adjust=False).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]
        if any(pd.isna(x) for x in [last["ema_s"], last["ema_m"], last["ema_l"], prev["ema_s"], prev["ema_m"], prev["ema_l"]]):
            return "HOLD", None, None

        bull_now = last["ema_s"] > last["ema_m"] > last["ema_l"]
        bull_prev = prev["ema_s"] > prev["ema_m"] > prev["ema_l"]
        bear_now = last["ema_s"] < last["ema_m"] < last["ema_l"]
        bear_prev = prev["ema_s"] < prev["ema_m"] < prev["ema_l"]

        if bull_now and not bull_prev:
            return "BUY", sl_pips, tp_pips
        if bear_now and not bear_prev:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
