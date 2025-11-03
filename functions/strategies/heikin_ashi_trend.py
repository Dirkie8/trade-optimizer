"""
Heikin Ashi Trend Strategy

Constructs Heikin-Ashi candles and uses their direction combined with an EMA filter.
Signals when HA flips direction and aligns with EMA trend.
"""
import pandas as pd
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


class HeikinAshiTrend(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        ema_period = int(self.config.get("ema_period", 50))
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data
        # Need a bit of history for EMA and HA computation
        if len(df) < max(ema_period, 20) + 3:
            return "HOLD", None, None

        # Compute EMA on Close (vectorized)
        ema = df["Close"].ewm(span=ema_period, adjust=False).mean()
        if pd.isna(ema.iloc[-1]) or pd.isna(ema.iloc[-2]):
            return "HOLD", None, None

        # Compute only a small tail of Heikin-Ashi to avoid O(n^2) per-bar work
        # Tail size anchored to EMA period but capped to reasonable bounds
        tail = max(ema_period, 50)
        tail = min(tail, len(df))
        o = df["Open"].iloc[-tail:]
        h = df["High"].iloc[-tail:]
        l = df["Low"].iloc[-tail:]
        c = df["Close"].iloc[-tail:]

        ha_close = (o + h + l + c) / 4.0
        # Initialize ha_open reasonably using mid of first bar
        ha_open = ha_close.copy()
        ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2.0
        # Iterate only over the small tail
        for i in range(1, len(ha_close)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

        # Determine HA direction flip on last two bars
        bull_prev = ha_close.iloc[-2] > ha_open.iloc[-2]
        bull_last = ha_close.iloc[-1] > ha_open.iloc[-1]

        uptrend = df["Close"].iloc[-1] > ema.iloc[-1]
        downtrend = df["Close"].iloc[-1] < ema.iloc[-1]

        if bull_last and not bull_prev and uptrend:
            return "BUY", sl_pips, tp_pips
        if (not bull_last) and bull_prev and downtrend:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
