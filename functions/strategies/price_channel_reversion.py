"""
Price Channel Reversion Strategy

Uses Donchian channel midline; fades when price deviates beyond a fraction of
channel width and turns back toward midline.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


class PriceChannelReversion(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("channel_period", 20))
        frac = float(cfg.get("fraction", 0.4))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < period + 3:
            return "HOLD", None, None

        h = df["High"].rolling(period).max()
        l = df["Low"].rolling(period).min()
        m = (h + l) / 2.0
        width = (h - l)

        c_prev, c_last = df["Close"].iloc[-2], df["Close"].iloc[-1]
        m_prev, m_last = m.iloc[-2], m.iloc[-1]
        w_prev = width.iloc[-2]
        if any(pd.isna(x) for x in [c_prev, c_last, m_prev, m_last, w_prev]):
            return "HOLD", None, None

        upper_band_prev = m_prev + frac * w_prev
        lower_band_prev = m_prev - frac * w_prev

        # Fade extremes when price moves back toward midline
        if c_prev <= lower_band_prev and c_last > c_prev:
            return "BUY", sl_pips, tp_pips
        if c_prev >= upper_band_prev and c_last < c_prev:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
