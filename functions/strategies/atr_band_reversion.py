"""
ATR Band Reversion Strategy

Uses SMA +/- k * ATR bands; fades extremes when price turns back inward.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


class ATRBandReversion(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        sma_period = int(cfg.get("sma_period", 20))
        atr_period = int(cfg.get("atr_period", 14))
        k = float(cfg.get("atr_mult", 1.5))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < max(sma_period, atr_period) + 3:
            return "HOLD", None, None

        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        sma = c.rolling(sma_period).mean()
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        sma_prev, sma_last = sma.iloc[-2], sma.iloc[-1]
        atr_prev = atr.iloc[-2]
        c_prev, c_last = c.iloc[-2], c.iloc[-1]
        if any(pd.isna(x) for x in [sma_prev, sma_last, atr_prev, c_prev, c_last]):
            return "HOLD", None, None

        upper_prev = sma_prev + k * atr_prev
        lower_prev = sma_prev - k * atr_prev

        if c_prev <= lower_prev and c_last > c_prev:
            return "BUY", sl_pips, tp_pips
        if c_prev >= upper_prev and c_last < c_prev:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
