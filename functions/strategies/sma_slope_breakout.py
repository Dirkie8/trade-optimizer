"""
SMA Slope Breakout Strategy

Requires SMA slope in the direction of a breakout of recent highs/lows.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


class SMASlopeBreakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        sma_n = int(cfg.get("sma_period", 50))
        lookback = int(cfg.get("lookback", 20))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        c = self.data["Close"]
        h = self.data["High"]
        l = self.data["Low"]
        if len(c) < max(sma_n, lookback) + 3:
            return "HOLD", None, None

        sma = _sma(c, sma_n)
        sma_prev, sma_last = sma.iloc[-2], sma.iloc[-1]
        if pd.isna(sma_prev) or pd.isna(sma_last):
            return "HOLD", None, None

        recent_high = h.iloc[-(lookback + 1):-1].max()
        recent_low = l.iloc[-(lookback + 1):-1].min()
        cl = c.iloc[-1]

        if cl > recent_high and sma_last > sma_prev:
            return "BUY", sl_pips, tp_pips
        if cl < recent_low and sma_last < sma_prev:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
