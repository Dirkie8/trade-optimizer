"""
Volatility Expansion Breakout Strategy

When ATR expands sharply vs its moving average, trade a breakout of recent
high/low.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


class VolatilityExpansionBreakout(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        atr_period = int(cfg.get("atr_period", 14))
        smooth = int(cfg.get("atr_smooth", 20))
        expansion_k = float(cfg.get("expansion_k", 1.5))
        lookback = int(cfg.get("lookback", 20))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < max(atr_period, smooth, lookback) + 3:
            return "HOLD", None, None

        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        atr_ma = atr.rolling(smooth).mean()

        # Expansion condition on previous bar
        if pd.isna(atr.iloc[-2]) or pd.isna(atr_ma.iloc[-2]):
            return "HOLD", None, None
        expand = atr.iloc[-2] > expansion_k * atr_ma.iloc[-2]

        recent_high = h.iloc[-(lookback + 1):-1].max()
        recent_low = l.iloc[-(lookback + 1):-1].min()
        cl = c.iloc[-1]

        if expand and cl > recent_high:
            return "BUY", sl_pips, tp_pips
        if expand and cl < recent_low:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
