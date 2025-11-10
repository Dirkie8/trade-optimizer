"""
EMA Pullback Strategy

Trend filter using slow EMA; enters on pullback to fast EMA and resumption.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


class EMAPullback(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        fast = int(cfg.get("fast_period", 20))
        slow = int(cfg.get("slow_period", 100))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        c = self.data["Close"]
        if len(c) < slow + 3:
            return "HOLD", None, None

        f = _ema(c, fast)
        s = _ema(c, slow)
        c_prev, c_last = c.iloc[-2], c.iloc[-1]
        f_prev, f_last = f.iloc[-2], f.iloc[-1]
        s_prev, s_last = s.iloc[-2], s.iloc[-1]
        if any(pd.isna(x) for x in [c_prev, c_last, f_prev, f_last, s_prev, s_last]):
            return "HOLD", None, None

        # Long: trend up, prior close below fast EMA, now close back above fast EMA
        if c_prev < f_prev and c_last > f_last and c_last > s_last and s_last >= s_prev:
            return "BUY", sl_pips, tp_pips
        # Short: trend down, prior close above fast EMA, now close back below fast EMA
        if c_prev > f_prev and c_last < f_last and c_last < s_last and s_last <= s_prev:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
