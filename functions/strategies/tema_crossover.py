"""
TEMA Crossover Strategy

Uses Triple Exponential Moving Averages (TEMA) for fast/slow crossover signals.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _tema(series: pd.Series, period: int) -> pd.Series:
    e1 = _ema(series, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    return 3 * (e1 - e2) + e3


class TEMACrossover(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        fast = int(cfg.get("fast_period", 7))
        slow = int(cfg.get("slow_period", 28))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        if slow <= fast:
            slow = fast + 1

        close = self.data["Close"]
        if len(close) < slow + 2:
            return "HOLD", None, None

        f = _tema(close, fast)
        s = _tema(close, slow)
        f_prev, f_last = f.iloc[-2], f.iloc[-1]
        s_prev, s_last = s.iloc[-2], s.iloc[-1]
        if any(pd.isna(x) for x in [f_prev, f_last, s_prev, s_last]):
            return "HOLD", None, None

        if f_prev <= s_prev and f_last > s_last:
            return "BUY", sl_pips, tp_pips
        if f_prev >= s_prev and f_last < s_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
