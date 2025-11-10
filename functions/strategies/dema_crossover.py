"""
DEMA Crossover Strategy

Uses Double Exponential Moving Averages (DEMA) for fast/slow crossover signals.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _dema(series: pd.Series, period: int) -> pd.Series:
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    return 2 * ema1 - ema2


class DEMACrossover(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        fast = int(cfg.get("fast_period", 10))
        slow = int(cfg.get("slow_period", 30))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        if slow <= fast:
            slow = fast + 1

        close = self.data["Close"]
        if len(close) < slow + 2:
            return "HOLD", None, None

        f = _dema(close, fast)
        s = _dema(close, slow)
        f_prev, f_last = f.iloc[-2], f.iloc[-1]
        s_prev, s_last = s.iloc[-2], s.iloc[-1]
        if any(pd.isna(x) for x in [f_prev, f_last, s_prev, s_last]):
            return "HOLD", None, None

        if f_prev <= s_prev and f_last > s_last:
            return "BUY", sl_pips, tp_pips
        if f_prev >= s_prev and f_last < s_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
