"""
TRIX Crossover Strategy

Uses TRIX (triple-smoothed EMA ROC) with signal line crossover.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def _trix(series: pd.Series, period: int) -> pd.Series:
    e1 = _ema(series, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    trix = e3.pct_change() * 100.0
    return trix


class TRIXCrossover(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("trix_period", 15))
        signal_n = int(cfg.get("signal_period", 9))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        close = self.data["Close"]
        if len(close) < period * 3 + signal_n + 2:
            return "HOLD", None, None

        t = _trix(close, period)
        s = _ema(t, signal_n)
        t_prev, t_last = t.iloc[-2], t.iloc[-1]
        s_prev, s_last = s.iloc[-2], s.iloc[-1]
        if any(pd.isna(x) for x in [t_prev, t_last, s_prev, s_last]):
            return "HOLD", None, None

        if t_prev <= s_prev and t_last > s_last:
            return "BUY", sl_pips, tp_pips
        if t_prev >= s_prev and t_last < s_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
