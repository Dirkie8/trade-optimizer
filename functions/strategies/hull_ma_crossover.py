"""
Hull MA Crossover Strategy

Uses fast/slow Hull Moving Averages (HMA) for crossover signals.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _wma(series: pd.Series, period: int) -> pd.Series:
    """Fast weighted moving average using numpy convolution (weights 1..period).

    Produces NaN for the first period-1 entries to align with pandas rolling.
    """
    if period <= 1:
        return series.copy()
    x = series.to_numpy(dtype=float, copy=False)
    w = np.arange(1, period + 1, dtype=float)
    denom = w.sum()
    conv = np.convolve(x, w[::-1], mode='valid') / denom
    out = np.empty_like(x, dtype=float)
    out[: period - 1] = np.nan
    out[period - 1 :] = conv
    return pd.Series(out, index=series.index)


def _hma(series: pd.Series, period: int) -> pd.Series:
    if period < 2:
        period = 2
    half = int(round(period / 2))
    sqrt = int(round(np.sqrt(period)))
    wma1 = _wma(series, half)
    wma2 = _wma(series, period)
    diff = 2 * wma1 - wma2
    return _wma(diff, sqrt)


class HullMACrossover(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        fast = int(cfg.get("fast_period", 12))
        slow = int(cfg.get("slow_period", 36))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        if slow <= fast:
            slow = fast + 1

        close = self.data["Close"]
        if len(close) < slow + 5:
            return "HOLD", None, None

        f = _hma(close, fast)
        s = _hma(close, slow)
        f_prev, f_last = f.iloc[-2], f.iloc[-1]
        s_prev, s_last = s.iloc[-2], s.iloc[-1]
        if any(pd.isna(x) for x in [f_prev, f_last, s_prev, s_last]):
            return "HOLD", None, None

        if f_prev <= s_prev and f_last > s_last:
            return "BUY", sl_pips, tp_pips
        if f_prev >= s_prev and f_last < s_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
