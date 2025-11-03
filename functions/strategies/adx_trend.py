"""
ADX Trend Strategy

Uses Average Directional Index (ADX) to confirm trend strength and DI+/DI- crossover for direction.
Signals when ADX is above threshold and DI lines favor a direction.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from functions.base_strategy import BaseStrategy, Action


def _adx(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di


class ADXTrend(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        period = int(self.config.get("adx_period", 14))
        threshold = float(self.config.get("adx_threshold", 20))
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])

        df = self.data.copy()
        if len(df) < period * 3:
            return "HOLD", None, None

        adx, plus_di, minus_di = _adx(df, period)
        adx_prev, adx_last = adx.iloc[-2], adx.iloc[-1]
        p_prev, p_last = plus_di.iloc[-2], plus_di.iloc[-1]
        m_prev, m_last = minus_di.iloc[-2], minus_di.iloc[-1]
        if any(pd.isna(x) for x in [adx_prev, adx_last, p_prev, p_last, m_prev, m_last]):
            return "HOLD", None, None

        strong = adx_last >= threshold
        if strong and p_prev <= m_prev and p_last > m_last:
            return "BUY", sl_pips, tp_pips
        if strong and p_prev >= m_prev and p_last < m_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
