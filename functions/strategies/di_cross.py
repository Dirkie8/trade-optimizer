"""
Directional Movement (DI) Cross Strategy

Uses +DI/-DI crossover without an ADX threshold for more frequent signals.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from functions.base_strategy import BaseStrategy, Action


def _di(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
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
    return plus_di, minus_di


class DICross(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("di_period", 14))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < period + 2:
            return "HOLD", None, None

        plus_di, minus_di = _di(df, period)
        p_prev, p_last = plus_di.iloc[-2], plus_di.iloc[-1]
        m_prev, m_last = minus_di.iloc[-2], minus_di.iloc[-1]
        if any(pd.isna(x) for x in [p_prev, p_last, m_prev, m_last]):
            return "HOLD", None, None

        if p_prev <= m_prev and p_last > m_last:
            return "BUY", sl_pips, tp_pips
        if p_prev >= m_prev and p_last < m_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
