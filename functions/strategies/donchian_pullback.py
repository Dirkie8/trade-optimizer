"""
Donchian Pullback Strategy

Uses Donchian channel middle line as trend filter; enters on pullbacks to the
opposite band and resumption toward the trend.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _donchian(df: pd.DataFrame, n: int):
    upper = df["High"].rolling(n).max()
    lower = df["Low"].rolling(n).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid


class DonchianPullback(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("channel_period", 20))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < period + 3:
            return "HOLD", None, None

        u, l, m = _donchian(df, period)
        c_prev, c_last = df["Close"].iloc[-2], df["Close"].iloc[-1]
        u_prev, l_prev, m_prev = u.iloc[-2], l.iloc[-2], m.iloc[-2]
        m_last = m.iloc[-1]
        if any(pd.isna(x) for x in [u_prev, l_prev, m_prev, m_last, c_prev, c_last]):
            return "HOLD", None, None

        # Long: uptrend (price above mid), previous bar touched/lower than lower band, now closes above mid
        if c_prev <= l_prev and c_last > m_last:
            return "BUY", sl_pips, tp_pips
        # Short: downtrend (price below mid), previous bar touched/higher than upper band, now closes below mid
        if c_prev >= u_prev and c_last < m_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
