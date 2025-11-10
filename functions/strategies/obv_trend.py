"""
OBV Trend Strategy

Uses On-Balance Volume (OBV) with EMA smoothing and crossover.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _obv(df: pd.DataFrame) -> pd.Series:
    c = df["Close"]
    v = df["Volume"].fillna(0.0)
    direction = (c.diff().fillna(0.0)).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * v).cumsum()


class OBVTrend(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        fast = int(cfg.get("fast_period", 10))
        slow = int(cfg.get("slow_period", 30))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if "Volume" not in df.columns or len(df) < slow + 2:
            return "HOLD", None, None

        obv = _obv(df)
        f = obv.ewm(span=fast, adjust=False).mean()
        s = obv.ewm(span=slow, adjust=False).mean()
        f_prev, f_last = f.iloc[-2], f.iloc[-1]
        s_prev, s_last = s.iloc[-2], s.iloc[-1]
        if any(pd.isna(x) for x in [f_prev, f_last, s_prev, s_last]):
            return "HOLD", None, None

        if f_prev <= s_prev and f_last > s_last:
            return "BUY", sl_pips, tp_pips
        if f_prev >= s_prev and f_last < s_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
