"""
Engulfing Pattern with Trend Filter Strategy

Uses bullish/bearish engulfing candlestick with EMA trend filter.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


class EngulfingTrend(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        ema_n = int(cfg.get("ema_period", 50))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < ema_n + 3:
            return "HOLD", None, None

        c1o, c1c = df["Open"].iloc[-2], df["Close"].iloc[-2]
        c2o, c2c = df["Open"].iloc[-1], df["Close"].iloc[-1]
        ema = _ema(df["Close"], ema_n)
        ema_last = ema.iloc[-1]
        if any(pd.isna(x) for x in [ema_last]):
            return "HOLD", None, None

        bullish_engulf = (c2c > c2o) and (c1c < c1o) and (c2c >= c1o) and (c2o <= c1c)
        bearish_engulf = (c2c < c2o) and (c1c > c1o) and (c2c <= c1o) and (c2o >= c1c)

        if bullish_engulf and c2c > ema_last:
            return "BUY", sl_pips, tp_pips
        if bearish_engulf and c2c < ema_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
