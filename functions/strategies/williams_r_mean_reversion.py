"""
Williams %R Mean Reversion Strategy

Buys when %R is deeply oversold and turns up; sells when %R is overbought and
turns down. Classic thresholds: -80 (oversold), -20 (overbought).
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from functions.base_strategy import BaseStrategy, Action


def _williams_r(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["High"].rolling(period).max()
    low = df["Low"].rolling(period).min()
    close = df["Close"]
    wr = -100 * (high - close) / (high - low).replace(0, np.nan)
    return wr


class WilliamsRMeanReversion(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("wr_period", 14))
        overbought = float(cfg.get("wr_overbought", -20))
        oversold = float(cfg.get("wr_oversold", -80))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < period + 2:
            return "HOLD", None, None

        wr = _williams_r(df, period)
        prev, last = wr.iloc[-2], wr.iloc[-1]
        if any(pd.isna(x) for x in [prev, last]):
            return "HOLD", None, None

        # Mean reversion with simple turn confirmation
        if prev <= oversold and last > prev:
            return "BUY", sl_pips, tp_pips
        if prev >= overbought and last < prev:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
