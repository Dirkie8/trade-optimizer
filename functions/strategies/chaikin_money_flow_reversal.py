"""
Chaikin Money Flow Reversal Strategy

Uses CMF to detect buying/selling pressure reversals; optional price EMA filter.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from functions.base_strategy import BaseStrategy, Action


def _cmf(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    vol = df["Volume"].fillna(0.0)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * vol
    cmf = mfv.rolling(period).sum() / vol.rolling(period).sum().replace(0, np.nan)
    return cmf


class CMFReversal(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("cmf_period", 20))
        threshold = float(cfg.get("threshold", 0.05))
        ema_filter = int(cfg.get("ema_filter", 50))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < max(period, ema_filter) + 2 or "Volume" not in df.columns:
            return "HOLD", None, None

        cmf = _cmf(df, period)
        c_prev, c_last = cmf.iloc[-2], cmf.iloc[-1]
        if any(pd.isna(x) for x in [c_prev, c_last]):
            return "HOLD", None, None

        ema = df["Close"].ewm(span=ema_filter, adjust=False).mean()
        ema_last = ema.iloc[-1]
        price_last = df["Close"].iloc[-1]

        long_ok = (c_prev <= -threshold and c_last > c_prev)
        short_ok = (c_prev >= threshold and c_last < c_prev)

        if long_ok and price_last > ema_last:
            return "BUY", sl_pips, tp_pips
        if short_ok and price_last < ema_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
