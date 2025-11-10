"""
MFI Mean Reversion Strategy

Uses Money Flow Index (MFI) oversold/overbought to fade extremes.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from functions.base_strategy import BaseStrategy, Action


def _mfi(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0.0)
    raw_money = tp * vol
    delta_tp = tp.diff()
    pos_flow = raw_money.where(delta_tp > 0, 0.0)
    neg_flow = raw_money.where(delta_tp < 0, 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum().replace(0, np.nan)
    mr = pos_sum / neg_sum
    mfi = 100 - (100 / (1 + mr))
    return mfi


class MFIMeanReversion(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("mfi_period", 14))
        oversold = float(cfg.get("oversold", 20))
        overbought = float(cfg.get("overbought", 80))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < period + 2 or "Volume" not in df.columns:
            return "HOLD", None, None

        mfi = _mfi(df, period)
        prev, last = mfi.iloc[-2], mfi.iloc[-1]
        if any(pd.isna(x) for x in [prev, last]):
            return "HOLD", None, None

        if prev <= oversold and last > prev:
            return "BUY", sl_pips, tp_pips
        if prev >= overbought and last < prev:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
