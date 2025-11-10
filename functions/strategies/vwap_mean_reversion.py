"""
VWAP Mean Reversion Strategy

Uses rolling VWAP bands to fade deviations.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _rolling_vwap(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0.0)
    pv = (tp * vol).rolling(period).sum()
    vv = vol.rolling(period).sum().replace(0, pd.NA)
    return pv / vv


class VWAPMeanReversion(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("vwap_period", 50))
        band = float(cfg.get("band", 0.0015))  # 15 bps default
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        df = self.data
        if len(df) < period + 2 or "Volume" not in df.columns:
            return "HOLD", None, None

        vwap = _rolling_vwap(df, period)
        c_prev, c_last = df["Close"].iloc[-2], df["Close"].iloc[-1]
        v_prev, v_last = vwap.iloc[-2], vwap.iloc[-1]
        if any(pd.isna(x) for x in [c_prev, c_last, v_prev, v_last]):
            return "HOLD", None, None

        upper = v_last * (1 + band)
        lower = v_last * (1 - band)

        if c_prev <= lower and c_last > v_last:
            return "BUY", sl_pips, tp_pips
        if c_prev >= upper and c_last < v_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
