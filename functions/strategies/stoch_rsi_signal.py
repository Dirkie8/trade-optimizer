"""
StochRSI Strategy

Uses Stochastic RSI to identify overbought/oversold conditions.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from functions.base_strategy import BaseStrategy, Action


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _stoch(x: pd.Series, k: int, d: int) -> tuple[pd.Series, pd.Series]:
    lowest = x.rolling(k).min()
    highest = x.rolling(k).max()
    stoch_k = 100 * (x - lowest) / (highest - lowest)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


class StochRSIStrategy(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        rsi_n = int(cfg.get("rsi_period", 14))
        k_n = int(cfg.get("stoch_k", 14))
        d_n = int(cfg.get("stoch_d", 3))
        oversold = float(cfg.get("oversold", 20))
        overbought = float(cfg.get("overbought", 80))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        c = self.data["Close"]
        if len(c) < rsi_n + k_n + d_n + 2:
            return "HOLD", None, None

        r = _rsi(c, rsi_n)
        k, d = _stoch(r, k_n, d_n)
        k_prev, k_last = k.iloc[-2], k.iloc[-1]
        d_prev, d_last = d.iloc[-2], d.iloc[-1]
        if any(pd.isna(x) for x in [k_prev, k_last, d_prev, d_last]):
            return "HOLD", None, None

        if k_prev <= oversold and k_last > d_last:
            return "BUY", sl_pips, tp_pips
        if k_prev >= overbought and k_last < d_last:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
