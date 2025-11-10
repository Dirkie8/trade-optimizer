"""
RSI2 Mean Reversion Strategy

Classic short-term RSI(2) mean reversion signals.
"""
from typing import Tuple, Optional
import pandas as pd

from functions.base_strategy import BaseStrategy, Action


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


class RSI2MeanReversion(BaseStrategy):
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        cfg = self.config
        period = int(cfg.get("rsi_period", 2))
        oversold = float(cfg.get("oversold", 10))
        overbought = float(cfg.get("overbought", 90))
        tp_pips = float(cfg["take_profit_pips"])
        sl_pips = float(cfg["stop_loss_pips"])

        close = self.data["Close"]
        if len(close) < period + 2:
            return "HOLD", None, None

        r = _rsi(close, period)
        prev, last = r.iloc[-2], r.iloc[-1]
        if pd.isna(prev) or pd.isna(last):
            return "HOLD", None, None

        # Mean reversion with simple turn confirmation
        if prev <= oversold and last > prev:
            return "BUY", sl_pips, tp_pips
        if prev >= overbought and last < prev:
            return "SELL", sl_pips, tp_pips
        return "HOLD", None, None
